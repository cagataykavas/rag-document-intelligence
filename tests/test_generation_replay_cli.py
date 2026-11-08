from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

import src.generation_replay_cli as replay_cli
from src.admission_authentication import AdmissionAuthenticator
from src.claims import claim_support_sha256, trace_claim_support
from src.evidence_snapshot import build_evidence_snapshot
from src.generation_policy import (
    CONTENT_BOUND_GENERATION_POLICY,
    admit_generation,
    bind_claim_support,
)
from src.generation_replay_cli import (
    AUTH_KEY_ENV,
    DECRYPT_KEY_ENV,
    PRIVATE_BUNDLE_SCHEMA_VERSION,
    ReplayCliError,
    main,
    run,
    seal_private_replay_bundle,
)
from src.prompting import EvidenceSnippet

DECRYPT_KEY = b"d" * 32
AUTH_KEY = b"a" * 32


def _encoded(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _payload() -> dict[str, object]:
    return {
        "answer": "Audit records are retained for seven years.",
        "citations": ["policy:7"],
        "claim_citations": [{"claim_index": 0, "citations": ["policy:7"]}],
        "confidence": 0.93,
        "insufficient_evidence": False,
    }


def _snippets() -> list[EvidenceSnippet]:
    return [
        EvidenceSnippet(
            "policy:7",
            "policy.md",
            "Audit records are retained for seven years.",
        )
    ]


def _audit() -> dict[str, object]:
    snippets = _snippets()
    snapshot = build_evidence_snapshot(snippets)
    admission = admit_generation(
        _payload(),
        snapshot.chunk_ids,
        CONTENT_BOUND_GENERATION_POLICY,
        evidence_snapshot_sha256=snapshot.snapshot_sha256,
    )
    support = trace_claim_support(
        admission.answer,
        admission.citations,
        [{"chunk_id": row.chunk_id, "text": row.text} for row in snippets],
        admission.claim_citations,
    )
    return bind_claim_support(admission, claim_support_sha256(support)).to_audit_dict()


def _private_bundle(
    *, authentication: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "schema_version": PRIVATE_BUNDLE_SCHEMA_VERSION,
        "payload": _payload(),
        "evidence": [
            {
                "chunk_id": row.chunk_id,
                "source": row.source,
                "text": row.text,
                "source_locator": row.source_locator,
            }
            for row in _snippets()
        ],
        "expected_admission_audit": _audit(),
        "authentication": authentication,
    }


def _write_bundle(
    path: Path,
    *,
    private_bundle: dict[str, object] | None = None,
    key_id: str = "archive-2026",
    key: bytes = DECRYPT_KEY,
) -> bytes:
    envelope = seal_private_replay_bundle(
        _private_bundle() if private_bundle is None else private_bundle,
        key_id=key_id,
        key=key,
        nonce=b"n" * 12,
    )
    path.write_bytes(envelope)
    path.chmod(0o600)
    return envelope


@pytest.fixture(autouse=True)
def decrypt_keyring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        DECRYPT_KEY_ENV, json.dumps({"archive-2026": _encoded(DECRYPT_KEY)})
    )
    monkeypatch.delenv(AUTH_KEY_ENV, raising=False)


def test_cli_decrypts_replays_and_writes_only_privacy_reduced_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = tmp_path / "private.bundle"
    receipt_path = tmp_path / "receipt.json"
    _write_bundle(bundle)

    assert main([str(bundle), "--receipt", str(receipt_path)]) == 0

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    rendered = receipt_path.read_text(encoding="utf-8")
    assert receipt["verified"] is True
    assert receipt["accepted"] is True
    assert receipt["authentication_verified"] is False
    assert stat_mode(receipt_path) == 0o600
    assert bundle.exists()
    assert _payload()["answer"] not in rendered
    assert "policy:7" not in rendered
    assert "policy.md" not in rendered
    assert json.loads(capsys.readouterr().out)["verified"] is True


def test_delete_flag_removes_exact_bundle_only_after_receipt_is_durable(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "private.bundle"
    receipt_path = tmp_path / "receipt.json"
    _write_bundle(bundle)

    receipt = run(bundle, receipt_path, delete_after_success=True)

    assert receipt["verified"] is True
    assert receipt_path.exists()
    assert not bundle.exists()


def test_failed_replay_preserves_bundle_and_does_not_write_receipt(
    tmp_path: Path,
) -> None:
    private_bundle = _private_bundle()
    private_bundle["payload"] = {**_payload(), "answer": "substituted"}
    bundle = tmp_path / "private.bundle"
    receipt_path = tmp_path / "receipt.json"
    _write_bundle(bundle, private_bundle=private_bundle)

    assert (
        main(
            [
                str(bundle),
                "--receipt",
                str(receipt_path),
                "--delete-private-bundle-after-success",
            ]
        )
        == 2
    )

    assert bundle.exists()
    assert not receipt_path.exists()


def test_delete_detects_in_place_change_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "private.bundle"
    receipt_path = tmp_path / "receipt.json"
    original = _write_bundle(bundle)
    write_receipt = replay_cli._write_receipt

    def write_then_change(path: Path, receipt: dict[str, object]) -> None:
        write_receipt(path, receipt)
        bundle.write_bytes(original)
        bundle.chmod(0o600)

    monkeypatch.setattr(replay_cli, "_write_receipt", write_then_change)

    with pytest.raises(ReplayCliError, match="BUNDLE_DELETE_RACE"):
        run(bundle, receipt_path, delete_after_success=True)
    assert bundle.exists()
    assert receipt_path.exists()


def test_ciphertext_authentication_and_key_selection_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "private.bundle"
    receipt_path = tmp_path / "receipt.json"
    envelope = json.loads(_write_bundle(bundle))
    ciphertext = bytearray(base64.urlsafe_b64decode(envelope["ciphertext_b64"]))
    ciphertext[-1] ^= 1
    envelope["ciphertext_b64"] = _encoded(bytes(ciphertext))
    bundle.write_text(json.dumps(envelope), encoding="utf-8")
    bundle.chmod(0o600)

    with pytest.raises(ReplayCliError, match="BUNDLE_AUTHENTICATION_FAILED"):
        run(bundle, receipt_path)

    _write_bundle(bundle)
    monkeypatch.setenv(DECRYPT_KEY_ENV, json.dumps({"new-key": _encoded(DECRYPT_KEY)}))
    with pytest.raises(ReplayCliError, match="BUNDLE_INVALID"):
        run(bundle, receipt_path)


def test_authenticated_audit_requires_independent_verification_keyring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authentication = AdmissionAuthenticator(
        "admission-2026", AUTH_KEY, clock=lambda: 1_700_000_000
    ).authenticate(_audit(), replay_id="a" * 32)
    bundle = tmp_path / "private.bundle"
    receipt_path = tmp_path / "receipt.json"
    _write_bundle(bundle, private_bundle=_private_bundle(authentication=authentication))

    with pytest.raises(ReplayCliError, match="KEYRING_INVALID"):
        run(bundle, receipt_path)

    monkeypatch.setenv(AUTH_KEY_ENV, json.dumps({"admission-2026": _encoded(AUTH_KEY)}))
    receipt = run(bundle, receipt_path)
    assert receipt["authentication_verified"] is True


def test_private_file_and_receipt_creation_are_fail_closed(tmp_path: Path) -> None:
    bundle = tmp_path / "private.bundle"
    receipt_path = tmp_path / "receipt.json"
    _write_bundle(bundle)
    bundle.chmod(0o640)
    with pytest.raises(ReplayCliError, match="BUNDLE_FILE_UNSAFE"):
        run(bundle, receipt_path)

    bundle.chmod(0o600)
    receipt_path.write_text("existing", encoding="utf-8")
    with pytest.raises(ReplayCliError, match="RECEIPT_CREATE_FAILED"):
        run(bundle, receipt_path)
    assert receipt_path.read_text(encoding="utf-8") == "existing"


def test_symlink_and_duplicate_outer_fields_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.bundle"
    link = tmp_path / "link.bundle"
    receipt_path = tmp_path / "receipt.json"
    _write_bundle(target)
    link.symlink_to(target)
    with pytest.raises(ReplayCliError, match="BUNDLE_OPEN_FAILED"):
        run(link, receipt_path)

    duplicate = tmp_path / "duplicate.bundle"
    duplicate.write_text(
        '{"schema_version":"generation-replay-encrypted-bundle/v1",'
        '"schema_version":"generation-replay-encrypted-bundle/v1",'
        '"key_id":"archive-2026","nonce_b64":"x","ciphertext_b64":"x"}',
        encoding="utf-8",
    )
    duplicate.chmod(0o600)
    with pytest.raises(ReplayCliError, match="JSON_DUPLICATE_KEY"):
        run(duplicate, receipt_path)


def test_keyring_and_private_schema_are_bounded(tmp_path: Path) -> None:
    bundle = tmp_path / "private.bundle"
    receipt_path = tmp_path / "receipt.json"
    invalid = _private_bundle()
    invalid["unexpected"] = True
    _write_bundle(bundle, private_bundle=invalid)
    with pytest.raises(ReplayCliError, match="PRIVATE_BUNDLE_INVALID"):
        run(bundle, receipt_path)

    with pytest.raises(ReplayCliError, match="KEY_ID_INVALID"):
        seal_private_replay_bundle(_private_bundle(), key_id="", key=DECRYPT_KEY)


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777
