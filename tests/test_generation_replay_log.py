from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.generation_evidence_replay import (
    SCHEMA_VERSION,
    GenerationEvidenceReplayError,
    validate_generation_evidence_receipt,
)
from src.generation_replay_log import (
    GENERATION_REPLAY_LOG_GENESIS_SHA256,
    GenerationReplayLogError,
    append_generation_replay_receipt,
    verify_generation_replay_log,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _receipt(seed: str = "one") -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "applied": True,
        "verified": True,
        "accepted": True,
        "authentication_verified": False,
        "evidence_chunk_count": 1,
        "claim_count": 1,
        "payload_sha256": _digest(f"payload-{seed}"),
        "evidence_snapshot_sha256": _digest(f"evidence-{seed}"),
        "claim_support_sha256": _digest(f"support-{seed}"),
        "decision_sha256": _digest(f"decision-{seed}"),
    }
    document["receipt_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    return document


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def test_append_and_verify_privacy_reduced_chain(tmp_path: Path) -> None:
    directory = _private_directory(tmp_path / "review")
    path = directory / "generation-replay.jsonl"
    receipt = _receipt()

    entry = append_generation_replay_receipt(
        path,
        receipt,
        expected_head_sha256=GENERATION_REPLAY_LOG_GENESIS_SHA256,
    )
    snapshot = verify_generation_replay_log(
        path,
        expected_head_sha256=entry.entry_sha256,
    )

    assert entry.sequence == 1
    assert entry.previous_entry_sha256 == GENERATION_REPLAY_LOG_GENESIS_SHA256
    assert entry.replay_receipt_sha256 == receipt["receipt_sha256"]
    assert entry.claim_support_sha256 == receipt["claim_support_sha256"]
    assert entry.evidence_chunk_count == 1
    assert entry.claim_count == 1
    assert snapshot.entries == (entry,)
    assert snapshot.head_sha256 == entry.entry_sha256
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_exact_concurrent_retries_append_once(tmp_path: Path) -> None:
    path = _private_directory(tmp_path / "review") / "generation-replay.jsonl"
    receipt = _receipt()

    def append(_index: int):
        return append_generation_replay_receipt(
            path,
            receipt,
            expected_head_sha256=GENERATION_REPLAY_LOG_GENESIS_SHA256,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(append, range(16)))

    snapshot = verify_generation_replay_log(path)
    assert len(snapshot.entries) == 1
    assert {entry.entry_sha256 for entry in entries} == {snapshot.head_sha256}


def test_compare_and_set_rejects_stale_writer_then_accepts_current_head(
    tmp_path: Path,
) -> None:
    path = _private_directory(tmp_path / "review") / "generation-replay.jsonl"
    first = append_generation_replay_receipt(
        path,
        _receipt("first"),
        expected_head_sha256=GENERATION_REPLAY_LOG_GENESIS_SHA256,
    )

    with pytest.raises(GenerationReplayLogError, match="STALE_HEAD"):
        append_generation_replay_receipt(
            path,
            _receipt("second"),
            expected_head_sha256=GENERATION_REPLAY_LOG_GENESIS_SHA256,
        )
    second = append_generation_replay_receipt(
        path,
        _receipt("second"),
        expected_head_sha256=first.entry_sha256,
    )

    snapshot = verify_generation_replay_log(path)
    assert second.sequence == 2
    assert second.previous_entry_sha256 == first.entry_sha256
    assert snapshot.head_sha256 == second.entry_sha256


def test_receipt_validation_rejects_tampering_and_schema_extension(
    tmp_path: Path,
) -> None:
    path = _private_directory(tmp_path / "review") / "generation-replay.jsonl"
    tampered = _receipt()
    tampered["accepted"] = False
    extended = {**_receipt(), "answer": "private content"}

    with pytest.raises(GenerationEvidenceReplayError, match="RECEIPT_INVALID"):
        validate_generation_evidence_receipt(tampered)
    with pytest.raises(GenerationReplayLogError, match="RECEIPT_INVALID"):
        append_generation_replay_receipt(
            path,
            extended,
            expected_head_sha256=GENERATION_REPLAY_LOG_GENESIS_SHA256,
        )
    assert not path.exists()


@pytest.mark.parametrize("mutation", ["truncated", "tampered", "extended"])
def test_verifier_rejects_chain_mutation(tmp_path: Path, mutation: str) -> None:
    path = _private_directory(tmp_path / "review") / "generation-replay.jsonl"
    append_generation_replay_receipt(
        path,
        _receipt(),
        expected_head_sha256=GENERATION_REPLAY_LOG_GENESIS_SHA256,
    )
    payload = path.read_bytes()
    if mutation == "truncated":
        path.write_bytes(payload.rstrip(b"\n"))
    else:
        document = json.loads(payload)
        if mutation == "tampered":
            document["payload_sha256"] = "f" * 64
        else:
            document["operator_note"] = "not allowed"
        path.write_bytes(_canonical(document) + b"\n")
    path.chmod(0o600)

    with pytest.raises(GenerationReplayLogError):
        verify_generation_replay_log(path)


def test_file_and_parent_safety_fail_closed(tmp_path: Path) -> None:
    public_directory = tmp_path / "public"
    public_directory.mkdir(mode=0o755)
    public_directory.chmod(0o755)
    with pytest.raises(GenerationReplayLogError, match="PARENT_UNSAFE"):
        append_generation_replay_receipt(
            public_directory / "log.jsonl",
            _receipt(),
            expected_head_sha256=GENERATION_REPLAY_LOG_GENESIS_SHA256,
        )

    directory = _private_directory(tmp_path / "review")
    target = directory / "target.jsonl"
    target.write_bytes(b"")
    target.chmod(0o600)
    hardlink = directory / "hardlink.jsonl"
    os.link(target, hardlink)
    with pytest.raises(GenerationReplayLogError, match="LOG_FILE_UNSAFE"):
        append_generation_replay_receipt(
            hardlink,
            _receipt(),
            expected_head_sha256=GENERATION_REPLAY_LOG_GENESIS_SHA256,
        )

    symlink = directory / "symlink.jsonl"
    symlink.symlink_to(target)
    with pytest.raises(GenerationReplayLogError, match="LOG_UNAVAILABLE"):
        verify_generation_replay_log(symlink)


def test_duplicate_json_fields_and_wrong_pinned_head_are_rejected(
    tmp_path: Path,
) -> None:
    path = _private_directory(tmp_path / "review") / "generation-replay.jsonl"
    entry = append_generation_replay_receipt(
        path,
        _receipt(),
        expected_head_sha256=GENERATION_REPLAY_LOG_GENESIS_SHA256,
    )
    original = path.read_text(encoding="utf-8").rstrip("\n")
    path.write_text(original[:-1] + ',"sequence":1}\n', encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(GenerationReplayLogError, match="DUPLICATE_FIELD"):
        verify_generation_replay_log(path)

    path.write_bytes(_canonical(entry.to_audit_dict()) + b"\n")
    path.chmod(0o600)
    with pytest.raises(GenerationReplayLogError, match="HEAD_MISMATCH"):
        verify_generation_replay_log(path, expected_head_sha256="f" * 64)
