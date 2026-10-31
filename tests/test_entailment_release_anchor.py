from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from src.entailment_release_anchor import (
    KmsEd25519Signer,
    ReleaseHeadAnchorError,
    create_release_head_anchor,
    dump_release_head_anchor,
    load_release_head_anchor,
    verify_release_head_anchor,
)
from src.entailment_release_log import (
    RELEASE_LOG_GENESIS_SHA256,
    append_verified_candidate_release,
)
from tests.test_entailment_release_manifest import (
    KEY,
    candidate,
    global_comparison,
    later_manifest,
    manifest,
    slice_comparison,
)

SOURCE_COMMIT = "a" * 40
KEY_ID = "kms-ed25519-release-2026-q4"


class LocalSigner:
    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        self.private_key = private_key

    def sign(self, message: bytes) -> bytes:
        return self.private_key.sign(message)


def release_log_payload(tmp_path: Path) -> bytes:
    path = tmp_path / "release-log.jsonl"
    append_verified_candidate_release(
        path,
        manifest(),
        global_comparison(),
        slice_comparison(),
        candidate(),
        keys={"release-key-2026-10": KEY},
        current_epoch=150,
        expected_head_sha256=RELEASE_LOG_GENESIS_SHA256,
    )
    return path.read_bytes()


def key_material() -> tuple[Ed25519PrivateKey, bytes]:
    private_key = Ed25519PrivateKey.generate()
    public_der = private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )
    return private_key, public_der


def test_asymmetric_anchor_round_trip_binds_exact_log_and_commit(
    tmp_path: Path,
) -> None:
    payload = release_log_payload(tmp_path)
    private_key, public_der = key_material()
    anchor = create_release_head_anchor(
        payload,
        key_id=KEY_ID,
        source_commit_sha=SOURCE_COMMIT,
        issued_at_epoch=1_800_000_000,
        signer=LocalSigner(private_key),
    )
    loaded = load_release_head_anchor(dump_release_head_anchor(anchor))

    snapshot = verify_release_head_anchor(
        loaded,
        payload,
        public_keys={KEY_ID: public_der},
        expected_source_commit_sha=SOURCE_COMMIT,
    )

    assert loaded == anchor
    assert snapshot.head_sha256 == anchor.head_sha256
    assert anchor.entry_count == 1
    assert len(anchor.anchor_sha256) == 64


def test_anchor_rejects_log_commit_key_and_signature_substitution(
    tmp_path: Path,
) -> None:
    payload = release_log_payload(tmp_path)
    private_key, public_der = key_material()
    anchor = create_release_head_anchor(
        payload,
        key_id=KEY_ID,
        source_commit_sha=SOURCE_COMMIT,
        issued_at_epoch=1_800_000_000,
        signer=LocalSigner(private_key),
    )
    _, wrong_public_der = key_material()
    changed_signature = ("A" if anchor.signature[0] != "A" else "B") + anchor.signature[
        1:
    ]

    cases = (
        (anchor, payload + b"\n", public_der, SOURCE_COMMIT, "release log"),
        (anchor, payload, public_der, "b" * 40, "source commit"),
        (anchor, payload, wrong_public_der, SOURCE_COMMIT, "signature mismatch"),
        (
            replace(anchor, signature=changed_signature),
            payload,
            public_der,
            SOURCE_COMMIT,
            "signature mismatch",
        ),
    )
    for candidate_anchor, candidate_payload, key, commit, error in cases:
        with pytest.raises(ReleaseHeadAnchorError, match=error):
            verify_release_head_anchor(
                candidate_anchor,
                candidate_payload,
                public_keys={KEY_ID: key},
                expected_source_commit_sha=commit,
            )


def test_anchor_detects_structurally_valid_tail_truncation(tmp_path: Path) -> None:
    path = tmp_path / "release-log.jsonl"
    first = append_verified_candidate_release(
        path,
        manifest(),
        global_comparison(),
        slice_comparison(),
        candidate(),
        keys={"release-key-2026-10": KEY},
        current_epoch=150,
        expected_head_sha256=RELEASE_LOG_GENESIS_SHA256,
    )
    first_payload = path.read_bytes()
    second_manifest, second_candidate = later_manifest()
    append_verified_candidate_release(
        path,
        second_manifest,
        global_comparison(),
        slice_comparison(),
        second_candidate,
        keys={"release-key-2026-10": KEY},
        current_epoch=250,
        expected_head_sha256=first.entry_sha256,
    )
    full_payload = path.read_bytes()
    private_key, public_der = key_material()
    anchor = create_release_head_anchor(
        full_payload,
        key_id=KEY_ID,
        source_commit_sha=SOURCE_COMMIT,
        issued_at_epoch=1_800_000_000,
        signer=LocalSigner(private_key),
    )

    with pytest.raises(ReleaseHeadAnchorError, match="does not bind"):
        verify_release_head_anchor(
            anchor,
            first_payload,
            public_keys={KEY_ID: public_der},
            expected_source_commit_sha=SOURCE_COMMIT,
        )


def test_anchor_loader_rejects_duplicate_extra_and_noncanonical_fields(
    tmp_path: Path,
) -> None:
    payload = release_log_payload(tmp_path)
    private_key, _ = key_material()
    anchor_payload = dump_release_head_anchor(
        create_release_head_anchor(
            payload,
            key_id=KEY_ID,
            source_commit_sha=SOURCE_COMMIT,
            issued_at_epoch=1_800_000_000,
            signer=LocalSigner(private_key),
        )
    )
    duplicate = anchor_payload[:-1] + b',"purpose":"rag-release-log-head-anchor"}'
    extra = json.loads(anchor_payload)
    extra["extra"] = True

    with pytest.raises(ReleaseHeadAnchorError, match="duplicate field"):
        load_release_head_anchor(duplicate)
    with pytest.raises(ReleaseHeadAnchorError, match="fields do not match"):
        load_release_head_anchor(
            json.dumps(extra, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
    with pytest.raises(ReleaseHeadAnchorError, match="not canonical"):
        load_release_head_anchor(json.dumps(json.loads(anchor_payload)).encode("ascii"))


def test_kms_adapter_uses_raw_ed25519_and_validates_response() -> None:
    private_key, _ = key_material()
    calls: list[dict[str, object]] = []

    class Client:
        def sign(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return {
                "KeyId": "arn:aws:kms:eu-central-1:123456789012:key/example",
                "SigningAlgorithm": "ED25519_SHA_512",
                "Signature": private_key.sign(kwargs["Message"]),
            }

    signature = KmsEd25519Signer(Client(), kms_key_id="alias/rag-release").sign(
        b"anchor"
    )

    assert len(signature) == 64
    assert calls == [
        {
            "KeyId": "alias/rag-release",
            "Message": b"anchor",
            "MessageType": "RAW",
            "SigningAlgorithm": "ED25519_SHA_512",
        }
    ]


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {
            "KeyId": "key",
            "SigningAlgorithm": "ECDSA_SHA_256",
            "Signature": b"x" * 64,
        },
        {
            "KeyId": "key",
            "SigningAlgorithm": "ED25519_SHA_512",
            "Signature": b"short",
        },
    ],
)
def test_kms_adapter_rejects_malformed_responses(response: object) -> None:
    client = SimpleNamespace(sign=lambda **kwargs: response)

    with pytest.raises(ReleaseHeadAnchorError, match="response is malformed"):
        KmsEd25519Signer(client, kms_key_id="alias/rag-release").sign(b"anchor")
