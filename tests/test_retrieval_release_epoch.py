from __future__ import annotations

import json
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from src.retrieval_release_epoch import (
    MAX_RELEASE_EPOCH_BYTES,
    RELEASE_EPOCH_RECEIPT_SCHEMA_VERSION,
    ReleaseEpochError,
    ReleaseEpochPolicy,
    create_release_epoch_attestation,
    dump_release_epoch_attestation,
    load_release_epoch_attestation,
    release_artifact_set_sha256,
    verify_release_epoch_attestation,
)

CANDIDATE = "a" * 64
BASELINE = "b" * 64
REVIEW = "c" * 64
STATUS = "d" * 64
VERIFICATION_EPOCH = 1_800_000_060
SOURCE_REVISION = "e" * 40


class _FailingSigner:
    def sign(self, _message: bytes) -> bytes:
        raise RuntimeError("private provider diagnostic")


def _public_key(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )


def _policy(
    private_key: Ed25519PrivateKey,
    **overrides: str,
) -> ReleaseEpochPolicy:
    arguments = {
        "repository": "cagataykavas/rag-document-intelligence",
        "workflow": ".github/workflows/release-retrieval.yml",
        "source_ref": "refs/heads/main",
        "source_revision": SOURCE_REVISION,
        "key_id": "retrieval-release-epoch-v1",
    }
    arguments.update(overrides)
    return ReleaseEpochPolicy(public_key=_public_key(private_key), **arguments)


def _attestation(
    private_key: Ed25519PrivateKey | None = None,
):
    private_key = private_key or Ed25519PrivateKey.generate()
    policy = _policy(private_key)
    attestation = create_release_epoch_attestation(
        candidate_snapshot_sha256=CANDIDATE,
        baseline_artifact_sha256=BASELINE,
        inventory_review_sha256=REVIEW,
        key_status_snapshot_sha256=STATUS,
        verification_epoch=VERIFICATION_EPOCH,
        policy=policy,
        signer=private_key,
    )
    return attestation, policy, private_key


def _verify(attestation, policy, **overrides: str):
    arguments = {
        "candidate_snapshot_sha256": CANDIDATE,
        "baseline_artifact_sha256": BASELINE,
        "inventory_review_sha256": REVIEW,
        "key_status_snapshot_sha256": STATUS,
    }
    arguments.update(overrides)
    return verify_release_epoch_attestation(
        attestation,
        policy=policy,
        **arguments,
    )


def test_round_trip_authenticates_epoch_policy_and_exact_artifacts() -> None:
    attestation, policy, _ = _attestation()
    loaded = load_release_epoch_attestation(dump_release_epoch_attestation(attestation))

    receipt = _verify(loaded, policy)

    assert loaded == attestation
    assert receipt.schema_version == RELEASE_EPOCH_RECEIPT_SCHEMA_VERSION
    assert receipt.attestation_sha256 == attestation.attestation_sha256
    assert receipt.policy_sha256 == policy.sha256
    assert receipt.verification_epoch == VERIFICATION_EPOCH
    assert receipt.artifact_set_sha256 == release_artifact_set_sha256(
        candidate_snapshot_sha256=CANDIDATE,
        baseline_artifact_sha256=BASELINE,
        inventory_review_sha256=REVIEW,
        key_status_snapshot_sha256=STATUS,
    )
    serialized = json.dumps(receipt.to_audit_dict(), sort_keys=True)
    assert policy.repository not in serialized
    assert policy.workflow not in serialized
    assert policy.key_id not in serialized


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        ("candidate_snapshot_sha256", "artifact set mismatch"),
        ("baseline_artifact_sha256", "artifact set mismatch"),
        ("inventory_review_sha256", "artifact set mismatch"),
        ("key_status_snapshot_sha256", "artifact set mismatch"),
    ],
)
def test_rejects_artifact_substitution(field: str, reason: str) -> None:
    attestation, policy, _ = _attestation()
    with pytest.raises(ReleaseEpochError, match=reason):
        _verify(attestation, policy, **{field: "f" * 64})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repository", "other/repository"),
        ("workflow", ".github/workflows/other.yml"),
        ("source_ref", "refs/heads/release"),
        ("source_revision", "f" * 40),
        ("key_id", "other-key"),
    ],
)
def test_rejects_release_policy_substitution(field: str, value: str) -> None:
    attestation, policy, private_key = _attestation()
    fields = {
        "repository": policy.repository,
        "workflow": policy.workflow,
        "source_ref": policy.source_ref,
        "source_revision": policy.source_revision,
        "key_id": policy.key_id,
    }
    fields[field] = value
    substituted = _policy(private_key, **fields)
    with pytest.raises(ReleaseEpochError, match="policy mismatch"):
        _verify(attestation, substituted)


def test_signature_is_checked_before_semantic_bindings() -> None:
    attestation, policy, _ = _attestation()
    tampered = replace(
        attestation,
        artifact_set_sha256="f" * 64,
    )
    with pytest.raises(ReleaseEpochError, match="signature mismatch"):
        _verify(tampered, policy, candidate_snapshot_sha256="f" * 64)


def test_release_epoch_key_must_be_independent_from_review_roles() -> None:
    attestation, policy, private_key = _attestation()
    with pytest.raises(ReleaseEpochError, match="must be independent"):
        verify_release_epoch_attestation(
            attestation,
            candidate_snapshot_sha256=CANDIDATE,
            baseline_artifact_sha256=BASELINE,
            inventory_review_sha256=REVIEW,
            key_status_snapshot_sha256=STATUS,
            policy=policy,
            disallowed_public_keys=(_public_key(private_key),),
        )


def test_creation_rejects_wrong_signer_and_sanitizes_provider_errors() -> None:
    policy = _policy(Ed25519PrivateKey.generate())
    arguments = {
        "candidate_snapshot_sha256": CANDIDATE,
        "baseline_artifact_sha256": BASELINE,
        "inventory_review_sha256": REVIEW,
        "key_status_snapshot_sha256": STATUS,
        "verification_epoch": VERIFICATION_EPOCH,
        "policy": policy,
    }
    with pytest.raises(ReleaseEpochError, match="does not match"):
        create_release_epoch_attestation(
            **arguments,
            signer=Ed25519PrivateKey.generate(),
        )
    with pytest.raises(ReleaseEpochError, match="signing failed") as error:
        create_release_epoch_attestation(**arguments, signer=_FailingSigner())
    assert "private provider diagnostic" not in str(error.value)


def test_loader_rejects_noncanonical_duplicate_oversized_and_unsupported() -> None:
    attestation, _, _ = _attestation()
    payload = dump_release_epoch_attestation(attestation)
    with pytest.raises(ReleaseEpochError, match="not canonical"):
        load_release_epoch_attestation(json.dumps(json.loads(payload)).encode("ascii"))
    duplicate = payload.replace(b'"purpose":', b'"purpose":"duplicate","purpose":', 1)
    with pytest.raises(ReleaseEpochError, match="duplicate field"):
        load_release_epoch_attestation(duplicate)
    with pytest.raises(ReleaseEpochError, match="payload is invalid"):
        load_release_epoch_attestation(b"{" + b"x" * MAX_RELEASE_EPOCH_BYTES)
    unsupported = replace(attestation, schema_version="release-epoch/v0")
    with pytest.raises(ReleaseEpochError, match="fields are invalid"):
        dump_release_epoch_attestation(unsupported)


@pytest.mark.parametrize(
    "overrides",
    [
        {"repository": "unsafe repository"},
        {"workflow": "../workflow.yml"},
        {"source_ref": "refs/pull/1/head"},
        {"source_revision": "not-a-revision"},
        {"key_id": "unsafe key"},
    ],
)
def test_policy_rejects_unsafe_context(overrides: dict[str, str]) -> None:
    with pytest.raises(ReleaseEpochError, match="policy is malformed"):
        _policy(Ed25519PrivateKey.generate(), **overrides)
