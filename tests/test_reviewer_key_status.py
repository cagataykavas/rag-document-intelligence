from __future__ import annotations

import json
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from src.reviewer_key_status import (
    MAX_STATUS_LIFETIME_SECONDS,
    ReviewerKeyStatusError,
    create_reviewer_key_status_snapshot,
    dump_reviewer_key_status_snapshot,
    load_reviewer_key_status_snapshot,
    verify_reviewer_key_status_snapshot,
)

REVIEW_POLICY_SHA256 = "a" * 64
ISSUED_AT = 1_800_000_000
EXPIRES_AT = ISSUED_AT + 3_600
VERIFICATION_EPOCH = ISSUED_AT + 60


class _FailingSigner:
    def sign(self, _message: bytes) -> bytes:
        raise RuntimeError("private provider diagnostic")


def _public_key(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )


def _keys() -> tuple[
    dict[str, Ed25519PrivateKey],
    dict[str, bytes],
    Ed25519PrivateKey,
    dict[str, bytes],
]:
    reviewer_private = {
        f"reviewer-{index}": Ed25519PrivateKey.generate() for index in range(1, 4)
    }
    reviewer_public = {
        key_id: _public_key(private_key)
        for key_id, private_key in reviewer_private.items()
    }
    authority = Ed25519PrivateKey.generate()
    return (
        reviewer_private,
        reviewer_public,
        authority,
        {"reviewer-status-authority": _public_key(authority)},
    )


def _snapshot(
    *,
    statuses: dict[str, str] | None = None,
    issued_at_epoch: int = ISSUED_AT,
    expires_at_epoch: int = EXPIRES_AT,
):
    _, reviewers, authority, authorities = _keys()
    snapshot = create_reviewer_key_status_snapshot(
        review_policy_sha256=REVIEW_POLICY_SHA256,
        statuses=statuses or {key_id: "active" for key_id in reviewers},
        issued_at_epoch=issued_at_epoch,
        expires_at_epoch=expires_at_epoch,
        authority_key_id="reviewer-status-authority",
        authority_public_key=authorities["reviewer-status-authority"],
        signer=authority,
    )
    return snapshot, reviewers, authority, authorities


def test_status_round_trip_binds_policy_authority_freshness_and_coverage() -> None:
    snapshot, reviewers, _, authorities = _snapshot()
    loaded = load_reviewer_key_status_snapshot(
        dump_reviewer_key_status_snapshot(snapshot)
    )

    receipt = verify_reviewer_key_status_snapshot(
        loaded,
        expected_review_policy_sha256=REVIEW_POLICY_SHA256,
        reviewer_public_keys=reviewers,
        required_active_key_ids=("reviewer-1", "reviewer-3"),
        authority_public_keys=authorities,
        verification_epoch=VERIFICATION_EPOCH,
    )

    assert loaded == snapshot
    assert receipt.snapshot_sha256 == snapshot.snapshot_sha256
    assert receipt.active_key_count == 3
    assert receipt.revoked_key_count == 0
    serialized = json.dumps(receipt.to_audit_dict(), sort_keys=True)
    assert "reviewer-1" not in serialized
    assert snapshot.signature not in serialized


def test_status_rejects_revoked_review_signer_before_quorum_use() -> None:
    snapshot, reviewers, _, authorities = _snapshot(
        statuses={
            "reviewer-1": "active",
            "reviewer-2": "active",
            "reviewer-3": "revoked",
        }
    )

    with pytest.raises(ReviewerKeyStatusError, match="revoked signer"):
        verify_reviewer_key_status_snapshot(
            snapshot,
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1", "reviewer-3"),
            authority_public_keys=authorities,
            verification_epoch=VERIFICATION_EPOCH,
        )


@pytest.mark.parametrize(
    ("issued_at", "expires_at", "verification_epoch", "message"),
    [
        (ISSUED_AT, EXPIRES_AT, EXPIRES_AT + 1, "expired"),
        (ISSUED_AT, EXPIRES_AT, ISSUED_AT - 1, "future"),
        (
            ISSUED_AT,
            ISSUED_AT + MAX_STATUS_LIFETIME_SECONDS + 1,
            VERIFICATION_EPOCH,
            "validity window",
        ),
    ],
)
def test_status_rejects_stale_future_and_overlong_snapshots(
    issued_at: int,
    expires_at: int,
    verification_epoch: int,
    message: str,
) -> None:
    if expires_at - issued_at > MAX_STATUS_LIFETIME_SECONDS:
        with pytest.raises(ReviewerKeyStatusError, match=message):
            _snapshot(issued_at_epoch=issued_at, expires_at_epoch=expires_at)
        return
    snapshot, reviewers, _, authorities = _snapshot(
        issued_at_epoch=issued_at,
        expires_at_epoch=expires_at,
    )
    with pytest.raises(ReviewerKeyStatusError, match=message):
        verify_reviewer_key_status_snapshot(
            snapshot,
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys=authorities,
            verification_epoch=verification_epoch,
        )


def test_status_rejects_policy_coverage_and_authority_substitution() -> None:
    snapshot, reviewers, authority, authorities = _snapshot()
    missing = create_reviewer_key_status_snapshot(
        review_policy_sha256=REVIEW_POLICY_SHA256,
        statuses={"reviewer-1": "active", "reviewer-2": "active"},
        issued_at_epoch=ISSUED_AT,
        expires_at_epoch=EXPIRES_AT,
        authority_key_id="reviewer-status-authority",
        authority_public_key=authorities["reviewer-status-authority"],
        signer=authority,
    )
    wrong_authority = Ed25519PrivateKey.generate()

    with pytest.raises(ReviewerKeyStatusError, match="review policy mismatch"):
        verify_reviewer_key_status_snapshot(
            snapshot,
            expected_review_policy_sha256="b" * 64,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys=authorities,
            verification_epoch=VERIFICATION_EPOCH,
        )
    with pytest.raises(ReviewerKeyStatusError, match="does not cover"):
        verify_reviewer_key_status_snapshot(
            missing,
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys=authorities,
            verification_epoch=VERIFICATION_EPOCH,
        )
    with pytest.raises(ReviewerKeyStatusError, match="not trusted"):
        verify_reviewer_key_status_snapshot(
            snapshot,
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys={"other-authority": _public_key(wrong_authority)},
            verification_epoch=VERIFICATION_EPOCH,
        )


def test_status_rejects_reviewer_key_as_status_authority() -> None:
    reviewer_private, reviewers, _, _ = _keys()
    snapshot = create_reviewer_key_status_snapshot(
        review_policy_sha256=REVIEW_POLICY_SHA256,
        statuses={key_id: "active" for key_id in reviewers},
        issued_at_epoch=ISSUED_AT,
        expires_at_epoch=EXPIRES_AT,
        authority_key_id="reviewer-1",
        authority_public_key=reviewers["reviewer-1"],
        signer=reviewer_private["reviewer-1"],
    )

    with pytest.raises(ReviewerKeyStatusError, match="must be independent"):
        verify_reviewer_key_status_snapshot(
            snapshot,
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys={"reviewer-1": reviewers["reviewer-1"]},
            verification_epoch=VERIFICATION_EPOCH,
        )


def test_status_creation_rejects_wrong_signer_and_sanitizes_provider_errors() -> None:
    _, reviewers, authority, authorities = _keys()
    arguments = {
        "review_policy_sha256": REVIEW_POLICY_SHA256,
        "statuses": {key_id: "active" for key_id in reviewers},
        "issued_at_epoch": ISSUED_AT,
        "expires_at_epoch": EXPIRES_AT,
        "authority_key_id": "reviewer-status-authority",
        "authority_public_key": authorities["reviewer-status-authority"],
    }
    with pytest.raises(ReviewerKeyStatusError, match="does not match"):
        create_reviewer_key_status_snapshot(
            **arguments,
            signer=Ed25519PrivateKey.generate(),
        )
    with pytest.raises(ReviewerKeyStatusError, match="status signing failed") as error:
        create_reviewer_key_status_snapshot(**arguments, signer=_FailingSigner())
    assert "private provider diagnostic" not in str(error.value)


def test_status_rejects_signature_tampering_and_noncanonical_json() -> None:
    snapshot, reviewers, _, authorities = _snapshot()
    changed = ("A" if snapshot.signature[0] != "A" else "B") + snapshot.signature[1:]

    with pytest.raises(ReviewerKeyStatusError, match="signature mismatch"):
        verify_reviewer_key_status_snapshot(
            replace(snapshot, signature=changed),
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys=authorities,
            verification_epoch=VERIFICATION_EPOCH,
        )
    payload = dump_reviewer_key_status_snapshot(snapshot)
    with pytest.raises(ReviewerKeyStatusError, match="not canonical"):
        load_reviewer_key_status_snapshot(
            json.dumps(json.loads(payload)).encode("ascii")
        )
    duplicate = payload.replace(b'"purpose":', b'"purpose":"duplicate","purpose":', 1)
    with pytest.raises(ReviewerKeyStatusError, match="duplicate field"):
        load_reviewer_key_status_snapshot(duplicate)
