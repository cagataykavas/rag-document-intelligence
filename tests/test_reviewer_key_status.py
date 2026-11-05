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
    reviewer_key_status_authority_policy_sha256,
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
    dict[str, Ed25519PrivateKey],
    dict[str, bytes],
]:
    reviewer_private = {
        f"reviewer-{index}": Ed25519PrivateKey.generate() for index in range(1, 4)
    }
    reviewer_public = {
        key_id: _public_key(private_key)
        for key_id, private_key in reviewer_private.items()
    }
    authority_private = {
        f"status-authority-{index}": Ed25519PrivateKey.generate()
        for index in range(1, 4)
    }
    authority_public = {
        key_id: _public_key(private_key)
        for key_id, private_key in authority_private.items()
    }
    return (
        reviewer_private,
        reviewer_public,
        authority_private,
        authority_public,
    )


def _snapshot(
    *,
    statuses: dict[str, str] | None = None,
    issued_at_epoch: int = ISSUED_AT,
    expires_at_epoch: int = EXPIRES_AT,
):
    _, reviewers, authority_private, authority_public = _keys()
    snapshot = create_reviewer_key_status_snapshot(
        review_policy_sha256=REVIEW_POLICY_SHA256,
        statuses=statuses or {key_id: "active" for key_id in reviewers},
        issued_at_epoch=issued_at_epoch,
        expires_at_epoch=expires_at_epoch,
        authority_public_keys=authority_public,
        required_authority_signatures=2,
        signers={
            key_id: authority_private[key_id]
            for key_id in ("status-authority-1", "status-authority-3")
        },
    )
    return snapshot, reviewers, authority_private, authority_public


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
        required_authority_signatures=2,
        verification_epoch=VERIFICATION_EPOCH,
    )

    assert loaded == snapshot
    assert receipt.snapshot_sha256 == snapshot.snapshot_sha256
    assert receipt.active_key_count == 3
    assert receipt.revoked_key_count == 0
    serialized = json.dumps(receipt.to_audit_dict(), sort_keys=True)
    assert "reviewer-1" not in serialized
    assert all(item.signature not in serialized for item in snapshot.signatures)
    assert (
        snapshot.authority_policy_sha256
        == reviewer_key_status_authority_policy_sha256(
            authorities, required_signatures=2
        )
    )


def test_status_rejects_authority_threshold_substitution_and_missing_quorum() -> None:
    snapshot, reviewers, authority_private, authorities = _snapshot()
    with pytest.raises(ReviewerKeyStatusError, match="authority policy mismatch"):
        verify_reviewer_key_status_snapshot(
            snapshot,
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys=authorities,
            required_authority_signatures=3,
            verification_epoch=VERIFICATION_EPOCH,
        )

    threshold_three = create_reviewer_key_status_snapshot(
        review_policy_sha256=REVIEW_POLICY_SHA256,
        statuses={key_id: "active" for key_id in reviewers},
        issued_at_epoch=ISSUED_AT,
        expires_at_epoch=EXPIRES_AT,
        authority_public_keys=authorities,
        required_authority_signatures=3,
        signers=authority_private,
    )
    with pytest.raises(ReviewerKeyStatusError, match="quorum is not satisfied"):
        verify_reviewer_key_status_snapshot(
            replace(threshold_three, signatures=threshold_three.signatures[:2]),
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys=authorities,
            required_authority_signatures=3,
            verification_epoch=VERIFICATION_EPOCH,
        )


def test_authority_policy_rejects_aliases_and_single_key_thresholds() -> None:
    authority = Ed25519PrivateKey.generate()
    aliases = {
        "status-authority-1": _public_key(authority),
        "status-authority-2": _public_key(authority),
    }
    with pytest.raises(ReviewerKeyStatusError, match="must be distinct"):
        reviewer_key_status_authority_policy_sha256(aliases, required_signatures=2)

    _, _, _, authorities = _keys()
    with pytest.raises(ReviewerKeyStatusError, match="must be in"):
        reviewer_key_status_authority_policy_sha256(authorities, required_signatures=1)


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
            required_authority_signatures=2,
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
            required_authority_signatures=2,
            verification_epoch=verification_epoch,
        )


def test_status_rejects_policy_coverage_and_authority_substitution() -> None:
    snapshot, reviewers, authority_private, authorities = _snapshot()
    missing = create_reviewer_key_status_snapshot(
        review_policy_sha256=REVIEW_POLICY_SHA256,
        statuses={"reviewer-1": "active", "reviewer-2": "active"},
        issued_at_epoch=ISSUED_AT,
        expires_at_epoch=EXPIRES_AT,
        authority_public_keys=authorities,
        required_authority_signatures=2,
        signers={
            key_id: authority_private[key_id]
            for key_id in ("status-authority-1", "status-authority-2")
        },
    )
    wrong_authority = Ed25519PrivateKey.generate()

    with pytest.raises(ReviewerKeyStatusError, match="review policy mismatch"):
        verify_reviewer_key_status_snapshot(
            snapshot,
            expected_review_policy_sha256="b" * 64,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys=authorities,
            required_authority_signatures=2,
            verification_epoch=VERIFICATION_EPOCH,
        )
    with pytest.raises(ReviewerKeyStatusError, match="does not cover"):
        verify_reviewer_key_status_snapshot(
            missing,
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys=authorities,
            required_authority_signatures=2,
            verification_epoch=VERIFICATION_EPOCH,
        )
    with pytest.raises(ReviewerKeyStatusError, match="authority policy mismatch"):
        verify_reviewer_key_status_snapshot(
            snapshot,
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys={
                "other-authority-1": _public_key(wrong_authority),
                "other-authority-2": _public_key(Ed25519PrivateKey.generate()),
            },
            required_authority_signatures=2,
            verification_epoch=VERIFICATION_EPOCH,
        )


def test_status_rejects_reviewer_key_as_status_authority() -> None:
    reviewer_private, reviewers, _, _ = _keys()
    independent = Ed25519PrivateKey.generate()
    authority_public = {
        "reviewer-1": reviewers["reviewer-1"],
        "status-authority-2": _public_key(independent),
    }
    snapshot = create_reviewer_key_status_snapshot(
        review_policy_sha256=REVIEW_POLICY_SHA256,
        statuses={key_id: "active" for key_id in reviewers},
        issued_at_epoch=ISSUED_AT,
        expires_at_epoch=EXPIRES_AT,
        authority_public_keys=authority_public,
        required_authority_signatures=2,
        signers={
            "reviewer-1": reviewer_private["reviewer-1"],
            "status-authority-2": independent,
        },
    )

    with pytest.raises(ReviewerKeyStatusError, match="must be independent"):
        verify_reviewer_key_status_snapshot(
            snapshot,
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys=authority_public,
            required_authority_signatures=2,
            verification_epoch=VERIFICATION_EPOCH,
        )


def test_status_creation_rejects_wrong_signer_and_sanitizes_provider_errors() -> None:
    _, reviewers, authority_private, authorities = _keys()
    arguments = {
        "review_policy_sha256": REVIEW_POLICY_SHA256,
        "statuses": {key_id: "active" for key_id in reviewers},
        "issued_at_epoch": ISSUED_AT,
        "expires_at_epoch": EXPIRES_AT,
        "authority_public_keys": authorities,
        "required_authority_signatures": 2,
    }
    with pytest.raises(ReviewerKeyStatusError, match="does not match"):
        create_reviewer_key_status_snapshot(
            **arguments,
            signers={
                "status-authority-1": Ed25519PrivateKey.generate(),
                "status-authority-2": authority_private["status-authority-2"],
            },
        )
    with pytest.raises(ReviewerKeyStatusError, match="status signing failed") as error:
        create_reviewer_key_status_snapshot(
            **arguments,
            signers={
                "status-authority-1": _FailingSigner(),
                "status-authority-2": authority_private["status-authority-2"],
            },
        )
    assert "private provider diagnostic" not in str(error.value)


def test_status_rejects_signature_tampering_and_noncanonical_json() -> None:
    snapshot, reviewers, _, authorities = _snapshot()
    first = snapshot.signatures[0]
    changed = ("A" if first.signature[0] != "A" else "B") + first.signature[1:]
    tampered = replace(
        snapshot,
        signatures=(
            replace(first, signature=changed),
            *snapshot.signatures[1:],
        ),
    )

    with pytest.raises(ReviewerKeyStatusError, match="signature mismatch"):
        verify_reviewer_key_status_snapshot(
            tampered,
            expected_review_policy_sha256=REVIEW_POLICY_SHA256,
            reviewer_public_keys=reviewers,
            required_active_key_ids=("reviewer-1",),
            authority_public_keys=authorities,
            required_authority_signatures=2,
            verification_epoch=VERIFICATION_EPOCH,
        )
    with pytest.raises(ReviewerKeyStatusError, match="unique sorted key ids"):
        dump_reviewer_key_status_snapshot(replace(snapshot, signatures=(first, first)))
    payload = dump_reviewer_key_status_snapshot(snapshot)
    with pytest.raises(ReviewerKeyStatusError, match="not canonical"):
        load_reviewer_key_status_snapshot(
            json.dumps(json.loads(payload)).encode("ascii")
        )
    duplicate = payload.replace(b'"purpose":', b'"purpose":"duplicate","purpose":', 1)
    with pytest.raises(ReviewerKeyStatusError, match="duplicate field"):
        load_reviewer_key_status_snapshot(duplicate)
