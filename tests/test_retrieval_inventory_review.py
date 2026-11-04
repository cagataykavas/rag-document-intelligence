from __future__ import annotations

import json
from dataclasses import replace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from src.retrieval_inventory_review import (
    InventoryReviewError,
    create_inventory_review_attestation,
    dump_inventory_review_attestation,
    inventory_review_policy_sha256,
    load_inventory_review_attestation,
    verify_inventory_review_attestation,
)

INVENTORY_SHA256 = "a" * 64
BASELINE_SHA256 = "b" * 64


def _reviewers(
    count: int = 3,
) -> tuple[dict[str, Ed25519PrivateKey], dict[str, bytes]]:
    private_keys = {
        f"reviewer-{index}": Ed25519PrivateKey.generate()
        for index in range(1, count + 1)
    }
    public_keys = {
        key_id: private_key.public_key().public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo
        )
        for key_id, private_key in private_keys.items()
    }
    return private_keys, public_keys


def _attestation():
    private_keys, public_keys = _reviewers()
    attestation = create_inventory_review_attestation(
        source_inventory_sha256=INVENTORY_SHA256,
        baseline_artifact_sha256=BASELINE_SHA256,
        public_keys=public_keys,
        required_signatures=2,
        signers={
            key_id: private_keys[key_id] for key_id in ("reviewer-1", "reviewer-3")
        },
    )
    return attestation, private_keys, public_keys


def test_quorum_review_round_trip_binds_inventory_baseline_and_policy() -> None:
    attestation, _, public_keys = _attestation()
    loaded = load_inventory_review_attestation(
        dump_inventory_review_attestation(attestation)
    )

    commitment = verify_inventory_review_attestation(
        loaded,
        expected_source_inventory_sha256=INVENTORY_SHA256,
        expected_baseline_artifact_sha256=BASELINE_SHA256,
        public_keys=public_keys,
        required_signatures=2,
    )

    assert loaded == attestation
    assert commitment == attestation.attestation_sha256
    assert loaded.review_policy_sha256 == inventory_review_policy_sha256(
        public_keys, required_signatures=2
    )
    assert tuple(item.key_id for item in loaded.signatures) == (
        "reviewer-1",
        "reviewer-3",
    )


@pytest.mark.parametrize(
    ("inventory", "baseline", "message"),
    [
        ("c" * 64, BASELINE_SHA256, "source inventory mismatch"),
        (INVENTORY_SHA256, "c" * 64, "baseline artifact mismatch"),
    ],
)
def test_review_rejects_artifact_substitution(
    inventory: str,
    baseline: str,
    message: str,
) -> None:
    attestation, _, public_keys = _attestation()

    with pytest.raises(InventoryReviewError, match=message):
        verify_inventory_review_attestation(
            attestation,
            expected_source_inventory_sha256=inventory,
            expected_baseline_artifact_sha256=baseline,
            public_keys=public_keys,
            required_signatures=2,
        )


def test_review_rejects_signature_keyring_and_threshold_substitution() -> None:
    attestation, _, public_keys = _attestation()
    changed = ("A" if attestation.signatures[0].signature[0] != "A" else "B") + (
        attestation.signatures[0].signature[1:]
    )
    tampered = replace(
        attestation,
        signatures=(
            replace(attestation.signatures[0], signature=changed),
            attestation.signatures[1],
        ),
    )
    _, other_public_keys = _reviewers()

    with pytest.raises(InventoryReviewError, match="signature mismatch"):
        verify_inventory_review_attestation(
            tampered,
            expected_source_inventory_sha256=INVENTORY_SHA256,
            expected_baseline_artifact_sha256=BASELINE_SHA256,
            public_keys=public_keys,
            required_signatures=2,
        )
    with pytest.raises(InventoryReviewError, match="policy mismatch"):
        verify_inventory_review_attestation(
            attestation,
            expected_source_inventory_sha256=INVENTORY_SHA256,
            expected_baseline_artifact_sha256=BASELINE_SHA256,
            public_keys=other_public_keys,
            required_signatures=2,
        )
    with pytest.raises(InventoryReviewError, match="policy mismatch"):
        verify_inventory_review_attestation(
            attestation,
            expected_source_inventory_sha256=INVENTORY_SHA256,
            expected_baseline_artifact_sha256=BASELINE_SHA256,
            public_keys=public_keys,
            required_signatures=3,
        )


def test_review_policy_rejects_aliases_for_the_same_public_key() -> None:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )

    with pytest.raises(InventoryReviewError, match="public keys must be distinct"):
        inventory_review_policy_sha256(
            {"reviewer-1": public_key, "reviewer-2": public_key},
            required_signatures=2,
        )


def test_creator_requires_matching_trusted_signers_and_quorum() -> None:
    private_keys, public_keys = _reviewers()
    unrelated = Ed25519PrivateKey.generate()

    with pytest.raises(InventoryReviewError, match="do not satisfy"):
        create_inventory_review_attestation(
            source_inventory_sha256=INVENTORY_SHA256,
            baseline_artifact_sha256=BASELINE_SHA256,
            public_keys=public_keys,
            required_signatures=2,
            signers={"reviewer-1": private_keys["reviewer-1"]},
        )
    with pytest.raises(InventoryReviewError, match="does not match"):
        create_inventory_review_attestation(
            source_inventory_sha256=INVENTORY_SHA256,
            baseline_artifact_sha256=BASELINE_SHA256,
            public_keys=public_keys,
            required_signatures=2,
            signers={"reviewer-1": unrelated, "reviewer-2": private_keys["reviewer-2"]},
        )


def test_loader_rejects_duplicate_unknown_and_noncanonical_fields() -> None:
    attestation, _, _ = _attestation()
    payload = dump_inventory_review_attestation(attestation)
    duplicate = payload.replace(
        b'"purpose":',
        b'"purpose":"duplicate","purpose":',
        1,
    )
    unknown = json.loads(payload)
    unknown["unexpected"] = True

    with pytest.raises(InventoryReviewError, match="duplicate field"):
        load_inventory_review_attestation(duplicate)
    with pytest.raises(InventoryReviewError, match="fields do not match"):
        load_inventory_review_attestation(
            json.dumps(unknown, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
    with pytest.raises(InventoryReviewError, match="not canonical"):
        load_inventory_review_attestation(
            json.dumps(json.loads(payload)).encode("ascii")
        )


def test_attestation_is_privacy_reduced() -> None:
    attestation, _, _ = _attestation()
    payload = dump_inventory_review_attestation(attestation)

    assert b"source text" not in payload
    assert b"document-id" not in payload
    assert set(json.loads(payload)) == {
        "algorithm",
        "baseline_artifact_sha256",
        "purpose",
        "review_policy_sha256",
        "schema_version",
        "signatures",
        "source_inventory_sha256",
    }
