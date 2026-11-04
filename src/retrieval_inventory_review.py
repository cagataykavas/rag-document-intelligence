from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_der_public_key,
)

REVIEW_SCHEMA_VERSION = "retrieval-source-inventory-review/v1"
REVIEW_POLICY_SCHEMA_VERSION = "retrieval-source-inventory-review-policy/v1"
REVIEW_PURPOSE = "rag-retrieval-source-inventory-review"
REVIEW_ALGORITHM = "ed25519"
MAX_REVIEW_ATTESTATION_BYTES = 32_768
MAX_REVIEWERS = 32
_DOMAIN = b"rag-document-intelligence/source-inventory-review/v1\0"
_POLICY_DOMAIN = b"rag-document-intelligence/source-inventory-review-policy/v1\0"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}")
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{86}")
_FIELDS = {
    "algorithm",
    "baseline_artifact_sha256",
    "purpose",
    "review_policy_sha256",
    "schema_version",
    "signatures",
    "source_inventory_sha256",
}
_SIGNATURE_FIELDS = {"key_id", "signature"}


class InventoryReviewError(ValueError):
    """The source-inventory review is malformed, mismatched, or unauthenticated."""

    code = "inventory_review_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


class MessageSigner(Protocol):
    def sign(self, message: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class InventoryReviewSignature:
    key_id: str
    signature: str


@dataclass(frozen=True, slots=True)
class InventoryReviewAttestation:
    schema_version: str
    purpose: str
    algorithm: str
    source_inventory_sha256: str
    baseline_artifact_sha256: str
    review_policy_sha256: str
    signatures: tuple[InventoryReviewSignature, ...]

    @property
    def attestation_sha256(self) -> str:
        return hashlib.sha256(dump_inventory_review_attestation(self)).hexdigest()

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise InventoryReviewError("review is not canonical JSON") from exc


def _validate_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise InventoryReviewError(f"{name} is malformed")
    return value


def _load_public_keys(
    public_keys: Mapping[str, bytes],
) -> tuple[dict[str, Ed25519PublicKey], tuple[dict[str, str], ...]]:
    if (
        not isinstance(public_keys, Mapping)
        or not 2 <= len(public_keys) <= MAX_REVIEWERS
    ):
        raise InventoryReviewError(
            "reviewer keyring must contain between 2 and 32 keys"
        )
    if any(
        not isinstance(key_id, str) or _IDENTIFIER_RE.fullmatch(key_id) is None
        for key_id in public_keys
    ):
        raise InventoryReviewError("reviewer keyring is malformed")
    loaded: dict[str, Ed25519PublicKey] = {}
    policy_rows: list[dict[str, str]] = []
    fingerprints: set[str] = set()
    for key_id in sorted(public_keys):
        encoded_key = public_keys[key_id]
        if not isinstance(encoded_key, bytes) or not 1 <= len(encoded_key) <= 8_192:
            raise InventoryReviewError("reviewer keyring is malformed")
        try:
            public_key = load_der_public_key(encoded_key)
        except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
            raise InventoryReviewError("reviewer public key is malformed") from exc
        if not isinstance(public_key, Ed25519PublicKey):
            raise InventoryReviewError("reviewer public key algorithm is unsupported")
        fingerprint = hashlib.sha256(
            public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        ).hexdigest()
        if fingerprint in fingerprints:
            raise InventoryReviewError("reviewer public keys must be distinct")
        fingerprints.add(fingerprint)
        loaded[key_id] = public_key
        policy_rows.append({"key_id": key_id, "public_key_sha256": fingerprint})
    return loaded, tuple(policy_rows)


def inventory_review_policy_sha256(
    public_keys: Mapping[str, bytes],
    *,
    required_signatures: int,
) -> str:
    """Commit to a threshold and exact set of distinct reviewer public keys."""
    _, reviewers = _load_public_keys(public_keys)
    if type(required_signatures) is not int or not 2 <= required_signatures <= len(
        reviewers
    ):
        raise InventoryReviewError("required signatures must be in [2, reviewer count]")
    document = {
        "schema_version": REVIEW_POLICY_SCHEMA_VERSION,
        "required_signatures": required_signatures,
        "reviewers": reviewers,
    }
    return hashlib.sha256(_POLICY_DOMAIN + _canonical_json(document)).hexdigest()


def _unsigned_document(attestation: InventoryReviewAttestation) -> dict[str, object]:
    return {
        key: value
        for key, value in attestation.to_audit_dict().items()
        if key != "signatures"
    }


def _message(unsigned: Mapping[str, object]) -> bytes:
    return _DOMAIN + _canonical_json(dict(unsigned))


def _encode_signature(signature: bytes) -> str:
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise InventoryReviewError("reviewer returned a malformed signature")
    return base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def _decode_signature(signature: str) -> bytes:
    try:
        decoded = base64.b64decode(signature + "==", altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise InventoryReviewError("review signature encoding is malformed") from exc
    if len(decoded) != 64:
        raise InventoryReviewError("review signature encoding is malformed")
    return decoded


def _validate_attestation(attestation: InventoryReviewAttestation) -> None:
    if not isinstance(attestation, InventoryReviewAttestation):
        raise TypeError("attestation must be an InventoryReviewAttestation")
    if (
        attestation.schema_version != REVIEW_SCHEMA_VERSION
        or attestation.purpose != REVIEW_PURPOSE
        or attestation.algorithm != REVIEW_ALGORITHM
    ):
        raise InventoryReviewError("review policy is unsupported")
    _validate_digest(attestation.source_inventory_sha256, "source inventory digest")
    _validate_digest(attestation.baseline_artifact_sha256, "baseline artifact digest")
    _validate_digest(attestation.review_policy_sha256, "review policy digest")
    if (
        not isinstance(attestation.signatures, tuple)
        or not 2 <= len(attestation.signatures) <= MAX_REVIEWERS
    ):
        raise InventoryReviewError("review signature count is invalid")
    key_ids: list[str] = []
    for item in attestation.signatures:
        if (
            not isinstance(item, InventoryReviewSignature)
            or not isinstance(item.key_id, str)
            or _IDENTIFIER_RE.fullmatch(item.key_id) is None
            or not isinstance(item.signature, str)
            or _SIGNATURE_RE.fullmatch(item.signature) is None
        ):
            raise InventoryReviewError("review signature is malformed")
        key_ids.append(item.key_id)
    if key_ids != sorted(key_ids) or len(set(key_ids)) != len(key_ids):
        raise InventoryReviewError("review signatures must have unique sorted key ids")


def create_inventory_review_attestation(
    *,
    source_inventory_sha256: str,
    baseline_artifact_sha256: str,
    public_keys: Mapping[str, bytes],
    required_signatures: int,
    signers: Mapping[str, MessageSigner],
) -> InventoryReviewAttestation:
    """Create a quorum review over one exact inventory and retrieval baseline."""
    _validate_digest(source_inventory_sha256, "source inventory digest")
    _validate_digest(baseline_artifact_sha256, "baseline artifact digest")
    loaded_keys, _ = _load_public_keys(public_keys)
    policy_sha256 = inventory_review_policy_sha256(
        public_keys, required_signatures=required_signatures
    )
    if (
        not isinstance(signers, Mapping)
        or not required_signatures <= len(signers) <= len(loaded_keys)
        or any(key_id not in loaded_keys for key_id in signers)
    ):
        raise InventoryReviewError("signers do not satisfy the review policy")
    unsigned: dict[str, object] = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "purpose": REVIEW_PURPOSE,
        "algorithm": REVIEW_ALGORITHM,
        "source_inventory_sha256": source_inventory_sha256,
        "baseline_artifact_sha256": baseline_artifact_sha256,
        "review_policy_sha256": policy_sha256,
    }
    message = _message(unsigned)
    signatures: list[InventoryReviewSignature] = []
    for key_id in sorted(signers):
        try:
            raw_signature = signers[key_id].sign(message)
            loaded_keys[key_id].verify(raw_signature, message)
        except InvalidSignature as exc:
            raise InventoryReviewError(
                "review signer does not match its trusted key"
            ) from exc
        except InventoryReviewError:
            raise
        except Exception as exc:
            raise InventoryReviewError("review signing failed") from exc
        signatures.append(
            InventoryReviewSignature(key_id, _encode_signature(raw_signature))
        )
    attestation = InventoryReviewAttestation(
        **unsigned,
        signatures=tuple(signatures),
    )
    _validate_attestation(attestation)
    return attestation


def verify_inventory_review_attestation(
    attestation: InventoryReviewAttestation,
    *,
    expected_source_inventory_sha256: str,
    expected_baseline_artifact_sha256: str,
    public_keys: Mapping[str, bytes],
    required_signatures: int,
) -> str:
    """Verify a distinct-key quorum and return the canonical review commitment."""
    _validate_attestation(attestation)
    expected_inventory = _validate_digest(
        expected_source_inventory_sha256, "expected source inventory digest"
    )
    expected_baseline = _validate_digest(
        expected_baseline_artifact_sha256, "expected baseline artifact digest"
    )
    if not hmac.compare_digest(attestation.source_inventory_sha256, expected_inventory):
        raise InventoryReviewError("review source inventory mismatch")
    if not hmac.compare_digest(attestation.baseline_artifact_sha256, expected_baseline):
        raise InventoryReviewError("review baseline artifact mismatch")
    loaded_keys, _ = _load_public_keys(public_keys)
    expected_policy = inventory_review_policy_sha256(
        public_keys, required_signatures=required_signatures
    )
    if not hmac.compare_digest(attestation.review_policy_sha256, expected_policy):
        raise InventoryReviewError("review policy mismatch")
    if len(attestation.signatures) < required_signatures:
        raise InventoryReviewError("review signature quorum is not met")
    message = _message(_unsigned_document(attestation))
    for item in attestation.signatures:
        public_key = loaded_keys.get(item.key_id)
        if public_key is None:
            raise InventoryReviewError("review signer is not trusted")
        try:
            public_key.verify(_decode_signature(item.signature), message)
        except InvalidSignature as exc:
            raise InventoryReviewError("review signature mismatch") from exc
    return attestation.attestation_sha256


def dump_inventory_review_attestation(
    attestation: InventoryReviewAttestation,
) -> bytes:
    _validate_attestation(attestation)
    payload = _canonical_json(attestation.to_audit_dict())
    if len(payload) > MAX_REVIEW_ATTESTATION_BYTES:
        raise InventoryReviewError("review attestation exceeds the byte limit")
    return payload


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InventoryReviewError("review attestation contains a duplicate field")
        result[key] = value
    return result


def load_inventory_review_attestation(payload: bytes) -> InventoryReviewAttestation:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload or len(payload) > MAX_REVIEW_ATTESTATION_BYTES:
        raise InventoryReviewError("review attestation size is invalid")
    try:
        document = json.loads(
            payload,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                InventoryReviewError(f"invalid JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InventoryReviewError("review attestation is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise InventoryReviewError("review attestation fields do not match the schema")
    rows = document.get("signatures")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or set(row) != _SIGNATURE_FIELDS for row in rows
    ):
        raise InventoryReviewError("review signature is malformed")
    try:
        signatures = tuple(InventoryReviewSignature(**row) for row in rows)
        attestation = InventoryReviewAttestation(
            **{key: value for key, value in document.items() if key != "signatures"},
            signatures=signatures,
        )
    except TypeError as exc:
        raise InventoryReviewError(
            "review attestation fields do not match the schema"
        ) from exc
    _validate_attestation(attestation)
    if _canonical_json(attestation.to_audit_dict()) != payload:
        raise InventoryReviewError("review attestation JSON is not canonical")
    return attestation
