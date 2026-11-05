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

STATUS_SCHEMA_VERSION = "retrieval-reviewer-key-status/v1"
STATUS_RECEIPT_SCHEMA_VERSION = "retrieval-reviewer-key-status-verification/v1"
STATUS_PURPOSE = "rag-retrieval-reviewer-key-status"
STATUS_ALGORITHM = "ed25519"
MAX_STATUS_BYTES = 32_768
MAX_STATUS_LIFETIME_SECONDS = 86_400
MAX_AUTHORITIES = 8

_DOMAIN = b"rag-document-intelligence/reviewer-key-status/v1\0"
_POLICY_DOMAIN = b"rag-document-intelligence/reviewer-key-status-policy/v1\0"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}")
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{86}")
_FIELDS = {
    "algorithm",
    "authority_key_id",
    "expires_at_epoch",
    "issued_at_epoch",
    "purpose",
    "review_policy_sha256",
    "schema_version",
    "signature",
    "statuses",
}
_STATUS_FIELDS = {"key_id", "status"}


class ReviewerKeyStatusError(ValueError):
    """A reviewer key-status snapshot is malformed, stale, or unauthenticated."""

    code = "reviewer_key_status_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


class MessageSigner(Protocol):
    def sign(self, message: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ReviewerKeyStatus:
    key_id: str
    status: str


@dataclass(frozen=True, slots=True)
class ReviewerKeyStatusSnapshot:
    schema_version: str
    purpose: str
    algorithm: str
    review_policy_sha256: str
    issued_at_epoch: int
    expires_at_epoch: int
    authority_key_id: str
    statuses: tuple[ReviewerKeyStatus, ...]
    signature: str

    @property
    def snapshot_sha256(self) -> str:
        return hashlib.sha256(dump_reviewer_key_status_snapshot(self)).hexdigest()

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReviewerKeyStatusReceipt:
    schema_version: str
    snapshot_sha256: str
    status_policy_sha256: str
    active_key_count: int
    revoked_key_count: int
    verification_epoch: int

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
        raise ReviewerKeyStatusError("status snapshot is not canonical JSON") from exc


def _validate_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ReviewerKeyStatusError(f"{name} is malformed")
    return value


def _load_public_keys(
    public_keys: Mapping[str, bytes], *, minimum: int, maximum: int, role: str
) -> tuple[dict[str, Ed25519PublicKey], dict[str, str]]:
    if (
        not isinstance(public_keys, Mapping)
        or not minimum <= len(public_keys) <= maximum
    ):
        raise ReviewerKeyStatusError(f"{role} keyring size is invalid")
    loaded: dict[str, Ed25519PublicKey] = {}
    fingerprints: dict[str, str] = {}
    for key_id in sorted(public_keys):
        encoded = public_keys[key_id]
        if (
            not isinstance(key_id, str)
            or _IDENTIFIER_RE.fullmatch(key_id) is None
            or not isinstance(encoded, bytes)
            or not 1 <= len(encoded) <= 8_192
        ):
            raise ReviewerKeyStatusError(f"{role} keyring is malformed")
        try:
            public_key = load_der_public_key(encoded)
        except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
            raise ReviewerKeyStatusError(f"{role} public key is malformed") from exc
        if not isinstance(public_key, Ed25519PublicKey):
            raise ReviewerKeyStatusError(f"{role} public key algorithm is unsupported")
        fingerprint = hashlib.sha256(
            public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
        ).hexdigest()
        if fingerprint in fingerprints.values():
            raise ReviewerKeyStatusError(f"{role} public keys must be distinct")
        loaded[key_id] = public_key
        fingerprints[key_id] = fingerprint
    return loaded, fingerprints


def _unsigned_document(snapshot: ReviewerKeyStatusSnapshot) -> dict[str, object]:
    return {
        key: value
        for key, value in snapshot.to_audit_dict().items()
        if key != "signature"
    }


def _message(unsigned: Mapping[str, object]) -> bytes:
    return _DOMAIN + _canonical_json(dict(unsigned))


def _encode_signature(signature: bytes) -> str:
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ReviewerKeyStatusError("status authority returned a malformed signature")
    return base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def _decode_signature(signature: str) -> bytes:
    try:
        decoded = base64.b64decode(signature + "==", altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ReviewerKeyStatusError("status signature encoding is malformed") from exc
    if len(decoded) != 64:
        raise ReviewerKeyStatusError("status signature encoding is malformed")
    return decoded


def _validate_snapshot(snapshot: ReviewerKeyStatusSnapshot) -> None:
    if not isinstance(snapshot, ReviewerKeyStatusSnapshot):
        raise TypeError("snapshot must be a ReviewerKeyStatusSnapshot")
    if (
        snapshot.schema_version != STATUS_SCHEMA_VERSION
        or snapshot.purpose != STATUS_PURPOSE
        or snapshot.algorithm != STATUS_ALGORITHM
    ):
        raise ReviewerKeyStatusError("status snapshot policy is unsupported")
    _validate_digest(snapshot.review_policy_sha256, "review policy digest")
    if (
        type(snapshot.issued_at_epoch) is not int
        or type(snapshot.expires_at_epoch) is not int
        or snapshot.issued_at_epoch < 1
        or snapshot.expires_at_epoch <= snapshot.issued_at_epoch
        or snapshot.expires_at_epoch - snapshot.issued_at_epoch
        > MAX_STATUS_LIFETIME_SECONDS
    ):
        raise ReviewerKeyStatusError("status snapshot validity window is invalid")
    if (
        not isinstance(snapshot.authority_key_id, str)
        or _IDENTIFIER_RE.fullmatch(snapshot.authority_key_id) is None
        or not isinstance(snapshot.signature, str)
        or _SIGNATURE_RE.fullmatch(snapshot.signature) is None
        or not isinstance(snapshot.statuses, tuple)
        or not 2 <= len(snapshot.statuses) <= 32
    ):
        raise ReviewerKeyStatusError("status snapshot is malformed")
    key_ids: list[str] = []
    for item in snapshot.statuses:
        if (
            not isinstance(item, ReviewerKeyStatus)
            or not isinstance(item.key_id, str)
            or _IDENTIFIER_RE.fullmatch(item.key_id) is None
            or item.status not in {"active", "revoked"}
        ):
            raise ReviewerKeyStatusError("reviewer key status is malformed")
        key_ids.append(item.key_id)
    if key_ids != sorted(key_ids) or len(set(key_ids)) != len(key_ids):
        raise ReviewerKeyStatusError(
            "reviewer key statuses must have unique sorted key ids"
        )


def create_reviewer_key_status_snapshot(
    *,
    review_policy_sha256: str,
    statuses: Mapping[str, str],
    issued_at_epoch: int,
    expires_at_epoch: int,
    authority_key_id: str,
    authority_public_key: bytes,
    signer: MessageSigner,
) -> ReviewerKeyStatusSnapshot:
    """Create one signed short-lived status view for an exact review policy."""
    rows = (
        tuple(
            ReviewerKeyStatus(key_id, status)
            for key_id, status in sorted(statuses.items())
        )
        if isinstance(statuses, Mapping)
        else ()
    )
    unsigned: dict[str, object] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "purpose": STATUS_PURPOSE,
        "algorithm": STATUS_ALGORITHM,
        "review_policy_sha256": review_policy_sha256,
        "issued_at_epoch": issued_at_epoch,
        "expires_at_epoch": expires_at_epoch,
        "authority_key_id": authority_key_id,
        "statuses": rows,
    }
    placeholder = ReviewerKeyStatusSnapshot(**unsigned, signature="A" * 86)
    _validate_snapshot(placeholder)
    authorities, _ = _load_public_keys(
        {authority_key_id: authority_public_key},
        minimum=1,
        maximum=1,
        role="status authority",
    )
    message = _message(_unsigned_document(placeholder))
    try:
        signature = signer.sign(message)
        authorities[authority_key_id].verify(signature, message)
    except InvalidSignature as exc:
        raise ReviewerKeyStatusError(
            "status signer does not match its trusted key"
        ) from exc
    except ReviewerKeyStatusError:
        raise
    except Exception as exc:
        raise ReviewerKeyStatusError("status signing failed") from exc
    snapshot = ReviewerKeyStatusSnapshot(
        **unsigned, signature=_encode_signature(signature)
    )
    _validate_snapshot(snapshot)
    return snapshot


def verify_reviewer_key_status_snapshot(
    snapshot: ReviewerKeyStatusSnapshot,
    *,
    expected_review_policy_sha256: str,
    reviewer_public_keys: Mapping[str, bytes],
    required_active_key_ids: tuple[str, ...],
    authority_public_keys: Mapping[str, bytes],
    verification_epoch: int,
) -> ReviewerKeyStatusReceipt:
    """Verify status authority, coverage, freshness, and active review signers."""
    _validate_snapshot(snapshot)
    expected_policy = _validate_digest(
        expected_review_policy_sha256, "expected review policy digest"
    )
    if not hmac.compare_digest(snapshot.review_policy_sha256, expected_policy):
        raise ReviewerKeyStatusError("status snapshot review policy mismatch")
    if type(verification_epoch) is not int or verification_epoch < 1:
        raise ReviewerKeyStatusError("verification epoch is invalid")
    if verification_epoch < snapshot.issued_at_epoch:
        raise ReviewerKeyStatusError("status snapshot is from the future")
    if verification_epoch > snapshot.expires_at_epoch:
        raise ReviewerKeyStatusError("status snapshot has expired")

    reviewers, reviewer_fingerprints = _load_public_keys(
        reviewer_public_keys, minimum=2, maximum=32, role="reviewer"
    )
    authorities, authority_fingerprints = _load_public_keys(
        authority_public_keys,
        minimum=1,
        maximum=MAX_AUTHORITIES,
        role="status authority",
    )
    authority = authorities.get(snapshot.authority_key_id)
    if authority is None:
        raise ReviewerKeyStatusError("status authority is not trusted")
    if (
        authority_fingerprints[snapshot.authority_key_id]
        in reviewer_fingerprints.values()
    ):
        raise ReviewerKeyStatusError(
            "status authority must be independent from reviewers"
        )
    try:
        authority.verify(
            _decode_signature(snapshot.signature),
            _message(_unsigned_document(snapshot)),
        )
    except InvalidSignature as exc:
        raise ReviewerKeyStatusError("status signature mismatch") from exc

    observed = {item.key_id: item.status for item in snapshot.statuses}
    if set(observed) != set(reviewers):
        raise ReviewerKeyStatusError(
            "status snapshot does not cover the reviewer keyring"
        )
    if (
        not isinstance(required_active_key_ids, tuple)
        or not required_active_key_ids
        or tuple(sorted(set(required_active_key_ids))) != required_active_key_ids
        or any(key_id not in reviewers for key_id in required_active_key_ids)
    ):
        raise ReviewerKeyStatusError("required active reviewer set is invalid")
    if any(observed[key_id] != "active" for key_id in required_active_key_ids):
        raise ReviewerKeyStatusError("review includes a revoked signer")

    policy_document = {
        "schema_version": STATUS_RECEIPT_SCHEMA_VERSION,
        "review_policy_sha256": expected_policy,
        "authorities": tuple(
            {
                "key_id": key_id,
                "public_key_sha256": authority_fingerprints[key_id],
            }
            for key_id in sorted(authority_fingerprints)
        ),
        "signing_authority_key_id": snapshot.authority_key_id,
        "maximum_lifetime_seconds": MAX_STATUS_LIFETIME_SECONDS,
    }
    active_count = sum(status == "active" for status in observed.values())
    return ReviewerKeyStatusReceipt(
        schema_version=STATUS_RECEIPT_SCHEMA_VERSION,
        snapshot_sha256=snapshot.snapshot_sha256,
        status_policy_sha256=hashlib.sha256(
            _POLICY_DOMAIN + _canonical_json(policy_document)
        ).hexdigest(),
        active_key_count=active_count,
        revoked_key_count=len(observed) - active_count,
        verification_epoch=verification_epoch,
    )


def dump_reviewer_key_status_snapshot(snapshot: ReviewerKeyStatusSnapshot) -> bytes:
    _validate_snapshot(snapshot)
    payload = _canonical_json(snapshot.to_audit_dict())
    if len(payload) > MAX_STATUS_BYTES:
        raise ReviewerKeyStatusError("status snapshot exceeds the byte limit")
    return payload


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewerKeyStatusError("status snapshot contains a duplicate field")
        result[key] = value
    return result


def load_reviewer_key_status_snapshot(payload: bytes) -> ReviewerKeyStatusSnapshot:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload or len(payload) > MAX_STATUS_BYTES:
        raise ReviewerKeyStatusError("status snapshot size is invalid")
    try:
        document = json.loads(
            payload,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReviewerKeyStatusError(f"invalid JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReviewerKeyStatusError("status snapshot is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise ReviewerKeyStatusError("status snapshot fields do not match the schema")
    rows = document.get("statuses")
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or set(row) != _STATUS_FIELDS for row in rows
    ):
        raise ReviewerKeyStatusError("reviewer key status is malformed")
    try:
        snapshot = ReviewerKeyStatusSnapshot(
            **{key: value for key, value in document.items() if key != "statuses"},
            statuses=tuple(ReviewerKeyStatus(**row) for row in rows),
        )
    except TypeError as exc:
        raise ReviewerKeyStatusError(
            "status snapshot fields do not match the schema"
        ) from exc
    _validate_snapshot(snapshot)
    if _canonical_json(snapshot.to_audit_dict()) != payload:
        raise ReviewerKeyStatusError("status snapshot JSON is not canonical")
    return snapshot
