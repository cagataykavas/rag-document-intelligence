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
from cryptography.hazmat.primitives.serialization import load_der_public_key

from src.entailment_release_log import (
    ReleaseLogError,
    ReleaseLogSnapshot,
    verify_release_log_payload,
)

RELEASE_HEAD_ANCHOR_SCHEMA_VERSION = "entailment-release-head-anchor/v1"
RELEASE_HEAD_ANCHOR_ALGORITHM = "ed25519-sha512"
RELEASE_HEAD_ANCHOR_PURPOSE = "rag-release-log-head-anchor"
_DOMAIN = b"rag-document-intelligence/release-head-anchor/v1\x00"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{86}")
_MAX_ANCHOR_BYTES = 4_096
_MAX_KMS_MESSAGE_BYTES = 4_096
_FIELDS = {
    "algorithm",
    "entry_count",
    "head_sha256",
    "issued_at_epoch",
    "key_id",
    "log_sha256",
    "purpose",
    "schema_version",
    "signature",
    "source_commit_sha",
}


class ReleaseHeadAnchorError(ValueError):
    """A release-log anchor is malformed, mismatched, or unauthenticated."""

    code = "release_head_anchor_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


class MessageSigner(Protocol):
    def sign(self, message: bytes) -> bytes: ...


class KmsEd25519Signer:
    """Minimal AWS KMS signer adapter; the private key never enters this process."""

    def __init__(self, client: Any, *, kms_key_id: str) -> None:
        if not isinstance(kms_key_id, str) or not 1 <= len(kms_key_id) <= 2_048:
            raise ReleaseHeadAnchorError("KMS key id is invalid")
        self._client = client
        self._kms_key_id = kms_key_id

    def sign(self, message: bytes) -> bytes:
        if (
            not isinstance(message, bytes)
            or not 1 <= len(message) <= _MAX_KMS_MESSAGE_BYTES
        ):
            raise ReleaseHeadAnchorError("KMS signing message size is invalid")
        try:
            response = self._client.sign(
                KeyId=self._kms_key_id,
                Message=message,
                MessageType="RAW",
                SigningAlgorithm="ED25519_SHA_512",
            )
        except Exception as exc:
            raise ReleaseHeadAnchorError("KMS signing failed") from exc
        if not isinstance(response, Mapping):
            raise ReleaseHeadAnchorError("KMS signing response is malformed")
        signature = response.get("Signature")
        if (
            response.get("SigningAlgorithm") != "ED25519_SHA_512"
            or not isinstance(response.get("KeyId"), str)
            or not isinstance(signature, bytes)
            or len(signature) != 64
        ):
            raise ReleaseHeadAnchorError("KMS signing response is malformed")
        return signature


@dataclass(frozen=True, slots=True)
class ReleaseHeadAnchor:
    schema_version: str
    purpose: str
    algorithm: str
    key_id: str
    source_commit_sha: str
    entry_count: int
    head_sha256: str
    log_sha256: str
    issued_at_epoch: int
    signature: str

    @property
    def anchor_sha256(self) -> str:
        return hashlib.sha256(dump_release_head_anchor(self)).hexdigest()

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
        raise ReleaseHeadAnchorError("anchor is not canonical JSON") from exc


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseHeadAnchorError("anchor contains a duplicate field")
        result[key] = value
    return result


def _validate_document(document: Mapping[str, object]) -> None:
    if set(document) != _FIELDS:
        raise ReleaseHeadAnchorError("anchor fields do not match the schema")
    if (
        document["schema_version"] != RELEASE_HEAD_ANCHOR_SCHEMA_VERSION
        or document["purpose"] != RELEASE_HEAD_ANCHOR_PURPOSE
        or document["algorithm"] != RELEASE_HEAD_ANCHOR_ALGORITHM
    ):
        raise ReleaseHeadAnchorError("anchor policy is unsupported")
    if (
        not isinstance(document["key_id"], str)
        or _IDENTIFIER_RE.fullmatch(document["key_id"]) is None
        or not isinstance(document["source_commit_sha"], str)
        or _GIT_SHA_RE.fullmatch(document["source_commit_sha"]) is None
        or type(document["entry_count"]) is not int
        or not 1 <= document["entry_count"] <= 10_000
        or type(document["issued_at_epoch"]) is not int
        or document["issued_at_epoch"] < 1
        or any(
            not isinstance(document[field], str)
            or _DIGEST_RE.fullmatch(document[field]) is None
            for field in ("head_sha256", "log_sha256")
        )
        or not isinstance(document["signature"], str)
        or _SIGNATURE_RE.fullmatch(document["signature"]) is None
    ):
        raise ReleaseHeadAnchorError("anchor fields are malformed")


def _unsigned_document(anchor: ReleaseHeadAnchor) -> dict[str, object]:
    return {
        key: value
        for key, value in anchor.to_audit_dict().items()
        if key != "signature"
    }


def _message(document: Mapping[str, object]) -> bytes:
    return _DOMAIN + _canonical_json(dict(document))


def _encode_signature(signature: bytes) -> str:
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ReleaseHeadAnchorError("anchor signer returned a malformed signature")
    return base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def _decode_signature(signature: str) -> bytes:
    try:
        decoded = base64.b64decode(signature + "==", altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ReleaseHeadAnchorError("anchor signature encoding is malformed") from exc
    if len(decoded) != 64:
        raise ReleaseHeadAnchorError("anchor signature encoding is malformed")
    return decoded


def create_release_head_anchor(
    log_payload: bytes,
    *,
    key_id: str,
    source_commit_sha: str,
    issued_at_epoch: int,
    signer: MessageSigner,
) -> ReleaseHeadAnchor:
    """Sign one captured, non-empty release log with an asymmetric signer."""

    try:
        snapshot = verify_release_log_payload(log_payload)
    except ReleaseLogError as exc:
        raise ReleaseHeadAnchorError("release log is invalid") from exc
    if not snapshot.entries:
        raise ReleaseHeadAnchorError("release log is empty")
    unsigned: dict[str, object] = {
        "schema_version": RELEASE_HEAD_ANCHOR_SCHEMA_VERSION,
        "purpose": RELEASE_HEAD_ANCHOR_PURPOSE,
        "algorithm": RELEASE_HEAD_ANCHOR_ALGORITHM,
        "key_id": key_id,
        "source_commit_sha": source_commit_sha,
        "entry_count": len(snapshot.entries),
        "head_sha256": snapshot.head_sha256,
        "log_sha256": hashlib.sha256(log_payload).hexdigest(),
        "issued_at_epoch": issued_at_epoch,
    }
    candidate = {**unsigned, "signature": "A" * 86}
    _validate_document(candidate)
    try:
        signature = signer.sign(_message(unsigned))
    except ReleaseHeadAnchorError:
        raise
    except Exception as exc:
        raise ReleaseHeadAnchorError("anchor signing failed") from exc
    return ReleaseHeadAnchor(**unsigned, signature=_encode_signature(signature))


def verify_release_head_anchor(
    anchor: ReleaseHeadAnchor,
    log_payload: bytes,
    *,
    public_keys: Mapping[str, bytes],
    expected_source_commit_sha: str,
) -> ReleaseLogSnapshot:
    """Verify the public signature and exact anchored log payload."""

    if not isinstance(anchor, ReleaseHeadAnchor):
        raise TypeError("anchor must be a ReleaseHeadAnchor")
    document = anchor.to_audit_dict()
    _validate_document(document)
    if not isinstance(expected_source_commit_sha, str) or (
        _GIT_SHA_RE.fullmatch(expected_source_commit_sha) is None
    ):
        raise ReleaseHeadAnchorError("expected source commit is malformed")
    if not hmac.compare_digest(anchor.source_commit_sha, expected_source_commit_sha):
        raise ReleaseHeadAnchorError("anchor source commit mismatch")
    if not isinstance(public_keys, Mapping) or anchor.key_id not in public_keys:
        raise ReleaseHeadAnchorError("anchor public key is unavailable")
    encoded_key = public_keys[anchor.key_id]
    if not isinstance(encoded_key, bytes) or not 1 <= len(encoded_key) <= 8_192:
        raise ReleaseHeadAnchorError("anchor public key is malformed")
    try:
        public_key = load_der_public_key(encoded_key)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise ReleaseHeadAnchorError("anchor public key is malformed") from exc
    if not isinstance(public_key, Ed25519PublicKey):
        raise ReleaseHeadAnchorError("anchor public key algorithm mismatch")
    try:
        public_key.verify(
            _decode_signature(anchor.signature),
            _message(_unsigned_document(anchor)),
        )
    except InvalidSignature as exc:
        raise ReleaseHeadAnchorError("anchor signature mismatch") from exc

    try:
        snapshot = verify_release_log_payload(log_payload)
    except ReleaseLogError as exc:
        raise ReleaseHeadAnchorError("supplied release log is invalid") from exc
    if (
        len(snapshot.entries) != anchor.entry_count
        or not hmac.compare_digest(snapshot.head_sha256, anchor.head_sha256)
        or not hmac.compare_digest(
            hashlib.sha256(log_payload).hexdigest(), anchor.log_sha256
        )
    ):
        raise ReleaseHeadAnchorError("anchor does not bind the supplied release log")
    return snapshot


def dump_release_head_anchor(anchor: ReleaseHeadAnchor) -> bytes:
    if not isinstance(anchor, ReleaseHeadAnchor):
        raise TypeError("anchor must be a ReleaseHeadAnchor")
    document = anchor.to_audit_dict()
    _validate_document(document)
    payload = _canonical_json(document)
    if len(payload) > _MAX_ANCHOR_BYTES:
        raise ReleaseHeadAnchorError("anchor exceeds the byte limit")
    return payload


def load_release_head_anchor(payload: bytes) -> ReleaseHeadAnchor:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if not payload or len(payload) > _MAX_ANCHOR_BYTES:
        raise ReleaseHeadAnchorError("anchor size is invalid")
    try:
        document = json.loads(payload, object_pairs_hook=_reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReleaseHeadAnchorError("anchor is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ReleaseHeadAnchorError("anchor fields do not match the schema")
    _validate_document(document)
    if _canonical_json(document) != payload:
        raise ReleaseHeadAnchorError("anchor JSON is not canonical")
    try:
        return ReleaseHeadAnchor(**document)
    except TypeError as exc:  # pragma: no cover - exact fields checked above
        raise ReleaseHeadAnchorError("anchor fields do not match the schema") from exc
