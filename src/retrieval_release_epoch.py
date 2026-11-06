from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_der_public_key,
)

RELEASE_EPOCH_SCHEMA_VERSION = "retrieval-release-epoch-attestation/v1"
RELEASE_EPOCH_POLICY_SCHEMA_VERSION = "retrieval-release-epoch-policy/v1"
RELEASE_EPOCH_RECEIPT_SCHEMA_VERSION = "retrieval-release-epoch-verification/v1"
RELEASE_EPOCH_PURPOSE = "rag-retrieval-release-epoch"
RELEASE_EPOCH_ALGORITHM = "ed25519"
MAX_RELEASE_EPOCH_BYTES = 16_384

_DOMAIN = b"rag-document-intelligence/retrieval-release-epoch/v1\0"
_ARTIFACT_SET_DOMAIN = b"rag-document-intelligence/retrieval-release-artifacts/v1\0"
_POLICY_DOMAIN = b"rag-document-intelligence/retrieval-release-epoch-policy/v1\0"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_GIT_OBJECT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}")
_REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]{1,100}/[A-Za-z0-9_.-]{1,100}")
_SOURCE_REF_RE = re.compile(r"refs/heads/[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")
_WORKFLOW_RE = re.compile(
    r"[.]github/workflows/[A-Za-z0-9][A-Za-z0-9_.-]{0,99}[.]ya?ml"
)
_SIGNATURE_RE = re.compile(r"[A-Za-z0-9_-]{86}")
_FIELDS = {
    "algorithm",
    "artifact_set_sha256",
    "key_id",
    "policy_sha256",
    "purpose",
    "schema_version",
    "signature",
    "verification_epoch",
}


class ReleaseEpochError(ValueError):
    """A release epoch was malformed, unauthenticated, or bound to other inputs."""

    code = "release_epoch_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


class MessageSigner(Protocol):
    def sign(self, message: bytes) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ReleaseEpochPolicy:
    repository: str
    workflow: str
    source_ref: str
    source_revision: str
    key_id: str
    public_key: bytes

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repository, str)
            or _REPOSITORY_RE.fullmatch(self.repository) is None
            or not isinstance(self.workflow, str)
            or _WORKFLOW_RE.fullmatch(self.workflow) is None
            or not isinstance(self.source_ref, str)
            or _SOURCE_REF_RE.fullmatch(self.source_ref) is None
            or not isinstance(self.source_revision, str)
            or _GIT_OBJECT_RE.fullmatch(self.source_revision) is None
            or not isinstance(self.key_id, str)
            or _IDENTIFIER_RE.fullmatch(self.key_id) is None
            or ".." in self.workflow
            or ".." in self.source_ref
            or "//" in self.source_ref
        ):
            raise ReleaseEpochError("release epoch policy is malformed")
        _load_public_key(self.public_key, "release epoch")

    @property
    def sha256(self) -> str:
        document = {
            "schema_version": RELEASE_EPOCH_POLICY_SCHEMA_VERSION,
            "key_id": self.key_id,
            "public_key_sha256": _fingerprint(self.public_key, "release epoch"),
            "repository_sha256": _text_sha256(self.repository),
            "source_ref_sha256": _text_sha256(self.source_ref),
            "source_revision": self.source_revision,
            "workflow_sha256": _text_sha256(f"{self.repository}/{self.workflow}"),
        }
        return hashlib.sha256(_POLICY_DOMAIN + _canonical_json(document)).hexdigest()


@dataclass(frozen=True, slots=True)
class ReleaseEpochAttestation:
    schema_version: str
    purpose: str
    algorithm: str
    artifact_set_sha256: str
    policy_sha256: str
    verification_epoch: int
    key_id: str
    signature: str

    @property
    def attestation_sha256(self) -> str:
        return hashlib.sha256(dump_release_epoch_attestation(self)).hexdigest()


@dataclass(frozen=True, slots=True)
class ReleaseEpochReceipt:
    schema_version: str
    attestation_sha256: str
    artifact_set_sha256: str
    policy_sha256: str
    verification_epoch: int

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


def release_artifact_set_sha256(
    *,
    candidate_snapshot_sha256: str,
    baseline_artifact_sha256: str,
    inventory_review_sha256: str,
    key_status_snapshot_sha256: str,
) -> str:
    document = {
        "baseline_artifact_sha256": _digest(
            baseline_artifact_sha256, "baseline artifact digest"
        ),
        "candidate_snapshot_sha256": _digest(
            candidate_snapshot_sha256, "candidate snapshot digest"
        ),
        "inventory_review_sha256": _digest(
            inventory_review_sha256, "inventory review digest"
        ),
        "key_status_snapshot_sha256": _digest(
            key_status_snapshot_sha256, "key status snapshot digest"
        ),
    }
    return hashlib.sha256(_ARTIFACT_SET_DOMAIN + _canonical_json(document)).hexdigest()


def create_release_epoch_attestation(
    *,
    candidate_snapshot_sha256: str,
    baseline_artifact_sha256: str,
    inventory_review_sha256: str,
    key_status_snapshot_sha256: str,
    verification_epoch: int,
    policy: ReleaseEpochPolicy,
    signer: MessageSigner,
) -> ReleaseEpochAttestation:
    """Sign the epoch for one exact reviewed retrieval release decision."""
    if not isinstance(policy, ReleaseEpochPolicy):
        raise TypeError("policy must be a ReleaseEpochPolicy")
    if type(verification_epoch) is not int or verification_epoch < 1:
        raise ReleaseEpochError("verification epoch is invalid")
    unsigned = {
        "schema_version": RELEASE_EPOCH_SCHEMA_VERSION,
        "purpose": RELEASE_EPOCH_PURPOSE,
        "algorithm": RELEASE_EPOCH_ALGORITHM,
        "artifact_set_sha256": release_artifact_set_sha256(
            candidate_snapshot_sha256=candidate_snapshot_sha256,
            baseline_artifact_sha256=baseline_artifact_sha256,
            inventory_review_sha256=inventory_review_sha256,
            key_status_snapshot_sha256=key_status_snapshot_sha256,
        ),
        "policy_sha256": policy.sha256,
        "verification_epoch": verification_epoch,
        "key_id": policy.key_id,
    }
    message = _message(unsigned)
    try:
        signature = signer.sign(message)
        _load_public_key(policy.public_key, "release epoch").verify(signature, message)
    except InvalidSignature as exc:
        raise ReleaseEpochError("release epoch signer does not match policy") from exc
    except ReleaseEpochError:
        raise
    except Exception as exc:
        raise ReleaseEpochError("release epoch signing failed") from exc
    attestation = ReleaseEpochAttestation(
        **unsigned,
        signature=_encode_signature(signature),
    )
    _validate_attestation(attestation)
    return attestation


def verify_release_epoch_attestation(
    attestation: ReleaseEpochAttestation,
    *,
    candidate_snapshot_sha256: str,
    baseline_artifact_sha256: str,
    inventory_review_sha256: str,
    key_status_snapshot_sha256: str,
    policy: ReleaseEpochPolicy,
    disallowed_public_keys: Iterable[bytes] = (),
) -> ReleaseEpochReceipt:
    """Verify signature, role separation, protected context, and exact artifacts."""
    _validate_attestation(attestation)
    if not isinstance(policy, ReleaseEpochPolicy):
        raise TypeError("policy must be a ReleaseEpochPolicy")
    release_key_fingerprint = _fingerprint(policy.public_key, "release epoch")
    try:
        disallowed = tuple(disallowed_public_keys)
    except TypeError as exc:
        raise ReleaseEpochError("disallowed key set is malformed") from exc
    if len(disallowed) > 64:
        raise ReleaseEpochError("disallowed key set is malformed")
    if any(
        hmac.compare_digest(
            release_key_fingerprint,
            _fingerprint(public_key, "disallowed role"),
        )
        for public_key in disallowed
    ):
        raise ReleaseEpochError("release epoch key must be independent")
    try:
        _load_public_key(policy.public_key, "release epoch").verify(
            _decode_signature(attestation.signature),
            _message(_unsigned_document(attestation)),
        )
    except InvalidSignature as exc:
        raise ReleaseEpochError("release epoch signature mismatch") from exc
    expected_artifacts = release_artifact_set_sha256(
        candidate_snapshot_sha256=candidate_snapshot_sha256,
        baseline_artifact_sha256=baseline_artifact_sha256,
        inventory_review_sha256=inventory_review_sha256,
        key_status_snapshot_sha256=key_status_snapshot_sha256,
    )
    if not hmac.compare_digest(attestation.artifact_set_sha256, expected_artifacts):
        raise ReleaseEpochError("release epoch artifact set mismatch")
    if (
        not hmac.compare_digest(attestation.policy_sha256, policy.sha256)
        or attestation.key_id != policy.key_id
    ):
        raise ReleaseEpochError("release epoch policy mismatch")
    return ReleaseEpochReceipt(
        schema_version=RELEASE_EPOCH_RECEIPT_SCHEMA_VERSION,
        attestation_sha256=attestation.attestation_sha256,
        artifact_set_sha256=expected_artifacts,
        policy_sha256=policy.sha256,
        verification_epoch=attestation.verification_epoch,
    )


def dump_release_epoch_attestation(attestation: ReleaseEpochAttestation) -> bytes:
    _validate_attestation(attestation)
    return _canonical_json(asdict(attestation))


def load_release_epoch_attestation(payload: bytes) -> ReleaseEpochAttestation:
    if (
        not isinstance(payload, bytes)
        or not 1 <= len(payload) <= MAX_RELEASE_EPOCH_BYTES
    ):
        raise ReleaseEpochError("release epoch payload is invalid")
    try:
        document = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ReleaseEpochError("release epoch JSON is invalid") from exc
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise ReleaseEpochError("release epoch schema is invalid")
    try:
        attestation = ReleaseEpochAttestation(**document)
    except TypeError as exc:
        raise ReleaseEpochError("release epoch schema is invalid") from exc
    _validate_attestation(attestation)
    if payload != _canonical_json(asdict(attestation)):
        raise ReleaseEpochError("release epoch is not canonical JSON")
    return attestation


def _validate_attestation(attestation: ReleaseEpochAttestation) -> None:
    if not isinstance(attestation, ReleaseEpochAttestation):
        raise TypeError("attestation must be a ReleaseEpochAttestation")
    if (
        attestation.schema_version != RELEASE_EPOCH_SCHEMA_VERSION
        or attestation.purpose != RELEASE_EPOCH_PURPOSE
        or attestation.algorithm != RELEASE_EPOCH_ALGORITHM
        or not isinstance(attestation.key_id, str)
        or _IDENTIFIER_RE.fullmatch(attestation.key_id) is None
        or type(attestation.verification_epoch) is not int
        or attestation.verification_epoch < 1
        or not isinstance(attestation.signature, str)
        or _SIGNATURE_RE.fullmatch(attestation.signature) is None
    ):
        raise ReleaseEpochError("release epoch fields are invalid")
    _digest(attestation.artifact_set_sha256, "artifact set digest")
    _digest(attestation.policy_sha256, "policy digest")


def _unsigned_document(attestation: ReleaseEpochAttestation) -> dict[str, object]:
    return {
        key: value for key, value in asdict(attestation).items() if key != "signature"
    }


def _message(unsigned: Mapping[str, object]) -> bytes:
    return _DOMAIN + _canonical_json(dict(unsigned))


def _load_public_key(encoded: bytes, role: str) -> Ed25519PublicKey:
    if not isinstance(encoded, bytes) or not 1 <= len(encoded) <= 8_192:
        raise ReleaseEpochError(f"{role} public key is malformed")
    try:
        key = load_der_public_key(encoded)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise ReleaseEpochError(f"{role} public key is malformed") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ReleaseEpochError(f"{role} public key algorithm is unsupported")
    return key


def _fingerprint(encoded: bytes, role: str) -> str:
    key = _load_public_key(encoded, role)
    return hashlib.sha256(key.public_bytes(Encoding.Raw, PublicFormat.Raw)).hexdigest()


def _encode_signature(signature: bytes) -> str:
    if not isinstance(signature, bytes) or len(signature) != 64:
        raise ReleaseEpochError("release epoch signature is malformed")
    return base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")


def _decode_signature(signature: str) -> bytes:
    try:
        decoded = base64.b64decode(signature + "==", altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ReleaseEpochError("release epoch signature is malformed") from exc
    if len(decoded) != 64:
        raise ReleaseEpochError("release epoch signature is malformed")
    return decoded


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ReleaseEpochError(f"{name} is malformed")
    return value


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        raise ReleaseEpochError("release epoch is not canonical JSON") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ReleaseEpochError("release epoch contains a duplicate field")
        document[key] = value
    return document
