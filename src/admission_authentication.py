from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import time
from collections.abc import Mapping
from typing import Any, Callable

AUTH_SCHEMA_VERSION = "generation-admission-authentication/v1"
ADMISSION_SCHEMA_VERSION = "generation-admission/v1"
ALGORITHM = "HMAC-SHA256"
PURPOSE = "rag-generation-admission"
MINIMUM_KEY_BYTES = 32
DEFAULT_MAX_AGE_SECONDS = 300
DEFAULT_CLOCK_SKEW_SECONDS = 5
KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "applied",
        "accepted",
        "reasons",
        "payload_bytes",
        "citation_count",
        "payload_sha256",
        "allowed_evidence_sha256",
        "policy_sha256",
        "decision_sha256",
    }
)
AUTH_FIELDS = frozenset(
    {
        "schema_version",
        "applied",
        "algorithm",
        "purpose",
        "key_id",
        "issued_at_epoch",
        "audit_sha256",
        "signature",
    }
)


class AdmissionAuthenticationError(ValueError):
    """Stable, non-sensitive admission-authentication failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AdmissionAuthenticationError("AUDIT_NOT_CANONICAL") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _normalize_audit(audit: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(audit, Mapping) or set(audit) != AUDIT_FIELDS:
        raise AdmissionAuthenticationError("AUDIT_SCHEMA_INVALID")
    if audit["schema_version"] != ADMISSION_SCHEMA_VERSION:
        raise AdmissionAuthenticationError("AUDIT_SCHEMA_INVALID")
    if audit["applied"] is not True or not isinstance(audit["accepted"], bool):
        raise AdmissionAuthenticationError("AUDIT_SCHEMA_INVALID")
    reasons = audit["reasons"]
    if not isinstance(reasons, (list, tuple)) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise AdmissionAuthenticationError("AUDIT_SCHEMA_INVALID")
    if len(reasons) != len(set(reasons)):
        raise AdmissionAuthenticationError("AUDIT_SCHEMA_INVALID")
    if not _valid_nonnegative_int(audit["payload_bytes"]) or not _valid_nonnegative_int(
        audit["citation_count"]
    ):
        raise AdmissionAuthenticationError("AUDIT_SCHEMA_INVALID")
    digest_fields = (
        "payload_sha256",
        "allowed_evidence_sha256",
        "policy_sha256",
        "decision_sha256",
    )
    if any(
        not isinstance(audit[field], str) or SHA256_RE.fullmatch(audit[field]) is None
        for field in digest_fields
    ):
        raise AdmissionAuthenticationError("AUDIT_SCHEMA_INVALID")

    normalized = dict(audit)
    normalized["reasons"] = list(reasons)
    decision = {
        "schema_version": normalized["schema_version"],
        "accepted": normalized["accepted"],
        "reasons": normalized["reasons"],
        "payload_sha256": normalized["payload_sha256"],
        "allowed_evidence_sha256": normalized["allowed_evidence_sha256"],
        "policy_sha256": normalized["policy_sha256"],
    }
    if not hmac.compare_digest(
        str(normalized["decision_sha256"]), _sha256(_canonical_json(decision))
    ):
        raise AdmissionAuthenticationError("DECISION_DIGEST_MISMATCH")
    return normalized


def _validated_key(key: object) -> bytes:
    if not isinstance(key, bytes) or len(key) < MINIMUM_KEY_BYTES:
        raise AdmissionAuthenticationError("SIGNING_KEY_INVALID")
    return key


def _validated_key_id(key_id: object) -> str:
    if not isinstance(key_id, str) or KEY_ID_RE.fullmatch(key_id) is None:
        raise AdmissionAuthenticationError("KEY_ID_INVALID")
    return key_id


class AdmissionAuthenticator:
    """Authenticate admission evidence without retaining generated content."""

    def __init__(
        self,
        key_id: str,
        key: bytes,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.key_id = _validated_key_id(key_id)
        self._key = _validated_key(key)
        self._clock = clock

    def __repr__(self) -> str:
        return f"AdmissionAuthenticator(key_id={self.key_id!r}, key=<redacted>)"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> AdmissionAuthenticator | None:
        source = os.environ if environ is None else environ
        key_id = source.get("RAG_ADMISSION_HMAC_KEY_ID")
        encoded_key = source.get("RAG_ADMISSION_HMAC_KEY_B64")
        if key_id is None and encoded_key is None:
            return None
        if not key_id or not encoded_key:
            raise AdmissionAuthenticationError("SIGNING_CONFIG_INCOMPLETE")
        try:
            key = base64.b64decode(encoded_key, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AdmissionAuthenticationError("SIGNING_KEY_INVALID") from exc
        return cls(key_id, key, clock=clock)

    def authenticate(
        self,
        audit: Mapping[str, object],
        *,
        issued_at_epoch: int | None = None,
    ) -> dict[str, object]:
        normalized_audit = _normalize_audit(audit)
        issued_at = int(self._clock()) if issued_at_epoch is None else issued_at_epoch
        if not _valid_nonnegative_int(issued_at):
            raise AdmissionAuthenticationError("ISSUED_AT_INVALID")
        claims: dict[str, object] = {
            "schema_version": AUTH_SCHEMA_VERSION,
            "applied": True,
            "algorithm": ALGORITHM,
            "purpose": PURPOSE,
            "key_id": self.key_id,
            "issued_at_epoch": issued_at,
            "audit_sha256": _sha256(_canonical_json(normalized_audit)),
        }
        signature = hmac.new(
            self._key, _canonical_json(claims), hashlib.sha256
        ).hexdigest()
        return {**claims, "signature": signature}


def verify_admission_authentication(
    audit: Mapping[str, object],
    authentication: Mapping[str, object],
    keyring: Mapping[str, bytes],
    *,
    now_epoch: int | None = None,
    max_age_seconds: int | None = DEFAULT_MAX_AGE_SECONDS,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
) -> None:
    """Fail closed unless the exact admission audit has a valid fresh HMAC."""

    normalized_audit = _normalize_audit(audit)
    if not isinstance(authentication, Mapping) or set(authentication) != AUTH_FIELDS:
        raise AdmissionAuthenticationError("AUTHENTICATION_SCHEMA_INVALID")
    if (
        authentication["schema_version"] != AUTH_SCHEMA_VERSION
        or authentication["applied"] is not True
        or authentication["algorithm"] != ALGORITHM
        or authentication["purpose"] != PURPOSE
    ):
        raise AdmissionAuthenticationError("AUTHENTICATION_SCHEMA_INVALID")
    key_id = _validated_key_id(authentication["key_id"])
    issued_at = authentication["issued_at_epoch"]
    audit_sha256 = authentication["audit_sha256"]
    signature = authentication["signature"]
    if not _valid_nonnegative_int(issued_at):
        raise AdmissionAuthenticationError("ISSUED_AT_INVALID")
    if not isinstance(audit_sha256, str) or SHA256_RE.fullmatch(audit_sha256) is None:
        raise AdmissionAuthenticationError("AUTHENTICATION_SCHEMA_INVALID")
    if not isinstance(signature, str) or SHA256_RE.fullmatch(signature) is None:
        raise AdmissionAuthenticationError("AUTHENTICATION_SCHEMA_INVALID")

    expected_audit_sha256 = _sha256(_canonical_json(normalized_audit))
    if not hmac.compare_digest(audit_sha256, expected_audit_sha256):
        raise AdmissionAuthenticationError("AUDIT_DIGEST_MISMATCH")
    key = keyring.get(key_id)
    if key is None:
        raise AdmissionAuthenticationError("SIGNING_KEY_UNKNOWN")
    key = _validated_key(key)
    claims = {field: authentication[field] for field in AUTH_FIELDS - {"signature"}}
    expected_signature = hmac.new(
        key, _canonical_json(claims), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise AdmissionAuthenticationError("SIGNATURE_INVALID")

    if not _valid_nonnegative_int(clock_skew_seconds):
        raise AdmissionAuthenticationError("FRESHNESS_POLICY_INVALID")
    if max_age_seconds is not None and not _valid_nonnegative_int(max_age_seconds):
        raise AdmissionAuthenticationError("FRESHNESS_POLICY_INVALID")
    current = int(time.time()) if now_epoch is None else now_epoch
    if not _valid_nonnegative_int(current):
        raise AdmissionAuthenticationError("FRESHNESS_POLICY_INVALID")
    if issued_at > current + clock_skew_seconds:
        raise AdmissionAuthenticationError("SIGNATURE_FROM_FUTURE")
    if (
        max_age_seconds is not None
        and current - issued_at > max_age_seconds + clock_skew_seconds
    ):
        raise AdmissionAuthenticationError("SIGNATURE_STALE")
