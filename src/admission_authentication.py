from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

AUTH_SCHEMA_VERSION = "generation-admission-authentication/v2"
ADMISSION_SCHEMA_VERSION = "generation-admission/v2"
ALGORITHM = "HMAC-SHA256"
PURPOSE = "rag-generation-admission"
MINIMUM_KEY_BYTES = 32
DEFAULT_MAX_AGE_SECONDS = 300
DEFAULT_CLOCK_SKEW_SECONDS = 5
KEY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPLAY_ID_RE = re.compile(r"^[0-9a-f]{32}$")
AUDIT_FIELDS = frozenset(
    {
        "schema_version",
        "applied",
        "accepted",
        "reasons",
        "payload_bytes",
        "citation_count",
        "claim_count",
        "claim_citation_link_count",
        "payload_sha256",
        "claim_map_sha256",
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
        "replay_id",
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
    count_fields = (
        "payload_bytes",
        "citation_count",
        "claim_count",
        "claim_citation_link_count",
    )
    if any(not _valid_nonnegative_int(audit[field]) for field in count_fields):
        raise AdmissionAuthenticationError("AUDIT_SCHEMA_INVALID")
    digest_fields = (
        "payload_sha256",
        "claim_map_sha256",
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


def _validated_replay_id(replay_id: object) -> str:
    if not isinstance(replay_id, str) or REPLAY_ID_RE.fullmatch(replay_id) is None:
        raise AdmissionAuthenticationError("REPLAY_ID_INVALID")
    return replay_id


class AdmissionReplayStore:
    """Durably consume authenticated replay identities exactly once."""

    def __init__(self, path: str | Path, *, busy_timeout_seconds: float = 5.0) -> None:
        if (
            not isinstance(busy_timeout_seconds, (int, float))
            or isinstance(busy_timeout_seconds, bool)
            or not math.isfinite(busy_timeout_seconds)
            or busy_timeout_seconds <= 0
        ):
            raise AdmissionAuthenticationError("REPLAY_STORE_CONFIG_INVALID")
        self._path = Path(path)
        self._busy_timeout_ms = max(1, int(busy_timeout_seconds * 1000))
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS admission_replays (
                        replay_sha256 TEXT PRIMARY KEY,
                        expires_at_epoch INTEGER NOT NULL
                    ) WITHOUT ROWID
                    """
                )
        except sqlite3.Error as exc:
            raise AdmissionAuthenticationError("REPLAY_STORE_UNAVAILABLE") from exc

    def __repr__(self) -> str:
        return "AdmissionReplayStore(path=<redacted>)"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
        )
        connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        return connection

    def consume(
        self,
        *,
        key_id: str,
        replay_id: str,
        current_epoch: int,
        expires_at_epoch: int,
    ) -> None:
        """Atomically reject a replay while pruning identities past their window."""

        replay_key = _sha256(
            _canonical_json(
                {
                    "key_id": _validated_key_id(key_id),
                    "purpose": PURPOSE,
                    "replay_id": _validated_replay_id(replay_id),
                }
            )
        )
        if (
            not _valid_nonnegative_int(current_epoch)
            or not _valid_nonnegative_int(expires_at_epoch)
            or expires_at_epoch < current_epoch
        ):
            raise AdmissionAuthenticationError("REPLAY_POLICY_INVALID")
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "DELETE FROM admission_replays WHERE expires_at_epoch < ?",
                    (current_epoch,),
                )
                try:
                    connection.execute(
                        """
                        INSERT INTO admission_replays(replay_sha256, expires_at_epoch)
                        VALUES (?, ?)
                        """,
                        (replay_key, expires_at_epoch),
                    )
                except sqlite3.IntegrityError as exc:
                    connection.rollback()
                    raise AdmissionAuthenticationError("ADMISSION_REPLAYED") from exc
                connection.commit()
        except AdmissionAuthenticationError:
            raise
        except sqlite3.Error as exc:
            raise AdmissionAuthenticationError("REPLAY_STORE_UNAVAILABLE") from exc


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
        replay_id: str | None = None,
    ) -> dict[str, object]:
        normalized_audit = _normalize_audit(audit)
        issued_at = int(self._clock()) if issued_at_epoch is None else issued_at_epoch
        if not _valid_nonnegative_int(issued_at):
            raise AdmissionAuthenticationError("ISSUED_AT_INVALID")
        checked_replay_id = _validated_replay_id(
            secrets.token_hex(16) if replay_id is None else replay_id
        )
        claims: dict[str, object] = {
            "schema_version": AUTH_SCHEMA_VERSION,
            "applied": True,
            "algorithm": ALGORITHM,
            "purpose": PURPOSE,
            "key_id": self.key_id,
            "issued_at_epoch": issued_at,
            "replay_id": checked_replay_id,
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
    replay_store: AdmissionReplayStore | None = None,
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
    replay_id = authentication["replay_id"]
    audit_sha256 = authentication["audit_sha256"]
    signature = authentication["signature"]
    if not _valid_nonnegative_int(issued_at):
        raise AdmissionAuthenticationError("ISSUED_AT_INVALID")
    replay_id = _validated_replay_id(replay_id)
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
    if replay_store is not None:
        if (
            not isinstance(replay_store, AdmissionReplayStore)
            or max_age_seconds is None
        ):
            raise AdmissionAuthenticationError("REPLAY_POLICY_INVALID")
        replay_store.consume(
            key_id=key_id,
            replay_id=replay_id,
            current_epoch=current,
            expires_at_epoch=issued_at + max_age_seconds + clock_skew_seconds,
        )
