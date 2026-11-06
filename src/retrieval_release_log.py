from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from src.retrieval_gate import (
    RetrievalCandidateSnapshot,
    gate_reviewed_candidate_snapshot,
)
from src.retrieval_release_epoch import ReleaseEpochPolicy

RETRIEVAL_RELEASE_LOG_ENTRY_SCHEMA_VERSION = "retrieval-release-log-entry/v1"
RETRIEVAL_RELEASE_LOG_GENESIS_SHA256 = "0" * 64
MAX_RETRIEVAL_RELEASE_LOG_BYTES = 1024 * 1024
MAX_RETRIEVAL_RELEASE_LOG_ENTRIES = 4096

_DOMAIN = b"rag-document-intelligence/retrieval-release-log-entry/v1\0"
_RECEIPT_DOMAIN = b"rag-document-intelligence/reviewed-retrieval-gate-receipt/v1\0"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_FIELDS = frozenset(
    {
        "baseline_artifact_sha256",
        "candidate_snapshot_sha256",
        "comparison_sha256",
        "entry_sha256",
        "inventory_review_sha256",
        "key_status_snapshot_sha256",
        "previous_entry_sha256",
        "release_epoch_attestation_sha256",
        "release_epoch_policy_sha256",
        "reviewed_gate_receipt_sha256",
        "schema_version",
        "sequence",
        "verification_epoch",
    }
)


class RetrievalReleaseLogError(RuntimeError):
    """A reviewed retrieval release could not be recorded without ambiguity."""

    code = "retrieval_release_log_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class RetrievalReleaseLogEntry:
    schema_version: str
    sequence: int
    previous_entry_sha256: str
    candidate_snapshot_sha256: str
    baseline_artifact_sha256: str
    comparison_sha256: str
    inventory_review_sha256: str
    key_status_snapshot_sha256: str
    release_epoch_attestation_sha256: str
    release_epoch_policy_sha256: str
    reviewed_gate_receipt_sha256: str
    verification_epoch: int
    entry_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalReleaseLogSnapshot:
    entries: tuple[RetrievalReleaseLogEntry, ...]
    head_sha256: str


def append_reviewed_retrieval_release(
    path: Path,
    snapshot: RetrievalCandidateSnapshot,
    baseline_payload: bytes,
    inventory_review_payload: bytes,
    reviewer_key_status_payload: bytes,
    release_epoch_attestation_payload: bytes,
    *,
    reviewer_public_keys: Mapping[str, bytes],
    required_review_signatures: int,
    status_authority_public_keys: Mapping[str, bytes],
    required_status_authority_signatures: int,
    release_epoch_policy: ReleaseEpochPolicy,
    expected_head_sha256: str,
) -> RetrievalReleaseLogEntry:
    """Re-run the complete reviewed gate, then append its accepted commitments."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    expected_head = _digest(expected_head_sha256, "expected head")
    comparison, receipt = gate_reviewed_candidate_snapshot(
        snapshot,
        baseline_payload,
        inventory_review_payload,
        reviewer_key_status_payload,
        release_epoch_attestation_payload,
        reviewer_public_keys=reviewer_public_keys,
        required_review_signatures=required_review_signatures,
        status_authority_public_keys=status_authority_public_keys,
        required_status_authority_signatures=required_status_authority_signatures,
        release_epoch_policy=release_epoch_policy,
    )
    if (
        comparison.meets_policy is not True
        or receipt.gate_receipt.meets_policy is not True
    ):
        raise RetrievalReleaseLogError("reviewed retrieval gate rejected the candidate")
    receipt_sha256 = hashlib.sha256(
        _RECEIPT_DOMAIN + _canonical_json(receipt.to_audit_dict())
    ).hexdigest()
    bindings: dict[str, object] = {
        "candidate_snapshot_sha256": snapshot.artifact_sha256,
        "baseline_artifact_sha256": receipt.gate_receipt.baseline_artifact_sha256,
        "comparison_sha256": comparison.comparison_sha256,
        "inventory_review_sha256": receipt.inventory_review_sha256,
        "key_status_snapshot_sha256": receipt.key_status_snapshot_sha256,
        "release_epoch_attestation_sha256": receipt.release_epoch_attestation_sha256,
        "release_epoch_policy_sha256": receipt.release_epoch_policy_sha256,
        "reviewed_gate_receipt_sha256": receipt_sha256,
        "verification_epoch": receipt.verification_epoch,
    }

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "r+b") as handle:
            descriptor = -1
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            metadata = os.fstat(handle.fileno())
            _validate_file(metadata)
            payload = _read_locked(handle, metadata.st_size)
            _require_unchanged(metadata, os.fstat(handle.fileno()))
            current = _load_payload(payload)
            replay = _find_release_epoch(
                current, receipt.release_epoch_attestation_sha256
            )
            if replay is not None:
                if all(
                    getattr(replay, key) == value for key, value in bindings.items()
                ):
                    return replay
                raise RetrievalReleaseLogError(
                    "release epoch was recorded with other evidence"
                )
            if not hmac.compare_digest(current.head_sha256, expected_head):
                raise RetrievalReleaseLogError(
                    "stale writer does not extend the current head"
                )
            if len(current.entries) >= MAX_RETRIEVAL_RELEASE_LOG_ENTRIES:
                raise RetrievalReleaseLogError("release log is full")
            if (
                current.entries
                and receipt.verification_epoch <= current.entries[-1].verification_epoch
            ):
                raise RetrievalReleaseLogError("verification epoch does not advance")
            unsigned = {
                "schema_version": RETRIEVAL_RELEASE_LOG_ENTRY_SCHEMA_VERSION,
                "sequence": len(current.entries) + 1,
                "previous_entry_sha256": current.head_sha256,
                **bindings,
            }
            entry = RetrievalReleaseLogEntry(
                **unsigned,
                entry_sha256=_entry_digest(unsigned),
            )
            encoded = _canonical_json(entry.to_audit_dict()) + b"\n"
            if len(payload) + len(encoded) > MAX_RETRIEVAL_RELEASE_LOG_BYTES:
                raise RetrievalReleaseLogError("release log is full")
            handle.seek(0, os.SEEK_END)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            return entry
    except RetrievalReleaseLogError:
        raise
    except OSError as exc:
        raise RetrievalReleaseLogError("release log is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_retrieval_release_log(
    path: Path,
    *,
    expected_head_sha256: str | None = None,
) -> RetrievalReleaseLogSnapshot:
    """Verify the complete canonical chain and an optional independently pinned head."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        metadata = os.fstat(descriptor)
        _validate_file(metadata)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = _read_locked(handle, metadata.st_size)
            _require_unchanged(metadata, os.fstat(handle.fileno()))
            result = _load_payload(payload)
    except RetrievalReleaseLogError:
        raise
    except OSError as exc:
        raise RetrievalReleaseLogError("release log is unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if expected_head_sha256 is not None:
        expected = _digest(expected_head_sha256, "expected head")
        if not hmac.compare_digest(result.head_sha256, expected):
            raise RetrievalReleaseLogError(
                "release log head does not match the pinned head"
            )
    return result


def _load_payload(payload: bytes) -> RetrievalReleaseLogSnapshot:
    if not payload:
        return RetrievalReleaseLogSnapshot((), RETRIEVAL_RELEASE_LOG_GENESIS_SHA256)
    if len(payload) > MAX_RETRIEVAL_RELEASE_LOG_BYTES:
        raise RetrievalReleaseLogError("release log exceeds the byte limit")
    if not payload.endswith(b"\n"):
        raise RetrievalReleaseLogError("release log ends with a partial entry")
    lines = payload.splitlines()
    if len(lines) > MAX_RETRIEVAL_RELEASE_LOG_ENTRIES:
        raise RetrievalReleaseLogError("release log exceeds the entry limit")
    entries: list[RetrievalReleaseLogEntry] = []
    release_epochs: set[str] = set()
    receipts: set[str] = set()
    previous = RETRIEVAL_RELEASE_LOG_GENESIS_SHA256
    previous_epoch = 0
    for sequence, line in enumerate(lines, start=1):
        try:
            document = json.loads(line, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalReleaseLogError("release log contains invalid JSON") from exc
        if not isinstance(document, dict) or set(document) != _FIELDS:
            raise RetrievalReleaseLogError("release log entry schema is invalid")
        if _canonical_json(document) != line:
            raise RetrievalReleaseLogError("release log entry is not canonical JSON")
        if (
            document["schema_version"] != RETRIEVAL_RELEASE_LOG_ENTRY_SCHEMA_VERSION
            or type(document["sequence"]) is not int
            or document["sequence"] != sequence
            or type(document["verification_epoch"]) is not int
            or document["verification_epoch"] <= previous_epoch
            or any(
                not isinstance(document[field], str)
                or _DIGEST_RE.fullmatch(document[field]) is None
                for field in _FIELDS
                - {"schema_version", "sequence", "verification_epoch"}
            )
        ):
            raise RetrievalReleaseLogError("release log entry is invalid")
        if not hmac.compare_digest(document["previous_entry_sha256"], previous):
            raise RetrievalReleaseLogError("release log chain is broken")
        claimed = document["entry_sha256"]
        unsigned = {
            key: value for key, value in document.items() if key != "entry_sha256"
        }
        if not hmac.compare_digest(claimed, _entry_digest(unsigned)):
            raise RetrievalReleaseLogError("release log entry digest mismatch")
        if document["release_epoch_attestation_sha256"] in release_epochs:
            raise RetrievalReleaseLogError("release epoch appears more than once")
        if document["reviewed_gate_receipt_sha256"] in receipts:
            raise RetrievalReleaseLogError(
                "reviewed gate receipt appears more than once"
            )
        try:
            entry = RetrievalReleaseLogEntry(**document)
        except TypeError as exc:  # pragma: no cover - exact fields checked above
            raise RetrievalReleaseLogError(
                "release log entry schema is invalid"
            ) from exc
        entries.append(entry)
        release_epochs.add(entry.release_epoch_attestation_sha256)
        receipts.add(entry.reviewed_gate_receipt_sha256)
        previous = entry.entry_sha256
        previous_epoch = entry.verification_epoch
    return RetrievalReleaseLogSnapshot(tuple(entries), previous)


def _validate_file(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or metadata.st_size > MAX_RETRIEVAL_RELEASE_LOG_BYTES
    ):
        raise RetrievalReleaseLogError("release log file is unsafe")


def _read_locked(handle: BinaryIO, expected_size: int) -> bytes:
    handle.seek(0)
    payload = handle.read(MAX_RETRIEVAL_RELEASE_LOG_BYTES + 1)
    if len(payload) > MAX_RETRIEVAL_RELEASE_LOG_BYTES:
        raise RetrievalReleaseLogError("release log exceeds the byte limit")
    if len(payload) != expected_size:
        raise RetrievalReleaseLogError("release log changed while being read")
    return payload


def _require_unchanged(before: os.stat_result, after: os.stat_result) -> None:
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise RetrievalReleaseLogError("release log changed while being read")


def _find_release_epoch(
    snapshot: RetrievalReleaseLogSnapshot,
    release_epoch_sha256: str,
) -> RetrievalReleaseLogEntry | None:
    return next(
        (
            entry
            for entry in snapshot.entries
            if entry.release_epoch_attestation_sha256 == release_epoch_sha256
        ),
        None,
    )


def _entry_digest(document: Mapping[str, object]) -> str:
    return hashlib.sha256(_DOMAIN + _canonical_json(dict(document))).hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise RetrievalReleaseLogError(f"{name} is not a canonical SHA-256")
    return value


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
        raise RetrievalReleaseLogError(
            "release log value is not canonical JSON"
        ) from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise RetrievalReleaseLogError(
                "release log entry contains a duplicate field"
            )
        document[key] = value
    return document
