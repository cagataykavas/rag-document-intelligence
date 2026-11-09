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
from typing import BinaryIO

from src.generation_evidence_replay import (
    GenerationEvidenceReplayError,
    validate_generation_evidence_receipt,
)

GENERATION_REPLAY_LOG_ENTRY_SCHEMA_VERSION = "generation-replay-log-entry/v1"
GENERATION_REPLAY_LOG_GENESIS_SHA256 = "0" * 64
MAX_GENERATION_REPLAY_LOG_BYTES = 4 * 1024 * 1024
MAX_GENERATION_REPLAY_LOG_ENTRIES = 8192

_DOMAIN = b"rag-document-intelligence/generation-replay-log-entry/v1\0"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = frozenset(
    {
        "schema_version",
        "sequence",
        "previous_entry_sha256",
        "replay_receipt_sha256",
        "payload_sha256",
        "evidence_snapshot_sha256",
        "claim_support_sha256",
        "decision_sha256",
        "evidence_chunk_count",
        "claim_count",
        "accepted",
        "authentication_verified",
        "entry_sha256",
    }
)


class GenerationReplayLogError(RuntimeError):
    """A replay receipt could not be recorded without ambiguity."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class GenerationReplayLogEntry:
    schema_version: str
    sequence: int
    previous_entry_sha256: str
    replay_receipt_sha256: str
    payload_sha256: str
    evidence_snapshot_sha256: str
    claim_support_sha256: str
    decision_sha256: str
    evidence_chunk_count: int
    claim_count: int
    accepted: bool
    authentication_verified: bool
    entry_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationReplayLogSnapshot:
    entries: tuple[GenerationReplayLogEntry, ...]
    head_sha256: str


def append_generation_replay_receipt(
    path: Path,
    receipt: Mapping[str, object],
    *,
    expected_head_sha256: str,
) -> GenerationReplayLogEntry:
    """Append one verified receipt under a lock and compare-and-set head."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    expected_head = _digest(expected_head_sha256)
    try:
        verified = validate_generation_evidence_receipt(receipt)
    except GenerationEvidenceReplayError as exc:
        raise GenerationReplayLogError("RECEIPT_INVALID") from exc
    bindings = {
        "replay_receipt_sha256": verified["receipt_sha256"],
        "payload_sha256": verified["payload_sha256"],
        "evidence_snapshot_sha256": verified["evidence_snapshot_sha256"],
        "claim_support_sha256": verified["claim_support_sha256"],
        "decision_sha256": verified["decision_sha256"],
        "evidence_chunk_count": verified["evidence_chunk_count"],
        "claim_count": verified["claim_count"],
        "accepted": verified["accepted"],
        "authentication_verified": verified["authentication_verified"],
    }

    parent_descriptor = _open_private_parent(path.parent)
    descriptor = -1
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, 0o600, dir_fd=parent_descriptor)
        with os.fdopen(descriptor, "r+b") as handle:
            descriptor = -1
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            metadata = os.fstat(handle.fileno())
            _validate_file(metadata)
            payload = _read_locked(handle, metadata.st_size)
            _require_unchanged(metadata, os.fstat(handle.fileno()))
            current = _load_payload(payload)
            replay = next(
                (
                    entry
                    for entry in current.entries
                    if hmac.compare_digest(
                        entry.replay_receipt_sha256,
                        verified["receipt_sha256"],
                    )
                ),
                None,
            )
            if replay is not None:
                if all(
                    getattr(replay, key) == value for key, value in bindings.items()
                ):
                    return replay
                raise GenerationReplayLogError("RECEIPT_CONFLICT")
            if not hmac.compare_digest(current.head_sha256, expected_head):
                raise GenerationReplayLogError("STALE_HEAD")
            if len(current.entries) >= MAX_GENERATION_REPLAY_LOG_ENTRIES:
                raise GenerationReplayLogError("LOG_FULL")
            unsigned = {
                "schema_version": GENERATION_REPLAY_LOG_ENTRY_SCHEMA_VERSION,
                "sequence": len(current.entries) + 1,
                "previous_entry_sha256": current.head_sha256,
                **bindings,
            }
            entry = GenerationReplayLogEntry(
                **unsigned,
                entry_sha256=_entry_digest(unsigned),
            )
            encoded = _canonical_json(entry.to_audit_dict()) + b"\n"
            if len(payload) + len(encoded) > MAX_GENERATION_REPLAY_LOG_BYTES:
                raise GenerationReplayLogError("LOG_FULL")
            handle.seek(0, os.SEEK_END)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            os.fsync(parent_descriptor)
            return entry
    except GenerationReplayLogError:
        raise
    except OSError as exc:
        raise GenerationReplayLogError("LOG_UNAVAILABLE") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)


def verify_generation_replay_log(
    path: Path,
    *,
    expected_head_sha256: str | None = None,
) -> GenerationReplayLogSnapshot:
    """Verify the canonical chain and an optional independently pinned head."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    parent_descriptor = _open_private_parent(path.parent)
    descriptor = -1
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path.name, flags, dir_fd=parent_descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        metadata = os.fstat(descriptor)
        _validate_file(metadata)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = _read_locked(handle, metadata.st_size)
            _require_unchanged(metadata, os.fstat(handle.fileno()))
            result = _load_payload(payload)
    except GenerationReplayLogError:
        raise
    except OSError as exc:
        raise GenerationReplayLogError("LOG_UNAVAILABLE") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_descriptor)
    if expected_head_sha256 is not None:
        expected = _digest(expected_head_sha256)
        if not hmac.compare_digest(result.head_sha256, expected):
            raise GenerationReplayLogError("HEAD_MISMATCH")
    return result


def _load_payload(payload: bytes) -> GenerationReplayLogSnapshot:
    if not payload:
        return GenerationReplayLogSnapshot((), GENERATION_REPLAY_LOG_GENESIS_SHA256)
    if len(payload) > MAX_GENERATION_REPLAY_LOG_BYTES:
        raise GenerationReplayLogError("LOG_TOO_LARGE")
    if not payload.endswith(b"\n"):
        raise GenerationReplayLogError("PARTIAL_ENTRY")
    lines = payload.splitlines()
    if len(lines) > MAX_GENERATION_REPLAY_LOG_ENTRIES:
        raise GenerationReplayLogError("LOG_TOO_LARGE")
    entries: list[GenerationReplayLogEntry] = []
    receipts: set[str] = set()
    previous = GENERATION_REPLAY_LOG_GENESIS_SHA256
    for sequence, line in enumerate(lines, start=1):
        try:
            document = json.loads(line, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GenerationReplayLogError("LOG_INVALID") from exc
        if not isinstance(document, dict) or set(document) != _FIELDS:
            raise GenerationReplayLogError("LOG_INVALID")
        if _canonical_json(document) != line:
            raise GenerationReplayLogError("LOG_NOT_CANONICAL")
        if (
            document["schema_version"] != GENERATION_REPLAY_LOG_ENTRY_SCHEMA_VERSION
            or type(document["sequence"]) is not int
            or document["sequence"] != sequence
            or type(document["accepted"]) is not bool
            or type(document["authentication_verified"]) is not bool
            or type(document["evidence_chunk_count"]) is not int
            or not 1 <= document["evidence_chunk_count"] <= 50
            or type(document["claim_count"]) is not int
            or not 0 <= document["claim_count"] <= 32
            or any(
                not isinstance(document[field], str)
                or _DIGEST_RE.fullmatch(document[field]) is None
                for field in _FIELDS
                - {
                    "schema_version",
                    "sequence",
                    "accepted",
                    "authentication_verified",
                    "evidence_chunk_count",
                    "claim_count",
                }
            )
        ):
            raise GenerationReplayLogError("LOG_INVALID")
        if not hmac.compare_digest(document["previous_entry_sha256"], previous):
            raise GenerationReplayLogError("CHAIN_BROKEN")
        claimed = document["entry_sha256"]
        unsigned = {
            key: value for key, value in document.items() if key != "entry_sha256"
        }
        if not hmac.compare_digest(claimed, _entry_digest(unsigned)):
            raise GenerationReplayLogError("ENTRY_DIGEST_MISMATCH")
        if document["replay_receipt_sha256"] in receipts:
            raise GenerationReplayLogError("DUPLICATE_RECEIPT")
        entry = GenerationReplayLogEntry(**document)
        entries.append(entry)
        receipts.add(entry.replay_receipt_sha256)
        previous = entry.entry_sha256
    return GenerationReplayLogSnapshot(tuple(entries), previous)


def _open_private_parent(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GenerationReplayLogError("PARENT_UNSAFE") from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        os.close(descriptor)
        raise GenerationReplayLogError("PARENT_UNSAFE")
    return descriptor


def _validate_file(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o077
        or metadata.st_size > MAX_GENERATION_REPLAY_LOG_BYTES
    ):
        raise GenerationReplayLogError("LOG_FILE_UNSAFE")


def _read_locked(handle: BinaryIO, expected_size: int) -> bytes:
    handle.seek(0)
    payload = handle.read(MAX_GENERATION_REPLAY_LOG_BYTES + 1)
    if len(payload) > MAX_GENERATION_REPLAY_LOG_BYTES:
        raise GenerationReplayLogError("LOG_TOO_LARGE")
    if len(payload) != expected_size:
        raise GenerationReplayLogError("LOG_CHANGED")
    return payload


def _require_unchanged(before: os.stat_result, after: os.stat_result) -> None:
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ctime_ns != after.st_ctime_ns
    ):
        raise GenerationReplayLogError("LOG_CHANGED")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GenerationReplayLogError("LOG_INVALID") from exc


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise GenerationReplayLogError("DUPLICATE_FIELD")
        document[key] = value
    return document


def _digest(value: object) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise GenerationReplayLogError("HEAD_INVALID")
    return value


def _entry_digest(unsigned: Mapping[str, object]) -> str:
    return hashlib.sha256(_DOMAIN + _canonical_json(dict(unsigned))).hexdigest()
