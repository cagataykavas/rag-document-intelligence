from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from src.entailment_paired_regression import PairedRegressionComparison
from src.entailment_release_manifest import (
    CandidateArtifact,
    CandidateReleaseManifest,
    verify_candidate_release_manifest,
)
from src.entailment_slice_regression import PairedSliceRegressionComparison

RELEASE_LOG_ENTRY_SCHEMA_VERSION = "entailment-release-log-entry/v1"
RELEASE_LOG_GENESIS_SHA256 = "0" * 64
_DOMAIN = b"rag-document-intelligence/release-log-entry/v1\x00"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_MAX_LOG_BYTES = 4 * 1024 * 1024
_MAX_LINE_BYTES = 2_048
_MAX_ENTRIES = 10_000
_FIELDS = {
    "schema_version",
    "sequence",
    "previous_entry_sha256",
    "manifest_sha256",
    "release_id",
    "artifact_sha256",
    "source_commit_sha",
    "issued_at_epoch",
    "key_id",
    "entry_sha256",
}


class ReleaseLogError(ValueError):
    """A release log is malformed, forked, rolled back, or unavailable."""

    code = "release_log_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


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
        raise ReleaseLogError("entry is not canonical JSON") from exc


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ReleaseLogError(f"{name} must be a canonical lowercase SHA-256")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ReleaseLogError(f"{name} is not a canonical identifier")
    return value


def _entry_digest(document: Mapping[str, object]) -> str:
    return hashlib.sha256(_DOMAIN + _canonical_json(dict(document))).hexdigest()


@dataclass(frozen=True, slots=True)
class ReleaseLogEntry:
    schema_version: str
    sequence: int
    previous_entry_sha256: str
    manifest_sha256: str
    release_id: str
    artifact_sha256: str
    source_commit_sha: str
    issued_at_epoch: int
    key_id: str
    entry_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReleaseLogSnapshot:
    entries: tuple[ReleaseLogEntry, ...]
    head_sha256: str


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseLogError("entry contains a duplicate field")
        result[key] = value
    return result


def _load_payload(payload: bytes) -> ReleaseLogSnapshot:
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    if len(payload) > _MAX_LOG_BYTES:
        raise ReleaseLogError("log exceeds the byte limit")
    if payload and not payload.endswith(b"\n"):
        raise ReleaseLogError("log ends with a partial entry")
    lines = payload.splitlines()
    if len(lines) > _MAX_ENTRIES:
        raise ReleaseLogError("log exceeds the entry limit")

    entries: list[ReleaseLogEntry] = []
    release_ids: set[str] = set()
    manifest_digests: set[str] = set()
    artifact_digests: set[str] = set()
    previous_digest = RELEASE_LOG_GENESIS_SHA256
    previous_issued_at = 0
    for expected_sequence, line in enumerate(lines, start=1):
        if not line or len(line) > _MAX_LINE_BYTES:
            raise ReleaseLogError("entry size is invalid")
        try:
            document = json.loads(line, object_pairs_hook=_reject_duplicates)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ReleaseLogError("entry is not valid JSON") from exc
        if not isinstance(document, dict) or set(document) != _FIELDS:
            raise ReleaseLogError("entry fields do not match the schema")
        if _canonical_json(document) != line:
            raise ReleaseLogError("entry JSON is not canonical")
        if document["schema_version"] != RELEASE_LOG_ENTRY_SCHEMA_VERSION:
            raise ReleaseLogError("entry schema is unsupported")
        if (
            type(document["sequence"]) is not int
            or document["sequence"] != expected_sequence
        ):
            raise ReleaseLogError("entry sequence is not contiguous")
        if (
            type(document["issued_at_epoch"]) is not int
            or document["issued_at_epoch"] < 1
        ):
            raise ReleaseLogError("issued_at_epoch must be a positive integer")
        if document["issued_at_epoch"] <= previous_issued_at:
            raise ReleaseLogError("manifest issue times do not advance")
        _digest(document["previous_entry_sha256"], "previous_entry_sha256")
        _digest(document["manifest_sha256"], "manifest_sha256")
        _digest(document["artifact_sha256"], "artifact_sha256")
        claimed_entry_digest = _digest(document["entry_sha256"], "entry_sha256")
        _identifier(document["release_id"], "release_id")
        _identifier(document["key_id"], "key_id")
        if (
            not isinstance(document["source_commit_sha"], str)
            or _GIT_SHA_RE.fullmatch(document["source_commit_sha"]) is None
        ):
            raise ReleaseLogError("source_commit_sha is not a canonical Git SHA")
        unsigned = {
            key: value for key, value in document.items() if key != "entry_sha256"
        }
        expected_digest = _entry_digest(unsigned)
        if not hmac.compare_digest(claimed_entry_digest, expected_digest):
            raise ReleaseLogError("entry digest mismatch")
        if not hmac.compare_digest(document["previous_entry_sha256"], previous_digest):
            raise ReleaseLogError("entry does not extend the previous head")
        if document["release_id"] in release_ids:
            raise ReleaseLogError("release_id is already present")
        if document["manifest_sha256"] in manifest_digests:
            raise ReleaseLogError("manifest is already present")
        if document["artifact_sha256"] in artifact_digests:
            raise ReleaseLogError("artifact is already present")
        try:
            entry = ReleaseLogEntry(**document)
        except TypeError as exc:  # pragma: no cover - exact fields checked above
            raise ReleaseLogError("entry fields do not match the schema") from exc
        entries.append(entry)
        release_ids.add(entry.release_id)
        manifest_digests.add(entry.manifest_sha256)
        artifact_digests.add(entry.artifact_sha256)
        previous_digest = entry.entry_sha256
        previous_issued_at = entry.issued_at_epoch
    return ReleaseLogSnapshot(tuple(entries), previous_digest)


def verify_release_log(
    path: Path,
    *,
    expected_head_sha256: str | None = None,
) -> ReleaseLogSnapshot:
    """Verify canonical entries, the complete hash chain, and an optional pinned head."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        payload = b""
    except OSError as exc:
        raise ReleaseLogError("log is unavailable") from exc
    snapshot = _load_payload(payload)
    if expected_head_sha256 is not None:
        expected = _digest(expected_head_sha256, "expected_head_sha256")
        if not hmac.compare_digest(snapshot.head_sha256, expected):
            raise ReleaseLogError("log head does not match the pinned head")
    return snapshot


def _locked_payload(handle: BinaryIO) -> bytes:
    handle.seek(0)
    payload = handle.read(_MAX_LOG_BYTES + 1)
    if len(payload) > _MAX_LOG_BYTES:
        raise ReleaseLogError("log exceeds the byte limit")
    return payload


def append_verified_candidate_release(
    path: Path,
    manifest: CandidateReleaseManifest,
    global_comparison: PairedRegressionComparison,
    slice_comparison: PairedSliceRegressionComparison,
    candidate: CandidateArtifact,
    *,
    keys: Mapping[str, bytes],
    current_epoch: int,
    expected_head_sha256: str,
) -> ReleaseLogEntry:
    """Authenticate and atomically append one candidate using head compare-and-swap."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    expected_head = _digest(expected_head_sha256, "expected_head_sha256")
    verify_candidate_release_manifest(
        manifest,
        global_comparison,
        slice_comparison,
        candidate,
        keys=keys,
        current_epoch=current_epoch,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ReleaseLogError("log is unavailable") from exc
    try:
        with os.fdopen(descriptor, "r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            payload = _locked_payload(handle)
            snapshot = _load_payload(payload)
            if not hmac.compare_digest(snapshot.head_sha256, expected_head):
                raise ReleaseLogError("stale writer does not extend the current head")
            if any(
                entry.release_id == manifest.release_id for entry in snapshot.entries
            ):
                raise ReleaseLogError("release_id is already present")
            if any(
                entry.manifest_sha256 == manifest.manifest_sha256
                for entry in snapshot.entries
            ):
                raise ReleaseLogError("manifest is already present")
            if any(
                entry.artifact_sha256 == manifest.artifact_sha256
                for entry in snapshot.entries
            ):
                raise ReleaseLogError("artifact is already present")
            if (
                snapshot.entries
                and manifest.issued_at_epoch <= snapshot.entries[-1].issued_at_epoch
            ):
                raise ReleaseLogError("manifest issue time does not advance")
            unsigned: dict[str, object] = {
                "schema_version": RELEASE_LOG_ENTRY_SCHEMA_VERSION,
                "sequence": len(snapshot.entries) + 1,
                "previous_entry_sha256": snapshot.head_sha256,
                "manifest_sha256": manifest.manifest_sha256,
                "release_id": manifest.release_id,
                "artifact_sha256": manifest.artifact_sha256,
                "source_commit_sha": manifest.source_commit_sha,
                "issued_at_epoch": manifest.issued_at_epoch,
                "key_id": manifest.key_id,
            }
            entry = ReleaseLogEntry(**unsigned, entry_sha256=_entry_digest(unsigned))
            encoded = _canonical_json(entry.to_audit_dict()) + b"\n"
            if (
                len(encoded) > _MAX_LINE_BYTES
                or len(payload) + len(encoded) > _MAX_LOG_BYTES
            ):
                raise ReleaseLogError("log capacity would be exceeded")
            handle.seek(0, os.SEEK_END)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            return entry
    except OSError as exc:
        raise ReleaseLogError("log write failed") from exc
