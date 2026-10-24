from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

from src.prompting import EvidenceSnippet

SCHEMA_VERSION = "evidence-snapshot/v1"
MAX_SNAPSHOT_CHUNKS = 50
MAX_CHUNK_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 4_096


class EvidenceSnapshotError(ValueError):
    """Retrieved evidence cannot be represented by the bounded snapshot contract."""


@dataclass(frozen=True, slots=True)
class EvidenceCommitment:
    chunk_id: str
    source_sha256: str
    text_sha256: str
    text_utf8_bytes: int


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    schema_version: str
    commitments: tuple[EvidenceCommitment, ...]
    total_text_utf8_bytes: int
    snapshot_sha256: str

    @property
    def chunk_ids(self) -> frozenset[str]:
        return frozenset(item.chunk_id for item in self.commitments)

    def to_audit_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": self.schema_version,
            "chunk_count": len(self.commitments),
            "total_text_utf8_bytes": self.total_text_utf8_bytes,
            "snapshot_sha256": self.snapshot_sha256,
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_evidence_snapshot(snippets: Iterable[EvidenceSnippet]) -> EvidenceSnapshot:
    """Bind chunk identity, source identity and exact text without retaining content."""

    rows = list(snippets)
    if not 1 <= len(rows) <= MAX_SNAPSHOT_CHUNKS:
        raise EvidenceSnapshotError("evidence snapshot must contain 1-50 chunks")

    commitments: list[EvidenceCommitment] = []
    seen_ids: set[str] = set()
    total_bytes = 0
    for snippet in rows:
        if not isinstance(snippet, EvidenceSnippet):
            raise EvidenceSnapshotError(
                "evidence snapshot contains an unsupported item"
            )
        chunk_id_bytes = snippet.chunk_id.encode("utf-8")
        source_bytes = snippet.source.encode("utf-8")
        text_bytes = snippet.text.encode("utf-8")
        if not chunk_id_bytes or len(chunk_id_bytes) > 200:
            raise EvidenceSnapshotError("chunk_id must contain 1-200 UTF-8 bytes")
        if snippet.chunk_id in seen_ids:
            raise EvidenceSnapshotError("evidence snapshot contains duplicate chunk_id")
        if not source_bytes or len(source_bytes) > MAX_SOURCE_BYTES:
            raise EvidenceSnapshotError("source must contain 1-4096 UTF-8 bytes")
        if not text_bytes or len(text_bytes) > MAX_CHUNK_BYTES:
            raise EvidenceSnapshotError("chunk text must contain 1-1048576 UTF-8 bytes")
        seen_ids.add(snippet.chunk_id)
        total_bytes += len(text_bytes)
        commitments.append(
            EvidenceCommitment(
                chunk_id=snippet.chunk_id,
                source_sha256=_sha256(source_bytes),
                text_sha256=_sha256(text_bytes),
                text_utf8_bytes=len(text_bytes),
            )
        )

    normalized = tuple(sorted(commitments, key=lambda item: item.chunk_id))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "commitments": [asdict(item) for item in normalized],
        "total_text_utf8_bytes": total_bytes,
    }
    return EvidenceSnapshot(
        schema_version=SCHEMA_VERSION,
        commitments=normalized,
        total_text_utf8_bytes=total_bytes,
        snapshot_sha256=_sha256(_canonical_json(manifest)),
    )


def verify_evidence_snapshot(
    snapshot: EvidenceSnapshot,
    snippets: Iterable[EvidenceSnippet],
) -> bool:
    """Rebuild and compare a snapshot without exposing evidence in the audit record."""

    try:
        rebuilt = build_evidence_snapshot(snippets)
    except EvidenceSnapshotError:
        return False
    return snapshot == rebuilt
