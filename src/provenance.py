from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.retrieval import Chunk

DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


class ProvenanceArtifactError(ValueError):
    def __init__(self, code: str, path: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.path = path


@dataclass(frozen=True)
class ProvenancePolicy:
    min_queries: int = 1
    min_results_per_query: int = 1
    max_snapshot_age_seconds: int = 7 * 24 * 60 * 60
    max_queries: int = 10_000
    max_inventory_chunks: int = 1_000_000
    max_results_per_query: int = 100
    max_chunk_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        integer_fields = (
            "min_queries",
            "min_results_per_query",
            "max_snapshot_age_seconds",
            "max_queries",
            "max_inventory_chunks",
            "max_results_per_query",
            "max_chunk_bytes",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_queries > self.max_queries:
            raise ValueError("min_queries cannot exceed max_queries")
        if self.min_results_per_query > self.max_results_per_query:
            raise ValueError("min_results_per_query cannot exceed max_results_per_query")


@dataclass(frozen=True)
class QueryProvenance:
    query_id: str
    result_count: int
    verified_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProvenanceReport:
    accepted: bool
    malformed: bool
    reason_codes: tuple[str, ...]
    snapshot_id: str | None
    snapshot_digest: str | None
    snapshot_age_seconds: float | None
    inventory_chunks: int
    query_count: int
    result_count: int
    queries: tuple[QueryProvenance, ...]
    error_path: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ProvenanceArtifactError(
            "INVALID_IDENTIFIER", path, "identifier must contain 1-200 characters"
        )
    if any(ord(character) < 32 for character in value):
        raise ProvenanceArtifactError(
            "INVALID_IDENTIFIER", path, "identifier must not contain control characters"
        )
    return value


def _digest(value: object, path: str) -> str:
    if not isinstance(value, str) or DIGEST_PATTERN.fullmatch(value) is None:
        raise ProvenanceArtifactError(
            "INVALID_DIGEST", path, "digest must be lowercase SHA-256 hex"
        )
    return value


def _timestamp(value: object, path: str) -> datetime:
    if not isinstance(value, str):
        raise ProvenanceArtifactError("INVALID_TIMESTAMP", path, "timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ProvenanceArtifactError(
            "INVALID_TIMESTAMP", path, "timestamp must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise ProvenanceArtifactError(
            "INVALID_TIMESTAMP", path, "timestamp must include a UTC offset"
        )
    return parsed.astimezone(UTC)


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProvenanceArtifactError("INVALID_OBJECT", path, "value must be an object")
    return value


def _sequence(value: object, path: str, *, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise ProvenanceArtifactError("INVALID_ARRAY", path, "value must be an array")
    if len(value) > maximum:
        raise ProvenanceArtifactError(
            "EVIDENCE_BUDGET_EXCEEDED", path, f"array exceeds the {maximum}-item budget"
        )
    return value


def _snapshot_payload(
    *,
    snapshot_id: str,
    retriever_id: str,
    chunking_policy_id: str,
    created_at: str,
    chunks: Iterable[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "retriever_id": retriever_id,
        "chunking_policy_id": chunking_policy_id,
        "created_at": created_at,
        "chunks": sorted(chunks, key=lambda item: item["chunk_id"]),
    }


def build_snapshot(
    chunks: Iterable[Chunk],
    *,
    snapshot_id: str,
    retriever_id: str,
    chunking_policy_id: str,
    created_at: datetime,
) -> dict[str, Any]:
    """Create a content-addressed index manifest from the chunks actually indexed."""
    normalized_created_at = _timestamp(created_at.isoformat(), "created_at").isoformat()
    inventory: list[dict[str, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = _identifier(chunk.chunk_id, "chunks[].chunk_id")
        if chunk_id in seen:
            raise ValueError(f"duplicate chunk_id {chunk_id!r}")
        seen.add(chunk_id)
        inventory.append(
            {
                "chunk_id": chunk_id,
                "document_id": _identifier(chunk.doc_id, "chunks[].document_id"),
                "content_sha256": _sha256(chunk.text.encode("utf-8")),
            }
        )
    if not inventory:
        raise ValueError("at least one indexed chunk is required")
    payload = _snapshot_payload(
        snapshot_id=_identifier(snapshot_id, "snapshot_id"),
        retriever_id=_identifier(retriever_id, "retriever_id"),
        chunking_policy_id=_identifier(chunking_policy_id, "chunking_policy_id"),
        created_at=normalized_created_at,
        chunks=inventory,
    )
    return {**payload, "snapshot_digest": _sha256(_canonical_json(payload))}


def bind_query_results(
    query_id: str,
    retrieval: Iterable[dict[str, Any]],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Bind ranked retrieval rows to a manifest without trusting caller-supplied digests."""
    inventory = {
        item["chunk_id"]: item
        for item in _sequence(snapshot.get("chunks"), "snapshot.chunks", maximum=1_000_000)
    }
    results: list[dict[str, Any]] = []
    for rank, raw in enumerate(retrieval, start=1):
        row = _mapping(raw, f"retrieval[{rank - 1}]")
        chunk_id = _identifier(row.get("chunk_id"), f"retrieval[{rank - 1}].chunk_id")
        text = row.get("text")
        if not isinstance(text, str):
            raise ProvenanceArtifactError(
                "INVALID_CHUNK_TEXT", f"retrieval[{rank - 1}].text", "text must be a string"
            )
        manifest_row = inventory.get(chunk_id)
        if not isinstance(manifest_row, dict):
            raise ProvenanceArtifactError(
                "UNKNOWN_CHUNK", f"retrieval[{rank - 1}].chunk_id", "chunk is absent from snapshot"
            )
        document_id = _identifier(row.get("doc_id"), f"retrieval[{rank - 1}].doc_id")
        content_digest = _sha256(text.encode("utf-8"))
        if (
            manifest_row.get("document_id") != document_id
            or manifest_row.get("content_sha256") != content_digest
        ):
            raise ProvenanceArtifactError(
                "SNAPSHOT_MEMBERSHIP_MISMATCH",
                f"retrieval[{rank - 1}]",
                "retrieval row differs from the snapshot inventory",
            )
        results.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "document_id": document_id,
                "content_sha256": content_digest,
                "text": text,
            }
        )
    return {
        "query_id": _identifier(query_id, "query_id"),
        "snapshot_id": snapshot.get("snapshot_id"),
        "retriever_id": snapshot.get("retriever_id"),
        "results": results,
    }


def _malformed(error: ProvenanceArtifactError) -> ProvenanceReport:
    return ProvenanceReport(
        accepted=False,
        malformed=True,
        reason_codes=(error.code,),
        snapshot_id=None,
        snapshot_digest=None,
        snapshot_age_seconds=None,
        inventory_chunks=0,
        query_count=0,
        result_count=0,
        queries=(),
        error_path=error.path,
        error_message=str(error),
    )


def audit_provenance(
    artifact: object,
    *,
    policy: ProvenancePolicy | None = None,
    evaluated_at: datetime | None = None,
) -> ProvenanceReport:
    """Verify that retrieval evidence belongs to one exact, fresh index snapshot."""
    try:
        selected_policy = policy or ProvenancePolicy()
        now = evaluated_at or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("evaluated_at must be timezone-aware")
        now = now.astimezone(UTC)
        root = _mapping(artifact, "$")
        snapshot = _mapping(root.get("snapshot"), "snapshot")
        if snapshot.get("schema_version") != 1:
            raise ProvenanceArtifactError(
                "UNSUPPORTED_SCHEMA", "snapshot.schema_version", "schema_version must be 1"
            )
        snapshot_id = _identifier(snapshot.get("snapshot_id"), "snapshot.snapshot_id")
        retriever_id = _identifier(snapshot.get("retriever_id"), "snapshot.retriever_id")
        chunking_policy_id = _identifier(
            snapshot.get("chunking_policy_id"), "snapshot.chunking_policy_id"
        )
        created_at = _timestamp(snapshot.get("created_at"), "snapshot.created_at")
        if created_at > now:
            raise ProvenanceArtifactError(
                "FUTURE_SNAPSHOT", "snapshot.created_at", "snapshot cannot be created in the future"
            )
        raw_inventory = _sequence(
            snapshot.get("chunks"),
            "snapshot.chunks",
            maximum=selected_policy.max_inventory_chunks,
        )
        if not raw_inventory:
            raise ProvenanceArtifactError(
                "EMPTY_SNAPSHOT", "snapshot.chunks", "snapshot must contain indexed chunks"
            )
        inventory: dict[str, dict[str, str]] = {}
        canonical_inventory: list[dict[str, str]] = []
        for index, raw in enumerate(raw_inventory):
            row = _mapping(raw, f"snapshot.chunks[{index}]")
            chunk_id = _identifier(row.get("chunk_id"), f"snapshot.chunks[{index}].chunk_id")
            if chunk_id in inventory:
                raise ProvenanceArtifactError(
                    "DUPLICATE_CHUNK_ID",
                    f"snapshot.chunks[{index}].chunk_id",
                    "snapshot chunk IDs must be unique",
                )
            normalized = {
                "chunk_id": chunk_id,
                "document_id": _identifier(
                    row.get("document_id"), f"snapshot.chunks[{index}].document_id"
                ),
                "content_sha256": _digest(
                    row.get("content_sha256"), f"snapshot.chunks[{index}].content_sha256"
                ),
            }
            inventory[chunk_id] = normalized
            canonical_inventory.append(normalized)

        claimed_snapshot_digest = _digest(
            snapshot.get("snapshot_digest"), "snapshot.snapshot_digest"
        )
        canonical_snapshot = _snapshot_payload(
            snapshot_id=snapshot_id,
            retriever_id=retriever_id,
            chunking_policy_id=chunking_policy_id,
            created_at=created_at.isoformat(),
            chunks=canonical_inventory,
        )
        computed_snapshot_digest = _sha256(_canonical_json(canonical_snapshot))
        if claimed_snapshot_digest != computed_snapshot_digest:
            raise ProvenanceArtifactError(
                "SNAPSHOT_DIGEST_MISMATCH",
                "snapshot.snapshot_digest",
                "snapshot manifest does not match its content digest",
            )

        raw_queries = _sequence(root.get("queries"), "queries", maximum=selected_policy.max_queries)
        seen_queries: set[str] = set()
        query_reports: list[QueryProvenance] = []
        total_results = 0
        for query_index, raw_query in enumerate(raw_queries):
            query = _mapping(raw_query, f"queries[{query_index}]")
            query_id = _identifier(query.get("query_id"), f"queries[{query_index}].query_id")
            if query_id in seen_queries:
                raise ProvenanceArtifactError(
                    "DUPLICATE_QUERY_ID",
                    f"queries[{query_index}].query_id",
                    "query IDs must be unique",
                )
            seen_queries.add(query_id)
            if query.get("snapshot_id") != snapshot_id:
                raise ProvenanceArtifactError(
                    "SNAPSHOT_BINDING_MISMATCH",
                    f"queries[{query_index}].snapshot_id",
                    "query is bound to a different snapshot",
                )
            if query.get("retriever_id") != retriever_id:
                raise ProvenanceArtifactError(
                    "RETRIEVER_BINDING_MISMATCH",
                    f"queries[{query_index}].retriever_id",
                    "query is bound to a different retriever",
                )
            raw_results = _sequence(
                query.get("results"),
                f"queries[{query_index}].results",
                maximum=selected_policy.max_results_per_query,
            )
            seen_chunks: set[str] = set()
            verified: list[str] = []
            for result_index, raw_result in enumerate(raw_results):
                path = f"queries[{query_index}].results[{result_index}]"
                result = _mapping(raw_result, path)
                rank = result.get("rank")
                if isinstance(rank, bool) or rank != result_index + 1:
                    raise ProvenanceArtifactError(
                        "INVALID_RANK_SEQUENCE",
                        f"{path}.rank",
                        "ranks must be contiguous and start at 1",
                    )
                chunk_id = _identifier(result.get("chunk_id"), f"{path}.chunk_id")
                if chunk_id in seen_chunks:
                    raise ProvenanceArtifactError(
                        "DUPLICATE_QUERY_CHUNK",
                        f"{path}.chunk_id",
                        "a query cannot return the same chunk twice",
                    )
                seen_chunks.add(chunk_id)
                manifest_row = inventory.get(chunk_id)
                if manifest_row is None:
                    raise ProvenanceArtifactError(
                        "UNKNOWN_CHUNK", f"{path}.chunk_id", "chunk is absent from snapshot"
                    )
                document_id = _identifier(result.get("document_id"), f"{path}.document_id")
                claimed_content_digest = _digest(
                    result.get("content_sha256"), f"{path}.content_sha256"
                )
                text = result.get("text")
                if not isinstance(text, str):
                    raise ProvenanceArtifactError(
                        "INVALID_CHUNK_TEXT", f"{path}.text", "text must be a string"
                    )
                if len(text.encode("utf-8")) > selected_policy.max_chunk_bytes:
                    raise ProvenanceArtifactError(
                        "CHUNK_BYTE_BUDGET_EXCEEDED",
                        f"{path}.text",
                        "retrieved chunk exceeds the configured byte budget",
                    )
                actual_content_digest = _sha256(text.encode("utf-8"))
                if claimed_content_digest != actual_content_digest:
                    raise ProvenanceArtifactError(
                        "RESULT_CONTENT_DIGEST_MISMATCH",
                        f"{path}.content_sha256",
                        "retrieved text does not match its content digest",
                    )
                if (
                    manifest_row["document_id"] != document_id
                    or manifest_row["content_sha256"] != claimed_content_digest
                ):
                    raise ProvenanceArtifactError(
                        "SNAPSHOT_MEMBERSHIP_MISMATCH",
                        path,
                        "retrieved chunk identity differs from the snapshot inventory",
                    )
                verified.append(chunk_id)
            total_results += len(verified)
            query_reports.append(QueryProvenance(query_id, len(verified), tuple(verified)))

        age_seconds = (now - created_at).total_seconds()
        reasons: list[str] = []
        if len(query_reports) < selected_policy.min_queries:
            reasons.append("INSUFFICIENT_QUERY_EVIDENCE")
        if any(item.result_count < selected_policy.min_results_per_query for item in query_reports):
            reasons.append("INSUFFICIENT_RESULTS_PER_QUERY")
        if age_seconds > selected_policy.max_snapshot_age_seconds:
            reasons.append("STALE_INDEX_SNAPSHOT")
        return ProvenanceReport(
            accepted=not reasons,
            malformed=False,
            reason_codes=tuple(reasons),
            snapshot_id=snapshot_id,
            snapshot_digest=computed_snapshot_digest,
            snapshot_age_seconds=age_seconds,
            inventory_chunks=len(inventory),
            query_count=len(query_reports),
            result_count=total_results,
            queries=tuple(query_reports),
        )
    except ProvenanceArtifactError as error:
        return _malformed(error)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit retrieval evidence against a content-addressed index snapshot."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--evaluated-at", help="Timezone-aware ISO-8601 time; defaults to now")
    parser.add_argument("--min-queries", type=int, default=1)
    parser.add_argument("--min-results-per-query", type=int, default=1)
    parser.add_argument("--max-snapshot-age-seconds", type=int, default=7 * 24 * 60 * 60)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = json.loads(args.artifact.read_text(encoding="utf-8"))
        evaluated_at = (
            _timestamp(args.evaluated_at, "--evaluated-at") if args.evaluated_at else None
        )
        policy = ProvenancePolicy(
            min_queries=args.min_queries,
            min_results_per_query=args.min_results_per_query,
            max_snapshot_age_seconds=args.max_snapshot_age_seconds,
        )
        report = audit_provenance(payload, policy=policy, evaluated_at=evaluated_at)
    except (OSError, json.JSONDecodeError, ProvenanceArtifactError, ValueError) as error:
        report = _malformed(
            error
            if isinstance(error, ProvenanceArtifactError)
            else ProvenanceArtifactError("INVALID_INPUT", "$", str(error))
        )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if report.malformed:
        return 2
    return 0 if report.accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
