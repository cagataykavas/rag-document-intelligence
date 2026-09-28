"""Fail-closed admission audit for RAG document lifecycle state.

The audit is intentionally independent from a vector database.  It consumes a
bounded manifest exported at the retrieval boundary and verifies that every
served chunk belonged to the active document revision at retrieval time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA = "rag-document-lifecycle/v1"
REPORT_SCHEMA = "rag-document-lifecycle-report/v1"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class MalformedArtifact(ValueError):
    """Raised when evidence cannot be interpreted safely."""


@dataclass(frozen=True)
class LifecyclePolicy:
    max_artifact_bytes: int = 262_144
    max_documents: int = 10_000
    max_chunks: int = 50_000
    max_queries: int = 500
    max_results_per_query: int = 100
    max_json_depth: int = 16
    max_json_nodes: int = 250_000
    max_reported_findings: int = 100
    max_artifact_age_seconds: int = 900
    max_query_age_seconds: int = 900
    max_future_skew_seconds: int = 30

    def validate(self) -> None:
        values = asdict(self)
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"policy {name} must be a positive integer")
        if self.max_reported_findings > 10_000:
            raise ValueError("policy max_reported_findings exceeds safety limit")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _private_ref(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _reject_constant(value: str) -> None:
    raise MalformedArtifact(f"non-finite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MalformedArtifact(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_artifact(path: Path, policy: LifecyclePolicy) -> dict[str, Any]:
    """Load a strict, bounded UTF-8 JSON artifact."""

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise MalformedArtifact("artifact cannot be inspected") from exc
    if size > policy.max_artifact_bytes:
        raise MalformedArtifact("artifact exceeds byte budget")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise MalformedArtifact("artifact must be readable UTF-8") from exc
    if len(raw) > policy.max_artifact_bytes:
        raise MalformedArtifact("artifact exceeds byte budget")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except MalformedArtifact:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise MalformedArtifact("artifact is not valid JSON") from exc
    if not isinstance(value, dict):
        raise MalformedArtifact("artifact root must be an object")
    _check_tree_budget(value, policy)
    return value


def _check_tree_budget(value: Any, policy: LifecyclePolicy) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > policy.max_json_nodes:
            raise MalformedArtifact("artifact exceeds JSON node budget")
        if depth > policy.max_json_depth:
            raise MalformedArtifact("artifact exceeds JSON depth budget")
        if isinstance(item, dict):
            stack.extend((key, depth + 1) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise MalformedArtifact("artifact contains a non-finite number")


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise MalformedArtifact(f"{context} fields mismatch; missing={missing}, unknown={unknown}")


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MalformedArtifact(f"{context} must be an object")
    return value


def _array(value: Any, context: str, maximum: int) -> list[Any]:
    if not isinstance(value, list):
        raise MalformedArtifact(f"{context} must be an array")
    if len(value) > maximum:
        raise MalformedArtifact(f"{context} exceeds item budget")
    return value


def _identifier(value: Any, context: str, pattern: re.Pattern[str] = _ID_RE) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise MalformedArtifact(f"{context} is not a valid identifier")
    return value


def _timestamp(value: Any, context: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise MalformedArtifact(f"{context} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise MalformedArtifact(f"{context} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise MalformedArtifact(f"{context} must use UTC")
    return parsed.astimezone(UTC)


def _integer(value: Any, context: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise MalformedArtifact(f"{context} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class _Document:
    document_id: str
    current_revision: str
    state: str
    state_changed_at: datetime


@dataclass(frozen=True)
class _Chunk:
    chunk_id: str
    document_id: str
    document_revision: str
    indexed_at: datetime


@dataclass(frozen=True)
class _Result:
    chunk_id: str
    rank: int


@dataclass(frozen=True)
class _Query:
    query_id: str
    retrieved_at: datetime
    results: tuple[_Result, ...]


def _parse_artifact(
    artifact: dict[str, Any], policy: LifecyclePolicy
) -> tuple[datetime, dict[str, _Document], dict[str, _Chunk], tuple[_Query, ...]]:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "observed_at",
            "catalog_revision",
            "index_revision",
            "documents",
            "chunks",
            "queries",
        },
        "artifact",
    )
    if artifact["schema_version"] != ARTIFACT_SCHEMA:
        raise MalformedArtifact("unsupported schema_version")
    observed_at = _timestamp(artifact["observed_at"], "observed_at")
    _identifier(artifact["catalog_revision"], "catalog_revision", _REVISION_RE)
    _identifier(artifact["index_revision"], "index_revision", _REVISION_RE)

    documents: dict[str, _Document] = {}
    for index, raw in enumerate(_array(artifact["documents"], "documents", policy.max_documents)):
        item = _object(raw, f"documents[{index}]")
        _exact_keys(
            item,
            {"document_id", "current_revision", "state", "state_changed_at"},
            f"documents[{index}]",
        )
        document_id = _identifier(item["document_id"], f"documents[{index}].document_id")
        revision = _identifier(
            item["current_revision"],
            f"documents[{index}].current_revision",
            _REVISION_RE,
        )
        state = item["state"]
        if state not in {"active", "deleted"}:
            raise MalformedArtifact(f"documents[{index}].state is invalid")
        changed = _timestamp(item["state_changed_at"], f"documents[{index}].state_changed_at")
        if changed > observed_at:
            raise MalformedArtifact("document state_changed_at is after observed_at")
        if document_id in documents:
            raise MalformedArtifact("duplicate document_id")
        documents[document_id] = _Document(document_id, revision, state, changed)

    chunks: dict[str, _Chunk] = {}
    for index, raw in enumerate(_array(artifact["chunks"], "chunks", policy.max_chunks)):
        item = _object(raw, f"chunks[{index}]")
        _exact_keys(
            item,
            {"chunk_id", "document_id", "document_revision", "indexed_at"},
            f"chunks[{index}]",
        )
        chunk_id = _identifier(item["chunk_id"], f"chunks[{index}].chunk_id")
        document_id = _identifier(item["document_id"], f"chunks[{index}].document_id")
        revision = _identifier(
            item["document_revision"],
            f"chunks[{index}].document_revision",
            _REVISION_RE,
        )
        indexed_at = _timestamp(item["indexed_at"], f"chunks[{index}].indexed_at")
        if indexed_at > observed_at:
            raise MalformedArtifact("chunk indexed_at is after observed_at")
        if chunk_id in chunks:
            raise MalformedArtifact("duplicate chunk_id")
        chunks[chunk_id] = _Chunk(chunk_id, document_id, revision, indexed_at)

    queries: list[_Query] = []
    seen_queries: set[str] = set()
    for index, raw in enumerate(_array(artifact["queries"], "queries", policy.max_queries)):
        item = _object(raw, f"queries[{index}]")
        _exact_keys(item, {"query_id", "retrieved_at", "results"}, f"queries[{index}]")
        query_id = _identifier(item["query_id"], f"queries[{index}].query_id")
        if query_id in seen_queries:
            raise MalformedArtifact("duplicate query_id")
        seen_queries.add(query_id)
        retrieved_at = _timestamp(item["retrieved_at"], f"queries[{index}].retrieved_at")
        if retrieved_at > observed_at:
            raise MalformedArtifact("query retrieved_at is after observed_at")
        results: list[_Result] = []
        seen_chunks: set[str] = set()
        seen_ranks: set[int] = set()
        for result_index, raw_result in enumerate(
            _array(
                item["results"],
                f"queries[{index}].results",
                policy.max_results_per_query,
            )
        ):
            result = _object(raw_result, f"queries[{index}].results[{result_index}]")
            _exact_keys(
                result,
                {"chunk_id", "rank"},
                f"queries[{index}].results[{result_index}]",
            )
            chunk_id = _identifier(result["chunk_id"], "result.chunk_id")
            rank = _integer(result["rank"], "result.rank")
            if chunk_id in seen_chunks or rank in seen_ranks:
                raise MalformedArtifact("query has duplicate chunk_id or rank")
            seen_chunks.add(chunk_id)
            seen_ranks.add(rank)
            results.append(_Result(chunk_id, rank))
        if seen_ranks and seen_ranks != set(range(1, len(results) + 1)):
            raise MalformedArtifact("query ranks must be contiguous from 1")
        queries.append(_Query(query_id, retrieved_at, tuple(results)))
    if not queries:
        raise MalformedArtifact("queries must contain at least one entry")
    return observed_at, documents, chunks, tuple(queries)


def audit_document_lifecycle(
    artifact: dict[str, Any],
    policy: LifecyclePolicy | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Audit served retrieval results against the authoritative lifecycle catalog."""

    policy = policy or LifecyclePolicy()
    policy.validate()
    current_time = (now or datetime.now(UTC)).astimezone(UTC)
    observed_at, documents, chunks, queries = _parse_artifact(artifact, policy)

    findings: list[dict[str, str]] = []

    def add(code: str, *, query_id: str | None = None, chunk_id: str | None = None) -> None:
        finding = {"code": code}
        if query_id is not None:
            finding["query_ref"] = _private_ref(query_id)
        if chunk_id is not None:
            finding["chunk_ref"] = _private_ref(chunk_id)
        findings.append(finding)

    future_skew = (observed_at - current_time).total_seconds()
    if future_skew > policy.max_future_skew_seconds:
        add("ARTIFACT_FROM_FUTURE")
    if (current_time - observed_at).total_seconds() > policy.max_artifact_age_seconds:
        add("STALE_ARTIFACT")

    served_results = 0
    for query in queries:
        if (observed_at - query.retrieved_at).total_seconds() > policy.max_query_age_seconds:
            add("STALE_QUERY_EVIDENCE", query_id=query.query_id)
        for result in sorted(query.results, key=lambda item: item.rank):
            served_results += 1
            chunk = chunks.get(result.chunk_id)
            if chunk is None:
                add("UNKNOWN_CHUNK", query_id=query.query_id, chunk_id=result.chunk_id)
                continue
            if chunk.indexed_at > query.retrieved_at:
                add(
                    "CHUNK_INDEXED_AFTER_RETRIEVAL",
                    query_id=query.query_id,
                    chunk_id=chunk.chunk_id,
                )
            document = documents.get(chunk.document_id)
            if document is None:
                add("UNKNOWN_DOCUMENT", query_id=query.query_id, chunk_id=chunk.chunk_id)
                continue
            if document.state_changed_at > query.retrieved_at:
                add(
                    "LIFECYCLE_STATE_AFTER_RETRIEVAL",
                    query_id=query.query_id,
                    chunk_id=chunk.chunk_id,
                )
                continue
            if document.state != "active":
                add("DOCUMENT_NOT_ACTIVE", query_id=query.query_id, chunk_id=chunk.chunk_id)
            elif chunk.document_revision != document.current_revision:
                add(
                    "SUPERSEDED_DOCUMENT_REVISION", query_id=query.query_id, chunk_id=chunk.chunk_id
                )

    findings.sort(
        key=lambda item: (
            item["code"],
            item.get("query_ref", ""),
            item.get("chunk_ref", ""),
        )
    )
    reason_codes = sorted({item["code"] for item in findings})
    reported = findings[: policy.max_reported_findings]
    policy_payload = asdict(policy)
    evidence_id = _digest({"artifact": artifact, "policy": policy_payload})
    return {
        "schema_version": REPORT_SCHEMA,
        "accepted": not findings,
        "reason_codes": reason_codes,
        "finding_count": len(findings),
        "reported_findings": reported,
        "truncated_findings": len(findings) - len(reported),
        "summary": {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "query_count": len(queries),
            "served_result_count": served_results,
        },
        "artifact_digest": _digest(artifact),
        "policy_digest": _digest(policy_payload),
        "evidence_id": evidence_id,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            os.chmod(temporary, 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-artifact-age-seconds", type=int, default=900)
    parser.add_argument("--max-query-age-seconds", type=int, default=900)
    parser.add_argument("--max-future-skew-seconds", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = LifecyclePolicy(
            max_artifact_age_seconds=args.max_artifact_age_seconds,
            max_query_age_seconds=args.max_query_age_seconds,
            max_future_skew_seconds=args.max_future_skew_seconds,
        )
        policy.validate()
        artifact = load_artifact(args.artifact, policy)
        report = audit_document_lifecycle(artifact, policy)
        output = _canonical_bytes(report) + b"\n"
        if args.output:
            _atomic_write(args.output, output)
        else:
            sys.stdout.buffer.write(output)
        return 0 if report["accepted"] else 2
    except (MalformedArtifact, ValueError) as exc:
        error = {
            "schema_version": REPORT_SCHEMA,
            "accepted": False,
            "error": "malformed_artifact",
            "detail": str(exc),
        }
        sys.stderr.buffer.write(_canonical_bytes(error) + b"\n")
        return 3
    except OSError:
        error = {
            "schema_version": REPORT_SCHEMA,
            "accepted": False,
            "error": "io_error",
        }
        sys.stderr.buffer.write(_canonical_bytes(error) + b"\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
