from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.hybrid import HybridSparseRetriever
from src.retrieval import Chunk, SourceLocator
from src.retrieval_baseline import (
    RetrievalBaselineError,
    RetrievalRegressionComparison,
    compare_retrieval_baseline,
    load_retrieval_baseline_json,
)
from src.retrieval_eval import RetrievalCase, RetrievalMetrics, evaluate_rankings

CASE_SCHEMA_VERSION = "retrieval-evaluation-cases/v1"
SNAPSHOT_SCHEMA_VERSION = "retrieval-candidate-snapshot/v1"
GATE_SCHEMA_VERSION = "retrieval-snapshot-gate/v1"
MAX_CASE_BYTES = 1_048_576
MAX_BASELINE_BYTES = 65_536
MAX_CASES = 10_000
MAX_DOCUMENTS = 100_000
MAX_CHUNKS = 1_000_000
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_DOCUMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}")


class RetrievalSnapshotError(RuntimeError):
    """A corpus or case snapshot could not be evaluated safely."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class CanonicalRetrievalCase:
    case_id: str
    query: str
    relevant_doc_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalCandidateSnapshot:
    schema_version: str
    k: int
    document_count: int
    chunk_count: int
    case_count: int
    corpus_snapshot_sha256: str
    evaluation_cases_sha256: str
    metrics: RetrievalMetrics
    artifact_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalSnapshotGateReceipt:
    schema_version: str
    candidate_snapshot_sha256: str
    baseline_artifact_sha256: str
    candidate_metrics_sha256: str
    comparison_sha256: str
    document_count: int
    chunk_count: int
    case_count: int
    meets_policy: bool
    failure_reasons: tuple[str, ...]

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _domain_sha256(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\0" + _canonical_json(value)).hexdigest()


def _read_regular_file(path: Path, maximum: int, code: str) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum:
            raise RetrievalSnapshotError(code)
        payload = os.read(descriptor, maximum + 1)
        if len(payload) != metadata.st_size:
            raise RetrievalSnapshotError(code)
        return payload
    except OSError as exc:
        raise RetrievalSnapshotError(code) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_canonical_cases(payload: bytes) -> tuple[tuple[CanonicalRetrievalCase, ...], str]:
    """Load a bounded canonical case set and return its domain-separated digest."""

    if not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_CASE_BYTES:
        raise RetrievalSnapshotError("cases_file_invalid")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RetrievalSnapshotError("cases_duplicate_json_key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> object:
        raise RetrievalSnapshotError("cases_json_invalid")

    try:
        document = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrievalSnapshotError("cases_json_invalid") from exc
    if not isinstance(document, dict) or set(document) != {"schema_version", "cases"}:
        raise RetrievalSnapshotError("cases_schema_invalid")
    if document["schema_version"] != CASE_SCHEMA_VERSION:
        raise RetrievalSnapshotError("cases_schema_unsupported")
    rows = document["cases"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CASES:
        raise RetrievalSnapshotError("cases_count_invalid")

    cases: list[CanonicalRetrievalCase] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "case_id",
            "query",
            "relevant_doc_ids",
        }:
            raise RetrievalSnapshotError("case_schema_invalid")
        case_id = row["case_id"]
        query = row["query"]
        relevant_doc_ids = row["relevant_doc_ids"]
        try:
            query_size = len(query.encode("utf-8"))
        except (AttributeError, UnicodeEncodeError):
            raise RetrievalSnapshotError("case_query_invalid") from None
        if not isinstance(relevant_doc_ids, list):
            raise RetrievalSnapshotError("case_value_invalid")
        valid_document_ids = all(
            isinstance(doc_id, str) and _DOCUMENT_ID_RE.fullmatch(doc_id) is not None
            for doc_id in relevant_doc_ids
        )
        if (
            not isinstance(case_id, str)
            or _IDENTIFIER_RE.fullmatch(case_id) is None
            or case_id in seen
            or not isinstance(query, str)
            or not 1 <= query_size <= 16_384
            or query != query.strip()
            or not 1 <= len(relevant_doc_ids) <= 100
            or not valid_document_ids
            or len(relevant_doc_ids) != len(set(relevant_doc_ids))
            or relevant_doc_ids != sorted(relevant_doc_ids)
        ):
            raise RetrievalSnapshotError("case_value_invalid")
        seen.add(case_id)
        cases.append(
            CanonicalRetrievalCase(
                case_id=case_id,
                query=query,
                relevant_doc_ids=tuple(relevant_doc_ids),
            )
        )
    if [case.case_id for case in cases] != sorted(seen):
        raise RetrievalSnapshotError("cases_order_invalid")
    expected = {
        "schema_version": CASE_SCHEMA_VERSION,
        "cases": [asdict(case) for case in cases],
    }
    if payload != _canonical_json(expected):
        raise RetrievalSnapshotError("cases_json_not_canonical")
    return tuple(cases), _domain_sha256(b"retrieval-evaluation-cases/v1", expected)


def _row_payload(row: sqlite3.Row, names: tuple[str, ...]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for name in names:
        value = row[name]
        if value is not None and not isinstance(value, (str, int)):
            raise RetrievalSnapshotError("corpus_row_invalid")
        payload[name] = value
    return payload


def _snapshot_corpus(
    connection: sqlite3.Connection,
) -> tuple[str, frozenset[str], list[Chunk], int, int]:
    checks = connection.execute("PRAGMA quick_check").fetchall()
    if [str(row[0]) for row in checks] != ["ok"]:
        raise RetrievalSnapshotError("corpus_integrity_check_failed")
    required = {
        "documents": {
            "doc_id",
            "source",
            "media_type",
            "content_sha256",
            "metadata_json",
        },
        "chunks": {
            "chunk_id",
            "doc_id",
            "source",
            "text",
            "ordinal",
            "page_number",
            "page_start_utf8",
            "page_end_utf8",
            "page_text_sha256",
        },
    }
    for table, columns in required.items():
        observed = {
            str(row["name"])
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if not columns.issubset(observed):
            raise RetrievalSnapshotError("corpus_schema_invalid")
    document_count = int(
        connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    )
    chunk_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    if not 1 <= document_count <= MAX_DOCUMENTS or not 1 <= chunk_count <= MAX_CHUNKS:
        raise RetrievalSnapshotError("corpus_size_invalid")

    digest = hashlib.sha256(b"retrieval-corpus-snapshot/v1\0")
    doc_ids: set[str] = set()
    document_names = (
        "doc_id",
        "source",
        "media_type",
        "content_sha256",
        "metadata_json",
    )
    documents = connection.execute(
        """
        SELECT doc_id, source, media_type, content_sha256, metadata_json
        FROM documents ORDER BY doc_id
        """
    )
    for row in documents:
        payload = _row_payload(row, document_names)
        if not all(isinstance(payload[name], str) for name in document_names):
            raise RetrievalSnapshotError("corpus_row_invalid")
        doc_id = str(payload["doc_id"])
        if _DOCUMENT_ID_RE.fullmatch(doc_id) is None or doc_id in doc_ids:
            raise RetrievalSnapshotError("corpus_row_invalid")
        doc_ids.add(doc_id)
        encoded = _canonical_json(payload)
        digest.update(b"D" + len(encoded).to_bytes(8, "big") + encoded)

    chunks: list[Chunk] = []
    chunk_names = (
        "chunk_id",
        "doc_id",
        "source",
        "text",
        "ordinal",
        "page_number",
        "page_start_utf8",
        "page_end_utf8",
        "page_text_sha256",
    )
    rows = connection.execute(
        """
        SELECT chunk_id, doc_id, source, text, ordinal,
               page_number, page_start_utf8, page_end_utf8, page_text_sha256
        FROM chunks ORDER BY doc_id, ordinal, chunk_id
        """
    )
    seen_chunks: set[str] = set()
    for row in rows:
        payload = _row_payload(row, chunk_names)
        chunk_id = payload["chunk_id"]
        doc_id = payload["doc_id"]
        source = payload["source"]
        text = payload["text"]
        ordinal = payload["ordinal"]
        locator_values = tuple(
            payload[name]
            for name in (
                "page_number",
                "page_start_utf8",
                "page_end_utf8",
                "page_text_sha256",
            )
        )
        if (
            not isinstance(chunk_id, str)
            or not chunk_id
            or chunk_id in seen_chunks
            or not isinstance(doc_id, str)
            or doc_id not in doc_ids
            or not isinstance(source, str)
            or not isinstance(text, str)
            or not text
            or type(ordinal) is not int
            or ordinal < 0
            or (any(value is None for value in locator_values) and any(
                value is not None for value in locator_values
            ))
        ):
            raise RetrievalSnapshotError("corpus_row_invalid")
        try:
            locator = (
                SourceLocator(*locator_values)
                if all(value is not None for value in locator_values)
                else None
            )
        except (TypeError, ValueError) as exc:
            raise RetrievalSnapshotError("corpus_row_invalid") from exc
        seen_chunks.add(chunk_id)
        chunks.append(Chunk(chunk_id, doc_id, source, text, locator))
        committed = {
            **payload,
            "text": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
        encoded = _canonical_json(committed)
        digest.update(b"C" + len(encoded).to_bytes(8, "big") + encoded)
    if len(doc_ids) != document_count or len(chunks) != chunk_count:
        raise RetrievalSnapshotError("corpus_count_changed")
    return digest.hexdigest(), frozenset(doc_ids), chunks, document_count, chunk_count


def evaluate_candidate_snapshot(
    database: Path,
    cases_payload: bytes,
    *,
    k: int,
) -> RetrievalCandidateSnapshot:
    """Evaluate one canonical case set against one consistent SQLite read snapshot."""

    if type(k) is not int or not 1 <= k <= 1_000:
        raise ValueError("k must be an integer in [1, 1000]")
    cases, cases_sha256 = load_canonical_cases(cases_payload)
    if database.is_symlink() or not database.is_file():
        raise RetrievalSnapshotError("corpus_database_invalid")
    try:
        with sqlite3.connect(
            f"{database.resolve().as_uri()}?mode=ro",
            uri=True,
            timeout=5.0,
        ) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            (
                corpus_sha256,
                doc_ids,
                chunks,
                document_count,
                chunk_count,
            ) = _snapshot_corpus(connection)
            if any(
                relevant not in doc_ids
                for case in cases
                for relevant in case.relevant_doc_ids
            ):
                raise RetrievalSnapshotError("case_relevant_document_missing")
            retriever = HybridSparseRetriever().fit(chunks)
            rankings = [retriever.search(case.query, k=k) for case in cases]
            metrics = evaluate_rankings(
                rankings,
                [
                    RetrievalCase(case.query, frozenset(case.relevant_doc_ids))
                    for case in cases
                ],
                k=k,
            )
    except RetrievalSnapshotError:
        raise
    except (OSError, sqlite3.Error, UnicodeEncodeError, ValueError) as exc:
        raise RetrievalSnapshotError("corpus_evaluation_failed") from exc
    document = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "k": k,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "case_count": len(cases),
        "corpus_snapshot_sha256": corpus_sha256,
        "evaluation_cases_sha256": cases_sha256,
        "metrics": asdict(metrics),
    }
    return RetrievalCandidateSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        k=k,
        document_count=document_count,
        chunk_count=chunk_count,
        case_count=len(cases),
        corpus_snapshot_sha256=corpus_sha256,
        evaluation_cases_sha256=cases_sha256,
        metrics=metrics,
        artifact_sha256=_domain_sha256(b"retrieval-candidate-snapshot/v1", document),
    )


def gate_candidate_snapshot(
    snapshot: RetrievalCandidateSnapshot,
    baseline_payload: bytes,
) -> tuple[RetrievalRegressionComparison, RetrievalSnapshotGateReceipt]:
    """Load a reviewed baseline and gate one computed candidate snapshot."""

    if not isinstance(snapshot, RetrievalCandidateSnapshot):
        raise TypeError("snapshot must be RetrievalCandidateSnapshot")
    baseline = load_retrieval_baseline_json(baseline_payload)
    comparison = compare_retrieval_baseline(
        baseline,
        snapshot.metrics,
        k=snapshot.k,
        corpus_snapshot_sha256=snapshot.corpus_snapshot_sha256,
        evaluation_cases_sha256=snapshot.evaluation_cases_sha256,
    )
    receipt = RetrievalSnapshotGateReceipt(
        schema_version=GATE_SCHEMA_VERSION,
        candidate_snapshot_sha256=snapshot.artifact_sha256,
        baseline_artifact_sha256=comparison.baseline_artifact_sha256,
        candidate_metrics_sha256=comparison.candidate_metrics_sha256,
        comparison_sha256=comparison.comparison_sha256,
        document_count=snapshot.document_count,
        chunk_count=snapshot.chunk_count,
        case_count=snapshot.case_count,
        meets_policy=comparison.meets_policy,
        failure_reasons=comparison.failure_reasons,
    )
    return comparison, receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute a canonical retrieval snapshot and apply a reviewed gate."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        cases_payload = _read_regular_file(
            args.cases, MAX_CASE_BYTES, "cases_file_invalid"
        )
        baseline_payload = _read_regular_file(
            args.baseline, MAX_BASELINE_BYTES, "baseline_file_invalid"
        )
        baseline = load_retrieval_baseline_json(baseline_payload)
        snapshot = evaluate_candidate_snapshot(
            args.database,
            cases_payload,
            k=baseline.k,
        )
        _, receipt = gate_candidate_snapshot(snapshot, baseline_payload)
    except (RetrievalSnapshotError, RetrievalBaselineError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, RetrievalSnapshotError) else "retrieval_gate_failed"
        print(code, file=sys.stderr)
        return 2
    print(json.dumps(receipt.to_audit_dict(), indent=2, sort_keys=True))
    return 0 if receipt.meets_policy else 2


if __name__ == "__main__":
    raise SystemExit(main())
