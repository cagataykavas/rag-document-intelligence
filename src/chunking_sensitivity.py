from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXIT_ACCEPTED = 0
EXIT_REJECTED = 2
EXIT_MALFORMED = 3

MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_POLICY_BYTES = 64 * 1024
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ChunkingAuditError(ValueError):
    """The chunking-sensitivity artifact or policy is malformed."""


@dataclass(frozen=True, slots=True)
class DocumentSpec:
    document_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class Span:
    document_sha256: str
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class RankedSpan:
    rank: int
    document_sha256: str
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class VariantResult:
    configuration_sha256: str
    results: tuple[RankedSpan, ...]


@dataclass(frozen=True, slots=True)
class QueryEvidence:
    query_id: str
    required_spans: tuple[Span, ...]
    variants: tuple[VariantResult, ...]


@dataclass(frozen=True, slots=True)
class ChunkingArtifact:
    schema_version: int
    corpus_sha256: str
    retriever_sha256: str
    baseline_configuration_sha256: str
    configurations: tuple[str, ...]
    documents: tuple[DocumentSpec, ...]
    queries: tuple[QueryEvidence, ...]


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    min_configurations: int = 2
    max_configurations: int = 16
    max_documents: int = 10_000
    max_queries: int = 1_000
    max_results_per_configuration: int = 100
    max_total_spans: int = 500_000
    min_span_coverage_jaccard: float = 0.60
    min_document_jaccard: float = 0.50
    min_rank_weighted_document_jaccard: float = 0.50
    min_required_span_recall: float = 0.80
    max_findings: int = 256

    def __post_init__(self) -> None:
        integer_fields = (
            "min_configurations",
            "max_configurations",
            "max_documents",
            "max_queries",
            "max_results_per_configuration",
            "max_total_spans",
            "max_findings",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ChunkingAuditError(f"{field} must be an integer")
        if not 2 <= self.min_configurations <= self.max_configurations <= 32:
            raise ChunkingAuditError("configuration bounds must satisfy 2 <= min <= max <= 32")
        if not 1 <= self.max_documents <= 100_000:
            raise ChunkingAuditError("max_documents must be between 1 and 100000")
        if not 1 <= self.max_queries <= 10_000:
            raise ChunkingAuditError("max_queries must be between 1 and 10000")
        if not 1 <= self.max_results_per_configuration <= 1_000:
            raise ChunkingAuditError("max_results_per_configuration must be between 1 and 1000")
        if not 1 <= self.max_total_spans <= 5_000_000:
            raise ChunkingAuditError("max_total_spans must be between 1 and 5000000")
        for field in (
            "min_span_coverage_jaccard",
            "min_document_jaccard",
            "min_rank_weighted_document_jaccard",
            "min_required_span_recall",
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ChunkingAuditError(f"{field} must be finite and between 0 and 1")
        if not 1 <= self.max_findings <= 4096:
            raise ChunkingAuditError("max_findings must be between 1 and 4096")

    @property
    def digest(self) -> str:
        return _sha256_json(asdict(self))


@dataclass(frozen=True, slots=True)
class Comparison:
    query_ref: str
    configuration_sha256: str
    baseline_required_span_recall: float
    variant_required_span_recall: float
    span_coverage_jaccard: float
    document_jaccard: float
    rank_weighted_document_jaccard: float


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    query_ref: str
    configuration_sha256: str
    observed: float
    limit: float


@dataclass(frozen=True, slots=True)
class ChunkingReport:
    schema_version: int
    outcome: str
    artifact_sha256: str
    policy_sha256: str
    corpus_sha256: str
    retriever_sha256: str
    baseline_configuration_sha256: str
    configurations: int
    documents: int
    queries: int
    comparisons: tuple[Comparison, ...]
    findings: tuple[Finding, ...]
    findings_truncated: bool
    evidence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_json(values: object) -> str:
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _reference(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ChunkingAuditError(f"{field} must be a canonical lowercase SHA-256 digest")
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise ChunkingAuditError(f"{field} must be a bounded canonical identifier")
    return value


def _integer(value: object, field: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ChunkingAuditError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise ChunkingAuditError(f"{field} is outside its allowed range")
    return value


def _expect_fields(values: dict[str, Any], expected: set[str], field: str) -> None:
    missing = expected - values.keys()
    unknown = values.keys() - expected
    if missing:
        raise ChunkingAuditError(f"{field} missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ChunkingAuditError(f"{field} has unknown fields: {', '.join(sorted(unknown))}")


def _document(values: object) -> DocumentSpec:
    if not isinstance(values, dict):
        raise ChunkingAuditError("document must be an object")
    _expect_fields(values, {"document_sha256", "size_bytes"}, "document")
    return DocumentSpec(
        document_sha256=_digest(values["document_sha256"], "document_sha256"),
        size_bytes=_integer(values["size_bytes"], "size_bytes", minimum=1, maximum=2**40),
    )


def _span(
    values: object,
    *,
    documents: dict[str, DocumentSpec],
    ranked: bool,
) -> Span | RankedSpan:
    if not isinstance(values, dict):
        raise ChunkingAuditError("span must be an object")
    expected = {"document_sha256", "start_byte", "end_byte"}
    if ranked:
        expected.add("rank")
    _expect_fields(values, expected, "ranked span" if ranked else "required span")
    document_sha256 = _digest(values["document_sha256"], "document_sha256")
    if document_sha256 not in documents:
        raise ChunkingAuditError("span references an unknown document")
    start = _integer(
        values["start_byte"], "start_byte", minimum=0, maximum=documents[document_sha256].size_bytes
    )
    end = _integer(
        values["end_byte"], "end_byte", minimum=1, maximum=documents[document_sha256].size_bytes
    )
    if start >= end:
        raise ChunkingAuditError("span must be a non-empty half-open interval")
    if ranked:
        return RankedSpan(
            rank=_integer(values["rank"], "rank", minimum=1, maximum=1_000),
            document_sha256=document_sha256,
            start_byte=start,
            end_byte=end,
        )
    return Span(document_sha256=document_sha256, start_byte=start, end_byte=end)


def _variant(
    values: object,
    *,
    documents: dict[str, DocumentSpec],
    policy: ChunkingPolicy,
) -> VariantResult:
    if not isinstance(values, dict):
        raise ChunkingAuditError("variant must be an object")
    _expect_fields(values, {"configuration_sha256", "results"}, "variant")
    raw_results = values["results"]
    if (
        not isinstance(raw_results, list)
        or not 1 <= len(raw_results) <= policy.max_results_per_configuration
    ):
        raise ChunkingAuditError("variant results violate the configured result budget")
    results = tuple(
        sorted(
            (_span(item, documents=documents, ranked=True) for item in raw_results),
            key=lambda item: item.rank,
        )
    )
    if [item.rank for item in results] != list(range(1, len(results) + 1)):
        raise ChunkingAuditError("variant ranks must be unique and contiguous from one")
    identities = [(item.document_sha256, item.start_byte, item.end_byte) for item in results]
    if len(set(identities)) != len(identities):
        raise ChunkingAuditError("variant contains duplicate ranked spans")
    return VariantResult(
        configuration_sha256=_digest(values["configuration_sha256"], "configuration_sha256"),
        results=results,  # type: ignore[arg-type]
    )


def _query(
    values: object,
    *,
    documents: dict[str, DocumentSpec],
    configurations: tuple[str, ...],
    policy: ChunkingPolicy,
) -> QueryEvidence:
    if not isinstance(values, dict):
        raise ChunkingAuditError("query evidence must be an object")
    _expect_fields(values, {"query_id", "required_spans", "variants"}, "query evidence")
    raw_required = values["required_spans"]
    if not isinstance(raw_required, list) or not raw_required:
        raise ChunkingAuditError("each query must contain required spans")
    required = tuple(
        sorted(
            (_span(item, documents=documents, ranked=False) for item in raw_required),
            key=lambda item: (item.document_sha256, item.start_byte, item.end_byte),
        )
    )
    required_identities = [
        (item.document_sha256, item.start_byte, item.end_byte) for item in required
    ]
    if len(set(required_identities)) != len(required_identities):
        raise ChunkingAuditError("query contains duplicate required spans")
    raw_variants = values["variants"]
    if not isinstance(raw_variants, list) or len(raw_variants) != len(configurations):
        raise ChunkingAuditError("query must contain exactly one result for every configuration")
    variants = tuple(
        sorted(
            (_variant(item, documents=documents, policy=policy) for item in raw_variants),
            key=lambda item: item.configuration_sha256,
        )
    )
    observed = tuple(item.configuration_sha256 for item in variants)
    if observed != configurations:
        raise ChunkingAuditError("query configuration set does not match the artifact manifest")
    return QueryEvidence(
        query_id=_identifier(values["query_id"], "query_id"),
        required_spans=required,  # type: ignore[arg-type]
        variants=variants,
    )


def parse_artifact(values: object, *, policy: ChunkingPolicy) -> ChunkingArtifact:
    if not isinstance(values, dict):
        raise ChunkingAuditError("artifact must be an object")
    _expect_fields(
        values,
        {
            "schema_version",
            "corpus_sha256",
            "retriever_sha256",
            "baseline_configuration_sha256",
            "configurations",
            "documents",
            "queries",
        },
        "artifact",
    )
    if values["schema_version"] != 1:
        raise ChunkingAuditError("unsupported schema_version")

    raw_configurations = values["configurations"]
    if not isinstance(raw_configurations, list) or not (
        policy.min_configurations <= len(raw_configurations) <= policy.max_configurations
    ):
        raise ChunkingAuditError("artifact violates configuration-count policy")
    configurations = tuple(sorted(_digest(item, "configuration") for item in raw_configurations))
    if len(set(configurations)) != len(configurations):
        raise ChunkingAuditError("configuration digests must be unique")
    baseline = _digest(
        values["baseline_configuration_sha256"],
        "baseline_configuration_sha256",
    )
    if baseline not in configurations:
        raise ChunkingAuditError("baseline configuration is not in the manifest")

    raw_documents = values["documents"]
    if not isinstance(raw_documents, list) or not 1 <= len(raw_documents) <= policy.max_documents:
        raise ChunkingAuditError("artifact violates the document budget")
    documents = tuple(
        sorted((_document(item) for item in raw_documents), key=lambda item: item.document_sha256)
    )
    document_map = {item.document_sha256: item for item in documents}
    if len(document_map) != len(documents):
        raise ChunkingAuditError("document digests must be unique")

    raw_queries = values["queries"]
    if not isinstance(raw_queries, list) or not 1 <= len(raw_queries) <= policy.max_queries:
        raise ChunkingAuditError("artifact violates the query budget")
    queries = tuple(
        sorted(
            (
                _query(
                    item,
                    documents=document_map,
                    configurations=configurations,
                    policy=policy,
                )
                for item in raw_queries
            ),
            key=lambda item: item.query_id,
        )
    )
    if len({item.query_id for item in queries}) != len(queries):
        raise ChunkingAuditError("query IDs must be unique")
    total_spans = sum(
        len(query.required_spans) + sum(len(variant.results) for variant in query.variants)
        for query in queries
    )
    if total_spans > policy.max_total_spans:
        raise ChunkingAuditError("artifact violates the total span budget")
    return ChunkingArtifact(
        schema_version=1,
        corpus_sha256=_digest(values["corpus_sha256"], "corpus_sha256"),
        retriever_sha256=_digest(values["retriever_sha256"], "retriever_sha256"),
        baseline_configuration_sha256=baseline,
        configurations=configurations,
        documents=documents,
        queries=queries,
    )


def _merge(spans: Sequence[Span | RankedSpan]) -> dict[str, tuple[tuple[int, int], ...]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for span in spans:
        grouped[span.document_sha256].append((span.start_byte, span.end_byte))
    merged: dict[str, tuple[tuple[int, int], ...]] = {}
    for document, intervals in grouped.items():
        output: list[list[int]] = []
        for start, end in sorted(intervals):
            if output and start <= output[-1][1]:
                output[-1][1] = max(output[-1][1], end)
            else:
                output.append([start, end])
        merged[document] = tuple((start, end) for start, end in output)
    return merged


def _length(values: dict[str, tuple[tuple[int, int], ...]]) -> int:
    return sum(end - start for intervals in values.values() for start, end in intervals)


def _intersection_length(
    left: dict[str, tuple[tuple[int, int], ...]],
    right: dict[str, tuple[tuple[int, int], ...]],
) -> int:
    total = 0
    for document in left.keys() & right.keys():
        first = left[document]
        second = right[document]
        i = j = 0
        while i < len(first) and j < len(second):
            start = max(first[i][0], second[j][0])
            end = min(first[i][1], second[j][1])
            if start < end:
                total += end - start
            if first[i][1] <= second[j][1]:
                i += 1
            else:
                j += 1
    return total


def span_coverage_jaccard(
    left: Sequence[Span | RankedSpan], right: Sequence[Span | RankedSpan]
) -> float:
    left_merged = _merge(left)
    right_merged = _merge(right)
    intersection = _intersection_length(left_merged, right_merged)
    union = _length(left_merged) + _length(right_merged) - intersection
    return intersection / union if union else 1.0


def required_span_recall(required: Sequence[Span], retrieved: Sequence[RankedSpan]) -> float:
    required_merged = _merge(required)
    denominator = _length(required_merged)
    if not denominator:
        raise ChunkingAuditError("required spans must cover at least one byte")
    return _intersection_length(required_merged, _merge(retrieved)) / denominator


def document_jaccard(left: Sequence[RankedSpan], right: Sequence[RankedSpan]) -> float:
    left_documents = {item.document_sha256 for item in left}
    right_documents = {item.document_sha256 for item in right}
    union = left_documents | right_documents
    return len(left_documents & right_documents) / len(union) if union else 1.0


def rank_weighted_document_jaccard(
    left: Sequence[RankedSpan], right: Sequence[RankedSpan]
) -> float:
    def weights(values: Sequence[RankedSpan]) -> dict[str, float]:
        output: dict[str, float] = {}
        for item in values:
            output.setdefault(item.document_sha256, 1.0 / item.rank)
        return output

    left_weights = weights(left)
    right_weights = weights(right)
    documents = left_weights.keys() | right_weights.keys()
    numerator = sum(
        min(left_weights.get(item, 0.0), right_weights.get(item, 0.0)) for item in documents
    )
    denominator = sum(
        max(left_weights.get(item, 0.0), right_weights.get(item, 0.0)) for item in documents
    )
    return numerator / denominator if denominator else 1.0


def _artifact_document(artifact: ChunkingArtifact) -> dict[str, Any]:
    return {
        "baseline_configuration_sha256": artifact.baseline_configuration_sha256,
        "configurations": list(artifact.configurations),
        "corpus_sha256": artifact.corpus_sha256,
        "documents": [asdict(item) for item in artifact.documents],
        "queries": [
            {
                "query_id": query.query_id,
                "required_spans": [asdict(item) for item in query.required_spans],
                "variants": [
                    {
                        "configuration_sha256": variant.configuration_sha256,
                        "results": [asdict(item) for item in variant.results],
                    }
                    for variant in query.variants
                ],
            }
            for query in artifact.queries
        ],
        "retriever_sha256": artifact.retriever_sha256,
        "schema_version": artifact.schema_version,
    }


def audit_chunking_sensitivity(
    artifact: ChunkingArtifact,
    *,
    policy: ChunkingPolicy | None = None,
) -> ChunkingReport:
    active_policy = policy or ChunkingPolicy()
    artifact = parse_artifact(_artifact_document(artifact), policy=active_policy)
    comparisons: list[Comparison] = []
    findings: list[Finding] = []
    for query in artifact.queries:
        query_ref = _reference(query.query_id)
        variants = {item.configuration_sha256: item for item in query.variants}
        baseline = variants[artifact.baseline_configuration_sha256]
        baseline_recall = required_span_recall(query.required_spans, baseline.results)
        if baseline_recall < active_policy.min_required_span_recall:
            findings.append(
                Finding(
                    code="baseline_required_span_recall_below_minimum",
                    query_ref=query_ref,
                    configuration_sha256=artifact.baseline_configuration_sha256,
                    observed=round(baseline_recall, 6),
                    limit=active_policy.min_required_span_recall,
                )
            )
        for configuration in artifact.configurations:
            if configuration == artifact.baseline_configuration_sha256:
                continue
            variant = variants[configuration]
            variant_recall = required_span_recall(query.required_spans, variant.results)
            span_jaccard = span_coverage_jaccard(baseline.results, variant.results)
            doc_jaccard = document_jaccard(baseline.results, variant.results)
            rank_jaccard = rank_weighted_document_jaccard(baseline.results, variant.results)
            comparisons.append(
                Comparison(
                    query_ref=query_ref,
                    configuration_sha256=configuration,
                    baseline_required_span_recall=round(baseline_recall, 6),
                    variant_required_span_recall=round(variant_recall, 6),
                    span_coverage_jaccard=round(span_jaccard, 6),
                    document_jaccard=round(doc_jaccard, 6),
                    rank_weighted_document_jaccard=round(rank_jaccard, 6),
                )
            )
            checks = (
                (
                    "variant_required_span_recall_below_minimum",
                    variant_recall,
                    active_policy.min_required_span_recall,
                ),
                (
                    "span_coverage_jaccard_below_minimum",
                    span_jaccard,
                    active_policy.min_span_coverage_jaccard,
                ),
                (
                    "document_jaccard_below_minimum",
                    doc_jaccard,
                    active_policy.min_document_jaccard,
                ),
                (
                    "rank_weighted_document_jaccard_below_minimum",
                    rank_jaccard,
                    active_policy.min_rank_weighted_document_jaccard,
                ),
            )
            for code, observed, limit in checks:
                if observed < limit:
                    findings.append(
                        Finding(
                            code=code,
                            query_ref=query_ref,
                            configuration_sha256=configuration,
                            observed=round(observed, 6),
                            limit=limit,
                        )
                    )

    comparisons.sort(key=lambda item: (item.query_ref, item.configuration_sha256))
    findings.sort(key=lambda item: (item.code, item.query_ref, item.configuration_sha256))
    bounded_findings = tuple(findings[: active_policy.max_findings])
    artifact_sha256 = _sha256_json(_artifact_document(artifact))
    report_document = {
        "artifact_sha256": artifact_sha256,
        "baseline_configuration_sha256": artifact.baseline_configuration_sha256,
        "comparisons": [asdict(item) for item in comparisons],
        "configurations": len(artifact.configurations),
        "corpus_sha256": artifact.corpus_sha256,
        "documents": len(artifact.documents),
        "findings": [asdict(item) for item in bounded_findings],
        "findings_truncated": len(findings) > len(bounded_findings),
        "outcome": "accepted" if not findings else "rejected",
        "policy_sha256": active_policy.digest,
        "queries": len(artifact.queries),
        "retriever_sha256": artifact.retriever_sha256,
        "schema_version": 1,
    }
    return ChunkingReport(
        schema_version=1,
        outcome=report_document["outcome"],
        artifact_sha256=artifact_sha256,
        policy_sha256=active_policy.digest,
        corpus_sha256=artifact.corpus_sha256,
        retriever_sha256=artifact.retriever_sha256,
        baseline_configuration_sha256=artifact.baseline_configuration_sha256,
        configurations=len(artifact.configurations),
        documents=len(artifact.documents),
        queries=len(artifact.queries),
        comparisons=tuple(comparisons),
        findings=bounded_findings,
        findings_truncated=report_document["findings_truncated"],
        evidence_sha256=_sha256_json(report_document),
    )


def _reject_constant(value: str) -> None:
    raise ChunkingAuditError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ChunkingAuditError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _load_json(path: Path, *, max_bytes: int) -> Any:
    encoded = path.read_bytes()
    if len(encoded) > max_bytes:
        raise ChunkingAuditError(f"{path.name} exceeds the {max_bytes}-byte budget")
    return json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
    )


def _load_policy(path: Path) -> ChunkingPolicy:
    values = _load_json(path, max_bytes=MAX_POLICY_BYTES)
    if not isinstance(values, dict):
        raise ChunkingAuditError("policy must be an object")
    expected = set(ChunkingPolicy.__dataclass_fields__)
    if set(values) - expected:
        raise ChunkingAuditError("policy contains unknown fields")
    try:
        return ChunkingPolicy(**values)
    except TypeError as exc:
        raise ChunkingAuditError("policy fields have invalid types") from exc


def _atomic_write(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(values, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit RAG retrieval sensitivity to chunk-boundary changes."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        policy = _load_policy(args.policy) if args.policy else ChunkingPolicy()
        artifact = parse_artifact(
            _load_json(args.artifact, max_bytes=MAX_ARTIFACT_BYTES),
            policy=policy,
        )
        report = audit_chunking_sensitivity(artifact, policy=policy)
        document = report.to_dict()
        if args.output:
            _atomic_write(args.output, document)
        print(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, UnicodeError, json.JSONDecodeError, ChunkingAuditError) as exc:
        print(
            json.dumps(
                {"outcome": "malformed", "reason": type(exc).__name__},
                sort_keys=True,
            )
        )
        return EXIT_MALFORMED
    return EXIT_ACCEPTED if report.outcome == "accepted" else EXIT_REJECTED


if __name__ == "__main__":
    raise SystemExit(main())
