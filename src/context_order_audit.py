"""Audit RAG answers for sensitivity to retrieved-context ordering."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ArtifactError(ValueError):
    """The benchmark artifact is malformed or exceeds a resource budget."""


@dataclass(frozen=True)
class AuditPolicy:
    min_queries: int = 1
    min_variants_per_query: int = 4
    min_unique_orders: int = 4
    min_max_order_distance: float = 0.50
    min_citation_jaccard: float = 0.80
    min_answer_token_jaccard: float = 0.70
    max_abstention_flips: int = 0
    max_artifact_age_seconds: float = 604_800.0
    max_future_skew_seconds: float = 30.0
    max_artifact_bytes: int = 1_048_576
    max_queries: int = 500
    max_variants_per_query: int = 32
    max_chunks_per_query: int = 64
    max_answer_bytes: int = 16_384

    def __post_init__(self) -> None:
        positive_ints = (
            "min_queries",
            "min_variants_per_query",
            "min_unique_orders",
            "max_artifact_bytes",
            "max_queries",
            "max_variants_per_query",
            "max_chunks_per_query",
            "max_answer_bytes",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_abstention_flips, bool)
            or not isinstance(self.max_abstention_flips, int)
            or self.max_abstention_flips < 0
        ):
            raise ValueError("max_abstention_flips must be a non-negative integer")
        for name in (
            "min_max_order_distance",
            "min_citation_jaccard",
            "min_answer_token_jaccard",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        for name in ("max_artifact_age_seconds", "max_future_skew_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.min_queries > self.max_queries:
            raise ValueError("min_queries must not exceed max_queries")
        if self.min_variants_per_query > self.max_variants_per_query:
            raise ValueError("min_variants_per_query must not exceed its maximum")
        if self.min_unique_orders > self.max_variants_per_query:
            raise ValueError("min_unique_orders must not exceed the variant maximum")
        if self.max_chunks_per_query < 2:
            raise ValueError("max_chunks_per_query must allow at least two chunks")


@dataclass(frozen=True)
class QueryOrderAudit:
    query_digest: str
    variant_count: int
    unique_order_count: int
    max_normalized_order_distance: float
    min_citation_jaccard: float
    min_answer_token_jaccard: float
    abstention_flips: int
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContextOrderAuditReport:
    accepted: bool
    schema_version: int
    artifact_digest: str
    benchmark_digest: str
    policy_digest: str
    query_count: int
    finding_codes: tuple[str, ...]
    queries: tuple[QueryOrderAudit, ...]
    evidence_digest: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_QUERY_REASON_ORDER = (
    "TOO_FEW_VARIANTS",
    "INSUFFICIENT_ORDER_DIVERSITY",
    "INSUFFICIENT_ORDER_PERTURBATION",
    "CITATION_SET_DRIFT",
    "ANSWER_LEXICAL_DRIFT",
    "ABSTENTION_DECISION_FLIP",
)
_FINDING_ORDER = (
    "TOO_FEW_QUERIES",
    "ARTIFACT_FROM_FUTURE",
    "ARTIFACT_STALE",
    *_QUERY_REASON_ORDER,
)
_FINDING_RANK = {code: index for index, code in enumerate(_FINDING_ORDER)}


def _identifier(value: object, path: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 128:
        raise ArtifactError(f"{path} must be a non-empty string of at most 128 bytes")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ArtifactError(f"{path} contains a control character")
    return value


def _digest_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactError("artifact is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _exact_keys(value: object, expected: set[str], path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactError(f"{path} must be an object")
    actual = set(value)
    if actual != expected:
        raise ArtifactError(f"{path} fields do not match the schema")
    return value


def _string_list(value: object, path: str, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise ArtifactError(f"{path} must contain between 1 and {maximum} identifiers")
    items = tuple(_identifier(item, f"{path}[]") for item in value)
    if len(set(items)) != len(items):
        raise ArtifactError(f"{path} contains duplicate identifiers")
    return items


def _parse_timestamp(value: object, path: str) -> datetime:
    if not isinstance(value, str):
        raise ArtifactError(f"{path} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ArtifactError(f"{path} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactError(f"{path} must include a timezone")
    return parsed.astimezone(UTC)


def _tokens(answer: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(answer)}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _normalized_kendall_distance(baseline: tuple[str, ...], candidate: tuple[str, ...]) -> float:
    if len(baseline) < 2:
        return 0.0
    positions = {item: index for index, item in enumerate(candidate)}
    inversions = 0
    pairs = 0
    for left_index, left in enumerate(baseline[:-1]):
        for right in baseline[left_index + 1 :]:
            pairs += 1
            if positions[left] > positions[right]:
                inversions += 1
    return inversions / pairs


def _variant(
    raw: object,
    *,
    path: str,
    expected_chunks: tuple[str, ...],
    policy: AuditPolicy,
) -> dict[str, object]:
    value = _exact_keys(
        raw,
        {
            "variant_id",
            "baseline",
            "ordered_chunk_ids",
            "cited_chunk_ids",
            "answer",
            "insufficient_evidence",
        },
        path,
    )
    variant_id = _identifier(value["variant_id"], f"{path}.variant_id")
    baseline = value["baseline"]
    insufficient = value["insufficient_evidence"]
    if not isinstance(baseline, bool) or not isinstance(insufficient, bool):
        raise ArtifactError(f"{path} boolean fields must be booleans")
    ordered = _string_list(
        value["ordered_chunk_ids"], f"{path}.ordered_chunk_ids", policy.max_chunks_per_query
    )
    if set(ordered) != set(expected_chunks) or len(ordered) != len(expected_chunks):
        raise ArtifactError(f"{path}.ordered_chunk_ids must be a permutation of chunk_ids")
    cited_raw = value["cited_chunk_ids"]
    if not isinstance(cited_raw, list) or len(cited_raw) > policy.max_chunks_per_query:
        raise ArtifactError(f"{path}.cited_chunk_ids must be a bounded list")
    cited = tuple(_identifier(item, f"{path}.cited_chunk_ids[]") for item in cited_raw)
    if len(set(cited)) != len(cited) or not set(cited).issubset(expected_chunks):
        raise ArtifactError(f"{path}.cited_chunk_ids must be unique retrieved chunk IDs")
    answer = value["answer"]
    if not isinstance(answer, str) or len(answer.encode("utf-8")) > policy.max_answer_bytes:
        raise ArtifactError(f"{path}.answer exceeds the text budget")
    if insufficient:
        if cited:
            raise ArtifactError(f"{path} abstention must not cite evidence")
    elif not answer.strip() or not cited:
        raise ArtifactError(f"{path} non-abstention requires an answer and citation")
    return {
        "variant_id": variant_id,
        "baseline": baseline,
        "ordered_chunk_ids": ordered,
        "cited_chunk_ids": cited,
        "answer_tokens": _tokens(answer),
        "insufficient_evidence": insufficient,
    }


def audit_context_order(
    artifact: object,
    *,
    policy: AuditPolicy | None = None,
    observed_at: datetime | None = None,
) -> ContextOrderAuditReport:
    """Validate and audit a context-order metamorphic benchmark artifact."""
    policy = policy or AuditPolicy()
    observed_at = observed_at or datetime.now(UTC)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ArtifactError("observed_at must include a timezone")
    observed = observed_at.astimezone(UTC)

    root = _exact_keys(
        artifact,
        {
            "schema_version",
            "benchmark_id",
            "model_id",
            "prompt_template_id",
            "retrieval_snapshot_digest",
            "generation_config_digest",
            "generated_at",
            "queries",
        },
        "artifact",
    )
    if root["schema_version"] != 1:
        raise ArtifactError("schema_version must be 1")
    benchmark_id = _identifier(root["benchmark_id"], "benchmark_id")
    _identifier(root["model_id"], "model_id")
    _identifier(root["prompt_template_id"], "prompt_template_id")
    for field in ("retrieval_snapshot_digest", "generation_config_digest"):
        value = root[field]
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ArtifactError(f"{field} must be a lowercase SHA-256 digest")
    generated_at = _parse_timestamp(root["generated_at"], "generated_at")
    queries_raw = root["queries"]
    if not isinstance(queries_raw, list) or len(queries_raw) > policy.max_queries:
        raise ArtifactError("queries must be a bounded list")

    query_ids: set[str] = set()
    query_reports: list[QueryOrderAudit] = []
    all_findings: set[str] = set()

    for query_index, query_raw in enumerate(queries_raw):
        path = f"queries[{query_index}]"
        query = _exact_keys(query_raw, {"query_id", "chunk_ids", "variants"}, path)
        query_id = _identifier(query["query_id"], f"{path}.query_id")
        if query_id in query_ids:
            raise ArtifactError("query_id values must be unique")
        query_ids.add(query_id)
        chunks = _string_list(query["chunk_ids"], f"{path}.chunk_ids", policy.max_chunks_per_query)
        if len(chunks) < 2:
            raise ArtifactError(f"{path}.chunk_ids needs at least two chunks for an order audit")
        variants_raw = query["variants"]
        if not isinstance(variants_raw, list) or len(variants_raw) > policy.max_variants_per_query:
            raise ArtifactError(f"{path}.variants must be a bounded list")
        variants = [
            _variant(
                raw,
                path=f"{path}.variants[{index}]",
                expected_chunks=chunks,
                policy=policy,
            )
            for index, raw in enumerate(variants_raw)
        ]
        variant_ids = [str(item["variant_id"]) for item in variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ArtifactError(f"{path}.variant_id values must be unique")
        baselines = [item for item in variants if item["baseline"]]
        if len(baselines) != 1:
            raise ArtifactError(f"{path} must contain exactly one baseline variant")
        baseline = baselines[0]
        baseline_order = baseline["ordered_chunk_ids"]
        assert isinstance(baseline_order, tuple)
        baseline_citations = set(baseline["cited_chunk_ids"])
        baseline_tokens = baseline["answer_tokens"]
        assert isinstance(baseline_tokens, set)

        orders = {item["ordered_chunk_ids"] for item in variants}
        distances = [
            _normalized_kendall_distance(baseline_order, item["ordered_chunk_ids"])
            for item in variants
        ]
        citation_scores = [
            _jaccard(baseline_citations, set(item["cited_chunk_ids"]))
            for item in variants
            if item is not baseline
        ]
        answer_scores = [
            _jaccard(baseline_tokens, item["answer_tokens"])
            for item in variants
            if item is not baseline
        ]
        abstention_flips = sum(
            item["insufficient_evidence"] != baseline["insufficient_evidence"]
            for item in variants
            if item is not baseline
        )
        min_citations = min(citation_scores, default=1.0)
        min_answers = min(answer_scores, default=1.0)
        max_distance = max(distances, default=0.0)

        reasons: set[str] = set()
        if len(variants) < policy.min_variants_per_query:
            reasons.add("TOO_FEW_VARIANTS")
        if len(orders) < policy.min_unique_orders:
            reasons.add("INSUFFICIENT_ORDER_DIVERSITY")
        if max_distance < policy.min_max_order_distance:
            reasons.add("INSUFFICIENT_ORDER_PERTURBATION")
        if min_citations < policy.min_citation_jaccard:
            reasons.add("CITATION_SET_DRIFT")
        if min_answers < policy.min_answer_token_jaccard:
            reasons.add("ANSWER_LEXICAL_DRIFT")
        if abstention_flips > policy.max_abstention_flips:
            reasons.add("ABSTENTION_DECISION_FLIP")
        all_findings.update(reasons)
        reason_codes = tuple(sorted(reasons, key=_FINDING_RANK.__getitem__))
        query_reports.append(
            QueryOrderAudit(
                query_digest=_digest_identifier(query_id),
                variant_count=len(variants),
                unique_order_count=len(orders),
                max_normalized_order_distance=round(max_distance, 12),
                min_citation_jaccard=round(min_citations, 12),
                min_answer_token_jaccard=round(min_answers, 12),
                abstention_flips=abstention_flips,
                reason_codes=reason_codes,
            )
        )

    if len(queries_raw) < policy.min_queries:
        all_findings.add("TOO_FEW_QUERIES")
    age_seconds = (observed - generated_at).total_seconds()
    if age_seconds < -policy.max_future_skew_seconds:
        all_findings.add("ARTIFACT_FROM_FUTURE")
    if age_seconds > policy.max_artifact_age_seconds:
        all_findings.add("ARTIFACT_STALE")

    artifact_digest = _canonical_digest(root)
    policy_digest = _canonical_digest(asdict(policy))
    finding_codes = tuple(sorted(all_findings, key=_FINDING_RANK.__getitem__))
    report_payload = {
        "accepted": not finding_codes,
        "schema_version": 1,
        "artifact_digest": artifact_digest,
        "benchmark_digest": _digest_identifier(benchmark_id),
        "policy_digest": policy_digest,
        "query_count": len(query_reports),
        "finding_codes": finding_codes,
        "queries": [item.to_dict() for item in query_reports],
    }
    return ContextOrderAuditReport(
        accepted=not finding_codes,
        schema_version=1,
        artifact_digest=artifact_digest,
        benchmark_digest=_digest_identifier(benchmark_id),
        policy_digest=policy_digest,
        query_count=len(query_reports),
        finding_codes=finding_codes,
        queries=tuple(query_reports),
        evidence_digest=_canonical_digest(report_payload),
    )


def load_artifact(path: str | Path, policy: AuditPolicy | None = None) -> object:
    policy = policy or AuditPolicy()
    raw = Path(path).read_bytes()
    if len(raw) > policy.max_artifact_bytes:
        raise ArtifactError("artifact byte budget exceeded")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ArtifactError("artifact contains a duplicate JSON key")
            value[key] = item
        return value

    try:
        return json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactError(f"invalid JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("artifact must be valid UTF-8 JSON") from exc


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", help="context-order benchmark JSON")
    parser.add_argument("--output", type=Path, help="atomically write the JSON report")
    args = parser.parse_args(argv)
    try:
        report = audit_context_order(load_artifact(args.artifact))
    except (ArtifactError, OSError):
        print('{"accepted":false,"error":"malformed_artifact"}')
        return 2
    payload = report.to_dict()
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    if args.output:
        _write_report(args.output, payload)
    print(rendered)
    return 0 if report.accepted else 3


if __name__ == "__main__":
    raise SystemExit(main())
