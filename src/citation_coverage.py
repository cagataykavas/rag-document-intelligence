from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_KEYS = {
    "schema_version",
    "generated_at",
    "response_kind",
    "query_id",
    "model_id",
    "prompt_sha256",
    "retrieval_snapshot_sha256",
    "answer",
    "retrieved_chunk_ids",
    "claims",
}
CLAIM_KEYS = {
    "claim_id",
    "start_char",
    "end_char",
    "text",
    "claim_type",
    "citation_ids",
}
CLAIM_TYPES = {
    "factual",
    "quantitative",
    "recommendation",
    "attribution",
    "opinion",
    "disclaimer",
}
MATERIAL_TYPES = {"factual", "quantitative", "recommendation", "attribution"}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
TOKEN = re.compile(r"[\w%+-]+", re.UNICODE)
STRUCTURED_LITERAL = re.compile(
    r"(?:\b\d+(?:[.,]\d+)?%?\b|https?://|\b[A-Fa-f0-9]{8}-[A-Fa-f0-9-]{27,}\b|"
    r"\b[A-Z]{2,10}-\d{1,12}\b)"
)


class CoverageArtifactError(ValueError):
    """Malformed artifact error with a stable, non-sensitive code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CitationCoveragePolicy:
    max_artifact_bytes: int = 262_144
    max_answer_chars: int = 32_768
    max_claims: int = 512
    max_retrieved_chunks: int = 2_048
    max_citations_per_claim: int = 16
    min_content_token_coverage: float = 1.0
    min_material_claim_coverage: float = 1.0
    max_age_seconds: float = 86_400.0
    max_future_skew_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("max_artifact_bytes", self.max_artifact_bytes, 1_024, 4_000_000),
            ("max_answer_chars", self.max_answer_chars, 1, 250_000),
            ("max_claims", self.max_claims, 1, 10_000),
            ("max_retrieved_chunks", self.max_retrieved_chunks, 1, 100_000),
            ("max_citations_per_claim", self.max_citations_per_claim, 1, 1_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be between {lower} and {upper}")
        for name, value, lower, upper in (
            ("min_content_token_coverage", self.min_content_token_coverage, 0.0, 1.0),
            ("min_material_claim_coverage", self.min_material_claim_coverage, 0.0, 1.0),
            ("max_age_seconds", self.max_age_seconds, 1.0, 31_536_000.0),
            ("max_future_skew_seconds", self.max_future_skew_seconds, 0.0, 86_400.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} must be between {lower} and {upper}")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CitationCoverageReport:
    schema_version: int
    accepted: bool
    reason_codes: tuple[str, ...]
    response_kind: str
    claim_count: int
    material_claim_count: int
    cited_material_claim_count: int
    structured_literal_claim_count: int
    cited_structured_literal_claim_count: int
    answer_content_token_count: int
    mapped_content_token_count: int
    content_token_coverage: float
    material_claim_coverage: float
    invalid_citation_count: int
    answer_sha256: str
    artifact_sha256: str
    policy_sha256: str
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        body = asdict(self)
        body.pop("evidence_sha256")
        body["reason_codes"] = list(self.reason_codes)
        return body

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CoverageArtifactError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def load_artifact(path: str | Path, policy: CitationCoveragePolicy) -> dict[str, Any]:
    artifact_path = Path(path)
    try:
        raw_bytes = artifact_path.read_bytes()
    except OSError as exc:
        raise CoverageArtifactError("ARTIFACT_UNREADABLE") from exc
    if len(raw_bytes) > policy.max_artifact_bytes:
        raise CoverageArtifactError("ARTIFACT_BYTE_BUDGET_EXCEEDED")
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CoverageArtifactError("ARTIFACT_NOT_UTF8") from exc
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                CoverageArtifactError("NON_FINITE_JSON_NUMBER")
            ),
        )
    except CoverageArtifactError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise CoverageArtifactError("INVALID_JSON") from exc
    if not isinstance(payload, dict):
        raise CoverageArtifactError("ARTIFACT_NOT_OBJECT")
    return payload


def _exact_keys(value: dict[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise CoverageArtifactError(code)


def _safe_id(value: object, code: str) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise CoverageArtifactError(code)
    return value


def _sha(value: object, code: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise CoverageArtifactError(code)
    return value


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise CoverageArtifactError("INVALID_GENERATED_AT")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CoverageArtifactError("INVALID_GENERATED_AT") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CoverageArtifactError("NAIVE_GENERATED_AT")
    return parsed.astimezone(timezone.utc)


def _validate_artifact(payload: dict[str, Any], policy: CitationCoveragePolicy) -> None:
    try:
        canonical_size = len(
            json.dumps(
                payload,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise CoverageArtifactError("NON_CANONICAL_JSON_VALUE") from exc
    if canonical_size > policy.max_artifact_bytes:
        raise CoverageArtifactError("ARTIFACT_BYTE_BUDGET_EXCEEDED")
    _exact_keys(payload, SCHEMA_KEYS, "ARTIFACT_SCHEMA_MISMATCH")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise CoverageArtifactError("UNSUPPORTED_SCHEMA_VERSION")
    if payload["response_kind"] not in {"answer", "abstention"}:
        raise CoverageArtifactError("INVALID_RESPONSE_KIND")
    _safe_id(payload["query_id"], "INVALID_QUERY_ID")
    _safe_id(payload["model_id"], "INVALID_MODEL_ID")
    _sha(payload["prompt_sha256"], "INVALID_PROMPT_DIGEST")
    _sha(payload["retrieval_snapshot_sha256"], "INVALID_RETRIEVAL_DIGEST")
    _timestamp(payload["generated_at"])
    answer = payload["answer"]
    if not isinstance(answer, str) or not answer.strip() or len(answer) > policy.max_answer_chars:
        raise CoverageArtifactError("INVALID_ANSWER")
    if any(ord(character) < 32 and character not in "\n\r\t" for character in answer):
        raise CoverageArtifactError("ANSWER_CONTROL_CHARACTER")

    retrieved = payload["retrieved_chunk_ids"]
    if not isinstance(retrieved, list) or len(retrieved) > policy.max_retrieved_chunks:
        raise CoverageArtifactError("INVALID_RETRIEVAL_MANIFEST")
    retrieved_ids = [_safe_id(value, "INVALID_RETRIEVED_CHUNK_ID") for value in retrieved]
    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise CoverageArtifactError("DUPLICATE_RETRIEVED_CHUNK_ID")

    claims = payload["claims"]
    if not isinstance(claims, list) or not claims or len(claims) > policy.max_claims:
        raise CoverageArtifactError("INVALID_CLAIM_LIST")
    seen_ids: set[str] = set()
    previous_end = 0
    for claim in claims:
        if not isinstance(claim, dict):
            raise CoverageArtifactError("CLAIM_NOT_OBJECT")
        _exact_keys(claim, CLAIM_KEYS, "CLAIM_SCHEMA_MISMATCH")
        claim_id = _safe_id(claim["claim_id"], "INVALID_CLAIM_ID")
        if claim_id in seen_ids:
            raise CoverageArtifactError("DUPLICATE_CLAIM_ID")
        seen_ids.add(claim_id)
        start, end = claim["start_char"], claim["end_char"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < previous_end
            or start < 0
            or end <= start
            or end > len(answer)
        ):
            raise CoverageArtifactError("INVALID_OR_OVERLAPPING_CLAIM_SPAN")
        previous_end = end
        text = claim["text"]
        if not isinstance(text, str) or text != answer[start:end] or not text.strip():
            raise CoverageArtifactError("CLAIM_TEXT_SPAN_MISMATCH")
        if claim["claim_type"] not in CLAIM_TYPES:
            raise CoverageArtifactError("INVALID_CLAIM_TYPE")
        citations = claim["citation_ids"]
        if not isinstance(citations, list) or len(citations) > policy.max_citations_per_claim:
            raise CoverageArtifactError("INVALID_CITATION_LIST")
        citation_ids = [_safe_id(item, "INVALID_CITATION_ID") for item in citations]
        if len(citation_ids) != len(set(citation_ids)):
            raise CoverageArtifactError("DUPLICATE_CITATION_ID")


def audit_citation_coverage(
    payload: dict[str, Any],
    policy: CitationCoveragePolicy | None = None,
    *,
    now: datetime | None = None,
) -> CitationCoverageReport:
    """Audit whether every material answer claim is mapped to retrieved evidence.

    The audit checks coverage and provenance contracts. It does not decide whether
    a cited chunk entails a claim; that is a separate evaluation boundary.
    """
    selected = policy or CitationCoveragePolicy()
    _validate_artifact(payload, selected)
    answer: str = payload["answer"]
    claims: list[dict[str, Any]] = payload["claims"]
    retrieved = set(payload["retrieved_chunk_ids"])
    reasons: set[str] = set()

    observed_at = now or datetime.now(timezone.utc)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    generated_at = _timestamp(payload["generated_at"])
    age = (observed_at.astimezone(timezone.utc) - generated_at).total_seconds()
    if age > selected.max_age_seconds:
        reasons.add("ARTIFACT_STALE")
    if age < -selected.max_future_skew_seconds:
        reasons.add("ARTIFACT_FROM_FUTURE")

    material_count = 0
    cited_material_count = 0
    structured_count = 0
    cited_structured_count = 0
    invalid_citations = 0
    spans: list[tuple[int, int]] = []
    for claim in claims:
        citations = tuple(claim["citation_ids"])
        valid_citations = tuple(item for item in citations if item in retrieved)
        invalid_citations += len(citations) - len(valid_citations)
        if len(valid_citations) != len(citations):
            reasons.add("CITATION_OUTSIDE_RETRIEVAL")

        is_material = claim["claim_type"] in MATERIAL_TYPES
        has_structured_literal = bool(STRUCTURED_LITERAL.search(claim["text"]))
        if is_material:
            material_count += 1
            if valid_citations:
                cited_material_count += 1
            else:
                reasons.add("MATERIAL_CLAIM_UNCITED")
        if has_structured_literal:
            structured_count += 1
            if valid_citations:
                cited_structured_count += 1
            else:
                reasons.add("STRUCTURED_LITERAL_UNCITED")
        spans.append((claim["start_char"], claim["end_char"]))

    token_matches = tuple(TOKEN.finditer(answer))
    mapped_tokens = sum(
        any(start <= token.start() and token.end() <= end for start, end in spans)
        for token in token_matches
    )
    token_coverage = mapped_tokens / len(token_matches) if token_matches else 1.0
    material_coverage = cited_material_count / material_count if material_count else 1.0
    if token_coverage < selected.min_content_token_coverage:
        reasons.add("CONTENT_COVERAGE_BELOW_THRESHOLD")
    if material_coverage < selected.min_material_claim_coverage:
        reasons.add("MATERIAL_CITATION_COVERAGE_BELOW_THRESHOLD")
    if payload["response_kind"] == "answer" and material_count == 0:
        reasons.add("ANSWER_HAS_NO_MATERIAL_CLAIMS")
    if payload["response_kind"] == "abstention" and material_count:
        reasons.add("ABSTENTION_CONTAINS_MATERIAL_CLAIMS")

    body: dict[str, object] = {
        "schema_version": 1,
        "accepted": not reasons,
        "reason_codes": tuple(sorted(reasons)),
        "response_kind": payload["response_kind"],
        "claim_count": len(claims),
        "material_claim_count": material_count,
        "cited_material_claim_count": cited_material_count,
        "structured_literal_claim_count": structured_count,
        "cited_structured_literal_claim_count": cited_structured_count,
        "answer_content_token_count": len(token_matches),
        "mapped_content_token_count": mapped_tokens,
        "content_token_coverage": token_coverage,
        "material_claim_coverage": material_coverage,
        "invalid_citation_count": invalid_citations,
        "answer_sha256": _sha256_text(answer),
        "artifact_sha256": _digest(payload),
        "policy_sha256": _digest(selected.as_dict()),
    }
    report = CitationCoverageReport(**body, evidence_sha256="")  # type: ignore[arg-type]
    return replace(report, evidence_sha256=_digest(report.body()))


def _error_report(code: str) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": 1,
        "accepted": False,
        "reason_codes": [code],
        "status": "malformed",
    }
    return {**body, "evidence_sha256": _digest(body)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit claim-level citation completeness")
    parser.add_argument("artifact", help="Path to the citation coverage artifact")
    parser.add_argument("--now", help="Timezone-aware ISO-8601 evaluation time")
    parser.add_argument("--min-content-coverage", type=float, default=1.0)
    parser.add_argument("--min-material-coverage", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        policy = CitationCoveragePolicy(
            min_content_token_coverage=args.min_content_coverage,
            min_material_claim_coverage=args.min_material_coverage,
        )
        now = _timestamp(args.now) if args.now else None
        payload = load_artifact(args.artifact, policy)
        report = audit_citation_coverage(payload, policy, now=now)
    except (CoverageArtifactError, ValueError) as exc:
        code = exc.code if isinstance(exc, CoverageArtifactError) else "INVALID_POLICY"
        print(json.dumps(_error_report(code), sort_keys=True, separators=(",", ":")))
        return 3
    print(json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":")))
    return 0 if report.accepted else 2


if __name__ == "__main__":
    sys.exit(main())
