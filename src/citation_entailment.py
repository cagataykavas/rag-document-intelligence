"""Fail-closed semantic support audit for claim-level RAG citations."""

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
from typing import Any, NoReturn

SCHEMA_VERSION = "rag-citation-entailment/v1"
REPORT_VERSION = "rag-citation-entailment-report/v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "benchmark_id",
        "model_id",
        "prompt_template_id",
        "retrieval_snapshot_sha256",
        "evaluator_model_id",
        "evaluator_revision",
        "created_at",
        "claims",
    }
)
_CLAIM_FIELDS = frozenset({"claim_id", "claim_sha256", "risk", "citations", "verdicts"})
_VERDICT_FIELDS = frozenset(
    {
        "citation_id",
        "evidence_sha256",
        "entailment_probability",
        "neutral_probability",
        "contradiction_probability",
    }
)
_RISKS = frozenset({"standard", "critical"})
_MAX_FINDINGS = 256


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _integer(name: str, value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        _invalid(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(name: str, value: object, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        _invalid(f"{name} must be finite and in [{minimum}, {maximum}]")
    return number


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        _invalid(f"{name} must be a bounded identifier")
    return value


def _sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        _invalid(f"{name} must be a canonical lowercase SHA-256 digest")
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _timestamp(name: str, value: object) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        _invalid(f"{name} must be a bounded RFC 3339 timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC 3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _invalid(f"{name} must include a timezone offset")
    return parsed.astimezone(UTC)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _invalid("JSON objects must not contain duplicate fields")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    _invalid(f"non-finite JSON constant is not allowed: {value}")


@dataclass(frozen=True)
class EntailmentPolicy:
    min_claim_entailment: float = 0.70
    min_citation_entailment: float = 0.50
    max_citation_contradiction: float = 0.10
    min_claim_support_rate: float = 0.95
    min_citation_support_rate: float = 0.80
    max_contradicted_claim_rate: float = 0.0
    require_all_critical_supported: bool = True
    require_no_critical_contradiction: bool = True
    max_artifact_age_seconds: float = 86_400.0
    max_future_skew_seconds: float = 300.0
    probability_sum_tolerance: float = 1e-6
    max_claims: int = 1_000
    max_citations_per_claim: int = 16
    max_total_verdicts: int = 5_000
    max_input_bytes: int = 1_048_576

    def validate(self) -> None:
        for name in (
            "min_claim_entailment",
            "min_citation_entailment",
            "max_citation_contradiction",
            "min_claim_support_rate",
            "min_citation_support_rate",
            "max_contradicted_claim_rate",
        ):
            _number(name, getattr(self, name), 0.0, 1.0)
        _number("max_artifact_age_seconds", self.max_artifact_age_seconds, 0.001, 31_536_000.0)
        _number("max_future_skew_seconds", self.max_future_skew_seconds, 0.0, 86_400.0)
        _number("probability_sum_tolerance", self.probability_sum_tolerance, 0.0, 0.01)
        _integer("max_claims", self.max_claims, 1, 100_000)
        _integer("max_citations_per_claim", self.max_citations_per_claim, 1, 10_000)
        _integer("max_total_verdicts", self.max_total_verdicts, 1, 1_000_000)
        _integer("max_input_bytes", self.max_input_bytes, 1, 16_777_216)
        if not isinstance(self.require_all_critical_supported, bool) or not isinstance(
            self.require_no_critical_contradiction, bool
        ):
            _invalid("critical-claim policy flags must be booleans")


@dataclass(frozen=True)
class Finding:
    code: str
    evidence: dict[str, object]
    blocking: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": "error" if self.blocking else "diagnostic",
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class EntailmentReport:
    accepted: bool
    artifact_sha256: str
    benchmark_sha256: str
    configuration_sha256: str
    policy_sha256: str
    evaluated_at: str
    metrics: dict[str, int | float]
    findings: tuple[Finding, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_VERSION,
            "accepted": self.accepted,
            "artifact_sha256": self.artifact_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "configuration_sha256": self.configuration_sha256,
            "policy_sha256": self.policy_sha256,
            "evaluated_at": self.evaluated_at,
            "metrics": self.metrics,
            "findings": [finding.as_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class _Verdict:
    citation_id: str
    evidence_sha256: str
    entailment: float
    neutral: float
    contradiction: float


@dataclass(frozen=True)
class _Claim:
    claim_id: str
    claim_sha256: str
    risk: str
    citations: tuple[str, ...]
    verdicts: tuple[_Verdict, ...]


def load_artifact(payload: bytes, *, max_bytes: int = 1_048_576) -> dict[str, Any]:
    """Load strict, bounded JSON evidence."""
    limit = _integer("max_bytes", max_bytes, 1, 16_777_216)
    if not payload or len(payload) > limit:
        _invalid("artifact byte size is outside the configured budget")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        _invalid("artifact root must be an object")
    return value


def _parse_verdict(value: object, tolerance: float) -> _Verdict:
    if not isinstance(value, dict) or set(value) != _VERDICT_FIELDS:
        _invalid("each verdict must contain exactly the documented fields")
    entailment = _number("entailment_probability", value["entailment_probability"], 0.0, 1.0)
    neutral = _number("neutral_probability", value["neutral_probability"], 0.0, 1.0)
    contradiction = _number(
        "contradiction_probability", value["contradiction_probability"], 0.0, 1.0
    )
    if abs(entailment + neutral + contradiction - 1.0) > tolerance:
        _invalid("verdict probabilities must sum to one within tolerance")
    return _Verdict(
        citation_id=_identifier("citation_id", value["citation_id"]),
        evidence_sha256=_sha256("evidence_sha256", value["evidence_sha256"]),
        entailment=entailment,
        neutral=neutral,
        contradiction=contradiction,
    )


def _parse_claim(value: object, policy: EntailmentPolicy) -> _Claim:
    if not isinstance(value, dict) or set(value) != _CLAIM_FIELDS:
        _invalid("each claim must contain exactly the documented fields")
    citations_value = value["citations"]
    verdicts_value = value["verdicts"]
    if not isinstance(citations_value, list) or not citations_value:
        _invalid("each claim requires at least one citation")
    if len(citations_value) > policy.max_citations_per_claim:
        _invalid("claim exceeds max_citations_per_claim")
    citations = tuple(_identifier("citation_id", item) for item in citations_value)
    if len(set(citations)) != len(citations):
        _invalid("claim citations must be unique")
    if not isinstance(verdicts_value, list) or not verdicts_value:
        _invalid("each claim requires citation verdicts")
    verdicts = tuple(
        _parse_verdict(item, policy.probability_sum_tolerance) for item in verdicts_value
    )
    verdict_ids = [item.citation_id for item in verdicts]
    if len(set(verdict_ids)) != len(verdict_ids):
        _invalid("citation verdicts must be unique")
    if set(verdict_ids) != set(citations):
        _invalid("citation verdicts must exactly cover declared citations")
    risk = value["risk"]
    if risk not in _RISKS:
        _invalid("risk must be standard or critical")
    return _Claim(
        claim_id=_identifier("claim_id", value["claim_id"]),
        claim_sha256=_sha256("claim_sha256", value["claim_sha256"]),
        risk=risk,
        citations=citations,
        verdicts=verdicts,
    )


def audit_entailment(
    document: dict[str, Any],
    *,
    evaluated_at: datetime,
    policy: EntailmentPolicy | None = None,
) -> EntailmentReport:
    """Audit claim support and citation correctness from calibrated NLI evidence."""
    active_policy = policy or EntailmentPolicy()
    active_policy.validate()
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        _invalid("evaluated_at must be timezone-aware")
    now = evaluated_at.astimezone(UTC)
    if not isinstance(document, dict) or set(document) != _DOCUMENT_FIELDS:
        _invalid("artifact must contain exactly the documented fields")
    if document["schema_version"] != SCHEMA_VERSION:
        _invalid("unsupported artifact schema_version")

    benchmark_id = _identifier("benchmark_id", document["benchmark_id"])
    model_id = _identifier("model_id", document["model_id"])
    prompt_template_id = _identifier("prompt_template_id", document["prompt_template_id"])
    retrieval_sha = _sha256("retrieval_snapshot_sha256", document["retrieval_snapshot_sha256"])
    evaluator_model_id = _identifier("evaluator_model_id", document["evaluator_model_id"])
    evaluator_revision = _identifier("evaluator_revision", document["evaluator_revision"])
    created_at = _timestamp("created_at", document["created_at"])
    claims_value = document["claims"]
    if not isinstance(claims_value, list) or not claims_value:
        _invalid("claims must be a non-empty array")
    if len(claims_value) > active_policy.max_claims:
        _invalid("claims exceeds max_claims")
    claims = [_parse_claim(value, active_policy) for value in claims_value]
    claim_ids = [claim.claim_id for claim in claims]
    if len(set(claim_ids)) != len(claim_ids):
        _invalid("claim_id values must be unique")
    claims.sort(key=lambda item: item.claim_id)
    total_verdicts = sum(len(claim.verdicts) for claim in claims)
    if total_verdicts > active_policy.max_total_verdicts:
        _invalid("artifact exceeds max_total_verdicts")

    canonical_document = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "model_id": model_id,
        "prompt_template_id": prompt_template_id,
        "retrieval_snapshot_sha256": retrieval_sha,
        "evaluator_model_id": evaluator_model_id,
        "evaluator_revision": evaluator_revision,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "claims": [
            {
                "claim_id": claim.claim_id,
                "claim_sha256": claim.claim_sha256,
                "risk": claim.risk,
                "citations": sorted(claim.citations),
                "verdicts": [
                    {
                        "citation_id": verdict.citation_id,
                        "evidence_sha256": verdict.evidence_sha256,
                        "entailment_probability": verdict.entailment,
                        "neutral_probability": verdict.neutral,
                        "contradiction_probability": verdict.contradiction,
                    }
                    for verdict in sorted(claim.verdicts, key=lambda item: item.citation_id)
                ],
            }
            for claim in claims
        ],
    }
    canonical = json.dumps(
        canonical_document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    artifact_sha256 = _digest(canonical)
    findings: list[Finding] = []
    age_seconds = (now - created_at).total_seconds()
    if age_seconds > active_policy.max_artifact_age_seconds:
        findings.append(
            Finding(
                "STALE_ARTIFACT",
                {"age_seconds": age_seconds, "limit": active_policy.max_artifact_age_seconds},
            )
        )
    if age_seconds < -active_policy.max_future_skew_seconds:
        findings.append(
            Finding(
                "FUTURE_ARTIFACT",
                {
                    "future_skew_seconds": -age_seconds,
                    "limit": active_policy.max_future_skew_seconds,
                },
            )
        )

    supported_claims = 0
    supported_citations = 0
    contradicted_claims = 0
    best_entailments: list[float] = []
    max_contradiction = 0.0
    for claim in claims:
        best_entailment = max(verdict.entailment for verdict in claim.verdicts)
        claim_contradiction = max(verdict.contradiction for verdict in claim.verdicts)
        best_entailments.append(best_entailment)
        max_contradiction = max(max_contradiction, claim_contradiction)
        supported = best_entailment >= active_policy.min_claim_entailment
        contradicted = claim_contradiction > active_policy.max_citation_contradiction
        supported_claims += int(supported)
        contradicted_claims += int(contradicted)
        useful_citations = sum(
            verdict.entailment >= active_policy.min_citation_entailment
            and verdict.contradiction <= active_policy.max_citation_contradiction
            for verdict in claim.verdicts
        )
        supported_citations += useful_citations
        identity = {"claim_sha256": _digest(claim.claim_id), "risk": claim.risk}
        if not supported:
            findings.append(
                Finding(
                    "CLAIM_UNSUPPORTED",
                    {
                        **identity,
                        "best_entailment": best_entailment,
                        "required": active_policy.min_claim_entailment,
                    },
                    blocking=False,
                )
            )
        if contradicted:
            findings.append(
                Finding(
                    "CLAIM_CONTRADICTED",
                    {
                        **identity,
                        "max_contradiction": claim_contradiction,
                        "limit": active_policy.max_citation_contradiction,
                    },
                    blocking=False,
                )
            )
        if (
            claim.risk == "critical"
            and active_policy.require_all_critical_supported
            and not supported
        ):
            findings.append(Finding("CRITICAL_CLAIM_UNSUPPORTED", identity))
        if (
            claim.risk == "critical"
            and active_policy.require_no_critical_contradiction
            and contradicted
        ):
            findings.append(Finding("CRITICAL_CLAIM_CONTRADICTED", identity))

    claim_support_rate = supported_claims / len(claims)
    citation_support_rate = supported_citations / total_verdicts
    contradicted_claim_rate = contradicted_claims / len(claims)
    if claim_support_rate < active_policy.min_claim_support_rate:
        findings.append(
            Finding(
                "CLAIM_SUPPORT_RATE",
                {"observed": claim_support_rate, "required": active_policy.min_claim_support_rate},
            )
        )
    if citation_support_rate < active_policy.min_citation_support_rate:
        findings.append(
            Finding(
                "CITATION_SUPPORT_RATE",
                {
                    "observed": citation_support_rate,
                    "required": active_policy.min_citation_support_rate,
                },
            )
        )
    if contradicted_claim_rate > active_policy.max_contradicted_claim_rate:
        findings.append(
            Finding(
                "CONTRADICTED_CLAIM_RATE",
                {
                    "observed": contradicted_claim_rate,
                    "limit": active_policy.max_contradicted_claim_rate,
                },
            )
        )

    findings.sort(key=lambda item: (item.code, json.dumps(item.evidence, sort_keys=True)))
    finding_count = len(findings)
    if finding_count > _MAX_FINDINGS:
        findings = findings[: _MAX_FINDINGS - 1]
        findings.append(
            Finding("FINDINGS_TRUNCATED", {"observed": finding_count, "reported": _MAX_FINDINGS})
        )
    configuration = {
        "model_id": model_id,
        "prompt_template_id": prompt_template_id,
        "retrieval_snapshot_sha256": retrieval_sha,
        "evaluator_model_id": evaluator_model_id,
        "evaluator_revision": evaluator_revision,
    }
    policy_sha256 = _digest(
        json.dumps(asdict(active_policy), sort_keys=True, separators=(",", ":"))
    )
    metrics: dict[str, int | float] = {
        "claim_count": len(claims),
        "critical_claim_count": sum(claim.risk == "critical" for claim in claims),
        "citation_count": total_verdicts,
        "supported_claim_count": supported_claims,
        "supported_citation_count": supported_citations,
        "contradicted_claim_count": contradicted_claims,
        "claim_support_rate": claim_support_rate,
        "citation_support_rate": citation_support_rate,
        "contradicted_claim_rate": contradicted_claim_rate,
        "mean_best_entailment": sum(best_entailments) / len(best_entailments),
        "max_contradiction": max_contradiction,
        "artifact_age_seconds": age_seconds,
    }
    return EntailmentReport(
        accepted=not any(finding.blocking for finding in findings),
        artifact_sha256=artifact_sha256,
        benchmark_sha256=_digest(benchmark_id),
        configuration_sha256=_digest(
            json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        ),
        policy_sha256=policy_sha256,
        evaluated_at=now.isoformat().replace("+00:00", "Z"),
        metrics=metrics,
        findings=tuple(findings),
    )


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=1_048_576)
    args = parser.parse_args(argv)
    try:
        policy = EntailmentPolicy(max_input_bytes=args.max_bytes)
        if args.artifact.stat().st_size > policy.max_input_bytes:
            _invalid("artifact byte size is outside the configured budget")
        document = load_artifact(args.artifact.read_bytes(), max_bytes=policy.max_input_bytes)
        report = audit_entailment(document, evaluated_at=datetime.now(UTC), policy=policy)
        payload = json.dumps(report.as_dict(), sort_keys=True, separators=(",", ":"))
        exit_code = 0 if report.accepted else 3
    except (OSError, ValueError):
        payload = json.dumps(
            {
                "schema_version": REPORT_VERSION,
                "accepted": False,
                "error_code": "MALFORMED_ARTIFACT",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        exit_code = 2
    if args.output is None:
        print(payload)
    else:
        _write_atomic(args.output, payload)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
