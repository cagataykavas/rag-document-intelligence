from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Callable

from src.evidence_snapshot import EvidenceSnapshotError, build_evidence_snapshot
from src.generation_policy import CONTENT_BOUND_GENERATION_POLICY, admit_generation
from src.prompting import EvidenceSnippet, PromptMode, build_prompt

Generator = Callable[[str], dict]


@dataclass(frozen=True)
class PromptEvalCase:
    question: str
    evidence: tuple[EvidenceSnippet, ...]
    expected_insufficient_evidence: bool = False


@dataclass(frozen=True)
class PromptReleasePolicy:
    """Minimum production-contract behavior required for every prompt mode."""

    min_admission_rate: float = 1.0
    min_citation_validity_rate: float = 1.0
    min_insufficiency_accuracy: float = 1.0
    max_generator_error_rate: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class PromptModeMetrics:
    mode: str
    cases: int
    schema_validity_rate: float
    citation_validity_rate: float
    insufficiency_accuracy: float
    average_confidence: float
    average_prompt_characters: float
    average_latency_ms: float
    production_admission_rate: float
    generator_error_rate: float
    admission_reason_counts: dict[str, int]
    release_accepted: bool
    release_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _schema_valid(payload: object) -> bool:
    required = {
        "answer",
        "citations",
        "claim_citations",
        "confidence",
        "insufficient_evidence",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        return False
    if not isinstance(payload["answer"], str):
        return False
    if not isinstance(payload["citations"], list):
        return False
    if not isinstance(payload["claim_citations"], list):
        return False
    if not isinstance(payload["insufficient_evidence"], bool):
        return False
    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError):
        return False
    return 0.0 <= confidence <= 1.0


def evaluate_prompt_modes(
    cases: list[PromptEvalCase],
    generator: Generator,
    *,
    modes: tuple[PromptMode, ...] = (
        PromptMode.ZERO_SHOT,
        PromptMode.ONE_SHOT,
        PromptMode.FEW_SHOT,
    ),
    release_policy: PromptReleasePolicy = PromptReleasePolicy(),
) -> dict[str, dict]:
    """Compare structured-output behavior across zero/one/few-shot prompts.

    The caller supplies the generator, making the harness usable with a hosted
    model, a local OpenAI-compatible gateway, or deterministic test doubles.
    Retrieval is held fixed so the comparison isolates prompt-mode behavior.
    """
    results: dict[str, dict] = {}
    for mode in modes:
        schema_scores: list[float] = []
        citation_scores: list[float] = []
        insufficiency_scores: list[float] = []
        confidences: list[float] = []
        prompt_sizes: list[float] = []
        latencies: list[float] = []
        admission_scores: list[float] = []
        generator_error_scores: list[float] = []
        admission_reasons: Counter[str] = Counter()

        for case in cases:
            prompt = build_prompt(case.question, list(case.evidence), mode)
            prompt_sizes.append(float(len(prompt)))
            started = time.perf_counter()
            try:
                payload = generator(prompt)
            except Exception:  # noqa: BLE001 - providers are an untrusted benchmark boundary
                latencies.append((time.perf_counter() - started) * 1000)
                schema_scores.append(0.0)
                citation_scores.append(0.0)
                insufficiency_scores.append(0.0)
                confidences.append(0.0)
                admission_scores.append(0.0)
                generator_error_scores.append(1.0)
                admission_reasons["generator_error"] += 1
                continue
            latencies.append((time.perf_counter() - started) * 1000)
            generator_error_scores.append(0.0)

            valid_schema = _schema_valid(payload)
            schema_scores.append(float(valid_schema))

            snapshot_sha256 = None
            if case.evidence:
                try:
                    snapshot_sha256 = build_evidence_snapshot(
                        case.evidence
                    ).snapshot_sha256
                except EvidenceSnapshotError:
                    admission_reasons["evidence_snapshot_invalid"] += 1
            admission = admit_generation(
                payload,
                {snippet.chunk_id for snippet in case.evidence},
                CONTENT_BOUND_GENERATION_POLICY,
                evidence_snapshot_sha256=snapshot_sha256,
            )
            admission_scores.append(float(admission.accepted))
            admission_reasons.update(admission.reasons)

            if not valid_schema:
                citation_scores.append(0.0)
                insufficiency_scores.append(0.0)
                confidences.append(0.0)
                continue

            allowed = {snippet.chunk_id for snippet in case.evidence}
            citations = [str(item) for item in payload["citations"]]
            citation_scores.append(
                sum(item in allowed for item in citations) / len(citations)
                if citations
                else float(payload["insufficient_evidence"])
            )
            insufficiency_scores.append(
                float(
                    bool(payload["insufficient_evidence"])
                    == case.expected_insufficient_evidence
                )
            )
            confidences.append(float(payload["confidence"]))

        admission_rate = mean(admission_scores) if admission_scores else 0.0
        citation_rate = mean(citation_scores) if citation_scores else 0.0
        insufficiency_accuracy = (
            mean(insufficiency_scores) if insufficiency_scores else 0.0
        )
        generator_error_rate = (
            mean(generator_error_scores) if generator_error_scores else 0.0
        )
        release_reasons: list[str] = []
        if admission_rate < release_policy.min_admission_rate:
            release_reasons.append("production_admission_rate_below_minimum")
        if citation_rate < release_policy.min_citation_validity_rate:
            release_reasons.append("citation_validity_rate_below_minimum")
        if insufficiency_accuracy < release_policy.min_insufficiency_accuracy:
            release_reasons.append("insufficiency_accuracy_below_minimum")
        if generator_error_rate > release_policy.max_generator_error_rate:
            release_reasons.append("generator_error_rate_above_maximum")

        metrics = PromptModeMetrics(
            mode=mode.value,
            cases=len(cases),
            schema_validity_rate=mean(schema_scores) if schema_scores else 0.0,
            citation_validity_rate=citation_rate,
            insufficiency_accuracy=insufficiency_accuracy,
            average_confidence=mean(confidences) if confidences else 0.0,
            average_prompt_characters=mean(prompt_sizes) if prompt_sizes else 0.0,
            average_latency_ms=mean(latencies) if latencies else 0.0,
            production_admission_rate=admission_rate,
            generator_error_rate=generator_error_rate,
            admission_reason_counts=dict(sorted(admission_reasons.items())),
            release_accepted=not release_reasons,
            release_reasons=tuple(release_reasons),
        )
        results[mode.value] = metrics.to_dict()

    return results
