from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Callable

from src.prompting import EvidenceSnippet, PromptMode, build_prompt

Generator = Callable[[str], dict]


@dataclass(frozen=True)
class PromptEvalCase:
    question: str
    evidence: tuple[EvidenceSnippet, ...]
    expected_insufficient_evidence: bool = False


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

    def to_dict(self) -> dict:
        return asdict(self)


def _schema_valid(payload: dict) -> bool:
    required = {"answer", "citations", "confidence", "insufficient_evidence"}
    if not required.issubset(payload):
        return False
    if not isinstance(payload["answer"], str):
        return False
    if not isinstance(payload["citations"], list):
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
) -> dict[str, dict]:
    """Compare structured-output behavior across zero/one/few-shot prompts.

    The caller supplies the generator, making the harness usable with a hosted
    model, a local OpenAI-compatible gateway, or deterministic test doubles.
    Retrieval is held fixed so the comparison isolates prompt-mode behavior.
    """
    if not cases:
        return {mode.value: PromptModeMetrics(mode.value, 0, 0, 0, 0, 0, 0, 0).to_dict() for mode in modes}

    results: dict[str, dict] = {}
    for mode in modes:
        schema_scores: list[float] = []
        citation_scores: list[float] = []
        insufficiency_scores: list[float] = []
        confidences: list[float] = []
        prompt_sizes: list[float] = []
        latencies: list[float] = []

        for case in cases:
            prompt = build_prompt(case.question, list(case.evidence), mode)
            prompt_sizes.append(float(len(prompt)))
            started = time.perf_counter()
            payload = generator(prompt)
            latencies.append((time.perf_counter() - started) * 1000)

            valid_schema = _schema_valid(payload)
            schema_scores.append(float(valid_schema))
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
                float(bool(payload["insufficient_evidence"]) == case.expected_insufficient_evidence)
            )
            confidences.append(float(payload["confidence"]))

        metrics = PromptModeMetrics(
            mode=mode.value,
            cases=len(cases),
            schema_validity_rate=mean(schema_scores),
            citation_validity_rate=mean(citation_scores),
            insufficiency_accuracy=mean(insufficiency_scores),
            average_confidence=mean(confidences),
            average_prompt_characters=mean(prompt_sizes),
            average_latency_ms=mean(latencies),
        )
        results[mode.value] = metrics.to_dict()

    return results
