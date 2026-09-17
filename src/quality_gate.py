from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class GateOutcome(str, Enum):
    PASS = "pass"
    ABSTAIN = "abstain"
    FAIL = "fail"


@dataclass(frozen=True)
class GroundingGateConfig:
    minimum_citation_validity: float = 1.0
    minimum_lexical_grounding: float = 0.6
    minimum_claim_support: float = 0.5
    minimum_supported_claim_rate: float = 1.0
    allow_rejected_citations: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_citation_validity", self.minimum_citation_validity),
            ("minimum_lexical_grounding", self.minimum_lexical_grounding),
            ("minimum_claim_support", self.minimum_claim_support),
            ("minimum_supported_claim_rate", self.minimum_supported_claim_rate),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class GroundingGateResult:
    outcome: GateOutcome
    publishable: bool
    reasons: tuple[str, ...]
    citation_validity_rate: float
    lexical_grounding_rate: float
    supported_claim_rate: float
    claims: int
    supported_claims: int
    rejected_citations: int
    quarantined_chunks: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outcome"] = self.outcome.value
        return payload


def _rate(audit: Mapping[str, Any], key: str) -> float:
    try:
        value = float(audit.get(key, 0.0))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"citation_audit.{key} must be numeric") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"citation_audit.{key} must be between 0 and 1")
    return value


def _sequence(answer: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = answer.get(key, ())
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{key} must be an array")
    return value


def evaluate_grounding_gate(
    answer: Mapping[str, Any],
    config: GroundingGateConfig | None = None,
) -> GroundingGateResult:
    """Turn RAG evidence diagnostics into a deterministic publish/abstain decision.

    The gate consumes the JSON shape returned by Answer. It enforces inspectable
    lexical and citation contracts; it is not a semantic entailment verifier.
    """
    policy = config or GroundingGateConfig()
    audit = answer.get("citation_audit")
    if not isinstance(audit, Mapping):
        raise TypeError("citation_audit must be an object")

    citation_validity = _rate(audit, "citation_validity_rate")
    lexical_grounding = _rate(audit, "lexical_grounding_rate")
    citations = _sequence(answer, "citations")
    rejected = _sequence(answer, "rejected_citations")
    claim_rows = _sequence(answer, "claim_support")
    quarantined = _sequence(answer, "quarantined_chunk_ids")
    insufficient = bool(answer.get("insufficient_evidence", False))
    answer_text = str(answer.get("answer", "")).strip()

    if insufficient:
        reasons = []
        if citations:
            reasons.append("abstention_contains_citations")
        if rejected:
            reasons.append("abstention_contains_rejected_citations")
        outcome = GateOutcome.FAIL if reasons else GateOutcome.ABSTAIN
        return GroundingGateResult(
            outcome=outcome,
            publishable=False,
            reasons=tuple(reasons) or ("insufficient_evidence",),
            citation_validity_rate=citation_validity,
            lexical_grounding_rate=lexical_grounding,
            supported_claim_rate=0.0,
            claims=0,
            supported_claims=0,
            rejected_citations=len(rejected),
            quarantined_chunks=len(quarantined),
        )

    supported_claims = 0
    for index, row in enumerate(claim_rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"claim_support[{index}] must be an object")
        try:
            support = float(row.get("lexical_support", 0.0))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"claim_support[{index}].lexical_support must be numeric"
            ) from exc
        if not 0.0 <= support <= 1.0:
            raise ValueError(
                f"claim_support[{index}].lexical_support must be between 0 and 1"
            )
        if row.get("best_chunk_id") is not None and support >= policy.minimum_claim_support:
            supported_claims += 1

    claim_count = len(claim_rows)
    supported_claim_rate = supported_claims / claim_count if claim_count else 0.0
    reasons: list[str] = []
    if not answer_text:
        reasons.append("empty_answer")
    if not citations:
        reasons.append("missing_citations")
    if citation_validity < policy.minimum_citation_validity:
        reasons.append("citation_validity_below_threshold")
    if lexical_grounding < policy.minimum_lexical_grounding:
        reasons.append("lexical_grounding_below_threshold")
    if not claim_rows:
        reasons.append("missing_claim_trace")
    elif supported_claim_rate < policy.minimum_supported_claim_rate:
        reasons.append("supported_claim_rate_below_threshold")
    if rejected and not policy.allow_rejected_citations:
        reasons.append("rejected_citations_present")

    return GroundingGateResult(
        outcome=GateOutcome.FAIL if reasons else GateOutcome.PASS,
        publishable=not reasons,
        reasons=tuple(reasons),
        citation_validity_rate=citation_validity,
        lexical_grounding_rate=lexical_grounding,
        supported_claim_rate=supported_claim_rate,
        claims=claim_count,
        supported_claims=supported_claims,
        rejected_citations=len(rejected),
        quarantined_chunks=len(quarantined),
    )
