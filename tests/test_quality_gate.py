from __future__ import annotations

import pytest

from src.quality_gate import (
    GateOutcome,
    GroundingGateConfig,
    evaluate_grounding_gate,
)


def _answer() -> dict:
    return {
        "answer": "Audit records are retained for seven years.",
        "citations": ["policy:1"],
        "rejected_citations": [],
        "insufficient_evidence": False,
        "citation_audit": {
            "citation_validity_rate": 1.0,
            "lexical_grounding_rate": 0.9,
        },
        "claim_support": [
            {
                "claim": "Audit records are retained for seven years.",
                "best_chunk_id": "policy:1",
                "lexical_support": 0.85,
            }
        ],
        "quarantined_chunk_ids": [],
    }


def test_grounded_answer_passes_publish_gate() -> None:
    result = evaluate_grounding_gate(_answer())

    assert result.outcome is GateOutcome.PASS
    assert result.publishable
    assert result.supported_claim_rate == 1.0
    assert result.to_dict()["outcome"] == "pass"


def test_safe_insufficient_evidence_response_abstains() -> None:
    answer = _answer()
    answer.update(
        {
            "insufficient_evidence": True,
            "citations": [],
            "claim_support": [],
            "citation_audit": {
                "citation_validity_rate": 0.0,
                "lexical_grounding_rate": 0.0,
            },
        }
    )

    result = evaluate_grounding_gate(answer)

    assert result.outcome is GateOutcome.ABSTAIN
    assert not result.publishable
    assert result.reasons == ("insufficient_evidence",)


def test_abstention_with_citations_fails_contract() -> None:
    answer = _answer()
    answer["insufficient_evidence"] = True

    result = evaluate_grounding_gate(answer)

    assert result.outcome is GateOutcome.FAIL
    assert "abstention_contains_citations" in result.reasons


def test_unsupported_claim_and_rejected_citation_fail_gate() -> None:
    answer = _answer()
    answer["rejected_citations"] = ["invented:404"]
    answer["citation_audit"]["citation_validity_rate"] = 0.5
    answer["claim_support"].append(
        {
            "claim": "The policy has no exceptions.",
            "best_chunk_id": None,
            "lexical_support": 0.0,
        }
    )

    result = evaluate_grounding_gate(answer)

    assert result.outcome is GateOutcome.FAIL
    assert result.supported_claim_rate == 0.5
    assert "citation_validity_below_threshold" in result.reasons
    assert "supported_claim_rate_below_threshold" in result.reasons
    assert "rejected_citations_present" in result.reasons


def test_thresholds_are_explicitly_configurable() -> None:
    answer = _answer()
    answer["citation_audit"]["lexical_grounding_rate"] = 0.4
    answer["claim_support"][0]["lexical_support"] = 0.3
    config = GroundingGateConfig(
        minimum_lexical_grounding=0.35,
        minimum_claim_support=0.25,
    )

    assert evaluate_grounding_gate(answer, config).outcome is GateOutcome.PASS


def test_invalid_metric_ranges_are_rejected() -> None:
    answer = _answer()
    answer["citation_audit"]["lexical_grounding_rate"] = 1.2

    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_grounding_gate(answer)

    with pytest.raises(ValueError, match="minimum_claim_support"):
        GroundingGateConfig(minimum_claim_support=-0.1)
