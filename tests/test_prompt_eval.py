import pytest

from src.prompt_eval import (
    PromptEvalCase,
    PromptReleasePolicy,
    evaluate_prompt_modes,
)
from src.prompting import EvidenceSnippet, PromptMode


def deterministic_generator(prompt: str) -> dict:
    return {
        "answer": "The supplied evidence requires authenticated access.",
        "citations": ["security:1"],
        "claim_citations": [{"claim_index": 0, "citations": ["security:1"]}],
        "confidence": 0.9,
        "insufficient_evidence": False,
    }


def test_prompt_mode_evaluation_compares_zero_one_few_shot():
    case = PromptEvalCase(
        question="Is authentication required?",
        evidence=(
            EvidenceSnippet(
                "security:1",
                "security.md",
                "Administrative access requires authentication.",
            ),
        ),
    )
    result = evaluate_prompt_modes([case], deterministic_generator)

    assert set(result) == {"zero_shot", "one_shot", "few_shot"}
    assert result["zero_shot"]["schema_validity_rate"] == 1.0
    assert result["one_shot"]["citation_validity_rate"] == 1.0
    assert result["few_shot"]["insufficiency_accuracy"] == 1.0
    assert result["zero_shot"]["production_admission_rate"] == 1.0
    assert result["zero_shot"]["release_accepted"] is True
    assert (
        result["zero_shot"]["average_prompt_characters"]
        < result["one_shot"]["average_prompt_characters"]
        < result["few_shot"]["average_prompt_characters"]
    )


def test_invalid_generator_schema_is_scored_not_crashed():
    case = PromptEvalCase(
        question="Question?",
        evidence=(EvidenceSnippet("c:1", "demo", "Evidence."),),
    )
    result = evaluate_prompt_modes([case], lambda _prompt: {"answer": "missing fields"})
    assert result["zero_shot"]["schema_validity_rate"] == 0.0
    assert result["zero_shot"]["citation_validity_rate"] == 0.0
    assert result["zero_shot"]["production_admission_rate"] == 0.0
    assert result["zero_shot"]["release_accepted"] is False
    assert (
        result["zero_shot"]["admission_reason_counts"]["missing_field:citations"] == 1
    )


def test_benchmark_uses_production_claim_citation_admission_contract():
    case = PromptEvalCase(
        question="Summarize both requirements.",
        evidence=(
            EvidenceSnippet(
                "security:1",
                "security.md",
                "Access requires authentication. Audit events are retained.",
            ),
        ),
    )

    def incomplete_claim_map(_prompt: str) -> dict:
        return {
            "answer": "Access requires authentication. Audit events are retained.",
            "citations": ["security:1"],
            "claim_citations": [{"claim_index": 0, "citations": ["security:1"]}],
            "confidence": 0.9,
            "insufficient_evidence": False,
        }

    result = evaluate_prompt_modes(
        [case],
        incomplete_claim_map,
        modes=(PromptMode.ZERO_SHOT,),
    )["zero_shot"]

    assert result["schema_validity_rate"] == 1.0
    assert result["citation_validity_rate"] == 1.0
    assert result["production_admission_rate"] == 0.0
    assert result["admission_reason_counts"] == {
        "incomplete_claim_citation_coverage": 1
    }
    assert result["release_reasons"] == ("production_admission_rate_below_minimum",)


def test_generator_failure_is_bounded_and_content_free():
    case = PromptEvalCase(
        question="Question?",
        evidence=(EvidenceSnippet("c:1", "demo", "Evidence."),),
    )

    def failing_generator(_prompt: str) -> dict:
        raise RuntimeError("provider body with sensitive material")

    result = evaluate_prompt_modes(
        [case],
        failing_generator,
        modes=(PromptMode.ZERO_SHOT,),
    )["zero_shot"]

    assert result["generator_error_rate"] == 1.0
    assert result["admission_reason_counts"] == {"generator_error": 1}
    assert "sensitive" not in repr(result)
    assert result["release_accepted"] is False


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"min_admission_rate": -0.1}, ValueError),
        ({"max_generator_error_rate": 1.1}, ValueError),
        ({"min_citation_validity_rate": True}, TypeError),
    ],
)
def test_release_policy_rejects_invalid_thresholds(kwargs, error):
    with pytest.raises(error):
        PromptReleasePolicy(**kwargs)
