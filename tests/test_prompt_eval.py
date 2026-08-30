from src.prompt_eval import PromptEvalCase, evaluate_prompt_modes
from src.prompting import EvidenceSnippet


def deterministic_generator(prompt: str) -> dict:
    return {
        "answer": "The supplied evidence requires authenticated access.",
        "citations": ["security:1"],
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
