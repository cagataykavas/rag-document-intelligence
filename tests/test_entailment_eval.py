from __future__ import annotations

import math

import pytest

from src.entailment_eval import (
    EntailmentEvaluationError,
    EntailmentLabel,
    EntailmentPair,
    EntailmentPolicy,
    evaluate_entailment_diagnostic,
)


def pair(
    case_id: str,
    label: EntailmentLabel,
    *,
    claim: str | None = None,
    evidence: str | None = None,
) -> EntailmentPair:
    return EntailmentPair(
        case_id=case_id,
        claim=claim or f"Claim {case_id}",
        evidence=evidence or f"Evidence {case_id}",
        label=label,
    )


def scorer_for(scores: dict[str, float]):
    return lambda claim, _evidence: scores[claim]


def test_calibrates_threshold_without_using_holdout_labels() -> None:
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.SUPPORTED),
        pair("c3", EntailmentLabel.UNSUPPORTED),
        pair("c4", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.SUPPORTED),
        pair("e3", EntailmentLabel.UNSUPPORTED),
        pair("e4", EntailmentLabel.UNSUPPORTED),
    ]
    scores = {
        "Claim c1": 0.95,
        "Claim c2": 0.82,
        "Claim c3": 0.40,
        "Claim c4": 0.10,
        "Claim e1": 0.91,
        "Claim e2": 0.86,
        "Claim e3": 0.30,
        "Claim e4": 0.20,
    }

    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        scorer_for(scores),
        scorer_id="cross-encoder-v1",
    )

    assert report.threshold == 0.82
    assert report.calibration.supported_precision == 1.0
    assert report.calibration.supported_recall == 1.0
    assert report.evaluation.balanced_accuracy == 1.0
    assert report.meets_policy
    assert report.failure_reasons == ()
    assert len(report.calibration_sha256) == 64
    assert len(report.evaluation_sha256) == 64
    assert len(report.report_sha256) == 64


def test_precision_target_controls_calibration_threshold() -> None:
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.SUPPORTED),
        pair("c3", EntailmentLabel.UNSUPPORTED),
        pair("c4", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.SUPPORTED),
        pair("e3", EntailmentLabel.UNSUPPORTED),
        pair("e4", EntailmentLabel.UNSUPPORTED),
    ]
    scores = {
        "Claim c1": 0.95,
        "Claim c2": 0.70,
        "Claim c3": 0.80,
        "Claim c4": 0.20,
        "Claim e1": 0.99,
        "Claim e2": 0.96,
        "Claim e3": 0.40,
        "Claim e4": 0.10,
    }

    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        scorer_for(scores),
        scorer_id="risk-calibrated-v1",
        policy=EntailmentPolicy(
            min_supported_precision=1.0,
            min_supported_recall=0.5,
        ),
    )

    assert report.threshold == 0.95
    assert report.calibration.supported_precision == 1.0
    assert report.calibration.supported_recall == 0.5
    assert report.meets_policy


def test_holdout_failure_is_reported_without_changing_admission() -> None:
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.SUPPORTED),
        pair("c3", EntailmentLabel.UNSUPPORTED),
        pair("c4", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]
    scores = {
        "Claim c1": 0.90,
        "Claim c2": 0.80,
        "Claim c3": 0.30,
        "Claim c4": 0.10,
        "Claim e1": 0.70,
        "Claim e2": 0.95,
    }

    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        scorer_for(scores),
        scorer_id="regressed-v2",
    )

    assert report.threshold == 0.8
    assert not report.meets_policy
    assert report.failure_reasons == (
        "evaluation_supported_precision_below_minimum",
        "evaluation_supported_recall_below_minimum",
    )
    assert report.evaluation.false_supported == 1
    assert report.evaluation.false_unsupported == 1


def test_calibration_and_holdout_content_overlap_fails_closed() -> None:
    calibration = [
        pair(
            "c1",
            EntailmentLabel.SUPPORTED,
            claim="Same claim",
            evidence="Same evidence",
        ),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair(
            "e1",
            EntailmentLabel.SUPPORTED,
            claim="Same claim",
            evidence="Same evidence",
        ),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]

    with pytest.raises(EntailmentEvaluationError, match="content overlap"):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            lambda _claim, _evidence: 0.5,
            scorer_id="scorer-v1",
        )


def test_calibration_and_holdout_case_id_overlap_fails_closed() -> None:
    calibration = [
        pair("shared", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair(
            "shared",
            EntailmentLabel.SUPPORTED,
            claim="Different claim",
            evidence="Different evidence",
        ),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]

    with pytest.raises(EntailmentEvaluationError, match="case_id overlap"):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            lambda _claim, _evidence: 0.5,
            scorer_id="scorer-v1",
        )


@pytest.mark.parametrize(
    ("calibration", "message"),
    [
        (
            [
                pair("same", EntailmentLabel.SUPPORTED),
                pair("same", EntailmentLabel.UNSUPPORTED),
            ],
            "duplicate case_id",
        ),
        (
            [
                pair("c1", EntailmentLabel.SUPPORTED, claim="x", evidence="y"),
                pair("c2", EntailmentLabel.UNSUPPORTED, claim="x", evidence="y"),
            ],
            "duplicate content",
        ),
        (
            [
                pair("c1", EntailmentLabel.SUPPORTED),
                pair("c2", EntailmentLabel.SUPPORTED),
            ],
            "both labels",
        ),
    ],
)
def test_invalid_calibration_split_fails_closed(calibration, message) -> None:
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]

    with pytest.raises(EntailmentEvaluationError, match=message):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            lambda _claim, _evidence: 0.5,
            scorer_id="scorer-v1",
        )


@pytest.mark.parametrize("invalid_score", [True, -0.1, 1.1, math.nan, math.inf, "0.9"])
def test_invalid_scorer_probability_fails_closed(invalid_score) -> None:
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]

    with pytest.raises(EntailmentEvaluationError, match="invalid probability"):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            lambda _claim, _evidence: invalid_score,
            scorer_id="scorer-v1",
        )


def test_scorer_failure_is_normalized_without_content_leakage() -> None:
    secret = "customer secret in provider diagnostic"
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]

    def failing_scorer(_claim: str, _evidence: str) -> float:
        raise RuntimeError(secret)

    with pytest.raises(EntailmentEvaluationError, match="scorer failed") as caught:
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            failing_scorer,
            scorer_id="scorer-v1",
        )

    assert secret not in str(caught.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_supported_precision": True},
        {"min_supported_recall": math.nan},
        {"min_supported_recall": 1.1},
        {"max_pairs_per_split": 1},
        {"max_claim_utf8_bytes": 0},
    ],
)
def test_policy_rejects_unsafe_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        EntailmentPolicy(**kwargs)


def test_text_and_total_byte_boundaries_are_enforced() -> None:
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED, claim="12345"),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]
    policy = EntailmentPolicy(max_claim_utf8_bytes=4)

    with pytest.raises(EntailmentEvaluationError, match="claim exceeds"):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            lambda _claim, _evidence: 0.5,
            scorer_id="scorer-v1",
            policy=policy,
        )


def test_audit_report_contains_commitments_not_raw_claims_or_evidence() -> None:
    calibration = [
        pair(
            "c1",
            EntailmentLabel.SUPPORTED,
            claim="private claim c1",
            evidence="private evidence c1",
        ),
        pair(
            "c2",
            EntailmentLabel.UNSUPPORTED,
            claim="private claim c2",
            evidence="private evidence c2",
        ),
    ]
    evaluation = [
        pair(
            "e1",
            EntailmentLabel.SUPPORTED,
            claim="private claim e1",
            evidence="private evidence e1",
        ),
        pair(
            "e2",
            EntailmentLabel.UNSUPPORTED,
            claim="private claim e2",
            evidence="private evidence e2",
        ),
    ]
    scores = {
        "private claim c1": 0.9,
        "private claim c2": 0.1,
        "private claim e1": 0.9,
        "private claim e2": 0.1,
    }

    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        scorer_for(scores),
        scorer_id="privacy-check-v1",
    )

    serialized = repr(report.to_audit_dict())
    assert "private claim" not in serialized
    assert "private evidence" not in serialized
    assert (
        report.report_sha256
        == evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            scorer_for(scores),
            scorer_id="privacy-check-v1",
        ).report_sha256
    )
