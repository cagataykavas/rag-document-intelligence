from __future__ import annotations

import math

import pytest

from src.entailment_eval import (
    DriftSlicePolicy,
    EntailmentEvaluationError,
    EntailmentLabel,
    EntailmentPair,
    EntailmentPolicy,
    EntailmentSliceMetadata,
    VersionedSemanticLeakageGuard,
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


def test_unicode_and_formatting_near_duplicate_leakage_fails_before_scoring() -> None:
    calibration = [
        pair(
            "c1",
            EntailmentLabel.SUPPORTED,
            claim="İdari erişim için güçlü kimlik doğrulama gerekir.",
            evidence="Tüm idari erişimler, güçlü kimlik doğrulama gerektirir.",
        ),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair(
            "e1",
            EntailmentLabel.SUPPORTED,
            claim="I\u0307DARI ERI\u0307ŞI\u0307M I\u0307ÇI\u0307N GÜÇLÜ KI\u0307MLI\u0307K DOĞRULAMA GEREKI\u0307R!",
            evidence="Tüm idari erişimler güçlü kimlik doğrulama gerektirir",
        ),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]
    scorer_called = False

    def scorer(_claim: str, _evidence: str) -> float:
        nonlocal scorer_called
        scorer_called = True
        return 0.5

    with pytest.raises(
        EntailmentEvaluationError, match="near-duplicate content overlap"
    ):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            scorer,
            scorer_id="scorer-v1",
        )

    assert not scorer_called


def test_near_duplicate_guard_requires_both_claim_and_evidence_similarity() -> None:
    calibration = [
        pair(
            "c1",
            EntailmentLabel.SUPPORTED,
            claim="Administrative access requires authentication.",
            evidence="Administrative access requires a hardware security key.",
        ),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair(
            "e1",
            EntailmentLabel.SUPPORTED,
            claim="Administrative access requires authentication!",
            evidence="Administrative access is permitted with an emailed one-time code.",
        ),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]
    scores = {
        "Administrative access requires authentication.": 0.9,
        "Claim c2": 0.1,
        "Administrative access requires authentication!": 0.9,
        "Claim e2": 0.1,
    }

    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        scorer_for(scores),
        scorer_id="scorer-v1",
    )

    assert report.meets_policy
    assert report.near_duplicate_comparisons > 0
    assert (
        report.leakage_guard_version == "unicode-nfkc-casefold-tr-dot-char5-jaccard/v1"
    )


def test_within_split_near_duplicate_content_fails_closed() -> None:
    calibration = [
        pair(
            "c1",
            EntailmentLabel.SUPPORTED,
            claim="The retention period is exactly ninety calendar days.",
            evidence="Records remain available for exactly ninety calendar days.",
        ),
        pair(
            "c2",
            EntailmentLabel.UNSUPPORTED,
            claim="The retention period is exactly ninety calendar days!",
            evidence="Records remain available for exactly ninety calendar days!",
        ),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]

    with pytest.raises(EntailmentEvaluationError, match="contains near-duplicate"):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            lambda _claim, _evidence: 0.5,
            scorer_id="scorer-v1",
        )


def test_near_duplicate_comparison_budget_fails_closed() -> None:
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]

    with pytest.raises(EntailmentEvaluationError, match="comparison budget"):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            lambda _claim, _evidence: 0.5,
            scorer_id="scorer-v1",
            policy=EntailmentPolicy(max_near_duplicate_comparisons=1),
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
        {"near_duplicate_similarity_threshold": math.nan},
        {"near_duplicate_similarity_threshold": 0.0},
        {"near_duplicate_similarity_threshold": 1.1},
        {"max_near_duplicate_comparisons": 0},
        {"semantic_duplicate_similarity_threshold": math.nan},
        {"semantic_duplicate_similarity_threshold": 0.0},
        {"semantic_duplicate_similarity_threshold": 1.1},
        {"max_semantic_duplicate_comparisons": 0},
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


def test_versioned_semantic_guard_rejects_turkish_paraphrase_before_scoring() -> None:
    calibration = [
        pair(
            "c1",
            EntailmentLabel.SUPPORTED,
            claim="Yönetici oturumlarında donanım anahtarı zorunludur.",
            evidence="İdari erişim için fiziksel bir güvenlik anahtarı kullanılmalıdır.",
        ),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair(
            "e1",
            EntailmentLabel.SUPPORTED,
            claim="Admin girişleri yalnızca fiziksel ikinci faktörle açılır.",
            evidence="Yönetim yetkisi, cihaz tabanlı kimlik doğrulaması gerektirir.",
        ),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]
    main_scorer_called = False

    def main_scorer(_claim: str, _evidence: str) -> float:
        nonlocal main_scorer_called
        main_scorer_called = True
        return 0.5

    def semantic_scorer(
        left_claim: str, _left_evidence: str, right_claim: str, _right_evidence: str
    ) -> float:
        target = {left_claim, right_claim}
        if target == {calibration[0].claim, evaluation[0].claim}:
            return 0.97
        return 0.05

    guard = VersionedSemanticLeakageGuard(
        scorer=semantic_scorer,
        scorer_id="multilingual-nli:turkish-v1",
        model_revision_sha256="b" * 64,
    )

    with pytest.raises(
        EntailmentEvaluationError, match="semantic duplicate content overlap"
    ):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            main_scorer,
            scorer_id="entailment-v1",
            semantic_leakage_guard=guard,
        )

    assert not main_scorer_called


def test_semantic_guard_provenance_and_work_are_bound_to_report() -> None:
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]
    guard = VersionedSemanticLeakageGuard(
        scorer=lambda *_texts: 0.1,
        scorer_id="multilingual-nli:v3",
        model_revision_sha256="c" * 64,
    )

    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        lambda claim, _evidence: 0.9 if claim.endswith("1") else 0.1,
        scorer_id="entailment-v1",
        semantic_leakage_guard=guard,
    )

    assert report.schema_version == "entailment-diagnostic/v4"
    assert (
        report.semantic_leakage_guard_version
        == "versioned-pairwise-semantic-similarity/v1"
    )
    assert report.semantic_scorer_id == "multilingual-nli:v3"
    assert report.semantic_model_revision_sha256 == "c" * 64
    assert report.semantic_duplicate_comparisons == 6
    assert "Claim c1" not in repr(report.to_audit_dict())


def test_semantic_comparison_budget_fails_before_entailment_scorer() -> None:
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]
    main_scorer_called = False

    def main_scorer(_claim: str, _evidence: str) -> float:
        nonlocal main_scorer_called
        main_scorer_called = True
        return 0.5

    with pytest.raises(EntailmentEvaluationError, match="comparison budget"):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            main_scorer,
            scorer_id="entailment-v1",
            policy=EntailmentPolicy(max_semantic_duplicate_comparisons=5),
            semantic_leakage_guard=VersionedSemanticLeakageGuard(
                scorer=lambda *_texts: 0.1,
                scorer_id="multilingual-nli:v3",
                model_revision_sha256="c" * 64,
            ),
        )

    assert not main_scorer_called


@pytest.mark.parametrize("invalid_similarity", [True, -0.1, 1.1, math.nan, "0.5"])
def test_invalid_semantic_similarity_fails_closed(invalid_similarity: object) -> None:
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]

    with pytest.raises(EntailmentEvaluationError, match="invalid similarity"):
        evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            lambda _claim, _evidence: 0.5,
            scorer_id="entailment-v1",
            semantic_leakage_guard=VersionedSemanticLeakageGuard(
                scorer=lambda *_texts: invalid_similarity,  # type: ignore[arg-type]
                scorer_id="multilingual-nli:v3",
                model_revision_sha256="c" * 64,
            ),
        )


def test_semantic_scorer_failure_is_normalized_without_content_leakage() -> None:
    secret = "private provider diagnostic with customer text"

    def failing_semantic_scorer(*_texts: str) -> float:
        raise RuntimeError(secret)

    with pytest.raises(
        EntailmentEvaluationError, match="semantic leakage scorer failed"
    ) as caught:
        evaluate_entailment_diagnostic(
            [
                pair("c1", EntailmentLabel.SUPPORTED),
                pair("c2", EntailmentLabel.UNSUPPORTED),
            ],
            [
                pair("e1", EntailmentLabel.SUPPORTED),
                pair("e2", EntailmentLabel.UNSUPPORTED),
            ],
            lambda _claim, _evidence: 0.5,
            scorer_id="entailment-v1",
            semantic_leakage_guard=VersionedSemanticLeakageGuard(
                scorer=failing_semantic_scorer,
                scorer_id="multilingual-nli:v3",
                model_revision_sha256="c" * 64,
            ),
        )

    assert secret not in str(caught.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scorer": None, "scorer_id": "nli-v1", "model_revision_sha256": "c" * 64},
        {
            "scorer": lambda *_: 0.1,
            "scorer_id": "bad id",
            "model_revision_sha256": "c" * 64,
        },
        {
            "scorer": lambda *_: 0.1,
            "scorer_id": "nli-v1",
            "model_revision_sha256": "latest",
        },
    ],
)
def test_semantic_guard_rejects_unversioned_configuration(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        VersionedSemanticLeakageGuard(**kwargs)  # type: ignore[arg-type]


def _slice_metadata(*case_ids: str) -> dict[str, EntailmentSliceMetadata]:
    return {
        case_id: EntailmentSliceMetadata(language="tr", document_domain="finance")
        for case_id in case_ids
    }


def test_privacy_reduced_drift_slices_report_metric_deltas_above_floor() -> None:
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
        "Claim c1": 0.9,
        "Claim c2": 0.8,
        "Claim c3": 0.2,
        "Claim c4": 0.1,
        "Claim e1": 0.9,
        "Claim e2": 0.3,
        "Claim e3": 0.85,
        "Claim e4": 0.1,
    }

    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        scorer_for(scores),
        scorer_id="drift-slice-v1",
        drift_slice_metadata=_slice_metadata(
            "c1", "c2", "c3", "c4", "e1", "e2", "e3", "e4"
        ),
        drift_slice_policy=DriftSlicePolicy(
            min_pairs_per_split=4,
            min_pairs_per_label_per_split=1,
        ),
    )

    assert report.schema_version == "entailment-diagnostic/v4"
    assert report.drift_slice_version == "privacy-reduced-language-domain-length/v1"
    assert len(report.drift_slice_policy_sha256 or "") == 64
    assert len(report.slice_assignment_sha256 or "") == 64
    assert [item.dimension for item in report.drift_slices] == [
        "language",
        "document_domain",
        "claim_length_utf8",
        "evidence_length_utf8",
    ]
    language_slice = report.drift_slices[0]
    assert language_slice.value == "tr"
    assert language_slice.metrics_reported
    assert language_slice.calibration is not None
    assert language_slice.evaluation is not None
    assert language_slice.calibration.balanced_accuracy == 1.0
    assert language_slice.evaluation.balanced_accuracy == 0.5
    assert language_slice.balanced_accuracy_delta == -0.5
    assert "Claim c1" not in repr(report.to_audit_dict())


def test_slice_metrics_are_suppressed_below_sample_floor() -> None:
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
    metadata = {
        case_id: EntailmentSliceMetadata(
            language="tr" if case_id[-1] in {"1", "3"} else "en",
            document_domain="finance",
        )
        for case_id in ("c1", "c2", "c3", "c4", "e1", "e2", "e3", "e4")
    }

    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        lambda claim, _evidence: 0.9 if claim[-1] in {"1", "2"} else 0.1,
        scorer_id="drift-slice-v1",
        drift_slice_metadata=metadata,
        drift_slice_policy=DriftSlicePolicy(
            min_pairs_per_split=4,
            min_pairs_per_label_per_split=1,
        ),
    )

    language_slices = [
        item for item in report.drift_slices if item.dimension == "language"
    ]
    assert {item.value for item in language_slices} == {"en", "tr"}
    assert all(item.calibration_pairs == 2 for item in language_slices)
    assert all(item.evaluation_pairs == 2 for item in language_slices)
    assert all(not item.metrics_reported for item in language_slices)
    assert all(item.calibration is None for item in language_slices)
    assert all(item.balanced_accuracy_delta is None for item in language_slices)


def test_slice_label_floor_suppresses_single_label_metrics() -> None:
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
    metadata = {
        case_id: EntailmentSliceMetadata(
            language="tr" if case_id[-1] in {"1", "2"} else "en",
            document_domain="finance",
        )
        for case_id in ("c1", "c2", "c3", "c4", "e1", "e2", "e3", "e4")
    }

    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        lambda claim, _evidence: 0.9 if claim[-1] in {"1", "2"} else 0.1,
        scorer_id="drift-slice-v1",
        drift_slice_metadata=metadata,
        drift_slice_policy=DriftSlicePolicy(
            min_pairs_per_split=2,
            min_pairs_per_label_per_split=1,
        ),
    )

    language_slices = [
        item for item in report.drift_slices if item.dimension == "language"
    ]
    assert all(not item.metrics_reported for item in language_slices)


def test_drift_slice_metadata_must_exactly_cover_both_splits_before_scoring() -> None:
    scorer_called = False

    def scorer(_claim: str, _evidence: str) -> float:
        nonlocal scorer_called
        scorer_called = True
        return 0.5

    with pytest.raises(EntailmentEvaluationError, match="exactly cover"):
        evaluate_entailment_diagnostic(
            [
                pair("c1", EntailmentLabel.SUPPORTED),
                pair("c2", EntailmentLabel.UNSUPPORTED),
            ],
            [
                pair("e1", EntailmentLabel.SUPPORTED),
                pair("e2", EntailmentLabel.UNSUPPORTED),
            ],
            scorer,
            scorer_id="drift-slice-v1",
            drift_slice_metadata=_slice_metadata("c1", "c2", "e1"),
            drift_slice_policy=DriftSlicePolicy(
                min_pairs_per_split=2,
                min_pairs_per_label_per_split=1,
            ),
        )

    assert not scorer_called


def test_drift_slice_category_limit_fails_before_scoring() -> None:
    scorer_called = False

    def scorer(_claim: str, _evidence: str) -> float:
        nonlocal scorer_called
        scorer_called = True
        return 0.5

    metadata = {
        "c1": EntailmentSliceMetadata(language="tr", document_domain="finance"),
        "c2": EntailmentSliceMetadata(language="en", document_domain="finance"),
        "e1": EntailmentSliceMetadata(language="tr", document_domain="finance"),
        "e2": EntailmentSliceMetadata(language="en", document_domain="finance"),
    }
    with pytest.raises(EntailmentEvaluationError, match="language.*category limit"):
        evaluate_entailment_diagnostic(
            [
                pair("c1", EntailmentLabel.SUPPORTED),
                pair("c2", EntailmentLabel.UNSUPPORTED),
            ],
            [
                pair("e1", EntailmentLabel.SUPPORTED),
                pair("e2", EntailmentLabel.UNSUPPORTED),
            ],
            scorer,
            scorer_id="drift-slice-v1",
            drift_slice_metadata=metadata,
            drift_slice_policy=DriftSlicePolicy(
                min_pairs_per_split=2,
                min_pairs_per_label_per_split=1,
                max_values_per_dimension=1,
            ),
        )

    assert not scorer_called


def test_slice_assignment_and_policy_are_bound_to_report_digest() -> None:
    calibration = [
        pair("c1", EntailmentLabel.SUPPORTED),
        pair("c2", EntailmentLabel.UNSUPPORTED),
    ]
    evaluation = [
        pair("e1", EntailmentLabel.SUPPORTED),
        pair("e2", EntailmentLabel.UNSUPPORTED),
    ]
    base_metadata = _slice_metadata("c1", "c2", "e1", "e2")
    changed_metadata = dict(base_metadata)
    changed_metadata["e1"] = EntailmentSliceMetadata(
        language="en", document_domain="finance"
    )

    def evaluate(metadata, policy):
        return evaluate_entailment_diagnostic(
            calibration,
            evaluation,
            lambda claim, _evidence: 0.9 if claim.endswith("1") else 0.1,
            scorer_id="drift-slice-v1",
            drift_slice_metadata=metadata,
            drift_slice_policy=policy,
        )

    first = evaluate(
        base_metadata,
        DriftSlicePolicy(
            min_pairs_per_split=2,
            min_pairs_per_label_per_split=1,
        ),
    )
    reassigned = evaluate(
        changed_metadata,
        DriftSlicePolicy(
            min_pairs_per_split=2,
            min_pairs_per_label_per_split=1,
        ),
    )
    rebucketed = evaluate(
        base_metadata,
        DriftSlicePolicy(
            min_pairs_per_split=2,
            min_pairs_per_label_per_split=1,
            claim_length_utf8_boundaries=(8, 32),
        ),
    )

    assert first.slice_assignment_sha256 != reassigned.slice_assignment_sha256
    assert first.report_sha256 != reassigned.report_sha256
    assert first.drift_slice_policy_sha256 != rebucketed.drift_slice_policy_sha256
    assert first.report_sha256 != rebucketed.report_sha256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_pairs_per_split": True},
        {"min_pairs_per_split": 2, "min_pairs_per_label_per_split": 2},
        {"claim_length_utf8_boundaries": ()},
        {"claim_length_utf8_boundaries": (64, 32)},
        {"evidence_length_utf8_boundaries": (0,)},
        {"max_values_per_dimension": 0},
    ],
)
def test_drift_slice_policy_rejects_unsafe_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        DriftSlicePolicy(**kwargs)


@pytest.mark.parametrize(
    "metadata",
    [
        {"language": "", "document_domain": "finance"},
        {"language": "tr", "document_domain": "customer notes"},
        {"language": "tr", "document_domain": "x" * 33},
    ],
)
def test_drift_slice_metadata_rejects_unsafe_categories(metadata) -> None:
    with pytest.raises(ValueError):
        EntailmentSliceMetadata(**metadata)
