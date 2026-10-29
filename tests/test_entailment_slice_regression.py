from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from src.entailment_baseline import MetricRegressionThresholds
from src.entailment_eval import (
    DriftSlicePolicy,
    EntailmentLabel,
    EntailmentPair,
    EntailmentPolicy,
    EntailmentSliceMetadata,
    evaluate_entailment_diagnostic,
)
from src.entailment_paired_regression import create_evaluation_score_ledger
from src.entailment_slice_assignment import create_evaluation_slice_assignment
from src.entailment_slice_regression import (
    SLICE_BOOTSTRAP_METHOD_VERSION,
    SLICE_COMPARISON_SCHEMA_VERSION,
    PairedSliceBootstrapPolicy,
    SliceRegressionError,
    SliceRegressionRule,
    compare_paired_slice_scores,
)


_ENTAILMENT_POLICY = EntailmentPolicy(
    min_supported_precision=0.0,
    min_supported_recall=0.0,
)
_SLICE_POLICY = DriftSlicePolicy(
    min_pairs_per_split=4,
    min_pairs_per_label_per_split=2,
    claim_length_utf8_boundaries=(64, 128),
    evidence_length_utf8_boundaries=(64, 128),
)


def _case(split: str, label: EntailmentLabel, index: int) -> EntailmentPair:
    case_id = f"{split}-{label.value}-{index:02d}"
    token = hashlib.sha256(case_id.encode()).hexdigest()
    return EntailmentPair(
        case_id=case_id,
        claim=f"Control {token} is satisfied.",
        evidence=f"Evidence {token[::-1]} records the reviewed outcome.",
        label=label,
    )


def _evidence(
    *,
    degraded_language: str | None = None,
    metadata_language_shift: bool = False,
    imbalanced_language: bool = False,
):
    calibration = [
        _case("cal", label, index) for label in EntailmentLabel for index in range(20)
    ]
    evaluation = [
        _case("eval", label, index) for label in EntailmentLabel for index in range(20)
    ]
    all_pairs = [*calibration, *evaluation]
    metadata = {
        pair.case_id: EntailmentSliceMetadata(
            language=(
                "en"
                if imbalanced_language
                and (
                    (pair.label is EntailmentLabel.SUPPORTED and index % 4 != 3)
                    or (pair.label is EntailmentLabel.UNSUPPORTED and index % 4 == 0)
                )
                else "en"
                if not imbalanced_language and index % 2 == 0
                else "tr"
            ),
            document_domain="security",
        )
        for index, pair in enumerate(all_pairs)
    }
    if metadata_language_shift:
        metadata[evaluation[0].case_id] = EntailmentSliceMetadata(
            language="tr",
            document_domain="security",
        )
    scores_by_claim: dict[str, float] = {}
    scores_by_case_id: dict[str, float] = {}
    for pair in all_pairs:
        score = 0.9 if pair.label is EntailmentLabel.SUPPORTED else 0.1
        if (
            pair in evaluation
            and pair.label is EntailmentLabel.SUPPORTED
            and metadata[pair.case_id].language == degraded_language
        ):
            score = 0.1
        scores_by_claim[pair.claim] = score
        if pair in evaluation:
            scores_by_case_id[pair.case_id] = score
    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        lambda claim, _evidence: scores_by_claim[claim],
        scorer_id=f"nli-{degraded_language or 'baseline'}",
        policy=_ENTAILMENT_POLICY,
        drift_slice_metadata=metadata,
        drift_slice_policy=_SLICE_POLICY,
    )
    return (
        create_evaluation_score_ledger(report, evaluation, scores_by_case_id),
        create_evaluation_slice_assignment(
            report,
            calibration,
            evaluation,
            metadata,
            _SLICE_POLICY,
        ),
    )


def _thresholds(max_drop: float = 0.05) -> MetricRegressionThresholds:
    return MetricRegressionThresholds(
        max_supported_precision_drop=max_drop,
        max_supported_recall_drop=max_drop,
        max_unsupported_recall_drop=max_drop,
        max_balanced_accuracy_drop=max_drop,
    )


def _rule(dimension: str, value: str, *, min_pairs: int = 20) -> SliceRegressionRule:
    return SliceRegressionRule(
        dimension=dimension,
        value=value,
        min_pairs=min_pairs,
        min_pairs_per_label=min_pairs // 2,
        thresholds=_thresholds(),
    )


def _policy(*rules: SliceRegressionRule, resamples: int = 1_000):
    return PairedSliceBootstrapPolicy(
        confidence_level=0.95,
        resamples=resamples,
        rules=tuple(rules) or (_rule("language", "en"), _rule("language", "tr")),
    )


def test_identical_slices_pass_with_reproducible_fwer_evidence() -> None:
    baseline, baseline_assignment = _evidence()
    candidate, candidate_assignment = _evidence()
    policy = _policy()

    first = compare_paired_slice_scores(
        baseline,
        candidate,
        baseline_assignment,
        candidate_assignment,
        policy,
    )
    second = compare_paired_slice_scores(
        baseline,
        candidate,
        baseline_assignment,
        candidate_assignment,
        policy,
    )

    assert first == second
    assert first.schema_version == SLICE_COMPARISON_SCHEMA_VERSION
    assert first.bootstrap_method_version == SLICE_BOOTSTRAP_METHOD_VERSION
    assert first.hypotheses == 8
    assert first.per_hypothesis_confidence_level == pytest.approx(0.99375)
    assert first.bootstrap_draws == 40_000
    assert first.meets_policy
    assert first.failure_reasons == ()
    assert [(item.dimension, item.value, item.pairs) for item in first.slices] == [
        ("language", "en", 20),
        ("language", "tr", 20),
    ]
    assert len(first.comparison_sha256) == 64


def test_regression_is_localized_to_reviewed_slice() -> None:
    baseline, baseline_assignment = _evidence()
    candidate, candidate_assignment = _evidence(degraded_language="tr")

    result = compare_paired_slice_scores(
        baseline,
        candidate,
        baseline_assignment,
        candidate_assignment,
        _policy(),
    )

    assert not result.meets_policy
    assert "slice:language=tr:supported_recall:regressed" in result.failure_reasons
    assert "slice:language=tr:balanced_accuracy:regressed" in result.failure_reasons
    en_slice, tr_slice = result.slices
    assert all(not metric.regressed for metric in en_slice.metrics)
    tr_recall = next(
        metric for metric in tr_slice.metrics if metric.metric == "supported_recall"
    )
    assert tr_recall.point_delta == -1.0
    assert tr_recall.adjusted_delta_upper == -1.0


def test_required_slice_and_per_label_floors_fail_closed() -> None:
    baseline, baseline_assignment = _evidence()
    candidate, candidate_assignment = _evidence()

    with pytest.raises(SliceRegressionError, match="has 20 pairs; requires 22"):
        compare_paired_slice_scores(
            baseline,
            candidate,
            baseline_assignment,
            candidate_assignment,
            _policy(_rule("language", "en", min_pairs=22)),
        )

    baseline, baseline_assignment = _evidence(imbalanced_language=True)
    candidate, candidate_assignment = _evidence(imbalanced_language=True)
    with pytest.raises(SliceRegressionError, match="label floor is not met"):
        compare_paired_slice_scores(
            baseline,
            candidate,
            baseline_assignment,
            candidate_assignment,
            _policy(_rule("language", "en")),
        )


def test_candidate_slice_reassignment_is_rejected() -> None:
    baseline, baseline_assignment = _evidence()
    candidate, candidate_assignment = _evidence(metadata_language_shift=True)

    with pytest.raises(SliceRegressionError, match="slice assignments differ"):
        compare_paired_slice_scores(
            baseline,
            candidate,
            baseline_assignment,
            candidate_assignment,
            _policy(),
        )


def test_bootstrap_work_is_bounded_before_sampling() -> None:
    baseline, baseline_assignment = _evidence()
    candidate, candidate_assignment = _evidence()
    rules = (
        _rule("document_domain", "security", min_pairs=40),
        _rule("language", "en"),
        _rule("language", "tr"),
    )

    with pytest.raises(SliceRegressionError, match="draw budget exceeded"):
        compare_paired_slice_scores(
            baseline,
            candidate,
            baseline_assignment,
            candidate_assignment,
            _policy(*rules, resamples=100_000),
        )


def test_policy_requires_canonical_unique_reviewed_rules() -> None:
    with pytest.raises(ValueError, match="unique and sorted"):
        _policy(_rule("language", "tr"), _rule("language", "en"))
    with pytest.raises(ValueError, match="unique and sorted"):
        _policy(_rule("language", "en"), _rule("language", "en"))
    with pytest.raises(ValueError, match="dimension must be"):
        _rule("tenant", "example")
    with pytest.raises(ValueError, match="family_wise_method"):
        replace(_policy(), family_wise_method="none")
