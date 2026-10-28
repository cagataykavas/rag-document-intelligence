from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from src.entailment_baseline import (
    BASELINE_SCHEMA_VERSION,
    COMPARISON_SCHEMA_VERSION,
    UNCERTAINTY_METHOD_VERSION,
    EntailmentBaselineError,
    EntailmentRegressionPolicy,
    MetricRegressionThresholds,
    SliceRegressionRule,
    compare_entailment_baseline,
    create_entailment_baseline,
    load_entailment_baseline,
)
from src.entailment_eval import (
    DriftSlicePolicy,
    EntailmentLabel,
    EntailmentPair,
    EntailmentPolicy,
    EntailmentSliceMetadata,
    evaluate_entailment_diagnostic,
)


_ENTAILMENT_POLICY = EntailmentPolicy(
    min_supported_precision=0.0,
    min_supported_recall=0.0,
)


def _case(split: str, index: int, label: EntailmentLabel) -> EntailmentPair:
    case_id = f"{split}-{index:03d}"
    token = hashlib.sha256(case_id.encode()).hexdigest()
    return EntailmentPair(
        case_id=case_id,
        claim=f"Control assertion {token} requires reviewed evidence.",
        evidence=f"Evidence record {token[::-1]} defines the control outcome.",
        label=label,
    )


def _report(
    *,
    pairs_per_label: int = 20,
    degraded_supported: int = 0,
    language: str = "tr",
    scorer_id: str = "nli-candidate-v1",
    entailment_policy: EntailmentPolicy = _ENTAILMENT_POLICY,
    claim_boundaries: tuple[int, ...] = (64, 256),
):
    calibration = [
        _case("cal-supported", index, EntailmentLabel.SUPPORTED)
        for index in range(pairs_per_label)
    ] + [
        _case("cal-unsupported", index, EntailmentLabel.UNSUPPORTED)
        for index in range(pairs_per_label)
    ]
    evaluation = [
        _case("eval-supported", index, EntailmentLabel.SUPPORTED)
        for index in range(pairs_per_label)
    ] + [
        _case("eval-unsupported", index, EntailmentLabel.UNSUPPORTED)
        for index in range(pairs_per_label)
    ]
    scores: dict[str, float] = {}
    for item in calibration:
        scores[item.claim] = 0.9 if item.label is EntailmentLabel.SUPPORTED else 0.1
    for index, item in enumerate(evaluation):
        if item.label is EntailmentLabel.SUPPORTED:
            scores[item.claim] = 0.2 if index < degraded_supported else 0.9
        else:
            scores[item.claim] = 0.1
    metadata = {
        item.case_id: EntailmentSliceMetadata(
            language=language,
            document_domain="finance",
        )
        for item in calibration + evaluation
    }
    return evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        lambda claim, _evidence: scores[claim],
        scorer_id=scorer_id,
        policy=entailment_policy,
        drift_slice_metadata=metadata,
        drift_slice_policy=DriftSlicePolicy(
            min_pairs_per_split=2 * pairs_per_label,
            min_pairs_per_label_per_split=pairs_per_label,
            claim_length_utf8_boundaries=claim_boundaries,
        ),
    )


def _thresholds(max_drop: float = 0.05) -> MetricRegressionThresholds:
    return MetricRegressionThresholds(
        max_supported_precision_drop=max_drop,
        max_supported_recall_drop=max_drop,
        max_unsupported_recall_drop=max_drop,
        max_balanced_accuracy_drop=max_drop,
    )


def _comparison_policy(baseline, max_drop: float = 0.05):
    thresholds = _thresholds(max_drop)
    return EntailmentRegressionPolicy(
        confidence_level=0.95,
        global_thresholds=thresholds,
        slice_rules=tuple(
            SliceRegressionRule(item.dimension, item.value, thresholds)
            for item in baseline.slices
        ),
    )


def _baseline(report=None):
    return create_entailment_baseline(
        report or _report(scorer_id="nli-baseline-v1"),
        baseline_id="sealed-holdout",
        baseline_version=1,
    )


def test_baseline_is_deterministic_content_minimizing_and_versioned() -> None:
    report = _report(scorer_id="nli-baseline-v1")
    first = _baseline(report)
    second = _baseline(report)

    assert first == second
    assert first.schema_version == BASELINE_SCHEMA_VERSION
    assert first.source_report_sha256 == report.report_sha256
    assert first.slice_assignment_sha256 == report.slice_assignment_sha256
    assert len(first.artifact_sha256) == 64
    assert len(first.slices) == 4
    serialized = json.dumps(first.to_audit_dict(), sort_keys=True)
    assert "Control assertion" not in serialized
    assert "eval-supported" not in serialized


def test_identical_candidate_passes_with_versioned_uncertainty_intervals() -> None:
    candidate = _report()
    baseline = _baseline(candidate)

    comparison = compare_entailment_baseline(
        baseline,
        candidate,
        _comparison_policy(baseline),
    )

    assert comparison.schema_version == COMPARISON_SCHEMA_VERSION
    assert comparison.uncertainty_method_version == UNCERTAINTY_METHOD_VERSION
    assert comparison.meets_policy
    assert comparison.failure_reasons == ()
    assert all(item.point_delta == 0 for item in comparison.global_metrics)
    assert all(not item.regressed for item in comparison.global_metrics)
    assert len(comparison.comparison_sha256) == 64


def test_json_round_trip_loader_rejects_tampering_and_extra_fields() -> None:
    baseline = _baseline()
    document = json.loads(json.dumps(baseline.to_audit_dict()))

    assert load_entailment_baseline(document) == baseline

    tampered = json.loads(json.dumps(document))
    tampered["baseline_version"] = 2
    with pytest.raises(EntailmentBaselineError, match="artifact digest mismatch"):
        load_entailment_baseline(tampered)

    document["unexpected"] = "field"
    with pytest.raises(EntailmentBaselineError, match="malformed"):
        load_entailment_baseline(document)


def test_confident_global_and_slice_regression_fails() -> None:
    baseline = _baseline()
    candidate = _report(degraded_supported=20, scorer_id="nli-candidate-v2")

    comparison = compare_entailment_baseline(
        baseline,
        candidate,
        _comparison_policy(baseline),
    )

    assert not comparison.meets_policy
    assert "global_supported_recall_regressed" in comparison.failure_reasons
    assert "global_balanced_accuracy_regressed" in comparison.failure_reasons
    assert all(item.regressed for item in comparison.slices)
    supported_recall = next(
        item for item in comparison.global_metrics if item.metric == "supported_recall"
    )
    assert supported_recall.point_delta == -1.0
    assert supported_recall.delta_upper < -supported_recall.max_drop


def test_small_sample_overlap_does_not_claim_a_regression() -> None:
    baseline = _baseline(_report(pairs_per_label=2, scorer_id="small-baseline"))
    candidate = _report(
        pairs_per_label=2,
        degraded_supported=1,
        scorer_id="small-candidate",
    )

    comparison = compare_entailment_baseline(
        baseline,
        candidate,
        _comparison_policy(baseline, max_drop=0.0),
    )

    supported_recall = next(
        item for item in comparison.global_metrics if item.metric == "supported_recall"
    )
    assert supported_recall.point_delta == -0.5
    assert supported_recall.delta_upper > 0
    assert not supported_recall.regressed
    assert comparison.meets_policy


def test_missing_and_unreviewed_candidate_slices_fail_closed() -> None:
    baseline = _baseline()
    candidate = _report(language="en")

    comparison = compare_entailment_baseline(
        baseline,
        candidate,
        _comparison_policy(baseline),
    )

    assert not comparison.meets_policy
    assert "slice_language_tr_metrics_unavailable" in comparison.failure_reasons
    assert "slice_language_en_unreviewed" in comparison.failure_reasons


def test_regression_policy_must_cover_baseline_slices_exactly() -> None:
    baseline = _baseline()
    incomplete = EntailmentRegressionPolicy(
        confidence_level=0.95,
        global_thresholds=_thresholds(),
        slice_rules=(),
    )

    with pytest.raises(EntailmentBaselineError, match="exactly cover"):
        compare_entailment_baseline(baseline, _report(), incomplete)


def test_tampered_baseline_and_candidate_digests_are_rejected() -> None:
    baseline = _baseline()
    policy = _comparison_policy(baseline)

    with pytest.raises(EntailmentBaselineError, match="artifact digest mismatch"):
        compare_entailment_baseline(
            replace(baseline, baseline_version=2),
            _report(),
            policy,
        )
    with pytest.raises(
        EntailmentBaselineError, match="candidate report digest mismatch"
    ):
        compare_entailment_baseline(
            baseline,
            replace(_report(), scorer_id="tampered"),
            policy,
        )


def test_incompatible_entailment_or_slice_policy_is_rejected() -> None:
    baseline = _baseline()
    different_entailment_policy = _report(
        entailment_policy=EntailmentPolicy(
            min_supported_precision=0.1,
            min_supported_recall=0.0,
        )
    )

    with pytest.raises(EntailmentBaselineError, match="entailment policy"):
        compare_entailment_baseline(
            baseline,
            different_entailment_policy,
            _comparison_policy(baseline),
        )
    different_slice_policy = _report(claim_boundaries=(80, 256))
    with pytest.raises(EntailmentBaselineError, match="drift slice policy"):
        compare_entailment_baseline(
            baseline,
            different_slice_policy,
            _comparison_policy(baseline),
        )


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), True])
def test_metric_drop_limits_are_bounded(value) -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        MetricRegressionThresholds(value, 0.0, 0.0, 0.0)


def test_duplicate_slice_rules_and_invalid_confidence_are_rejected() -> None:
    rule = SliceRegressionRule("language", "tr", _thresholds())
    with pytest.raises(ValueError, match="duplicate"):
        EntailmentRegressionPolicy(0.95, _thresholds(), (rule, rule))
    with pytest.raises(ValueError, match="confidence_level"):
        EntailmentRegressionPolicy(0.5, _thresholds(), ())


def test_baseline_requires_accepted_sliced_report() -> None:
    report = _report()
    with pytest.raises(EntailmentBaselineError, match="does not meet"):
        _baseline(replace(report, meets_policy=False))

    calibration = [
        _case("plain-cal", index, label)
        for index, label in enumerate(
            (EntailmentLabel.SUPPORTED, EntailmentLabel.UNSUPPORTED)
        )
    ]
    evaluation = [
        _case("plain-eval", index, label)
        for index, label in enumerate(
            (EntailmentLabel.SUPPORTED, EntailmentLabel.UNSUPPORTED)
        )
    ]
    scores = {
        item.claim: 0.9 if item.label is EntailmentLabel.SUPPORTED else 0.1
        for item in calibration + evaluation
    }
    unsliced = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        lambda claim, _evidence: scores[claim],
        scorer_id="plain",
        policy=_ENTAILMENT_POLICY,
    )
    with pytest.raises(
        EntailmentBaselineError, match="no privacy-reduced drift slices"
    ):
        _baseline(unsliced)


def test_comparison_artifact_omits_raw_examples_and_case_ids() -> None:
    baseline = _baseline()
    comparison = compare_entailment_baseline(
        baseline,
        _report(),
        _comparison_policy(baseline),
    )

    serialized = json.dumps(comparison.to_audit_dict(), sort_keys=True)
    assert "Control assertion" not in serialized
    assert "cal-supported" not in serialized
