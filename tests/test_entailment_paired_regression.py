from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from src.entailment_baseline import MetricRegressionThresholds
from src.entailment_eval import (
    EntailmentLabel,
    EntailmentPair,
    EntailmentPolicy,
    evaluate_entailment_diagnostic,
)
from src.entailment_paired_regression import (
    PAIRED_BOOTSTRAP_METHOD_VERSION,
    PAIRED_COMPARISON_SCHEMA_VERSION,
    SCORE_LEDGER_SCHEMA_VERSION,
    PairedBootstrapPolicy,
    PairedRegressionError,
    compare_paired_score_ledgers,
    create_evaluation_score_ledger,
    load_evaluation_score_ledger,
)


_POLICY = EntailmentPolicy(
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


def _ledger(
    *,
    pairs_per_label: int = 20,
    degraded_supported: int = 0,
    scorer_id: str = "nli-v1",
    claim_suffix: str = "",
    uncommitted_score: float | None = None,
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
    if claim_suffix:
        evaluation[0] = replace(evaluation[0], claim=evaluation[0].claim + claim_suffix)
    score_by_claim: dict[str, float] = {}
    score_by_case_id: dict[str, float] = {}
    for item in calibration:
        score_by_claim[item.claim] = (
            0.9 if item.label is EntailmentLabel.SUPPORTED else 0.1
        )
    supported_seen = 0
    for item in evaluation:
        if item.label is EntailmentLabel.SUPPORTED:
            score = 0.2 if supported_seen < degraded_supported else 0.9
            supported_seen += 1
        else:
            score = 0.1
        score_by_claim[item.claim] = score
        score_by_case_id[item.case_id] = score
    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        lambda claim, _evidence: score_by_claim[claim],
        scorer_id=scorer_id,
        policy=_POLICY,
    )
    if uncommitted_score is not None:
        score_by_case_id[evaluation[0].case_id] = uncommitted_score
    return create_evaluation_score_ledger(report, evaluation, score_by_case_id)


def _thresholds(max_drop: float = 0.05) -> MetricRegressionThresholds:
    return MetricRegressionThresholds(
        max_supported_precision_drop=max_drop,
        max_supported_recall_drop=max_drop,
        max_unsupported_recall_drop=max_drop,
        max_balanced_accuracy_drop=max_drop,
    )


def _bootstrap_policy(
    max_drop: float = 0.05, *, resamples: int = 1_000
) -> PairedBootstrapPolicy:
    return PairedBootstrapPolicy(
        confidence_level=0.95,
        resamples=resamples,
        thresholds=_thresholds(max_drop),
    )


def test_score_ledger_is_bound_deterministic_and_content_minimizing() -> None:
    first = _ledger()
    second = _ledger()

    assert first == second
    assert first.schema_version == SCORE_LEDGER_SCHEMA_VERSION
    assert first.evaluation_sha256
    assert first.policy_sha256
    assert len(first.rows) == 40
    serialized = json.dumps(first.to_audit_dict(), sort_keys=True)
    assert "Control assertion" not in serialized
    assert "eval-supported" not in serialized


def test_score_ledger_round_trip_rejects_tampering_and_extra_fields() -> None:
    ledger = _ledger()
    document = json.loads(json.dumps(ledger.to_audit_dict()))

    assert load_evaluation_score_ledger(document) == ledger

    tampered = json.loads(json.dumps(document))
    tampered["rows"][0]["score"] = 0.42
    with pytest.raises(PairedRegressionError, match="evaluation commitment mismatch"):
        load_evaluation_score_ledger(tampered)

    document["unexpected"] = True
    with pytest.raises(PairedRegressionError, match="malformed"):
        load_evaluation_score_ledger(document)


def test_ledger_creation_rejects_scores_not_committed_by_report() -> None:
    with pytest.raises(PairedRegressionError, match="do not match report commitment"):
        _ledger(uncommitted_score=0.37)


def test_identical_population_passes_with_reproducible_fwer_evidence() -> None:
    baseline = _ledger(scorer_id="baseline-v1")
    candidate = _ledger(scorer_id="candidate-v1")
    policy = _bootstrap_policy()

    first = compare_paired_score_ledgers(baseline, candidate, policy)
    second = compare_paired_score_ledgers(baseline, candidate, policy)

    assert first == second
    assert first.schema_version == PAIRED_COMPARISON_SCHEMA_VERSION
    assert first.bootstrap_method_version == PAIRED_BOOTSTRAP_METHOD_VERSION
    assert first.family_wise_method == "bonferroni"
    assert first.hypotheses == 4
    assert first.per_hypothesis_confidence_level == pytest.approx(0.9875)
    assert first.meets_policy
    assert first.failure_reasons == ()
    assert all(item.point_delta == 0 for item in first.metrics)
    assert len(first.bootstrap_seed_sha256) == 64
    assert len(first.comparison_sha256) == 64


def test_confident_paired_regression_fails_closed() -> None:
    baseline = _ledger(scorer_id="baseline-v1")
    candidate = _ledger(
        degraded_supported=20,
        scorer_id="candidate-v2",
    )

    comparison = compare_paired_score_ledgers(
        baseline,
        candidate,
        _bootstrap_policy(),
    )

    assert not comparison.meets_policy
    assert "global_supported_recall_regressed" in comparison.failure_reasons
    assert "global_balanced_accuracy_regressed" in comparison.failure_reasons
    supported_recall = next(
        item for item in comparison.metrics if item.metric == "supported_recall"
    )
    assert supported_recall.point_delta == -1.0
    assert supported_recall.adjusted_delta_upper == -1.0
    assert supported_recall.regressed


def test_unresolved_single_case_drop_does_not_claim_regression() -> None:
    baseline = _ledger(scorer_id="baseline-v1")
    candidate = _ledger(degraded_supported=1, scorer_id="candidate-v2")

    comparison = compare_paired_score_ledgers(
        baseline,
        candidate,
        _bootstrap_policy(max_drop=0.0),
    )

    supported_recall = next(
        item for item in comparison.metrics if item.metric == "supported_recall"
    )
    assert supported_recall.point_delta == pytest.approx(-0.05)
    assert supported_recall.adjusted_delta_upper == 0.0
    assert not supported_recall.regressed


def test_population_or_policy_mismatch_is_rejected() -> None:
    baseline = _ledger()
    different_population = _ledger(claim_suffix=" altered")
    with pytest.raises(PairedRegressionError, match="same sealed cases"):
        compare_paired_score_ledgers(
            baseline,
            different_population,
            _bootstrap_policy(),
        )

    incompatible_policy = replace(baseline, policy_sha256="f" * 64)
    document = {
        key: value
        for key, value in incompatible_policy.to_audit_dict().items()
        if key != "artifact_sha256"
    }
    incompatible_policy = replace(
        incompatible_policy,
        artifact_sha256=hashlib.sha256(
            json.dumps(
                document,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest(),
    )
    with pytest.raises(PairedRegressionError, match="policies are incompatible"):
        compare_paired_score_ledgers(
            baseline,
            incompatible_policy,
            _bootstrap_policy(),
        )


def test_bootstrap_work_is_bounded_before_sampling() -> None:
    baseline = _ledger(pairs_per_label=50, scorer_id="baseline-v1")
    candidate = _ledger(pairs_per_label=50, scorer_id="candidate-v1")

    with pytest.raises(PairedRegressionError, match="draw budget exceeded"):
        compare_paired_score_ledgers(
            baseline,
            candidate,
            _bootstrap_policy(resamples=100_000),
        )


@pytest.mark.parametrize("resamples", [999, 100_001, True])
def test_bootstrap_policy_rejects_unbounded_resample_counts(resamples) -> None:
    with pytest.raises(ValueError, match="resamples"):
        PairedBootstrapPolicy(0.95, resamples, _thresholds())


def test_bootstrap_policy_requires_explicit_supported_fwer_method() -> None:
    with pytest.raises(ValueError, match="bonferroni"):
        PairedBootstrapPolicy(0.95, 1_000, _thresholds(), "none")
