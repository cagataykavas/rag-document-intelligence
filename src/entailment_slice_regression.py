from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from src.entailment_baseline import MetricRegressionThresholds
from src.entailment_eval import EntailmentLabel, EntailmentMetrics
from src.entailment_paired_regression import EvaluationScoreLedger
from src.entailment_slice_assignment import (
    EvaluationSliceAssignmentArtifact,
    EvaluationSliceScoreRow,
    join_evaluation_slice_assignments,
)

SLICE_COMPARISON_SCHEMA_VERSION = "entailment-paired-slice-comparison/v1"
SLICE_BOOTSTRAP_METHOD_VERSION = "stratified-paired-slice-bootstrap-bonferroni/v1"
_DIMENSIONS = (
    "claim_length_utf8",
    "document_domain",
    "evidence_length_utf8",
    "language",
)
_METRICS = (
    "supported_precision",
    "supported_recall",
    "unsupported_recall",
    "balanced_accuracy",
)
_SLICE_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+:-]{0,63}")
_MAX_RULES = 64
_MAX_BOOTSTRAP_DRAWS = 5_000_000


class SliceRegressionError(ValueError):
    """A paired slice comparison is unsafe or statistically underpowered."""

    code = "paired_entailment_slice_regression_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class SliceRegressionRule:
    dimension: str
    value: str
    min_pairs: int
    min_pairs_per_label: int
    thresholds: MetricRegressionThresholds

    def __post_init__(self) -> None:
        if self.dimension not in _DIMENSIONS:
            raise ValueError(f"dimension must be one of {_DIMENSIONS}")
        if (
            not isinstance(self.value, str)
            or _SLICE_VALUE_RE.fullmatch(self.value) is None
        ):
            raise ValueError("value must be a canonical low-cardinality slice value")
        if type(self.min_pairs) is not int or not 4 <= self.min_pairs <= 100_000:
            raise ValueError("min_pairs must be an integer in [4, 100000]")
        if (
            type(self.min_pairs_per_label) is not int
            or not 2 <= self.min_pairs_per_label <= 50_000
            or 2 * self.min_pairs_per_label > self.min_pairs
        ):
            raise ValueError(
                "min_pairs_per_label must be in [2, 50000] and fit within min_pairs"
            )
        if not isinstance(self.thresholds, MetricRegressionThresholds):
            raise TypeError("thresholds must be MetricRegressionThresholds")


@dataclass(frozen=True, slots=True)
class PairedSliceBootstrapPolicy:
    confidence_level: float
    resamples: int
    rules: tuple[SliceRegressionRule, ...]
    family_wise_method: str = "bonferroni"

    def __post_init__(self) -> None:
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not math.isfinite(self.confidence_level)
            or not 0.8 <= float(self.confidence_level) <= 0.999
        ):
            raise ValueError("confidence_level must be finite and in [0.8, 0.999]")
        if type(self.resamples) is not int or not 1_000 <= self.resamples <= 100_000:
            raise ValueError("resamples must be an integer in [1000, 100000]")
        if not isinstance(self.rules, tuple) or not 1 <= len(self.rules) <= _MAX_RULES:
            raise ValueError("rules must be a tuple containing between 1 and 64 rules")
        if any(not isinstance(rule, SliceRegressionRule) for rule in self.rules):
            raise TypeError("rules must contain SliceRegressionRule values")
        keys = tuple((rule.dimension, rule.value) for rule in self.rules)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("rules must be unique and sorted by dimension then value")
        if self.family_wise_method != "bonferroni":
            raise ValueError("family_wise_method must be 'bonferroni'")


@dataclass(frozen=True, slots=True)
class SliceMetricComparison:
    metric: str
    baseline_value: float
    candidate_value: float
    point_delta: float
    adjusted_delta_upper: float
    max_drop: float
    regressed: bool


@dataclass(frozen=True, slots=True)
class SliceComparison:
    dimension: str
    value: str
    pairs: int
    supported_pairs: int
    unsupported_pairs: int
    metrics: tuple[SliceMetricComparison, ...]


@dataclass(frozen=True, slots=True)
class PairedSliceRegressionComparison:
    schema_version: str
    bootstrap_method_version: str
    baseline_ledger_sha256: str
    candidate_ledger_sha256: str
    baseline_assignment_sha256: str
    candidate_assignment_sha256: str
    evaluation_assignment_sha256: str
    confidence_level: float
    family_wise_method: str
    hypotheses: int
    per_hypothesis_confidence_level: float
    resamples: int
    bootstrap_draws: int
    bootstrap_seed_sha256: str
    policy_sha256: str
    slices: tuple[SliceComparison, ...]
    meets_policy: bool
    failure_reasons: tuple[str, ...]
    comparison_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _metrics(
    rows: Sequence[EvaluationSliceScoreRow], threshold: float
) -> EntailmentMetrics:
    true_supported = false_supported = true_unsupported = false_unsupported = 0
    for row in rows:
        predicted_supported = row.score >= threshold
        if row.label == EntailmentLabel.SUPPORTED.value:
            if predicted_supported:
                true_supported += 1
            else:
                false_unsupported += 1
        elif predicted_supported:
            false_supported += 1
        else:
            true_unsupported += 1
    supported_precision = (
        true_supported / (true_supported + false_supported)
        if true_supported + false_supported
        else 0.0
    )
    supported_recall = true_supported / (true_supported + false_unsupported)
    unsupported_recall = true_unsupported / (true_unsupported + false_supported)
    return EntailmentMetrics(
        pairs=len(rows),
        true_supported=true_supported,
        false_supported=false_supported,
        true_unsupported=true_unsupported,
        false_unsupported=false_unsupported,
        supported_precision=supported_precision,
        supported_recall=supported_recall,
        unsupported_recall=unsupported_recall,
        balanced_accuracy=(supported_recall + unsupported_recall) / 2,
    )


class _DeterministicSampler:
    def __init__(self, seed_hex: str) -> None:
        self._seed = bytes.fromhex(seed_hex)
        self._counter = 0

    def index(self, size: int) -> int:
        limit = (1 << 256) - ((1 << 256) % size)
        while True:
            value = int.from_bytes(
                hashlib.sha256(self._seed + self._counter.to_bytes(16, "big")).digest(),
                "big",
            )
            self._counter += 1
            if value < limit:
                return value % size


def _upper_quantile(values: list[float], probability: float) -> float:
    values.sort()
    return values[min(len(values) - 1, math.ceil(probability * len(values)) - 1)]


def _joined_rows(
    ledger: EvaluationScoreLedger,
    assignment: EvaluationSliceAssignmentArtifact,
    name: str,
) -> tuple[EvaluationSliceScoreRow, ...]:
    try:
        return join_evaluation_slice_assignments(ledger, assignment)
    except (TypeError, ValueError) as exc:
        raise SliceRegressionError(f"{name} evidence is invalid: {exc}") from exc


def compare_paired_slice_scores(
    baseline: EvaluationScoreLedger,
    candidate: EvaluationScoreLedger,
    baseline_assignment: EvaluationSliceAssignmentArtifact,
    candidate_assignment: EvaluationSliceAssignmentArtifact,
    policy: PairedSliceBootstrapPolicy,
) -> PairedSliceRegressionComparison:
    """Run reviewed per-slice paired bootstraps under one FWER budget."""

    if not isinstance(policy, PairedSliceBootstrapPolicy):
        raise TypeError("policy must be a PairedSliceBootstrapPolicy")
    baseline_rows = _joined_rows(baseline, baseline_assignment, "baseline")
    candidate_rows = _joined_rows(candidate, candidate_assignment, "candidate")
    if (
        baseline.report_schema_version != candidate.report_schema_version
        or baseline.policy_sha256 != candidate.policy_sha256
        or baseline_assignment.drift_slice_policy_sha256
        != candidate_assignment.drift_slice_policy_sha256
    ):
        raise SliceRegressionError("baseline and candidate policies are incompatible")
    baseline_population = tuple(
        (row.case_id_sha256, row.content_sha256, row.label) for row in baseline_rows
    )
    candidate_population = tuple(
        (row.case_id_sha256, row.content_sha256, row.label) for row in candidate_rows
    )
    if baseline_population != candidate_population:
        raise SliceRegressionError(
            "baseline and candidate do not cover the same sealed cases"
        )
    baseline_slices = tuple(
        tuple(getattr(row, dimension) for dimension in _DIMENSIONS)
        for row in baseline_rows
    )
    candidate_slices = tuple(
        tuple(getattr(row, dimension) for dimension in _DIMENSIONS)
        for row in candidate_rows
    )
    if (
        baseline_slices != candidate_slices
        or baseline_assignment.evaluation_assignment_sha256
        != candidate_assignment.evaluation_assignment_sha256
    ):
        raise SliceRegressionError("baseline and candidate slice assignments differ")

    selected: list[tuple[SliceRegressionRule, list[int], list[list[int]]]] = []
    total_pairs = 0
    for rule in policy.rules:
        indices = [
            index
            for index, row in enumerate(baseline_rows)
            if getattr(row, rule.dimension) == rule.value
        ]
        strata = [
            [index for index in indices if baseline_rows[index].label == label.value]
            for label in EntailmentLabel
        ]
        counts = tuple(len(stratum) for stratum in strata)
        if len(indices) < rule.min_pairs:
            raise SliceRegressionError(
                f"slice {rule.dimension}={rule.value} has {len(indices)} pairs; "
                f"requires {rule.min_pairs}"
            )
        if any(count < rule.min_pairs_per_label for count in counts):
            raise SliceRegressionError(
                f"slice {rule.dimension}={rule.value} label floor is not met: "
                f"{counts}; requires {rule.min_pairs_per_label} per label"
            )
        selected.append((rule, indices, strata))
        total_pairs += len(indices)
    bootstrap_draws = total_pairs * policy.resamples
    if bootstrap_draws > _MAX_BOOTSTRAP_DRAWS:
        raise SliceRegressionError(
            f"bootstrap draw budget exceeded: {bootstrap_draws} > {_MAX_BOOTSTRAP_DRAWS}"
        )

    hypotheses = len(policy.rules) * len(_METRICS)
    alpha = 1 - float(policy.confidence_level)
    per_hypothesis_confidence = 1 - alpha / hypotheses
    policy_sha256 = _sha256(asdict(policy))
    seed_sha256 = _sha256(
        {
            "baseline_ledger_sha256": baseline.artifact_sha256,
            "candidate_ledger_sha256": candidate.artifact_sha256,
            "baseline_assignment_sha256": baseline_assignment.artifact_sha256,
            "candidate_assignment_sha256": candidate_assignment.artifact_sha256,
            "method": SLICE_BOOTSTRAP_METHOD_VERSION,
            "policy_sha256": policy_sha256,
        }
    )
    sampler = _DeterministicSampler(seed_sha256)
    comparisons: list[SliceComparison] = []
    failure_reasons: list[str] = []
    for rule, indices, strata in selected:
        baseline_slice = [baseline_rows[index] for index in indices]
        candidate_slice = [candidate_rows[index] for index in indices]
        baseline_metrics = _metrics(baseline_slice, baseline.threshold)
        candidate_metrics = _metrics(candidate_slice, candidate.threshold)
        deltas = {metric: [] for metric in _METRICS}
        for _ in range(policy.resamples):
            sample_indices = [
                stratum[sampler.index(len(stratum))]
                for stratum in strata
                for _ in range(len(stratum))
            ]
            baseline_sample = [baseline_rows[index] for index in sample_indices]
            candidate_sample = [candidate_rows[index] for index in sample_indices]
            baseline_sample_metrics = _metrics(baseline_sample, baseline.threshold)
            candidate_sample_metrics = _metrics(candidate_sample, candidate.threshold)
            for metric in _METRICS:
                deltas[metric].append(
                    getattr(candidate_sample_metrics, metric)
                    - getattr(baseline_sample_metrics, metric)
                )
        metric_comparisons: list[SliceMetricComparison] = []
        for metric in _METRICS:
            baseline_value = float(getattr(baseline_metrics, metric))
            candidate_value = float(getattr(candidate_metrics, metric))
            adjusted_upper = _upper_quantile(deltas[metric], per_hypothesis_confidence)
            max_drop = rule.thresholds.for_metric(metric)
            regressed = adjusted_upper < -max_drop
            if regressed:
                failure_reasons.append(
                    f"slice:{rule.dimension}={rule.value}:{metric}:regressed"
                )
            metric_comparisons.append(
                SliceMetricComparison(
                    metric=metric,
                    baseline_value=baseline_value,
                    candidate_value=candidate_value,
                    point_delta=candidate_value - baseline_value,
                    adjusted_delta_upper=adjusted_upper,
                    max_drop=max_drop,
                    regressed=regressed,
                )
            )
        comparisons.append(
            SliceComparison(
                dimension=rule.dimension,
                value=rule.value,
                pairs=len(indices),
                supported_pairs=len(strata[0]),
                unsupported_pairs=len(strata[1]),
                metrics=tuple(metric_comparisons),
            )
        )

    document = {
        "schema_version": SLICE_COMPARISON_SCHEMA_VERSION,
        "bootstrap_method_version": SLICE_BOOTSTRAP_METHOD_VERSION,
        "baseline_ledger_sha256": baseline.artifact_sha256,
        "candidate_ledger_sha256": candidate.artifact_sha256,
        "baseline_assignment_sha256": baseline_assignment.artifact_sha256,
        "candidate_assignment_sha256": candidate_assignment.artifact_sha256,
        "evaluation_assignment_sha256": baseline_assignment.evaluation_assignment_sha256,
        "confidence_level": float(policy.confidence_level),
        "family_wise_method": policy.family_wise_method,
        "hypotheses": hypotheses,
        "per_hypothesis_confidence_level": per_hypothesis_confidence,
        "resamples": policy.resamples,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed_sha256": seed_sha256,
        "policy_sha256": policy_sha256,
        "slices": [asdict(item) for item in comparisons],
        "meets_policy": not failure_reasons,
        "failure_reasons": failure_reasons,
    }
    return PairedSliceRegressionComparison(
        schema_version=SLICE_COMPARISON_SCHEMA_VERSION,
        bootstrap_method_version=SLICE_BOOTSTRAP_METHOD_VERSION,
        baseline_ledger_sha256=baseline.artifact_sha256,
        candidate_ledger_sha256=candidate.artifact_sha256,
        baseline_assignment_sha256=baseline_assignment.artifact_sha256,
        candidate_assignment_sha256=candidate_assignment.artifact_sha256,
        evaluation_assignment_sha256=baseline_assignment.evaluation_assignment_sha256,
        confidence_level=float(policy.confidence_level),
        family_wise_method=policy.family_wise_method,
        hypotheses=hypotheses,
        per_hypothesis_confidence_level=per_hypothesis_confidence,
        resamples=policy.resamples,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed_sha256=seed_sha256,
        policy_sha256=policy_sha256,
        slices=tuple(comparisons),
        meets_policy=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        comparison_sha256=_sha256(document),
    )
