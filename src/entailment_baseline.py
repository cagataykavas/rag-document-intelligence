from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from statistics import NormalDist

from src.entailment_eval import (
    EntailmentDiagnosticReport,
    EntailmentMetrics,
)

BASELINE_SCHEMA_VERSION = "entailment-regression-baseline/v1"
COMPARISON_SCHEMA_VERSION = "entailment-regression-comparison/v1"
UNCERTAINTY_METHOD_VERSION = "wilson-newcombe-independent-proportions/v1"
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,127}")
_SLICE_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+:-]{0,63}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_METRIC_NAMES = (
    "supported_precision",
    "supported_recall",
    "unsupported_recall",
    "balanced_accuracy",
)
_METRIC_FIELDS = {
    "pairs",
    "true_supported",
    "false_supported",
    "true_unsupported",
    "false_unsupported",
    *_METRIC_NAMES,
}
_BASELINE_FIELDS = {
    "schema_version",
    "baseline_id",
    "baseline_version",
    "source_report_schema_version",
    "source_report_sha256",
    "scorer_id",
    "threshold",
    "policy_sha256",
    "drift_slice_version",
    "drift_slice_policy_sha256",
    "slice_assignment_sha256",
    "slice_inventory_sha256",
    "evaluation",
    "slices",
    "artifact_sha256",
}


class EntailmentBaselineError(ValueError):
    """The baseline or comparison contract is unsafe or incompatible."""

    code = "entailment_baseline_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class MetricRegressionThresholds:
    max_supported_precision_drop: float
    max_supported_recall_drop: float
    max_unsupported_recall_drop: float
    max_balanced_accuracy_drop: float

    def __post_init__(self) -> None:
        for name in (
            "max_supported_precision_drop",
            "max_supported_recall_drop",
            "max_unsupported_recall_drop",
            "max_balanced_accuracy_drop",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= float(value) <= 1
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")

    def for_metric(self, metric: str) -> float:
        return float(getattr(self, f"max_{metric}_drop"))


@dataclass(frozen=True, slots=True)
class SliceRegressionRule:
    dimension: str
    value: str
    thresholds: MetricRegressionThresholds

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dimension, str)
            or _IDENTIFIER_RE.fullmatch(self.dimension) is None
        ):
            raise ValueError("slice dimension must be a safe identifier")
        if (
            not isinstance(self.value, str)
            or _SLICE_VALUE_RE.fullmatch(self.value) is None
        ):
            raise ValueError("slice value must contain 1-64 safe characters")
        if not isinstance(self.thresholds, MetricRegressionThresholds):
            raise TypeError("slice thresholds must be MetricRegressionThresholds")

    @property
    def key(self) -> tuple[str, str]:
        return (self.dimension, self.value)


@dataclass(frozen=True, slots=True)
class EntailmentRegressionPolicy:
    confidence_level: float
    global_thresholds: MetricRegressionThresholds
    slice_rules: tuple[SliceRegressionRule, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not math.isfinite(self.confidence_level)
            or not 0.8 <= float(self.confidence_level) <= 0.999
        ):
            raise ValueError("confidence_level must be finite and in [0.8, 0.999]")
        if not isinstance(self.global_thresholds, MetricRegressionThresholds):
            raise TypeError("global_thresholds must be MetricRegressionThresholds")
        if (
            not isinstance(self.slice_rules, tuple)
            or len(self.slice_rules) > 512
            or any(
                not isinstance(rule, SliceRegressionRule) for rule in self.slice_rules
            )
        ):
            raise TypeError("slice_rules must be a tuple of at most 512 rules")
        keys = [rule.key for rule in self.slice_rules]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "slice_rules must not contain duplicate dimension/value keys"
            )


@dataclass(frozen=True, slots=True)
class EntailmentBaselineSlice:
    dimension: str
    value: str
    evaluation: EntailmentMetrics

    @property
    def key(self) -> tuple[str, str]:
        return (self.dimension, self.value)


@dataclass(frozen=True, slots=True)
class EntailmentBaselineArtifact:
    schema_version: str
    baseline_id: str
    baseline_version: int
    source_report_schema_version: str
    source_report_sha256: str
    scorer_id: str
    threshold: float
    policy_sha256: str
    drift_slice_version: str
    drift_slice_policy_sha256: str
    slice_assignment_sha256: str
    slice_inventory_sha256: str
    evaluation: EntailmentMetrics
    slices: tuple[EntailmentBaselineSlice, ...]
    artifact_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricInterval:
    value: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class MetricRegressionComparison:
    metric: str
    baseline: MetricInterval
    candidate: MetricInterval
    point_delta: float
    delta_lower: float
    delta_upper: float
    max_drop: float
    regressed: bool


@dataclass(frozen=True, slots=True)
class SliceRegressionComparison:
    dimension: str
    value: str
    metrics: tuple[MetricRegressionComparison, ...]
    regressed: bool


@dataclass(frozen=True, slots=True)
class EntailmentRegressionComparison:
    schema_version: str
    uncertainty_method_version: str
    baseline_id: str
    baseline_version: int
    baseline_artifact_sha256: str
    candidate_report_sha256: str
    candidate_scorer_id: str
    confidence_level: float
    policy_sha256: str
    global_metrics: tuple[MetricRegressionComparison, ...]
    slices: tuple[SliceRegressionComparison, ...]
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


def _validate_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EntailmentBaselineError(f"{name} must be canonical lowercase SHA-256")
    return value


def _validate_metrics(metrics: EntailmentMetrics, name: str) -> None:
    if not isinstance(metrics, EntailmentMetrics):
        raise EntailmentBaselineError(f"{name} metrics are malformed")
    counts = (
        metrics.pairs,
        metrics.true_supported,
        metrics.false_supported,
        metrics.true_unsupported,
        metrics.false_unsupported,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise EntailmentBaselineError(f"{name} metric counts are malformed")
    if sum(counts[1:]) != metrics.pairs or metrics.pairs == 0:
        raise EntailmentBaselineError(f"{name} confusion matrix is inconsistent")
    if (
        metrics.true_supported + metrics.false_unsupported == 0
        or metrics.true_unsupported + metrics.false_supported == 0
    ):
        raise EntailmentBaselineError(f"{name} must contain both labels")
    for metric in _METRIC_NAMES:
        value = getattr(metrics, metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= float(value) <= 1
        ):
            raise EntailmentBaselineError(f"{name} metric values are malformed")

    expected = {
        "supported_precision": (
            metrics.true_supported / (metrics.true_supported + metrics.false_supported)
            if metrics.true_supported + metrics.false_supported
            else 0.0
        ),
        "supported_recall": metrics.true_supported
        / (metrics.true_supported + metrics.false_unsupported),
        "unsupported_recall": metrics.true_unsupported
        / (metrics.true_unsupported + metrics.false_supported),
    }
    expected["balanced_accuracy"] = (
        expected["supported_recall"] + expected["unsupported_recall"]
    ) / 2
    if any(
        not math.isclose(getattr(metrics, metric), value, rel_tol=0, abs_tol=1e-12)
        for metric, value in expected.items()
    ):
        raise EntailmentBaselineError(f"{name} metric values do not match counts")


def _validate_report_digest(report: EntailmentDiagnosticReport, name: str) -> None:
    _validate_digest(report.report_sha256, f"{name} digest")
    document = {
        key: value
        for key, value in report.to_audit_dict().items()
        if key != "report_sha256"
    }
    if report.report_sha256 != _sha256(document):
        raise EntailmentBaselineError(f"{name} digest mismatch")


def _slice_inventory(report: EntailmentDiagnosticReport) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "calibration_pairs": item.calibration_pairs,
                "dimension": item.dimension,
                "evaluation_pairs": item.evaluation_pairs,
                "metrics_reported": item.metrics_reported,
                "value": item.value,
            }
            for item in report.drift_slices
        ),
        key=lambda item: (str(item["dimension"]), str(item["value"])),
    )


def create_entailment_baseline(
    report: EntailmentDiagnosticReport,
    *,
    baseline_id: str,
    baseline_version: int,
) -> EntailmentBaselineArtifact:
    """Create an immutable privacy-reduced baseline from one accepted report."""

    if not isinstance(report, EntailmentDiagnosticReport):
        raise TypeError("report must be an EntailmentDiagnosticReport")
    if (
        not isinstance(baseline_id, str)
        or _IDENTIFIER_RE.fullmatch(baseline_id) is None
    ):
        raise ValueError("baseline_id must contain 1-128 safe characters")
    if type(baseline_version) is not int or not 1 <= baseline_version <= 1_000_000:
        raise ValueError("baseline_version must be an integer in [1, 1000000]")
    if not report.meets_policy:
        raise EntailmentBaselineError(
            "source report does not meet its entailment policy"
        )
    if (
        report.drift_slice_version is None
        or report.drift_slice_policy_sha256 is None
        or report.slice_assignment_sha256 is None
        or not report.drift_slices
    ):
        raise EntailmentBaselineError(
            "source report has no privacy-reduced drift slices"
        )
    _validate_report_digest(report, "source report")
    _validate_digest(report.policy_sha256, "source policy digest")
    _validate_digest(report.drift_slice_policy_sha256, "drift slice policy digest")
    _validate_metrics(report.evaluation, "global evaluation")
    slices: list[EntailmentBaselineSlice] = []
    for item in report.drift_slices:
        if not item.metrics_reported:
            continue
        if item.evaluation is None:
            raise EntailmentBaselineError(
                "reported source slice is missing evaluation metrics"
            )
        _validate_metrics(item.evaluation, f"slice {item.dimension}:{item.value}")
        slices.append(
            EntailmentBaselineSlice(
                dimension=item.dimension,
                value=item.value,
                evaluation=item.evaluation,
            )
        )
    if not slices:
        raise EntailmentBaselineError(
            "source report has no statistically supported slices"
        )
    slices.sort(key=lambda item: item.key)
    document = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "baseline_version": baseline_version,
        "source_report_schema_version": report.schema_version,
        "source_report_sha256": report.report_sha256,
        "scorer_id": report.scorer_id,
        "threshold": report.threshold,
        "policy_sha256": report.policy_sha256,
        "drift_slice_version": report.drift_slice_version,
        "drift_slice_policy_sha256": report.drift_slice_policy_sha256,
        "slice_assignment_sha256": report.slice_assignment_sha256,
        "slice_inventory_sha256": _sha256(_slice_inventory(report)),
        "evaluation": asdict(report.evaluation),
        "slices": [asdict(item) for item in slices],
    }
    return EntailmentBaselineArtifact(
        schema_version=BASELINE_SCHEMA_VERSION,
        baseline_id=baseline_id,
        baseline_version=baseline_version,
        source_report_schema_version=report.schema_version,
        source_report_sha256=report.report_sha256,
        scorer_id=report.scorer_id,
        threshold=report.threshold,
        policy_sha256=report.policy_sha256,
        drift_slice_version=report.drift_slice_version,
        drift_slice_policy_sha256=report.drift_slice_policy_sha256,
        slice_assignment_sha256=report.slice_assignment_sha256,
        slice_inventory_sha256=str(document["slice_inventory_sha256"]),
        evaluation=report.evaluation,
        slices=tuple(slices),
        artifact_sha256=_sha256(document),
    )


def _metrics_from_mapping(value: object) -> EntailmentMetrics:
    if not isinstance(value, Mapping) or set(value) != _METRIC_FIELDS:
        raise EntailmentBaselineError("baseline artifact is malformed")
    try:
        return EntailmentMetrics(**value)
    except TypeError as exc:
        raise EntailmentBaselineError("baseline artifact is malformed") from exc


def load_entailment_baseline(
    document: Mapping[str, object],
) -> EntailmentBaselineArtifact:
    """Load and integrity-check a baseline decoded from canonical JSON."""

    if not isinstance(document, Mapping) or set(document) != _BASELINE_FIELDS:
        raise EntailmentBaselineError("baseline artifact is malformed")
    raw_slices = document.get("slices")
    if not isinstance(raw_slices, list) or not 1 <= len(raw_slices) <= 512:
        raise EntailmentBaselineError("baseline artifact is malformed")
    slices: list[EntailmentBaselineSlice] = []
    for item in raw_slices:
        if not isinstance(item, Mapping) or set(item) != {
            "dimension",
            "value",
            "evaluation",
        }:
            raise EntailmentBaselineError("baseline artifact is malformed")
        slices.append(
            EntailmentBaselineSlice(
                dimension=item["dimension"],
                value=item["value"],
                evaluation=_metrics_from_mapping(item["evaluation"]),
            )
        )
    try:
        baseline = EntailmentBaselineArtifact(
            schema_version=document["schema_version"],
            baseline_id=document["baseline_id"],
            baseline_version=document["baseline_version"],
            source_report_schema_version=document["source_report_schema_version"],
            source_report_sha256=document["source_report_sha256"],
            scorer_id=document["scorer_id"],
            threshold=document["threshold"],
            policy_sha256=document["policy_sha256"],
            drift_slice_version=document["drift_slice_version"],
            drift_slice_policy_sha256=document["drift_slice_policy_sha256"],
            slice_assignment_sha256=document["slice_assignment_sha256"],
            slice_inventory_sha256=document["slice_inventory_sha256"],
            evaluation=_metrics_from_mapping(document["evaluation"]),
            slices=tuple(slices),
            artifact_sha256=document["artifact_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise EntailmentBaselineError("baseline artifact is malformed") from exc
    _validate_baseline(baseline)
    return baseline


def _wilson(successes: int, trials: int, confidence_level: float) -> MetricInterval:
    value = successes / trials if trials else 0.0
    if trials == 0:
        return MetricInterval(value=value, lower=0.0, upper=1.0)
    z = NormalDist().inv_cdf((1 + confidence_level) / 2)
    z_squared = z * z
    denominator = 1 + z_squared / trials
    center = (value + z_squared / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(value * (1 - value) / trials + z_squared / (4 * trials * trials))
        / denominator
    )
    return MetricInterval(
        value=value,
        lower=max(0.0, center - half_width),
        upper=min(1.0, center + half_width),
    )


def _metric_interval(
    metrics: EntailmentMetrics,
    metric: str,
    confidence_level: float,
) -> MetricInterval:
    if metric == "supported_precision":
        return _wilson(
            metrics.true_supported,
            metrics.true_supported + metrics.false_supported,
            confidence_level,
        )
    supported_recall = _wilson(
        metrics.true_supported,
        metrics.true_supported + metrics.false_unsupported,
        confidence_level,
    )
    unsupported_recall = _wilson(
        metrics.true_unsupported,
        metrics.true_unsupported + metrics.false_supported,
        confidence_level,
    )
    if metric == "supported_recall":
        return supported_recall
    if metric == "unsupported_recall":
        return unsupported_recall
    if metric == "balanced_accuracy":
        return MetricInterval(
            value=(supported_recall.value + unsupported_recall.value) / 2,
            lower=(supported_recall.lower + unsupported_recall.lower) / 2,
            upper=(supported_recall.upper + unsupported_recall.upper) / 2,
        )
    raise AssertionError(f"unsupported metric: {metric}")


def _compare_metrics(
    baseline: EntailmentMetrics,
    candidate: EntailmentMetrics,
    thresholds: MetricRegressionThresholds,
    confidence_level: float,
) -> tuple[MetricRegressionComparison, ...]:
    comparisons: list[MetricRegressionComparison] = []
    for metric in _METRIC_NAMES:
        baseline_interval = _metric_interval(baseline, metric, confidence_level)
        candidate_interval = _metric_interval(candidate, metric, confidence_level)
        max_drop = thresholds.for_metric(metric)
        point_delta = candidate_interval.value - baseline_interval.value
        delta_lower = point_delta - math.sqrt(
            (candidate_interval.value - candidate_interval.lower) ** 2
            + (baseline_interval.upper - baseline_interval.value) ** 2
        )
        delta_upper = point_delta + math.sqrt(
            (candidate_interval.upper - candidate_interval.value) ** 2
            + (baseline_interval.value - baseline_interval.lower) ** 2
        )
        comparisons.append(
            MetricRegressionComparison(
                metric=metric,
                baseline=baseline_interval,
                candidate=candidate_interval,
                point_delta=point_delta,
                delta_lower=delta_lower,
                delta_upper=delta_upper,
                max_drop=max_drop,
                regressed=delta_upper < -max_drop,
            )
        )
    return tuple(comparisons)


def _validate_baseline(
    baseline: EntailmentBaselineArtifact,
) -> dict[tuple[str, str], EntailmentBaselineSlice]:
    if (
        not isinstance(baseline.slices, tuple)
        or not 1 <= len(baseline.slices) <= 512
        or any(
            not isinstance(item, EntailmentBaselineSlice) for item in baseline.slices
        )
    ):
        raise EntailmentBaselineError("baseline slices are malformed")
    expected_hash = _sha256(
        {
            key: value
            for key, value in baseline.to_audit_dict().items()
            if key != "artifact_sha256"
        }
    )
    if baseline.schema_version != BASELINE_SCHEMA_VERSION:
        raise EntailmentBaselineError("baseline schema is unsupported")
    if (
        not isinstance(baseline.baseline_id, str)
        or _IDENTIFIER_RE.fullmatch(baseline.baseline_id) is None
        or type(baseline.baseline_version) is not int
        or not 1 <= baseline.baseline_version <= 1_000_000
        or not isinstance(baseline.scorer_id, str)
        or _IDENTIFIER_RE.fullmatch(baseline.scorer_id) is None
        or not isinstance(baseline.source_report_schema_version, str)
        or _VERSION_RE.fullmatch(baseline.source_report_schema_version) is None
        or not isinstance(baseline.drift_slice_version, str)
        or _VERSION_RE.fullmatch(baseline.drift_slice_version) is None
        or isinstance(baseline.threshold, bool)
        or not isinstance(baseline.threshold, (int, float))
        or not math.isfinite(baseline.threshold)
        or not 0 <= float(baseline.threshold) <= 1
    ):
        raise EntailmentBaselineError("baseline metadata is malformed")
    for value, name in (
        (baseline.source_report_sha256, "source report digest"),
        (baseline.policy_sha256, "source policy digest"),
        (baseline.drift_slice_policy_sha256, "drift slice policy digest"),
        (baseline.slice_assignment_sha256, "slice assignment digest"),
        (baseline.slice_inventory_sha256, "slice inventory digest"),
        (baseline.artifact_sha256, "baseline artifact digest"),
    ):
        _validate_digest(value, name)
    if baseline.artifact_sha256 != expected_hash:
        raise EntailmentBaselineError("baseline artifact digest mismatch")
    _validate_metrics(baseline.evaluation, "baseline global evaluation")
    for item in baseline.slices:
        if (
            not isinstance(item.dimension, str)
            or _IDENTIFIER_RE.fullmatch(item.dimension) is None
            or not isinstance(item.value, str)
            or _SLICE_VALUE_RE.fullmatch(item.value) is None
        ):
            raise EntailmentBaselineError("baseline slice key is malformed")
        _validate_metrics(
            item.evaluation, f"baseline slice {item.dimension}:{item.value}"
        )
    slice_map = {item.key: item for item in baseline.slices}
    if len(slice_map) != len(baseline.slices):
        raise EntailmentBaselineError("baseline slices are duplicated")
    return slice_map


def compare_entailment_baseline(
    baseline: EntailmentBaselineArtifact,
    candidate: EntailmentDiagnosticReport,
    policy: EntailmentRegressionPolicy,
) -> EntailmentRegressionComparison:
    """Compare a candidate with uncertainty-aware global and per-slice gates."""

    if not isinstance(baseline, EntailmentBaselineArtifact):
        raise TypeError("baseline must be an EntailmentBaselineArtifact")
    if not isinstance(candidate, EntailmentDiagnosticReport):
        raise TypeError("candidate must be an EntailmentDiagnosticReport")
    if not isinstance(policy, EntailmentRegressionPolicy):
        raise TypeError("policy must be an EntailmentRegressionPolicy")
    baseline_slice_map = _validate_baseline(baseline)
    _validate_report_digest(candidate, "candidate report")
    if candidate.schema_version != baseline.source_report_schema_version:
        raise EntailmentBaselineError("candidate report schema is incompatible")
    if candidate.policy_sha256 != baseline.policy_sha256:
        raise EntailmentBaselineError("candidate entailment policy is incompatible")
    if (
        candidate.drift_slice_version != baseline.drift_slice_version
        or candidate.drift_slice_policy_sha256 != baseline.drift_slice_policy_sha256
    ):
        raise EntailmentBaselineError("candidate drift slice policy is incompatible")
    _validate_metrics(candidate.evaluation, "candidate global evaluation")
    rule_map = {rule.key: rule for rule in policy.slice_rules}
    if set(rule_map) != set(baseline_slice_map):
        raise EntailmentBaselineError(
            "regression policy must exactly cover every reportable baseline slice"
        )
    candidate_slice_map = {
        (item.dimension, item.value): item
        for item in candidate.drift_slices
        if item.metrics_reported
    }
    global_metrics = _compare_metrics(
        baseline.evaluation,
        candidate.evaluation,
        policy.global_thresholds,
        float(policy.confidence_level),
    )
    failure_reasons = [
        f"global_{item.metric}_regressed" for item in global_metrics if item.regressed
    ]
    if not candidate.meets_policy:
        failure_reasons.append("candidate_entailment_policy_failed")
    failure_reasons.extend(
        f"slice_{dimension}_{value}_unreviewed"
        for dimension, value in sorted(
            set(candidate_slice_map) - set(baseline_slice_map)
        )
    )
    slice_comparisons: list[SliceRegressionComparison] = []
    for key in sorted(baseline_slice_map):
        dimension, value = key
        candidate_slice = candidate_slice_map.get(key)
        if candidate_slice is None or candidate_slice.evaluation is None:
            failure_reasons.append(f"slice_{dimension}_{value}_metrics_unavailable")
            continue
        _validate_metrics(
            candidate_slice.evaluation, f"candidate slice {dimension}:{value}"
        )
        metrics = _compare_metrics(
            baseline_slice_map[key].evaluation,
            candidate_slice.evaluation,
            rule_map[key].thresholds,
            float(policy.confidence_level),
        )
        regressed = any(item.regressed for item in metrics)
        if regressed:
            failure_reasons.extend(
                f"slice_{dimension}_{value}_{item.metric}_regressed"
                for item in metrics
                if item.regressed
            )
        slice_comparisons.append(
            SliceRegressionComparison(
                dimension=dimension,
                value=value,
                metrics=metrics,
                regressed=regressed,
            )
        )
    policy_sha256 = _sha256(asdict(policy))
    document = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "uncertainty_method_version": UNCERTAINTY_METHOD_VERSION,
        "baseline_id": baseline.baseline_id,
        "baseline_version": baseline.baseline_version,
        "baseline_artifact_sha256": baseline.artifact_sha256,
        "candidate_report_sha256": candidate.report_sha256,
        "candidate_scorer_id": candidate.scorer_id,
        "confidence_level": float(policy.confidence_level),
        "policy_sha256": policy_sha256,
        "global_metrics": [asdict(item) for item in global_metrics],
        "slices": [asdict(item) for item in slice_comparisons],
        "meets_policy": not failure_reasons,
        "failure_reasons": failure_reasons,
    }
    return EntailmentRegressionComparison(
        schema_version=COMPARISON_SCHEMA_VERSION,
        uncertainty_method_version=UNCERTAINTY_METHOD_VERSION,
        baseline_id=baseline.baseline_id,
        baseline_version=baseline.baseline_version,
        baseline_artifact_sha256=baseline.artifact_sha256,
        candidate_report_sha256=candidate.report_sha256,
        candidate_scorer_id=candidate.scorer_id,
        confidence_level=float(policy.confidence_level),
        policy_sha256=policy_sha256,
        global_metrics=global_metrics,
        slices=tuple(slice_comparisons),
        meets_policy=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        comparison_sha256=_sha256(document),
    )
