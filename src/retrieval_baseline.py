from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from src.retrieval_eval import RetrievalMetrics

BASELINE_SCHEMA_VERSION = "retrieval-regression-baseline/v1"
COMPARISON_SCHEMA_VERSION = "retrieval-regression-comparison/v1"
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HIGHER_IS_BETTER = (
    "hit_rate_at_k",
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "mean_unique_documents_at_k",
)
_LOWER_IS_BETTER = (
    "duplicate_document_rate_at_k",
    "mean_largest_document_share_at_k",
)
_METRIC_NAMES = (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER)
_METRIC_FIELDS = {"cases", *_METRIC_NAMES}
_THRESHOLD_FIELDS = {f"max_{name}_regression" for name in _METRIC_NAMES}
_BASELINE_FIELDS = {
    "schema_version",
    "baseline_id",
    "baseline_version",
    "source_commit",
    "k",
    "corpus_snapshot_sha256",
    "evaluation_cases_sha256",
    "metrics",
    "thresholds",
    "artifact_sha256",
}
_MAX_BASELINE_BYTES = 64 * 1024


class RetrievalBaselineError(ValueError):
    """The retrieval baseline or candidate is unsafe or incompatible."""

    code = "retrieval_baseline_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class RetrievalRegressionThresholds:
    max_hit_rate_at_k_regression: float
    max_recall_at_k_regression: float
    max_precision_at_k_regression: float
    max_mean_reciprocal_rank_regression: float
    max_ndcg_at_k_regression: float
    max_mean_unique_documents_at_k_regression: float
    max_duplicate_document_rate_at_k_regression: float
    max_mean_largest_document_share_at_k_regression: float

    def __post_init__(self) -> None:
        for name in _METRIC_NAMES:
            field = f"max_{name}_regression"
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or float(value) < 0
            ):
                raise ValueError(f"{field} must be finite and non-negative")
            upper = 1_000.0 if name == "mean_unique_documents_at_k" else 1.0
            if float(value) > upper:
                raise ValueError(f"{field} exceeds its metric range")

    def for_metric(self, metric: str) -> float:
        if metric not in _METRIC_NAMES:
            raise KeyError(metric)
        return float(getattr(self, f"max_{metric}_regression"))


@dataclass(frozen=True, slots=True)
class RetrievalBaselineArtifact:
    schema_version: str
    baseline_id: str
    baseline_version: int
    source_commit: str
    k: int
    corpus_snapshot_sha256: str
    evaluation_cases_sha256: str
    metrics: RetrievalMetrics
    thresholds: RetrievalRegressionThresholds
    artifact_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalMetricComparison:
    metric: str
    direction: str
    baseline: float
    candidate: float
    observed_regression: float
    max_regression: float
    regressed: bool


@dataclass(frozen=True, slots=True)
class RetrievalRegressionComparison:
    schema_version: str
    baseline_id: str
    baseline_version: int
    baseline_artifact_sha256: str
    candidate_metrics_sha256: str
    metrics: tuple[RetrievalMetricComparison, ...]
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
        raise RetrievalBaselineError(f"{name} must be canonical lowercase SHA-256")
    return value


def _validate_metrics(metrics: RetrievalMetrics, *, k: int, name: str) -> None:
    if not isinstance(metrics, RetrievalMetrics):
        raise RetrievalBaselineError(f"{name} metrics are malformed")
    if type(metrics.cases) is not int or not 1 <= metrics.cases <= 10_000_000:
        raise RetrievalBaselineError(f"{name} case count is malformed")
    for metric in _METRIC_NAMES:
        value = getattr(metrics, metric)
        upper = float(k) if metric == "mean_unique_documents_at_k" else 1.0
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= float(value) <= upper
        ):
            raise RetrievalBaselineError(f"{name} metric {metric} is malformed")


def _metrics_from_mapping(value: object) -> RetrievalMetrics:
    if not isinstance(value, Mapping) or set(value) != _METRIC_FIELDS:
        raise RetrievalBaselineError("baseline artifact is malformed")
    try:
        return RetrievalMetrics(**value)
    except TypeError as exc:
        raise RetrievalBaselineError("baseline artifact is malformed") from exc


def _thresholds_from_mapping(value: object) -> RetrievalRegressionThresholds:
    if not isinstance(value, Mapping) or set(value) != _THRESHOLD_FIELDS:
        raise RetrievalBaselineError("baseline artifact is malformed")
    try:
        return RetrievalRegressionThresholds(**value)
    except (TypeError, ValueError) as exc:
        raise RetrievalBaselineError("baseline artifact is malformed") from exc


def create_retrieval_baseline(
    metrics: RetrievalMetrics,
    thresholds: RetrievalRegressionThresholds,
    *,
    baseline_id: str,
    baseline_version: int,
    source_commit: str,
    k: int,
    corpus_snapshot_sha256: str,
    evaluation_cases_sha256: str,
) -> RetrievalBaselineArtifact:
    """Seal one reviewed retrieval result and its explicit regression budget."""

    if (
        not isinstance(baseline_id, str)
        or _IDENTIFIER_RE.fullmatch(baseline_id) is None
    ):
        raise ValueError("baseline_id must contain 1-128 safe characters")
    if type(baseline_version) is not int or not 1 <= baseline_version <= 1_000_000:
        raise ValueError("baseline_version must be an integer in [1, 1000000]")
    if (
        not isinstance(source_commit, str)
        or _GIT_COMMIT_RE.fullmatch(source_commit) is None
    ):
        raise ValueError("source_commit must be a full lowercase Git commit SHA")
    if type(k) is not int or not 1 <= k <= 1_000:
        raise ValueError("k must be an integer in [1, 1000]")
    if not isinstance(thresholds, RetrievalRegressionThresholds):
        raise TypeError("thresholds must be RetrievalRegressionThresholds")
    _validate_digest(corpus_snapshot_sha256, "corpus snapshot digest")
    _validate_digest(evaluation_cases_sha256, "evaluation cases digest")
    _validate_metrics(metrics, k=k, name="baseline")
    if thresholds.max_mean_unique_documents_at_k_regression > k:
        raise ValueError("unique-document regression threshold cannot exceed k")
    document = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline_id": baseline_id,
        "baseline_version": baseline_version,
        "source_commit": source_commit,
        "k": k,
        "corpus_snapshot_sha256": corpus_snapshot_sha256,
        "evaluation_cases_sha256": evaluation_cases_sha256,
        "metrics": asdict(metrics),
        "thresholds": asdict(thresholds),
    }
    return RetrievalBaselineArtifact(
        schema_version=BASELINE_SCHEMA_VERSION,
        baseline_id=baseline_id,
        baseline_version=baseline_version,
        source_commit=source_commit,
        k=k,
        corpus_snapshot_sha256=corpus_snapshot_sha256,
        evaluation_cases_sha256=evaluation_cases_sha256,
        metrics=metrics,
        thresholds=thresholds,
        artifact_sha256=_sha256(document),
    )


def load_retrieval_baseline(
    document: Mapping[str, object],
) -> RetrievalBaselineArtifact:
    """Load and integrity-check a decoded baseline artifact."""

    if not isinstance(document, Mapping) or set(document) != _BASELINE_FIELDS:
        raise RetrievalBaselineError("baseline artifact is malformed")
    try:
        baseline = RetrievalBaselineArtifact(
            schema_version=document["schema_version"],
            baseline_id=document["baseline_id"],
            baseline_version=document["baseline_version"],
            source_commit=document["source_commit"],
            k=document["k"],
            corpus_snapshot_sha256=document["corpus_snapshot_sha256"],
            evaluation_cases_sha256=document["evaluation_cases_sha256"],
            metrics=_metrics_from_mapping(document["metrics"]),
            thresholds=_thresholds_from_mapping(document["thresholds"]),
            artifact_sha256=document["artifact_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RetrievalBaselineError("baseline artifact is malformed") from exc
    _validate_baseline(baseline)
    return baseline


def serialize_retrieval_baseline(baseline: RetrievalBaselineArtifact) -> bytes:
    """Serialize one validated artifact as bounded canonical ASCII JSON."""

    if not isinstance(baseline, RetrievalBaselineArtifact):
        raise TypeError("baseline must be RetrievalBaselineArtifact")
    _validate_baseline(baseline)
    payload = _canonical_json(baseline.to_audit_dict())
    if len(payload) > _MAX_BASELINE_BYTES:
        raise RetrievalBaselineError("baseline artifact exceeds size limit")
    return payload


def load_retrieval_baseline_json(payload: bytes) -> RetrievalBaselineArtifact:
    """Parse one bounded, duplicate-free, canonical baseline JSON document."""

    if not isinstance(payload, bytes) or not 1 <= len(payload) <= _MAX_BASELINE_BYTES:
        raise RetrievalBaselineError("baseline JSON is empty or exceeds size limit")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RetrievalBaselineError("baseline JSON contains duplicate keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise RetrievalBaselineError(
            f"baseline JSON contains invalid constant: {value}"
        )

    try:
        document = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrievalBaselineError("baseline JSON is malformed") from exc
    baseline = load_retrieval_baseline(document)
    if payload != _canonical_json(baseline.to_audit_dict()):
        raise RetrievalBaselineError("baseline JSON is not canonical")
    return baseline


def _validate_baseline(baseline: RetrievalBaselineArtifact) -> None:
    if baseline.schema_version != BASELINE_SCHEMA_VERSION:
        raise RetrievalBaselineError("baseline schema is unsupported")
    if (
        not isinstance(baseline.baseline_id, str)
        or _IDENTIFIER_RE.fullmatch(baseline.baseline_id) is None
        or type(baseline.baseline_version) is not int
        or not 1 <= baseline.baseline_version <= 1_000_000
        or not isinstance(baseline.source_commit, str)
        or _GIT_COMMIT_RE.fullmatch(baseline.source_commit) is None
        or type(baseline.k) is not int
        or not 1 <= baseline.k <= 1_000
        or not isinstance(baseline.thresholds, RetrievalRegressionThresholds)
    ):
        raise RetrievalBaselineError("baseline metadata is malformed")
    _validate_digest(baseline.corpus_snapshot_sha256, "corpus snapshot digest")
    _validate_digest(baseline.evaluation_cases_sha256, "evaluation cases digest")
    _validate_digest(baseline.artifact_sha256, "baseline artifact digest")
    _validate_metrics(baseline.metrics, k=baseline.k, name="baseline")
    if baseline.thresholds.max_mean_unique_documents_at_k_regression > baseline.k:
        raise RetrievalBaselineError("baseline threshold is incompatible with k")
    document = {
        key: value
        for key, value in baseline.to_audit_dict().items()
        if key != "artifact_sha256"
    }
    if baseline.artifact_sha256 != _sha256(document):
        raise RetrievalBaselineError("baseline artifact digest mismatch")


def compare_retrieval_baseline(
    baseline: RetrievalBaselineArtifact,
    candidate: RetrievalMetrics,
    *,
    k: int,
    corpus_snapshot_sha256: str,
    evaluation_cases_sha256: str,
) -> RetrievalRegressionComparison:
    """Compare like-for-like aggregate retrieval results against a sealed baseline."""

    if not isinstance(baseline, RetrievalBaselineArtifact):
        raise TypeError("baseline must be RetrievalBaselineArtifact")
    _validate_baseline(baseline)
    _validate_digest(corpus_snapshot_sha256, "candidate corpus snapshot digest")
    _validate_digest(evaluation_cases_sha256, "candidate evaluation cases digest")
    if k != baseline.k:
        raise RetrievalBaselineError("candidate k does not match baseline")
    if corpus_snapshot_sha256 != baseline.corpus_snapshot_sha256:
        raise RetrievalBaselineError(
            "candidate corpus snapshot does not match baseline"
        )
    if evaluation_cases_sha256 != baseline.evaluation_cases_sha256:
        raise RetrievalBaselineError("candidate evaluation cases do not match baseline")
    _validate_metrics(candidate, k=k, name="candidate")
    if candidate.cases != baseline.metrics.cases:
        raise RetrievalBaselineError("candidate case count does not match baseline")

    comparisons: list[RetrievalMetricComparison] = []
    for metric in _METRIC_NAMES:
        baseline_value = float(getattr(baseline.metrics, metric))
        candidate_value = float(getattr(candidate, metric))
        higher_is_better = metric in _HIGHER_IS_BETTER
        regression = (
            baseline_value - candidate_value
            if higher_is_better
            else candidate_value - baseline_value
        )
        max_regression = baseline.thresholds.for_metric(metric)
        comparisons.append(
            RetrievalMetricComparison(
                metric=metric,
                direction="higher_is_better" if higher_is_better else "lower_is_better",
                baseline=baseline_value,
                candidate=candidate_value,
                observed_regression=max(0.0, regression),
                max_regression=max_regression,
                regressed=(
                    regression > max_regression
                    and not math.isclose(
                        regression, max_regression, rel_tol=0.0, abs_tol=1e-12
                    )
                ),
            )
        )

    failure_reasons = tuple(
        f"metric_regressed:{item.metric}" for item in comparisons if item.regressed
    )
    candidate_document = {
        "k": k,
        "corpus_snapshot_sha256": corpus_snapshot_sha256,
        "evaluation_cases_sha256": evaluation_cases_sha256,
        "metrics": asdict(candidate),
    }
    comparison_document = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "baseline_id": baseline.baseline_id,
        "baseline_version": baseline.baseline_version,
        "baseline_artifact_sha256": baseline.artifact_sha256,
        "candidate_metrics_sha256": _sha256(candidate_document),
        "metrics": [asdict(item) for item in comparisons],
        "meets_policy": not failure_reasons,
        "failure_reasons": list(failure_reasons),
    }
    return RetrievalRegressionComparison(
        schema_version=COMPARISON_SCHEMA_VERSION,
        baseline_id=baseline.baseline_id,
        baseline_version=baseline.baseline_version,
        baseline_artifact_sha256=baseline.artifact_sha256,
        candidate_metrics_sha256=str(comparison_document["candidate_metrics_sha256"]),
        metrics=tuple(comparisons),
        meets_policy=not failure_reasons,
        failure_reasons=failure_reasons,
        comparison_sha256=_sha256(comparison_document),
    )
