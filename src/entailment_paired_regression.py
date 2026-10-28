from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from src.entailment_baseline import MetricRegressionThresholds
from src.entailment_eval import (
    EntailmentDiagnosticReport,
    EntailmentLabel,
    EntailmentMetrics,
    EntailmentPair,
)

SCORE_LEDGER_SCHEMA_VERSION = "entailment-paired-score-ledger/v1"
PAIRED_COMPARISON_SCHEMA_VERSION = "entailment-paired-comparison/v1"
PAIRED_BOOTSTRAP_METHOD_VERSION = "stratified-paired-bootstrap-bonferroni/v1"
_CASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SCORER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,127}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_METRIC_NAMES = (
    "supported_precision",
    "supported_recall",
    "unsupported_recall",
    "balanced_accuracy",
)
_LEDGER_FIELDS = {
    "schema_version",
    "report_schema_version",
    "report_sha256",
    "evaluation_sha256",
    "scorer_id",
    "threshold",
    "policy_sha256",
    "rows",
    "artifact_sha256",
}
_ROW_FIELDS = {"case_id_sha256", "content_sha256", "label", "score"}
_MAX_CASES = 100_000
_MAX_BOOTSTRAP_DRAWS = 5_000_000


class PairedRegressionError(ValueError):
    """A score ledger or paired comparison is unsafe or incompatible."""

    code = "paired_entailment_regression_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class EvaluationScoreRow:
    case_id_sha256: str
    content_sha256: str
    label: str
    score: float


@dataclass(frozen=True, slots=True)
class EvaluationScoreLedger:
    schema_version: str
    report_schema_version: str
    report_sha256: str
    evaluation_sha256: str
    scorer_id: str
    threshold: float
    policy_sha256: str
    rows: tuple[EvaluationScoreRow, ...]
    artifact_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedBootstrapPolicy:
    confidence_level: float
    resamples: int
    thresholds: MetricRegressionThresholds
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
        if not isinstance(self.thresholds, MetricRegressionThresholds):
            raise TypeError("thresholds must be MetricRegressionThresholds")
        if self.family_wise_method != "bonferroni":
            raise ValueError("family_wise_method must be 'bonferroni'")


@dataclass(frozen=True, slots=True)
class PairedMetricComparison:
    metric: str
    baseline_value: float
    candidate_value: float
    point_delta: float
    adjusted_delta_upper: float
    max_drop: float
    regressed: bool


@dataclass(frozen=True, slots=True)
class PairedRegressionComparison:
    schema_version: str
    bootstrap_method_version: str
    baseline_ledger_sha256: str
    candidate_ledger_sha256: str
    baseline_report_sha256: str
    candidate_report_sha256: str
    confidence_level: float
    family_wise_method: str
    hypotheses: int
    per_hypothesis_confidence_level: float
    resamples: int
    bootstrap_seed_sha256: str
    policy_sha256: str
    metrics: tuple[PairedMetricComparison, ...]
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


def _content_sha256(pair: EntailmentPair) -> str:
    payload = json.dumps(
        {"claim": pair.claim, "evidence": pair.evidence},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PairedRegressionError(f"{name} must be canonical lowercase SHA-256")
    return value


def _validate_report_digest(report: EntailmentDiagnosticReport) -> None:
    _validate_digest(report.report_sha256, "report digest")
    document = {
        key: value
        for key, value in report.to_audit_dict().items()
        if key != "report_sha256"
    }
    if report.report_sha256 != _sha256(document):
        raise PairedRegressionError("report digest mismatch")


def _metrics(rows: Sequence[EvaluationScoreRow], threshold: float) -> EntailmentMetrics:
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


def _evaluation_digest(rows: Sequence[EvaluationScoreRow]) -> str:
    return _sha256(
        [
            {
                "case_id_sha256": row.case_id_sha256,
                "content_sha256": row.content_sha256,
                "label": row.label,
                "score": row.score,
            }
            for row in sorted(rows, key=lambda item: item.case_id_sha256)
        ]
    )


def create_evaluation_score_ledger(
    report: EntailmentDiagnosticReport,
    evaluation_pairs: Sequence[EntailmentPair],
    scores_by_case_id: Mapping[str, float],
) -> EvaluationScoreLedger:
    """Seal the scored holdout rows already committed by a diagnostic report."""

    if not isinstance(report, EntailmentDiagnosticReport):
        raise TypeError("report must be an EntailmentDiagnosticReport")
    if isinstance(evaluation_pairs, (str, bytes)) or not isinstance(
        evaluation_pairs, Sequence
    ):
        raise TypeError("evaluation_pairs must be a sequence of EntailmentPair values")
    if not isinstance(scores_by_case_id, Mapping):
        raise TypeError("scores_by_case_id must be a mapping")
    if not 2 <= len(evaluation_pairs) <= _MAX_CASES:
        raise PairedRegressionError("evaluation size is outside [2, 100000]")
    _validate_report_digest(report)
    if not report.meets_policy:
        raise PairedRegressionError("report does not meet its entailment policy")
    expected_ids: set[str] = set()
    content_digests: set[str] = set()
    rows: list[EvaluationScoreRow] = []
    for pair in evaluation_pairs:
        if not isinstance(pair, EntailmentPair):
            raise TypeError("evaluation_pairs contains an unsupported pair")
        if _CASE_ID_RE.fullmatch(pair.case_id) is None or pair.case_id in expected_ids:
            raise PairedRegressionError(
                "evaluation case IDs are malformed or duplicated"
            )
        if not isinstance(pair.claim, str) or not isinstance(pair.evidence, str):
            raise PairedRegressionError("evaluation text must be UTF-8 strings")
        if (
            not pair.claim
            or not pair.evidence
            or not isinstance(pair.label, EntailmentLabel)
        ):
            raise PairedRegressionError("evaluation pair is malformed")
        content_digest = _content_sha256(pair)
        if content_digest in content_digests:
            raise PairedRegressionError("evaluation content is duplicated")
        expected_ids.add(pair.case_id)
        content_digests.add(content_digest)
        score = scores_by_case_id.get(pair.case_id)
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= float(score) <= 1
        ):
            raise PairedRegressionError("evaluation score is missing or invalid")
        rows.append(
            EvaluationScoreRow(
                case_id_sha256=hashlib.sha256(pair.case_id.encode("utf-8")).hexdigest(),
                content_sha256=content_digest,
                label=pair.label.value,
                score=float(score),
            )
        )
    if set(scores_by_case_id) != expected_ids:
        raise PairedRegressionError(
            "score mapping must exactly cover evaluation case IDs"
        )
    rows.sort(key=lambda item: item.case_id_sha256)
    if {row.label for row in rows} != {
        EntailmentLabel.SUPPORTED.value,
        EntailmentLabel.UNSUPPORTED.value,
    }:
        raise PairedRegressionError("evaluation ledger must contain both labels")
    evaluation_sha256 = _evaluation_digest(rows)
    if evaluation_sha256 != report.evaluation_sha256:
        raise PairedRegressionError("evaluation rows do not match report commitment")
    if _metrics(rows, report.threshold) != report.evaluation:
        raise PairedRegressionError("evaluation rows do not match report metrics")
    document = {
        "schema_version": SCORE_LEDGER_SCHEMA_VERSION,
        "report_schema_version": report.schema_version,
        "report_sha256": report.report_sha256,
        "evaluation_sha256": evaluation_sha256,
        "scorer_id": report.scorer_id,
        "threshold": report.threshold,
        "policy_sha256": report.policy_sha256,
        "rows": [asdict(row) for row in rows],
    }
    return EvaluationScoreLedger(
        schema_version=SCORE_LEDGER_SCHEMA_VERSION,
        report_schema_version=report.schema_version,
        report_sha256=report.report_sha256,
        evaluation_sha256=evaluation_sha256,
        scorer_id=report.scorer_id,
        threshold=report.threshold,
        policy_sha256=report.policy_sha256,
        rows=tuple(rows),
        artifact_sha256=_sha256(document),
    )


def _validate_ledger(ledger: EvaluationScoreLedger) -> None:
    if ledger.schema_version != SCORE_LEDGER_SCHEMA_VERSION:
        raise PairedRegressionError("score ledger schema is unsupported")
    if not isinstance(ledger.rows, tuple) or not 2 <= len(ledger.rows) <= _MAX_CASES:
        raise PairedRegressionError("score ledger rows are malformed")
    for value, name in (
        (ledger.report_sha256, "report digest"),
        (ledger.evaluation_sha256, "evaluation digest"),
        (ledger.policy_sha256, "policy digest"),
        (ledger.artifact_sha256, "artifact digest"),
    ):
        _validate_digest(value, name)
    if (
        not isinstance(ledger.report_schema_version, str)
        or _VERSION_RE.fullmatch(ledger.report_schema_version) is None
        or not isinstance(ledger.scorer_id, str)
        or _SCORER_ID_RE.fullmatch(ledger.scorer_id) is None
        or isinstance(ledger.threshold, bool)
        or not isinstance(ledger.threshold, (int, float))
        or not math.isfinite(ledger.threshold)
        or not 0 <= float(ledger.threshold) <= 1
    ):
        raise PairedRegressionError("score ledger metadata is malformed")
    case_ids: set[str] = set()
    content_digests: set[str] = set()
    for row in ledger.rows:
        if not isinstance(row, EvaluationScoreRow):
            raise PairedRegressionError("score ledger rows are malformed")
        _validate_digest(row.case_id_sha256, "case commitment")
        _validate_digest(row.content_sha256, "content commitment")
        if row.label not in {item.value for item in EntailmentLabel}:
            raise PairedRegressionError("score ledger label is malformed")
        if (
            isinstance(row.score, bool)
            or not isinstance(row.score, (int, float))
            or not math.isfinite(row.score)
            or not 0 <= float(row.score) <= 1
        ):
            raise PairedRegressionError("score ledger score is malformed")
        if row.case_id_sha256 in case_ids or row.content_sha256 in content_digests:
            raise PairedRegressionError("score ledger rows are duplicated")
        case_ids.add(row.case_id_sha256)
        content_digests.add(row.content_sha256)
    if tuple(sorted(ledger.rows, key=lambda item: item.case_id_sha256)) != ledger.rows:
        raise PairedRegressionError("score ledger rows are not canonical")
    if {row.label for row in ledger.rows} != {
        EntailmentLabel.SUPPORTED.value,
        EntailmentLabel.UNSUPPORTED.value,
    }:
        raise PairedRegressionError("score ledger must contain both labels")
    if ledger.evaluation_sha256 != _evaluation_digest(ledger.rows):
        raise PairedRegressionError("evaluation commitment mismatch")
    expected = _sha256(
        {
            key: value
            for key, value in ledger.to_audit_dict().items()
            if key != "artifact_sha256"
        }
    )
    if ledger.artifact_sha256 != expected:
        raise PairedRegressionError("score ledger artifact digest mismatch")


def load_evaluation_score_ledger(
    document: Mapping[str, object],
) -> EvaluationScoreLedger:
    """Load and integrity-check a score ledger decoded from canonical JSON."""

    if not isinstance(document, Mapping) or set(document) != _LEDGER_FIELDS:
        raise PairedRegressionError("score ledger is malformed")
    raw_rows = document.get("rows")
    if not isinstance(raw_rows, list) or not 2 <= len(raw_rows) <= _MAX_CASES:
        raise PairedRegressionError("score ledger is malformed")
    rows: list[EvaluationScoreRow] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != _ROW_FIELDS:
            raise PairedRegressionError("score ledger is malformed")
        try:
            rows.append(EvaluationScoreRow(**raw))
        except TypeError as exc:
            raise PairedRegressionError("score ledger is malformed") from exc
    try:
        ledger = EvaluationScoreLedger(
            schema_version=document["schema_version"],
            report_schema_version=document["report_schema_version"],
            report_sha256=document["report_sha256"],
            evaluation_sha256=document["evaluation_sha256"],
            scorer_id=document["scorer_id"],
            threshold=document["threshold"],
            policy_sha256=document["policy_sha256"],
            rows=tuple(rows),
            artifact_sha256=document["artifact_sha256"],
        )
    except TypeError as exc:
        raise PairedRegressionError("score ledger is malformed") from exc
    _validate_ledger(ledger)
    return ledger


class _DeterministicSampler:
    def __init__(self, seed_hex: str) -> None:
        self._seed = bytes.fromhex(seed_hex)
        self._counter = 0

    def index(self, size: int) -> int:
        limit = (1 << 256) - ((1 << 256) % size)
        while True:
            digest = hashlib.sha256(
                self._seed + self._counter.to_bytes(16, "big")
            ).digest()
            self._counter += 1
            value = int.from_bytes(digest, "big")
            if value < limit:
                return value % size


def _upper_quantile(values: list[float], probability: float) -> float:
    values.sort()
    index = min(len(values) - 1, math.ceil(probability * len(values)) - 1)
    return values[index]


def compare_paired_score_ledgers(
    baseline: EvaluationScoreLedger,
    candidate: EvaluationScoreLedger,
    policy: PairedBootstrapPolicy,
) -> PairedRegressionComparison:
    """Run a deterministic stratified paired bootstrap with FWER control."""

    if not isinstance(baseline, EvaluationScoreLedger):
        raise TypeError("baseline must be an EvaluationScoreLedger")
    if not isinstance(candidate, EvaluationScoreLedger):
        raise TypeError("candidate must be an EvaluationScoreLedger")
    if not isinstance(policy, PairedBootstrapPolicy):
        raise TypeError("policy must be a PairedBootstrapPolicy")
    _validate_ledger(baseline)
    _validate_ledger(candidate)
    if baseline.report_schema_version != candidate.report_schema_version:
        raise PairedRegressionError("report schemas are incompatible")
    if baseline.policy_sha256 != candidate.policy_sha256:
        raise PairedRegressionError("entailment policies are incompatible")
    baseline_population = tuple(
        (row.case_id_sha256, row.content_sha256, row.label) for row in baseline.rows
    )
    candidate_population = tuple(
        (row.case_id_sha256, row.content_sha256, row.label) for row in candidate.rows
    )
    if baseline_population != candidate_population:
        raise PairedRegressionError("score ledgers do not cover the same sealed cases")
    total_draws = len(baseline.rows) * policy.resamples
    if total_draws > _MAX_BOOTSTRAP_DRAWS:
        raise PairedRegressionError(
            f"bootstrap draw budget exceeded: {total_draws} > {_MAX_BOOTSTRAP_DRAWS}"
        )
    policy_sha256 = _sha256(asdict(policy))
    seed_sha256 = _sha256(
        {
            "baseline_ledger_sha256": baseline.artifact_sha256,
            "candidate_ledger_sha256": candidate.artifact_sha256,
            "method": PAIRED_BOOTSTRAP_METHOD_VERSION,
            "policy_sha256": policy_sha256,
        }
    )
    sampler = _DeterministicSampler(seed_sha256)
    strata = [
        [index for index, row in enumerate(baseline.rows) if row.label == label.value]
        for label in EntailmentLabel
    ]
    baseline_metrics = _metrics(baseline.rows, baseline.threshold)
    candidate_metrics = _metrics(candidate.rows, candidate.threshold)
    bootstrap_deltas = {name: [] for name in _METRIC_NAMES}
    for _ in range(policy.resamples):
        sample_indices = [
            stratum[sampler.index(len(stratum))]
            for stratum in strata
            for _ in range(len(stratum))
        ]
        baseline_sample = [baseline.rows[index] for index in sample_indices]
        candidate_sample = [candidate.rows[index] for index in sample_indices]
        baseline_sample_metrics = _metrics(baseline_sample, baseline.threshold)
        candidate_sample_metrics = _metrics(candidate_sample, candidate.threshold)
        for metric in _METRIC_NAMES:
            bootstrap_deltas[metric].append(
                getattr(candidate_sample_metrics, metric)
                - getattr(baseline_sample_metrics, metric)
            )
    hypotheses = len(_METRIC_NAMES)
    alpha = 1 - float(policy.confidence_level)
    per_hypothesis_confidence = 1 - alpha / hypotheses
    comparisons: list[PairedMetricComparison] = []
    failure_reasons: list[str] = []
    for metric in _METRIC_NAMES:
        baseline_value = float(getattr(baseline_metrics, metric))
        candidate_value = float(getattr(candidate_metrics, metric))
        adjusted_upper = _upper_quantile(
            bootstrap_deltas[metric], per_hypothesis_confidence
        )
        max_drop = policy.thresholds.for_metric(metric)
        regressed = adjusted_upper < -max_drop
        if regressed:
            failure_reasons.append(f"global_{metric}_regressed")
        comparisons.append(
            PairedMetricComparison(
                metric=metric,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                point_delta=candidate_value - baseline_value,
                adjusted_delta_upper=adjusted_upper,
                max_drop=max_drop,
                regressed=regressed,
            )
        )
    document = {
        "schema_version": PAIRED_COMPARISON_SCHEMA_VERSION,
        "bootstrap_method_version": PAIRED_BOOTSTRAP_METHOD_VERSION,
        "baseline_ledger_sha256": baseline.artifact_sha256,
        "candidate_ledger_sha256": candidate.artifact_sha256,
        "baseline_report_sha256": baseline.report_sha256,
        "candidate_report_sha256": candidate.report_sha256,
        "confidence_level": float(policy.confidence_level),
        "family_wise_method": policy.family_wise_method,
        "hypotheses": hypotheses,
        "per_hypothesis_confidence_level": per_hypothesis_confidence,
        "resamples": policy.resamples,
        "bootstrap_seed_sha256": seed_sha256,
        "policy_sha256": policy_sha256,
        "metrics": [asdict(item) for item in comparisons],
        "meets_policy": not failure_reasons,
        "failure_reasons": failure_reasons,
    }
    return PairedRegressionComparison(
        schema_version=PAIRED_COMPARISON_SCHEMA_VERSION,
        bootstrap_method_version=PAIRED_BOOTSTRAP_METHOD_VERSION,
        baseline_ledger_sha256=baseline.artifact_sha256,
        candidate_ledger_sha256=candidate.artifact_sha256,
        baseline_report_sha256=baseline.report_sha256,
        candidate_report_sha256=candidate.report_sha256,
        confidence_level=float(policy.confidence_level),
        family_wise_method=policy.family_wise_method,
        hypotheses=hypotheses,
        per_hypothesis_confidence_level=per_hypothesis_confidence,
        resamples=policy.resamples,
        bootstrap_seed_sha256=seed_sha256,
        policy_sha256=policy_sha256,
        metrics=tuple(comparisons),
        meets_policy=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        comparison_sha256=_sha256(document),
    )
