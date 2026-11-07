from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean

from src.retrieval_baseline import RetrievalRegressionThresholds
from src.retrieval_eval import RetrievalCase, RetrievalMetrics, evaluate_rankings
from src.retrieval_gate import (
    CanonicalRetrievalCase,
    RetrievalCandidateSnapshot,
    validate_candidate_snapshot,
)

SCORE_LEDGER_SCHEMA_VERSION = "retrieval-paired-score-ledger/v1"
PAIRED_COMPARISON_SCHEMA_VERSION = "retrieval-paired-comparison/v1"
PAIRED_BOOTSTRAP_METHOD_VERSION = "paired-bootstrap-bonferroni/v1"
_CASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
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
_ROW_FIELDS = {"case_id_sha256", "case_content_sha256", "ranking_sha256", *_METRIC_NAMES}
_LEDGER_FIELDS = {
    "schema_version",
    "candidate_snapshot_sha256",
    "corpus_snapshot_sha256",
    "evaluation_cases_sha256",
    "source_inventory_sha256",
    "k",
    "rows",
    "artifact_sha256",
}
_MAX_CASES = 10_000
_MAX_BOOTSTRAP_DRAWS = 5_000_000


class RetrievalPairedRegressionError(ValueError):
    """A retrieval score ledger or paired comparison is unsafe or incompatible."""

    code = "paired_retrieval_regression_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class RetrievalScoreRow:
    case_id_sha256: str
    case_content_sha256: str
    ranking_sha256: str
    hit_rate_at_k: float
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    mean_unique_documents_at_k: float
    duplicate_document_rate_at_k: float
    mean_largest_document_share_at_k: float


@dataclass(frozen=True, slots=True)
class RetrievalScoreLedger:
    schema_version: str
    candidate_snapshot_sha256: str
    corpus_snapshot_sha256: str
    evaluation_cases_sha256: str
    source_inventory_sha256: str
    k: int
    rows: tuple[RetrievalScoreRow, ...]
    artifact_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalPairedBootstrapPolicy:
    confidence_level: float
    resamples: int
    thresholds: RetrievalRegressionThresholds
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
        if not isinstance(self.thresholds, RetrievalRegressionThresholds):
            raise TypeError("thresholds must be RetrievalRegressionThresholds")
        if self.family_wise_method != "bonferroni":
            raise ValueError("family_wise_method must be 'bonferroni'")


@dataclass(frozen=True, slots=True)
class RetrievalPairedMetricComparison:
    metric: str
    direction: str
    baseline_value: float
    candidate_value: float
    improvement_delta: float
    adjusted_delta_upper: float
    max_regression: float
    regressed: bool


@dataclass(frozen=True, slots=True)
class RetrievalPairedComparison:
    schema_version: str
    bootstrap_method_version: str
    baseline_ledger_sha256: str
    candidate_ledger_sha256: str
    confidence_level: float
    family_wise_method: str
    hypotheses: int
    per_hypothesis_confidence_level: float
    resamples: int
    bootstrap_seed_sha256: str
    policy_sha256: str
    metrics: tuple[RetrievalPairedMetricComparison, ...]
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
        raise RetrievalPairedRegressionError(
            f"{name} must be canonical lowercase SHA-256"
        )
    return value


def _case_content_sha256(case: CanonicalRetrievalCase) -> str:
    payload = json.dumps(
        {"query": case.query, "relevant_doc_ids": sorted(case.relevant_doc_ids)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ranking_sha256(results: Sequence[Mapping[str, object]], k: int) -> str:
    doc_ids: list[str] = []
    for item in results[:k]:
        if not isinstance(item, Mapping):
            raise RetrievalPairedRegressionError("ranking row is malformed")
        doc_id = item.get("doc_id")
        if not isinstance(doc_id, str) or not 1 <= len(doc_id) <= 256:
            raise RetrievalPairedRegressionError("ranking document ID is malformed")
        doc_ids.append(doc_id)
    return _sha256(doc_ids)


def _aggregate_metrics(rows: Sequence[RetrievalScoreRow]) -> RetrievalMetrics:
    return RetrievalMetrics(
        cases=len(rows),
        **{name: mean(getattr(row, name) for row in rows) for name in _METRIC_NAMES},
    )


def create_retrieval_score_ledger(
    snapshot: RetrievalCandidateSnapshot,
    cases: Sequence[CanonicalRetrievalCase],
    rankings: Sequence[Sequence[Mapping[str, object]]],
) -> RetrievalScoreLedger:
    """Seal privacy-reduced case scores for one validated retrieval snapshot."""

    validate_candidate_snapshot(snapshot)
    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise TypeError("cases must be a sequence of CanonicalRetrievalCase values")
    if isinstance(rankings, (str, bytes)) or not isinstance(rankings, Sequence):
        raise TypeError("rankings must be a sequence")
    if not 2 <= len(cases) <= _MAX_CASES or len(rankings) != len(cases):
        raise RetrievalPairedRegressionError(
            "cases and rankings must have equal size in [2, 10000]"
        )
    if len(cases) != snapshot.case_count:
        raise RetrievalPairedRegressionError("case count does not match snapshot")
    rows: list[RetrievalScoreRow] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    for case, results in zip(cases, rankings, strict=True):
        if not isinstance(case, CanonicalRetrievalCase):
            raise TypeError("cases contains an unsupported value")
        if _CASE_ID_RE.fullmatch(case.case_id) is None or case.case_id in seen_ids:
            raise RetrievalPairedRegressionError("case IDs are malformed or duplicated")
        if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
            raise RetrievalPairedRegressionError("ranking is malformed")
        content_sha256 = _case_content_sha256(case)
        if content_sha256 in seen_content:
            raise RetrievalPairedRegressionError("case content is duplicated")
        seen_ids.add(case.case_id)
        seen_content.add(content_sha256)
        retrieval_case = RetrievalCase(case.query, frozenset(case.relevant_doc_ids))
        metrics = evaluate_rankings([list(results)], [retrieval_case], k=snapshot.k)
        rows.append(
            RetrievalScoreRow(
                case_id_sha256=hashlib.sha256(case.case_id.encode("utf-8")).hexdigest(),
                case_content_sha256=content_sha256,
                ranking_sha256=_ranking_sha256(results, snapshot.k),
                **{name: float(getattr(metrics, name)) for name in _METRIC_NAMES},
            )
        )
    rows.sort(key=lambda item: item.case_id_sha256)
    if _aggregate_metrics(rows) != snapshot.metrics:
        raise RetrievalPairedRegressionError(
            "case scores do not reproduce snapshot metrics"
        )
    document = {
        "schema_version": SCORE_LEDGER_SCHEMA_VERSION,
        "candidate_snapshot_sha256": snapshot.artifact_sha256,
        "corpus_snapshot_sha256": snapshot.corpus_snapshot_sha256,
        "evaluation_cases_sha256": snapshot.evaluation_cases_sha256,
        "source_inventory_sha256": snapshot.source_inventory_sha256,
        "k": snapshot.k,
        "rows": [asdict(row) for row in rows],
    }
    return RetrievalScoreLedger(
        schema_version=SCORE_LEDGER_SCHEMA_VERSION,
        candidate_snapshot_sha256=snapshot.artifact_sha256,
        corpus_snapshot_sha256=snapshot.corpus_snapshot_sha256,
        evaluation_cases_sha256=snapshot.evaluation_cases_sha256,
        source_inventory_sha256=snapshot.source_inventory_sha256,
        k=snapshot.k,
        rows=tuple(rows),
        artifact_sha256=_sha256(document),
    )


def _validate_row(row: RetrievalScoreRow, k: int) -> None:
    if not isinstance(row, RetrievalScoreRow):
        raise RetrievalPairedRegressionError("score ledger row is malformed")
    _validate_digest(row.case_id_sha256, "case ID commitment")
    _validate_digest(row.case_content_sha256, "case content commitment")
    _validate_digest(row.ranking_sha256, "ranking commitment")
    for name in _METRIC_NAMES:
        value = getattr(row, name)
        upper = float(k) if name == "mean_unique_documents_at_k" else 1.0
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= float(value) <= upper
        ):
            raise RetrievalPairedRegressionError(f"score ledger {name} is malformed")


def _validate_ledger(ledger: RetrievalScoreLedger) -> None:
    if ledger.schema_version != SCORE_LEDGER_SCHEMA_VERSION:
        raise RetrievalPairedRegressionError("score ledger schema is unsupported")
    if type(ledger.k) is not int or not 1 <= ledger.k <= 1_000:
        raise RetrievalPairedRegressionError("score ledger k is malformed")
    if not isinstance(ledger.rows, tuple) or not 2 <= len(ledger.rows) <= _MAX_CASES:
        raise RetrievalPairedRegressionError("score ledger rows are malformed")
    for value, name in (
        (ledger.candidate_snapshot_sha256, "candidate snapshot commitment"),
        (ledger.corpus_snapshot_sha256, "corpus snapshot commitment"),
        (ledger.evaluation_cases_sha256, "evaluation cases commitment"),
        (ledger.source_inventory_sha256, "source inventory commitment"),
        (ledger.artifact_sha256, "artifact commitment"),
    ):
        _validate_digest(value, name)
    for row in ledger.rows:
        _validate_row(row, ledger.k)
    if tuple(sorted(ledger.rows, key=lambda item: item.case_id_sha256)) != ledger.rows:
        raise RetrievalPairedRegressionError("score ledger rows are not canonical")
    case_ids = {row.case_id_sha256 for row in ledger.rows}
    case_contents = {row.case_content_sha256 for row in ledger.rows}
    if len(case_ids) != len(ledger.rows) or len(case_contents) != len(ledger.rows):
        raise RetrievalPairedRegressionError("score ledger rows are duplicated")
    expected = _sha256(
        {key: value for key, value in ledger.to_audit_dict().items() if key != "artifact_sha256"}
    )
    if ledger.artifact_sha256 != expected:
        raise RetrievalPairedRegressionError("score ledger artifact digest mismatch")


def load_retrieval_score_ledger(document: Mapping[str, object]) -> RetrievalScoreLedger:
    """Load and integrity-check a decoded score ledger."""

    if not isinstance(document, Mapping) or set(document) != _LEDGER_FIELDS:
        raise RetrievalPairedRegressionError("score ledger is malformed")
    raw_rows = document.get("rows")
    if not isinstance(raw_rows, list) or not 2 <= len(raw_rows) <= _MAX_CASES:
        raise RetrievalPairedRegressionError("score ledger is malformed")
    rows: list[RetrievalScoreRow] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != _ROW_FIELDS:
            raise RetrievalPairedRegressionError("score ledger is malformed")
        try:
            rows.append(RetrievalScoreRow(**raw))
        except TypeError as exc:
            raise RetrievalPairedRegressionError("score ledger is malformed") from exc
    try:
        ledger = RetrievalScoreLedger(
            schema_version=document["schema_version"],
            candidate_snapshot_sha256=document["candidate_snapshot_sha256"],
            corpus_snapshot_sha256=document["corpus_snapshot_sha256"],
            evaluation_cases_sha256=document["evaluation_cases_sha256"],
            source_inventory_sha256=document["source_inventory_sha256"],
            k=document["k"],
            rows=tuple(rows),
            artifact_sha256=document["artifact_sha256"],
        )
    except TypeError as exc:
        raise RetrievalPairedRegressionError("score ledger is malformed") from exc
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


def compare_paired_retrieval_ledgers(
    baseline: RetrievalScoreLedger,
    candidate: RetrievalScoreLedger,
    policy: RetrievalPairedBootstrapPolicy,
) -> RetrievalPairedComparison:
    """Run a deterministic case-paired bootstrap with family-wise error control."""

    if not isinstance(baseline, RetrievalScoreLedger):
        raise TypeError("baseline must be a RetrievalScoreLedger")
    if not isinstance(candidate, RetrievalScoreLedger):
        raise TypeError("candidate must be a RetrievalScoreLedger")
    if not isinstance(policy, RetrievalPairedBootstrapPolicy):
        raise TypeError("policy must be a RetrievalPairedBootstrapPolicy")
    _validate_ledger(baseline)
    _validate_ledger(candidate)
    if (
        baseline.k,
        baseline.corpus_snapshot_sha256,
        baseline.evaluation_cases_sha256,
        baseline.source_inventory_sha256,
    ) != (
        candidate.k,
        candidate.corpus_snapshot_sha256,
        candidate.evaluation_cases_sha256,
        candidate.source_inventory_sha256,
    ):
        raise RetrievalPairedRegressionError("score ledger snapshots are incompatible")
    baseline_population = tuple(
        (row.case_id_sha256, row.case_content_sha256) for row in baseline.rows
    )
    candidate_population = tuple(
        (row.case_id_sha256, row.case_content_sha256) for row in candidate.rows
    )
    if baseline_population != candidate_population:
        raise RetrievalPairedRegressionError("score ledgers do not cover the same cases")
    if policy.thresholds.max_mean_unique_documents_at_k_regression > baseline.k:
        raise RetrievalPairedRegressionError("policy threshold is incompatible with k")
    total_draws = len(baseline.rows) * policy.resamples
    if total_draws > _MAX_BOOTSTRAP_DRAWS:
        raise RetrievalPairedRegressionError(
            f"bootstrap draw budget exceeded: {total_draws} > {_MAX_BOOTSTRAP_DRAWS}"
        )
    policy_sha256 = _sha256(asdict(policy))
    seed_sha256 = _sha256(
        {
            "evaluation_cases_sha256": baseline.evaluation_cases_sha256,
            "method": PAIRED_BOOTSTRAP_METHOD_VERSION,
            "policy_sha256": policy_sha256,
        }
    )
    sampler = _DeterministicSampler(seed_sha256)
    bootstrap_deltas = {name: [] for name in _METRIC_NAMES}
    for _ in range(policy.resamples):
        indices = [sampler.index(len(baseline.rows)) for _ in baseline.rows]
        for metric in _METRIC_NAMES:
            baseline_value = mean(getattr(baseline.rows[index], metric) for index in indices)
            candidate_value = mean(getattr(candidate.rows[index], metric) for index in indices)
            delta = candidate_value - baseline_value
            if metric in _LOWER_IS_BETTER:
                delta = -delta
            bootstrap_deltas[metric].append(delta)
    hypotheses = len(_METRIC_NAMES)
    per_hypothesis_confidence = 1 - (1 - float(policy.confidence_level)) / hypotheses
    baseline_metrics = _aggregate_metrics(baseline.rows)
    candidate_metrics = _aggregate_metrics(candidate.rows)
    comparisons: list[RetrievalPairedMetricComparison] = []
    failure_reasons: list[str] = []
    for metric in _METRIC_NAMES:
        baseline_value = float(getattr(baseline_metrics, metric))
        candidate_value = float(getattr(candidate_metrics, metric))
        improvement_delta = candidate_value - baseline_value
        direction = "higher_is_better"
        if metric in _LOWER_IS_BETTER:
            improvement_delta = -improvement_delta
            direction = "lower_is_better"
        adjusted_upper = _upper_quantile(
            bootstrap_deltas[metric], per_hypothesis_confidence
        )
        max_regression = policy.thresholds.for_metric(metric)
        regressed = adjusted_upper < -max_regression
        if regressed:
            failure_reasons.append(f"metric_regressed:{metric}")
        comparisons.append(
            RetrievalPairedMetricComparison(
                metric=metric,
                direction=direction,
                baseline_value=baseline_value,
                candidate_value=candidate_value,
                improvement_delta=improvement_delta,
                adjusted_delta_upper=adjusted_upper,
                max_regression=max_regression,
                regressed=regressed,
            )
        )
    document = {
        "schema_version": PAIRED_COMPARISON_SCHEMA_VERSION,
        "bootstrap_method_version": PAIRED_BOOTSTRAP_METHOD_VERSION,
        "baseline_ledger_sha256": baseline.artifact_sha256,
        "candidate_ledger_sha256": candidate.artifact_sha256,
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
    return RetrievalPairedComparison(
        schema_version=PAIRED_COMPARISON_SCHEMA_VERSION,
        bootstrap_method_version=PAIRED_BOOTSTRAP_METHOD_VERSION,
        baseline_ledger_sha256=baseline.artifact_sha256,
        candidate_ledger_sha256=candidate.artifact_sha256,
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
