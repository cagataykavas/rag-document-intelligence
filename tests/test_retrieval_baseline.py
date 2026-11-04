from dataclasses import asdict, replace

import pytest

from src.retrieval_baseline import (
    BASELINE_SCHEMA_VERSION,
    COMPARISON_SCHEMA_VERSION,
    RetrievalBaselineError,
    RetrievalRegressionThresholds,
    compare_retrieval_baseline,
    create_retrieval_baseline,
    load_retrieval_baseline,
    load_retrieval_baseline_json,
    serialize_retrieval_baseline,
)
from src.retrieval_eval import RetrievalMetrics

CORPUS_SHA256 = "1" * 64
CASES_SHA256 = "2" * 64
INVENTORY_SHA256 = "4" * 64
SOURCE_COMMIT = "3" * 40


def _metrics(**overrides: float | int) -> RetrievalMetrics:
    values: dict[str, float | int] = {
        "cases": 20,
        "hit_rate_at_k": 0.9,
        "recall_at_k": 0.8,
        "precision_at_k": 0.4,
        "mean_reciprocal_rank": 0.7,
        "ndcg_at_k": 0.75,
        "mean_unique_documents_at_k": 3.5,
        "duplicate_document_rate_at_k": 0.2,
        "mean_largest_document_share_at_k": 0.4,
    }
    values.update(overrides)
    return RetrievalMetrics(**values)


def _thresholds(**overrides: float) -> RetrievalRegressionThresholds:
    values = {
        "max_hit_rate_at_k_regression": 0.02,
        "max_recall_at_k_regression": 0.03,
        "max_precision_at_k_regression": 0.02,
        "max_mean_reciprocal_rank_regression": 0.03,
        "max_ndcg_at_k_regression": 0.025,
        "max_mean_unique_documents_at_k_regression": 0.25,
        "max_duplicate_document_rate_at_k_regression": 0.02,
        "max_mean_largest_document_share_at_k_regression": 0.03,
    }
    values.update(overrides)
    return RetrievalRegressionThresholds(**values)


def _baseline():
    return create_retrieval_baseline(
        _metrics(),
        _thresholds(),
        baseline_id="retrieval-prod-k5",
        baseline_version=1,
        source_commit=SOURCE_COMMIT,
        k=5,
        corpus_snapshot_sha256=CORPUS_SHA256,
        evaluation_cases_sha256=CASES_SHA256,
        source_inventory_sha256=INVENTORY_SHA256,
    )


def _compare(candidate: RetrievalMetrics):
    return compare_retrieval_baseline(
        _baseline(),
        candidate,
        k=5,
        corpus_snapshot_sha256=CORPUS_SHA256,
        evaluation_cases_sha256=CASES_SHA256,
        source_inventory_sha256=INVENTORY_SHA256,
    )


def test_baseline_is_deterministic_content_addressed_and_privacy_reduced() -> None:
    first = _baseline()
    second = _baseline()

    assert first == second
    assert first.schema_version == BASELINE_SCHEMA_VERSION
    assert len(first.artifact_sha256) == 64
    serialized = str(first.to_audit_dict())
    assert "query" not in serialized
    assert "doc_id" not in serialized
    assert "chunk" not in serialized


def test_baseline_round_trip_rejects_tampering_and_extra_fields() -> None:
    baseline = _baseline()
    document = baseline.to_audit_dict()

    assert load_retrieval_baseline(document) == baseline

    tampered = {**document, "k": 4}
    with pytest.raises(RetrievalBaselineError, match="digest mismatch"):
        load_retrieval_baseline(tampered)

    legacy = {**document, "schema_version": "retrieval-regression-baseline/v1"}
    with pytest.raises(RetrievalBaselineError, match="schema is unsupported"):
        load_retrieval_baseline(legacy)

    with pytest.raises(RetrievalBaselineError, match="malformed"):
        load_retrieval_baseline({**document, "unreviewed": True})


def test_canonical_json_round_trip_rejects_whitespace_and_duplicate_keys() -> None:
    baseline = _baseline()
    payload = serialize_retrieval_baseline(baseline)

    assert load_retrieval_baseline_json(payload) == baseline
    with pytest.raises(RetrievalBaselineError, match="not canonical"):
        load_retrieval_baseline_json(payload + b"\n")
    with pytest.raises(RetrievalBaselineError, match="duplicate keys"):
        load_retrieval_baseline_json(
            b'{"schema_version":"first","schema_version":"second"}'
        )


@pytest.mark.parametrize("payload", [b"", b"{" + b" " * (64 * 1024)])
def test_baseline_json_is_bounded(payload: bytes) -> None:
    with pytest.raises(RetrievalBaselineError, match="size limit"):
        load_retrieval_baseline_json(payload)


def test_improvements_and_regressions_within_budget_pass() -> None:
    comparison = _compare(
        _metrics(
            hit_rate_at_k=0.92,
            recall_at_k=0.78,
            mean_unique_documents_at_k=3.25,
            duplicate_document_rate_at_k=0.18,
            mean_largest_document_share_at_k=0.42,
        )
    )

    assert comparison.schema_version == COMPARISON_SCHEMA_VERSION
    assert comparison.meets_policy is True
    assert comparison.failure_reasons == ()
    assert len(comparison.metrics) == 8
    assert len(comparison.comparison_sha256) == 64


@pytest.mark.parametrize(
    ("metric", "candidate"),
    [
        ("hit_rate_at_k", 0.87),
        ("mean_unique_documents_at_k", 3.0),
        ("duplicate_document_rate_at_k", 0.23),
        ("mean_largest_document_share_at_k", 0.44),
    ],
)
def test_gate_handles_both_metric_directions(metric: str, candidate: float) -> None:
    comparison = _compare(_metrics(**{metric: candidate}))

    assert comparison.meets_policy is False
    assert comparison.failure_reasons == (f"metric_regressed:{metric}",)
    failed = next(item for item in comparison.metrics if item.metric == metric)
    assert failed.regressed is True
    assert failed.observed_regression > failed.max_regression


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"k": 4}, "candidate k"),
        ({"corpus_snapshot_sha256": "4" * 64}, "corpus snapshot"),
        ({"evaluation_cases_sha256": "5" * 64}, "evaluation cases"),
        ({"source_inventory_sha256": "6" * 64}, "source inventory"),
    ],
)
def test_comparison_requires_exact_evaluation_identity(
    kwargs: dict, reason: str
) -> None:
    call = {
        "k": 5,
        "corpus_snapshot_sha256": CORPUS_SHA256,
        "evaluation_cases_sha256": CASES_SHA256,
        "source_inventory_sha256": INVENTORY_SHA256,
    }
    call.update(kwargs)

    with pytest.raises(RetrievalBaselineError, match=reason):
        compare_retrieval_baseline(_baseline(), _metrics(), **call)


def test_comparison_rejects_case_count_drift() -> None:
    with pytest.raises(RetrievalBaselineError, match="case count"):
        _compare(_metrics(cases=19))


@pytest.mark.parametrize(
    "metrics",
    [
        _metrics(hit_rate_at_k=float("nan")),
        _metrics(recall_at_k=True),
        _metrics(mean_unique_documents_at_k=5.1),
        _metrics(cases=0),
    ],
)
def test_non_finite_boolean_out_of_range_and_empty_metrics_fail_closed(
    metrics: RetrievalMetrics,
) -> None:
    with pytest.raises(RetrievalBaselineError, match="malformed"):
        create_retrieval_baseline(
            metrics,
            _thresholds(),
            baseline_id="bad",
            baseline_version=1,
            source_commit=SOURCE_COMMIT,
            k=5,
            corpus_snapshot_sha256=CORPUS_SHA256,
            evaluation_cases_sha256=CASES_SHA256,
            source_inventory_sha256=INVENTORY_SHA256,
        )


def test_every_metric_requires_an_explicit_reviewed_threshold() -> None:
    document = _baseline().to_audit_dict()
    thresholds = dict(document["thresholds"])
    thresholds.pop("max_ndcg_at_k_regression")
    document["thresholds"] = thresholds

    with pytest.raises(RetrievalBaselineError, match="malformed"):
        load_retrieval_baseline(document)


def test_comparison_revalidates_in_memory_baseline_digest() -> None:
    tampered = replace(
        _baseline(),
        thresholds=_thresholds(max_recall_at_k_regression=0.5),
    )

    with pytest.raises(RetrievalBaselineError, match="digest mismatch"):
        compare_retrieval_baseline(
            tampered,
            _metrics(),
            k=5,
            corpus_snapshot_sha256=CORPUS_SHA256,
            evaluation_cases_sha256=CASES_SHA256,
            source_inventory_sha256=INVENTORY_SHA256,
        )


def test_candidate_commitment_and_comparison_digest_change_with_metrics() -> None:
    first = _compare(_metrics())
    second = _compare(_metrics(ndcg_at_k=0.74))

    assert first.candidate_metrics_sha256 != second.candidate_metrics_sha256
    assert first.comparison_sha256 != second.comparison_sha256
    assert asdict(first)["baseline_artifact_sha256"] == _baseline().artifact_sha256
