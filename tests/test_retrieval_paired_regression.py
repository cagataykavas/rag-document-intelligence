import hashlib
import json
from dataclasses import asdict, replace

import pytest

from src.retrieval_baseline import RetrievalRegressionThresholds
from src.retrieval_eval import RetrievalCase, evaluate_rankings
from src.retrieval_gate import (
    SNAPSHOT_SCHEMA_VERSION,
    CanonicalRetrievalCase,
    RetrievalCandidateSnapshot,
)
from src.retrieval_paired_regression import (
    PAIRED_BOOTSTRAP_METHOD_VERSION,
    PAIRED_COMPARISON_SCHEMA_VERSION,
    RetrievalPairedBootstrapPolicy,
    RetrievalPairedRegressionError,
    compare_paired_retrieval_ledgers,
    create_retrieval_score_ledger,
    load_retrieval_score_ledger,
)

CORPUS_SHA256 = "1" * 64
CASES_SHA256 = "2" * 64
INVENTORY_SHA256 = "3" * 64


def _sha256(value: object, *, domain: bytes = b"") -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain + (b"\0" if domain else b"") + payload).hexdigest()


def _cases(count: int = 20) -> list[CanonicalRetrievalCase]:
    return [
        CanonicalRetrievalCase(f"case-{index:03d}", f"query {index}", (f"rel-{index}",))
        for index in range(count)
    ]


def _rankings(
    cases: list[CanonicalRetrievalCase], *, degraded: bool = False
) -> list[list[dict[str, str]]]:
    if degraded:
        return [[{"doc_id": "noise"}] * 3 for _ in cases]
    return [
        [
            {"doc_id": case.relevant_doc_ids[0]},
            {"doc_id": f"noise-a-{index}"},
            {"doc_id": f"noise-b-{index}"},
        ]
        for index, case in enumerate(cases)
    ]


def _snapshot(
    cases: list[CanonicalRetrievalCase], rankings: list[list[dict[str, str]]]
) -> RetrievalCandidateSnapshot:
    metrics = evaluate_rankings(
        rankings,
        [
            RetrievalCase(case.query, frozenset(case.relevant_doc_ids))
            for case in cases
        ],
        k=3,
    )
    document = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "k": 3,
        "document_count": 100,
        "chunk_count": 100,
        "case_count": len(cases),
        "corpus_snapshot_sha256": CORPUS_SHA256,
        "evaluation_cases_sha256": CASES_SHA256,
        "source_inventory_sha256": INVENTORY_SHA256,
        "metrics": asdict(metrics),
    }
    return RetrievalCandidateSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        k=3,
        document_count=100,
        chunk_count=100,
        case_count=len(cases),
        corpus_snapshot_sha256=CORPUS_SHA256,
        evaluation_cases_sha256=CASES_SHA256,
        source_inventory_sha256=INVENTORY_SHA256,
        metrics=metrics,
        artifact_sha256=_sha256(
            document, domain=b"retrieval-candidate-snapshot/v2"
        ),
    )


def _ledger(*, degraded: bool = False, count: int = 20):
    cases = _cases(count)
    rankings = _rankings(cases, degraded=degraded)
    return create_retrieval_score_ledger(_snapshot(cases, rankings), cases, rankings)


def _thresholds(value: float = 0.0) -> RetrievalRegressionThresholds:
    return RetrievalRegressionThresholds(
        max_hit_rate_at_k_regression=value,
        max_recall_at_k_regression=value,
        max_precision_at_k_regression=value,
        max_mean_reciprocal_rank_regression=value,
        max_ndcg_at_k_regression=value,
        max_mean_unique_documents_at_k_regression=value,
        max_duplicate_document_rate_at_k_regression=value,
        max_mean_largest_document_share_at_k_regression=value,
    )


def _policy(*, resamples: int = 1_000) -> RetrievalPairedBootstrapPolicy:
    return RetrievalPairedBootstrapPolicy(0.95, resamples, _thresholds())


def test_score_ledger_is_deterministic_bound_and_privacy_reduced() -> None:
    first = _ledger()
    second = _ledger()

    assert first == second
    assert len(first.rows) == 20
    assert len(first.artifact_sha256) == 64
    decoded = json.loads(json.dumps(first.to_audit_dict()))
    assert load_retrieval_score_ledger(decoded) == first
    serialized = json.dumps(first.to_audit_dict())
    assert "query 0" not in serialized
    assert "rel-0" not in serialized
    assert "noise-a" not in serialized


def test_ledger_rejects_scores_that_do_not_reproduce_snapshot() -> None:
    cases = _cases()
    good = _rankings(cases)
    bad = _rankings(cases, degraded=True)

    with pytest.raises(RetrievalPairedRegressionError, match="reproduce"):
        create_retrieval_score_ledger(_snapshot(cases, good), cases, bad)


def test_loader_rejects_tampering_and_extra_fields() -> None:
    ledger = _ledger()
    document = json.loads(json.dumps(ledger.to_audit_dict()))
    document["rows"][0]["recall_at_k"] = 0.0

    with pytest.raises(RetrievalPairedRegressionError, match="digest mismatch"):
        load_retrieval_score_ledger(document)
    with pytest.raises(RetrievalPairedRegressionError, match="malformed"):
        load_retrieval_score_ledger({**ledger.to_audit_dict(), "unreviewed": True})


def test_identical_ledgers_pass_with_reproducible_fwer_evidence() -> None:
    ledger = _ledger()

    first = compare_paired_retrieval_ledgers(ledger, ledger, _policy())
    second = compare_paired_retrieval_ledgers(ledger, ledger, _policy())

    assert first == second
    assert first.schema_version == PAIRED_COMPARISON_SCHEMA_VERSION
    assert first.bootstrap_method_version == PAIRED_BOOTSTRAP_METHOD_VERSION
    assert first.hypotheses == 8
    assert first.per_hypothesis_confidence_level == pytest.approx(0.99375)
    assert first.meets_policy is True
    assert first.failure_reasons == ()
    assert all(item.adjusted_delta_upper == 0.0 for item in first.metrics)


def test_confident_quality_and_source_diversity_regressions_fail_closed() -> None:
    comparison = compare_paired_retrieval_ledgers(
        _ledger(), _ledger(degraded=True), _policy()
    )

    assert comparison.meets_policy is False
    assert "metric_regressed:recall_at_k" in comparison.failure_reasons
    assert (
        "metric_regressed:duplicate_document_rate_at_k"
        in comparison.failure_reasons
    )
    duplicate_rate = next(
        item
        for item in comparison.metrics
        if item.metric == "duplicate_document_rate_at_k"
    )
    assert duplicate_rate.direction == "lower_is_better"
    assert duplicate_rate.improvement_delta < 0
    assert duplicate_rate.regressed is True


def test_changed_case_population_is_not_treated_as_paired() -> None:
    baseline = _ledger()
    changed_cases = _cases()
    changed_cases[0] = replace(changed_cases[0], query="different query")
    rankings = _rankings(changed_cases)
    candidate = create_retrieval_score_ledger(
        _snapshot(changed_cases, rankings), changed_cases, rankings
    )

    with pytest.raises(RetrievalPairedRegressionError, match="same cases"):
        compare_paired_retrieval_ledgers(baseline, candidate, _policy())


def test_bootstrap_work_is_bounded_before_sampling() -> None:
    ledger = _ledger(count=51)

    with pytest.raises(RetrievalPairedRegressionError, match="budget exceeded"):
        compare_paired_retrieval_ledgers(
            ledger, ledger, _policy(resamples=100_000)
        )
