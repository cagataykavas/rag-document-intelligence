from pathlib import Path

import pytest

from src.engine import OpenAICompatibleGenerator, RAGEngine
from src.grounding import audit_grounding
from src.hybrid import HybridSparseRetriever
from src.parsers import ParsedDocument
from src.retrieval import Chunk
from src.retrieval_eval import (
    RetrievalCase,
    evaluate_rankings,
    evaluate_ranking_stability,
    gate_ranking_stability,
)
from src.store import DocumentStore


def test_hybrid_retriever_exposes_word_and_character_scores():
    chunks = [
        Chunk(
            "a:0",
            "a",
            "security.md",
            "REQ-42X administrative authentication is mandatory",
        ),
        Chunk(
            "b:0", "b", "latency.md", "API latency shall remain below 250 milliseconds"
        ),
    ]
    results = HybridSparseRetriever().fit(chunks).search("REQ42X authentication", k=2)
    assert results[0]["doc_id"] == "a"
    assert results[0]["word_score"] >= 0.0
    assert results[0]["character_score"] > 0.0
    assert results[0]["rank"] == 1


def test_grounding_audit_detects_fabricated_citation():
    retrieval = [
        {
            "chunk_id": "policy:1",
            "text": "Audit records are retained for seven years.",
        }
    ]
    audit = audit_grounding(
        "Audit records are retained for seven years.",
        ["policy:1", "invented:9"],
        retrieval,
    )
    assert audit.valid_citations == ("policy:1",)
    assert audit.invalid_citations == ("invented:9",)
    assert audit.citation_validity_rate == pytest.approx(0.5)
    assert audit.lexical_grounding_rate > 0.7


def test_ranked_retrieval_metrics_include_mrr_and_ndcg():
    cases = [RetrievalCase("query", frozenset({"good"}))]
    rankings = [[{"doc_id": "bad"}, {"doc_id": "good"}, {"doc_id": "other"}]]
    metrics = evaluate_rankings(rankings, cases, k=3)
    assert metrics.hit_rate_at_k == pytest.approx(1.0)
    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.mean_reciprocal_rank == pytest.approx(0.5)
    assert 0.0 < metrics.ndcg_at_k < 1.0


def stability_trial(*rows: tuple[str, float]) -> list[dict]:
    return [
        {"chunk_id": chunk_id, "score": score, "trace": "retained"}
        for chunk_id, score in rows
    ]


def test_identical_retrieval_trials_pass_strict_stability_gate():
    trials = [
        stability_trial(("a:0", 0.9), ("b:0", 0.8), ("c:0", 0.7)),
        stability_trial(("a:0", 0.9), ("b:0", 0.8), ("c:0", 0.7)),
        stability_trial(("a:0", 0.9), ("b:0", 0.8), ("c:0", 0.7)),
    ]

    metrics = evaluate_ranking_stability(trials, k=3)
    report = gate_ranking_stability(
        metrics,
        min_exact_order_pair_rate=1.0,
        min_mean_jaccard_at_k=1.0,
        min_pairwise_jaccard_at_k=1.0,
        max_common_score_delta=0.0,
    )

    assert metrics.pairs == 3
    assert report["decision"] == "pass"
    assert report["violations"] == []


def test_tie_order_flapping_is_visible_even_when_top_k_set_is_stable():
    metrics = evaluate_ranking_stability(
        [
            stability_trial(("a:0", 0.9), ("b:0", 0.8), ("c:0", 0.8)),
            stability_trial(("a:0", 0.9), ("c:0", 0.8), ("b:0", 0.8)),
        ],
        k=3,
    )
    report = gate_ranking_stability(
        metrics,
        min_exact_order_pair_rate=1.0,
        min_mean_jaccard_at_k=1.0,
        min_pairwise_jaccard_at_k=1.0,
        max_common_score_delta=0.0,
    )

    assert metrics.mean_jaccard_at_k == 1.0
    assert metrics.exact_order_pair_rate == 0.0
    assert [item["code"] for item in report["violations"]] == [
        "exact_order_rate_below_minimum"
    ]


def test_candidate_and_score_drift_have_separate_gate_reasons():
    metrics = evaluate_ranking_stability(
        [
            stability_trial(("a:0", 0.9), ("b:0", 0.8)),
            stability_trial(("a:0", 0.7), ("c:0", 0.6)),
        ],
        k=2,
    )
    report = gate_ranking_stability(
        metrics,
        min_exact_order_pair_rate=0.0,
        min_mean_jaccard_at_k=0.5,
        min_pairwise_jaccard_at_k=0.5,
        max_common_score_delta=0.1,
    )

    assert metrics.mean_jaccard_at_k == pytest.approx(1 / 3)
    assert metrics.maximum_common_score_delta == pytest.approx(0.2)
    assert [item["code"] for item in report["violations"]] == [
        "common_score_delta_above_maximum",
        "mean_jaccard_below_minimum",
        "pairwise_jaccard_below_minimum",
    ]


@pytest.mark.parametrize(
    ("trials", "k", "message"),
    [
        ([stability_trial(("a", 1.0))], 1, "between 2 and 20"),
        ([stability_trial(("a", 1.0)), stability_trial(("a", 1.0))], 0, "k must"),
        (
            [stability_trial(("a", 1.0), ("a", 0.9)), stability_trial(("a", 1.0))],
            1,
            "duplicate",
        ),
        (
            [
                stability_trial(("a", 0.8), ("b", 0.9)),
                stability_trial(("a", 0.8), ("b", 0.7)),
            ],
            2,
            "descending",
        ),
        (
            [stability_trial(("a", float("nan"))), stability_trial(("a", 1.0))],
            1,
            "invalid score",
        ),
    ],
)
def test_stability_evidence_fails_closed(trials, k: int, message: str):
    with pytest.raises(ValueError, match=message):
        evaluate_ranking_stability(trials, k=k)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True])
def test_stability_policy_rejects_invalid_rates(threshold):
    metrics = evaluate_ranking_stability(
        [stability_trial(("a", 1.0)), stability_trial(("a", 1.0))],
        k=1,
    )

    with pytest.raises(ValueError, match="minimum stability"):
        gate_ranking_stability(
            metrics,
            min_exact_order_pair_rate=threshold,
            min_mean_jaccard_at_k=1.0,
            min_pairwise_jaccard_at_k=1.0,
            max_common_score_delta=0.0,
        )


def test_engine_reports_rejected_llm_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = DocumentStore(tmp_path / "rag.db")
    store.ingest(
        ParsedDocument(
            "policy.md",
            "text/markdown",
            "Administrative endpoints require authenticated access and audit logging.",
            {},
        ),
        words_per_chunk=30,
        overlap=5,
    )
    valid_chunk_id = store.chunks()[0].chunk_id

    def fake_generate(_self, _prompt: str) -> dict:
        return {
            "answer": "Administrative endpoints require authenticated access.",
            "citations": [valid_chunk_id, "fabricated:404"],
            "confidence": 0.91,
            "insufficient_evidence": False,
        }

    monkeypatch.setattr(OpenAICompatibleGenerator, "generate", fake_generate)
    answer = RAGEngine(store).answer(
        "Do admin endpoints require authentication?", use_llm=True
    )
    assert answer.citations == (valid_chunk_id,)
    assert answer.rejected_citations == ("fabricated:404",)
    assert answer.citation_audit["citation_validity_rate"] == pytest.approx(0.5)
