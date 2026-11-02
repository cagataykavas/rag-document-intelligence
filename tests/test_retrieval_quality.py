from pathlib import Path

import pytest

from src.engine import OpenAICompatibleGenerator, RAGEngine
from src.grounding import audit_grounding
from src.hybrid import HybridSparseRetriever
from src.parsers import ParsedDocument
from src.retrieval import Chunk
from src.retrieval_eval import RetrievalCase, evaluate_rankings
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


def test_duplicate_chunks_from_one_relevant_document_receive_credit_once():
    cases = [RetrievalCase("query", frozenset({"policy"}))]
    rankings = [
        [
            {"doc_id": "policy", "chunk_id": "policy:0"},
            {"doc_id": "policy", "chunk_id": "policy:1"},
            {"doc_id": "policy", "chunk_id": "policy:2"},
        ]
    ]

    metrics = evaluate_rankings(rankings, cases, k=3)

    assert metrics.hit_rate_at_k == pytest.approx(1.0)
    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.precision_at_k == pytest.approx(1 / 3)
    assert metrics.mean_reciprocal_rank == pytest.approx(1.0)
    assert metrics.ndcg_at_k == pytest.approx(1.0)
    assert metrics.mean_unique_documents_at_k == pytest.approx(1.0)
    assert metrics.duplicate_document_rate_at_k == pytest.approx(2 / 3)
    assert metrics.mean_largest_document_share_at_k == pytest.approx(1.0)


def test_duplicate_chunk_consumes_rank_before_a_second_relevant_document():
    cases = [RetrievalCase("query", frozenset({"policy", "runbook"}))]
    rankings = [
        [
            {"doc_id": "policy", "chunk_id": "policy:0"},
            {"doc_id": "policy", "chunk_id": "policy:1"},
            {"doc_id": "runbook", "chunk_id": "runbook:0"},
        ]
    ]

    metrics = evaluate_rankings(rankings, cases, k=3)

    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.precision_at_k == pytest.approx(2 / 3)
    assert metrics.mean_reciprocal_rank == pytest.approx(1.0)
    assert 0.0 < metrics.ndcg_at_k < 1.0


def test_document_level_retrieval_metrics_remain_bounded_with_duplicate_chunks():
    cases = [RetrievalCase("query", frozenset({"policy"}))]
    rankings = [[{"doc_id": "policy"} for _ in range(20)]]

    metrics = evaluate_rankings(rankings, cases, k=20)

    for value in (
        metrics.hit_rate_at_k,
        metrics.recall_at_k,
        metrics.precision_at_k,
        metrics.mean_reciprocal_rank,
        metrics.ndcg_at_k,
    ):
        assert 0.0 <= value <= 1.0


def test_source_saturation_diagnostics_include_irrelevant_documents():
    cases = [RetrievalCase("query", frozenset({"relevant"}))]
    rankings = [
        [
            {"doc_id": "a", "chunk_id": "a:0"},
            {"doc_id": "a", "chunk_id": "a:1"},
            {"doc_id": "b", "chunk_id": "b:0"},
            {"doc_id": "b", "chunk_id": "b:1"},
            {"doc_id": "c", "chunk_id": "c:0"},
        ]
    ]

    metrics = evaluate_rankings(rankings, cases, k=5)

    assert metrics.mean_unique_documents_at_k == pytest.approx(3.0)
    assert metrics.duplicate_document_rate_at_k == pytest.approx(2 / 5)
    assert metrics.mean_largest_document_share_at_k == pytest.approx(2 / 5)


@pytest.mark.parametrize(
    ("ranking", "unique_documents", "duplicate_rate", "largest_share"),
    [
        ([], 0.0, 0.0, 0.0),
        ([{"doc_id": "only"}], 1.0, 0.0, 1.0),
        (
            [{"doc_id": "a"}, {"doc_id": "b"}, {"doc_id": "c"}],
            3.0,
            0.0,
            1 / 3,
        ),
    ],
)
def test_source_saturation_uses_observed_slots_for_short_rankings(
    ranking: list[dict],
    unique_documents: float,
    duplicate_rate: float,
    largest_share: float,
) -> None:
    metrics = evaluate_rankings(
        [ranking],
        [RetrievalCase("query", frozenset({"relevant"}))],
        k=5,
    )

    assert metrics.mean_unique_documents_at_k == pytest.approx(unique_documents)
    assert metrics.duplicate_document_rate_at_k == pytest.approx(duplicate_rate)
    assert metrics.mean_largest_document_share_at_k == pytest.approx(largest_share)


def test_engine_fails_closed_on_rejected_llm_citations(
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
            "claim_citations": [
                {
                    "claim_index": 0,
                    "citations": [valid_chunk_id, "fabricated:404"],
                }
            ],
            "confidence": 0.91,
            "insufficient_evidence": False,
        }

    monkeypatch.setattr(OpenAICompatibleGenerator, "generate", fake_generate)
    answer = RAGEngine(store).answer(
        "Do admin endpoints require authentication?", use_llm=True
    )
    assert answer.insufficient_evidence is True
    assert answer.citations == ()
    assert answer.rejected_citations == ("fabricated:404",)
    assert answer.citation_audit["citations"] == ()
    assert "citation_not_allowed" in answer.generation_admission["reasons"]
