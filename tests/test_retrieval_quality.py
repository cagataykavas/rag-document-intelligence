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
        Chunk("a:0", "a", "security.md", "REQ-42X administrative authentication is mandatory"),
        Chunk("b:0", "b", "latency.md", "API latency shall remain below 250 milliseconds"),
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


def test_engine_reports_rejected_llm_citations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    answer = RAGEngine(store).answer("Do admin endpoints require authentication?", use_llm=True)
    assert answer.citations == (valid_chunk_id,)
    assert answer.rejected_citations == ("fabricated:404",)
    assert answer.citation_audit["citation_validity_rate"] == pytest.approx(0.5)
