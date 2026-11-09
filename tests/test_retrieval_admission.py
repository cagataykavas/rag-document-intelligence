from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.engine import RAGEngine
from src.hybrid import HybridSparseRetriever, RetrievalAdmissionPolicy
from src.parsers import ParsedDocument
from src.retrieval import Chunk
from src.store import DocumentStore


def chunks() -> list[Chunk]:
    return [
        Chunk("auth:0", "auth", "auth.md", "administrators require strong authentication"),
        Chunk("latency:0", "latency", "latency.md", "API latency is below 250 milliseconds"),
    ]


def test_out_of_vocabulary_query_returns_no_arbitrary_evidence() -> None:
    results = HybridSparseRetriever().fit(chunks()).search("🛸🛸🛸", k=2)

    assert results == []


def test_engine_abstains_when_retrieval_has_no_positive_signal(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "rag.db")
    store.ingest(
        ParsedDocument("policy.md", "text/markdown", "administrators require authentication", {}),
        words_per_chunk=20,
        overlap=2,
    )

    answer = RAGEngine(store).answer("🛸🛸🛸")

    assert answer.insufficient_evidence is True
    assert answer.retrieval == ()
    assert answer.citations == ()


def test_equal_score_order_is_stable_across_index_order() -> None:
    left = Chunk("a:0", "a", "a.md", "shared authentication requirement")
    right = Chunk("b:0", "b", "b.md", "shared authentication requirement")

    forward = HybridSparseRetriever().fit([left, right]).search("authentication", k=2)
    reverse = HybridSparseRetriever().fit([right, left]).search("authentication", k=2)

    assert [item["chunk_id"] for item in forward] == ["a:0", "b:0"]
    assert [item["chunk_id"] for item in reverse] == ["a:0", "b:0"]


def test_relative_floor_excludes_weak_tail_results() -> None:
    policy = RetrievalAdmissionPolicy(min_relative_score=0.9)
    values = [
        Chunk("strong:0", "strong", "strong.md", "alpha beta gamma delta"),
        Chunk("weak:0", "weak", "weak.md", "alpha unrelated material"),
    ]

    results = (
        HybridSparseRetriever(admission_policy=policy)
        .fit(values)
        .search("alpha beta gamma delta", k=2)
    )

    assert [item["chunk_id"] for item in results] == ["strong:0"]
    assert results[0]["admission_threshold"] == pytest.approx(results[0]["score"] * 0.9)


def test_absolute_floor_can_fail_closed_on_weak_signal() -> None:
    policy = RetrievalAdmissionPolicy(min_fused_score=1.0)

    results = (
        HybridSparseRetriever(admission_policy=policy).fit(chunks()).search("authentication", k=2)
    )

    assert results == []


def test_diversity_reranking_remains_deterministic() -> None:
    values = [
        Chunk("a:0", "same", "a.md", "authentication policy"),
        Chunk("a:1", "same", "a.md", "authentication policy"),
        Chunk("b:0", "other", "b.md", "authentication policy"),
    ]

    results = (
        HybridSparseRetriever()
        .fit(list(reversed(values)))
        .search("authentication policy", k=3, diversity_penalty=0.1)
    )

    assert [item["chunk_id"] for item in results] == ["a:0", "b:0", "a:1"]


@pytest.mark.parametrize(
    "policy",
    [
        RetrievalAdmissionPolicy(min_fused_score=0.0),
        RetrievalAdmissionPolicy(min_relative_score=1.0),
        RetrievalAdmissionPolicy(max_query_chars=1),
        RetrievalAdmissionPolicy(max_results=1),
    ],
)
def test_policy_boundary_values_are_usable(policy: RetrievalAdmissionPolicy) -> None:
    assert HybridSparseRetriever(admission_policy=policy).fit(chunks())


@pytest.mark.parametrize(
    "values",
    [
        {"min_fused_score": -0.1},
        {"min_fused_score": float("nan")},
        {"min_fused_score": True},
        {"min_relative_score": 1.1},
        {"min_relative_score": float("inf")},
        {"min_relative_score": "0.5"},
        {"max_query_chars": True},
        {"max_results": 0},
    ],
)
def test_invalid_policy_fails_closed(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RetrievalAdmissionPolicy(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("query", "k", "penalty"),
    [
        ("", 1, 0.0),
        ("valid", True, 0.0),
        ("valid", 101, 0.0),
        ("valid", 1, -0.1),
        ("valid", 1, float("nan")),
        ("valid", 1, True),
        ("bad\x00query", 1, 0.0),
    ],
)
def test_invalid_search_contract_fails_closed(query: str, k: int, penalty: float) -> None:
    with pytest.raises(ValueError):
        HybridSparseRetriever().fit(chunks()).search(query, k=k, diversity_penalty=penalty)


def test_query_and_result_budgets_are_enforced() -> None:
    policy = RetrievalAdmissionPolicy(max_query_chars=4, max_results=1)
    retriever = HybridSparseRetriever(admission_policy=policy).fit(chunks())

    with pytest.raises(ValueError, match="character budget"):
        retriever.search("12345", k=1)
    with pytest.raises(ValueError, match="result budget"):
        retriever.search("auth", k=2)


def test_duplicate_chunk_identity_is_rejected() -> None:
    duplicate = replace(chunks()[0], text="different bytes under the same identity")

    with pytest.raises(ValueError, match="unique"):
        HybridSparseRetriever().fit([chunks()[0], duplicate])
