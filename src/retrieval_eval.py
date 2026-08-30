from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
from statistics import mean


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_doc_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalMetrics:
    cases: int
    hit_rate_at_k: float
    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float

    def to_dict(self) -> dict:
        return asdict(self)


def _dcg(binary_relevance: list[int]) -> float:
    return sum(value / log2(index + 2) for index, value in enumerate(binary_relevance))


def evaluate_rankings(
    ranked_results: list[list[dict]],
    cases: list[RetrievalCase],
    *,
    k: int,
) -> RetrievalMetrics:
    if len(ranked_results) != len(cases):
        raise ValueError("ranked_results and cases must have equal length")
    if k < 1:
        raise ValueError("k must be positive")
    if not cases:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    hits: list[float] = []
    recalls: list[float] = []
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []

    for results, case in zip(ranked_results, cases, strict=True):
        if not case.relevant_doc_ids:
            raise ValueError("every retrieval case must contain at least one relevant document")
        top = results[:k]
        retrieved_ids = [str(item["doc_id"]) for item in top]
        relevance = [int(doc_id in case.relevant_doc_ids) for doc_id in retrieved_ids]
        relevant_retrieved = sum(relevance)

        hits.append(float(relevant_retrieved > 0))
        recalls.append(relevant_retrieved / len(case.relevant_doc_ids))
        precisions.append(relevant_retrieved / k)

        reciprocal_rank = 0.0
        for rank, is_relevant in enumerate(relevance, start=1):
            if is_relevant:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

        ideal_relevance = [1] * min(len(case.relevant_doc_ids), k)
        ideal_relevance.extend([0] * (k - len(ideal_relevance)))
        ideal_dcg = _dcg(ideal_relevance)
        ndcgs.append(_dcg(relevance) / ideal_dcg if ideal_dcg else 0.0)

    return RetrievalMetrics(
        cases=len(cases),
        hit_rate_at_k=mean(hits),
        recall_at_k=mean(recalls),
        precision_at_k=mean(precisions),
        mean_reciprocal_rank=mean(reciprocal_ranks),
        ndcg_at_k=mean(ndcgs),
    )
