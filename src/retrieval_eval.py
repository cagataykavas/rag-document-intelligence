from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import isfinite, log2
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


@dataclass(frozen=True)
class RankingStabilityMetrics:
    trials: int
    pairs: int
    k: int
    exact_order_pair_rate: float
    mean_jaccard_at_k: float
    minimum_jaccard_at_k: float
    maximum_common_score_delta: float

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
            raise ValueError(
                "every retrieval case must contain at least one relevant document"
            )
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


def _validated_top_k(
    results: list[dict], *, k: int, trial_index: int
) -> list[tuple[str, float]]:
    if not isinstance(results, list) or not k <= len(results) <= 10_000:
        raise ValueError(
            f"trial {trial_index} must contain between k and 10000 results"
        )
    validated: list[tuple[str, float]] = []
    seen: set[str] = set()
    previous_score: float | None = None
    for rank, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"trial {trial_index} rank {rank} must be an object")
        chunk_id = item.get("chunk_id")
        score_value = item.get("score")
        if not isinstance(chunk_id, str) or not 1 <= len(chunk_id) <= 256:
            raise ValueError(f"trial {trial_index} rank {rank} has an invalid chunk_id")
        if chunk_id in seen:
            raise ValueError(
                f"trial {trial_index} contains duplicate chunk_id {chunk_id!r}"
            )
        if type(score_value) not in {int, float}:
            raise ValueError(f"trial {trial_index} rank {rank} has a non-numeric score")
        score = float(score_value)
        if not isfinite(score) or score < 0:
            raise ValueError(f"trial {trial_index} rank {rank} has an invalid score")
        if previous_score is not None and score > previous_score:
            raise ValueError(f"trial {trial_index} is not sorted by descending score")
        seen.add(chunk_id)
        previous_score = score
        if rank <= k:
            validated.append((chunk_id, score))
    return validated


def evaluate_ranking_stability(
    ranked_trials: list[list[dict]],
    *,
    k: int,
) -> RankingStabilityMetrics:
    """Measure pairwise repeatability for one fixed query, corpus, and retriever config."""
    if type(k) is not int or not 1 <= k <= 1_000:
        raise ValueError("k must be an integer in [1, 1000]")
    if not isinstance(ranked_trials, list) or not 2 <= len(ranked_trials) <= 20:
        raise ValueError("ranked_trials must contain between 2 and 20 trials")
    top_k = [
        _validated_top_k(results, k=k, trial_index=index)
        for index, results in enumerate(ranked_trials)
    ]
    exact: list[float] = []
    jaccards: list[float] = []
    score_deltas: list[float] = []
    for left, right in combinations(top_k, 2):
        left_ids = [chunk_id for chunk_id, _ in left]
        right_ids = [chunk_id for chunk_id, _ in right]
        exact.append(float(left_ids == right_ids))
        left_set = set(left_ids)
        right_set = set(right_ids)
        jaccards.append(len(left_set & right_set) / len(left_set | right_set))
        left_scores = dict(left)
        right_scores = dict(right)
        score_deltas.extend(
            abs(left_scores[chunk_id] - right_scores[chunk_id])
            for chunk_id in left_set & right_set
        )
    return RankingStabilityMetrics(
        trials=len(top_k),
        pairs=len(exact),
        k=k,
        exact_order_pair_rate=mean(exact),
        mean_jaccard_at_k=mean(jaccards),
        minimum_jaccard_at_k=min(jaccards),
        maximum_common_score_delta=max(score_deltas, default=0.0),
    )


def gate_ranking_stability(
    metrics: RankingStabilityMetrics,
    *,
    min_exact_order_pair_rate: float,
    min_mean_jaccard_at_k: float,
    min_pairwise_jaccard_at_k: float,
    max_common_score_delta: float,
) -> dict:
    """Apply reviewed repeatability thresholds and return stable reason codes."""
    minimums = {
        "min_exact_order_pair_rate": min_exact_order_pair_rate,
        "min_mean_jaccard_at_k": min_mean_jaccard_at_k,
        "min_pairwise_jaccard_at_k": min_pairwise_jaccard_at_k,
    }
    if any(
        type(value) not in {int, float}
        or not isfinite(float(value))
        or not 0 <= value <= 1
        for value in minimums.values()
    ):
        raise ValueError("minimum stability thresholds must be finite values in [0, 1]")
    if (
        type(max_common_score_delta) not in {int, float}
        or not isfinite(float(max_common_score_delta))
        or max_common_score_delta < 0
    ):
        raise ValueError("max_common_score_delta must be finite and non-negative")

    violations: list[dict[str, float | str]] = []

    def minimum(code: str, observed: float, threshold: float) -> None:
        if observed < threshold:
            violations.append(
                {"code": code, "observed": observed, "threshold": threshold}
            )

    minimum(
        "exact_order_rate_below_minimum",
        metrics.exact_order_pair_rate,
        float(min_exact_order_pair_rate),
    )
    minimum(
        "mean_jaccard_below_minimum",
        metrics.mean_jaccard_at_k,
        float(min_mean_jaccard_at_k),
    )
    minimum(
        "pairwise_jaccard_below_minimum",
        metrics.minimum_jaccard_at_k,
        float(min_pairwise_jaccard_at_k),
    )
    if metrics.maximum_common_score_delta > max_common_score_delta:
        violations.append(
            {
                "code": "common_score_delta_above_maximum",
                "observed": metrics.maximum_common_score_delta,
                "threshold": float(max_common_score_delta),
            }
        )
    violations.sort(key=lambda item: str(item["code"]))
    return {
        "schema_version": 1,
        "decision": "pass" if not violations else "fail",
        "metrics": metrics.to_dict(),
        "policy": {
            **minimums,
            "max_common_score_delta": float(max_common_score_delta),
        },
        "violations": violations,
    }
