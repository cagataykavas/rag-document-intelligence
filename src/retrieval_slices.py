from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean


@dataclass(frozen=True)
class SlicedRetrievalCase:
    case_id: str
    slice_name: str
    relevant_doc_ids: frozenset[str]
    ranked_doc_ids: tuple[str, ...]


@dataclass(frozen=True)
class SlicePolicy:
    k: int = 5
    min_cases_per_slice: int = 5
    min_recall_at_k: float = 0.8
    min_mrr: float = 0.7
    max_recall_gap: float = 0.15

    def __post_init__(self) -> None:
        if self.k < 1 or self.min_cases_per_slice < 1:
            raise ValueError("k and min_cases_per_slice must be positive integers")
        for name, value in (
            ("min_recall_at_k", self.min_recall_at_k),
            ("min_mrr", self.min_mrr),
            ("max_recall_gap", self.max_recall_gap),
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True)
class SliceMetrics:
    slice_name: str
    cases: int
    hit_rate_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float


@dataclass(frozen=True)
class SliceAuditReport:
    passed: bool
    policy: SlicePolicy
    overall: SliceMetrics
    slices: tuple[SliceMetrics, ...]
    excluded_slices: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "policy": asdict(self.policy),
            "overall": asdict(self.overall),
            "slices": [asdict(item) for item in self.slices],
            "excluded_slices": list(self.excluded_slices),
            "reasons": list(self.reasons),
        }


def _metrics(name: str, cases: list[SlicedRetrievalCase], k: int) -> SliceMetrics:
    recalls: list[float] = []
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    for case in cases:
        top = case.ranked_doc_ids[:k]
        relevant_retrieved = sum(doc_id in case.relevant_doc_ids for doc_id in top)
        recalls.append(relevant_retrieved / len(case.relevant_doc_ids))
        hits.append(float(relevant_retrieved > 0))
        reciprocal_ranks.append(
            next(
                (
                    1.0 / rank
                    for rank, doc_id in enumerate(top, start=1)
                    if doc_id in case.relevant_doc_ids
                ),
                0.0,
            )
        )
    return SliceMetrics(
        slice_name=name,
        cases=len(cases),
        hit_rate_at_k=mean(hits),
        recall_at_k=mean(recalls),
        mean_reciprocal_rank=mean(reciprocal_ranks),
    )


def audit_retrieval_slices(
    cases: list[SlicedRetrievalCase], policy: SlicePolicy | None = None
) -> SliceAuditReport:
    """Fail a release when a sufficiently represented query slice degrades."""

    active_policy = policy or SlicePolicy()
    if not cases:
        raise ValueError("cases must not be empty")

    seen: set[str] = set()
    grouped: dict[str, list[SlicedRetrievalCase]] = {}
    for case in cases:
        if not case.case_id or case.case_id in seen:
            raise ValueError(f"case_id must be non-empty and unique: {case.case_id!r}")
        seen.add(case.case_id)
        if not case.slice_name.strip():
            raise ValueError(f"slice_name must not be empty for case {case.case_id}")
        if not case.relevant_doc_ids:
            raise ValueError(f"relevant_doc_ids must not be empty for case {case.case_id}")
        if any(not doc_id for doc_id in (*case.relevant_doc_ids, *case.ranked_doc_ids)):
            raise ValueError(f"document IDs must not be empty for case {case.case_id}")
        if len(case.ranked_doc_ids) != len(set(case.ranked_doc_ids)):
            raise ValueError(f"ranked_doc_ids contains duplicates for case {case.case_id}")
        grouped.setdefault(case.slice_name, []).append(case)

    overall = _metrics("__overall__", cases, active_policy.k)
    included: list[SliceMetrics] = []
    excluded: list[str] = []
    reasons: list[str] = []
    for slice_name in sorted(grouped):
        slice_cases = grouped[slice_name]
        if len(slice_cases) < active_policy.min_cases_per_slice:
            excluded.append(slice_name)
            continue
        metrics = _metrics(slice_name, slice_cases, active_policy.k)
        included.append(metrics)
        if metrics.recall_at_k < active_policy.min_recall_at_k:
            reasons.append(f"slice_recall_below_minimum:{slice_name}")
        if metrics.mean_reciprocal_rank < active_policy.min_mrr:
            reasons.append(f"slice_mrr_below_minimum:{slice_name}")
        if overall.recall_at_k - metrics.recall_at_k > active_policy.max_recall_gap:
            reasons.append(f"slice_recall_gap_exceeded:{slice_name}")

    if not included:
        reasons.append("no_eligible_slices")
    return SliceAuditReport(
        passed=not reasons,
        policy=active_policy,
        overall=overall,
        slices=tuple(included),
        excluded_slices=tuple(excluded),
        reasons=tuple(reasons),
    )
