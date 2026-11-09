from __future__ import annotations

import json
import math

import pytest

from src.retrieval_slices import (
    SlicedRetrievalCase,
    SlicePolicy,
    audit_retrieval_slices,
)


def case(case_id: str, slice_name: str, ranked: tuple[str, ...]) -> SlicedRetrievalCase:
    return SlicedRetrievalCase(case_id, slice_name, frozenset({"target"}), ranked)


def test_passes_when_every_eligible_slice_meets_policy():
    cases = [case(f"a{i}", "short_query", ("target", "other")) for i in range(3)]
    cases += [case(f"b{i}", "identifier_query", ("other", "target")) for i in range(3)]

    report = audit_retrieval_slices(
        cases,
        SlicePolicy(k=2, min_cases_per_slice=3, min_recall_at_k=1.0, min_mrr=0.5),
    )

    assert report.passed
    assert [item.slice_name for item in report.slices] == ["identifier_query", "short_query"]
    assert json.loads(json.dumps(report.to_dict()))["passed"] is True


def test_fails_slice_even_when_aggregate_recall_looks_acceptable():
    cases = [case(f"good{i}", "common", ("target",)) for i in range(8)]
    cases += [case(f"bad{i}", "numeric", ("wrong",)) for i in range(2)]

    report = audit_retrieval_slices(
        cases,
        SlicePolicy(
            k=1,
            min_cases_per_slice=2,
            min_recall_at_k=0.5,
            min_mrr=0.5,
            max_recall_gap=0.2,
        ),
    )

    assert report.overall.recall_at_k == 0.8
    assert not report.passed
    assert "slice_recall_below_minimum:numeric" in report.reasons
    assert "slice_recall_gap_exceeded:numeric" in report.reasons


def test_reports_but_does_not_gate_underpowered_slices():
    cases = [case(f"a{i}", "eligible", ("target",)) for i in range(3)]
    cases.append(case("rare", "rare", ("wrong",)))

    report = audit_retrieval_slices(cases, SlicePolicy(min_cases_per_slice=3))

    assert report.passed
    assert report.excluded_slices == ("rare",)


def test_fails_closed_when_every_slice_is_underpowered():
    report = audit_retrieval_slices(
        [case("a", "one", ("target",)), case("b", "two", ("target",))],
        SlicePolicy(min_cases_per_slice=2),
    )

    assert not report.passed
    assert report.reasons == ("no_eligible_slices",)


@pytest.mark.parametrize(
    "cases, match",
    [
        ([], "must not be empty"),
        ([case("same", "a", ("target",)), case("same", "b", ("target",))], "unique"),
        ([case("a", "", ("target",))], "slice_name"),
        ([case("a", "slice", ("target", "target"))], "duplicates"),
        ([SlicedRetrievalCase("a", "slice", frozenset(), ("target",))], "relevant_doc_ids"),
    ],
)
def test_rejects_ambiguous_evaluation_evidence(cases, match):
    with pytest.raises(ValueError, match=match):
        audit_retrieval_slices(cases, SlicePolicy(min_cases_per_slice=1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"min_cases_per_slice": 0},
        {"min_recall_at_k": math.nan},
        {"min_mrr": 1.1},
        {"max_recall_gap": -0.1},
    ],
)
def test_rejects_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        SlicePolicy(**kwargs)
