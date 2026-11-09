from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.chunking_sensitivity import (
    EXIT_ACCEPTED,
    EXIT_MALFORMED,
    EXIT_REJECTED,
    ChunkingArtifact,
    ChunkingAuditError,
    ChunkingPolicy,
    RankedSpan,
    Span,
    audit_chunking_sensitivity,
    document_jaccard,
    main,
    parse_artifact,
    rank_weighted_document_jaccard,
    required_span_recall,
    span_coverage_jaccard,
)

BASELINE = "1" * 64
CANDIDATE = "2" * 64
DOCUMENT_A = "a" * 64
DOCUMENT_B = "b" * 64


def _span(document: str, start: int, end: int, *, rank: int | None = None) -> dict:
    value = {
        "document_sha256": document,
        "start_byte": start,
        "end_byte": end,
    }
    if rank is not None:
        value["rank"] = rank
    return value


def _artifact_document(*, query_count: int = 1) -> dict:
    queries = []
    for index in range(query_count):
        queries.append(
            {
                "query_id": f"refund-policy-{index}",
                "required_spans": [_span(DOCUMENT_A, 100, 200)],
                "variants": [
                    {
                        "configuration_sha256": BASELINE,
                        "results": [
                            _span(DOCUMENT_A, 80, 220, rank=1),
                            _span(DOCUMENT_B, 0, 100, rank=2),
                        ],
                    },
                    {
                        "configuration_sha256": CANDIDATE,
                        "results": [
                            _span(DOCUMENT_A, 90, 230, rank=1),
                            _span(DOCUMENT_B, 10, 110, rank=2),
                        ],
                    },
                ],
            }
        )
    return {
        "schema_version": 1,
        "corpus_sha256": "c" * 64,
        "retriever_sha256": "d" * 64,
        "baseline_configuration_sha256": BASELINE,
        "configurations": [BASELINE, CANDIDATE],
        "documents": [
            {"document_sha256": DOCUMENT_A, "size_bytes": 1_000},
            {"document_sha256": DOCUMENT_B, "size_bytes": 1_000},
        ],
        "queries": queries,
    }


def _parse(values: dict, policy: ChunkingPolicy | None = None) -> ChunkingArtifact:
    return parse_artifact(values, policy=policy or ChunkingPolicy())


def _codes(values: dict, policy: ChunkingPolicy | None = None) -> set[str]:
    report = audit_chunking_sensitivity(_parse(values, policy), policy=policy)
    return {finding.code for finding in report.findings}


def test_span_metrics_merge_overlap_and_use_half_open_offsets() -> None:
    left = (
        RankedSpan(1, DOCUMENT_A, 0, 10),
        RankedSpan(2, DOCUMENT_A, 5, 15),
    )
    right = (RankedSpan(1, DOCUMENT_A, 10, 20),)

    assert span_coverage_jaccard(left, right) == pytest.approx(5 / 20)
    assert required_span_recall((Span(DOCUMENT_A, 5, 12),), left) == 1.0


def test_document_metrics_capture_presence_and_rank_shift() -> None:
    left = (
        RankedSpan(1, DOCUMENT_A, 0, 10),
        RankedSpan(2, DOCUMENT_B, 0, 10),
    )
    right = (
        RankedSpan(1, DOCUMENT_B, 0, 10),
        RankedSpan(2, DOCUMENT_A, 0, 10),
    )

    assert document_jaccard(left, right) == 1.0
    assert rank_weighted_document_jaccard(left, right) == 0.5


def test_accepts_stable_chunking_variants() -> None:
    report = audit_chunking_sensitivity(_parse(_artifact_document()))

    assert report.outcome == "accepted"
    assert report.findings == ()
    assert report.comparisons[0].span_coverage_jaccard == pytest.approx(0.846154)
    assert len(report.artifact_sha256) == 64
    assert len(report.evidence_sha256) == 64


def test_input_order_does_not_change_evidence() -> None:
    original = _artifact_document(query_count=2)
    reordered = copy.deepcopy(original)
    reordered["configurations"].reverse()
    reordered["documents"].reverse()
    reordered["queries"].reverse()
    for query in reordered["queries"]:
        query["variants"].reverse()
        for variant in query["variants"]:
            variant["results"].reverse()

    first = audit_chunking_sensitivity(_parse(original))
    second = audit_chunking_sensitivity(_parse(reordered))

    assert first == second


def test_rejects_candidate_that_misses_required_evidence() -> None:
    values = _artifact_document()
    values["queries"][0]["variants"][1]["results"][0] = _span(DOCUMENT_A, 300, 440, rank=1)

    assert "variant_required_span_recall_below_minimum" in _codes(values)


def test_rejects_baseline_that_misses_required_evidence() -> None:
    values = _artifact_document()
    values["queries"][0]["variants"][0]["results"][0] = _span(DOCUMENT_A, 300, 440, rank=1)

    assert "baseline_required_span_recall_below_minimum" in _codes(values)


def test_stable_but_wrong_retrieval_is_rejected() -> None:
    values = _artifact_document()
    for variant in values["queries"][0]["variants"]:
        variant["results"] = [_span(DOCUMENT_B, 0, 100, rank=1)]

    codes = _codes(values)

    assert "baseline_required_span_recall_below_minimum" in codes
    assert "variant_required_span_recall_below_minimum" in codes
    assert "span_coverage_jaccard_below_minimum" not in codes


def test_rejects_span_coverage_drift() -> None:
    values = _artifact_document()
    values["queries"][0]["variants"][1]["results"] = [_span(DOCUMENT_A, 100, 200, rank=1)]

    assert "span_coverage_jaccard_below_minimum" in _codes(values)


def test_rejects_document_set_drift() -> None:
    values = _artifact_document()
    values["queries"][0]["variants"][1]["results"] = [_span(DOCUMENT_A, 80, 220, rank=1)]
    policy = ChunkingPolicy(
        min_span_coverage_jaccard=0.0,
        min_document_jaccard=0.75,
        min_rank_weighted_document_jaccard=0.0,
    )

    assert "document_jaccard_below_minimum" in _codes(values, policy)


def test_rejects_rank_weighted_drift() -> None:
    values = _artifact_document()
    values["queries"][0]["variants"][1]["results"].reverse()
    for rank, result in enumerate(values["queries"][0]["variants"][1]["results"], start=1):
        result["rank"] = rank
    policy = ChunkingPolicy(
        min_span_coverage_jaccard=0.0,
        min_document_jaccard=0.0,
        min_rank_weighted_document_jaccard=0.75,
    )

    assert "rank_weighted_document_jaccard_below_minimum" in _codes(values, policy)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(schema_version=2), "schema_version"),
        (lambda value: value.update(extra=True), "unknown fields"),
        (lambda value: value.update(corpus_sha256="A" * 64), "lowercase SHA-256"),
        (lambda value: value.update(configurations=[BASELINE, BASELINE]), "unique"),
        (lambda value: value.update(baseline_configuration_sha256="3" * 64), "not in"),
        (lambda value: value.update(documents=[]), "document budget"),
        (lambda value: value.update(queries=[]), "query budget"),
    ],
)
def test_rejects_malformed_top_level_artifacts(mutation, match: str) -> None:
    values = _artifact_document()
    mutation(values)

    with pytest.raises(ChunkingAuditError, match=match):
        _parse(values)


@pytest.mark.parametrize(
    ("target", "field", "value", "match"),
    [
        ("document", "size_bytes", True, "integer"),
        ("document", "unexpected", 1, "unknown fields"),
        ("query", "query_id", "contains whitespace", "identifier"),
        ("query", "unexpected", 1, "unknown fields"),
        ("variant", "unexpected", 1, "unknown fields"),
        ("result", "unexpected", 1, "unknown fields"),
    ],
)
def test_rejects_invalid_nested_fields(target: str, field: str, value: object, match: str) -> None:
    values = _artifact_document()
    targets = {
        "document": values["documents"][0],
        "query": values["queries"][0],
        "variant": values["queries"][0]["variants"][0],
        "result": values["queries"][0]["variants"][0]["results"][0],
    }
    targets[target][field] = value

    with pytest.raises(ChunkingAuditError, match=match):
        _parse(values)


@pytest.mark.parametrize(
    ("start", "end", "match"),
    [
        (100, 100, "non-empty"),
        (200, 100, "non-empty"),
        (0, 1_001, "allowed range"),
    ],
)
def test_rejects_invalid_span_bounds(start: int, end: int, match: str) -> None:
    values = _artifact_document()
    result = values["queries"][0]["variants"][0]["results"][0]
    result["start_byte"] = start
    result["end_byte"] = end

    with pytest.raises(ChunkingAuditError, match=match):
        _parse(values)


def test_rejects_unknown_span_document() -> None:
    values = _artifact_document()
    values["queries"][0]["required_spans"][0]["document_sha256"] = "f" * 64

    with pytest.raises(ChunkingAuditError, match="unknown document"):
        _parse(values)


def test_rejects_duplicate_documents_queries_and_required_spans() -> None:
    values = _artifact_document()
    values["documents"].append(copy.deepcopy(values["documents"][0]))
    with pytest.raises(ChunkingAuditError, match="document digests must be unique"):
        _parse(values)

    values = _artifact_document(query_count=2)
    values["queries"][1]["query_id"] = values["queries"][0]["query_id"]
    with pytest.raises(ChunkingAuditError, match="query IDs must be unique"):
        _parse(values)

    values = _artifact_document()
    values["queries"][0]["required_spans"] *= 2
    with pytest.raises(ChunkingAuditError, match="duplicate required spans"):
        _parse(values)


def test_rejects_duplicate_spans_and_noncontiguous_ranks() -> None:
    values = _artifact_document()
    values["queries"][0]["variants"][0]["results"][1] = copy.deepcopy(
        values["queries"][0]["variants"][0]["results"][0]
    )
    values["queries"][0]["variants"][0]["results"][1]["rank"] = 2
    with pytest.raises(ChunkingAuditError, match="duplicate ranked spans"):
        _parse(values)

    values = _artifact_document()
    values["queries"][0]["variants"][0]["results"][1]["rank"] = 3
    with pytest.raises(ChunkingAuditError, match="contiguous"):
        _parse(values)


def test_rejects_missing_or_duplicate_configuration_variants() -> None:
    values = _artifact_document()
    values["queries"][0]["variants"].pop()
    with pytest.raises(ChunkingAuditError, match="exactly one"):
        _parse(values)

    values = _artifact_document()
    values["queries"][0]["variants"][1]["configuration_sha256"] = BASELINE
    with pytest.raises(ChunkingAuditError, match="configuration set"):
        _parse(values)


def test_rejects_empty_required_spans_and_results() -> None:
    values = _artifact_document()
    values["queries"][0]["required_spans"] = []
    with pytest.raises(ChunkingAuditError, match="required spans"):
        _parse(values)

    values = _artifact_document()
    values["queries"][0]["variants"][0]["results"] = []
    with pytest.raises(ChunkingAuditError, match="result budget"):
        _parse(values)


def test_enforces_total_span_budget() -> None:
    policy = ChunkingPolicy(max_total_spans=4)

    with pytest.raises(ChunkingAuditError, match="total span budget"):
        _parse(_artifact_document(), policy)


def test_bounds_findings_deterministically() -> None:
    values = _artifact_document(query_count=2)
    for query in values["queries"]:
        for variant in query["variants"]:
            variant["results"] = [_span(DOCUMENT_B, 0, 10, rank=1)]
    policy = ChunkingPolicy(max_findings=2)

    report = audit_chunking_sensitivity(_parse(values, policy), policy=policy)

    assert report.outcome == "rejected"
    assert report.findings_truncated is True
    assert len(report.findings) == 2


def test_report_does_not_expose_query_identifiers() -> None:
    values = _artifact_document()
    query_id = values["queries"][0]["query_id"]

    encoded = json.dumps(audit_chunking_sensitivity(_parse(values)).to_dict())

    assert query_id not in encoded


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_configurations": True},
        {"max_documents": 0},
        {"max_queries": 10_001},
        {"max_results_per_configuration": 0},
        {"max_total_spans": 5_000_001},
        {"min_span_coverage_jaccard": float("nan")},
        {"min_document_jaccard": True},
        {"min_required_span_recall": 1.1},
        {"max_findings": 0},
    ],
)
def test_rejects_invalid_policy(kwargs: dict) -> None:
    with pytest.raises(ChunkingAuditError):
        ChunkingPolicy(**kwargs)


def _write_json(path: Path, values: object) -> None:
    path.write_text(json.dumps(values), encoding="utf-8")


def test_cli_returns_distinct_success_and_policy_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_path = tmp_path / "artifact.json"
    output_path = tmp_path / "nested" / "report.json"
    _write_json(artifact_path, _artifact_document())

    assert main([str(artifact_path), "--output", str(output_path)]) == EXIT_ACCEPTED
    assert json.loads(output_path.read_text())["outcome"] == "accepted"
    assert json.loads(capsys.readouterr().out)["outcome"] == "accepted"

    values = _artifact_document()
    values["queries"][0]["variants"][1]["results"] = [_span(DOCUMENT_B, 400, 500, rank=1)]
    _write_json(artifact_path, values)

    assert main([str(artifact_path)]) == EXIT_REJECTED
    assert json.loads(capsys.readouterr().out)["outcome"] == "rejected"


@pytest.mark.parametrize(
    "encoded",
    [
        b'{"schema_version":1,"schema_version":1}',
        b'{"value": NaN}',
        b"not-json",
        b"\xff",
    ],
)
def test_cli_rejects_malformed_json(
    encoded: bytes, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(encoded)

    assert main([str(path)]) == EXIT_MALFORMED
    output = json.loads(capsys.readouterr().out)
    assert output["outcome"] == "malformed"


def test_cli_rejects_unknown_and_nonfinite_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact_path = tmp_path / "artifact.json"
    policy_path = tmp_path / "policy.json"
    _write_json(artifact_path, _artifact_document())
    _write_json(policy_path, {"unknown": 1})

    assert main([str(artifact_path), "--policy", str(policy_path)]) == EXIT_MALFORMED
    assert json.loads(capsys.readouterr().out)["outcome"] == "malformed"

    policy_path.write_text('{"min_document_jaccard": Infinity}', encoding="utf-8")
    assert main([str(artifact_path), "--policy", str(policy_path)]) == EXIT_MALFORMED


def test_cli_rejects_oversized_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from src import chunking_sensitivity

    path = tmp_path / "artifact.json"
    path.write_bytes(b"{} " * 10)
    monkeypatch.setattr(chunking_sensitivity, "MAX_ARTIFACT_BYTES", 8)

    assert main([str(path)]) == EXIT_MALFORMED
    assert json.loads(capsys.readouterr().out)["outcome"] == "malformed"
