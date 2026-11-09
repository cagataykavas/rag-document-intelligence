import json

import pytest

from src.context_budget import (
    ContextBudgetError,
    ContextBudgetPolicy,
    ContextCandidate,
    main,
    pack_context,
    render_evidence,
)


def _candidate(
    chunk_id: str,
    *,
    source: str = "policy.md",
    text: str = "evidence text",
    score: float = 0.8,
    required: bool = False,
) -> ContextCandidate:
    return ContextCandidate(chunk_id, source, text, score, required)


def test_pack_is_rank_preserving_and_accounts_for_rendered_context():
    candidates = [
        _candidate("a", source="one.md", text="first"),
        _candidate("b", source="two.md", text="second", required=True),
    ]
    expected_chars = sum(len(render_evidence(item)) for item in candidates) + 2

    result = pack_context(
        candidates,
        ContextBudgetPolicy(max_context_chars=expected_chars, max_chunk_chars=20),
    )

    assert result.selected == tuple(candidates)
    assert result.report["complete"] is True
    assert result.report["selected_context_chars"] == expected_chars
    assert result.report["remaining_context_chars"] == 0
    assert result.report["required_chunk_ids"] == ["b"]


def test_required_evidence_reserves_capacity_before_higher_ranked_optional_evidence():
    optional = _candidate("optional", source="one.md", text="x" * 30)
    required = _candidate("required", source="two.md", text="y" * 30, required=True)
    budget = len(render_evidence(required))

    result = pack_context(
        [optional, required],
        ContextBudgetPolicy(max_context_chars=budget, max_chunk_chars=30),
    )

    assert [item.chunk_id for item in result.selected] == ["required"]
    assert result.report["exclusions"] == [
        {"chunk_id": "optional", "reason": "context_budget_exhausted"}
    ]


def test_source_quota_prevents_one_document_from_monopolizing_context():
    result = pack_context(
        [
            _candidate("a", source="one.md"),
            _candidate("b", source="one.md"),
            _candidate("c", source="two.md"),
        ],
        ContextBudgetPolicy(
            max_context_chars=1_000,
            max_chunk_chars=100,
            max_chunks_per_source=1,
            min_selected_chunks=2,
            min_selected_sources=2,
        ),
    )

    assert [item.chunk_id for item in result.selected] == ["a", "c"]
    assert result.report["selected_source_count"] == 2
    assert result.report["exclusions"] == [{"chunk_id": "b", "reason": "source_quota_exceeded"}]


def test_oversized_optional_chunk_is_excluded_without_truncation():
    result = pack_context(
        [_candidate("large", text="x" * 11), _candidate("small", text="ok")],
        ContextBudgetPolicy(max_context_chars=500, max_chunk_chars=10),
    )

    assert [item.chunk_id for item in result.selected] == ["small"]
    assert result.report["exclusions"] == [{"chunk_id": "large", "reason": "chunk_too_large"}]


@pytest.mark.parametrize(
    ("candidates", "policy", "message"),
    [
        ([], ContextBudgetPolicy(), "at least one"),
        (
            [_candidate("a"), _candidate("a")],
            ContextBudgetPolicy(),
            "duplicate chunk_id",
        ),
        (
            [_candidate("required", text="x" * 11, required=True)],
            ContextBudgetPolicy(max_context_chars=100, max_chunk_chars=10),
            "required chunk",
        ),
        (
            [_candidate("a", source="one"), _candidate("b", source="one")],
            ContextBudgetPolicy(
                max_context_chars=500,
                max_chunk_chars=100,
                min_selected_chunks=2,
                min_selected_sources=2,
            ),
            "min_selected_sources",
        ),
        (
            [_candidate("a", score=float("nan"))],
            ContextBudgetPolicy(),
            "score must be finite",
        ),
    ],
)
def test_invalid_or_insufficient_evidence_fails_closed(candidates, policy, message):
    with pytest.raises(ContextBudgetError, match=message):
        pack_context(candidates, policy)


@pytest.mark.parametrize(
    "policy",
    [
        ContextBudgetPolicy(max_context_chars=0),
        ContextBudgetPolicy(max_context_chars=10, max_chunk_chars=11),
        ContextBudgetPolicy(min_selected_chunks=1, min_selected_sources=2),
    ],
)
def test_invalid_policy_fails_closed(policy):
    with pytest.raises(ContextBudgetError):
        pack_context([_candidate("a")], policy)


def test_cli_uses_distinct_exit_code_for_valid_exclusions(tmp_path, capsys):
    payload = [
        {
            "chunk_id": "large",
            "source": "policy.md",
            "text": "x" * 11,
            "score": 0.9,
        },
        {
            "chunk_id": "small",
            "source": "manual.md",
            "text": "ok",
            "score": 0.8,
        },
    ]
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = main(
        [
            "--input",
            str(path),
            "--max-context-chars",
            "500",
            "--max-chunk-chars",
            "10",
            "--require-all",
        ]
    )
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 3
    assert report["complete"] is False
    assert report["selected_chunk_ids"] == ["small"]


def test_cli_rejects_malformed_input(tmp_path, capsys):
    path = tmp_path / "candidates.json"
    path.write_text("{}", encoding="utf-8")

    exit_code = main(["--input", str(path)])

    assert exit_code == 2
    assert "JSON array" in capsys.readouterr().err
