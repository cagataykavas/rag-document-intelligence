from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ContextBudgetError(ValueError):
    """Raised when evidence cannot be packed without violating its contract."""


@dataclass(frozen=True)
class ContextCandidate:
    chunk_id: str
    source: str
    text: str
    score: float
    required: bool = False


@dataclass(frozen=True)
class ContextBudgetPolicy:
    max_context_chars: int = 6_000
    max_chunk_chars: int = 1_800
    max_chunks_per_source: int = 2
    min_selected_chunks: int = 1
    min_selected_sources: int = 1

    def validate(self) -> None:
        values = {
            "max_context_chars": self.max_context_chars,
            "max_chunk_chars": self.max_chunk_chars,
            "max_chunks_per_source": self.max_chunks_per_source,
            "min_selected_chunks": self.min_selected_chunks,
            "min_selected_sources": self.min_selected_sources,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContextBudgetError(f"{name} must be a positive integer")
        if self.max_chunk_chars > self.max_context_chars:
            raise ContextBudgetError("max_chunk_chars cannot exceed max_context_chars")
        if self.min_selected_sources > self.min_selected_chunks:
            raise ContextBudgetError("min_selected_sources cannot exceed min_selected_chunks")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_context_chars": self.max_context_chars,
            "max_chunk_chars": self.max_chunk_chars,
            "max_chunks_per_source": self.max_chunks_per_source,
            "min_selected_chunks": self.min_selected_chunks,
            "min_selected_sources": self.min_selected_sources,
        }


@dataclass(frozen=True)
class ContextPack:
    selected: tuple[ContextCandidate, ...]
    report: dict[str, Any]


def render_evidence(candidate: ContextCandidate) -> str:
    """Render exactly the evidence block whose size is budgeted."""

    return (
        f"<evidence chunk_id={json.dumps(candidate.chunk_id)} "
        f"source={json.dumps(candidate.source)}>\n"
        f"{candidate.text}\n"
        "</evidence>"
    )


def _validate_candidate(candidate: ContextCandidate, index: int) -> None:
    prefix = f"candidates[{index}]"
    if not isinstance(candidate.chunk_id, str) or not candidate.chunk_id.strip():
        raise ContextBudgetError(f"{prefix}.chunk_id must be a non-empty string")
    if not isinstance(candidate.source, str) or not candidate.source.strip():
        raise ContextBudgetError(f"{prefix}.source must be a non-empty string")
    if not isinstance(candidate.text, str) or not candidate.text.strip():
        raise ContextBudgetError(f"{prefix}.text must be a non-empty string")
    if isinstance(candidate.score, bool) or not isinstance(candidate.score, (int, float)):
        raise ContextBudgetError(f"{prefix}.score must be numeric")
    if not math.isfinite(float(candidate.score)):
        raise ContextBudgetError(f"{prefix}.score must be finite")
    if not isinstance(candidate.required, bool):
        raise ContextBudgetError(f"{prefix}.required must be boolean")


def pack_context(
    candidates: Sequence[ContextCandidate],
    policy: ContextBudgetPolicy | None = None,
) -> ContextPack:
    """Select a rank-preserving, bounded set of evidence chunks.

    Required chunks reserve capacity before optional chunks are considered. The
    returned order always matches the original retrieval order.
    """

    active_policy = policy or ContextBudgetPolicy()
    active_policy.validate()
    if not candidates:
        raise ContextBudgetError("at least one context candidate is required")

    seen_ids: set[str] = set()
    rendered_sizes: list[int] = []
    for index, candidate in enumerate(candidates):
        _validate_candidate(candidate, index)
        if candidate.chunk_id in seen_ids:
            raise ContextBudgetError(f"duplicate chunk_id: {candidate.chunk_id}")
        seen_ids.add(candidate.chunk_id)
        rendered_sizes.append(len(render_evidence(candidate)))

    selected_indexes: set[int] = set()
    source_counts: dict[str, int] = {}
    used_chars = 0
    required_indexes = [index for index, item in enumerate(candidates) if item.required]
    for index in required_indexes:
        candidate = candidates[index]
        size = rendered_sizes[index]
        if len(candidate.text) > active_policy.max_chunk_chars:
            raise ContextBudgetError(
                f"required chunk {candidate.chunk_id!r} exceeds max_chunk_chars"
            )
        if source_counts.get(candidate.source, 0) >= active_policy.max_chunks_per_source:
            raise ContextBudgetError(
                f"required chunks from {candidate.source!r} exceed max_chunks_per_source"
            )
        separator = 2 if selected_indexes else 0
        if used_chars + separator + size > active_policy.max_context_chars:
            raise ContextBudgetError("required chunks exceed max_context_chars")
        selected_indexes.add(index)
        source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
        used_chars += separator + size

    exclusions: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates):
        if index in selected_indexes:
            continue
        size = rendered_sizes[index]
        if len(candidate.text) > active_policy.max_chunk_chars:
            reason = "chunk_too_large"
        elif source_counts.get(candidate.source, 0) >= active_policy.max_chunks_per_source:
            reason = "source_quota_exceeded"
        else:
            separator = 2 if selected_indexes else 0
            if used_chars + separator + size > active_policy.max_context_chars:
                reason = "context_budget_exhausted"
            else:
                selected_indexes.add(index)
                source_counts[candidate.source] = source_counts.get(candidate.source, 0) + 1
                used_chars += separator + size
                continue
        exclusions.append({"chunk_id": candidate.chunk_id, "reason": reason})

    selected = tuple(candidates[index] for index in sorted(selected_indexes))
    selected_sources = {item.source for item in selected}
    if len(selected) < active_policy.min_selected_chunks:
        raise ContextBudgetError("selected evidence is below min_selected_chunks")
    if len(selected_sources) < active_policy.min_selected_sources:
        raise ContextBudgetError("selected evidence is below min_selected_sources")

    return ContextPack(
        selected=selected,
        report={
            "schema_version": "context-budget/1.0",
            "complete": not exclusions,
            "policy": active_policy.to_dict(),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "selected_source_count": len(selected_sources),
            "selected_context_chars": used_chars,
            "remaining_context_chars": active_policy.max_context_chars - used_chars,
            "required_chunk_ids": [candidates[index].chunk_id for index in required_indexes],
            "selected_chunk_ids": [item.chunk_id for item in selected],
            "exclusions": exclusions,
        },
    )


def _candidate_from_mapping(value: Any, index: int) -> ContextCandidate:
    if not isinstance(value, Mapping):
        raise ContextBudgetError(f"candidates[{index}] must be an object")
    return ContextCandidate(
        chunk_id=value.get("chunk_id"),
        source=value.get("source"),
        text=value.get("text"),
        score=value.get("score"),
        required=value.get("required", False),
    )


def _load_candidates(path: Path) -> list[ContextCandidate]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContextBudgetError(f"input does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContextBudgetError("input must be valid JSON") from exc
    if not isinstance(payload, list):
        raise ContextBudgetError("input must contain a JSON array")
    return [_candidate_from_mapping(value, index) for index, value in enumerate(payload)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack ranked RAG evidence into a bounded context")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--max-context-chars", type=int, default=6_000)
    parser.add_argument("--max-chunk-chars", type=int, default=1_800)
    parser.add_argument("--max-chunks-per-source", type=int, default=2)
    parser.add_argument("--min-selected-chunks", type=int, default=1)
    parser.add_argument("--min-selected-sources", type=int, default=1)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="return exit code 3 if any valid candidate is excluded",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = pack_context(
            _load_candidates(args.input),
            ContextBudgetPolicy(
                max_context_chars=args.max_context_chars,
                max_chunk_chars=args.max_chunk_chars,
                max_chunks_per_source=args.max_chunks_per_source,
                min_selected_chunks=args.min_selected_chunks,
                min_selected_sources=args.min_selected_sources,
            ),
        )
    except (ContextBudgetError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.report, indent=2, sort_keys=True))
    return 3 if args.require_all and not result.report["complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
