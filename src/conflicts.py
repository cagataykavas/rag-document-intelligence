from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ConflictKind(str, Enum):
    MODALITY = "modality"
    NUMERIC = "numeric"
    NEGATION = "negation"
    NONE = "none"


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    text: str
    source: str


@dataclass(frozen=True)
class Conflict:
    left_id: str
    right_id: str
    kind: ConflictKind
    score: float
    explanation: str
    evidence: tuple[str, str]


MODALITIES = {
    "shall": 3,
    "must": 3,
    "required": 3,
    "should": 2,
    "recommended": 2,
    "may": 1,
    "optional": 1,
}
NEGATIONS = {"not", "never", "prohibited", "forbidden", "without"}


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if len(token) > 2 and token not in MODALITIES and token not in NEGATIONS
    }


def numeric_facts(text: str) -> list[tuple[float, str]]:
    pattern = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|sec|seconds?|minutes?|hours?|days?|%|percent|mb|gb|kb)?", re.I)
    facts: list[tuple[float, str]] = []
    for match in pattern.finditer(text):
        value = float(match.group("value"))
        unit = (match.group("unit") or "number").lower()
        facts.append((value, unit))
    return facts


def strongest_modality(text: str) -> tuple[str | None, int]:
    lowered = text.lower()
    found = [(word, rank) for word, rank in MODALITIES.items() if re.search(rf"\b{re.escape(word)}\b", lowered)]
    return max(found, key=lambda item: item[1]) if found else (None, 0)


def has_negation(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in NEGATIONS)


def semantic_overlap(left: Requirement, right: Requirement) -> float:
    a, b = tokens(left.text), tokens(right.text)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compare(left: Requirement, right: Requirement, minimum_overlap: float = 0.18) -> Conflict:
    overlap = semantic_overlap(left, right)
    evidence = (left.requirement_id, right.requirement_id)
    if overlap < minimum_overlap:
        return Conflict(left.requirement_id, right.requirement_id, ConflictKind.NONE, overlap, "Requirements do not discuss sufficiently similar concepts.", evidence)

    left_negated, right_negated = has_negation(left.text), has_negation(right.text)
    if left_negated != right_negated:
        return Conflict(
            left.requirement_id,
            right.requirement_id,
            ConflictKind.NEGATION,
            min(1.0, 0.72 + 0.28 * overlap),
            "Requirements have overlapping subject matter but opposite negation polarity.",
            evidence,
        )

    left_numbers, right_numbers = numeric_facts(left.text), numeric_facts(right.text)
    for left_value, left_unit in left_numbers:
        for right_value, right_unit in right_numbers:
            if left_unit == right_unit and left_value != right_value:
                relative_gap = abs(left_value - right_value) / max(abs(left_value), abs(right_value), 1.0)
                return Conflict(
                    left.requirement_id,
                    right.requirement_id,
                    ConflictKind.NUMERIC,
                    min(1.0, 0.60 + 0.30 * overlap + 0.10 * relative_gap),
                    f"Requirements specify different values for the same unit: {left_value:g} {left_unit} vs {right_value:g} {right_unit}.",
                    evidence,
                )

    left_word, left_rank = strongest_modality(left.text)
    right_word, right_rank = strongest_modality(right.text)
    if left_rank and right_rank and abs(left_rank - right_rank) >= 2:
        return Conflict(
            left.requirement_id,
            right.requirement_id,
            ConflictKind.MODALITY,
            min(1.0, 0.55 + 0.35 * overlap),
            f"Normative strength differs materially: {left_word!r} versus {right_word!r}.",
            evidence,
        )

    return Conflict(left.requirement_id, right.requirement_id, ConflictKind.NONE, overlap, "No deterministic contradiction pattern was detected.", evidence)


def find_conflicts(requirements: Iterable[Requirement], minimum_overlap: float = 0.18) -> list[Conflict]:
    rows = list(requirements)
    conflicts: list[Conflict] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            result = compare(left, right, minimum_overlap=minimum_overlap)
            if result.kind is not ConflictKind.NONE:
                conflicts.append(result)
    return sorted(conflicts, key=lambda item: item.score, reverse=True)
