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


@dataclass(frozen=True)
class NumericFact:
    raw_text: str
    raw_value: float
    raw_unit: str
    normalized_value: float
    dimension: str
    canonical_unit: str


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
UNIT_FACTORS: dict[str, tuple[str, float, str]] = {
    "ms": ("time", 0.001, "s"),
    "millisecond": ("time", 0.001, "s"),
    "milliseconds": ("time", 0.001, "s"),
    "s": ("time", 1.0, "s"),
    "sec": ("time", 1.0, "s"),
    "secs": ("time", 1.0, "s"),
    "second": ("time", 1.0, "s"),
    "seconds": ("time", 1.0, "s"),
    "min": ("time", 60.0, "s"),
    "mins": ("time", 60.0, "s"),
    "minute": ("time", 60.0, "s"),
    "minutes": ("time", 60.0, "s"),
    "hr": ("time", 3600.0, "s"),
    "hrs": ("time", 3600.0, "s"),
    "hour": ("time", 3600.0, "s"),
    "hours": ("time", 3600.0, "s"),
    "day": ("time", 86400.0, "s"),
    "days": ("time", 86400.0, "s"),
    "%": ("percentage", 1.0, "%"),
    "percent": ("percentage", 1.0, "%"),
    "kb": ("data_size", 1_000.0, "bytes"),
    "kilobyte": ("data_size", 1_000.0, "bytes"),
    "kilobytes": ("data_size", 1_000.0, "bytes"),
    "mb": ("data_size", 1_000_000.0, "bytes"),
    "megabyte": ("data_size", 1_000_000.0, "bytes"),
    "megabytes": ("data_size", 1_000_000.0, "bytes"),
    "gb": ("data_size", 1_000_000_000.0, "bytes"),
    "gigabyte": ("data_size", 1_000_000_000.0, "bytes"),
    "gigabytes": ("data_size", 1_000_000_000.0, "bytes"),
}
_UNITS = sorted(UNIT_FACTORS, key=len, reverse=True)
NUMERIC_FACT_RE = re.compile(
    rf"(?<![\w.-])(?P<value>[+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
    rf"(?P<unit>{'|'.join(re.escape(unit) for unit in _UNITS)})"
    r"(?=\s|[.,;:)]|$)",
    re.I,
)


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]+", text.lower())
        if len(token) > 2 and token not in MODALITIES and token not in NEGATIONS
    }


def numeric_facts(text: str) -> list[NumericFact]:
    """Extract comparable values with explicit, supported units.

    Time uses seconds, data size uses decimal bytes, and percentages retain
    percentage points. Bare numbers are ignored so versions and requirement IDs
    do not become false contradiction evidence.
    """
    facts: list[NumericFact] = []
    for match in NUMERIC_FACT_RE.finditer(text):
        raw_value = float(match.group("value"))
        raw_unit = match.group("unit")
        dimension, factor, canonical_unit = UNIT_FACTORS[raw_unit.lower()]
        facts.append(
            NumericFact(
                raw_text=match.group(0).strip(),
                raw_value=raw_value,
                raw_unit=raw_unit,
                normalized_value=raw_value * factor,
                dimension=dimension,
                canonical_unit=canonical_unit,
            )
        )
    return facts


def strongest_modality(text: str) -> tuple[str | None, int]:
    lowered = text.lower()
    found = [
        (word, rank)
        for word, rank in MODALITIES.items()
        if re.search(rf"\b{re.escape(word)}\b", lowered)
    ]
    return max(found, key=lambda item: item[1]) if found else (None, 0)


def has_negation(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", lowered) for word in NEGATIONS)


def semantic_overlap(left: Requirement, right: Requirement) -> float:
    a, b = tokens(left.text), tokens(right.text)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compare(
    left: Requirement, right: Requirement, minimum_overlap: float = 0.18
) -> Conflict:
    overlap = semantic_overlap(left, right)
    evidence = (left.requirement_id, right.requirement_id)
    if overlap < minimum_overlap:
        return Conflict(
            left.requirement_id,
            right.requirement_id,
            ConflictKind.NONE,
            overlap,
            "Requirements do not discuss sufficiently similar concepts.",
            evidence,
        )

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
    mismatch = _numeric_mismatch(left_numbers, right_numbers)
    if mismatch is not None:
        left_fact, right_fact, relative_gap = mismatch
        return Conflict(
            left.requirement_id,
            right.requirement_id,
            ConflictKind.NUMERIC,
            min(1.0, 0.60 + 0.30 * overlap + 0.10 * relative_gap),
            "Requirements specify different values for the same dimension: "
            f"{left_fact.raw_text} "
            f"({left_fact.normalized_value:g} {left_fact.canonical_unit}) vs "
            f"{right_fact.raw_text} "
            f"({right_fact.normalized_value:g} {right_fact.canonical_unit}).",
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

    return Conflict(
        left.requirement_id,
        right.requirement_id,
        ConflictKind.NONE,
        overlap,
        "No deterministic contradiction pattern was detected.",
        evidence,
    )


def _equivalent_values(left: float, right: float) -> bool:
    tolerance = 1e-9 * max(abs(left), abs(right), 1.0)
    return abs(left - right) <= tolerance


def _numeric_mismatch(
    left: list[NumericFact], right: list[NumericFact]
) -> tuple[NumericFact, NumericFact, float] | None:
    """Return the strongest mismatch not explained by an equivalent peer fact.

    Comparing every numeric pair creates false positives when requirements list
    the same measurements in a different order. Only values that have no
    equivalent in the other requirement are candidates for a contradiction.
    """
    candidates: list[tuple[float, NumericFact, NumericFact]] = []
    for left_fact in left:
        peers = [fact for fact in right if fact.dimension == left_fact.dimension]
        if any(
            _equivalent_values(left_fact.normalized_value, fact.normalized_value)
            for fact in peers
        ):
            continue
        for right_fact in peers:
            if any(
                fact.dimension == right_fact.dimension
                and _equivalent_values(
                    right_fact.normalized_value, fact.normalized_value
                )
                for fact in left
            ):
                continue
            relative_gap = abs(
                left_fact.normalized_value - right_fact.normalized_value
            ) / max(
                abs(left_fact.normalized_value),
                abs(right_fact.normalized_value),
                1.0,
            )
            candidates.append((relative_gap, left_fact, right_fact))
    if not candidates:
        return None
    relative_gap, left_fact, right_fact = max(candidates, key=lambda item: item[0])
    return left_fact, right_fact, relative_gap


def find_conflicts(
    requirements: Iterable[Requirement], minimum_overlap: float = 0.18
) -> list[Conflict]:
    rows = list(requirements)
    conflicts: list[Conflict] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            result = compare(left, right, minimum_overlap=minimum_overlap)
            if result.kind is not ConflictKind.NONE:
                conflicts.append(result)
    return sorted(conflicts, key=lambda item: item.score, reverse=True)
