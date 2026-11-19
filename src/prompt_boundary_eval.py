from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from src.evidence_policy import apply_evidence_policy
from src.prompt_eval import Generator, PromptEvalCase, evaluate_prompt_modes
from src.prompting import (
    EVIDENCE_ENVELOPE_VERSION,
    EvidenceSnippet,
    PromptMode,
    build_prompt,
)

MAX_CASE_BYTES = 256_000
MAX_CASES = 200
MAX_SNIPPETS_PER_CASE = 100
MAX_FIELD_CHARACTERS = 100_000
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
EVIDENCE_MARKER = "Untrusted evidence JSON follows:\n"
QUESTION_MARKER = "\n\n---\n\nQuestion (JSON string):\n"


@dataclass(frozen=True)
class PromptBoundaryCase:
    case_id: str
    question: str
    evidence: tuple[EvidenceSnippet, ...]
    expected_quarantined_chunk_ids: frozenset[str]
    expected_insufficient_evidence: bool = False


@dataclass(frozen=True)
class PromptBoundaryCaseResult:
    case_id: str
    expected_quarantined_chunk_ids: tuple[str, ...]
    actual_quarantined_chunk_ids: tuple[str, ...]
    policy_matched: bool
    structural_roundtrip_valid: bool
    finding_signals: tuple[str, ...]


@dataclass(frozen=True)
class PromptBoundaryReport:
    cases: int
    case_set_sha256: str
    boundary_sha256: str
    policy_accuracy: float
    structural_integrity_rate: float
    model_output_evaluated: bool
    model_modes: dict[str, dict] | None
    results: tuple[PromptBoundaryCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["results"] = [asdict(item) for item in self.results]
        return payload


def load_boundary_cases(path: str | Path) -> list[PromptBoundaryCase]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("boundary benchmark cases must be a regular file")
    if source.stat().st_size > MAX_CASE_BYTES:
        raise ValueError(f"boundary benchmark cases exceed {MAX_CASE_BYTES} bytes")
    raw = json.loads(
        source.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
    )
    if not isinstance(raw, list) or not raw or len(raw) > MAX_CASES:
        raise ValueError(f"boundary benchmark requires 1..{MAX_CASES} cases")

    cases = [_parse_case(value) for value in raw]
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("boundary benchmark case IDs must be unique")
    return cases


def evaluate_prompt_boundary(
    cases: list[PromptBoundaryCase],
    *,
    generator: Generator | None = None,
    modes: tuple[PromptMode, ...] = (PromptMode.ZERO_SHOT,),
) -> PromptBoundaryReport:
    if not cases or len(cases) > MAX_CASES:
        raise ValueError(f"boundary benchmark requires 1..{MAX_CASES} cases")
    _validate_case_contract(cases)

    case_results: list[PromptBoundaryCaseResult] = []
    prompt_cases: list[PromptEvalCase] = []
    for case in cases:
        safe_evidence, findings = apply_evidence_policy(list(case.evidence))
        actual_quarantined = tuple(
            sorted(item.chunk_id for item in findings if item.quarantined)
        )
        expected_quarantined = tuple(sorted(case.expected_quarantined_chunk_ids))
        prompt = build_prompt(case.question, safe_evidence, PromptMode.ZERO_SHOT)
        structural_valid = _roundtrip_matches(prompt, case.question, safe_evidence)
        case_results.append(
            PromptBoundaryCaseResult(
                case_id=case.case_id,
                expected_quarantined_chunk_ids=expected_quarantined,
                actual_quarantined_chunk_ids=actual_quarantined,
                policy_matched=actual_quarantined == expected_quarantined,
                structural_roundtrip_valid=structural_valid,
                finding_signals=tuple(
                    sorted(
                        {signal for finding in findings for signal in finding.signals}
                    )
                ),
            )
        )
        prompt_cases.append(
            PromptEvalCase(
                question=case.question,
                evidence=tuple(safe_evidence),
                expected_insufficient_evidence=case.expected_insufficient_evidence,
            )
        )

    policy_accuracy = sum(item.policy_matched for item in case_results) / len(
        case_results
    )
    structural_rate = sum(
        item.structural_roundtrip_valid for item in case_results
    ) / len(case_results)
    model_modes = (
        evaluate_prompt_modes(prompt_cases, generator, modes=modes)
        if generator is not None
        else None
    )
    case_set_sha256 = _case_set_digest(cases)
    boundary_payload = {
        "case_set_sha256": case_set_sha256,
        "policy_accuracy": policy_accuracy,
        "structural_integrity_rate": structural_rate,
        "results": [
            asdict(item) for item in sorted(case_results, key=lambda item: item.case_id)
        ],
    }
    return PromptBoundaryReport(
        cases=len(case_results),
        case_set_sha256=case_set_sha256,
        boundary_sha256=_digest(boundary_payload),
        policy_accuracy=policy_accuracy,
        structural_integrity_rate=structural_rate,
        model_output_evaluated=generator is not None,
        model_modes=model_modes,
        results=tuple(case_results),
    )


def _roundtrip_matches(
    prompt: str,
    question: str,
    safe_evidence: list[EvidenceSnippet],
) -> bool:
    if prompt.count(EVIDENCE_MARKER) != 1 or prompt.count(QUESTION_MARKER) != 1:
        return False
    envelope_and_question = prompt.split(EVIDENCE_MARKER, 1)[1]
    envelope_text, question_text = envelope_and_question.split(QUESTION_MARKER, 1)
    try:
        envelope = json.loads(envelope_text)
        decoded_question = json.loads(question_text)
    except (json.JSONDecodeError, TypeError):
        return False
    expected_envelope = {
        "version": EVIDENCE_ENVELOPE_VERSION,
        "snippets": [
            {"chunk_id": item.chunk_id, "source": item.source, "text": item.text}
            for item in safe_evidence
        ],
    }
    return envelope == expected_envelope and decoded_question == question


def _parse_case(value: object) -> PromptBoundaryCase:
    if not isinstance(value, dict):
        raise ValueError("each boundary benchmark case must be an object")
    allowed_keys = {
        "case_id",
        "question",
        "evidence",
        "expected_quarantined_chunk_ids",
        "expected_insufficient_evidence",
    }
    if set(value) - allowed_keys:
        raise ValueError("boundary benchmark case contains unknown fields")
    case_id = _required_string(value, "case_id", MAX_FIELD_CHARACTERS)
    if not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError("boundary benchmark case_id is invalid")
    question = _required_string(value, "question", MAX_FIELD_CHARACTERS)
    rows = value.get("evidence")
    if not isinstance(rows, list) or not rows or len(rows) > MAX_SNIPPETS_PER_CASE:
        raise ValueError(f"case evidence requires 1..{MAX_SNIPPETS_PER_CASE} snippets")
    snippets = tuple(_parse_snippet(row) for row in rows)
    chunk_ids = [item.chunk_id for item in snippets]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("evidence chunk IDs must be unique within each case")

    expected = value.get("expected_quarantined_chunk_ids", [])
    if not isinstance(expected, list) or any(
        not isinstance(item, str) for item in expected
    ):
        raise ValueError("expected_quarantined_chunk_ids must be a string array")
    expected_set = frozenset(expected)
    if len(expected) != len(expected_set) or not expected_set.issubset(chunk_ids):
        raise ValueError("expected quarantined IDs must be unique evidence chunk IDs")
    insufficient = value.get("expected_insufficient_evidence", False)
    if not isinstance(insufficient, bool):
        raise ValueError("expected_insufficient_evidence must be boolean")
    return PromptBoundaryCase(
        case_id=case_id,
        question=question,
        evidence=snippets,
        expected_quarantined_chunk_ids=expected_set,
        expected_insufficient_evidence=insufficient,
    )


def _validate_case_contract(cases: list[PromptBoundaryCase]) -> None:
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("boundary benchmark case IDs must be unique")
    for case in cases:
        if not isinstance(case.case_id, str) or not CASE_ID_PATTERN.fullmatch(
            case.case_id
        ):
            raise ValueError("boundary benchmark case_id is invalid")
        if (
            not isinstance(case.question, str)
            or not case.question
            or len(case.question) > MAX_FIELD_CHARACTERS
        ):
            raise ValueError("boundary benchmark question is invalid")
        if not 1 <= len(case.evidence) <= MAX_SNIPPETS_PER_CASE:
            raise ValueError(
                f"case evidence requires 1..{MAX_SNIPPETS_PER_CASE} snippets"
            )
        chunk_ids: list[str] = []
        for snippet in case.evidence:
            if not isinstance(snippet, EvidenceSnippet):
                raise ValueError("case evidence must contain EvidenceSnippet values")
            for value, limit in (
                (snippet.chunk_id, 256),
                (snippet.source, 2_000),
                (snippet.text, MAX_FIELD_CHARACTERS),
            ):
                if not isinstance(value, str) or not value or len(value) > limit:
                    raise ValueError("evidence snippet contains an invalid field")
            chunk_ids.append(snippet.chunk_id)
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("evidence chunk IDs must be unique within each case")
        if not isinstance(
            case.expected_quarantined_chunk_ids, frozenset
        ) or not case.expected_quarantined_chunk_ids.issubset(chunk_ids):
            raise ValueError("expected quarantined IDs must be evidence chunk IDs")
        if not isinstance(case.expected_insufficient_evidence, bool):
            raise ValueError("expected_insufficient_evidence must be boolean")


def _case_set_digest(cases: list[PromptBoundaryCase]) -> str:
    payload = [
        {
            "case_id": case.case_id,
            "question": case.question,
            "evidence": [asdict(item) for item in case.evidence],
            "expected_quarantined_chunk_ids": sorted(
                case.expected_quarantined_chunk_ids
            ),
            "expected_insufficient_evidence": case.expected_insufficient_evidence,
        }
        for case in sorted(cases, key=lambda item: item.case_id)
    ]
    return _digest(payload)


def _digest(value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _parse_snippet(value: object) -> EvidenceSnippet:
    if not isinstance(value, dict) or set(value) != {"chunk_id", "source", "text"}:
        raise ValueError("evidence snippets require exactly chunk_id, source, and text")
    return EvidenceSnippet(
        chunk_id=_required_string(value, "chunk_id", 256),
        source=_required_string(value, "source", 2_000),
        text=_required_string(value, "text", MAX_FIELD_CHARACTERS),
    )


def _required_string(value: dict, key: str, limit: int) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item or len(item) > limit:
        raise ValueError(f"{key} must be a non-empty string up to {limit} characters")
    return item


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result
