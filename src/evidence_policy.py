from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from urllib.parse import unquote

from src.prompting import EvidenceSnippet


@dataclass(frozen=True)
class EvidenceFinding:
    chunk_id: str
    severity: str
    signals: tuple[str, ...]
    quarantined: bool

    def to_dict(self) -> dict:
        return asdict(self)


SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,40}\b(instruction|prompt|rule)s?\b", re.I
        ),
    ),
    (
        "system_prompt_request",
        re.compile(r"\b(system prompt|developer message|hidden instruction)s?\b", re.I),
    ),
    (
        "credential_request",
        re.compile(r"\b(api key|password|secret|access token|credential)s?\b", re.I),
    ),
    (
        "tool_execution_request",
        re.compile(
            r"\b(run|execute|call|invoke)\b.{0,30}\b(tool|command|shell|terminal|function)\b",
            re.I,
        ),
    ),
    (
        "role_reassignment",
        re.compile(r"\b(you are now|act as|new role|switch role)\b", re.I),
    ),
)

MAX_DECODE_SOURCE_CHARACTERS = 100_000
DECODED_VIEW_SIGNALS = ("html_entity_decoding", "percent_decoding")


def _matched_signals(value: str) -> tuple[str, ...]:
    raw_matches = {name for name, pattern in SIGNALS if pattern.search(value)}
    decoded_matches: set[str] = set()
    decoding_signals: list[str] = []
    if len(value) <= MAX_DECODE_SOURCE_CHARACTERS:
        decoded_views = (
            ("html_entity_decoding", html.unescape(value)),
            ("percent_decoding", unquote(value, errors="replace")),
        )
        for decoding_signal, decoded in decoded_views:
            if decoded == value:
                continue
            view_matches = {
                name for name, pattern in SIGNALS if pattern.search(decoded)
            }
            if view_matches - raw_matches:
                decoding_signals.append(decoding_signal)
            decoded_matches.update(view_matches)

    matched = raw_matches | decoded_matches
    ordered_policy_signals = [name for name, _ in SIGNALS if name in matched]
    return tuple(ordered_policy_signals + decoding_signals)


def assess_evidence(snippets: list[EvidenceSnippet]) -> list[EvidenceFinding]:
    """Flag instruction-like content inside untrusted retrieved documents.

    This is a transparent heuristic policy layer, not a complete prompt-injection
    detector. High-severity chunks are quarantined from the generation context
    but remain visible in the retrieval trace for investigation.
    """
    findings: list[EvidenceFinding] = []
    for snippet in snippets:
        text_signals = _matched_signals(snippet.text)
        source_signals = _matched_signals(snippet.source)
        matched_set = set(text_signals) | set(source_signals)
        matched = tuple(
            name
            for name in (*[item[0] for item in SIGNALS], *DECODED_VIEW_SIGNALS)
            if name in matched_set
        )
        if not matched:
            continue
        severity = (
            "high"
            if {"instruction_override", "system_prompt_request", "credential_request"}
            & set(matched)
            else "medium"
        )
        findings.append(
            EvidenceFinding(
                chunk_id=snippet.chunk_id,
                severity=severity,
                signals=matched,
                quarantined=severity == "high",
            )
        )
    return findings


def apply_evidence_policy(
    snippets: list[EvidenceSnippet],
) -> tuple[list[EvidenceSnippet], list[EvidenceFinding]]:
    findings = assess_evidence(snippets)
    quarantined = {item.chunk_id for item in findings if item.quarantined}
    allowed = [item for item in snippets if item.chunk_id not in quarantined]
    return allowed, findings
