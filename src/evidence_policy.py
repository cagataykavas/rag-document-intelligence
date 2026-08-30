from __future__ import annotations

import re
from dataclasses import asdict, dataclass

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
    ("instruction_override", re.compile(r"\b(ignore|disregard|forget)\b.{0,40}\b(instruction|prompt|rule)s?\b", re.I)),
    ("system_prompt_request", re.compile(r"\b(system prompt|developer message|hidden instruction)s?\b", re.I)),
    ("credential_request", re.compile(r"\b(api key|password|secret|access token|credential)s?\b", re.I)),
    ("tool_execution_request", re.compile(r"\b(run|execute|call|invoke)\b.{0,30}\b(tool|command|shell|terminal|function)\b", re.I)),
    ("role_reassignment", re.compile(r"\b(you are now|act as|new role|switch role)\b", re.I)),
)


def assess_evidence(snippets: list[EvidenceSnippet]) -> list[EvidenceFinding]:
    """Flag instruction-like content inside untrusted retrieved documents.

    This is a transparent heuristic policy layer, not a complete prompt-injection
    detector. High-severity chunks are quarantined from the generation context
    but remain visible in the retrieval trace for investigation.
    """
    findings: list[EvidenceFinding] = []
    for snippet in snippets:
        matched = tuple(name for name, pattern in SIGNALS if pattern.search(snippet.text))
        if not matched:
            continue
        severity = "high" if {"instruction_override", "system_prompt_request", "credential_request"} & set(matched) else "medium"
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
