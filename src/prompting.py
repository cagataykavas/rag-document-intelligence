from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum


class PromptMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"
    FEW_SHOT = "few_shot"


@dataclass(frozen=True)
class EvidenceSnippet:
    chunk_id: str
    source: str
    text: str


EXAMPLES = [
    {
        "question": "What retention period is required?",
        "answer": {
            "answer": "The supplied evidence states a retention period of seven years.",
            "citations": ["policy:2"],
            "confidence": 0.93,
            "insufficient_evidence": False,
        },
    },
    {
        "question": "Does the requirement permit anonymous access?",
        "answer": {
            "answer": "No. The cited requirement explicitly requires authenticated access.",
            "citations": ["security:5"],
            "confidence": 0.96,
            "insufficient_evidence": False,
        },
    },
]


def build_prompt(question: str, evidence: list[EvidenceSnippet], mode: PromptMode) -> str:
    schema = {
        "answer": "string",
        "citations": ["chunk-id"],
        "confidence": "0..1",
        "insufficient_evidence": "boolean",
    }
    sections = [
        "You are an evidence-grounded document analysis assistant.",
        "Use only the supplied evidence. Do not invent sources or facts.",
        "If the evidence is insufficient, say so explicitly.",
        "Return JSON matching this schema:\n" + json.dumps(schema, indent=2),
    ]

    example_count = {
        PromptMode.ZERO_SHOT: 0,
        PromptMode.ONE_SHOT: 1,
        PromptMode.FEW_SHOT: 2,
    }[mode]
    if example_count:
        formatted_examples = []
        for example in EXAMPLES[:example_count]:
            formatted_examples.append(
                "Example question:\n"
                + example["question"]
                + "\nExample answer:\n"
                + json.dumps(example["answer"], ensure_ascii=False)
            )
        sections.append("\n\n".join(formatted_examples))

    evidence_block = "\n\n".join(
        f"[{item.chunk_id}] source={item.source}\n{item.text}" for item in evidence
    )
    sections.append("Evidence:\n" + evidence_block)
    sections.append("Question:\n" + question)
    return "\n\n---\n\n".join(sections)
