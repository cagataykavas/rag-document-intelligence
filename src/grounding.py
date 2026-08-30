from __future__ import annotations

import re
from dataclasses import asdict, dataclass

TOKEN_RE = re.compile(r"[A-Za-z0-9_%-]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


@dataclass(frozen=True)
class CitationAudit:
    citations: tuple[str, ...]
    valid_citations: tuple[str, ...]
    invalid_citations: tuple[str, ...]
    citation_validity_rate: float
    lexical_grounding_rate: float
    answer_content_tokens: int
    supported_content_tokens: int
    insufficient_citation_support: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }


def audit_grounding(
    answer: str,
    citations: tuple[str, ...] | list[str],
    retrieval: tuple[dict, ...] | list[dict],
) -> CitationAudit:
    """Audit citation IDs and lexical support from the cited evidence.

    `lexical_grounding_rate` is intentionally a conservative lexical overlap
    diagnostic. It is useful for detecting obviously unsupported answers, but it
    is not presented as a semantic factuality or entailment score.
    """
    cited = tuple(dict.fromkeys(str(item) for item in citations))
    retrieved_by_id = {
        str(item["chunk_id"]): item
        for item in retrieval
        if isinstance(item, dict) and "chunk_id" in item
    }
    valid = tuple(item for item in cited if item in retrieved_by_id)
    invalid = tuple(item for item in cited if item not in retrieved_by_id)

    validity = len(valid) / len(cited) if cited else 0.0
    answer_tokens = _content_tokens(answer)
    evidence_tokens: set[str] = set()
    for chunk_id in valid:
        evidence_tokens.update(_content_tokens(str(retrieved_by_id[chunk_id].get("text", ""))))
    supported = answer_tokens & evidence_tokens
    grounding = len(supported) / len(answer_tokens) if answer_tokens else 1.0

    return CitationAudit(
        citations=cited,
        valid_citations=valid,
        invalid_citations=invalid,
        citation_validity_rate=validity,
        lexical_grounding_rate=grounding,
        answer_content_tokens=len(answer_tokens),
        supported_content_tokens=len(supported),
        insufficient_citation_support=bool(answer_tokens) and not valid,
    )
