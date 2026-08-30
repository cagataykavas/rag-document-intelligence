from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.grounding import STOPWORDS, TOKEN_RE

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ClaimSupport:
    claim: str
    best_chunk_id: str | None
    lexical_support: float
    supported_tokens: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }


def trace_claim_support(
    answer: str,
    citation_ids: tuple[str, ...] | list[str],
    retrieval: tuple[dict, ...] | list[dict],
) -> list[ClaimSupport]:
    """Map answer sentences to their strongest cited chunk by lexical overlap.

    The score is intentionally lexical and inspectable. It highlights claims
    with weak surface support but is not presented as semantic entailment.
    """
    retrieved = {
        str(row["chunk_id"]): row
        for row in retrieval
        if isinstance(row, dict) and "chunk_id" in row
    }
    cited_rows = [retrieved[item] for item in citation_ids if item in retrieved]
    claims = [item.strip() for item in SENTENCE_RE.split(answer.strip()) if item.strip()]
    results: list[ClaimSupport] = []

    for claim in claims:
        claim_tokens = _tokens(claim)
        best_chunk_id: str | None = None
        best_score = 0.0
        best_supported: set[str] = set()
        for row in cited_rows:
            evidence_tokens = _tokens(str(row.get("text", "")))
            supported = claim_tokens & evidence_tokens
            score = len(supported) / len(claim_tokens) if claim_tokens else 1.0
            if score > best_score or (score == best_score and best_chunk_id is None):
                best_score = score
                best_chunk_id = str(row["chunk_id"])
                best_supported = supported
        results.append(
            ClaimSupport(
                claim=claim,
                best_chunk_id=best_chunk_id,
                lexical_support=best_score,
                supported_tokens=tuple(sorted(best_supported)),
            )
        )
    return results
