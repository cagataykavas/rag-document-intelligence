from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import asdict, dataclass

from src.grounding import STOPWORDS, TOKEN_RE

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CLAIM_SUPPORT_SCHEMA_VERSION = "claim-support/v1"
SOURCE_SPAN_SCHEMA_VERSION = "claim-source-span/v1"


def split_claims(answer: str) -> tuple[str, ...]:
    """Return the deterministic sentence-like units used by policy and tracing."""

    return tuple(
        item.strip() for item in SENTENCE_RE.split(answer.strip()) if item.strip()
    )


@dataclass(frozen=True)
class ClaimSupport:
    claim: str
    claim_sha256: str
    best_chunk_id: str | None
    lexical_support: float
    supported_tokens: tuple[str, ...]
    source_span_start_utf8: int | None
    source_span_end_utf8: int | None
    source_span_sha256: str | None
    source_span_commitment_sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) >= 3 and token.lower() not in STOPWORDS
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    """Split text like claims while retaining exact, trimmed character offsets."""

    spans: list[tuple[int, int, str]] = []
    start = 0
    boundaries = [match for match in SENTENCE_RE.finditer(text)]
    for boundary in boundaries:
        raw = text[start : boundary.start()]
        leading = len(raw) - len(raw.lstrip())
        trimmed = raw.strip()
        if trimmed:
            span_start = start + leading
            spans.append((span_start, span_start + len(trimmed), trimmed))
        start = boundary.end()
    raw = text[start:]
    leading = len(raw) - len(raw.lstrip())
    trimmed = raw.strip()
    if trimmed:
        span_start = start + leading
        spans.append((span_start, span_start + len(trimmed), trimmed))
    return tuple(spans)


def _span_commitment(
    *,
    claim_sha256: str,
    chunk_id: str | None,
    start_utf8: int | None,
    end_utf8: int | None,
    span_sha256: str | None,
) -> str:
    return _sha256(
        _canonical_json(
            {
                "schema_version": SOURCE_SPAN_SCHEMA_VERSION,
                "claim_sha256": claim_sha256,
                "chunk_id": chunk_id,
                "source_span_start_utf8": start_utf8,
                "source_span_end_utf8": end_utf8,
                "source_span_sha256": span_sha256,
            }
        )
    )


def claim_support_sha256(rows: tuple[ClaimSupport, ...] | list[ClaimSupport]) -> str:
    """Commit to ordered claim/span links without copying claims or passages into audit logs."""

    commitments = []
    for row in rows:
        if not isinstance(row, ClaimSupport):
            raise TypeError("claim support digest requires ClaimSupport rows")
        commitments.append(
            {
                "claim_sha256": row.claim_sha256,
                "best_chunk_id": row.best_chunk_id,
                "source_span_start_utf8": row.source_span_start_utf8,
                "source_span_end_utf8": row.source_span_end_utf8,
                "source_span_sha256": row.source_span_sha256,
                "source_span_commitment_sha256": row.source_span_commitment_sha256,
            }
        )
    return _sha256(
        _canonical_json(
            {
                "schema_version": CLAIM_SUPPORT_SCHEMA_VERSION,
                "supports": commitments,
            }
        )
    )


EMPTY_CLAIM_SUPPORT_SHA256 = claim_support_sha256([])


def verify_claim_source_span(support: ClaimSupport, chunk_text: str) -> bool:
    """Verify one half-open UTF-8 byte span and its claim/chunk commitment."""

    if not isinstance(support, ClaimSupport) or not isinstance(chunk_text, str):
        return False
    start = support.source_span_start_utf8
    end = support.source_span_end_utf8
    span_sha256 = support.source_span_sha256
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end
        or not isinstance(span_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", span_sha256) is None
    ):
        return False
    encoded = chunk_text.encode("utf-8")
    if end > len(encoded):
        return False
    expected_span = _sha256(encoded[start:end])
    expected_claim = _sha256(support.claim.encode("utf-8"))
    expected_commitment = _span_commitment(
        claim_sha256=expected_claim,
        chunk_id=support.best_chunk_id,
        start_utf8=start,
        end_utf8=end,
        span_sha256=expected_span,
    )
    return (
        hmac.compare_digest(support.claim_sha256, expected_claim)
        and hmac.compare_digest(span_sha256, expected_span)
        and hmac.compare_digest(
            support.source_span_commitment_sha256,
            expected_commitment,
        )
    )


def verify_claim_support_trace(
    rows: tuple[ClaimSupport, ...] | list[ClaimSupport],
    chunk_text_by_id: dict[str, str],
    expected_sha256: str,
) -> bool:
    """Verify the ordered manifest plus every claim and exact source span."""

    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        or not isinstance(chunk_text_by_id, dict)
    ):
        return False
    try:
        actual_sha256 = claim_support_sha256(rows)
    except (TypeError, ValueError):
        return False
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        return False
    for row in rows:
        if row.best_chunk_id is None:
            expected_claim = _sha256(row.claim.encode("utf-8"))
            empty_commitment = _span_commitment(
                claim_sha256=expected_claim,
                chunk_id=None,
                start_utf8=None,
                end_utf8=None,
                span_sha256=None,
            )
            if (
                row.source_span_start_utf8 is not None
                or row.source_span_end_utf8 is not None
                or row.source_span_sha256 is not None
                or not hmac.compare_digest(row.claim_sha256, expected_claim)
                or not hmac.compare_digest(
                    row.source_span_commitment_sha256,
                    empty_commitment,
                )
            ):
                return False
            continue
        chunk_text = chunk_text_by_id.get(row.best_chunk_id)
        if chunk_text is None or not verify_claim_source_span(row, chunk_text):
            return False
    return True


def trace_claim_support(
    answer: str,
    citation_ids: tuple[str, ...] | list[str],
    retrieval: tuple[dict, ...] | list[dict],
    claim_citations: tuple[tuple[int, tuple[str, ...]], ...] | None = None,
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
    claims = split_claims(answer)
    citations_by_claim = dict(claim_citations or ())
    results: list[ClaimSupport] = []

    for claim_index, claim in enumerate(claims):
        scoped_ids = citations_by_claim.get(claim_index, tuple(citation_ids))
        cited_rows = [retrieved[item] for item in scoped_ids if item in retrieved]
        claim_tokens = _tokens(claim)
        candidates: list[tuple[float, int, int, str, int, int, str, set[str]]] = []
        for row in cited_rows:
            text = str(row.get("text", ""))
            for start_char, end_char, span in _source_spans(text):
                supported = claim_tokens & _tokens(span)
                score = len(supported) / len(claim_tokens) if claim_tokens else 1.0
                start_utf8 = len(text[:start_char].encode("utf-8"))
                end_utf8 = start_utf8 + len(span.encode("utf-8"))
                candidates.append(
                    (
                        score,
                        len(supported),
                        end_utf8 - start_utf8,
                        str(row["chunk_id"]),
                        start_utf8,
                        end_utf8,
                        span,
                        supported,
                    )
                )
        best = min(
            candidates,
            key=lambda item: (-item[0], -item[1], item[2], item[3], item[4]),
            default=None,
        )
        claim_sha256 = _sha256(claim.encode("utf-8"))
        if best is None:
            best_score = 0.0
            best_supported: set[str] = set()
            best_chunk_id = None
            start_utf8 = None
            end_utf8 = None
            span_sha256 = None
        else:
            (
                best_score,
                _supported_count,
                _span_bytes,
                best_chunk_id,
                start_utf8,
                end_utf8,
                span,
                best_supported,
            ) = best
            span_sha256 = _sha256(span.encode("utf-8"))
        results.append(
            ClaimSupport(
                claim=claim,
                claim_sha256=claim_sha256,
                best_chunk_id=best_chunk_id,
                lexical_support=best_score,
                supported_tokens=tuple(sorted(best_supported)),
                source_span_start_utf8=start_utf8,
                source_span_end_utf8=end_utf8,
                source_span_sha256=span_sha256,
                source_span_commitment_sha256=_span_commitment(
                    claim_sha256=claim_sha256,
                    chunk_id=best_chunk_id,
                    start_utf8=start_utf8,
                    end_utf8=end_utf8,
                    span_sha256=span_sha256,
                ),
            )
        )
    return results
