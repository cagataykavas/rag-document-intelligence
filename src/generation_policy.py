from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any

from src.claims import split_claims

SCHEMA_VERSION = "generation-admission/v2"
EXPECTED_FIELDS = frozenset(
    {
        "answer",
        "citations",
        "claim_citations",
        "confidence",
        "insufficient_evidence",
    }
)
CLAIM_CITATION_FIELDS = frozenset({"claim_index", "citations"})
CITATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")


@dataclass(frozen=True)
class GenerationPolicy:
    """Resource and schema limits for untrusted generator responses."""

    max_payload_bytes: int = 64 * 1024
    max_answer_chars: int = 4_000
    max_citations: int = 20
    max_claims: int = 32
    max_citations_per_claim: int = 5


DEFAULT_GENERATION_POLICY = GenerationPolicy()


@dataclass(frozen=True)
class GenerationAdmission:
    """Validated response plus a content-minimizing decision trace."""

    accepted: bool
    reasons: tuple[str, ...]
    answer: str
    citations: tuple[str, ...]
    claim_citations: tuple[tuple[int, tuple[str, ...]], ...]
    confidence: float
    insufficient_evidence: bool
    payload_bytes: int
    citation_count: int
    claim_count: int
    claim_citation_link_count: int
    payload_sha256: str
    claim_map_sha256: str
    allowed_evidence_sha256: str
    policy_sha256: str
    decision_sha256: str

    def to_audit_dict(self) -> dict[str, Any]:
        """Return an audit record without raw answer text or citation IDs."""

        return {
            "schema_version": SCHEMA_VERSION,
            "applied": True,
            "accepted": self.accepted,
            "reasons": self.reasons,
            "payload_bytes": self.payload_bytes,
            "citation_count": self.citation_count,
            "claim_count": self.claim_count,
            "claim_citation_link_count": self.claim_citation_link_count,
            "payload_sha256": self.payload_sha256,
            "claim_map_sha256": self.claim_map_sha256,
            "allowed_evidence_sha256": self.allowed_evidence_sha256,
            "policy_sha256": self.policy_sha256,
            "decision_sha256": self.decision_sha256,
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
        default=lambda item: {"unsupported_python_type": type(item).__name__},
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _add_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def admit_generation(
    payload: Any,
    allowed_evidence_ids: set[str] | frozenset[str],
    policy: GenerationPolicy = DEFAULT_GENERATION_POLICY,
) -> GenerationAdmission:
    """Validate untrusted model output and fail closed on every contract breach.

    The decision trace hashes the payload, policy and evidence whitelist so a run
    can be reproduced without copying model-generated text into logs.
    """

    payload_bytes = _canonical_json(payload)
    allowed = frozenset(str(item) for item in allowed_evidence_ids)
    allowed_bytes = _canonical_json(sorted(allowed))
    policy_bytes = _canonical_json(asdict(policy))
    reasons: list[str] = []
    answer = ""
    citations: tuple[str, ...] = ()
    claim_citations: tuple[tuple[int, tuple[str, ...]], ...] = ()
    claims: tuple[str, ...] = ()
    confidence = 0.0
    insufficient_evidence = True

    if len(payload_bytes) > policy.max_payload_bytes:
        _add_reason(reasons, "payload_too_large")

    if not isinstance(payload, dict):
        _add_reason(reasons, "payload_not_object")
    else:
        keys = set(payload)
        for field in sorted(EXPECTED_FIELDS - keys):
            _add_reason(reasons, f"missing_field:{field}")
        if keys - EXPECTED_FIELDS:
            _add_reason(reasons, "unexpected_field")

        raw_answer = payload.get("answer")
        if not isinstance(raw_answer, str):
            _add_reason(reasons, "answer_not_string")
        else:
            answer = raw_answer.strip()
            if not answer:
                _add_reason(reasons, "answer_empty")
            if len(answer) > policy.max_answer_chars:
                _add_reason(reasons, "answer_too_long")
            claims = split_claims(answer)
            if len(claims) > policy.max_claims:
                _add_reason(reasons, "too_many_claims")

        raw_citations = payload.get("citations")
        if not isinstance(raw_citations, list):
            _add_reason(reasons, "citations_not_array")
        else:
            if len(raw_citations) > policy.max_citations:
                _add_reason(reasons, "too_many_citations")
            if not all(isinstance(item, str) for item in raw_citations):
                _add_reason(reasons, "citation_not_string")
            else:
                citations = tuple(raw_citations)
                if any(not CITATION_ID_RE.fullmatch(item) for item in citations):
                    _add_reason(reasons, "citation_id_invalid")
                if len(set(citations)) != len(citations):
                    _add_reason(reasons, "duplicate_citation")
                if any(item not in allowed for item in citations):
                    _add_reason(reasons, "citation_not_allowed")

        raw_claim_citations = payload.get("claim_citations")
        normalized_claim_citations: list[tuple[int, tuple[str, ...]]] = []
        if not isinstance(raw_claim_citations, list):
            _add_reason(reasons, "claim_citations_not_array")
        else:
            if len(raw_claim_citations) > policy.max_claims:
                _add_reason(reasons, "too_many_claim_citation_entries")
            seen_indexes: set[int] = set()
            for entry in raw_claim_citations:
                if not isinstance(entry, dict) or set(entry) != CLAIM_CITATION_FIELDS:
                    _add_reason(reasons, "claim_citation_entry_invalid")
                    continue
                claim_index = entry["claim_index"]
                if isinstance(claim_index, bool) or not isinstance(claim_index, int):
                    _add_reason(reasons, "claim_index_not_integer")
                    continue
                if claim_index < 0 or claim_index >= len(claims):
                    _add_reason(reasons, "claim_index_out_of_range")
                if claim_index in seen_indexes:
                    _add_reason(reasons, "duplicate_claim_index")
                seen_indexes.add(claim_index)

                raw_claim_citation_ids = entry["citations"]
                if not isinstance(raw_claim_citation_ids, list):
                    _add_reason(reasons, "claim_citation_ids_not_array")
                    continue
                if not raw_claim_citation_ids:
                    _add_reason(reasons, "claim_without_citations")
                if len(raw_claim_citation_ids) > policy.max_citations_per_claim:
                    _add_reason(reasons, "too_many_citations_for_claim")
                if not all(isinstance(item, str) for item in raw_claim_citation_ids):
                    _add_reason(reasons, "claim_citation_id_not_string")
                    continue
                claim_ids = tuple(raw_claim_citation_ids)
                if any(not CITATION_ID_RE.fullmatch(item) for item in claim_ids):
                    _add_reason(reasons, "claim_citation_id_invalid")
                if len(set(claim_ids)) != len(claim_ids):
                    _add_reason(reasons, "duplicate_claim_citation")
                if any(item not in citations for item in claim_ids):
                    _add_reason(reasons, "claim_citation_not_declared")
                if any(item not in allowed for item in claim_ids):
                    _add_reason(reasons, "claim_citation_not_allowed")
                normalized_claim_citations.append((claim_index, claim_ids))

            claim_citations = tuple(sorted(normalized_claim_citations))

        raw_confidence = payload.get("confidence")
        if isinstance(raw_confidence, bool) or not isinstance(
            raw_confidence, (int, float)
        ):
            _add_reason(reasons, "confidence_not_number")
        elif not math.isfinite(float(raw_confidence)):
            _add_reason(reasons, "confidence_not_finite")
        else:
            confidence = float(raw_confidence)
            if not 0.0 <= confidence <= 1.0:
                _add_reason(reasons, "confidence_out_of_range")

        raw_insufficient = payload.get("insufficient_evidence")
        if not isinstance(raw_insufficient, bool):
            _add_reason(reasons, "insufficient_evidence_not_boolean")
        else:
            insufficient_evidence = raw_insufficient
            if insufficient_evidence and citations:
                _add_reason(reasons, "abstention_with_citations")
            if insufficient_evidence and claim_citations:
                _add_reason(reasons, "abstention_with_claim_citations")
            if not insufficient_evidence and not citations:
                _add_reason(reasons, "answer_without_citations")
            if not insufficient_evidence:
                covered_indexes = {index for index, _ in claim_citations}
                if covered_indexes != set(range(len(claims))):
                    _add_reason(reasons, "incomplete_claim_citation_coverage")
                used_citations = {
                    citation for _, claim_ids in claim_citations for citation in claim_ids
                }
                if used_citations != set(citations):
                    _add_reason(reasons, "citation_not_mapped_to_claim")

    accepted = not reasons
    payload_sha256 = _sha256(payload_bytes)
    claim_map_sha256 = _sha256(_canonical_json(claim_citations))
    allowed_evidence_sha256 = _sha256(allowed_bytes)
    policy_sha256 = _sha256(policy_bytes)
    decision_bytes = _canonical_json(
        {
            "schema_version": SCHEMA_VERSION,
            "accepted": accepted,
            "reasons": reasons,
            "payload_sha256": payload_sha256,
            "allowed_evidence_sha256": allowed_evidence_sha256,
            "policy_sha256": policy_sha256,
        }
    )

    return GenerationAdmission(
        accepted=accepted,
        reasons=tuple(reasons),
        answer=answer,
        citations=citations,
        claim_citations=claim_citations,
        confidence=confidence,
        insufficient_evidence=insufficient_evidence,
        payload_bytes=len(payload_bytes),
        citation_count=len(citations),
        claim_count=len(claims),
        claim_citation_link_count=sum(len(items) for _, items in claim_citations),
        payload_sha256=payload_sha256,
        claim_map_sha256=claim_map_sha256,
        allowed_evidence_sha256=allowed_evidence_sha256,
        policy_sha256=policy_sha256,
        decision_sha256=_sha256(decision_bytes),
    )
