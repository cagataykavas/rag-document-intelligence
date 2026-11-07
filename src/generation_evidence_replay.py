from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping
from typing import Any

from src.admission_authentication import (
    DEFAULT_CLOCK_SKEW_SECONDS,
    AdmissionAuthenticationError,
    verify_admission_authentication,
)
from src.claims import claim_support_sha256, trace_claim_support
from src.evidence_snapshot import EvidenceSnapshotError, build_evidence_snapshot
from src.generation_policy import (
    CONTENT_BOUND_GENERATION_POLICY,
    GenerationPolicy,
    admit_generation,
    bind_claim_support,
)
from src.prompting import EvidenceSnippet

SCHEMA_VERSION = "generation-evidence-replay/v1"


class GenerationEvidenceReplayError(ValueError):
    """Stable, non-sensitive replay verification failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GenerationEvidenceReplayError("ADMISSION_AUDIT_INVALID") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_audit(audit: Mapping[str, object], fields: set[str]) -> dict[str, object]:
    if not isinstance(audit, Mapping) or set(audit) != fields:
        raise GenerationEvidenceReplayError("ADMISSION_AUDIT_INVALID")
    normalized = dict(audit)
    reasons = normalized.get("reasons")
    if not isinstance(reasons, (list, tuple)):
        raise GenerationEvidenceReplayError("ADMISSION_AUDIT_INVALID")
    normalized["reasons"] = list(reasons)
    return normalized


def replay_generation_evidence(
    payload: Any,
    evidence: Iterable[EvidenceSnippet],
    expected_admission_audit: Mapping[str, object],
    *,
    policy: GenerationPolicy = CONTENT_BOUND_GENERATION_POLICY,
    authentication: Mapping[str, object] | None = None,
    keyring: Mapping[str, bytes] | None = None,
    now_epoch: int | None = None,
    max_age_seconds: int | None = None,
    clock_skew_seconds: int = DEFAULT_CLOCK_SKEW_SECONDS,
) -> dict[str, object]:
    """Rebuild and verify one generation decision from retained private inputs.

    The caller supplies the original model payload and the exact post-quarantine
    evidence. The returned receipt contains commitments and counts only; it does
    not copy answer text, citation IDs, source names, or passages.
    """

    if not isinstance(policy, GenerationPolicy):
        raise GenerationEvidenceReplayError("POLICY_INVALID")
    try:
        evidence_rows = tuple(evidence)
        snapshot = build_evidence_snapshot(evidence_rows)
    except (EvidenceSnapshotError, AttributeError, TypeError, UnicodeError) as exc:
        raise GenerationEvidenceReplayError("EVIDENCE_INVALID") from exc

    admission = admit_generation(
        payload,
        snapshot.chunk_ids,
        policy,
        evidence_snapshot_sha256=snapshot.snapshot_sha256,
    )
    if admission.accepted:
        retrieval = [
            {"chunk_id": item.chunk_id, "text": item.text} for item in evidence_rows
        ]
        support = trace_claim_support(
            admission.answer,
            admission.citations,
            retrieval,
            admission.claim_citations,
        )
        admission = bind_claim_support(admission, claim_support_sha256(support))

    actual_audit = admission.to_audit_dict()
    fields = set(actual_audit)
    expected = _normalized_audit(expected_admission_audit, fields)
    actual = _normalized_audit(actual_audit, fields)
    if not hmac.compare_digest(
        _sha256(_canonical_json(expected)),
        _sha256(_canonical_json(actual)),
    ):
        raise GenerationEvidenceReplayError("ADMISSION_AUDIT_MISMATCH")

    authentication_verified = False
    if authentication is not None and not isinstance(authentication, Mapping):
        raise GenerationEvidenceReplayError("AUTHENTICATION_INVALID")
    if authentication is not None and authentication.get("applied") is False:
        if dict(authentication) != {"applied": False}:
            raise GenerationEvidenceReplayError("AUTHENTICATION_INVALID")
    elif authentication is not None:
        if keyring is None:
            raise GenerationEvidenceReplayError("KEYRING_REQUIRED")
        try:
            verify_admission_authentication(
                expected,
                authentication,
                keyring,
                now_epoch=now_epoch,
                max_age_seconds=max_age_seconds,
                clock_skew_seconds=clock_skew_seconds,
            )
        except AdmissionAuthenticationError as exc:
            raise GenerationEvidenceReplayError("AUTHENTICATION_INVALID") from exc
        authentication_verified = True
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "applied": True,
        "verified": True,
        "accepted": admission.accepted,
        "authentication_verified": authentication_verified,
        "evidence_chunk_count": len(evidence_rows),
        "claim_count": admission.claim_count,
        "payload_sha256": admission.payload_sha256,
        "evidence_snapshot_sha256": snapshot.snapshot_sha256,
        "claim_support_sha256": admission.claim_support_sha256,
        "decision_sha256": admission.decision_sha256,
    }
    receipt["receipt_sha256"] = _sha256(_canonical_json(receipt))
    return receipt
