"""Fail-closed temporal validity audit for claim-to-citation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA = "rag-temporal-citation-artifact/v1"
REPORT_SCHEMA = "rag-temporal-citation-report/v1"
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_SOURCES = 5_000
MAX_CLAIMS = 1_000
MAX_CITATIONS_PER_CLAIM = 32
MAX_REPORTED_FINDINGS = 2_048
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
TEMPORAL_MODES = {"query_as_of", "explicit", "timeless"}
IMPACT_LEVELS = {"standard", "critical"}


class EvidenceError(ValueError):
    """Raised when temporal evidence cannot be safely interpreted."""


@dataclass(frozen=True)
class AuditPolicy:
    max_evidence_age_seconds: int = 300
    max_future_skew_seconds: int = 30
    min_claims: int = 1
    require_citations: bool = True

    def __post_init__(self) -> None:
        values = (self.max_evidence_age_seconds, self.max_future_skew_seconds, self.min_claims)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError("integer policy values must be non-negative integers")
        if self.min_claims < 1:
            raise ValueError("min_claims must be positive")
        if not isinstance(self.require_citations, bool):
            raise TypeError("boolean policy values must be booleans")


@dataclass(frozen=True)
class SourceRevision:
    citation_id: str
    source_id: str
    revision: str
    published_at: datetime
    indexed_at: datetime
    valid_from: datetime
    valid_until: datetime | None
    content_sha256: str

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "source_id": self.source_id,
            "revision": self.revision,
            "published_at": _iso(self.published_at),
            "indexed_at": _iso(self.indexed_at),
            "valid_from": _iso(self.valid_from),
            "valid_until": _iso(self.valid_until) if self.valid_until else None,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class ClaimReference:
    claim_id: str
    temporal_mode: str
    effective_at: datetime | None
    impact: str
    citation_ids: tuple[str, ...]

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "temporal_mode": self.temporal_mode,
            "effective_at": _iso(self.effective_at) if self.effective_at else None,
            "impact": self.impact,
            "citation_ids": list(self.citation_ids),
        }


@dataclass(frozen=True)
class TemporalArtifact:
    generated_at: datetime
    query_id: str
    query_as_of: datetime
    model_id: str
    prompt_id: str
    retrieval_snapshot_id: str
    sources: tuple[SourceRevision, ...]
    claims: tuple[ClaimReference, ...]

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA,
            "generated_at": _iso(self.generated_at),
            "query_id": self.query_id,
            "query_as_of": _iso(self.query_as_of),
            "model_id": self.model_id,
            "prompt_id": self.prompt_id,
            "retrieval_snapshot_id": self.retrieval_snapshot_id,
            "sources": [source.canonical_dict() for source in self.sources],
            "claims": [claim.canonical_dict() for claim in self.claims],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TemporalArtifact:
        _exact_keys(
            values,
            {
                "schema_version",
                "generated_at",
                "query_id",
                "query_as_of",
                "model_id",
                "prompt_id",
                "retrieval_snapshot_id",
                "sources",
                "claims",
            },
            "artifact",
        )
        if values["schema_version"] != ARTIFACT_SCHEMA:
            raise EvidenceError("unsupported artifact schema")
        raw_sources = values["sources"]
        raw_claims = values["claims"]
        if not isinstance(raw_sources, list) or not 1 <= len(raw_sources) <= MAX_SOURCES:
            raise EvidenceError(f"sources must contain between 1 and {MAX_SOURCES} items")
        if not isinstance(raw_claims, list) or not 1 <= len(raw_claims) <= MAX_CLAIMS:
            raise EvidenceError(f"claims must contain between 1 and {MAX_CLAIMS} items")

        sources: list[SourceRevision] = []
        citation_ids: set[str] = set()
        for index, raw in enumerate(raw_sources):
            _exact_keys(
                raw,
                {
                    "citation_id",
                    "source_id",
                    "revision",
                    "published_at",
                    "indexed_at",
                    "valid_from",
                    "valid_until",
                    "content_sha256",
                },
                f"sources[{index}]",
            )
            citation_id = _identifier(raw["citation_id"], f"sources[{index}].citation_id")
            if citation_id in citation_ids:
                raise EvidenceError("citation_id values must be unique")
            citation_ids.add(citation_id)
            valid_until = (
                _timestamp(raw["valid_until"], f"sources[{index}].valid_until")
                if raw["valid_until"] is not None
                else None
            )
            valid_from = _timestamp(raw["valid_from"], f"sources[{index}].valid_from")
            if valid_until is not None and valid_until <= valid_from:
                raise EvidenceError("source valid_until must be later than valid_from")
            digest = raw["content_sha256"]
            if not isinstance(digest, str) or DIGEST.fullmatch(digest) is None:
                raise EvidenceError(f"sources[{index}].content_sha256 must be lowercase SHA-256")
            sources.append(
                SourceRevision(
                    citation_id=citation_id,
                    source_id=_identifier(raw["source_id"], f"sources[{index}].source_id"),
                    revision=_identifier(raw["revision"], f"sources[{index}].revision"),
                    published_at=_timestamp(raw["published_at"], f"sources[{index}].published_at"),
                    indexed_at=_timestamp(raw["indexed_at"], f"sources[{index}].indexed_at"),
                    valid_from=valid_from,
                    valid_until=valid_until,
                    content_sha256=digest,
                )
            )

        claims: list[ClaimReference] = []
        claim_ids: set[str] = set()
        for index, raw in enumerate(raw_claims):
            _exact_keys(
                raw,
                {"claim_id", "temporal_mode", "effective_at", "impact", "citation_ids"},
                f"claims[{index}]",
            )
            claim_id = _identifier(raw["claim_id"], f"claims[{index}].claim_id")
            if claim_id in claim_ids:
                raise EvidenceError("claim_id values must be unique")
            claim_ids.add(claim_id)
            mode = raw["temporal_mode"]
            if mode not in TEMPORAL_MODES:
                raise EvidenceError(f"claims[{index}].temporal_mode is unsupported")
            impact = raw["impact"]
            if impact not in IMPACT_LEVELS:
                raise EvidenceError(f"claims[{index}].impact is unsupported")
            effective_at = (
                _timestamp(raw["effective_at"], f"claims[{index}].effective_at")
                if raw["effective_at"] is not None
                else None
            )
            if mode == "timeless" and effective_at is not None:
                raise EvidenceError("timeless claims cannot define effective_at")
            if mode != "timeless" and effective_at is None:
                raise EvidenceError("temporal claims must define effective_at")
            raw_citations = raw["citation_ids"]
            if not isinstance(raw_citations, list) or len(raw_citations) > MAX_CITATIONS_PER_CLAIM:
                raise EvidenceError("claim citations must be a bounded list")
            if any(
                not isinstance(item, str) or IDENTIFIER.fullmatch(item) is None
                for item in raw_citations
            ):
                raise EvidenceError("claim citation IDs must be bounded identifiers")
            if len(set(raw_citations)) != len(raw_citations):
                raise EvidenceError("claim citation IDs must be unique")
            claims.append(
                ClaimReference(
                    claim_id,
                    mode,
                    effective_at,
                    impact,
                    tuple(sorted(raw_citations)),
                )
            )

        return cls(
            generated_at=_timestamp(values["generated_at"], "generated_at"),
            query_id=_identifier(values["query_id"], "query_id"),
            query_as_of=_timestamp(values["query_as_of"], "query_as_of"),
            model_id=_identifier(values["model_id"], "model_id"),
            prompt_id=_identifier(values["prompt_id"], "prompt_id"),
            retrieval_snapshot_id=_identifier(
                values["retrieval_snapshot_id"], "retrieval_snapshot_id"
            ),
            sources=tuple(sorted(sources, key=lambda source: source.citation_id)),
            claims=tuple(sorted(claims, key=lambda claim: claim.claim_id)),
        )


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    claim_ref: str | None = None
    source_ref: str | None = None


def _exact_keys(value: Any, expected: set[str], field: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise EvidenceError(f"{field} keys do not match the schema")


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise EvidenceError(f"{field} is not a bounded identifier")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise EvidenceError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise EvidenceError(f"{field} is not valid ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise EvidenceError("timestamps must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _reference(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def audit_artifact(
    artifact: TemporalArtifact,
    policy: AuditPolicy | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    policy = policy or AuditPolicy()
    artifact = TemporalArtifact.from_dict(artifact.canonical_dict())
    evaluated_source = now or datetime.now(UTC)
    if evaluated_source.tzinfo is None or evaluated_source.utcoffset() is None:
        raise EvidenceError("evaluation time must include a timezone")
    evaluated_at = evaluated_source.astimezone(UTC)
    findings: list[Finding] = []
    age = evaluated_at - artifact.generated_at
    future_limit = timedelta(seconds=policy.max_future_skew_seconds)

    if age < -future_limit:
        findings.append(Finding("TIME001", "artifact timestamp exceeds allowed future skew"))
    elif age > timedelta(seconds=policy.max_evidence_age_seconds):
        findings.append(Finding("TIME002", "artifact is older than the evidence-age budget"))
    if artifact.query_as_of > artifact.generated_at + future_limit:
        findings.append(Finding("TIME003", "query as-of time is later than answer generation"))
    if len(artifact.claims) < policy.min_claims:
        findings.append(Finding("CLAIM001", "claim count is below policy"))

    source_by_citation = {source.citation_id: source for source in artifact.sources}
    for source in artifact.sources:
        source_ref = _reference(source.source_id)
        if source.published_at > source.indexed_at:
            findings.append(
                Finding("SOURCE001", "source was indexed before publication", source_ref=source_ref)
            )
        if source.indexed_at > artifact.generated_at + future_limit:
            findings.append(
                Finding(
                    "SOURCE002", "source was indexed after answer generation", source_ref=source_ref
                )
            )

    invalid_citations = 0
    temporally_valid_citations = 0
    checked_citations = 0
    revision_conflicts = 0
    for claim in artifact.claims:
        claim_ref = _reference(claim.claim_id)
        if policy.require_citations and not claim.citation_ids:
            findings.append(Finding("COVER001", "claim has no citations", claim_ref=claim_ref))
            continue
        if claim.temporal_mode == "query_as_of" and claim.effective_at != artifact.query_as_of:
            findings.append(
                Finding(
                    "CLAIM002",
                    "query-as-of claim does not match the query timestamp",
                    claim_ref=claim_ref,
                )
            )
        if claim.impact == "critical" and claim.temporal_mode == "timeless":
            findings.append(
                Finding(
                    "CLAIM004",
                    "critical claims require an explicit temporal reference",
                    claim_ref=claim_ref,
                )
            )
        if (
            claim.effective_at is not None
            and claim.effective_at > artifact.generated_at + future_limit
        ):
            findings.append(
                Finding("CLAIM003", "claim effective time is in the future", claim_ref=claim_ref)
            )

        target = claim.effective_at
        revisions_by_source: dict[str, set[str]] = {}
        for citation_id in claim.citation_ids:
            checked_citations += 1
            source = source_by_citation.get(citation_id)
            if source is None:
                invalid_citations += 1
                findings.append(
                    Finding("CIT001", "claim cites an unknown source", claim_ref=claim_ref)
                )
                continue
            source_ref = _reference(source.source_id)
            revisions_by_source.setdefault(source.source_id, set()).add(source.revision)
            reference_time = target or artifact.generated_at
            valid = True
            if source.published_at > reference_time:
                valid = False
                findings.append(
                    Finding(
                        "CIT002",
                        "citation was published after the claim reference time",
                        claim_ref,
                        source_ref,
                    )
                )
            if source.valid_from > reference_time:
                valid = False
                findings.append(
                    Finding(
                        "CIT003",
                        "citation was not yet effective at the claim reference time",
                        claim_ref,
                        source_ref,
                    )
                )
            if source.valid_until is not None and reference_time >= source.valid_until:
                valid = False
                findings.append(
                    Finding(
                        "CIT004",
                        "citation was no longer effective at the claim reference time",
                        claim_ref,
                        source_ref,
                    )
                )
            if valid:
                temporally_valid_citations += 1
            else:
                invalid_citations += 1
        for source_id, revisions in revisions_by_source.items():
            if len(revisions) > 1:
                revision_conflicts += 1
                findings.append(
                    Finding(
                        "CIT005",
                        "claim mixes multiple revisions of one source",
                        claim_ref,
                        _reference(source_id),
                    )
                )

    findings.sort(
        key=lambda finding: (
            finding.code,
            finding.claim_ref or "",
            finding.source_ref or "",
        )
    )
    total_findings = len(findings)
    reported_findings = findings[:MAX_REPORTED_FINDINGS]
    artifact_bytes = json.dumps(
        artifact.canonical_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "status": "accepted" if not findings else "rejected",
        "evaluated_at": _iso(evaluated_at),
        "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "query_ref": _reference(artifact.query_id),
        "model_id": artifact.model_id,
        "prompt_id": artifact.prompt_id,
        "retrieval_snapshot_id": artifact.retrieval_snapshot_id,
        "query_as_of": _iso(artifact.query_as_of),
        "audit_policy": asdict(policy),
        "metrics": {
            "sources": len(artifact.sources),
            "claims": len(artifact.claims),
            "checked_citations": checked_citations,
            "temporally_valid_citations": temporally_valid_citations,
            "invalid_citations": invalid_citations,
            "revision_conflicts": revision_conflicts,
            "temporal_validity_rate": temporally_valid_citations / checked_citations
            if checked_citations
            else 0.0,
            "critical_claims": sum(claim.impact == "critical" for claim in artifact.claims),
            "total_findings": total_findings,
            "reported_findings": len(reported_findings),
            "findings_truncated": total_findings > len(reported_findings),
        },
        "findings": [
            {key: value for key, value in asdict(finding).items() if value is not None}
            for finding in reported_findings
        ],
        "limitations": [
            "The audit trusts source timestamps, revision metadata, and claim segmentation.",
            "Temporal validity does not prove relevance, entailment, authority, or factual truth.",
        ],
    }
    report_bytes = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    report["evidence_sha256"] = hashlib.sha256(report_bytes).hexdigest()
    return report


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise EvidenceError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON value: {value}")


def load_artifact(path: Path) -> TemporalArtifact:
    metadata = path.stat(follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EvidenceError("artifact must be a regular, non-symlink file")
    if metadata.st_size > MAX_INPUT_BYTES:
        raise EvidenceError(f"artifact exceeds the {MAX_INPUT_BYTES}-byte budget")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise EvidenceError("artifact changed before it could be read")
        payload = handle.read(MAX_INPUT_BYTES + 1)
        final = os.fstat(handle.fileno())
    if len(payload) > MAX_INPUT_BYTES:
        raise EvidenceError(f"artifact exceeds the {MAX_INPUT_BYTES}-byte budget")
    if (opened.st_size, opened.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise EvidenceError("artifact changed while it was being read")
    try:
        values = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError("artifact is not valid JSON") from error
    if not isinstance(values, dict):
        raise EvidenceError("artifact root must be an object")
    return TemporalArtifact.from_dict(values)


def _atomic_write(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit temporal validity of RAG citations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-evidence-age-seconds", type=int, default=300)
    parser.add_argument("--max-future-skew-seconds", type=int, default=30)
    values = parser.parse_args(argv)
    try:
        policy = AuditPolicy(
            max_evidence_age_seconds=values.max_evidence_age_seconds,
            max_future_skew_seconds=values.max_future_skew_seconds,
        )
        report = audit_artifact(load_artifact(values.input), policy)
        _atomic_write(values.output, report)
    except (EvidenceError, OSError, TypeError, ValueError) as error:
        print(json.dumps({"status": "malformed", "error": str(error)}))
        return 3
    print(json.dumps({"status": report["status"], "evidence_sha256": report["evidence_sha256"]}))
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
