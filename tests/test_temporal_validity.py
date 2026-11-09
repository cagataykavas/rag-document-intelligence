from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from src.temporal_validity import (
    ARTIFACT_SCHEMA,
    MAX_INPUT_BYTES,
    AuditPolicy,
    ClaimReference,
    EvidenceError,
    SourceRevision,
    TemporalArtifact,
    audit_artifact,
    load_artifact,
    main,
)

NOW = datetime(2026, 9, 30, 19, 0, tzinfo=UTC)
AS_OF = datetime(2025, 6, 1, tzinfo=UTC)


def _source(index: int = 1, **changes) -> SourceRevision:
    values = {
        "citation_id": f"citation-{index}",
        "source_id": f"policy-{index}",
        "revision": "v2",
        "published_at": datetime(2025, 1, 1, tzinfo=UTC),
        "indexed_at": datetime(2025, 1, 2, tzinfo=UTC),
        "valid_from": datetime(2025, 1, 1, tzinfo=UTC),
        "valid_until": datetime(2026, 1, 1, tzinfo=UTC),
        "content_sha256": f"{index:064x}",
    }
    values.update(changes)
    return SourceRevision(**values)


def _claim(index: int = 1, **changes) -> ClaimReference:
    values = {
        "claim_id": f"claim-{index}",
        "temporal_mode": "query_as_of",
        "effective_at": AS_OF,
        "impact": "standard",
        "citation_ids": (f"citation-{index}",),
    }
    values.update(changes)
    return ClaimReference(**values)


def _artifact(**changes) -> TemporalArtifact:
    values = {
        "generated_at": NOW - timedelta(seconds=10),
        "query_id": "query-001",
        "query_as_of": AS_OF,
        "model_id": "rag-model-v4",
        "prompt_id": "prompt-v8",
        "retrieval_snapshot_id": "snapshot-2026-09-30",
        "sources": (_source(),),
        "claims": (_claim(),),
    }
    values.update(changes)
    return TemporalArtifact(**values)


def _write(path: Path, artifact: TemporalArtifact | None = None) -> None:
    path.write_text(json.dumps((artifact or _artifact()).canonical_dict()), encoding="utf-8")


def _codes(report: dict) -> set[str]:
    return {finding["code"] for finding in report["findings"]}


def test_valid_as_of_citation_is_accepted() -> None:
    report = audit_artifact(_artifact(), now=NOW)
    assert report["status"] == "accepted"
    assert report["metrics"]["temporal_validity_rate"] == 1.0
    assert report["metrics"]["temporally_valid_citations"] == 1


def test_report_is_deterministic_and_redacts_entity_identifiers() -> None:
    first = audit_artifact(_artifact(), now=NOW)
    second = audit_artifact(_artifact(), now=NOW)
    assert first == second
    serialized = json.dumps(first)
    assert "query-001" not in serialized
    assert "claim-1" not in serialized
    assert "policy-1" not in serialized
    assert len(first["evidence_sha256"]) == 64


@pytest.mark.parametrize(
    "source,code",
    [
        (_source(published_at=AS_OF + timedelta(days=1)), "CIT002"),
        (_source(valid_from=AS_OF + timedelta(days=1)), "CIT003"),
        (_source(valid_until=AS_OF), "CIT004"),
    ],
)
def test_temporally_invalid_source_is_rejected(source: SourceRevision, code: str) -> None:
    report = audit_artifact(_artifact(sources=(source,)), now=NOW)
    assert report["status"] == "rejected"
    assert code in _codes(report)
    assert report["metrics"]["invalid_citations"] == 1


def test_unknown_and_missing_citations_are_rejected() -> None:
    unknown = audit_artifact(
        _artifact(claims=(_claim(citation_ids=("citation-missing",)),)), now=NOW
    )
    missing = audit_artifact(_artifact(claims=(_claim(citation_ids=()),)), now=NOW)
    assert "CIT001" in _codes(unknown)
    assert "COVER001" in _codes(missing)


def test_query_as_of_claim_must_match_query_timestamp() -> None:
    claim = _claim(effective_at=AS_OF - timedelta(days=1))
    assert "CLAIM002" in _codes(audit_artifact(_artifact(claims=(claim,)), now=NOW))


def test_explicit_historical_claim_can_use_its_own_effective_time() -> None:
    historical = datetime(2025, 3, 1, tzinfo=UTC)
    claim = _claim(temporal_mode="explicit", effective_at=historical)
    report = audit_artifact(_artifact(claims=(claim,)), now=NOW)
    assert report["status"] == "accepted"


def test_timeless_claim_uses_generation_time_without_effective_at() -> None:
    source = _source(valid_until=None)
    claim = _claim(temporal_mode="timeless", effective_at=None)
    report = audit_artifact(_artifact(sources=(source,), claims=(claim,)), now=NOW)
    assert report["status"] == "accepted"


def test_critical_claim_cannot_bypass_temporal_binding() -> None:
    source = _source(valid_until=None)
    claim = _claim(temporal_mode="timeless", effective_at=None, impact="critical")
    report = audit_artifact(_artifact(sources=(source,), claims=(claim,)), now=NOW)
    assert "CLAIM004" in _codes(report)


def test_mixed_revisions_of_same_source_are_rejected() -> None:
    first = _source(1, citation_id="citation-old", revision="v1")
    second = _source(2, citation_id="citation-new", source_id=first.source_id, revision="v2")
    claim = _claim(citation_ids=(first.citation_id, second.citation_id))
    report = audit_artifact(_artifact(sources=(first, second), claims=(claim,)), now=NOW)
    assert "CIT005" in _codes(report)
    assert report["metrics"]["revision_conflicts"] == 1


def test_semantically_irrelevant_collection_order_is_canonical() -> None:
    first = _source(1)
    second = _source(2)
    first_claim = _claim(1)
    second_claim = _claim(2)
    forward = _artifact(sources=(first, second), claims=(first_claim, second_claim))
    reverse = _artifact(sources=(second, first), claims=(second_claim, first_claim))
    assert audit_artifact(forward, now=NOW) == audit_artifact(reverse, now=NOW)


def test_maximum_finding_volume_is_report_bounded() -> None:
    claims = tuple(
        ClaimReference(
            claim_id=f"claim-{index}",
            temporal_mode="query_as_of",
            effective_at=AS_OF,
            impact="standard",
            citation_ids=tuple(f"unknown-{index}-{offset}" for offset in range(32)),
        )
        for index in range(1_000)
    )
    report = audit_artifact(_artifact(claims=claims), now=NOW)
    assert report["metrics"]["total_findings"] == 32_000
    assert report["metrics"]["reported_findings"] == 2_048
    assert report["metrics"]["findings_truncated"] is True
    assert len(report["findings"]) == 2_048


def test_source_chronology_and_generation_binding_are_rejected() -> None:
    before_publication = _source(indexed_at=datetime(2024, 12, 1, tzinfo=UTC))
    after_generation = _source(indexed_at=NOW + timedelta(minutes=1))
    assert "SOURCE001" in _codes(audit_artifact(_artifact(sources=(before_publication,)), now=NOW))
    assert "SOURCE002" in _codes(audit_artifact(_artifact(sources=(after_generation,)), now=NOW))


def test_stale_future_and_impossible_query_times_are_rejected() -> None:
    stale = audit_artifact(_artifact(generated_at=NOW - timedelta(hours=1)), now=NOW)
    future = audit_artifact(_artifact(generated_at=NOW + timedelta(minutes=1)), now=NOW)
    query_future = audit_artifact(
        _artifact(
            query_as_of=NOW + timedelta(minutes=1), claims=(_claim(temporal_mode="explicit"),)
        ),
        now=NOW,
    )
    assert "TIME002" in _codes(stale)
    assert "TIME001" in _codes(future)
    assert "TIME003" in _codes(query_future)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda values: values.update(schema_version="wrong"), "schema"),
        (lambda values: values.update(generated_at="2026-09-30T19:00:00"), "timezone"),
        (lambda values: values["sources"][0].update(content_sha256="BAD"), "SHA-256"),
        (lambda values: values["claims"][0].update(temporal_mode="current"), "unsupported"),
        (lambda values: values["claims"][0].update(impact="severe"), "unsupported"),
    ],
)
def test_strict_schema_validation(mutation, match: str) -> None:
    values = _artifact().canonical_dict()
    mutation(values)
    with pytest.raises(EvidenceError, match=match):
        TemporalArtifact.from_dict(values)


def test_interval_duplicate_ids_and_duplicate_citations_fail_closed() -> None:
    values = _artifact().canonical_dict()
    values["sources"][0]["valid_until"] = values["sources"][0]["valid_from"]
    with pytest.raises(EvidenceError, match="later"):
        TemporalArtifact.from_dict(values)

    values = _artifact(sources=(_source(1), _source(2))).canonical_dict()
    values["sources"][1]["citation_id"] = values["sources"][0]["citation_id"]
    with pytest.raises(EvidenceError, match="unique"):
        TemporalArtifact.from_dict(values)

    values = _artifact().canonical_dict()
    values["claims"][0]["citation_ids"] = ["citation-1", "citation-1"]
    with pytest.raises(EvidenceError, match="unique"):
        TemporalArtifact.from_dict(values)


def test_loader_rejects_duplicate_nonfinite_oversized_and_symlink(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="duplicate"):
        load_artifact(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(EvidenceError, match="non-finite"):
        load_artifact(nonfinite)

    oversized = tmp_path / "oversized.json"
    oversized.write_text("x" * (MAX_INPUT_BYTES + 1), encoding="utf-8")
    with pytest.raises(EvidenceError, match="budget"):
        load_artifact(oversized)

    valid = tmp_path / "valid.json"
    linked = tmp_path / "linked.json"
    _write(valid)
    linked.symlink_to(valid)
    with pytest.raises(EvidenceError, match="non-symlink"):
        load_artifact(linked)


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW if tz is not None else NOW.replace(tzinfo=None)


def test_cli_has_distinct_accept_reject_and_malformed_exits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("src.temporal_validity.datetime", _FixedDateTime)
    accepted = tmp_path / "accepted.json"
    rejected = tmp_path / "rejected.json"
    malformed = tmp_path / "malformed.json"
    _write(accepted)
    _write(rejected, _artifact(sources=(_source(valid_until=AS_OF),)))
    malformed.write_text("{", encoding="utf-8")

    assert main(["--input", str(accepted), "--output", str(tmp_path / "a.json")]) == 0
    assert main(["--input", str(rejected), "--output", str(tmp_path / "r.json")]) == 2
    assert main(["--input", str(malformed), "--output", str(tmp_path / "m.json")]) == 3
    assert json.loads((tmp_path / "a.json").read_text())["status"] == "accepted"


def test_policy_validation_and_schema_constant() -> None:
    with pytest.raises(ValueError, match="positive"):
        AuditPolicy(min_claims=0)
    assert _artifact().canonical_dict()["schema_version"] == ARTIFACT_SCHEMA
