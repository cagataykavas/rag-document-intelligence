from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.document_lifecycle_audit import (
    LifecyclePolicy,
    MalformedArtifact,
    audit_document_lifecycle,
    load_artifact,
)

NOW = datetime(2026, 9, 29, 0, 0, tzinfo=UTC)


def artifact() -> dict:
    return {
        "schema_version": "rag-document-lifecycle/v1",
        "observed_at": "2026-09-28T23:59:50Z",
        "catalog_revision": "catalog-42",
        "index_revision": "index-17",
        "documents": [
            {
                "document_id": "retention-policy",
                "current_revision": "rev-3",
                "state": "active",
                "state_changed_at": "2026-09-28T23:00:00Z",
            }
        ],
        "chunks": [
            {
                "chunk_id": "retention-policy:rev-3:0",
                "document_id": "retention-policy",
                "document_revision": "rev-3",
                "indexed_at": "2026-09-28T23:30:00Z",
            }
        ],
        "queries": [
            {
                "query_id": "query-001",
                "retrieved_at": "2026-09-28T23:59:40Z",
                "results": [{"chunk_id": "retention-policy:rev-3:0", "rank": 1}],
            }
        ],
    }


def audit(value: dict, policy: LifecyclePolicy | None = None) -> dict:
    return audit_document_lifecycle(value, policy, now=NOW)


def test_accepts_active_current_revision() -> None:
    report = audit(artifact())

    assert report["accepted"] is True
    assert report["reason_codes"] == []
    assert report["summary"] == {
        "document_count": 1,
        "chunk_count": 1,
        "query_count": 1,
        "served_result_count": 1,
    }
    assert len(report["evidence_id"]) == 64


def test_evidence_is_deterministic() -> None:
    first = audit(artifact())
    second = audit(deepcopy(artifact()))

    assert first == second


def test_report_does_not_expose_raw_identifiers() -> None:
    value = artifact()
    value["documents"][0]["state"] = "deleted"

    rendered = json.dumps(audit(value), sort_keys=True)

    assert "retention-policy" not in rendered
    assert "query-001" not in rendered
    assert "DOCUMENT_NOT_ACTIVE" in rendered


def test_rejects_deleted_document_result() -> None:
    value = artifact()
    value["documents"][0]["state"] = "deleted"

    report = audit(value)

    assert report["accepted"] is False
    assert report["reason_codes"] == ["DOCUMENT_NOT_ACTIVE"]


def test_rejects_superseded_revision() -> None:
    value = artifact()
    value["documents"][0]["current_revision"] = "rev-4"

    report = audit(value)

    assert report["reason_codes"] == ["SUPERSEDED_DOCUMENT_REVISION"]


def test_rejects_unknown_chunk() -> None:
    value = artifact()
    value["queries"][0]["results"][0]["chunk_id"] = "missing:chunk"

    assert audit(value)["reason_codes"] == ["UNKNOWN_CHUNK"]


def test_rejects_unknown_document() -> None:
    value = artifact()
    value["chunks"][0]["document_id"] = "missing-document"

    assert audit(value)["reason_codes"] == ["UNKNOWN_DOCUMENT"]


def test_rejects_chunk_indexed_after_retrieval() -> None:
    value = artifact()
    value["chunks"][0]["indexed_at"] = "2026-09-28T23:59:45Z"

    assert audit(value)["reason_codes"] == ["CHUNK_INDEXED_AFTER_RETRIEVAL"]


def test_rejects_lifecycle_state_recorded_after_retrieval() -> None:
    value = artifact()
    value["documents"][0]["state_changed_at"] = "2026-09-28T23:59:45Z"

    assert audit(value)["reason_codes"] == ["LIFECYCLE_STATE_AFTER_RETRIEVAL"]


def test_rejects_stale_artifact() -> None:
    value = artifact()
    value["observed_at"] = "2026-09-28T23:40:00Z"
    value["queries"][0]["retrieved_at"] = "2026-09-28T23:39:50Z"
    value["documents"][0]["state_changed_at"] = "2026-09-28T23:00:00Z"
    value["chunks"][0]["indexed_at"] = "2026-09-28T23:30:00Z"

    assert audit(value)["reason_codes"] == ["STALE_ARTIFACT"]


def test_rejects_stale_query_evidence() -> None:
    value = artifact()
    value["queries"][0]["retrieved_at"] = "2026-09-28T23:40:00Z"
    value["chunks"][0]["indexed_at"] = "2026-09-28T23:30:00Z"

    assert audit(value)["reason_codes"] == ["STALE_QUERY_EVIDENCE"]


def test_rejects_artifact_too_far_in_future() -> None:
    value = artifact()
    value["observed_at"] = "2026-09-29T00:00:31Z"
    value["queries"][0]["retrieved_at"] = "2026-09-28T23:59:40Z"

    assert audit(value)["reason_codes"] == ["ARTIFACT_FROM_FUTURE"]


def test_accepts_empty_result_set() -> None:
    value = artifact()
    value["queries"][0]["results"] = []

    report = audit(value)

    assert report["accepted"] is True
    assert report["summary"]["served_result_count"] == 0


def test_findings_are_sorted_and_bounded() -> None:
    value = artifact()
    value["queries"][0]["results"] = [
        {"chunk_id": "missing-c", "rank": 3},
        {"chunk_id": "missing-a", "rank": 1},
        {"chunk_id": "missing-b", "rank": 2},
    ]
    policy = LifecyclePolicy(max_reported_findings=2)

    report = audit(value, policy)

    assert report["finding_count"] == 3
    assert len(report["reported_findings"]) == 2
    assert report["truncated_findings"] == 1
    assert report["reason_codes"] == ["UNKNOWN_CHUNK"]


def test_unserved_stale_chunk_is_not_a_release_failure() -> None:
    value = artifact()
    value["chunks"].append(
        {
            "chunk_id": "retention-policy:rev-2:0",
            "document_id": "retention-policy",
            "document_revision": "rev-2",
            "indexed_at": "2026-09-28T22:00:00Z",
        }
    )

    assert audit(value)["accepted"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(schema_version="wrong"), "schema_version"),
        (lambda value: value.update(extra=True), "fields mismatch"),
        (lambda value: value.update(queries=[]), "at least one"),
        (
            lambda value: value["documents"][0].update(state="archived"),
            "state is invalid",
        ),
        (
            lambda value: value["documents"][0].update(document_id=" whitespace"),
            "identifier",
        ),
        (
            lambda value: value["queries"][0].update(retrieved_at="2026-09-28"),
            "timestamp",
        ),
        (
            lambda value: value["queries"][0]["results"][0].update(rank=True),
            "integer",
        ),
    ],
)
def test_rejects_malformed_shape(mutate, message: str) -> None:
    value = artifact()
    mutate(value)

    with pytest.raises(MalformedArtifact, match=message):
        audit(value)


def test_rejects_duplicate_document_id() -> None:
    value = artifact()
    value["documents"].append(deepcopy(value["documents"][0]))

    with pytest.raises(MalformedArtifact, match="duplicate document_id"):
        audit(value)


def test_rejects_duplicate_chunk_id() -> None:
    value = artifact()
    value["chunks"].append(deepcopy(value["chunks"][0]))

    with pytest.raises(MalformedArtifact, match="duplicate chunk_id"):
        audit(value)


def test_rejects_duplicate_query_id() -> None:
    value = artifact()
    value["queries"].append(deepcopy(value["queries"][0]))

    with pytest.raises(MalformedArtifact, match="duplicate query_id"):
        audit(value)


@pytest.mark.parametrize(
    "results",
    [
        [
            {"chunk_id": "retention-policy:rev-3:0", "rank": 1},
            {"chunk_id": "retention-policy:rev-3:0", "rank": 2},
        ],
        [
            {"chunk_id": "retention-policy:rev-3:0", "rank": 1},
            {"chunk_id": "other", "rank": 1},
        ],
        [{"chunk_id": "retention-policy:rev-3:0", "rank": 2}],
    ],
)
def test_rejects_ambiguous_result_ranking(results: list[dict]) -> None:
    value = artifact()
    value["queries"][0]["results"] = results

    with pytest.raises(MalformedArtifact, match="duplicate|contiguous"):
        audit(value)


def test_rejects_query_after_observation() -> None:
    value = artifact()
    value["queries"][0]["retrieved_at"] = "2026-09-29T00:00:00Z"

    with pytest.raises(MalformedArtifact, match="after observed_at"):
        audit(value)


def test_rejects_state_change_after_observation() -> None:
    value = artifact()
    value["documents"][0]["state_changed_at"] = "2026-09-29T00:00:00Z"

    with pytest.raises(MalformedArtifact, match="after observed_at"):
        audit(value)


def test_rejects_chunk_indexed_after_observation() -> None:
    value = artifact()
    value["chunks"][0]["indexed_at"] = "2026-09-29T00:00:00Z"

    with pytest.raises(MalformedArtifact, match="after observed_at"):
        audit(value)


def test_resource_budgets_fail_closed() -> None:
    value = artifact()
    value["queries"][0]["results"] = []
    value["chunks"].append(
        {
            "chunk_id": "second",
            "document_id": "retention-policy",
            "document_revision": "rev-3",
            "indexed_at": "2026-09-28T23:30:00Z",
        }
    )
    with pytest.raises(MalformedArtifact, match="item budget"):
        audit(value, LifecyclePolicy(max_chunks=1))


def test_policy_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        audit(artifact(), LifecyclePolicy(max_queries=0))


def test_loader_rejects_duplicate_json_key(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")

    with pytest.raises(MalformedArtifact, match="duplicate JSON key"):
        load_artifact(path, LifecyclePolicy())


def test_loader_rejects_non_finite_number(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"value": NaN}', encoding="utf-8")

    with pytest.raises(MalformedArtifact, match="non-finite"):
        load_artifact(path, LifecyclePolicy())


def test_loader_enforces_byte_budget(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(" " * 101, encoding="utf-8")

    with pytest.raises(MalformedArtifact, match="byte budget"):
        load_artifact(path, LifecyclePolicy(max_artifact_bytes=100))


def test_cli_exit_codes_and_atomic_output(tmp_path: Path) -> None:
    input_path = tmp_path / "artifact.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(artifact()), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "src.document_lifecycle_audit",
        str(input_path),
        "--output",
        str(output_path),
        "--max-artifact-age-seconds",
        "31536000",
        "--max-future-skew-seconds",
        "3600",
    ]

    accepted = subprocess.run(command, check=False, capture_output=True)

    assert accepted.returncode == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["accepted"] is True
    assert not list(tmp_path.glob(".report.json.*"))

    rejected_value = artifact()
    rejected_value["documents"][0]["state"] = "deleted"
    input_path.write_text(json.dumps(rejected_value), encoding="utf-8")
    rejected = subprocess.run(command, check=False, capture_output=True)
    assert rejected.returncode == 2

    input_path.write_text("{", encoding="utf-8")
    malformed = subprocess.run(command, check=False, capture_output=True)
    assert malformed.returncode == 3
    assert json.loads(malformed.stderr)["error"] == "malformed_artifact"
