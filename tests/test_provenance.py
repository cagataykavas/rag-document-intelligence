from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from src.provenance import (
    ProvenanceArtifactError,
    ProvenancePolicy,
    audit_provenance,
    bind_query_results,
    build_snapshot,
)
from src.retrieval import Document, SparseRetriever, chunk_documents

NOW = datetime(2026, 9, 24, tzinfo=UTC)


def artifact() -> dict:
    chunks = chunk_documents(
        [
            Document("doc-a", "administrators require hardware security keys", "policy-a"),
            Document("doc-b", "audit evidence is retained for seven years", "policy-b"),
        ],
        words_per_chunk=20,
        overlap=0,
    )
    snapshot = build_snapshot(
        chunks,
        snapshot_id="index-2026-09-24",
        retriever_id="word-tfidf-v1",
        chunking_policy_id="words-20-overlap-0-v1",
        created_at=NOW - timedelta(hours=1),
    )
    retrieval = SparseRetriever().fit(chunks).search("hardware security keys", k=2)
    return {
        "snapshot": snapshot,
        "queries": [bind_query_results("query-1", retrieval, snapshot)],
    }


def test_accepts_retrieval_from_exact_snapshot() -> None:
    report = audit_provenance(
        artifact(),
        policy=ProvenancePolicy(min_results_per_query=2),
        evaluated_at=NOW,
    )
    assert report.accepted is True
    assert report.malformed is False
    assert report.reason_codes == ()
    assert report.inventory_chunks == 2
    assert report.result_count == 2
    assert report.queries[0].verified_chunk_ids == ("doc-a:0", "doc-b:0")


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda value: value["snapshot"].update(snapshot_id="other"), "SNAPSHOT_DIGEST_MISMATCH"),
        (
            lambda value: value["snapshot"]["chunks"].append(value["snapshot"]["chunks"][0]),
            "DUPLICATE_CHUNK_ID",
        ),
        (
            lambda value: value["queries"][0].update(snapshot_id="other"),
            "SNAPSHOT_BINDING_MISMATCH",
        ),
        (
            lambda value: value["queries"][0].update(retriever_id="other"),
            "RETRIEVER_BINDING_MISMATCH",
        ),
        (lambda value: value["queries"][0]["results"][0].update(rank=2), "INVALID_RANK_SEQUENCE"),
        (
            lambda value: value["queries"][0]["results"][1].update(
                chunk_id=value["queries"][0]["results"][0]["chunk_id"]
            ),
            "DUPLICATE_QUERY_CHUNK",
        ),
        (
            lambda value: value["queries"][0]["results"][0].update(chunk_id="unknown"),
            "UNKNOWN_CHUNK",
        ),
        (
            lambda value: value["queries"][0]["results"][0].update(text="changed"),
            "RESULT_CONTENT_DIGEST_MISMATCH",
        ),
        (
            lambda value: value["queries"][0]["results"][0].update(document_id="doc-b"),
            "SNAPSHOT_MEMBERSHIP_MISMATCH",
        ),
    ],
)
def test_malformed_provenance_fails_closed(mutation, reason: str) -> None:
    payload = artifact()
    mutation(payload)
    report = audit_provenance(payload, evaluated_at=NOW)
    assert report.accepted is False
    assert report.malformed is True
    assert report.reason_codes == (reason,)
    assert report.error_message


def test_policy_rejects_stale_but_well_formed_snapshot() -> None:
    report = audit_provenance(
        artifact(),
        policy=ProvenancePolicy(max_snapshot_age_seconds=30),
        evaluated_at=NOW,
    )
    assert report.accepted is False
    assert report.malformed is False
    assert report.reason_codes == ("STALE_INDEX_SNAPSHOT",)


def test_policy_reports_all_evidence_shortfalls() -> None:
    payload = artifact()
    payload["queries"][0]["results"].clear()
    report = audit_provenance(
        payload,
        policy=ProvenancePolicy(min_queries=2, min_results_per_query=2),
        evaluated_at=NOW,
    )
    assert report.reason_codes == (
        "INSUFFICIENT_QUERY_EVIDENCE",
        "INSUFFICIENT_RESULTS_PER_QUERY",
    )


def test_future_snapshot_and_oversized_chunk_fail_closed() -> None:
    payload = artifact()
    payload["snapshot"]["created_at"] = (NOW + timedelta(seconds=1)).isoformat()
    report = audit_provenance(payload, evaluated_at=NOW)
    assert report.reason_codes == ("FUTURE_SNAPSHOT",)

    payload = artifact()
    report = audit_provenance(
        payload,
        policy=ProvenancePolicy(max_chunk_bytes=4),
        evaluated_at=NOW,
    )
    assert report.reason_codes == ("CHUNK_BYTE_BUDGET_EXCEEDED",)


def test_builders_reject_duplicate_or_unknown_chunks() -> None:
    chunks = chunk_documents([Document("doc-a", "one two three")], words_per_chunk=5, overlap=0)
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        build_snapshot(
            [chunks[0], chunks[0]],
            snapshot_id="snapshot",
            retriever_id="retriever",
            chunking_policy_id="chunker",
            created_at=NOW,
        )
    snapshot = build_snapshot(
        chunks,
        snapshot_id="snapshot",
        retriever_id="retriever",
        chunking_policy_id="chunker",
        created_at=NOW,
    )
    with pytest.raises(ProvenanceArtifactError, match="absent from snapshot"):
        bind_query_results(
            "query",
            [{"chunk_id": "unknown", "doc_id": "doc-a", "text": "one"}],
            snapshot,
        )
    with pytest.raises(ProvenanceArtifactError, match="differs from the snapshot"):
        bind_query_results(
            "query",
            [{"chunk_id": "doc-a:0", "doc_id": "doc-a", "text": "changed"}],
            snapshot,
        )


def test_invalid_policy_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ProvenancePolicy(min_queries=True)
    with pytest.raises(ValueError, match="cannot exceed"):
        ProvenancePolicy(min_results_per_query=2, max_results_per_query=1)


def test_cli_exit_codes_distinguish_accept_reject_and_malformed(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact()), encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "src.provenance",
        str(path),
        "--evaluated-at",
        NOW.isoformat(),
    ]
    accepted = subprocess.run(command, check=False, capture_output=True, text=True)
    assert accepted.returncode == 0
    assert json.loads(accepted.stdout)["accepted"] is True

    rejected = subprocess.run(
        [*command, "--max-snapshot-age-seconds", "30"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 3
    assert json.loads(rejected.stdout)["malformed"] is False

    malformed_payload = deepcopy(artifact())
    malformed_payload["queries"][0]["results"][0]["text"] = "tampered"
    path.write_text(json.dumps(malformed_payload), encoding="utf-8")
    malformed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert malformed.returncode == 2
    assert json.loads(malformed.stdout)["malformed"] is True
