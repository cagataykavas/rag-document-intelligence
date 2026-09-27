from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.citation_coverage import (
    CitationCoveragePolicy,
    CoverageArtifactError,
    audit_citation_coverage,
    load_artifact,
)

NOW = datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc)
SHA_A = "a" * 64
SHA_B = "b" * 64


def artifact() -> dict:
    answer = "Audit logs are retained for seven years. The API p95 target is 250 ms."
    first = "Audit logs are retained for seven years."
    second = "The API p95 target is 250 ms."
    second_start = answer.index(second)
    return {
        "schema_version": 1,
        "generated_at": "2026-09-27T09:59:00Z",
        "response_kind": "answer",
        "query_id": "query-17",
        "model_id": "generator-v3",
        "prompt_sha256": SHA_A,
        "retrieval_snapshot_sha256": SHA_B,
        "answer": answer,
        "retrieved_chunk_ids": ["chunk-policy", "chunk-performance"],
        "claims": [
            {
                "claim_id": "claim-1",
                "start_char": 0,
                "end_char": len(first),
                "text": first,
                "claim_type": "factual",
                "citation_ids": ["chunk-policy"],
            },
            {
                "claim_id": "claim-2",
                "start_char": second_start,
                "end_char": second_start + len(second),
                "text": second,
                "claim_type": "quantitative",
                "citation_ids": ["chunk-performance"],
            },
        ],
    }


def test_accepts_complete_material_claim_coverage() -> None:
    report = audit_citation_coverage(artifact(), now=NOW)
    assert report.accepted is True
    assert report.reason_codes == ()
    assert report.material_claim_coverage == 1.0
    assert report.content_token_coverage == 1.0
    assert len(report.evidence_sha256) == 64


def test_report_is_deterministic_and_does_not_expose_answer() -> None:
    first = audit_citation_coverage(artifact(), now=NOW).as_dict()
    second = audit_citation_coverage(artifact(), now=NOW).as_dict()
    assert first == second
    encoded = json.dumps(first)
    assert "seven years" not in encoded
    assert "chunk-policy" not in encoded


def test_rejects_uncited_material_claim() -> None:
    payload = artifact()
    payload["claims"][0]["citation_ids"] = []
    report = audit_citation_coverage(payload, now=NOW)
    assert report.accepted is False
    assert "MATERIAL_CLAIM_UNCITED" in report.reason_codes
    assert "MATERIAL_CITATION_COVERAGE_BELOW_THRESHOLD" in report.reason_codes


def test_rejects_citation_not_in_retrieval_manifest() -> None:
    payload = artifact()
    payload["claims"][0]["citation_ids"] = ["chunk-other"]
    report = audit_citation_coverage(payload, now=NOW)
    assert report.accepted is False
    assert "CITATION_OUTSIDE_RETRIEVAL" in report.reason_codes
    assert report.invalid_citation_count == 1


def test_quantitative_literal_cannot_hide_as_opinion() -> None:
    payload = artifact()
    payload["claims"][1]["claim_type"] = "opinion"
    payload["claims"][1]["citation_ids"] = []
    report = audit_citation_coverage(payload, now=NOW)
    assert report.accepted is False
    assert "STRUCTURED_LITERAL_UNCITED" in report.reason_codes


def test_rejects_unmapped_answer_content() -> None:
    payload = artifact()
    payload["answer"] += " This extra factual sentence is omitted from the claim manifest."
    report = audit_citation_coverage(payload, now=NOW)
    assert report.accepted is False
    assert "CONTENT_COVERAGE_BELOW_THRESHOLD" in report.reason_codes


def test_rejects_answer_without_material_claims() -> None:
    payload = artifact()
    for claim in payload["claims"]:
        claim["claim_type"] = "opinion"
    report = audit_citation_coverage(payload, now=NOW)
    assert report.accepted is False
    assert "ANSWER_HAS_NO_MATERIAL_CLAIMS" in report.reason_codes


def test_accepts_bounded_abstention_without_material_claim() -> None:
    answer = "Insufficient retrieved evidence."
    payload = artifact()
    payload["response_kind"] = "abstention"
    payload["answer"] = answer
    payload["claims"] = [
        {
            "claim_id": "claim-abstain",
            "start_char": 0,
            "end_char": len(answer),
            "text": answer,
            "claim_type": "disclaimer",
            "citation_ids": [],
        }
    ]
    assert audit_citation_coverage(payload, now=NOW).accepted is True


def test_rejects_abstention_containing_material_claim() -> None:
    payload = artifact()
    payload["response_kind"] = "abstention"
    report = audit_citation_coverage(payload, now=NOW)
    assert "ABSTENTION_CONTAINS_MATERIAL_CLAIMS" in report.reason_codes


@pytest.mark.parametrize(
    ("generated_at", "code"),
    [
        ("2026-09-25T09:59:00Z", "ARTIFACT_STALE"),
        ("2026-09-27T10:06:00Z", "ARTIFACT_FROM_FUTURE"),
    ],
)
def test_rejects_stale_or_future_artifact(generated_at: str, code: str) -> None:
    payload = artifact()
    payload["generated_at"] = generated_at
    assert code in audit_citation_coverage(payload, now=NOW).reason_codes


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.update(extra=True),
        lambda value: value["claims"][0].update(extra=True),
        lambda value: value["claims"][0].update(text="mismatch"),
        lambda value: value["claims"][1].update(start_char=0),
        lambda value: value["claims"][1].update(claim_id="claim-1"),
        lambda value: value.update(retrieved_chunk_ids=["chunk-policy", "chunk-policy"]),
        lambda value: value["claims"][0].update(citation_ids=["chunk-policy", "chunk-policy"]),
        lambda value: value.update(generated_at="2026-09-27T09:59:00"),
    ],
)
def test_rejects_malformed_contract(mutator) -> None:
    payload = artifact()
    mutator(payload)
    with pytest.raises(CoverageArtifactError):
        audit_citation_coverage(payload, now=NOW)


def test_canonical_digest_is_independent_of_json_key_order() -> None:
    original = artifact()
    reordered = {key: original[key] for key in reversed(list(original))}
    assert (
        audit_citation_coverage(original, now=NOW).artifact_sha256
        == audit_citation_coverage(reordered, now=NOW).artifact_sha256
    )


def test_policy_rejects_non_finite_and_boolean_values() -> None:
    with pytest.raises(ValueError):
        CitationCoveragePolicy(min_content_token_coverage=float("nan"))
    with pytest.raises(ValueError):
        CitationCoveragePolicy(max_claims=True)


def test_strict_loader_rejects_duplicate_keys_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(CoverageArtifactError, match="DUPLICATE_JSON_KEY"):
        load_artifact(duplicate, CitationCoveragePolicy())
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"score":NaN}', encoding="utf-8")
    with pytest.raises(CoverageArtifactError, match="NON_FINITE_JSON_NUMBER"):
        load_artifact(nonfinite, CitationCoveragePolicy())


def test_loader_enforces_byte_budget(tmp_path: Path) -> None:
    path = tmp_path / "large.json"
    path.write_text("x" * 1_025, encoding="utf-8")
    with pytest.raises(CoverageArtifactError, match="ARTIFACT_BYTE_BUDGET_EXCEEDED"):
        load_artifact(path, CitationCoveragePolicy(max_artifact_bytes=1_024))


def run_cli(tmp_path: Path, payload: dict | str) -> subprocess.CompletedProcess[str]:
    path = tmp_path / "artifact.json"
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "src.citation_coverage",
            str(path),
            "--now",
            "2026-09-27T10:00:00Z",
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_has_distinct_success_rejection_and_malformed_codes(tmp_path: Path) -> None:
    accepted = run_cli(tmp_path, artifact())
    rejected_payload = deepcopy(artifact())
    rejected_payload["claims"][0]["citation_ids"] = []
    rejected = run_cli(tmp_path, rejected_payload)
    malformed = run_cli(tmp_path, "{broken")
    assert accepted.returncode == 0
    assert rejected.returncode == 2
    assert malformed.returncode == 3
    assert json.loads(accepted.stdout)["accepted"] is True
    assert json.loads(rejected.stdout)["accepted"] is False
    assert json.loads(malformed.stdout)["status"] == "malformed"


def test_finding_order_is_stable() -> None:
    payload = artifact()
    payload["claims"][1]["claim_type"] = "opinion"
    payload["claims"][1]["citation_ids"] = ["missing"]
    report = audit_citation_coverage(payload, now=NOW)
    assert report.reason_codes == tuple(sorted(report.reason_codes))
