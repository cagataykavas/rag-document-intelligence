from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.context_order_audit import (
    ArtifactError,
    AuditPolicy,
    audit_context_order,
    load_artifact,
)


NOW = datetime(2026, 9, 25, 23, 0, tzinfo=UTC)


def variant(
    variant_id: str,
    order: list[str],
    *,
    baseline: bool = False,
    citations: list[str] | None = None,
    answer: str = "Administrators must use phishing-resistant MFA.",
    insufficient: bool = False,
) -> dict[str, object]:
    return {
        "variant_id": variant_id,
        "baseline": baseline,
        "ordered_chunk_ids": order,
        "cited_chunk_ids": citations if citations is not None else ["chunk-a", "chunk-c"],
        "answer": answer,
        "insufficient_evidence": insufficient,
    }


def valid_artifact() -> dict[str, object]:
    chunks = ["chunk-a", "chunk-b", "chunk-c", "chunk-d"]
    return {
        "schema_version": 1,
        "benchmark_id": "context-order-release-17",
        "model_id": "local-model-v3",
        "prompt_template_id": "grounded-answer-v2",
        "retrieval_snapshot_digest": "a" * 64,
        "generation_config_digest": "b" * 64,
        "generated_at": NOW.isoformat(),
        "queries": [
            {
                "query_id": "query-auth-1",
                "chunk_ids": chunks,
                "variants": [
                    variant("baseline", chunks, baseline=True),
                    variant("reverse", list(reversed(chunks))),
                    variant("rotate", ["chunk-b", "chunk-c", "chunk-d", "chunk-a"]),
                    variant("interleave", ["chunk-c", "chunk-a", "chunk-d", "chunk-b"]),
                ],
            }
        ],
    }


def audit(artifact: object, policy: AuditPolicy | None = None):
    return audit_context_order(artifact, policy=policy, observed_at=NOW + timedelta(seconds=5))


def test_stable_context_permutations_are_accepted_without_raw_text() -> None:
    report = audit(valid_artifact())

    assert report.accepted is True
    assert report.finding_codes == ()
    query = report.queries[0]
    assert query.variant_count == 4
    assert query.unique_order_count == 4
    assert query.max_normalized_order_distance == 1.0
    assert query.min_citation_jaccard == 1.0
    assert query.min_answer_token_jaccard == 1.0
    rendered = json.dumps(report.to_dict())
    assert "Administrators" not in rendered
    assert "query-auth-1" not in rendered


def test_citation_drift_is_rejected() -> None:
    artifact = valid_artifact()
    artifact["queries"][0]["variants"][2]["cited_chunk_ids"] = ["chunk-b"]

    report = audit(artifact)

    assert report.accepted is False
    assert report.queries[0].min_citation_jaccard == 0.0
    assert "CITATION_SET_DRIFT" in report.finding_codes


def test_answer_lexical_drift_is_rejected() -> None:
    artifact = valid_artifact()
    artifact["queries"][0]["variants"][1]["answer"] = (
        "The retention schedule applies to archived invoices."
    )

    report = audit(artifact)

    assert report.accepted is False
    assert "ANSWER_LEXICAL_DRIFT" in report.finding_codes


def test_abstention_flip_is_rejected() -> None:
    artifact = valid_artifact()
    changed = artifact["queries"][0]["variants"][3]
    changed["insufficient_evidence"] = True
    changed["cited_chunk_ids"] = []
    changed["answer"] = "Insufficient evidence."

    report = audit(artifact)

    assert report.queries[0].abstention_flips == 1
    assert "ABSTENTION_DECISION_FLIP" in report.finding_codes


def test_repeated_or_weak_orders_do_not_count_as_a_permutation_audit() -> None:
    artifact = valid_artifact()
    baseline_order = artifact["queries"][0]["variants"][0]["ordered_chunk_ids"]
    for item in artifact["queries"][0]["variants"][1:]:
        item["ordered_chunk_ids"] = baseline_order.copy()

    report = audit(artifact)

    assert report.queries[0].unique_order_count == 1
    assert "INSUFFICIENT_ORDER_DIVERSITY" in report.finding_codes
    assert "INSUFFICIENT_ORDER_PERTURBATION" in report.finding_codes


def test_too_few_variants_is_explicit() -> None:
    artifact = valid_artifact()
    artifact["queries"][0]["variants"] = artifact["queries"][0]["variants"][:3]
    policy = AuditPolicy(min_unique_orders=3)

    report = audit(artifact, policy)

    assert report.queries[0].reason_codes == ("TOO_FEW_VARIANTS",)


def test_stale_and_future_artifacts_are_rejected() -> None:
    stale = valid_artifact()
    stale["generated_at"] = (NOW - timedelta(days=8)).isoformat()
    future = valid_artifact()
    future["generated_at"] = (NOW + timedelta(minutes=2)).isoformat()

    assert "ARTIFACT_STALE" in audit(stale).finding_codes
    assert "ARTIFACT_FROM_FUTURE" in audit(future).finding_codes


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version=2),
        lambda value: value.update(retrieval_snapshot_digest="not-a-digest"),
        lambda value: value["queries"].append(deepcopy(value["queries"][0])),
        lambda value: value["queries"][0]["chunk_ids"].append("chunk-a"),
        lambda value: value["queries"][0]["variants"].append(
            deepcopy(value["queries"][0]["variants"][0])
        ),
        lambda value: value["queries"][0]["variants"][1]["ordered_chunk_ids"].pop(),
        lambda value: value["queries"][0]["variants"][1]["cited_chunk_ids"].append("unknown-chunk"),
        lambda value: value["queries"][0]["variants"][1].update(insufficient_evidence=True),
        lambda value: value["queries"][0]["variants"][1].update(answer=""),
        lambda value: value["queries"][0]["variants"][1].update(baseline=True),
        lambda value: value["queries"][0].update(extra="unexpected"),
    ],
)
def test_malformed_artifacts_fail_closed(mutation) -> None:
    artifact = valid_artifact()
    mutation(artifact)

    with pytest.raises(ArtifactError):
        audit(artifact)


def test_answer_and_collection_budgets_are_enforced() -> None:
    artifact = valid_artifact()
    artifact["queries"][0]["variants"][0]["answer"] = "x" * 100
    with pytest.raises(ArtifactError):
        audit(artifact, AuditPolicy(max_answer_bytes=32))

    artifact = valid_artifact()
    artifact["queries"] = []
    report = audit(artifact)
    assert report.finding_codes == ("TOO_FEW_QUERIES",)


def test_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        AuditPolicy(min_citation_jaccard=float("nan"))
    with pytest.raises(ValueError):
        AuditPolicy(max_variants_per_query=0)
    with pytest.raises(ValueError):
        AuditPolicy(max_abstention_flips=-1)
    with pytest.raises(ValueError):
        AuditPolicy(min_queries=2, max_queries=1)
    with pytest.raises(ValueError):
        AuditPolicy(min_unique_orders=5, max_variants_per_query=4)
    with pytest.raises(ValueError):
        AuditPolicy(max_chunks_per_query=1)


def test_report_is_deterministic() -> None:
    first = audit(valid_artifact())
    second = audit(valid_artifact())
    assert first == second
    assert len(first.evidence_digest) == 64


def test_loader_rejects_duplicate_keys_non_finite_and_oversize(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ArtifactError):
        load_artifact(duplicate)

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"value":NaN}')
    with pytest.raises(ArtifactError):
        load_artifact(non_finite)

    oversized = tmp_path / "oversized.json"
    oversized.write_text("x" * 33)
    with pytest.raises(ArtifactError):
        load_artifact(oversized, AuditPolicy(max_artifact_bytes=32))


def test_cli_has_distinct_accept_reject_and_malformed_exit_codes(tmp_path: Path) -> None:
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(json.dumps(valid_artifact()))
    output_path = tmp_path / "report.json"
    accepted = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.context_order_audit",
            str(accepted_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert accepted.returncode == 0
    assert json.loads(accepted.stdout)["accepted"] is True
    assert json.loads(output_path.read_text())["accepted"] is True

    rejected_artifact = valid_artifact()
    rejected_artifact["queries"][0]["variants"][1]["answer"] = "Unrelated output."
    rejected_path = tmp_path / "rejected.json"
    rejected_path.write_text(json.dumps(rejected_artifact))
    rejected = subprocess.run(
        [sys.executable, "-m", "src.context_order_audit", str(rejected_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 3
    assert json.loads(rejected.stdout)["accepted"] is False

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{")
    malformed = subprocess.run(
        [sys.executable, "-m", "src.context_order_audit", str(malformed_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert malformed.returncode == 2
    assert json.loads(malformed.stdout) == {
        "accepted": False,
        "error": "malformed_artifact",
    }
