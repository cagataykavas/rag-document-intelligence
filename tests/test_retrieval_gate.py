from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from src.parsers import ParsedDocument
from src.retrieval_baseline import (
    RetrievalBaselineError,
    RetrievalRegressionThresholds,
    create_retrieval_baseline,
    serialize_retrieval_baseline,
)
from src.retrieval_gate import (
    CASE_SCHEMA_VERSION,
    RetrievalSnapshotError,
    evaluate_candidate_snapshot,
    gate_candidate_snapshot,
    load_canonical_cases,
    main,
)
from src.store import DocumentStore

SOURCE_COMMIT = "0" * 40


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _database(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    database = tmp_path / "retrieval.db"
    store = DocumentStore(database)
    documents = {
        "access": "Access tokens expire after fifteen minutes and require device binding.",
        "network": "Private services reject public ingress and require mutual TLS authentication.",
        "retention": "Audit records remain encrypted for seven years before verified deletion.",
    }
    doc_ids = {
        name: store.ingest(
            ParsedDocument(f"{name}.md", "text/markdown", text, {"fixture": name}),
            words_per_chunk=4,
            overlap=0,
        ).doc_id
        for name, text in documents.items()
    }
    return database, doc_ids


def _cases(doc_ids: dict[str, str]) -> bytes:
    return _canonical(
        {
            "schema_version": CASE_SCHEMA_VERSION,
            "cases": [
                {
                    "case_id": "access-token-expiry",
                    "query": "How quickly do access tokens expire?",
                    "relevant_doc_ids": [doc_ids["access"]],
                },
                {
                    "case_id": "audit-retention",
                    "query": "How long are audit records retained?",
                    "relevant_doc_ids": [doc_ids["retention"]],
                },
                {
                    "case_id": "private-service-ingress",
                    "query": "Can private services accept public ingress?",
                    "relevant_doc_ids": [doc_ids["network"]],
                },
            ],
        }
    )


def _thresholds() -> RetrievalRegressionThresholds:
    return RetrievalRegressionThresholds(
        max_hit_rate_at_k_regression=0.0,
        max_recall_at_k_regression=0.0,
        max_precision_at_k_regression=0.0,
        max_mean_reciprocal_rank_regression=0.0,
        max_ndcg_at_k_regression=0.0,
        max_mean_unique_documents_at_k_regression=0.0,
        max_duplicate_document_rate_at_k_regression=0.0,
        max_mean_largest_document_share_at_k_regression=0.0,
    )


def _baseline(snapshot, *, metrics=None) -> bytes:
    baseline = create_retrieval_baseline(
        metrics or snapshot.metrics,
        _thresholds(),
        baseline_id="canonical-fixture-k4",
        baseline_version=1,
        source_commit=SOURCE_COMMIT,
        k=snapshot.k,
        corpus_snapshot_sha256=snapshot.corpus_snapshot_sha256,
        evaluation_cases_sha256=snapshot.evaluation_cases_sha256,
    )
    return serialize_retrieval_baseline(baseline)


def test_case_snapshot_is_canonical_bounded_and_content_addressed() -> None:
    payload = _canonical(
        {
            "schema_version": CASE_SCHEMA_VERSION,
            "cases": [
                {
                    "case_id": "one",
                    "query": "token expiry",
                    "relevant_doc_ids": ["doc-a", "doc-b"],
                }
            ],
        }
    )

    first, first_digest = load_canonical_cases(payload)
    second, second_digest = load_canonical_cases(payload)

    assert first == second
    assert first_digest == second_digest
    assert len(first_digest) == 64
    with pytest.raises(RetrievalSnapshotError, match="cases_json_not_canonical"):
        load_canonical_cases(payload + b"\n")
    with pytest.raises(RetrievalSnapshotError, match="cases_duplicate_json_key"):
        load_canonical_cases(b'{"schema_version":"x","schema_version":"y"}')


@pytest.mark.parametrize(
    "relevant_doc_ids",
    [["doc-b", "doc-a"], ["doc-a", "doc-a"], [[]]],
)
def test_case_snapshot_rejects_ambiguous_document_labels(
    relevant_doc_ids: list[object],
) -> None:
    payload = _canonical(
        {
            "schema_version": CASE_SCHEMA_VERSION,
            "cases": [
                {
                    "case_id": "one",
                    "query": "query",
                    "relevant_doc_ids": relevant_doc_ids,
                }
            ],
        }
    )

    with pytest.raises(RetrievalSnapshotError, match="case_value_invalid"):
        load_canonical_cases(payload)


def test_candidate_snapshot_is_repeatable_and_privacy_reduced(tmp_path: Path) -> None:
    database, doc_ids = _database(tmp_path)
    cases = _cases(doc_ids)

    first = evaluate_candidate_snapshot(database, cases, k=4)
    second = evaluate_candidate_snapshot(database, cases, k=4)
    _, receipt = gate_candidate_snapshot(first, _baseline(first))

    assert first == second
    assert first.document_count == 3
    assert first.chunk_count == 9
    assert first.case_count == 3
    assert receipt.meets_policy is True
    serialized = json.dumps(receipt.to_audit_dict())
    assert "access-token-expiry" not in serialized
    assert "How quickly" not in serialized
    assert all(doc_id not in serialized for doc_id in doc_ids.values())


def test_corpus_mutation_invalidates_reviewed_identity(tmp_path: Path) -> None:
    database, doc_ids = _database(tmp_path)
    cases = _cases(doc_ids)
    accepted = evaluate_candidate_snapshot(database, cases, k=4)
    baseline = _baseline(accepted)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE chunks SET text = text || ' changed' WHERE chunk_id = "
            "(SELECT chunk_id FROM chunks ORDER BY chunk_id LIMIT 1)"
        )

    candidate = evaluate_candidate_snapshot(database, cases, k=4)

    assert candidate.corpus_snapshot_sha256 != accepted.corpus_snapshot_sha256
    with pytest.raises(RetrievalBaselineError, match="corpus snapshot"):
        gate_candidate_snapshot(candidate, baseline)


def test_gate_reports_metric_regression_without_sensitive_rows(tmp_path: Path) -> None:
    database, doc_ids = _database(tmp_path)
    snapshot = evaluate_candidate_snapshot(database, _cases(doc_ids), k=4)
    stronger = replace(
        snapshot.metrics,
        mean_unique_documents_at_k=snapshot.metrics.mean_unique_documents_at_k + 0.5,
    )

    comparison, receipt = gate_candidate_snapshot(snapshot, _baseline(snapshot, metrics=stronger))

    assert comparison.meets_policy is False
    assert receipt.failure_reasons == (
        "metric_regressed:mean_unique_documents_at_k",
    )


def test_cli_exits_zero_on_pass_and_two_on_identity_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database, doc_ids = _database(tmp_path)
    cases_path = tmp_path / "cases.json"
    cases_path.write_bytes(_cases(doc_ids))
    snapshot = evaluate_candidate_snapshot(database, cases_path.read_bytes(), k=4)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_bytes(_baseline(snapshot))

    arguments = [
        "--database",
        str(database),
        "--cases",
        str(cases_path),
        "--baseline",
        str(baseline_path),
    ]
    assert main(arguments) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["meets_policy"] is True

    cases_path.write_bytes(
        _cases({**doc_ids, "access": doc_ids["retention"]})
    )
    assert main(arguments) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "retrieval_gate_failed\n"


def test_missing_relevant_document_and_symlinked_database_fail_closed(
    tmp_path: Path,
) -> None:
    database, doc_ids = _database(tmp_path)
    missing = _cases({**doc_ids, "access": "doc-not-indexed"})
    with pytest.raises(RetrievalSnapshotError, match="case_relevant_document_missing"):
        evaluate_candidate_snapshot(database, missing, k=4)

    database_link = tmp_path / "linked.db"
    database_link.symlink_to(database)
    with pytest.raises(RetrievalSnapshotError, match="corpus_database_invalid"):
        evaluate_candidate_snapshot(database_link, _cases(doc_ids), k=4)
