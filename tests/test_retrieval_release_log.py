from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from src.parsers import ParsedDocument
from src.retrieval_baseline import (
    RetrievalRegressionThresholds,
    create_retrieval_baseline,
    serialize_retrieval_baseline,
)
from src.retrieval_gate import (
    CASE_SCHEMA_VERSION,
    INVENTORY_SCHEMA_VERSION,
    RetrievalCandidateSnapshot,
    evaluate_candidate_snapshot,
)
from src.retrieval_inventory_review import (
    create_inventory_review_attestation,
    dump_inventory_review_attestation,
)
from src.retrieval_release_epoch import (
    ReleaseEpochPolicy,
    create_release_epoch_attestation,
    dump_release_epoch_attestation,
)
from src.retrieval_release_log import (
    RETRIEVAL_RELEASE_LOG_GENESIS_SHA256,
    RetrievalReleaseLogError,
    append_reviewed_retrieval_release,
    verify_retrieval_release_log,
)
from src.reviewer_key_status import (
    create_reviewer_key_status_snapshot,
    dump_reviewer_key_status_snapshot,
)
from src.store import DocumentStore

SOURCE_REVISION = "a" * 40
DOCUMENTS = {
    "access": "Access tokens expire after fifteen minutes and require device binding.",
    "network": "Private services reject public ingress and require mutual TLS authentication.",
    "retention": "Audit records remain encrypted for seven years before verified deletion.",
}


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


@dataclass(slots=True)
class ReleaseFixture:
    snapshot: RetrievalCandidateSnapshot
    baseline: bytes
    review: bytes
    status: bytes
    release_epoch: bytes
    reviewer_public_keys: dict[str, bytes]
    status_authority_public_keys: dict[str, bytes]
    release_policy: ReleaseEpochPolicy
    release_private_key: Ed25519PrivateKey
    review_sha256: str
    status_sha256: str


def build_release(
    tmp_path: Path, *, baseline_metrics=None, epoch: int = 1_800_000_060
) -> ReleaseFixture:
    database = tmp_path / "retrieval.db"
    store = DocumentStore(database)
    doc_ids = {
        name: store.ingest(
            ParsedDocument(f"{name}.md", "text/markdown", text, {"fixture": name}),
            words_per_chunk=4,
            overlap=0,
        ).doc_id
        for name, text in DOCUMENTS.items()
    }
    inventory = canonical(
        {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "sources": sorted(
                (
                    {
                        "source": f"{name}.md",
                        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
                    for name, text in DOCUMENTS.items()
                ),
                key=lambda item: (item["source"], item["content_sha256"]),
            ),
        }
    )
    cases = canonical(
        {
            "schema_version": CASE_SCHEMA_VERSION,
            "cases": [
                {
                    "case_id": name,
                    "query": query,
                    "relevant_doc_ids": [doc_ids[name]],
                }
                for name, query in (
                    ("access", "How quickly do access tokens expire?"),
                    ("network", "Can private services accept public ingress?"),
                    ("retention", "How long are audit records retained?"),
                )
            ],
        }
    )
    snapshot = evaluate_candidate_snapshot(database, cases, inventory, k=4)
    thresholds = RetrievalRegressionThresholds(
        max_hit_rate_at_k_regression=0.0,
        max_recall_at_k_regression=0.0,
        max_precision_at_k_regression=0.0,
        max_mean_reciprocal_rank_regression=0.0,
        max_ndcg_at_k_regression=0.0,
        max_mean_unique_documents_at_k_regression=0.0,
        max_duplicate_document_rate_at_k_regression=0.0,
        max_mean_largest_document_share_at_k_regression=0.0,
    )
    baseline = create_retrieval_baseline(
        baseline_metrics or snapshot.metrics,
        thresholds,
        baseline_id="retained-reviewed-retrieval-k4",
        baseline_version=1,
        source_commit=SOURCE_REVISION,
        k=snapshot.k,
        corpus_snapshot_sha256=snapshot.corpus_snapshot_sha256,
        evaluation_cases_sha256=snapshot.evaluation_cases_sha256,
        source_inventory_sha256=snapshot.source_inventory_sha256,
    )
    baseline_payload = serialize_retrieval_baseline(baseline)
    reviewer_private_keys = {
        key_id: Ed25519PrivateKey.generate()
        for key_id in ("reviewer-a", "reviewer-b", "reviewer-c")
    }
    reviewer_public_keys = {
        key_id: key.public_key().public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo
        )
        for key_id, key in reviewer_private_keys.items()
    }
    review = create_inventory_review_attestation(
        source_inventory_sha256=snapshot.source_inventory_sha256,
        baseline_artifact_sha256=baseline.artifact_sha256,
        public_keys=reviewer_public_keys,
        required_signatures=2,
        signers={
            key_id: reviewer_private_keys[key_id]
            for key_id in ("reviewer-a", "reviewer-c")
        },
    )
    authority_private_keys = {
        key_id: Ed25519PrivateKey.generate()
        for key_id in ("status-a", "status-b", "status-c")
    }
    authority_public_keys = {
        key_id: key.public_key().public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo
        )
        for key_id, key in authority_private_keys.items()
    }
    status = create_reviewer_key_status_snapshot(
        review_policy_sha256=review.review_policy_sha256,
        statuses={key_id: "active" for key_id in reviewer_public_keys},
        issued_at_epoch=1_800_000_000,
        expires_at_epoch=1_800_003_600,
        authority_public_keys=authority_public_keys,
        required_authority_signatures=2,
        signers={
            key_id: authority_private_keys[key_id]
            for key_id in ("status-a", "status-c")
        },
    )
    release_private_key = Ed25519PrivateKey.generate()
    release_policy = ReleaseEpochPolicy(
        repository="cagataykavas/rag-document-intelligence",
        workflow=".github/workflows/release-retrieval.yml",
        source_ref="refs/heads/main",
        source_revision=SOURCE_REVISION,
        key_id="retrieval-release-epoch-v1",
        public_key=release_private_key.public_key().public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo
        ),
    )
    release_epoch = create_release_epoch_attestation(
        candidate_snapshot_sha256=snapshot.artifact_sha256,
        baseline_artifact_sha256=baseline.artifact_sha256,
        inventory_review_sha256=review.attestation_sha256,
        key_status_snapshot_sha256=status.snapshot_sha256,
        verification_epoch=epoch,
        policy=release_policy,
        signer=release_private_key,
    )
    return ReleaseFixture(
        snapshot=snapshot,
        baseline=baseline_payload,
        review=dump_inventory_review_attestation(review),
        status=dump_reviewer_key_status_snapshot(status),
        release_epoch=dump_release_epoch_attestation(release_epoch),
        reviewer_public_keys=reviewer_public_keys,
        status_authority_public_keys=authority_public_keys,
        release_policy=release_policy,
        release_private_key=release_private_key,
        review_sha256=review.attestation_sha256,
        status_sha256=status.snapshot_sha256,
    )


def append(path: Path, release: ReleaseFixture, *, expected_head: str):
    return append_reviewed_retrieval_release(
        path,
        release.snapshot,
        release.baseline,
        release.review,
        release.status,
        release.release_epoch,
        reviewer_public_keys=release.reviewer_public_keys,
        required_review_signatures=2,
        status_authority_public_keys=release.status_authority_public_keys,
        required_status_authority_signatures=2,
        release_epoch_policy=release.release_policy,
        expected_head_sha256=expected_head,
    )


def replace_epoch(release: ReleaseFixture, epoch: int) -> ReleaseFixture:
    baseline = json.loads(release.baseline)
    attestation = create_release_epoch_attestation(
        candidate_snapshot_sha256=release.snapshot.artifact_sha256,
        baseline_artifact_sha256=baseline["artifact_sha256"],
        inventory_review_sha256=release.review_sha256,
        key_status_snapshot_sha256=release.status_sha256,
        verification_epoch=epoch,
        policy=release.release_policy,
        signer=release.release_private_key,
    )
    return replace(release, release_epoch=dump_release_epoch_attestation(attestation))


def test_records_only_a_fully_reviewed_acceptance_and_replay_is_idempotent(
    tmp_path: Path,
) -> None:
    release = build_release(tmp_path)
    path = tmp_path / "retrieval-releases.jsonl"

    first = append(path, release, expected_head=RETRIEVAL_RELEASE_LOG_GENESIS_SHA256)
    replay = append(path, release, expected_head=RETRIEVAL_RELEASE_LOG_GENESIS_SHA256)
    verified = verify_retrieval_release_log(
        path, expected_head_sha256=first.entry_sha256
    )

    assert replay == first
    assert verified.entries == (first,)
    assert first.sequence == 1
    assert first.verification_epoch == 1_800_000_060
    serialized = path.read_text(encoding="ascii")
    assert "access.md" not in serialized
    assert "reviewer-a" not in serialized
    assert "cagataykavas" not in serialized


def test_concurrent_exact_retries_produce_one_entry(tmp_path: Path) -> None:
    release = build_release(tmp_path)
    path = tmp_path / "retrieval-releases.jsonl"

    with ThreadPoolExecutor(max_workers=8) as executor:
        entries = list(
            executor.map(
                lambda _: append(
                    path,
                    release,
                    expected_head=RETRIEVAL_RELEASE_LOG_GENESIS_SHA256,
                ),
                range(8),
            )
        )

    verified = verify_retrieval_release_log(path)
    assert len(set(entries)) == 1
    assert verified.entries == (entries[0],)


def test_rejected_candidate_does_not_create_a_release_log(tmp_path: Path) -> None:
    accepted = build_release(tmp_path)
    stronger = replace(
        accepted.snapshot.metrics,
        mean_unique_documents_at_k=accepted.snapshot.metrics.mean_unique_documents_at_k
        + 0.5,
    )
    rejected = build_release(tmp_path / "rejected", baseline_metrics=stronger)
    path = tmp_path / "retrieval-releases.jsonl"

    with pytest.raises(RetrievalReleaseLogError, match="gate rejected"):
        append(path, rejected, expected_head=RETRIEVAL_RELEASE_LOG_GENESIS_SHA256)

    assert not path.exists()


def test_stale_writer_and_nonadvancing_epoch_leave_chain_unchanged(
    tmp_path: Path,
) -> None:
    release = build_release(tmp_path)
    path = tmp_path / "retrieval-releases.jsonl"
    first = append(path, release, expected_head=RETRIEVAL_RELEASE_LOG_GENESIS_SHA256)
    before = path.read_bytes()

    later = replace_epoch(release, 1_800_000_120)
    with pytest.raises(RetrievalReleaseLogError, match="stale writer"):
        append(path, later, expected_head=RETRIEVAL_RELEASE_LOG_GENESIS_SHA256)
    earlier = replace_epoch(release, 1_800_000_030)
    with pytest.raises(RetrievalReleaseLogError, match="does not advance"):
        append(path, earlier, expected_head=first.entry_sha256)

    assert path.read_bytes() == before


def test_verifier_detects_tampering_truncation_and_unsafe_files(tmp_path: Path) -> None:
    release = build_release(tmp_path)
    path = tmp_path / "retrieval-releases.jsonl"
    append(path, release, expected_head=RETRIEVAL_RELEASE_LOG_GENESIS_SHA256)
    original = path.read_bytes()

    document = json.loads(original)
    document["comparison_sha256"] = "f" * 64
    path.write_bytes(canonical(document) + b"\n")
    with pytest.raises(RetrievalReleaseLogError, match="digest mismatch"):
        verify_retrieval_release_log(path)

    path.write_bytes(original[:-1])
    with pytest.raises(RetrievalReleaseLogError, match="partial entry"):
        verify_retrieval_release_log(path)

    path.write_bytes(original)
    hard_link = tmp_path / "hard-linked.jsonl"
    os.link(path, hard_link)
    with pytest.raises(RetrievalReleaseLogError, match="unsafe"):
        verify_retrieval_release_log(path)
