from dataclasses import FrozenInstanceError

import pytest

from src.evidence_snapshot import (
    MAX_CHUNK_BYTES,
    MAX_SNAPSHOT_CHUNKS,
    MAX_SOURCE_BYTES,
    EvidenceSnapshotError,
    build_evidence_snapshot,
    verify_evidence_snapshot,
)
from src.prompting import EvidenceSnippet


def snippet(
    chunk_id: str = "policy:7",
    source: str = "policy.md",
    text: str = "Audit records must be retained for seven years.",
) -> EvidenceSnippet:
    return EvidenceSnippet(chunk_id=chunk_id, source=source, text=text)


def test_snapshot_is_order_independent_and_content_minimizing():
    evidence = [snippet(), snippet("standard:2", "standard.pdf", "Access is logged.")]

    first = build_evidence_snapshot(evidence)
    second = build_evidence_snapshot(reversed(evidence))
    audit = first.to_audit_dict()

    assert first == second
    assert first.chunk_ids == {"policy:7", "standard:2"}
    assert audit["snapshot_sha256"] == first.snapshot_sha256
    assert audit["chunk_count"] == 2
    assert evidence[0].text not in repr(audit)
    assert evidence[0].source not in repr(audit)


@pytest.mark.parametrize(
    "changed",
    [
        snippet(text="Audit records must be retained for six years."),
        snippet(source="revised-policy.md"),
        snippet(chunk_id="policy:8"),
    ],
)
def test_snapshot_changes_when_bound_evidence_changes(changed: EvidenceSnippet):
    baseline = build_evidence_snapshot([snippet()])

    assert (
        build_evidence_snapshot([changed]).snapshot_sha256 != baseline.snapshot_sha256
    )
    assert verify_evidence_snapshot(baseline, [changed]) is False


def test_exact_evidence_verifies_against_snapshot():
    evidence = [snippet(), snippet("standard:2", "standard.pdf", "Access is logged.")]
    snapshot = build_evidence_snapshot(evidence)

    assert verify_evidence_snapshot(snapshot, reversed(evidence)) is True


@pytest.mark.parametrize(
    "evidence",
    [
        [],
        [snippet()] * (MAX_SNAPSHOT_CHUNKS + 1),
        [snippet(), snippet()],
        [snippet(chunk_id="")],
        [snippet(source="")],
        [snippet(text="")],
        [snippet(chunk_id="x" * 201)],
        [snippet(source="x" * (MAX_SOURCE_BYTES + 1))],
        [snippet(text="x" * (MAX_CHUNK_BYTES + 1))],
        [object()],
    ],
)
def test_invalid_or_unbounded_evidence_is_rejected(evidence):
    with pytest.raises(EvidenceSnapshotError):
        build_evidence_snapshot(evidence)


def test_snapshot_objects_are_immutable():
    snapshot = build_evidence_snapshot([snippet()])

    with pytest.raises(FrozenInstanceError):
        snapshot.snapshot_sha256 = "0" * 64
