from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from src.entailment_eval import (
    DriftSlicePolicy,
    EntailmentLabel,
    EntailmentPair,
    EntailmentPolicy,
    EntailmentSliceMetadata,
    evaluate_entailment_diagnostic,
)
from src.entailment_paired_regression import create_evaluation_score_ledger
from src.entailment_slice_assignment import (
    SLICE_ASSIGNMENT_SCHEMA_VERSION,
    SliceAssignmentError,
    create_evaluation_slice_assignment,
    join_evaluation_slice_assignments,
    load_evaluation_slice_assignment,
)


_ENTAILMENT_POLICY = EntailmentPolicy(
    min_supported_precision=0.0,
    min_supported_recall=0.0,
)
_SLICE_POLICY = DriftSlicePolicy(
    min_pairs_per_split=2,
    min_pairs_per_label_per_split=1,
    claim_length_utf8_boundaries=(64, 128),
    evidence_length_utf8_boundaries=(64, 128),
)


def _case(split: str, label: EntailmentLabel, index: int) -> EntailmentPair:
    case_id = f"{split}-{label.value}-{index}"
    token = hashlib.sha256(case_id.encode()).hexdigest()
    return EntailmentPair(
        case_id=case_id,
        claim=f"Control {token} is satisfied.",
        evidence=f"Evidence {token[::-1]} records the reviewed outcome.",
        label=label,
    )


def _fixture(*, scorer_id: str = "nli-v1"):
    calibration = [
        _case("cal", label, index)
        for label in EntailmentLabel
        for index in range(4)
    ]
    evaluation = [
        _case("eval", label, index)
        for label in EntailmentLabel
        for index in range(4)
    ]
    all_pairs = [*calibration, *evaluation]
    metadata = {
        pair.case_id: EntailmentSliceMetadata(
            language="en" if index % 2 == 0 else "tr",
            document_domain="security" if index % 4 < 2 else "finance",
        )
        for index, pair in enumerate(all_pairs)
    }
    scores_by_claim = {
        pair.claim: 0.9 if pair.label is EntailmentLabel.SUPPORTED else 0.1
        for pair in all_pairs
    }
    scores_by_case_id = {
        pair.case_id: scores_by_claim[pair.claim] for pair in evaluation
    }
    report = evaluate_entailment_diagnostic(
        calibration,
        evaluation,
        lambda claim, _evidence: scores_by_claim[claim],
        scorer_id=scorer_id,
        policy=_ENTAILMENT_POLICY,
        drift_slice_metadata=metadata,
        drift_slice_policy=_SLICE_POLICY,
    )
    ledger = create_evaluation_score_ledger(
        report,
        evaluation,
        scores_by_case_id,
    )
    assignments = create_evaluation_slice_assignment(
        report,
        calibration,
        evaluation,
        metadata,
        _SLICE_POLICY,
    )
    return report, calibration, evaluation, metadata, ledger, assignments


def test_assignment_is_deterministic_content_bound_and_privacy_reduced() -> None:
    report, calibration, evaluation, metadata, _ledger, first = _fixture()
    second = create_evaluation_slice_assignment(
        report,
        calibration,
        evaluation,
        metadata,
        _SLICE_POLICY,
    )

    assert first == second
    assert first.schema_version == SLICE_ASSIGNMENT_SCHEMA_VERSION
    assert first.combined_assignment_sha256 == report.slice_assignment_sha256
    assert len(first.rows) == len(evaluation)
    serialized = json.dumps(first.to_audit_dict(), sort_keys=True)
    assert "Control " not in serialized
    assert "Evidence " not in serialized
    assert evaluation[0].case_id not in serialized
    assert "security" in serialized


def test_assignment_round_trip_rejects_tampering_and_extra_fields() -> None:
    *_unused, assignments = _fixture()
    document = json.loads(json.dumps(assignments.to_audit_dict()))

    assert load_evaluation_slice_assignment(document) == assignments

    tampered = json.loads(json.dumps(document))
    tampered["rows"][0]["language"] = "de"
    with pytest.raises(SliceAssignmentError, match="evaluation assignment digest"):
        load_evaluation_slice_assignment(tampered)

    document["unexpected"] = True
    with pytest.raises(SliceAssignmentError, match="malformed"):
        load_evaluation_slice_assignment(document)


def test_assignment_loader_rejects_noncanonical_rows() -> None:
    *_unused, assignments = _fixture()
    document = json.loads(json.dumps(assignments.to_audit_dict()))
    document["rows"][0], document["rows"][1] = (
        document["rows"][1],
        document["rows"][0],
    )

    with pytest.raises(SliceAssignmentError, match="not canonical"):
        load_evaluation_slice_assignment(document)


def test_creation_rejects_policy_or_assignment_not_committed_by_report() -> None:
    report, calibration, evaluation, metadata, _ledger, _assignments = _fixture()
    incompatible_policy = replace(_SLICE_POLICY, claim_length_utf8_boundaries=(32, 64))
    with pytest.raises(SliceAssignmentError, match="policy does not match"):
        create_evaluation_slice_assignment(
            report,
            calibration,
            evaluation,
            metadata,
            incompatible_policy,
        )

    changed_metadata = dict(metadata)
    changed_metadata[evaluation[0].case_id] = EntailmentSliceMetadata(
        language="de",
        document_domain="security",
    )
    with pytest.raises(SliceAssignmentError, match="does not match report"):
        create_evaluation_slice_assignment(
            report,
            calibration,
            evaluation,
            changed_metadata,
            _SLICE_POLICY,
        )


def test_creation_requires_exact_metadata_coverage() -> None:
    report, calibration, evaluation, metadata, _ledger, _assignments = _fixture()
    incomplete = dict(metadata)
    incomplete.pop(evaluation[0].case_id)

    with pytest.raises(SliceAssignmentError, match="exactly cover"):
        create_evaluation_slice_assignment(
            report,
            calibration,
            evaluation,
            incomplete,
            _SLICE_POLICY,
        )


def test_verified_assignments_join_exact_score_ledger_population() -> None:
    *_prefix, ledger, assignments = _fixture()

    rows = join_evaluation_slice_assignments(ledger, assignments)

    assert len(rows) == len(ledger.rows)
    assert [row.case_id_sha256 for row in rows] == [
        row.case_id_sha256 for row in ledger.rows
    ]
    assert {row.language for row in rows} == {"en", "tr"}
    assert all(row.score in {0.1, 0.9} for row in rows)


def test_join_rejects_assignment_from_different_report() -> None:
    *_prefix, ledger, _assignments = _fixture(scorer_id="baseline-v1")
    *_other_prefix, other_assignments = _fixture(scorer_id="candidate-v2")

    with pytest.raises(SliceAssignmentError, match="incompatible"):
        join_evaluation_slice_assignments(ledger, other_assignments)


def test_join_rejects_content_change_that_preserves_slice_bucket() -> None:
    report, calibration, evaluation, metadata, ledger, _assignments = _fixture()
    changed_evaluation = list(evaluation)
    changed_evaluation[0] = replace(
        changed_evaluation[0],
        claim=changed_evaluation[0].claim[:-1] + "!",
    )
    changed_assignments = create_evaluation_slice_assignment(
        report,
        calibration,
        changed_evaluation,
        metadata,
        _SLICE_POLICY,
    )

    with pytest.raises(SliceAssignmentError, match="populations differ"):
        join_evaluation_slice_assignments(ledger, changed_assignments)
