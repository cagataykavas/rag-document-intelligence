from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from src.entailment_eval import (
    DRIFT_SLICE_VERSION,
    DriftSlicePolicy,
    EntailmentDiagnosticReport,
    EntailmentPair,
    EntailmentSliceMetadata,
)
from src.entailment_paired_regression import (
    EvaluationScoreLedger,
    load_evaluation_score_ledger,
)

SLICE_ASSIGNMENT_SCHEMA_VERSION = "entailment-evaluation-slice-assignment/v1"
_CASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{0,127}")
_SLICE_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+:-]{0,63}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DIMENSIONS = (
    "language",
    "document_domain",
    "claim_length_utf8",
    "evidence_length_utf8",
)
_ARTIFACT_FIELDS = {
    "schema_version",
    "report_schema_version",
    "report_sha256",
    "evaluation_sha256",
    "drift_slice_version",
    "drift_slice_policy_sha256",
    "combined_assignment_sha256",
    "evaluation_assignment_sha256",
    "rows",
    "artifact_sha256",
}
_ROW_FIELDS = {
    "case_id_sha256",
    "content_sha256",
    *_DIMENSIONS,
}
_MAX_CASES = 100_000


class SliceAssignmentError(ValueError):
    """An evaluation slice assignment is unsafe or incompatible."""

    code = "entailment_slice_assignment_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class EvaluationSliceAssignmentRow:
    case_id_sha256: str
    content_sha256: str
    language: str
    document_domain: str
    claim_length_utf8: str
    evidence_length_utf8: str


@dataclass(frozen=True, slots=True)
class EvaluationSliceAssignmentArtifact:
    schema_version: str
    report_schema_version: str
    report_sha256: str
    evaluation_sha256: str
    drift_slice_version: str
    drift_slice_policy_sha256: str
    combined_assignment_sha256: str
    evaluation_assignment_sha256: str
    rows: tuple[EvaluationSliceAssignmentRow, ...]
    artifact_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSliceScoreRow:
    case_id_sha256: str
    content_sha256: str
    label: str
    score: float
    language: str
    document_domain: str
    claim_length_utf8: str
    evidence_length_utf8: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _content_sha256(pair: EntailmentPair) -> str:
    payload = json.dumps(
        {"claim": pair.claim, "evidence": pair.evidence},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SliceAssignmentError(f"{name} must be canonical lowercase SHA-256")
    return value


def _validate_report_digest(report: EntailmentDiagnosticReport) -> None:
    _validate_digest(report.report_sha256, "report digest")
    document = {
        key: value
        for key, value in report.to_audit_dict().items()
        if key != "report_sha256"
    }
    if report.report_sha256 != _sha256(document):
        raise SliceAssignmentError("report digest mismatch")


def _length_bucket(length: int, boundaries: tuple[int, ...]) -> str:
    lower = 0
    for upper in boundaries:
        if length <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f"{lower}+"


def _slice_values(
    pair: EntailmentPair,
    metadata: EntailmentSliceMetadata,
    policy: DriftSlicePolicy,
) -> dict[str, str]:
    return {
        "language": metadata.language,
        "document_domain": metadata.document_domain,
        "claim_length_utf8": _length_bucket(
            len(pair.claim.encode("utf-8")), policy.claim_length_utf8_boundaries
        ),
        "evidence_length_utf8": _length_bucket(
            len(pair.evidence.encode("utf-8")), policy.evidence_length_utf8_boundaries
        ),
    }


def _validate_pairs(
    pairs: Sequence[EntailmentPair], split: str
) -> tuple[EntailmentPair, ...]:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise TypeError(f"{split}_pairs must be a sequence of EntailmentPair values")
    if not 2 <= len(pairs) <= _MAX_CASES:
        raise SliceAssignmentError(f"{split} size is outside [2, 100000]")
    normalized: list[EntailmentPair] = []
    case_ids: set[str] = set()
    content_digests: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, EntailmentPair):
            raise TypeError(f"{split}_pairs contains an unsupported value")
        if _CASE_ID_RE.fullmatch(pair.case_id) is None:
            raise SliceAssignmentError(f"{split} case_id is malformed")
        content_sha256 = _content_sha256(pair)
        if pair.case_id in case_ids or content_sha256 in content_digests:
            raise SliceAssignmentError(f"{split} contains duplicate cases")
        case_ids.add(pair.case_id)
        content_digests.add(content_sha256)
        normalized.append(pair)
    return tuple(normalized)


def _validate_slice_counts(
    report: EntailmentDiagnosticReport,
    rows: Sequence[dict[str, str]],
) -> None:
    expected: dict[tuple[str, str], tuple[int, int]] = {}
    for item in report.drift_slices:
        key = (item.dimension, item.value)
        if key in expected or item.dimension not in _DIMENSIONS:
            raise SliceAssignmentError("report slice inventory is malformed")
        expected[key] = (item.calibration_pairs, item.evaluation_pairs)
    observed: dict[tuple[str, str], tuple[int, int]] = {}
    for dimension in _DIMENSIONS:
        calibration = Counter(
            row[dimension] for row in rows if row["split"] == "calibration"
        )
        evaluation = Counter(
            row[dimension] for row in rows if row["split"] == "evaluation"
        )
        for value in calibration.keys() | evaluation.keys():
            observed[(dimension, value)] = (
                calibration[value],
                evaluation[value],
            )
    if observed != expected:
        raise SliceAssignmentError("report slice counts do not match assignments")


def create_evaluation_slice_assignment(
    report: EntailmentDiagnosticReport,
    calibration_pairs: Sequence[EntailmentPair],
    evaluation_pairs: Sequence[EntailmentPair],
    metadata: Mapping[str, EntailmentSliceMetadata],
    policy: DriftSlicePolicy,
) -> EvaluationSliceAssignmentArtifact:
    """Seal evaluation-only assignments after verifying the report commitment."""

    if not isinstance(report, EntailmentDiagnosticReport):
        raise TypeError("report must be an EntailmentDiagnosticReport")
    if not isinstance(policy, DriftSlicePolicy):
        raise TypeError("policy must be a DriftSlicePolicy")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    calibration = _validate_pairs(calibration_pairs, "calibration")
    evaluation = _validate_pairs(evaluation_pairs, "evaluation")
    calibration_ids = {pair.case_id for pair in calibration}
    evaluation_ids = {pair.case_id for pair in evaluation}
    if calibration_ids & evaluation_ids:
        raise SliceAssignmentError("calibration and evaluation case_id overlap")
    calibration_content = {_content_sha256(pair) for pair in calibration}
    evaluation_content = {_content_sha256(pair) for pair in evaluation}
    if calibration_content & evaluation_content:
        raise SliceAssignmentError("calibration and evaluation content overlap")
    expected_ids = calibration_ids | evaluation_ids
    if set(metadata) != expected_ids:
        raise SliceAssignmentError("metadata must exactly cover both split case IDs")
    if any(
        not isinstance(metadata[case_id], EntailmentSliceMetadata)
        for case_id in expected_ids
    ):
        raise TypeError("metadata contains an unsupported value")

    _validate_report_digest(report)
    if (
        report.drift_slice_version != DRIFT_SLICE_VERSION
        or report.drift_slice_policy_sha256 is None
        or report.slice_assignment_sha256 is None
        or not report.drift_slices
    ):
        raise SliceAssignmentError("report has no supported drift-slice commitment")
    if (
        not isinstance(report.schema_version, str)
        or _VERSION_RE.fullmatch(report.schema_version) is None
    ):
        raise SliceAssignmentError("report schema version is malformed")
    for value, name in (
        (report.evaluation_sha256, "evaluation digest"),
        (report.drift_slice_policy_sha256, "drift slice policy digest"),
        (report.slice_assignment_sha256, "combined assignment digest"),
    ):
        _validate_digest(value, name)
    if (
        report.calibration.pairs != len(calibration)
        or report.evaluation.pairs != len(evaluation)
    ):
        raise SliceAssignmentError("split sizes do not match report")
    policy_sha256 = _sha256(asdict(policy))
    if policy_sha256 != report.drift_slice_policy_sha256:
        raise SliceAssignmentError("drift slice policy does not match report")

    combined_rows: list[dict[str, str]] = []
    evaluation_rows: list[EvaluationSliceAssignmentRow] = []
    for split, pairs in (("calibration", calibration), ("evaluation", evaluation)):
        for pair in pairs:
            values = _slice_values(pair, metadata[pair.case_id], policy)
            case_id_sha256 = hashlib.sha256(pair.case_id.encode("utf-8")).hexdigest()
            combined_rows.append(
                {"case_id_sha256": case_id_sha256, "split": split, **values}
            )
            if split == "evaluation":
                evaluation_rows.append(
                    EvaluationSliceAssignmentRow(
                        case_id_sha256=case_id_sha256,
                        content_sha256=_content_sha256(pair),
                        **values,
                    )
                )
    for dimension in _DIMENSIONS:
        if len({row[dimension] for row in combined_rows}) > policy.max_values_per_dimension:
            raise SliceAssignmentError(f"{dimension} exceeds the configured category limit")
    combined_rows.sort(key=lambda row: (row["split"], row["case_id_sha256"]))
    combined_assignment_sha256 = _sha256(combined_rows)
    if combined_assignment_sha256 != report.slice_assignment_sha256:
        raise SliceAssignmentError("combined slice assignment does not match report")
    _validate_slice_counts(report, combined_rows)

    evaluation_rows.sort(key=lambda row: row.case_id_sha256)
    row_documents = [asdict(row) for row in evaluation_rows]
    evaluation_assignment_sha256 = _sha256(row_documents)
    document = {
        "schema_version": SLICE_ASSIGNMENT_SCHEMA_VERSION,
        "report_schema_version": report.schema_version,
        "report_sha256": report.report_sha256,
        "evaluation_sha256": report.evaluation_sha256,
        "drift_slice_version": report.drift_slice_version,
        "drift_slice_policy_sha256": policy_sha256,
        "combined_assignment_sha256": combined_assignment_sha256,
        "evaluation_assignment_sha256": evaluation_assignment_sha256,
        "rows": row_documents,
    }
    return EvaluationSliceAssignmentArtifact(
        schema_version=SLICE_ASSIGNMENT_SCHEMA_VERSION,
        report_schema_version=report.schema_version,
        report_sha256=report.report_sha256,
        evaluation_sha256=report.evaluation_sha256,
        drift_slice_version=report.drift_slice_version,
        drift_slice_policy_sha256=policy_sha256,
        combined_assignment_sha256=combined_assignment_sha256,
        evaluation_assignment_sha256=evaluation_assignment_sha256,
        rows=tuple(evaluation_rows),
        artifact_sha256=_sha256(document),
    )


def _validate_artifact(artifact: EvaluationSliceAssignmentArtifact) -> None:
    if artifact.schema_version != SLICE_ASSIGNMENT_SCHEMA_VERSION:
        raise SliceAssignmentError("slice assignment schema is unsupported")
    if (
        not isinstance(artifact.report_schema_version, str)
        or _VERSION_RE.fullmatch(artifact.report_schema_version) is None
        or artifact.drift_slice_version != DRIFT_SLICE_VERSION
        or not isinstance(artifact.rows, tuple)
        or not 2 <= len(artifact.rows) <= _MAX_CASES
    ):
        raise SliceAssignmentError("slice assignment metadata is malformed")
    for value, name in (
        (artifact.report_sha256, "report digest"),
        (artifact.evaluation_sha256, "evaluation digest"),
        (artifact.drift_slice_policy_sha256, "drift slice policy digest"),
        (artifact.combined_assignment_sha256, "combined assignment digest"),
        (artifact.evaluation_assignment_sha256, "evaluation assignment digest"),
        (artifact.artifact_sha256, "artifact digest"),
    ):
        _validate_digest(value, name)
    case_ids: set[str] = set()
    content_digests: set[str] = set()
    for row in artifact.rows:
        if not isinstance(row, EvaluationSliceAssignmentRow):
            raise SliceAssignmentError("slice assignment rows are malformed")
        _validate_digest(row.case_id_sha256, "case commitment")
        _validate_digest(row.content_sha256, "content commitment")
        if row.case_id_sha256 in case_ids or row.content_sha256 in content_digests:
            raise SliceAssignmentError("slice assignment rows are duplicated")
        case_ids.add(row.case_id_sha256)
        content_digests.add(row.content_sha256)
        if any(
            _SLICE_VALUE_RE.fullmatch(getattr(row, dimension)) is None
            for dimension in _DIMENSIONS
        ):
            raise SliceAssignmentError("slice assignment value is malformed")
    if tuple(sorted(artifact.rows, key=lambda row: row.case_id_sha256)) != artifact.rows:
        raise SliceAssignmentError("slice assignment rows are not canonical")
    row_documents = [asdict(row) for row in artifact.rows]
    if artifact.evaluation_assignment_sha256 != _sha256(row_documents):
        raise SliceAssignmentError("evaluation assignment digest mismatch")
    expected = _sha256(
        {
            key: value
            for key, value in artifact.to_audit_dict().items()
            if key != "artifact_sha256"
        }
    )
    if artifact.artifact_sha256 != expected:
        raise SliceAssignmentError("slice assignment artifact digest mismatch")


def load_evaluation_slice_assignment(
    document: Mapping[str, object],
) -> EvaluationSliceAssignmentArtifact:
    """Load and integrity-check a slice assignment decoded from canonical JSON."""

    if not isinstance(document, Mapping) or set(document) != _ARTIFACT_FIELDS:
        raise SliceAssignmentError("slice assignment artifact is malformed")
    raw_rows = document.get("rows")
    if not isinstance(raw_rows, list) or not 2 <= len(raw_rows) <= _MAX_CASES:
        raise SliceAssignmentError("slice assignment artifact is malformed")
    rows: list[EvaluationSliceAssignmentRow] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != _ROW_FIELDS:
            raise SliceAssignmentError("slice assignment artifact is malformed")
        try:
            rows.append(EvaluationSliceAssignmentRow(**raw))
        except TypeError as exc:
            raise SliceAssignmentError("slice assignment artifact is malformed") from exc
    try:
        artifact = EvaluationSliceAssignmentArtifact(
            schema_version=document["schema_version"],
            report_schema_version=document["report_schema_version"],
            report_sha256=document["report_sha256"],
            evaluation_sha256=document["evaluation_sha256"],
            drift_slice_version=document["drift_slice_version"],
            drift_slice_policy_sha256=document["drift_slice_policy_sha256"],
            combined_assignment_sha256=document["combined_assignment_sha256"],
            evaluation_assignment_sha256=document["evaluation_assignment_sha256"],
            rows=tuple(rows),
            artifact_sha256=document["artifact_sha256"],
        )
    except TypeError as exc:
        raise SliceAssignmentError("slice assignment artifact is malformed") from exc
    _validate_artifact(artifact)
    return artifact


def join_evaluation_slice_assignments(
    score_ledger: EvaluationScoreLedger,
    assignments: EvaluationSliceAssignmentArtifact,
) -> tuple[EvaluationSliceScoreRow, ...]:
    """Join verified assignments to a score ledger for future paired slice tests."""

    if not isinstance(score_ledger, EvaluationScoreLedger):
        raise TypeError("score_ledger must be an EvaluationScoreLedger")
    if not isinstance(assignments, EvaluationSliceAssignmentArtifact):
        raise TypeError("assignments must be an EvaluationSliceAssignmentArtifact")
    ledger_document = score_ledger.to_audit_dict()
    ledger_document["rows"] = list(ledger_document["rows"])
    score_ledger = load_evaluation_score_ledger(ledger_document)
    _validate_artifact(assignments)
    if (
        score_ledger.report_schema_version != assignments.report_schema_version
        or score_ledger.report_sha256 != assignments.report_sha256
        or score_ledger.evaluation_sha256 != assignments.evaluation_sha256
    ):
        raise SliceAssignmentError("score ledger and slice assignment are incompatible")
    score_population = tuple(
        (row.case_id_sha256, row.content_sha256) for row in score_ledger.rows
    )
    assignment_population = tuple(
        (row.case_id_sha256, row.content_sha256) for row in assignments.rows
    )
    if score_population != assignment_population:
        raise SliceAssignmentError("score ledger and slice assignment populations differ")
    return tuple(
        EvaluationSliceScoreRow(
            case_id_sha256=score.case_id_sha256,
            content_sha256=score.content_sha256,
            label=score.label,
            score=score.score,
            language=assignment.language,
            document_domain=assignment.document_domain,
            claim_length_utf8=assignment.claim_length_utf8,
            evidence_length_utf8=assignment.evidence_length_utf8,
        )
        for score, assignment in zip(score_ledger.rows, assignments.rows, strict=True)
    )
