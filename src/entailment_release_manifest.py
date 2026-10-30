from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from src.entailment_paired_regression import (
    PAIRED_COMPARISON_SCHEMA_VERSION,
    PairedRegressionComparison,
)
from src.entailment_slice_regression import (
    SLICE_COMPARISON_SCHEMA_VERSION,
    PairedSliceRegressionComparison,
)

RELEASE_MANIFEST_SCHEMA_VERSION = "entailment-candidate-release-manifest/v1"
RELEASE_MANIFEST_PURPOSE = "rag-candidate-release"
RELEASE_MANIFEST_ALGORITHM = "hmac-sha256"
_DOMAIN = b"rag-document-intelligence/release-manifest/v1\x00"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_FIELDS = {
    "schema_version",
    "purpose",
    "release_id",
    "artifact_sha256",
    "source_commit_sha",
    "build_recipe_sha256",
    "global_comparison_sha256",
    "slice_comparison_sha256",
    "baseline_ledger_sha256",
    "candidate_ledger_sha256",
    "candidate_report_sha256",
    "evaluation_assignment_sha256",
    "issued_at_epoch",
    "expires_at_epoch",
    "key_id",
    "algorithm",
    "signature",
}
_SIGNED_FIELDS = _FIELDS - {"signature"}
_MINIMUM_KEY_BYTES = 32


class ReleaseManifestError(ValueError):
    """A candidate release manifest is malformed, stale, or unauthenticated."""

    code = "candidate_release_manifest_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ReleaseManifestError("value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ReleaseManifestError(f"{name} must be canonical lowercase SHA-256")
    return value


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise ReleaseManifestError(f"{name} is not a canonical identifier")
    return value


def _positive_epoch(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ReleaseManifestError(f"{name} must be a positive integer")
    return value


def _key(value: object) -> bytes:
    if not isinstance(value, bytes) or len(value) < _MINIMUM_KEY_BYTES:
        raise ReleaseManifestError("signing key must contain at least 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    artifact_sha256: str
    source_commit_sha: str
    build_recipe_sha256: str

    def __post_init__(self) -> None:
        _digest(self.artifact_sha256, "artifact_sha256")
        if (
            not isinstance(self.source_commit_sha, str)
            or _GIT_SHA_RE.fullmatch(self.source_commit_sha) is None
        ):
            raise ReleaseManifestError(
                "source_commit_sha must be a lowercase 40-character Git SHA"
            )
        _digest(self.build_recipe_sha256, "build_recipe_sha256")


@dataclass(frozen=True, slots=True)
class CandidateReleaseManifest:
    schema_version: str
    purpose: str
    release_id: str
    artifact_sha256: str
    source_commit_sha: str
    build_recipe_sha256: str
    global_comparison_sha256: str
    slice_comparison_sha256: str
    baseline_ledger_sha256: str
    candidate_ledger_sha256: str
    candidate_report_sha256: str
    evaluation_assignment_sha256: str
    issued_at_epoch: int
    expires_at_epoch: int
    key_id: str
    algorithm: str
    signature: str

    @property
    def manifest_sha256(self) -> str:
        return _sha256(asdict(self))

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


def _comparison_digest(comparison: object) -> str:
    document = asdict(comparison)
    claimed = document.pop("comparison_sha256")
    expected = _sha256(document)
    if not isinstance(claimed, str) or not hmac.compare_digest(claimed, expected):
        raise ReleaseManifestError("comparison digest mismatch")
    return claimed


def _validated_inputs(
    global_comparison: PairedRegressionComparison,
    slice_comparison: PairedSliceRegressionComparison,
    candidate: CandidateArtifact,
) -> tuple[str, str]:
    if not isinstance(global_comparison, PairedRegressionComparison):
        raise TypeError("global_comparison must be a PairedRegressionComparison")
    if not isinstance(slice_comparison, PairedSliceRegressionComparison):
        raise TypeError("slice_comparison must be a PairedSliceRegressionComparison")
    if not isinstance(candidate, CandidateArtifact):
        raise TypeError("candidate must be a CandidateArtifact")
    if not global_comparison.meets_policy or global_comparison.failure_reasons:
        raise ReleaseManifestError("global comparison is not accepted")
    if not slice_comparison.meets_policy or slice_comparison.failure_reasons:
        raise ReleaseManifestError("slice comparison is not accepted")
    if global_comparison.schema_version != PAIRED_COMPARISON_SCHEMA_VERSION:
        raise ReleaseManifestError("unsupported global comparison schema")
    if slice_comparison.schema_version != SLICE_COMPARISON_SCHEMA_VERSION:
        raise ReleaseManifestError("unsupported slice comparison schema")
    if not global_comparison.metrics:
        raise ReleaseManifestError("global comparison contains no metrics")
    if not slice_comparison.slices:
        raise ReleaseManifestError("slice comparison contains no reviewed slices")
    if any(metric.regressed for metric in global_comparison.metrics):
        raise ReleaseManifestError("global comparison contradicts its accepted result")
    if any(
        metric.regressed for item in slice_comparison.slices for metric in item.metrics
    ):
        raise ReleaseManifestError("slice comparison contradicts its accepted result")
    for name, value in (
        ("baseline_ledger_sha256", global_comparison.baseline_ledger_sha256),
        ("candidate_ledger_sha256", global_comparison.candidate_ledger_sha256),
        ("candidate_report_sha256", global_comparison.candidate_report_sha256),
        (
            "evaluation_assignment_sha256",
            slice_comparison.evaluation_assignment_sha256,
        ),
    ):
        _digest(value, name)
    if (
        global_comparison.baseline_ledger_sha256
        != slice_comparison.baseline_ledger_sha256
        or global_comparison.candidate_ledger_sha256
        != slice_comparison.candidate_ledger_sha256
    ):
        raise ReleaseManifestError(
            "global and slice comparisons bind different ledgers"
        )
    return _comparison_digest(global_comparison), _comparison_digest(slice_comparison)


def _signed_document(manifest: CandidateReleaseManifest) -> dict[str, object]:
    return {
        key: value
        for key, value in manifest.to_audit_dict().items()
        if key in _SIGNED_FIELDS
    }


def _signature(document: Mapping[str, object], key: bytes) -> str:
    return hmac.new(
        _key(key), _DOMAIN + _canonical_json(document), hashlib.sha256
    ).hexdigest()


def sign_candidate_release_manifest(
    global_comparison: PairedRegressionComparison,
    slice_comparison: PairedSliceRegressionComparison,
    candidate: CandidateArtifact,
    *,
    release_id: str,
    key_id: str,
    key: bytes,
    issued_at_epoch: int,
    expires_at_epoch: int,
) -> CandidateReleaseManifest:
    """Authenticate one accepted candidate and its exact release evidence."""

    global_digest, slice_digest = _validated_inputs(
        global_comparison, slice_comparison, candidate
    )
    issued = _positive_epoch(issued_at_epoch, "issued_at_epoch")
    expires = _positive_epoch(expires_at_epoch, "expires_at_epoch")
    if expires <= issued:
        raise ReleaseManifestError("expires_at_epoch must be after issued_at_epoch")
    unsigned = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "purpose": RELEASE_MANIFEST_PURPOSE,
        "release_id": _identifier(release_id, "release_id"),
        "artifact_sha256": candidate.artifact_sha256,
        "source_commit_sha": candidate.source_commit_sha,
        "build_recipe_sha256": candidate.build_recipe_sha256,
        "global_comparison_sha256": global_digest,
        "slice_comparison_sha256": slice_digest,
        "baseline_ledger_sha256": global_comparison.baseline_ledger_sha256,
        "candidate_ledger_sha256": global_comparison.candidate_ledger_sha256,
        "candidate_report_sha256": global_comparison.candidate_report_sha256,
        "evaluation_assignment_sha256": slice_comparison.evaluation_assignment_sha256,
        "issued_at_epoch": issued,
        "expires_at_epoch": expires,
        "key_id": _identifier(key_id, "key_id"),
        "algorithm": RELEASE_MANIFEST_ALGORITHM,
    }
    return CandidateReleaseManifest(**unsigned, signature=_signature(unsigned, key))


def verify_candidate_release_manifest(
    manifest: CandidateReleaseManifest,
    global_comparison: PairedRegressionComparison,
    slice_comparison: PairedSliceRegressionComparison,
    candidate: CandidateArtifact,
    *,
    keys: Mapping[str, bytes],
    current_epoch: int,
) -> None:
    """Fail closed unless the manifest authenticates the supplied release evidence."""

    if not isinstance(manifest, CandidateReleaseManifest):
        raise TypeError("manifest must be a CandidateReleaseManifest")
    global_digest, slice_digest = _validated_inputs(
        global_comparison, slice_comparison, candidate
    )
    if manifest.schema_version != RELEASE_MANIFEST_SCHEMA_VERSION:
        raise ReleaseManifestError("unsupported manifest schema")
    if manifest.purpose != RELEASE_MANIFEST_PURPOSE:
        raise ReleaseManifestError("manifest purpose mismatch")
    if manifest.algorithm != RELEASE_MANIFEST_ALGORITHM:
        raise ReleaseManifestError("unsupported signature algorithm")
    _identifier(manifest.release_id, "release_id")
    _identifier(manifest.key_id, "key_id")
    _digest(manifest.signature, "signature")
    now = _positive_epoch(current_epoch, "current_epoch")
    issued = _positive_epoch(manifest.issued_at_epoch, "issued_at_epoch")
    expires = _positive_epoch(manifest.expires_at_epoch, "expires_at_epoch")
    if expires <= issued or now < issued or now > expires:
        raise ReleaseManifestError("manifest is outside its validity window")
    expected_bindings = {
        "artifact_sha256": candidate.artifact_sha256,
        "source_commit_sha": candidate.source_commit_sha,
        "build_recipe_sha256": candidate.build_recipe_sha256,
        "global_comparison_sha256": global_digest,
        "slice_comparison_sha256": slice_digest,
        "baseline_ledger_sha256": global_comparison.baseline_ledger_sha256,
        "candidate_ledger_sha256": global_comparison.candidate_ledger_sha256,
        "candidate_report_sha256": global_comparison.candidate_report_sha256,
        "evaluation_assignment_sha256": slice_comparison.evaluation_assignment_sha256,
    }
    document = _signed_document(manifest)
    if any(document[name] != value for name, value in expected_bindings.items()):
        raise ReleaseManifestError("manifest does not bind the supplied evidence")
    if not isinstance(keys, Mapping) or manifest.key_id not in keys:
        raise ReleaseManifestError("signing key is unavailable")
    expected_signature = _signature(document, keys[manifest.key_id])
    if not hmac.compare_digest(manifest.signature, expected_signature):
        raise ReleaseManifestError("manifest signature mismatch")


def dump_candidate_release_manifest(manifest: CandidateReleaseManifest) -> bytes:
    if not isinstance(manifest, CandidateReleaseManifest):
        raise TypeError("manifest must be a CandidateReleaseManifest")
    return _canonical_json(manifest.to_audit_dict())


def load_candidate_release_manifest(payload: bytes) -> CandidateReleaseManifest:
    """Load only the exact canonical JSON representation without duplicate keys."""

    if not isinstance(payload, bytes) or not payload:
        raise ReleaseManifestError("manifest payload must be non-empty bytes")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseManifestError("manifest contains a duplicate field")
            result[key] = value
        return result

    try:
        document = json.loads(payload, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReleaseManifestError("manifest is not valid JSON") from exc
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise ReleaseManifestError("manifest fields do not match the schema")
    if _canonical_json(document) != payload:
        raise ReleaseManifestError("manifest JSON is not canonical")
    try:
        return CandidateReleaseManifest(**document)
    except TypeError as exc:
        raise ReleaseManifestError("manifest fields do not match the schema") from exc
