from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from src.entailment_paired_regression import (
    PAIRED_BOOTSTRAP_METHOD_VERSION,
    PAIRED_COMPARISON_SCHEMA_VERSION,
    PairedMetricComparison,
    PairedRegressionComparison,
)
from src.entailment_release_manifest import (
    CandidateArtifact,
    CandidateReleaseManifest,
    ReleaseManifestError,
    dump_candidate_release_manifest,
    load_candidate_release_manifest,
    sign_candidate_release_manifest,
    verify_candidate_release_manifest,
)
from src.entailment_slice_regression import (
    SLICE_BOOTSTRAP_METHOD_VERSION,
    SLICE_COMPARISON_SCHEMA_VERSION,
    PairedSliceRegressionComparison,
    SliceComparison,
    SliceMetricComparison,
)

KEY = b"release-manifest-test-key-material-32"
ZERO, ONE, TWO, THREE = (character * 64 for character in "0123")


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()


def sealed(value: object) -> object:
    document = asdict(value)
    document.pop("comparison_sha256")
    return replace(value, comparison_sha256=digest(document))


def global_comparison() -> PairedRegressionComparison:
    metric = PairedMetricComparison(
        metric="balanced_accuracy",
        baseline_value=0.9,
        candidate_value=0.91,
        point_delta=0.01,
        adjusted_delta_upper=0.02,
        max_drop=0.01,
        regressed=False,
    )
    return sealed(
        PairedRegressionComparison(
            PAIRED_COMPARISON_SCHEMA_VERSION,
            PAIRED_BOOTSTRAP_METHOD_VERSION,
            ZERO,
            ONE,
            TWO,
            THREE,
            0.95,
            "bonferroni",
            1,
            0.95,
            1000,
            ZERO,
            ONE,
            (metric,),
            True,
            (),
            "",
        )
    )


def slice_comparison() -> PairedSliceRegressionComparison:
    metric = SliceMetricComparison(
        metric="balanced_accuracy",
        baseline_value=0.9,
        candidate_value=0.91,
        point_delta=0.01,
        adjusted_delta_upper=0.02,
        max_drop=0.01,
        regressed=False,
    )
    reviewed_slice = SliceComparison("language", "en", 20, 10, 10, (metric,))
    return sealed(
        PairedSliceRegressionComparison(
            SLICE_COMPARISON_SCHEMA_VERSION,
            SLICE_BOOTSTRAP_METHOD_VERSION,
            ZERO,
            ONE,
            TWO,
            THREE,
            TWO,
            0.95,
            "bonferroni",
            1,
            0.95,
            1000,
            20_000,
            ZERO,
            ONE,
            (reviewed_slice,),
            True,
            (),
            "",
        )
    )


def candidate() -> CandidateArtifact:
    return CandidateArtifact(TWO, "a" * 40, THREE)


def manifest() -> CandidateReleaseManifest:
    return sign_candidate_release_manifest(
        global_comparison(),
        slice_comparison(),
        candidate(),
        release_id="candidate-2026-10-30",
        key_id="release-key-2026-10",
        key=KEY,
        issued_at_epoch=100,
        expires_at_epoch=200,
    )


def test_round_trip_binds_exact_candidate_and_release_evidence() -> None:
    signed = manifest()
    loaded = load_candidate_release_manifest(dump_candidate_release_manifest(signed))

    verify_candidate_release_manifest(
        loaded,
        global_comparison(),
        slice_comparison(),
        candidate(),
        keys={"release-key-2026-10": KEY},
        current_epoch=150,
    )
    assert loaded == signed
    assert len(loaded.manifest_sha256) == 64


def test_signing_rejects_failed_or_mismatched_gates() -> None:
    failed = replace(
        global_comparison(), meets_policy=False, failure_reasons=("regression",)
    )
    mismatched = replace(slice_comparison(), candidate_ledger_sha256=TWO)

    with pytest.raises(ReleaseManifestError, match="global comparison is not accepted"):
        sign_candidate_release_manifest(
            failed,
            slice_comparison(),
            candidate(),
            release_id="candidate-1",
            key_id="key-1",
            key=KEY,
            issued_at_epoch=100,
            expires_at_epoch=200,
        )
    with pytest.raises(ReleaseManifestError, match="different ledgers"):
        sign_candidate_release_manifest(
            global_comparison(),
            mismatched,
            candidate(),
            release_id="candidate-1",
            key_id="key-1",
            key=KEY,
            issued_at_epoch=100,
            expires_at_epoch=200,
        )


def test_verifier_rejects_candidate_and_comparison_substitution() -> None:
    signed = manifest()

    with pytest.raises(ReleaseManifestError, match="does not bind"):
        verify_candidate_release_manifest(
            signed,
            global_comparison(),
            slice_comparison(),
            replace(candidate(), artifact_sha256=ZERO),
            keys={"release-key-2026-10": KEY},
            current_epoch=150,
        )
    with pytest.raises(ReleaseManifestError, match="comparison digest mismatch"):
        verify_candidate_release_manifest(
            signed,
            replace(global_comparison(), candidate_report_sha256=TWO),
            slice_comparison(),
            candidate(),
            keys={"release-key-2026-10": KEY},
            current_epoch=150,
        )


def test_verifier_rejects_tampering_unknown_keys_and_invalid_time() -> None:
    signed = manifest()
    tampered = replace(signed, release_id="candidate-substituted")

    with pytest.raises(ReleaseManifestError, match="signature mismatch"):
        verify_candidate_release_manifest(
            tampered,
            global_comparison(),
            slice_comparison(),
            candidate(),
            keys={"release-key-2026-10": KEY},
            current_epoch=150,
        )
    with pytest.raises(ReleaseManifestError, match="key is unavailable"):
        verify_candidate_release_manifest(
            signed,
            global_comparison(),
            slice_comparison(),
            candidate(),
            keys={},
            current_epoch=150,
        )
    for current_epoch in (99, 201):
        with pytest.raises(ReleaseManifestError, match="validity window"):
            verify_candidate_release_manifest(
                signed,
                global_comparison(),
                slice_comparison(),
                candidate(),
                keys={"release-key-2026-10": KEY},
                current_epoch=current_epoch,
            )


def test_loader_rejects_duplicate_extra_and_noncanonical_fields() -> None:
    payload = dump_candidate_release_manifest(manifest())
    duplicate = payload[:-1] + b',"purpose":"rag-candidate-release"}'
    document = json.loads(payload)
    document["extra"] = True

    with pytest.raises(ReleaseManifestError, match="duplicate field"):
        load_candidate_release_manifest(duplicate)
    with pytest.raises(ReleaseManifestError, match="fields do not match"):
        load_candidate_release_manifest(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
    with pytest.raises(ReleaseManifestError, match="not canonical"):
        load_candidate_release_manifest(json.dumps(json.loads(payload)).encode("ascii"))
