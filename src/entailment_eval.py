from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

SCHEMA_VERSION = "entailment-diagnostic/v4"
LEAKAGE_GUARD_VERSION = "unicode-nfkc-casefold-tr-dot-char5-jaccard/v1"
SEMANTIC_LEAKAGE_GUARD_VERSION = "versioned-pairwise-semantic-similarity/v1"
DRIFT_SLICE_VERSION = "privacy-reduced-language-domain-length/v1"
_CASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SCORER_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
_SLICE_VALUE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}")
_NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)
_SHINGLE_SIZE = 5

EntailmentScorer = Callable[[str, str], float]
SemanticLeakageScorer = Callable[[str, str, str, str], float]


class EntailmentEvaluationError(ValueError):
    """The diagnostic could not produce a safe, reproducible result."""

    code = "entailment_evaluation_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


class EntailmentLabel(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class EntailmentPair:
    case_id: str
    claim: str
    evidence: str
    label: EntailmentLabel


@dataclass(frozen=True, slots=True)
class EntailmentSliceMetadata:
    """Low-cardinality, non-sensitive categories used for drift slicing."""

    language: str
    document_domain: str

    def __post_init__(self) -> None:
        for name in ("language", "document_domain"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SLICE_VALUE_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must contain 1-32 safe characters")


@dataclass(frozen=True, slots=True)
class DriftSlicePolicy:
    """Privacy and statistical floors for optional drift diagnostics."""

    min_pairs_per_split: int = 20
    min_pairs_per_label_per_split: int = 5
    claim_length_utf8_boundaries: tuple[int, ...] = (64, 256)
    evidence_length_utf8_boundaries: tuple[int, ...] = (256, 1_024)
    max_values_per_dimension: int = 32

    def __post_init__(self) -> None:
        if (
            type(self.min_pairs_per_split) is not int
            or not 2 <= self.min_pairs_per_split <= 100_000
        ):
            raise ValueError("min_pairs_per_split must be an integer in [2, 100000]")
        if (
            type(self.min_pairs_per_label_per_split) is not int
            or not 1 <= self.min_pairs_per_label_per_split <= 50_000
        ):
            raise ValueError(
                "min_pairs_per_label_per_split must be an integer in [1, 50000]"
            )
        if 2 * self.min_pairs_per_label_per_split > self.min_pairs_per_split:
            raise ValueError("min_pairs_per_split must cover both label floors")
        if (
            type(self.max_values_per_dimension) is not int
            or not 1 <= self.max_values_per_dimension <= 128
        ):
            raise ValueError("max_values_per_dimension must be an integer in [1, 128]")
        for name in (
            "claim_length_utf8_boundaries",
            "evidence_length_utf8_boundaries",
        ):
            boundaries = getattr(self, name)
            if (
                not isinstance(boundaries, tuple)
                or not 1 <= len(boundaries) <= 16
                or any(
                    type(value) is not int or not 1 <= value <= 4_194_304
                    for value in boundaries
                )
                or tuple(sorted(set(boundaries))) != boundaries
            ):
                raise ValueError(
                    f"{name} must contain 1-16 strictly increasing positive byte limits"
                )


@dataclass(frozen=True, slots=True)
class EntailmentPolicy:
    min_supported_precision: float = 0.9
    min_supported_recall: float = 0.8
    max_pairs_per_split: int = 10_000
    max_claim_utf8_bytes: int = 16_384
    max_evidence_utf8_bytes: int = 65_536
    max_split_utf8_bytes: int = 16 * 1024 * 1024
    near_duplicate_similarity_threshold: float = 0.9
    max_near_duplicate_comparisons: int = 1_000_000
    semantic_duplicate_similarity_threshold: float = 0.9
    max_semantic_duplicate_comparisons: int = 100_000

    def __post_init__(self) -> None:
        for name in ("min_supported_precision", "min_supported_recall"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and between 0 and 1")
        for name in (
            "near_duplicate_similarity_threshold",
            "semantic_duplicate_similarity_threshold",
        ):
            threshold = getattr(self, name)
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold)
                or not 0.0 < float(threshold) <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in (0, 1]")
        limits = {
            "max_pairs_per_split": (self.max_pairs_per_split, 2, 100_000),
            "max_claim_utf8_bytes": (self.max_claim_utf8_bytes, 1, 1_048_576),
            "max_evidence_utf8_bytes": (
                self.max_evidence_utf8_bytes,
                1,
                4 * 1024 * 1024,
            ),
            "max_split_utf8_bytes": (
                self.max_split_utf8_bytes,
                2,
                256 * 1024 * 1024,
            ),
            "max_near_duplicate_comparisons": (
                self.max_near_duplicate_comparisons,
                1,
                50_000_000,
            ),
            "max_semantic_duplicate_comparisons": (
                self.max_semantic_duplicate_comparisons,
                1,
                50_000_000,
            ),
        }
        for name, (value, minimum, maximum) in limits.items():
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


@dataclass(frozen=True, slots=True)
class EntailmentMetrics:
    pairs: int
    true_supported: int
    false_supported: int
    true_unsupported: int
    false_unsupported: int
    supported_precision: float
    supported_recall: float
    unsupported_recall: float
    balanced_accuracy: float


@dataclass(frozen=True, slots=True)
class EntailmentDriftSlice:
    dimension: str
    value: str
    calibration_pairs: int
    evaluation_pairs: int
    metrics_reported: bool
    calibration: EntailmentMetrics | None
    evaluation: EntailmentMetrics | None
    supported_precision_delta: float | None
    supported_recall_delta: float | None
    unsupported_recall_delta: float | None
    balanced_accuracy_delta: float | None


@dataclass(frozen=True, slots=True)
class EntailmentDiagnosticReport:
    schema_version: str
    leakage_guard_version: str
    scorer_id: str
    threshold: float
    calibration: EntailmentMetrics
    evaluation: EntailmentMetrics
    calibration_sha256: str
    evaluation_sha256: str
    policy_sha256: str
    report_sha256: str
    meets_policy: bool
    failure_reasons: tuple[str, ...]
    near_duplicate_comparisons: int
    semantic_leakage_guard_version: str | None
    semantic_scorer_id: str | None
    semantic_model_revision_sha256: str | None
    semantic_duplicate_comparisons: int
    drift_slice_version: str | None
    drift_slice_policy_sha256: str | None
    slice_assignment_sha256: str | None
    drift_slices: tuple[EntailmentDriftSlice, ...]

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ValidatedPair:
    pair: EntailmentPair
    content_sha256: str
    case_id_sha256: str
    claim_shingles: frozenset[str]
    evidence_shingles: frozenset[str]


@dataclass(frozen=True, slots=True)
class _ScoredPair:
    validated: _ValidatedPair
    score: float


@dataclass(frozen=True, slots=True)
class VersionedSemanticLeakageGuard:
    """Bound provenance for an untrusted multilingual similarity scorer."""

    scorer: SemanticLeakageScorer
    scorer_id: str
    model_revision_sha256: str

    def __post_init__(self) -> None:
        if not callable(self.scorer):
            raise TypeError("semantic leakage scorer must be callable")
        if (
            not isinstance(self.scorer_id, str)
            or _SCORER_ID_RE.fullmatch(self.scorer_id) is None
        ):
            raise ValueError("semantic scorer_id must contain 1-128 safe characters")
        if (
            not isinstance(self.model_revision_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.model_revision_sha256) is None
        ):
            raise ValueError(
                "model_revision_sha256 must be canonical lowercase SHA-256"
            )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _similarity_shingles(text: str) -> frozenset[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = unicodedata.normalize("NFKC", normalized).replace("\u0307", "")
    normalized = _NON_WORD_RE.sub(" ", normalized).strip()
    if len(normalized) <= _SHINGLE_SIZE:
        return frozenset({normalized})
    return frozenset(
        normalized[index : index + _SHINGLE_SIZE]
        for index in range(len(normalized) - _SHINGLE_SIZE + 1)
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right)


@dataclass(slots=True)
class _SimilarityBudget:
    maximum: int
    used: int = 0

    def consume(self) -> None:
        self.used += 1
        if self.used > self.maximum:
            raise EntailmentEvaluationError("near-duplicate comparison budget exceeded")


def _is_near_duplicate(
    left: _ValidatedPair,
    right: _ValidatedPair,
    *,
    threshold: float,
) -> bool:
    return (
        _jaccard(left.claim_shingles, right.claim_shingles) >= threshold
        and _jaccard(left.evidence_shingles, right.evidence_shingles) >= threshold
    )


def _candidate_indexes(
    shingles: frozenset[str],
    index: dict[str, set[int]],
) -> set[int]:
    candidates: set[int] = set()
    for shingle in shingles:
        candidates.update(index.get(shingle, ()))
    return candidates


def _index_pair(
    pair_index: int,
    shingles: frozenset[str],
    index: dict[str, set[int]],
) -> None:
    for shingle in shingles:
        index.setdefault(shingle, set()).add(pair_index)


def _reject_within_split_near_duplicates(
    pairs: tuple[_ValidatedPair, ...],
    *,
    split_name: str,
    threshold: float,
    budget: _SimilarityBudget,
) -> None:
    claim_index: dict[str, set[int]] = {}
    for pair_index, pair in enumerate(pairs):
        for candidate_index in sorted(
            _candidate_indexes(pair.claim_shingles, claim_index)
        ):
            budget.consume()
            if _is_near_duplicate(pair, pairs[candidate_index], threshold=threshold):
                raise EntailmentEvaluationError(
                    f"{split_name} contains near-duplicate content"
                )
        _index_pair(pair_index, pair.claim_shingles, claim_index)


def _reject_cross_split_near_duplicates(
    calibration: tuple[_ValidatedPair, ...],
    evaluation: tuple[_ValidatedPair, ...],
    *,
    threshold: float,
    budget: _SimilarityBudget,
) -> None:
    claim_index: dict[str, set[int]] = {}
    for pair_index, pair in enumerate(calibration):
        _index_pair(pair_index, pair.claim_shingles, claim_index)
    for pair in evaluation:
        for candidate_index in sorted(
            _candidate_indexes(pair.claim_shingles, claim_index)
        ):
            budget.consume()
            if _is_near_duplicate(
                pair,
                calibration[candidate_index],
                threshold=threshold,
            ):
                raise EntailmentEvaluationError(
                    "calibration and evaluation near-duplicate content overlap"
                )


def _semantic_similarity(
    left: _ValidatedPair,
    right: _ValidatedPair,
    guard: VersionedSemanticLeakageGuard,
    *,
    split_name: str,
) -> float:
    try:
        score = guard.scorer(
            left.pair.claim,
            left.pair.evidence,
            right.pair.claim,
            right.pair.evidence,
        )
    except Exception as exc:  # noqa: BLE001 - scorer is an untrusted evaluation boundary
        raise EntailmentEvaluationError(
            f"{split_name} semantic leakage scorer failed"
        ) from exc
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0.0 <= float(score) <= 1.0
    ):
        raise EntailmentEvaluationError(
            f"{split_name} semantic leakage scorer returned an invalid similarity"
        )
    return float(score)


def _reject_semantic_duplicates(
    calibration: tuple[_ValidatedPair, ...],
    evaluation: tuple[_ValidatedPair, ...],
    guard: VersionedSemanticLeakageGuard,
    *,
    threshold: float,
    budget: _SimilarityBudget,
) -> None:
    for split_name, pairs in (
        ("calibration split", calibration),
        ("evaluation split", evaluation),
    ):
        for left_index, left in enumerate(pairs):
            for right in pairs[left_index + 1 :]:
                budget.consume()
                if (
                    _semantic_similarity(left, right, guard, split_name=split_name)
                    >= threshold
                ):
                    raise EntailmentEvaluationError(
                        f"{split_name} contains semantic duplicate content"
                    )
    for calibration_pair in calibration:
        for evaluation_pair in evaluation:
            budget.consume()
            if (
                _semantic_similarity(
                    calibration_pair,
                    evaluation_pair,
                    guard,
                    split_name="cross-split",
                )
                >= threshold
            ):
                raise EntailmentEvaluationError(
                    "calibration and evaluation semantic duplicate content overlap"
                )


def _validate_split(
    pairs: Sequence[EntailmentPair],
    *,
    split_name: str,
    policy: EntailmentPolicy,
) -> tuple[_ValidatedPair, ...]:
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise TypeError(f"{split_name} must be a sequence of EntailmentPair values")
    if not 2 <= len(pairs) <= policy.max_pairs_per_split:
        raise EntailmentEvaluationError(
            f"{split_name} size must be in [2, {policy.max_pairs_per_split}]"
        )
    validated: list[_ValidatedPair] = []
    case_ids: set[str] = set()
    content_digests: set[str] = set()
    labels: set[EntailmentLabel] = set()
    total_bytes = 0
    for pair in pairs:
        if not isinstance(pair, EntailmentPair):
            raise TypeError(f"{split_name} contains an unsupported pair")
        if (
            not isinstance(pair.case_id, str)
            or _CASE_ID_RE.fullmatch(pair.case_id) is None
        ):
            raise EntailmentEvaluationError(f"{split_name} case_id is malformed")
        if pair.case_id in case_ids:
            raise EntailmentEvaluationError(
                f"{split_name} contains a duplicate case_id"
            )
        if not isinstance(pair.claim, str) or not isinstance(pair.evidence, str):
            raise EntailmentEvaluationError(f"{split_name} text must be UTF-8 strings")
        claim_bytes = pair.claim.encode("utf-8")
        evidence_bytes = pair.evidence.encode("utf-8")
        if not 0 < len(claim_bytes) <= policy.max_claim_utf8_bytes:
            raise EntailmentEvaluationError(
                f"{split_name} claim exceeds its byte boundary"
            )
        if not 0 < len(evidence_bytes) <= policy.max_evidence_utf8_bytes:
            raise EntailmentEvaluationError(
                f"{split_name} evidence exceeds its byte boundary"
            )
        if not isinstance(pair.label, EntailmentLabel):
            raise EntailmentEvaluationError(f"{split_name} label is unsupported")
        content_sha256 = _sha256(
            _canonical_json({"claim": pair.claim, "evidence": pair.evidence})
        )
        if content_sha256 in content_digests:
            raise EntailmentEvaluationError(f"{split_name} contains duplicate content")
        case_ids.add(pair.case_id)
        content_digests.add(content_sha256)
        labels.add(pair.label)
        total_bytes += len(claim_bytes) + len(evidence_bytes)
        if total_bytes > policy.max_split_utf8_bytes:
            raise EntailmentEvaluationError(
                f"{split_name} exceeds its total byte boundary"
            )
        validated.append(
            _ValidatedPair(
                pair=pair,
                content_sha256=content_sha256,
                case_id_sha256=_sha256(pair.case_id.encode("utf-8")),
                claim_shingles=_similarity_shingles(pair.claim),
                evidence_shingles=_similarity_shingles(pair.evidence),
            )
        )
    if labels != {EntailmentLabel.SUPPORTED, EntailmentLabel.UNSUPPORTED}:
        raise EntailmentEvaluationError(f"{split_name} must contain both labels")
    return tuple(validated)


def _score_split(
    pairs: tuple[_ValidatedPair, ...],
    scorer: EntailmentScorer,
    *,
    split_name: str,
) -> tuple[_ScoredPair, ...]:
    scored: list[_ScoredPair] = []
    for item in pairs:
        try:
            score = scorer(item.pair.claim, item.pair.evidence)
        except Exception as exc:  # noqa: BLE001 - scorer is an untrusted evaluation boundary
            raise EntailmentEvaluationError(f"{split_name} scorer failed") from exc
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0.0 <= float(score) <= 1.0
        ):
            raise EntailmentEvaluationError(
                f"{split_name} scorer returned an invalid probability"
            )
        scored.append(_ScoredPair(validated=item, score=float(score)))
    return tuple(scored)


def _metrics(pairs: tuple[_ScoredPair, ...], threshold: float) -> EntailmentMetrics:
    true_supported = false_supported = true_unsupported = false_unsupported = 0
    for item in pairs:
        predicted_supported = item.score >= threshold
        actually_supported = item.validated.pair.label is EntailmentLabel.SUPPORTED
        if predicted_supported and actually_supported:
            true_supported += 1
        elif predicted_supported:
            false_supported += 1
        elif actually_supported:
            false_unsupported += 1
        else:
            true_unsupported += 1
    predicted_positive = true_supported + false_supported
    actual_positive = true_supported + false_unsupported
    actual_negative = true_unsupported + false_supported
    precision = true_supported / predicted_positive if predicted_positive else 0.0
    recall = true_supported / actual_positive
    unsupported_recall = true_unsupported / actual_negative
    return EntailmentMetrics(
        pairs=len(pairs),
        true_supported=true_supported,
        false_supported=false_supported,
        true_unsupported=true_unsupported,
        false_unsupported=false_unsupported,
        supported_precision=precision,
        supported_recall=recall,
        unsupported_recall=unsupported_recall,
        balanced_accuracy=(recall + unsupported_recall) / 2,
    )


def _length_bucket(length: int, boundaries: tuple[int, ...]) -> str:
    lower = 0
    for upper in boundaries:
        if length <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f"{lower}+"


def _slice_values(
    item: _ValidatedPair,
    metadata: EntailmentSliceMetadata,
    policy: DriftSlicePolicy,
) -> dict[str, str]:
    return {
        "language": metadata.language,
        "document_domain": metadata.document_domain,
        "claim_length_utf8": _length_bucket(
            len(item.pair.claim.encode("utf-8")),
            policy.claim_length_utf8_boundaries,
        ),
        "evidence_length_utf8": _length_bucket(
            len(item.pair.evidence.encode("utf-8")),
            policy.evidence_length_utf8_boundaries,
        ),
    }


def _validate_slice_metadata(
    calibration: tuple[_ValidatedPair, ...],
    evaluation: tuple[_ValidatedPair, ...],
    metadata: Mapping[str, EntailmentSliceMetadata],
    policy: DriftSlicePolicy,
) -> tuple[dict[str, EntailmentSliceMetadata], str]:
    if not isinstance(metadata, Mapping):
        raise TypeError("drift_slice_metadata must be a mapping")
    expected_case_ids = {item.pair.case_id for item in (*calibration, *evaluation)}
    if set(metadata) != expected_case_ids:
        raise EntailmentEvaluationError(
            "drift slice metadata must exactly cover both split case IDs"
        )
    normalized: dict[str, EntailmentSliceMetadata] = {}
    for case_id in expected_case_ids:
        item = metadata[case_id]
        if not isinstance(item, EntailmentSliceMetadata):
            raise TypeError("drift slice metadata contains an unsupported value")
        normalized[case_id] = item

    rows: list[dict[str, str]] = []
    for split_name, pairs in (("calibration", calibration), ("evaluation", evaluation)):
        for item in pairs:
            values = _slice_values(item, normalized[item.pair.case_id], policy)
            rows.append(
                {
                    "case_id_sha256": item.case_id_sha256,
                    "split": split_name,
                    **values,
                }
            )
    for dimension in (
        "language",
        "document_domain",
        "claim_length_utf8",
        "evidence_length_utf8",
    ):
        if len({row[dimension] for row in rows}) > policy.max_values_per_dimension:
            raise EntailmentEvaluationError(
                f"{dimension} exceeds the configured category limit"
            )
    rows.sort(key=lambda row: (row["split"], row["case_id_sha256"]))
    return normalized, _sha256(_canonical_json(rows))


def _slice_has_metric_support(
    pairs: tuple[_ScoredPair, ...],
    policy: DriftSlicePolicy,
) -> bool:
    if len(pairs) < policy.min_pairs_per_split:
        return False
    supported = sum(
        item.validated.pair.label is EntailmentLabel.SUPPORTED for item in pairs
    )
    unsupported = len(pairs) - supported
    return min(supported, unsupported) >= policy.min_pairs_per_label_per_split


def _build_drift_slices(
    calibration: tuple[_ScoredPair, ...],
    evaluation: tuple[_ScoredPair, ...],
    metadata: Mapping[str, EntailmentSliceMetadata],
    policy: DriftSlicePolicy,
    threshold: float,
) -> tuple[EntailmentDriftSlice, ...]:
    dimensions = (
        "language",
        "document_domain",
        "claim_length_utf8",
        "evidence_length_utf8",
    )
    calibration_values = {
        item.validated.pair.case_id: _slice_values(
            item.validated,
            metadata[item.validated.pair.case_id],
            policy,
        )
        for item in calibration
    }
    evaluation_values = {
        item.validated.pair.case_id: _slice_values(
            item.validated,
            metadata[item.validated.pair.case_id],
            policy,
        )
        for item in evaluation
    }
    slices: list[EntailmentDriftSlice] = []
    for dimension in dimensions:
        values = sorted(
            {row[dimension] for row in calibration_values.values()}
            | {row[dimension] for row in evaluation_values.values()}
        )
        if len(values) > policy.max_values_per_dimension:
            raise EntailmentEvaluationError(
                f"{dimension} exceeds the configured category limit"
            )
        for value in values:
            calibration_slice = tuple(
                item
                for item in calibration
                if calibration_values[item.validated.pair.case_id][dimension] == value
            )
            evaluation_slice = tuple(
                item
                for item in evaluation
                if evaluation_values[item.validated.pair.case_id][dimension] == value
            )
            metrics_reported = _slice_has_metric_support(
                calibration_slice, policy
            ) and _slice_has_metric_support(evaluation_slice, policy)
            calibration_metrics = (
                _metrics(calibration_slice, threshold) if metrics_reported else None
            )
            evaluation_metrics = (
                _metrics(evaluation_slice, threshold) if metrics_reported else None
            )
            slices.append(
                EntailmentDriftSlice(
                    dimension=dimension,
                    value=value,
                    calibration_pairs=len(calibration_slice),
                    evaluation_pairs=len(evaluation_slice),
                    metrics_reported=metrics_reported,
                    calibration=calibration_metrics,
                    evaluation=evaluation_metrics,
                    supported_precision_delta=(
                        evaluation_metrics.supported_precision
                        - calibration_metrics.supported_precision
                        if metrics_reported
                        and evaluation_metrics is not None
                        and calibration_metrics is not None
                        else None
                    ),
                    supported_recall_delta=(
                        evaluation_metrics.supported_recall
                        - calibration_metrics.supported_recall
                        if metrics_reported
                        and evaluation_metrics is not None
                        and calibration_metrics is not None
                        else None
                    ),
                    unsupported_recall_delta=(
                        evaluation_metrics.unsupported_recall
                        - calibration_metrics.unsupported_recall
                        if metrics_reported
                        and evaluation_metrics is not None
                        and calibration_metrics is not None
                        else None
                    ),
                    balanced_accuracy_delta=(
                        evaluation_metrics.balanced_accuracy
                        - calibration_metrics.balanced_accuracy
                        if metrics_reported
                        and evaluation_metrics is not None
                        and calibration_metrics is not None
                        else None
                    ),
                )
            )
    return tuple(slices)


def _select_threshold(
    calibration: tuple[_ScoredPair, ...],
    policy: EntailmentPolicy,
) -> tuple[float, EntailmentMetrics]:
    candidates = sorted({item.score for item in calibration})
    results = [
        (threshold, _metrics(calibration, threshold)) for threshold in candidates
    ]
    precision_eligible = [
        result
        for result in results
        if result[1].supported_precision >= policy.min_supported_precision
    ]
    pool = precision_eligible or results
    return max(
        pool,
        key=lambda item: (
            item[1].supported_recall,
            item[1].supported_precision,
            item[1].balanced_accuracy,
            item[0],
        ),
    )


def _scored_split_sha256(pairs: tuple[_ScoredPair, ...]) -> str:
    rows = [
        {
            "case_id_sha256": item.validated.case_id_sha256,
            "content_sha256": item.validated.content_sha256,
            "label": item.validated.pair.label.value,
            "score": item.score,
        }
        for item in pairs
    ]
    return _sha256(_canonical_json(sorted(rows, key=lambda row: row["case_id_sha256"])))


def evaluate_entailment_diagnostic(
    calibration_pairs: Sequence[EntailmentPair],
    evaluation_pairs: Sequence[EntailmentPair],
    scorer: EntailmentScorer,
    *,
    scorer_id: str,
    policy: EntailmentPolicy = EntailmentPolicy(),
    semantic_leakage_guard: VersionedSemanticLeakageGuard | None = None,
    drift_slice_metadata: Mapping[str, EntailmentSliceMetadata] | None = None,
    drift_slice_policy: DriftSlicePolicy | None = None,
) -> EntailmentDiagnosticReport:
    """Calibrate on one labeled split and evaluate a disjoint holdout split.

    This probabilistic diagnostic never changes generation-admission authority.
    """

    if not callable(scorer):
        raise TypeError("scorer must be callable")
    if not isinstance(scorer_id, str) or _SCORER_ID_RE.fullmatch(scorer_id) is None:
        raise ValueError("scorer_id must contain 1-128 safe characters")
    if not isinstance(policy, EntailmentPolicy):
        raise TypeError("policy must be an EntailmentPolicy")
    if semantic_leakage_guard is not None and not isinstance(
        semantic_leakage_guard, VersionedSemanticLeakageGuard
    ):
        raise TypeError(
            "semantic_leakage_guard must be a VersionedSemanticLeakageGuard"
        )
    if (drift_slice_metadata is None) != (drift_slice_policy is None):
        raise ValueError(
            "drift_slice_metadata and drift_slice_policy must be provided together"
        )
    if drift_slice_policy is not None and not isinstance(
        drift_slice_policy, DriftSlicePolicy
    ):
        raise TypeError("drift_slice_policy must be a DriftSlicePolicy")
    calibration_validated = _validate_split(
        calibration_pairs,
        split_name="calibration split",
        policy=policy,
    )
    evaluation_validated = _validate_split(
        evaluation_pairs,
        split_name="evaluation split",
        policy=policy,
    )
    calibration_content = {item.content_sha256 for item in calibration_validated}
    evaluation_content = {item.content_sha256 for item in evaluation_validated}
    calibration_case_ids = {item.pair.case_id for item in calibration_validated}
    evaluation_case_ids = {item.pair.case_id for item in evaluation_validated}
    if calibration_case_ids & evaluation_case_ids:
        raise EntailmentEvaluationError("calibration and evaluation case_id overlap")
    if calibration_content & evaluation_content:
        raise EntailmentEvaluationError("calibration and evaluation content overlap")
    similarity_budget = _SimilarityBudget(policy.max_near_duplicate_comparisons)
    _reject_within_split_near_duplicates(
        calibration_validated,
        split_name="calibration split",
        threshold=policy.near_duplicate_similarity_threshold,
        budget=similarity_budget,
    )
    _reject_within_split_near_duplicates(
        evaluation_validated,
        split_name="evaluation split",
        threshold=policy.near_duplicate_similarity_threshold,
        budget=similarity_budget,
    )
    _reject_cross_split_near_duplicates(
        calibration_validated,
        evaluation_validated,
        threshold=policy.near_duplicate_similarity_threshold,
        budget=similarity_budget,
    )
    semantic_similarity_budget = _SimilarityBudget(
        policy.max_semantic_duplicate_comparisons
    )
    if semantic_leakage_guard is not None:
        _reject_semantic_duplicates(
            calibration_validated,
            evaluation_validated,
            semantic_leakage_guard,
            threshold=policy.semantic_duplicate_similarity_threshold,
            budget=semantic_similarity_budget,
        )
    normalized_slice_metadata: dict[str, EntailmentSliceMetadata] = {}
    slice_assignment_sha256: str | None = None
    if drift_slice_metadata is not None and drift_slice_policy is not None:
        normalized_slice_metadata, slice_assignment_sha256 = _validate_slice_metadata(
            calibration_validated,
            evaluation_validated,
            drift_slice_metadata,
            drift_slice_policy,
        )

    calibration_scored = _score_split(
        calibration_validated,
        scorer,
        split_name="calibration split",
    )
    evaluation_scored = _score_split(
        evaluation_validated,
        scorer,
        split_name="evaluation split",
    )
    threshold, calibration_metrics = _select_threshold(calibration_scored, policy)
    evaluation_metrics = _metrics(evaluation_scored, threshold)
    drift_slices: tuple[EntailmentDriftSlice, ...] = ()
    if drift_slice_policy is not None:
        drift_slices = _build_drift_slices(
            calibration_scored,
            evaluation_scored,
            normalized_slice_metadata,
            drift_slice_policy,
            threshold,
        )
    failure_reasons: list[str] = []
    for split_name, metrics in (
        ("calibration", calibration_metrics),
        ("evaluation", evaluation_metrics),
    ):
        if metrics.supported_precision < policy.min_supported_precision:
            failure_reasons.append(f"{split_name}_supported_precision_below_minimum")
        if metrics.supported_recall < policy.min_supported_recall:
            failure_reasons.append(f"{split_name}_supported_recall_below_minimum")

    calibration_sha256 = _scored_split_sha256(calibration_scored)
    evaluation_sha256 = _scored_split_sha256(evaluation_scored)
    policy_sha256 = _sha256(_canonical_json(asdict(policy)))
    semantic_guard_version = (
        SEMANTIC_LEAKAGE_GUARD_VERSION if semantic_leakage_guard is not None else None
    )
    semantic_scorer_id = (
        semantic_leakage_guard.scorer_id if semantic_leakage_guard is not None else None
    )
    semantic_model_revision_sha256 = (
        semantic_leakage_guard.model_revision_sha256
        if semantic_leakage_guard is not None
        else None
    )
    drift_slice_version = (
        DRIFT_SLICE_VERSION if drift_slice_policy is not None else None
    )
    drift_slice_policy_sha256 = (
        _sha256(_canonical_json(asdict(drift_slice_policy)))
        if drift_slice_policy is not None
        else None
    )
    report_document = {
        "schema_version": SCHEMA_VERSION,
        "leakage_guard_version": LEAKAGE_GUARD_VERSION,
        "scorer_id": scorer_id,
        "threshold": threshold,
        "calibration": asdict(calibration_metrics),
        "evaluation": asdict(evaluation_metrics),
        "calibration_sha256": calibration_sha256,
        "evaluation_sha256": evaluation_sha256,
        "policy_sha256": policy_sha256,
        "meets_policy": not failure_reasons,
        "failure_reasons": failure_reasons,
        "near_duplicate_comparisons": similarity_budget.used,
        "semantic_leakage_guard_version": semantic_guard_version,
        "semantic_scorer_id": semantic_scorer_id,
        "semantic_model_revision_sha256": semantic_model_revision_sha256,
        "semantic_duplicate_comparisons": semantic_similarity_budget.used,
        "drift_slice_version": drift_slice_version,
        "drift_slice_policy_sha256": drift_slice_policy_sha256,
        "slice_assignment_sha256": slice_assignment_sha256,
        "drift_slices": [asdict(item) for item in drift_slices],
    }
    return EntailmentDiagnosticReport(
        schema_version=SCHEMA_VERSION,
        leakage_guard_version=LEAKAGE_GUARD_VERSION,
        scorer_id=scorer_id,
        threshold=threshold,
        calibration=calibration_metrics,
        evaluation=evaluation_metrics,
        calibration_sha256=calibration_sha256,
        evaluation_sha256=evaluation_sha256,
        policy_sha256=policy_sha256,
        report_sha256=_sha256(_canonical_json(report_document)),
        meets_policy=not failure_reasons,
        failure_reasons=tuple(failure_reasons),
        near_duplicate_comparisons=similarity_budget.used,
        semantic_leakage_guard_version=semantic_guard_version,
        semantic_scorer_id=semantic_scorer_id,
        semantic_model_revision_sha256=semantic_model_revision_sha256,
        semantic_duplicate_comparisons=semantic_similarity_budget.used,
        drift_slice_version=drift_slice_version,
        drift_slice_policy_sha256=drift_slice_policy_sha256,
        slice_assignment_sha256=slice_assignment_sha256,
        drift_slices=drift_slices,
    )
