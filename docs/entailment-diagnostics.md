# Calibrated entailment diagnostics

The deterministic generation-admission policy answers a contract question: did the model return a
bounded payload whose claims cite allowed evidence? It deliberately does not claim that the evidence
semantically supports each claim.

`src/entailment_eval.py` adds a separate, model-agnostic harness for that probabilistic question. A
caller supplies labeled `supported` / `unsupported` claim-evidence pairs and a scorer that returns a
finite probability in `[0, 1]`:

```python
from src.entailment_eval import (
    EntailmentLabel,
    EntailmentPair,
    EntailmentPolicy,
    evaluate_entailment_diagnostic,
)

calibration = [
    EntailmentPair(
        case_id="cal-1",
        claim="Administrative access requires authentication.",
        evidence="All administrative access shall require authentication.",
        label=EntailmentLabel.SUPPORTED,
    ),
    EntailmentPair(
        case_id="cal-2",
        claim="Audit events are retained for seven years.",
        evidence="Audit events are retained for ninety days.",
        label=EntailmentLabel.UNSUPPORTED,
    ),
]

report = evaluate_entailment_diagnostic(
    calibration,
    evaluation_pairs,
    scorer=my_cross_encoder,
    scorer_id="multilingual-nli-2026-10",
    policy=EntailmentPolicy(
        min_supported_precision=0.95,
        min_supported_recall=0.80,
    ),
)
```

## Versioned semantic leakage adapter

The surface guard cannot identify paraphrases that share little character-level text. For sealed
offline suites, callers can additionally provide a `VersionedSemanticLeakageGuard` backed by a
reviewed multilingual NLI or semantic-similarity model:

```python
from src.entailment_eval import VersionedSemanticLeakageGuard

semantic_guard = VersionedSemanticLeakageGuard(
    scorer=compare_claim_evidence_pairs,
    scorer_id="multilingual-nli:turkish-v3",
    model_revision_sha256=model_artifact_sha256,
)

report = evaluate_entailment_diagnostic(
    calibration,
    evaluation_pairs,
    scorer=my_cross_encoder,
    scorer_id="multilingual-nli-2026-10",
    semantic_leakage_guard=semantic_guard,
)
```

The adapter compares every pair within calibration, within holdout and across the two splits after
the cheaper exact and character-shingle guards pass. The policy's
`max_semantic_duplicate_comparisons` is a hard work budget: if a complete scan cannot finish inside
that budget, evaluation fails before the entailment scorer runs. This deliberately favors an
auditable complete check over a silent partial scan. The default budget of 100,000 comparisons is
appropriate only for bounded offline suites; larger corpora need a reviewed semantic candidate
index and a separate recall benchmark.

Adapter exceptions and invalid, boolean, non-finite or out-of-range similarities are normalized at
the untrusted boundary without copying provider diagnostics or evaluated text. The v4 report binds
the guard version, scorer ID, immutable model-artifact SHA-256 and actual comparison count. It does
not store raw pair text.

## Privacy-reduced drift slices

Drift slicing is explicit and opt-in. Callers provide a low-cardinality language and document-domain
category for every calibration and holdout case, plus a policy that sets sample floors and UTF-8
length buckets:

```python
from src.entailment_eval import (
    DriftSlicePolicy,
    EntailmentSliceMetadata,
)

slice_metadata = {
    "cal-1": EntailmentSliceMetadata(language="en", document_domain="security"),
    "cal-2": EntailmentSliceMetadata(language="tr", document_domain="finance"),
    # Every calibration and holdout case ID must appear exactly once.
}

report = evaluate_entailment_diagnostic(
    calibration,
    evaluation_pairs,
    scorer=my_cross_encoder,
    scorer_id="multilingual-nli-2026-10",
    drift_slice_metadata=slice_metadata,
    drift_slice_policy=DriftSlicePolicy(
        min_pairs_per_split=20,
        min_pairs_per_label_per_split=5,
        claim_length_utf8_boundaries=(64, 256),
        evidence_length_utf8_boundaries=(256, 1024),
    ),
)
```

The same calibration-selected threshold is applied to every slice. For language, document domain,
claim length and evidence length, the report compares calibration and holdout confusion metrics and
their deltas. Metrics are emitted only when both splits meet the total-pair floor and contain the
configured minimum number of both `supported` and `unsupported` labels. Below-floor slices expose
only aggregate pair counts; they do not emit unstable confusion metrics or deltas.

Category values are restricted to 32 safe characters and their count is bounded per dimension.
They are intentionally visible in the audit report, so callers must use reviewed, non-sensitive
taxonomy labels rather than tenant, customer or document names. Raw claim/evidence text and case IDs
remain absent. A SHA-256 commitment binds every hashed case ID to its split, categories and computed
length buckets; a separate commitment binds the slicing policy.

## Leakage-safe calibration

The supported threshold is selected only from the calibration split. Among thresholds that satisfy
the configured minimum precision, the harness chooses the one with the highest supported recall;
precision, balanced accuracy and the more conservative threshold break ties deterministically. The
selected threshold is then applied unchanged to the holdout evaluation split.

Both splits must contain both labels. Duplicate case IDs or claim-evidence contents are rejected
within a split, and case-ID or content overlap across calibration and holdout fails closed. A
versioned Unicode NFKC + casefold + punctuation normalization guard also canonicalizes the combining
dot produced by Turkish dotted-I casing and compares five-character shingle sets. A pair is
near-duplicate only when both its claim and evidence meet the configured Jaccard threshold. This
catches renamed examples with superficial casing, spacing, punctuation, or minor textual edits
without treating a shared generic claim alone as leakage.

Candidate generation uses a shingle index rather than comparing every pair blindly. The policy
also bounds the total exact similarity comparisons; a highly repetitive or unexpectedly large suite
that exceeds that budget fails before invoking the scorer. The normalization version, threshold,
comparison budget, and observed comparison count are committed into the audit report.

## Bounded untrusted scorer boundary

The harness bounds pair counts, claim bytes, evidence bytes and total split bytes before invoking the
scorer. Boolean, non-numeric, non-finite and out-of-range scores are rejected. Scorer exceptions are
normalized to a stable error and do not copy provider messages or evaluated text into the report.

The report contains:

- the calibration-selected threshold;
- supported precision and recall, unsupported recall and balanced accuracy;
- the full confusion matrix for calibration and holdout;
- explicit policy-failure reason codes;
- SHA-256 commitments to case identity, exact claim/evidence contents, labels and scores;
- policy and final report commitments;
- the versioned near-duplicate guard and number of bounded similarity comparisons;
- optional semantic-leakage guard provenance and completed comparison count;
- optional privacy-reduced drift slices, suppression decisions and metric deltas.

Raw claim and evidence text are intentionally omitted from the audit report. These hashes are
integrity commitments, not anonymization; predictable text may still be vulnerable to guessing.

## Authority boundary and limitations

`meets_policy` is an evaluation result, not serving authority. It does not alter
`GenerationAdmission`, bypass citation validation or release model text. A production team may use
the report as one reviewed release input, but deterministic admission remains the online safety
boundary.

The harness does not provide a pretrained entailment model, establish source authority or prove
factual truth. The optional semantic guard trusts the supplied adapter and artifact commitment; a
SHA-256 revision proves byte identity, not model quality. Both lexical and semantic thresholds can
reject legitimately related but non-duplicate examples. Performance depends on representative
labels and may drift by language, domain, claim granularity and evidence length. Deployments should
keep a sealed holdout set, review class balance and recalibrate after any scorer or data-distribution
change. Slice deltas do not provide confidence intervals, hypothesis tests or serving-population
weights. A useful next increment is a versioned baseline comparator with uncertainty bounds and
reviewed per-slice regression thresholds.
