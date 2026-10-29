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

## Versioned baseline regression gate

An accepted sliced report can be sealed as an immutable baseline. The artifact contains only the
source commitments, confusion matrices and reportable low-cardinality slice keys. It can be stored
as canonical JSON and loaded in a later release job with exact-field, type, metric-consistency and
SHA-256 integrity checks:

```python
import json

from src.entailment_baseline import (
    EntailmentRegressionPolicy,
    MetricRegressionThresholds,
    SliceRegressionRule,
    compare_entailment_baseline,
    create_entailment_baseline,
    load_entailment_baseline,
)

baseline = create_entailment_baseline(
    accepted_report,
    baseline_id="sealed-holdout",
    baseline_version=3,
)
baseline_json = json.dumps(baseline.to_audit_dict(), sort_keys=True)
baseline = load_entailment_baseline(json.loads(baseline_json))

global_limits = MetricRegressionThresholds(
    max_supported_precision_drop=0.02,
    max_supported_recall_drop=0.03,
    max_unsupported_recall_drop=0.03,
    max_balanced_accuracy_drop=0.02,
)

# This mapping is reviewed configuration, not inferred from the candidate run.
reviewed_slice_limits = {
    ("language", "tr"): global_limits,
    ("document_domain", "finance"): global_limits,
    ("claim_length_utf8", "65-256"): global_limits,
    ("evidence_length_utf8", "0-256"): global_limits,
}
comparison_policy = EntailmentRegressionPolicy(
    confidence_level=0.95,
    global_thresholds=global_limits,
    slice_rules=tuple(
        SliceRegressionRule(dimension, value, thresholds)
        for (dimension, value), thresholds in reviewed_slice_limits.items()
    ),
)

comparison = compare_entailment_baseline(
    baseline,
    candidate_report,
    comparison_policy,
)
```

Each rate receives a Wilson score interval. Candidate-minus-baseline bounds use Newcombe's
independent-proportions construction: the point difference is widened by the root-sum-square of the
relevant Wilson deviations. A metric regresses only when the difference interval's upper bound is
below the negative reviewed tolerance. A lower point estimate whose interval still overlaps the
tolerance is therefore not presented as a statistically resolved regression.

Supported precision uses predicted-supported trials, supported recall uses labeled-supported trials
and unsupported recall uses labeled-unsupported trials. Balanced-accuracy bounds average the two
recall bounds. The uncertainty method has an explicit version in every comparison artifact so a
future statistical change cannot silently alter historical gates.

The candidate must use the same diagnostic schema, entailment policy and drift-slice policy as the
baseline. Its own entailment policy must still pass. Reviewed rules must exactly cover all reportable
baseline slices; a missing or newly reportable unreviewed slice fails closed instead of being silently
excluded. Both baseline and comparison artifacts omit claims, evidence and case IDs. Their SHA-256
digests detect accidental or malicious modification but do not authenticate who approved an artifact;
production promotion should add repository review or an external signature.

## Paired holdout regression gate

When two scorers are evaluated on the exact same sealed holdout, the paired comparator can retain
that case-level correlation instead of treating their aggregate rates as independent. Callers must
capture the score mapping used by each diagnostic run; the ledger builder does not invoke or trust a
scorer a second time:

```python
import json

from src.entailment_paired_regression import (
    PairedBootstrapPolicy,
    compare_paired_score_ledgers,
    create_evaluation_score_ledger,
    load_evaluation_score_ledger,
)

baseline_ledger = create_evaluation_score_ledger(
    accepted_report,
    evaluation_pairs,
    accepted_scores_by_case_id,
)
baseline_ledger = load_evaluation_score_ledger(
    json.loads(json.dumps(baseline_ledger.to_audit_dict()))
)
candidate_ledger = create_evaluation_score_ledger(
    candidate_report,
    evaluation_pairs,
    candidate_scores_by_case_id,
)

paired_policy = PairedBootstrapPolicy(
    confidence_level=0.95,
    resamples=10_000,
    thresholds=global_limits,
    family_wise_method="bonferroni",
)
paired_comparison = compare_paired_score_ledgers(
    baseline_ledger,
    candidate_ledger,
    paired_policy,
)
```

Ledger creation reproduces the diagnostic's evaluation SHA-256 and confusion matrix before sealing
the artifact. It stores sorted SHA-256 commitments to case IDs and exact content, labels and scores,
but never raw IDs, claims or evidence. Comparison requires identical case/content/label commitments,
diagnostic schemas and entailment policies, so a changed population cannot masquerade as a paired
experiment.

The bootstrap resamples supported and unsupported cases separately and uses each sampled case for
both scorers. Its seed is derived from the two ledger digests, method version and policy digest; a
SHA-256 counter sampler makes repeated runs deterministic. For the four global metrics, a regression
is reported only when the Bonferroni-adjusted one-sided upper delta bound is below the negative
reviewed tolerance. The method, seed commitment, adjusted confidence, hypothesis count and bounded
resample count are included in the comparison artifact. Jobs fail before sampling when requested
work exceeds five million case draws.

### Verified evaluation-only slice assignments

The diagnostic report commits to calibration and evaluation assignments together. A separate
artifact reproduces that combined commitment and every aggregate slice count before sealing only
the evaluation rows needed for paired analysis:

```python
from src.entailment_slice_assignment import (
    create_evaluation_slice_assignment,
    join_evaluation_slice_assignments,
    load_evaluation_slice_assignment,
)

assignments = create_evaluation_slice_assignment(
    accepted_report,
    calibration_pairs,
    evaluation_pairs,
    slice_metadata,
    drift_slice_policy,
)
assignments = load_evaluation_slice_assignment(
    json.loads(json.dumps(assignments.to_audit_dict()))
)
slice_score_rows = join_evaluation_slice_assignments(
    baseline_ledger,
    assignments,
)
```

Rows contain stable case and content SHA-256 commitments plus the four low-cardinality slice values;
they omit raw case IDs, claims and evidence. The join integrity-checks both artifacts and requires an
exact report schema, report digest, evaluation digest, row ordering and case/content population match.
The artifact is reusable only with a score ledger from the same report, preventing unrelated slice
metadata from being attached to scored cases.

The paired comparator still gates only global metrics; the verified joined rows are evidence-ready
input rather than a claim of per-slice statistical support. A next increment should add reviewed
per-slice thresholds and family-wise-error accounting across both metrics and slices. The score and
assignment ledgers reveal labels, scores, low-cardinality categories and stable pseudonymous
commitments; hashes are integrity commitments rather than anonymization, and access should be
restricted. The percentile bootstrap is an offline diagnostic, not a sequential test,
serving-population weighting scheme, baseline approval signature or online serving authority.

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
change. The baseline comparator's per-slice independent-proportions bounds remain conservative. The
paired gate models global case correlation and controls the four-metric family-wise error rate.
Evaluation-only slice assignments can now be verified and joined to scored rows, but the comparator
does not yet provide paired slice inference, serving-population weights or repeated-look error
control. A useful next increment is reviewed per-slice bootstrap policy with explicit multiplicity
accounting and minimum sample floors.
