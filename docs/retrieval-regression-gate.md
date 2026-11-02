# Retrieval regression gate

`src/retrieval_baseline.py` turns aggregate retrieval diagnostics into a reviewed,
content-addressed release gate. It is intentionally separate from ranking: the gate
can reject a candidate, but it cannot silently tune or replace retrieval behavior.

## What the baseline binds

Each `retrieval-regression-baseline/v1` artifact commits to:

- the full Git commit that produced the accepted result;
- the configured top-K and exact evaluation case count;
- independently computed SHA-256 commitments for the indexed corpus snapshot and
  evaluation case set;
- all document-level relevance and source-saturation metrics; and
- an explicit maximum regression for every metric.

Hit rate, recall, precision, reciprocal rank, nDCG, and unique-document coverage are
higher-is-better. Duplicate-document rate and largest-document share are
lower-is-better. Improvements never consume a regression budget.

```python
from src.retrieval_baseline import (
    RetrievalRegressionThresholds,
    compare_retrieval_baseline,
    create_retrieval_baseline,
)

thresholds = RetrievalRegressionThresholds(
    max_hit_rate_at_k_regression=0.01,
    max_recall_at_k_regression=0.02,
    max_precision_at_k_regression=0.02,
    max_mean_reciprocal_rank_regression=0.02,
    max_ndcg_at_k_regression=0.02,
    max_mean_unique_documents_at_k_regression=0.25,
    max_duplicate_document_rate_at_k_regression=0.02,
    max_mean_largest_document_share_at_k_regression=0.02,
)

baseline = create_retrieval_baseline(
    accepted_metrics,
    thresholds,
    baseline_id="production-retrieval-k5",
    baseline_version=1,
    source_commit="0123456789abcdef0123456789abcdef01234567",
    k=5,
    corpus_snapshot_sha256=corpus_snapshot_sha256,
    evaluation_cases_sha256=evaluation_cases_sha256,
)

comparison = compare_retrieval_baseline(
    baseline,
    candidate_metrics,
    k=5,
    corpus_snapshot_sha256=corpus_snapshot_sha256,
    evaluation_cases_sha256=evaluation_cases_sha256,
)
if not comparison.meets_policy:
    raise SystemExit(2)
```

Persist `serialize_retrieval_baseline(baseline)` in a reviewed release artifact.
Load external bytes through `load_retrieval_baseline_json`; its 64 KiB boundary
rejects duplicate keys, non-ASCII or noncanonical JSON, missing or extra fields,
unsupported schemas, malformed ranges, and any artifact-digest change.
The comparison report contains only aggregate metrics, stable failure reasons, and
content commitments—never queries, document IDs, chunk IDs, rankings, or text.

## Fail-closed compatibility

A candidate is not comparable when any of these change:

- K;
- case count;
- corpus snapshot digest; or
- evaluation-case digest.

This prevents a release from passing by silently dropping difficult cases, switching
the indexed corpus, or evaluating at an easier K. A changed corpus or suite requires a
new reviewed baseline version instead of a relaxed comparison.

## Security and statistical boundary

The module validates supplied commitments; it does not decide how corpus and case-set
digests are produced or prove that those inputs are complete. Keep their canonical
snapshot procedure under review. The current gate compares point estimates and is
appropriate for deterministic evaluation. It does not estimate uncertainty, correct
for repeated tuning, or prove semantic diversity, source authority, answer quality,
or production traffic performance. A larger stochastic benchmark should add a
predeclared resampling or confidence-bound policy before using the same release gate.
