# Retrieval regression gate

`src/retrieval_baseline.py` turns aggregate retrieval diagnostics into a reviewed,
content-addressed release gate. `src/retrieval_gate.py` supplies the missing operational
boundary: it computes the corpus and case commitments, runs deterministic retrieval, and
applies that gate in one command. Neither module can silently tune or replace ranking.

## Deterministic CI command

```bash
python -m src.retrieval_gate \
  --database data/rag.db \
  --cases evaluation/retrieval-cases.json \
  --baseline evaluation/retrieval-baseline.json
```

The command exits `0` only when the computed candidate meets the reviewed baseline and
`2` for a regression, identity mismatch, malformed artifact, unsafe input, or evaluation
failure. Its JSON receipt contains counts, content commitments, comparison commitments,
and stable failure codes. It never emits queries, text, document/chunk IDs, or rankings.

The case file is canonical ASCII JSON with no trailing newline:

```json
{"cases":[{"case_id":"retention","query":"How long are audit records retained?","relevant_doc_ids":["doc-0123456789abcdef"]}],"schema_version":"retrieval-evaluation-cases/v1"}
```

Cases must be sorted by `case_id`; relevant document IDs must be sorted and unique. The
loader rejects duplicate JSON keys, missing or extra fields, noncanonical serialization,
invalid identifiers, empty labels, and files over 1 MiB. These constraints make the case
commitment reproducible across machines and keep labels reviewable in version control.

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

## Corpus snapshot procedure

The command opens SQLite read-only, enables `query_only`, starts one transaction, and
runs `PRAGMA quick_check`. Within that same consistent read snapshot it:

1. validates the required `documents` and `chunks` schema and bounded row counts;
2. streams every document in `doc_id` order into a domain-separated SHA-256 digest;
3. streams every chunk in `(doc_id, ordinal, chunk_id)` order, committing all provenance
   fields and a SHA-256 of text rather than text itself;
4. rejects incomplete provenance, duplicate IDs, orphan chunks, and labels for documents
   absent from the snapshot; and
5. fits and evaluates the sparse hybrid retriever from those exact chunk rows.

Length-framed canonical records avoid concatenation ambiguity. The candidate artifact
commits to K, row counts, case count, both snapshot digests, and all aggregate metrics.
The checked-in CI fixture demonstrates the complete ingest-to-gate path rather than
comparing hand-entered metrics.

## Security and statistical boundary

The snapshot covers the rows visible in the supplied SQLite database; it cannot prove
that upstream ingestion discovered every intended source or that labels are correct.
The current gate compares point estimates and is appropriate for deterministic
evaluation. It does not estimate uncertainty, correct for repeated tuning, or prove
semantic diversity, source authority, answer quality, or production traffic performance.
A larger stochastic benchmark should add a predeclared resampling or confidence-bound
policy before using the same release gate.
