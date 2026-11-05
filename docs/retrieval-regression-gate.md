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
  --inventory evaluation/retrieval-source-inventory.json \
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

The source inventory is a second canonical artifact containing the exact source identifier and
normalized-content SHA-256 expected for every indexed document:

```json
{"schema_version":"retrieval-source-inventory/v1","sources":[{"content_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef","source":"policies/retention.md"}]}
```

Rows must be sorted by `(source, content_sha256)`. Content digests must be unique, matching the
store's content-addressed admission rule. The gate compares this reviewed population to every
document row inside the same read transaction used for corpus hashing and evaluation. A missing,
unexpected, renamed, or content-substituted source therefore fails before retrieval runs. The
inventory is bounded to 4 MiB and rejects duplicate keys, duplicate content identities, malformed
UTF-8, noncanonical JSON, unsupported schemas, and more than 100,000 sources.

## What the baseline binds

Each `retrieval-regression-baseline/v2` artifact commits to:

- the full Git commit that produced the accepted result;
- the configured top-K and exact evaluation case count;
- independently computed SHA-256 commitments for the indexed corpus snapshot and
  evaluation case set;
- the exact reviewed source-inventory commitment;
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
    source_inventory_sha256=source_inventory_sha256,
)

comparison = compare_retrieval_baseline(
    baseline,
    candidate_metrics,
    k=5,
    corpus_snapshot_sha256=corpus_snapshot_sha256,
    evaluation_cases_sha256=evaluation_cases_sha256,
    source_inventory_sha256=source_inventory_sha256,
)
if not comparison.meets_policy:
    raise SystemExit(2)
```

Persist `serialize_retrieval_baseline(baseline)` in a reviewed release artifact.
Load external bytes through `load_retrieval_baseline_json`; its 64 KiB boundary
rejects duplicate keys, non-ASCII or noncanonical JSON, missing or extra fields,
unsupported schemas, malformed ranges, and any artifact-digest change.
Version 1 baselines are intentionally rejected because they do not bind a reviewed
source inventory; regenerate and review a version 2 baseline instead of silently upgrading it.
The comparison report contains only aggregate metrics, stable failure reasons, and
content commitments—never queries, document IDs, chunk IDs, rankings, or text.

## Independent inventory review quorum

`src/retrieval_inventory_review.py` authenticates the review decision separately from the
baseline author. An attestation binds the exact source-inventory digest, baseline artifact digest,
and a policy digest under a domain-separated Ed25519 message. The policy digest commits to a
required threshold of at least two, the sorted reviewer key IDs, and the SHA-256 fingerprint of
each distinct Ed25519 public key. Reusing one public key under multiple aliases is rejected, so an
alias cannot satisfy a quorum.

```python
from src.retrieval_gate import gate_reviewed_candidate_snapshot

comparison, receipt = gate_reviewed_candidate_snapshot(
    candidate_snapshot,
    baseline_payload,
    inventory_review_payload,
    reviewer_key_status_payload,
    reviewer_public_keys=protected_reviewer_keyring,
    required_review_signatures=2,
    status_authority_public_keys=protected_status_authority_keyring,
    required_status_authority_signatures=2,
    verification_epoch=trusted_verification_epoch,
)
```

The high-level gate first verifies a canonical, domain-separated Ed25519 status snapshot. The
snapshot must cover the exact reviewer keyring, bind the review-policy digest, remain within a
24-hour maximum validity window, be current at the supplied verification epoch, and satisfy the
configured quorum of trusted authority keys. The snapshot binds a policy digest over the exact
authority threshold, sorted key IDs, and distinct public-key fingerprints. Every authority key must
also be cryptographically distinct from every reviewer key. Every signer in the inventory review
must be active. Only then does the gate verify the review quorum, cross-check the attested inventory
and baseline, and apply the retrieval comparison.

The authority policy accepts 2–8 distinct Ed25519 keys and requires a threshold between two and the
key count. Version 1 single-authority snapshots are deliberately rejected; regenerate them under a
reviewed version 2 authority policy instead of silently treating one old signature as a quorum.

The wrapper receipt cross-binds the review, status snapshot, status policy, verification epoch and
aggregate active/revoked counts, but not reviewer IDs, authority IDs, signatures, source names,
queries, or content. Treat the reviewer keyring, both thresholds, status-authority keyring and
verification epoch as protected deployment policy; accepting them from the untrusted change being
reviewed would remove the trust boundary.

A valid snapshot does not prove reviewer organizational independence, inventory authority or
completeness, trustworthy time, or that a compromised authority quorum cannot collude. A protected
release job should source time from authenticated infrastructure and eventually publish status
snapshots or their commitments to an independently monitored transparency log.

## Fail-closed compatibility

A candidate is not comparable when any of these change:

- K;
- case count;
- corpus snapshot digest; or
- evaluation-case digest; or
- reviewed source-inventory digest.

This prevents a release from passing by silently dropping difficult cases, switching
the indexed corpus, or evaluating at an easier K. A changed corpus or suite requires a
new reviewed baseline version instead of a relaxed comparison.

## Corpus snapshot procedure

The command opens SQLite read-only, enables `query_only`, starts one transaction, and
runs `PRAGMA quick_check`. Within that same consistent read snapshot it:

1. validates the required `documents` and `chunks` schema and bounded row counts;
2. requires the exact reviewed source/content inventory, without emitting source names;
3. streams every document in `doc_id` order into a domain-separated SHA-256 digest;
4. streams every chunk in `(doc_id, ordinal, chunk_id)` order, committing all provenance
   fields and a SHA-256 of text rather than text itself;
5. rejects incomplete provenance, duplicate IDs, orphan chunks, and labels for documents
   absent from the snapshot; and
6. fits and evaluates the sparse hybrid retriever from those exact chunk rows.

Length-framed canonical records avoid concatenation ambiguity. The candidate artifact
commits to K, row counts, case count, the corpus, case and reviewed-inventory commitments,
and all aggregate metrics.
The checked-in CI fixture demonstrates the complete ingest-to-gate path rather than
comparing hand-entered metrics.

## Security and statistical boundary

The snapshot proves completeness only relative to the reviewed inventory. It cannot prove that the
inventory's source selection is authoritative, current, or exhaustive, nor that labels are correct.
The current gate compares point estimates and is appropriate for deterministic
evaluation. It does not estimate uncertainty, correct for repeated tuning, or prove
semantic diversity, source authority, answer quality, or production traffic performance.
A larger stochastic benchmark should add a predeclared resampling or confidence-bound
policy before using the same release gate.
