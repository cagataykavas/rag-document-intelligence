# Retrieval provenance gate

A citation ID proves little if the underlying index can change between retrieval, generation and
audit. This gate binds every retrieved chunk to one content-addressed index snapshot before the
evidence can be treated as reproducible.

The snapshot manifest records the retriever identity, chunking-policy identity, creation time and
the `(chunk_id, document_id, content_sha256)` inventory. Its `snapshot_digest` covers the canonical,
sorted manifest. Each query then declares the same snapshot and retriever and supplies contiguous
ranked results whose text digest must match both the result row and snapshot inventory.

```bash
python -m src.provenance evidence/retrieval.json \
  --evaluated-at 2026-09-24T00:00:00+00:00 \
  --min-queries 10 \
  --min-results-per-query 3 \
  --max-snapshot-age-seconds 86400
```

Exit codes are stable for CI: `0` accepts, `2` identifies malformed or internally inconsistent
evidence, and `3` rejects well-formed evidence under the configured freshness or sample policy.
Reports never include text content, only bounded identifiers, counts, digests and reason codes.

`build_snapshot` accepts the repository's existing `Chunk` objects. `bind_query_results` converts
actual retriever rows into the audit format while recomputing content digests rather than trusting
caller-provided hashes.

## Guarantees and limits

The gate detects mixed snapshots, retriever drift, changed chunk text, manifest tampering,
non-contiguous rankings, unknown chunks and stale/underpowered evidence. It does not prove that a
document is true, that chunking is semantically appropriate, that the retriever is relevant, or
that the snapshot was produced by an authorized indexer. SHA-256 is an identity mechanism, not a
signature. Production systems should sign manifests, retain them in immutable storage and record
the accepted snapshot digest with the prompt, model response and citation audit.

The reference manifest contains a full chunk inventory for transparent verification. Very large
corpora should replace this with a signed Merkle tree or registry lookup while retaining the same
per-result membership and content checks.
