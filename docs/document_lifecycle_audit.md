# Document lifecycle admission audit

Retrieval quality and citation checks do not answer a separate production
question: **was the returned chunk still eligible to be served?** A vector
index can retain an old revision after a source document has been replaced or
deleted. A high similarity score does not make that stale content valid.

`src/document_lifecycle_audit.py` is a fail-closed release/admission gate for
that boundary. It compares the exact served Top-K projection with an
authoritative lifecycle catalog and rejects results that refer to deleted,
unknown, superseded, or temporally impossible content.

## Contract

The producer exports a bounded JSON artifact:

```json
{
  "schema_version": "rag-document-lifecycle/v1",
  "observed_at": "2026-09-28T23:59:50Z",
  "catalog_revision": "catalog-42",
  "index_revision": "index-17",
  "documents": [
    {
      "document_id": "retention-policy",
      "current_revision": "rev-3",
      "state": "active",
      "state_changed_at": "2026-09-28T23:00:00Z"
    }
  ],
  "chunks": [
    {
      "chunk_id": "retention-policy:rev-3:0",
      "document_id": "retention-policy",
      "document_revision": "rev-3",
      "indexed_at": "2026-09-28T23:30:00Z"
    }
  ],
  "queries": [
    {
      "query_id": "query-001",
      "retrieved_at": "2026-09-28T23:59:40Z",
      "results": [
        {"chunk_id": "retention-policy:rev-3:0", "rank": 1}
      ]
    }
  ]
}
```

`documents` is the authoritative catalog view. `chunks` is the index manifest
for the retrieval snapshot. `queries[].results` must be the actual served
projection, not a reconstructed sample of candidates. Ranks are unique and
contiguous so omission or post-audit reordering is visible to the producer.

Each served result must satisfy all of these conditions:

1. the chunk exists in the supplied index manifest;
2. its document exists in the lifecycle catalog;
3. the lifecycle state was established before retrieval;
4. the document is `active`;
5. the chunk revision equals the catalog's `current_revision`; and
6. the chunk was indexed before the retrieval event.

The observation and query windows are independently bounded. This prevents an
otherwise valid but old artifact from being reused as current release
evidence.

## Run the gate

```bash
python -m src.document_lifecycle_audit lifecycle.json \
  --output lifecycle-report.json
```

Exit codes are stable for CI and deployment policy:

| Code | Meaning |
| ---: | --- |
| `0` | Artifact is well formed and every served result is lifecycle-valid |
| `2` | Artifact is well formed but the lifecycle policy rejected it |
| `3` | Artifact, policy, or output is malformed/unavailable |

The report contains only bounded reason codes and SHA-256-derived private
references. Raw document, query, and chunk identifiers are not copied into the
finding list. `artifact_digest`, `policy_digest`, and `evidence_id` make the
decision reproducible without claiming that an unsigned digest authenticates
the producer.

Policy failures include:

| Reason code | Interpretation |
| --- | --- |
| `DOCUMENT_NOT_ACTIVE` | A served chunk belongs to a deleted document |
| `SUPERSEDED_DOCUMENT_REVISION` | A served chunk is not from the current revision |
| `UNKNOWN_CHUNK` | A served chunk is absent from the index manifest |
| `UNKNOWN_DOCUMENT` | A manifest chunk has no lifecycle catalog entry |
| `CHUNK_INDEXED_AFTER_RETRIEVAL` | The supplied event order is impossible |
| `LIFECYCLE_STATE_AFTER_RETRIEVAL` | Catalog state cannot establish eligibility at retrieval time |
| `STALE_QUERY_EVIDENCE` | Query evidence exceeds its freshness window |
| `STALE_ARTIFACT` | The artifact observation is too old for admission |
| `ARTIFACT_FROM_FUTURE` | Clock skew exceeds the configured allowance |

Schema ambiguity is not treated as a normal policy failure. Duplicate JSON
keys, non-finite values, unknown fields, invalid UTC timestamps, duplicate
identifiers/ranks, broken rank sequences, and exceeded resource budgets return
the malformed exit code.

## Integration boundary

The gate should run after ranking and access filtering, but before chunk text
is placed in a model prompt:

```text
catalog snapshot + index manifest + served Top-K
                         |
                         v
                lifecycle admission
                    |          |
                  accept     reject
                    |          |
                    v          v
              prompt build   abstain / refresh index
```

In a live system, obtain the lifecycle catalog revision and index manifest from
the same transactional or causally ordered snapshot. Bind the resulting
`evidence_id` to the generation trace. On rejection, refresh or tombstone the
index before retrying; do not silently discard only the failing rows and allow
the prompt to proceed under a weaker evidence set.

## Security and correctness limits

- The audit trusts the catalog and index-manifest producer. It does not query
  the source-of-truth database or prove that the supplied served projection is
  complete.
- SHA-256 evidence IDs provide deterministic identity, not authenticity. Sign
  artifacts or store them in an append-only system when producer integrity is
  required.
- The gate does not delete stale vectors, implement tombstone propagation, or
  stop a downstream component from bypassing the admitted result set.
- A current revision can still contain factually wrong, poisoned, or
  unauthorized text. Existing access-scope, prompt-injection, retrieval, and
  grounding controls remain necessary.
- A single catalog view cannot reconstruct arbitrary historical eligibility.
  `LIFECYCLE_STATE_AFTER_RETRIEVAL` therefore rejects ambiguous time ordering
  rather than guessing.

The next production increment is to emit this artifact inside the real query
transaction, sign the catalog/index revisions, and reconcile lifecycle
tombstones against the vector store as an observable background process.
