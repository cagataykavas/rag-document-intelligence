# Chunk-boundary sensitivity audit

Changing chunk size or overlap rebuilds a RAG index without changing the source
corpus. Retrieval metrics can still look healthy while the actual document regions
sent to generation change materially. Comparing chunk IDs cannot measure this risk:
chunk IDs and boundaries necessarily change between configurations.

`src.chunking_sensitivity` compares the byte regions retrieved from the immutable
source documents. It merges overlapping chunks before measuring coverage, so the
gate evaluates evidence rather than chunk representation.

For each query and non-baseline configuration it measures:

- byte-span coverage Jaccard against the baseline retrieval;
- retrieved-document set Jaccard;
- reciprocal-rank-weighted document Jaccard;
- coverage of reviewed required-evidence spans in both configurations.

Required spans prevent two consistently wrong configurations from passing merely
because they retrieve the same irrelevant text. The artifact binds an immutable
corpus digest, retriever digest, every chunking configuration, document byte sizes,
queries, required spans and ranked retrieval spans. Reports contain hashed query
references and metrics, not query text or source content.

## Artifact contract

Spans are zero-based, half-open byte offsets into the exact document bytes covered by
`document_sha256`:

```json
{
  "schema_version": 1,
  "corpus_sha256": "<64 lowercase hex>",
  "retriever_sha256": "<64 lowercase hex>",
  "baseline_configuration_sha256": "<64 lowercase hex>",
  "configurations": ["<baseline digest>", "<candidate digest>"],
  "documents": [
    {"document_sha256": "<document digest>", "size_bytes": 4096}
  ],
  "queries": [
    {
      "query_id": "refund-policy",
      "required_spans": [
        {"document_sha256": "<document digest>", "start_byte": 800, "end_byte": 940}
      ],
      "variants": [
        {
          "configuration_sha256": "<baseline digest>",
          "results": [
            {"rank": 1, "document_sha256": "<document digest>", "start_byte": 760, "end_byte": 960}
          ]
        }
      ]
    }
  ]
}
```

Every query must contain exactly one result list for every declared configuration.
Ranks are contiguous from one. Overlap inside a configuration is allowed and counted
once; duplicate identical spans are rejected.

```bash
python -m src.chunking_sensitivity chunking-evidence.json \
  --policy chunking-policy.json \
  --output artifacts/chunking-sensitivity.json
```

Exit codes are `0` for accepted, `2` for a policy rejection and `3` for malformed or
operational input. The default policy requires at least two configurations, 0.60
span Jaccard, 0.50 document and rank-weighted Jaccard, and 0.80 required-span recall.

## Producer and trust boundaries

- Record offsets during parsing/chunk creation. Searching chunk text back inside a
  document is ambiguous when text repeats or normalization changes.
- Offsets are over bytes, not Unicode code points. Hash and measure the same immutable
  byte representation before decoding, OCR cleanup or whitespace normalization.
- Required spans are relevance judgements. They can be incomplete, biased or stale;
  version them with the benchmark and review them independently from chunking changes.
- Stable retrieved spans do not prove factuality, authority, entailment or answer
  correctness. Compose this result with lifecycle, access, citation and grounding
  admission before publication.
- SHA-256 identifies exact content but does not authenticate the producer. Sign or
  append-only persist the artifact before using it as release authority.
- Thresholds are calibration policy. Legitimate boundary changes can improve evidence
  while reducing overlap with a weak baseline; required-span recall keeps that tradeoff
  visible but does not decide it universally.

The next integration step is to persist source byte offsets in the real chunker, run
baseline and candidate indexes against the same frozen query suite, and bind an
accepted audit digest to the index promotion transaction.
