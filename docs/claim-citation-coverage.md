# Claim-level citation coverage gate

A citation can be valid and even entail the sentence beside it while other material claims in the same answer remain uncited. `src.citation_coverage` adds a release boundary for this **citation completeness** failure mode.

The input artifact binds an answer to its query, model, prompt digest and retrieval-snapshot digest. Its claim manifest records exact character spans, claim types and citation IDs. The audit then verifies:

- claim IDs, spans, text and citation lists are structurally valid and bounded;
- claim spans are ordered, non-overlapping and, by default, cover every answer content token;
- every factual, quantitative, recommendation and attribution claim has a citation;
- every citation belongs to the bound retrieval manifest;
- structured literals such as numbers, percentages, URLs, UUIDs and requirement IDs cannot be hidden under an exempt claim type;
- answer and abstention modes follow different fail-closed contracts;
- artifacts are timezone-aware, fresh and not materially future-dated.

Reports contain only counts, rates, stable reason codes and canonical SHA-256 identities. They do not repeat the answer, claim text, query ID or chunk IDs.

## Artifact contract

```json
{
  "schema_version": 1,
  "generated_at": "2026-09-27T09:59:00Z",
  "response_kind": "answer",
  "query_id": "query-17",
  "model_id": "generator-v3",
  "prompt_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "retrieval_snapshot_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "answer": "Audit logs are retained for seven years.",
  "retrieved_chunk_ids": ["chunk-policy"],
  "claims": [
    {
      "claim_id": "claim-1",
      "start_char": 0,
      "end_char": 40,
      "text": "Audit logs are retained for seven years.",
      "claim_type": "factual",
      "citation_ids": ["chunk-policy"]
    }
  ]
}
```

Run the gate with an explicit evaluation time for reproducible CI evidence:

```bash
python -m src.citation_coverage artifact.json --now 2026-09-27T10:00:00Z
```

Exit code `0` means accepted, `2` means a well-formed artifact violated policy, and `3` means the artifact or policy was malformed.

## Trust boundary

This gate proves that declared material claims are comprehensively mapped to retrieved evidence. It does **not** prove that a citation entails a claim, that retrieval is correct, or that the claim segmenter produced semantically atomic claims. Exact span validation and the content-token coverage threshold make omission visible, but they cannot replace semantic evaluation.

The next increment is to compose this report with the citation-entailment audit, require both digests before serving an answer, and produce the claim manifest directly from a versioned generation trace rather than accepting an independently assembled artifact.
