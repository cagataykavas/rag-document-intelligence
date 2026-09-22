# Bounded RAG context packing

Retrieval rank alone is not a safe prompt-assembly policy. A long chunk can overflow
the model context, repeated chunks from one document can crowd out independent
evidence, and a late-ranked mandatory source can be lost after optional evidence has
already consumed the budget.

`src.context_budget` provides a deterministic, fail-closed packing boundary for ranked
evidence. It:

- reserves space for chunks explicitly marked `required` before optional selection;
- measures the exact rendered evidence-block character cost, including identifiers and
  delimiters;
- keeps the original retrieval order in the packed result;
- limits raw chunk size without silently truncating evidence;
- caps chunks per source to reduce single-document domination;
- enforces minimum selected-chunk and distinct-source evidence;
- reports every exclusion with a stable reason code.

## CLI evidence preflight

Prepare a JSON array from a retrieval artifact:

```json
[
  {
    "chunk_id": "policy:4",
    "source": "security-policy.md",
    "text": "Administrative access requires multi-factor authentication.",
    "score": 0.91,
    "required": true
  },
  {
    "chunk_id": "guide:2",
    "source": "operator-guide.md",
    "text": "Operators enroll a second factor during account activation.",
    "score": 0.84
  }
]
```

Then run:

```bash
python -m src.context_budget \
  --input artifacts/retrieval-candidates.json \
  --max-context-chars 6000 \
  --max-chunk-chars 1800 \
  --max-chunks-per-source 2 \
  --min-selected-chunks 2 \
  --min-selected-sources 2
```

The JSON report includes selected chunk IDs, exact character utilization and exclusions.
`--require-all` returns exit code `3` when a valid candidate set requires any exclusion;
malformed evidence or an impossible policy returns exit code `2`.

## Trust boundary and limitations

The character budget applies only to rendered evidence blocks. Reserve separate model
capacity for system instructions, examples, the question and generated answer. Character
counts are deterministic across tokenizers but are not token counts; a provider adapter
should translate the model's token limit into a conservative evidence budget or add an
exact tokenizer at the boundary.

Source diversity is a structural guardrail, not proof of independent corroboration.
Aliases, mirrors and duplicated content can still appear as different sources. Required
flags are trusted policy inputs and must not be taken from untrusted document text.
Packing also cannot recover relevant evidence that retrieval failed to return.
