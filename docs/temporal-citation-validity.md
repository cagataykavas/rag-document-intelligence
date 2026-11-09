# Temporal citation-validity audit

A citation may be relevant and semantically supportive but still be invalid for the date asked by a
user. A superseded policy, not-yet-effective regulation or document published after the requested
historical cutoff creates temporal leakage even when ordinary grounding metrics look healthy.

`src.temporal_validity` audits a generation trace against bitemporal source metadata:

- publication and indexing time;
- inclusive `valid_from` and exclusive `valid_until` intervals;
- query and claim effective times;
- source identity and revision;
- model, prompt and retrieval-snapshot identity.

```bash
python -m src.temporal_validity \
  --input temporal-citations.json \
  --output temporal-audit.json
```

The gate rejects citations published after the claim reference time, revisions that were not yet in
force or had already expired, post-generation indexing, mixed revisions of one source, unknown
citations, missing citation coverage and inconsistent query-as-of claims. Exit codes are `0` for
acceptance, `2` for policy rejection and `3` for malformed or operational evidence.

Reports include aggregate validity metrics, bounded reason codes and hashed query/claim/source
references. They do not repeat document text or raw identifiers.

## Trust boundary

The producer must obtain validity intervals from an authoritative catalog and record them in the
same retrieval transaction. SHA-256 provides content identity, not producer authenticity. Temporal
validity does not establish relevance, entailment, document authority or factual truth; it should be
composed with retrieval, citation-completeness and semantic-support gates.
