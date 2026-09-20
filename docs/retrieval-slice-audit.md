# Retrieval slice audit

Aggregate Recall@K or MRR can hide a severe failure for a smaller but important
query class. `src.retrieval_slices` evaluates labelled retrieval cases by an
explicit slice such as language, document family, query length, numeric lookup,
or requirement-ID lookup.

```python
from src.retrieval_slices import SlicePolicy, SlicedRetrievalCase, audit_retrieval_slices

report = audit_retrieval_slices(
    cases,
    SlicePolicy(
        k=5,
        min_cases_per_slice=20,
        min_recall_at_k=0.80,
        min_mrr=0.70,
        max_recall_gap=0.15,
    ),
)
```

The release fails when an eligible slice misses its absolute Recall@K or MRR
floor, or trails aggregate recall by more than the configured gap. Underpowered
slices are explicitly listed rather than silently treated as reliable. A run
with no eligible slices fails closed. The JSON-ready report includes the policy,
overall metrics, per-slice metrics, excluded slices, and stable reason codes.

## Limits

Slice labels and relevance judgements must come from a versioned evaluation set;
this utility does not infer them. Thresholds are release policy, not universal
quality constants. Reusing queries during retriever tuning can overfit the gate,
and small slices have high variance even above a minimum-count threshold. For a
high-stakes release, pair this audit with confidence intervals, fresh holdout
queries, latency/cost checks, and manual error analysis.
