# Retrieval evidence admission

The generation path must not receive a chunk merely because the caller requested
`k` results. A sparse retriever can assign every indexed chunk a score of zero for an
out-of-vocabulary query; returning the first `k` entries in that case creates an
arbitrary evidence path and can turn a clean abstention into an unsupported answer.

`HybridSparseRetriever` now admits only chunks with positive lexical signal that also
meet the configured absolute and top-result-relative score floors. With the default
policy, zero-signal rows are removed while existing positive rankings remain eligible.

```python
from src.hybrid import HybridSparseRetriever, RetrievalAdmissionPolicy

policy = RetrievalAdmissionPolicy(
    min_fused_score=0.05,
    min_relative_score=0.25,
    max_query_chars=2_000,
    max_results=20,
)
retriever = HybridSparseRetriever(admission_policy=policy).fit(chunks)
evidence = retriever.search(question, k=5)
```

Each accepted row exposes the effective `admission_threshold`. Empty admitted output
flows through the existing `RAGEngine` insufficient-evidence response and never reaches
the prompt builder.

## Determinism and malformed inputs

Equal-score rows are ordered by fused score, component scores, and finally stable chunk
identity. The same corpus therefore produces the same ranking when ingestion order
changes. Document-diversity reranking uses the same total order after its penalty.

The boundary rejects duplicate chunk IDs, empty or over-budget queries, unsupported
control characters, non-finite policy values or penalties, and result counts outside
the configured budget. Non-finite computed scores also fail closed.

## Calibration and limitations

TF-IDF similarity is corpus-dependent, so absolute or relative floors are deployment
policy rather than universal relevance thresholds. A positive lexical score does not
prove semantic relevance, factual accuracy, or document authority. Character n-grams
can admit superficially similar identifiers, while synonyms may receive weak scores.

Calibrate floors on held-out queries and inspect coverage alongside retrieval quality;
do not tune them only on successful examples. The next increment should record admitted
and rejected score distributions in shadow mode, then derive per-query-class thresholds
with explicit abstention and recall budgets.
