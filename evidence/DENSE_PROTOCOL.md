# Dense retrieval development diagnostic

This extends the existing SciFact evaluation, not the separate agentic generation pipeline. It is a development experiment on an already exposed split, not a held-out test or official BEIR leaderboard result. No settings are selected after observing these results.

## Frozen settings

- Same 5,183 title-plus-abstract chunks and 188 evidence-bearing SciFact development claims as `PROTOCOL.md`. Supporting and refuting documents both count as relevant; no-evidence claims are excluded.
- Existing word TF-IDF and weighted word/character sparse implementations remain unchanged.
- Local encoder: `sentence-transformers/all-MiniLM-L6-v2`, pinned revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`, Apache-2.0. Official source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/c9745ed1d9f207416be6d2e6f8de32d1f16199bf.
- Official `sentence_bert_config.json`, `1_Pooling/config.json` and `modules.json` were inspected: 256-token maximum, attention-mask mean pooling, 384 dimensions and L2 normalization. Uses official tokenizer, CPU float32 safetensors, no remote code, batch size 32, one Torch thread, fixed seed and deterministic algorithms. No training or paid API.
- Queries and documents use the same encoder without prefixes. Long abstracts are truncated, not chunked or averaged across passages. The manifest reports truncation counts/rates from full tokenization; a tokenizer warning on the length census does not mean oversized sequences are passed to the model.
- Dense search is exact normalized dot product. Dense ties resolve by document ID then chunk ID; existing sparse tie behavior is retained for baseline parity.
- Equal-weight reciprocal rank fusion of top 100 word and top 100 dense results, constant 60, document-level duplicate suppression, document-ID tie breaking. No weight sweep, reranker or prompt optimization.
- Metrics at K=1,5,10 plus per-query rankings and paired 10,000-resample nDCG@10 bootstrap differences against hybrid sparse. These descriptive intervals do not establish universal superiority or correct generated answers. Pretraining overlap with SciFact is unknown.

## Runtime and provenance

The corpus vector cache key hashes complete ordered chunk content/metadata and encoder settings; `.npy` payload SHA-256 and vector shape/norm are verified on load. Query embeddings are not cached. Input/code hashes and dependency versions are included in the manifest. Cache files and model weights are not committed.

Index bytes count numeric arrays only, excluding vocabulary, text, Python overhead and model weights, so this is not resident memory or complete disk footprint. Index building includes encoding unless the manifest says `cache_hit: true`. Model initialization/download is separately timed. Search latency includes query encoding and exhaustive scoring; one warmup query is excluded. RRF latency is the sum of measured component timings plus measured fusion, explicitly an estimated sequential pipeline rather than an independently rerun end-to-end measurement. These single-run shared-host measurements are not production capacity or SLA claims.

## Reproduce

Install normal `requirements.txt` plus `requirements-dense.txt` in a virtual environment. From the repository root, with the SciFact files acquired as in `PROTOCOL.md`:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.dense_benchmark \
  --data /tmp/scifact/data --output evidence/scifact-dense-dev.json \
  --cache /tmp/scifact-dense-cache --threads 1 --batch-size 32
python -m pytest -q
ruff check src app tests
```

First run downloads public pinned model artifacts. No sign-in is required. Do not interpret offline fake-encoder unit tests as evidence of semantic retrieval performance; the separate real-model manifest provides that evidence.

## Observed results and validation

On the 188 exposed development queries, Recall@10 / nDCG@10 were:

- Word sparse: 0.8719 / 0.7307.
- Word/character sparse: 0.8996 / 0.7721.
- MiniLM dense: 0.9117 / 0.7743.
- Word+dense RRF: 0.9426 / 0.7958.

The RRF-minus-hybrid nDCG difference is +0.0237, with paired percentile-bootstrap 95% interval [-0.0057, 0.0539]. Dense-minus-hybrid is +0.0022, interval [-0.0455, 0.0516]. Both intervals include zero: neither is established as superior on this small diagnostic. No claim of generalization or answer accuracy follows. 3,681 abstracts (71.0%) exceeded the 256-token cap; no query did. This limitation is material.

Dense fit/cache-build took 418.9 seconds, including encoding, normalization and cache serialization (not isolated encoder time). Numeric vector storage was 7,961,088 bytes (7.59 MiB), excluding the model and other overhead. Single-run warm-search p95 was 18.9 ms word, 170.1 ms hybrid and 25.5 ms dense. Estimated sequential RRF p95 was 41.9 ms. Word and dense timings retrieve 100 candidates for fusion; hybrid retrieves 10. These are local measurements with one thread, not production latency promises.

After timing, validation was hardened to reject overflowing normalization, wrong embedding dimensions and mismatched cache signatures. Valid-vector arithmetic was unchanged. The exact measured dense source is preserved in `dense-measured-source.py.txt`, and the final-code replay command is in `dense-replay-command.py.txt`. The original measured code hashes are retained in `scifact-dense-dev.json`; final-code replay reproduced all 188 dense top-100 and RRF top-10 rankings exactly, recorded in `dense-postrun-validation.json`. Both sparse baselines reproduce all prior metrics and top-10 rankings exactly on matching input hashes. Timing refers to the pre-hardening run, not the validation replay.

Final full suite: 323 passed, including 16 new cases. Repository-pinned Ruff 0.12.11 passes across src/app/tests, and dependency compatibility checks pass. The shared environment’s newer Ruff 0.17.4 reports 38 pre-existing baseline violations, identically in the unchanged source; the new dense files pass that checker too. No Docker validation or deployed service test was performed. The benchmark is optional and does not silently replace the application's sparse retriever.

Suggested CV wording: “Added pinned local MiniLM embeddings and reciprocal-rank fusion to a reproducible SciFact retrieval benchmark across 5,183 abstracts and 188 development queries; measured 0.943 Recall@10 and 0.796 nDCG@10 for fusion, with cache-integrity checks, sparse-baseline parity and paired-bootstrap uncertainty.” Keep the truncation and non-significant paired interval available in the linked evidence; do not claim a proven uplift.

## Public dataset attribution and packaged evidence

These diagnostic runs use [SciFact](https://github.com/allenai/scifact),
introduced by Wadden et al. (2020), [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/).
SciFact claims and evidence annotations are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); its abstracts originate
from [Semantic Scholar S2ORC](https://github.com/allenai/s2orc) and are licensed
under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/), as documented in
the [upstream dataset license](https://github.com/allenai/scifact/blob/master/LICENSE.md).
The public benchmark output retains claim/document identifiers and derived rankings
for audit; source data retains its original licenses. MiniLM model attribution and
Apache-2.0 licensing above remain unchanged.

Run `python tools/restore_evidence.py` before reading or auditing
`evidence/scifact-dense-dev.json`. Its UTF-8 chunk packaging preserves every original
byte, metric and recorded source hash. See [packaging details](../docs/evidence-packaging.md).
The 323-test result above describes the original measured-source validation;
packaging adds 15 independent integrity/safety tests, without rerunning the experiment.
