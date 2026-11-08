# SciFact retrieval diagnostic

This independently labelled external corpus replaces the previous three-query demo as an evaluation of retrieval. It does not evaluate generated answers, semantic entailment, real users, employer work, or production deployment.

## Frozen protocol
- Source: Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (EMNLP 2020), https://arxiv.org/abs/2004.14974; official https://github.com/allenai/scifact.
- Download: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz. Because `latest` is mutable, SHA-256 hashes of exact input files are in the result manifest.
- All 5,183 scientific abstracts indexed; one title-plus-abstract chunk per document, no prefiltering by labels.
- Only the 188 evidence-bearing dev claims are scored; 112 claims with no gold evidence are excluded, not treated as successful abstentions. Gold keys in `evidence` are relevant, including refuting evidence. This is deliberately not the official BEIR split or leaderboard protocol.
- Existing settings fixed before measurement: word TF-IDF unigrams/bigrams with English stop words; hybrid 0.65 word + 0.35 character 3–5 grams. No training on relevance labels and no tuning after observing outcomes.
- Report Hit/Recall/Precision/MRR/nDCG at K=1,5,10 and all per-query rankings. Macro averages; each relevant document credited only once.
- Paired bootstrap compares per-query nDCG@10 differences (10,000 resamples, seed 2026, percentile 95% interval). It assumes sampled queries are independent; related scientific claims may violate this. It is descriptive uncertainty, not a universal superiority guarantee.
- Timing is one sequential local run per retriever with the index already fit, not a production load test; shared-host timings are not capacity claims.

## Changes
- Reused existing `fix/document-level-retrieval-metrics` (4ab61dc) instead of duplicating its document-credit fix. This is an unmerged stacked feature branch, NOT current main fd098bc.
- Fixed a separate early-stop defect: the candidate list shrank during selection, causing search to return only half the documents when K approached corpus size. The loop now stops at requested K or exhaustion.
- Added a dataset validator, standalone benchmark, paired uncertainty, manifests and six regression cases.

## Reproduce
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o /tmp/scifact-data.tar.gz
mkdir -p /tmp/scifact
tar -xzf /tmp/scifact-data.tar.gz -C /tmp/scifact
.venv/bin/python -m src.scifact_benchmark --data /tmp/scifact/data --output evidence/scifact-dev.json
.venv/bin/pytest -q
.venv/bin/ruff check src app tests
.venv/bin/pip check
```

The dataset is not redistributed in this patch. SciFact claims/evidence use CC BY 4.0 and abstracts ODC-By 1.0; see https://github.com/allenai/scifact/blob/master/LICENSE.md. Result artifacts contain IDs and scores only, no abstract text.

## Validation limitations
Docker is not installed in this executor, so container build and Compose validation were not run. Pytest has a Starlette/AnyIO deprecation warning. Passing contract tests is not proof of model accuracy. No merge, deployment, external provider calls, or paid model execution occurred.

## Observed results

At K=10, word-only Recall=0.8719 and nDCG=0.7307; hybrid Recall=0.8996 and nDCG=0.7721. The paired nDCG difference was +0.0414 with a percentile-bootstrap interval [0.0204, 0.0639]. This is an absolute nDCG gain, not a 4.14% gain in answer accuracy. In the final recorded run, hybrid p95 retrieval time was 458 ms versus 74 ms for word-only on this shared executor. An earlier run measured approximately 220 ms versus 22 ms, illustrating shared-host timing variability; ranking metrics reproduced exactly. The higher-quality ranking costs substantially more CPU latency.

CV wording: “Benchmarked hybrid sparse retrieval against a word-only baseline on 5,183 SciFact abstracts and 188 evidence-labelled development queries, achieving 0.900 Recall@10 and 0.772 nDCG@10 versus 0.872/0.731; added reproducible input manifests, paired-bootstrap analysis and retrieval regression tests.”

Do not describe these as new neural-model improvements, BEIR leaderboard results, semantic factuality, or live production performance. The hybrid algorithm already existed; this work establishes external evaluation evidence and fixes top-K completeness. Test suite result: 307 passed, one dependency deprecation warning.
