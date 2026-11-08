"""Frozen-setting SciFact dev retrieval diagnostic, not a BEIR leaderboard run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from src.hybrid import HybridSparseRetriever
from src.retrieval import Chunk, SparseRetriever
from src.retrieval_eval import RetrievalCase, evaluate_rankings


def load_scifact(data: Path):
    docs = [json.loads(line) for line in (data / 'corpus.jsonl').read_text().splitlines()]
    claims = [json.loads(line) for line in (data / 'claims_dev.jsonl').read_text().splitlines()]
    ids = [str(d['doc_id']) for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate corpus IDs')
    if len({c['id'] for c in claims}) != len(claims):
        raise ValueError('duplicate claim IDs')
    chunks = [Chunk(str(d['doc_id']), str(d['doc_id']), 'SciFact corpus',
                    d['title'] + '\n' + ' '.join(d['abstract'])) for d in docs]
    labelled = [c for c in claims if c['evidence']]
    cases = [RetrievalCase(c['claim'], frozenset(map(str, c['evidence']))) for c in labelled]
    if not cases:
        raise ValueError('no evidence-labelled claims')
    for case in cases:
        if not case.relevant_doc_ids <= set(ids):
            raise ValueError('unknown relevant document')
    return chunks, cases, [c['id'] for c in labelled], len(claims) - len(labelled)


def paired_bootstrap(deltas, *, seed=2026, samples=10000):
    values = np.asarray(deltas, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('finite nonempty deltas required')
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {'mean': float(values.mean()), 'lower_95': float(np.quantile(means, .025)),
            'upper_95': float(np.quantile(means, .975)), 'resamples': samples, 'seed': seed}


def run(data: Path, output: Path):
    chunks, cases, claim_ids, excluded = load_scifact(data)
    result = {
        'schema': 'scifact-retrieval-diagnostic/v1',
        'code_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                        ('src/scifact_benchmark.py', 'src/hybrid.py', 'src/retrieval.py',
                         'src/retrieval_eval.py')},
        'source_url': 'https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz',
        'source_repository': 'https://github.com/allenai/scifact',
        'files_sha256': {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                         for name in ('corpus.jsonl', 'claims_dev.jsonl')},
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty_worktree': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        'python': platform.python_version(), 'platform': platform.platform(),
        'dependencies': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')},
        'protocol': 'All abstracts, title + abstract as one chunk. Evidence-bearing dev claims only; '
                    'supporting and refuting abstracts both relevant. No relevance-label fitting or tuning. '
                    'Word TF-IDF (1,2) vs existing 0.65 word/0.35 char(3,5). k=1,5,10. '
                    'This is not official BEIR SciFact or semantic answer correctness.',
        'documents': len(chunks), 'queries': len(cases), 'excluded_no_evidence_queries': excluded,
        'relevant_pairs': sum(len(c.relevant_doc_ids) for c in cases), 'models': {},
    }
    per_case = {}
    for name, retriever in [('word', SparseRetriever()), ('hybrid', HybridSparseRetriever())]:
        start = time.perf_counter()
        retriever.fit(chunks)
        fit_seconds = time.perf_counter() - start
        rankings, latency = [], []
        for case in cases:
            start = time.perf_counter()
            rankings.append(retriever.search(case.query, k=10))
            latency.append((time.perf_counter() - start) * 1000)
        metrics = {str(k): evaluate_rankings(rankings, cases, k=k).to_dict() for k in (1, 5, 10)}
        per_case[name] = [evaluate_rankings([ranking], [case], k=10).ndcg_at_k
                          for ranking, case in zip(rankings, cases, strict=True)]
        result['models'][name] = {'fit_seconds': fit_seconds, 'metrics': metrics,
                                 'query_latency_ms': {'p50': float(np.median(latency)),
                                                      'p95': float(np.quantile(latency, .95))},
                                 'rankings': [{'claim_id': cid, 'relevant': sorted(case.relevant_doc_ids),
                                               'retrieved': [r['doc_id'] for r in rows]}
                                              for cid, case, rows in zip(claim_ids, cases, rankings, strict=True)]}
        print(name, metrics, flush=True)
    result['paired_ndcg_at_10_hybrid_minus_word'] = paired_bootstrap(
        np.array(per_case['hybrid']) - np.array(per_case['word']))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.output)
