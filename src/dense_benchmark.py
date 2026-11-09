"""Fixed-setting dense vs sparse SciFact development diagnostic, never a held-out test."""
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

from src.dense import MAX_LENGTH, DenseRetriever, MiniLMEncoder, reciprocal_rank_fusion
from src.hybrid import HybridSparseRetriever
from src.retrieval import SparseRetriever
from src.retrieval_eval import evaluate_rankings
from src.scifact_benchmark import load_scifact, paired_bootstrap


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_bytes(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix.nbytes
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def latency_summary(values):
    return {'p50': float(np.median(values)), 'p95': float(np.quantile(values, .95)),
            'samples_ms': values}


def truncation(tokenizer, texts):
    lengths = [len(ids) for ids in tokenizer(texts, truncation=False, padding=False)['input_ids']]
    return {'inputs': len(lengths), 'over_limit': sum(n > MAX_LENGTH for n in lengths),
            'fraction_over_limit': sum(n > MAX_LENGTH for n in lengths) / len(lengths),
            'max_tokens_before_truncation': max(lengths), 'limit_including_special_tokens': MAX_LENGTH}


def run(data, output, cache, *, threads=1, batch_size=32):
    chunks, cases, claim_ids, excluded = load_scifact(data)
    start = time.perf_counter()
    encoder = MiniLMEncoder(threads=threads, batch_size=batch_size)
    load_seconds = time.perf_counter() - start
    result = {
        'schema': 'scifact-dense-diagnostic/v1', 'encoder': encoder.signature,
        'encoder_load_seconds_including_download_if_uncached': load_seconds,
        'batch_size': batch_size, 'torch_threads': threads,
        'python': platform.python_version(), 'platform': platform.platform(),
        'dependencies': {p: importlib.metadata.version(p) for p in
                         ('numpy', 'scipy', 'scikit-learn', 'torch', 'transformers')},
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty_worktree': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        'code_sha256': {str(p): sha(p) for p in sorted(Path('src').glob('*.py'))},
        'files_sha256': {p: sha(data / p) for p in ('corpus.jsonl', 'claims_dev.jsonl')},
        'documents': len(chunks), 'queries': len(cases), 'excluded_no_evidence_queries': excluded,
        'protocol': 'Exposed SciFact development claims; title + abstract one chunk. No tuning. '
                    'Word and hybrid sparse unchanged. Dense exact normalized cosine. '
                    'RRF word+dense equal weights, constant 60, top 100 candidates each. '
                    'Dense input truncated to 256 tokens; not whole-document semantics. '
                    'Single sequential run, no load/capacity claims. Pretrained data overlap unknown.',
        'truncation': {'corpus': truncation(encoder.tokenizer, [c.text for c in chunks]),
                       'queries': truncation(encoder.tokenizer, [c.query for c in cases])},
        'models': {},
    }
    rankings, latencies, per_case = {}, {}, {}
    for name, retriever in [('word', SparseRetriever()), ('hybrid', HybridSparseRetriever()),
                            ('dense', DenseRetriever(encoder, cache))]:
        start = time.perf_counter()
        retriever.fit(chunks)
        fit_seconds = time.perf_counter() - start
        # Warmup is excluded; all reported queries include encoding and exact search.
        retriever.search(cases[0].query, k=100 if name != 'hybrid' else 10)
        rankings[name], latencies[name] = [], []
        for case in cases:
            start = time.perf_counter()
            rankings[name].append(retriever.search(case.query, k=100 if name != 'hybrid' else 10))
            latencies[name].append((time.perf_counter() - start) * 1000)
        matrices = [retriever.word_matrix, retriever.char_matrix] if name == 'hybrid' else [retriever.matrix]
        result['models'][name] = {'fit_seconds': fit_seconds,
                                  'numeric_index_bytes_excludes_vocab_model_text': sum(map(matrix_bytes, matrices)),
                                  'query_latency_ms': latency_summary(latencies[name])}
        if name == 'dense':
            result['models'][name].update(cache_hit=retriever.cache_hit, cache_key=retriever.cache_key,
                                           vector_sha256=retriever.vector_sha256)
    rankings['rrf'], latencies['rrf'] = [], []
    for i, (word, dense) in enumerate(zip(rankings['word'], rankings['dense'], strict=True)):
        start = time.perf_counter()
        rankings['rrf'].append(reciprocal_rank_fusion(word, dense, k=10))
        latencies['rrf'].append(latencies['word'][i] + latencies['dense'][i]
                                + (time.perf_counter() - start) * 1000)
    result['models']['rrf'] = {
        'query_latency_ms': latency_summary(latencies['rrf']),
        'timing_note': 'Estimated sequential pipeline: measured word + dense + fusion, not a separate rerun',
        'numeric_index_bytes_excludes_vocab_model_text': sum(
            result['models'][n]['numeric_index_bytes_excludes_vocab_model_text'] for n in ('word', 'dense'))}
    for name, rows in rankings.items():
        result['models'][name]['metrics'] = {
            str(k): evaluate_rankings(rows, cases, k=k).to_dict() for k in (1, 5, 10)}
        result['models'][name]['rankings'] = [
            {'claim_id': cid, 'relevant': sorted(case.relevant_doc_ids),
             'retrieved': [r['doc_id'] for r in ranking]}
            for cid, case, ranking in zip(claim_ids, cases, rows, strict=True)]
        per_case[name] = np.array([evaluate_rankings([ranking], [case], k=10).ndcg_at_k
                                  for ranking, case in zip(rows, cases, strict=True)])
        print(name, result['models'][name]['metrics']['10'], flush=True)
    result['paired_ndcg_at_10'] = {name + '_minus_hybrid': paired_bootstrap(per_case[name] - per_case['hybrid'])
                                   for name in ('dense', 'rrf')}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    run(args.data, args.output, args.cache, threads=args.threads, batch_size=args.batch_size)
