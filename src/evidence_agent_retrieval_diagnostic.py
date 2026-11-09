"""Retrieval-only integration diagnostic on the frozen agent cases; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path

from src.hybrid import HybridSparseRetriever
from src.retrieval import Chunk, SparseRetriever
from src.retrieval_eval import RetrievalCase, evaluate_rankings


def run(data, manifest, output):
    protocol = json.loads(manifest.read_text())
    for filename, expected in protocol['data_sha256'].items():
        if hashlib.sha256((data / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('dataset differs from frozen agent protocol')
    documents = list(map(json.loads, (data / 'corpus.jsonl').read_text().splitlines()))
    claims = {c['id']: c for c in map(json.loads, (data / 'claims_dev.jsonl').read_text().splitlines())}
    selected = [claims[cid] for cid in protocol['claim_ids']]
    chunks = [Chunk(str(d['doc_id']), str(d['doc_id']), 'SciFact',
                    d['title'] + ' ' + ' '.join(d['abstract'])) for d in documents]
    result = {'schema': 'agent-retrieval-integration/v1', 'claim_ids': protocol['claim_ids'],
        'scope': 'Post-hoc retrieval-only diagnostic. No model generation, no policy changes, no new heldout claim.',
        'data_sha256': protocol['data_sha256'],
        'code_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['src/evidence_agent_retrieval_diagnostic.py', 'src/retrieval.py', 'src/hybrid.py', 'src/retrieval_eval.py']},
        'settings': 'Same title + space + joined abstract as agent. One chunk/document. Full corpus. Word TF-IDF vs existing hybrid 0.65 word/0.35 char(3,5); default diversity penalty irrelevant with unique documents.',
        'denominator': 'All24 queries have recorded rankings. Retrieval metrics use16 evidence-bearing cases;8 NEI cases lack gold documents and are excluded from retrieval metrics only.',
        'models': {}}
    for name, index in [('word', SparseRetriever()), ('hybrid', HybridSparseRetriever())]:
        index.fit(chunks)
        rows, rankings, cases = [], [], []
        for claim in selected:
            found = index.search(claim['claim'], k=10)
            gold = frozenset(map(str, claim['evidence']))
            rows.append({'claim_id': claim['id'], 'gold_docs': sorted(gold),
                         'retrieved': [r['doc_id'] for r in found]})
            if gold:
                rankings.append(found)
                cases.append(RetrievalCase(claim['claim'], gold))
        metrics = {str(k): evaluate_rankings(rankings, cases, k=k).to_dict() for k in [3, 10]}
        hits = {str(k): sum(bool(set(row['retrieved'][:k]) & set(row['gold_docs']))
                           for row in rows if row['gold_docs']) for k in [3, 10]}
        result['models'][name] = {'metrics': metrics, 'cases_with_any_gold_doc': hits, 'rankings': rows}
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {'metrics': v['metrics'], 'cases_with_any_gold_doc': v['cases_with_any_gold_doc']}
                      for name, v in result['models'].items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.manifest, args.output)
