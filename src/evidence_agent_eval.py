"""Paired small public-dev diagnostic; never a claim of production answer quality."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from src.evidence_agent import CorpusTools, EvidenceAgent, LocalQwenPolicy, digest


def labels(claim):
    return {e['label'] for evidence in claim['evidence'].values() for e in evidence}


def select(claims):
    groups = {'SUPPORT': [], 'CONTRADICT': [], 'NOT_ENOUGH_INFO': []}
    for c in claims:
        target = labels(c)
        if len(target) > 1:
            continue
        groups[next(iter(target), 'NOT_ENOUGH_INFO')].append(c)
    return [c for group in groups.values()
            for c in sorted(group, key=lambda c: digest(['agent-dev-v1', c['id']]))[:8]]


def freeze(data, output):
    claims = [json.loads(s) for s in (data / 'claims_dev.jsonl').read_text().splitlines()]
    chosen = select(claims)
    manifest = {'schema': 'agent-dev-protocol/v1', 'claim_ids': [c['id'] for c in chosen],
        'selection': '8 per gold class, ordered by SHA256(JSON([agent-dev-v1, claim_id])); mixed-label claims excluded',
        'prior_exposure': 'SciFact dev was used by earlier retrieval evaluation. This is NOT an untouched test set. Selection frozen before agent generation, no prompt tuning on selected answers.',
        'agent': '6 model actions max; search top3 word TF-IDF; read; finish',
        'baseline': 'Fixed claim search top3 plus all three abstracts, then one same-model finish attempt. Stronger evidence availability, fewer model calls. Not equal compute.',
        'metrics': 'All 24 denominators include malformed, exhausted and failed runs as incorrect. Separate label accuracy, evidence-complete accuracy, abstention, invalid actions, tool errors, calls and latency. NEI evidence-complete means correct abstention, not proof that corpus lacks evidence.',
        'data_sha256': {p: __import__('hashlib').sha256((data / p).read_bytes()).hexdigest()
                        for p in ['corpus.jsonl', 'claims_dev.jsonl']}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def score(state, claim):
    answer = state['answer']
    gold = next(iter(labels(claim)), 'NOT_ENOUGH_INFO')
    correct = bool(answer and answer['verdict'] == gold)
    cited = {(c['doc_id'], c['sentence']) for c in answer['citations']} if answer else set()
    gold_sets = [{(str(doc), sentence) for sentence in item['sentences']}
                 for doc, items in claim['evidence'].items() for item in items if item['label'] == gold]
    complete = correct and (gold == 'NOT_ENOUGH_INFO' or any(g <= cited for g in gold_sets))
    return {'claim_id': claim['id'], 'gold': gold, 'predicted': answer['verdict'] if answer else None,
        'label_correct': correct, 'evidence_complete_correct': complete,
        'status': state['status'], 'model_calls': sum('raw' in t for t in state['trace']),
        'invalid_or_failed_steps': sum('error' in t for t in state['trace']),
        'generation_seconds': sum(t.get('generation_seconds', 0) for t in state['trace']),
        'citations': sorted(cited), 'read_documents': sorted(state['read'])}


def run(data, output, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    for filename, expected in manifest['data_sha256'].items():
        import hashlib
        if hashlib.sha256((data / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('dataset differs from frozen protocol')
    claims = {c['id']: c for c in map(json.loads, (data / 'claims_dev.jsonl').read_text().splitlines())}
    tools = CorpusTools(list(map(json.loads, (data / 'corpus.jsonl').read_text().splitlines())))
    policy = LocalQwenPolicy()
    output.mkdir(parents=True, exist_ok=True)
    results = {'protocol': manifest, 'policy': policy.identity, 'python': platform.python_version(),
        'dependencies': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'scikit-learn']},
        'code_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                       ['src/evidence_agent.py', 'src/evidence_agent_eval.py']}, 'cases': []}
    for cid in manifest['claim_ids']:
        claim = claims[cid]
        pair = {}
        for mode in ('fixed_top3', 'agent'):
            agent = EvidenceAgent(output / 'runs.sqlite', tools, policy, max_steps=6 if mode == 'agent' else 5)
            run_id = f'{mode}:{cid}'
            if mode == 'fixed_top3' and agent.load(run_id) is None:
                found = tools.search(claim['claim'])
                reads = {r['doc_id']: tools.read(r['doc_id']) for r in found}
                state = {'id': run_id, 'claim': claim['claim'], 'binding': agent.binding,
                    'steps': 4, 'trace': [{'action': {'tool': 'search', 'query': claim['claim']}, 'result': found}]
                    + [{'action': {'tool': 'read', 'doc_id': k}, 'result': v} for k, v in reads.items()],
                    'seen': sorted(reads), 'read': reads, 'pending': None, 'status': 'running', 'answer': None}
                agent._save(state)
            start = time.perf_counter()
            state = agent.run(run_id, claim['claim'])
            pair[mode] = score(state, claim)
            pair[mode]['wall_seconds_this_invocation'] = time.perf_counter() - start
            (output / f'{mode}-{cid}.json').write_text(json.dumps(state, indent=2) + '\n')
            print(mode, cid, pair[mode]['predicted'], pair[mode]['label_correct'], flush=True)
        results['cases'].append(pair)
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    n = len(results['cases'])
    results['summary'] = {mode: {'n': n, **{metric: sum(p[mode][metric] for p in results['cases'])
        for metric in ['label_correct', 'evidence_complete_correct', 'model_calls', 'invalid_or_failed_steps', 'generation_seconds']},
        'completed': sum(p[mode]['status'] == 'completed' for p in results['cases']),
        'abstentions': sum(p[mode]['predicted'] == 'NOT_ENOUGH_INFO' for p in results['cases'])}
        for mode in ['fixed_top3', 'agent']}
    results['paired_label_outcomes'] = {name: sum(bool(p['agent']['label_correct']) == a
        and bool(p['fixed_top3']['label_correct']) == b for p in results['cases'])
        for name, a, b in [('both_correct', True, True), ('agent_only', True, False),
                           ('baseline_only', False, True), ('both_wrong', False, False)]}
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results['summary'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze(args.data, args.manifest)
    elif args.output:
        run(args.data, args.output, args.manifest)
    else:
        parser.error('--output required unless --freeze')
