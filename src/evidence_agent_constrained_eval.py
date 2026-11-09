"""Development comparison of a constrained controller with fixed-retrieval/abstain baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from src.evidence_agent import CorpusTools, EvidenceAgent
from src.evidence_agent_constrained import ConstrainedQwenPolicy
from src.evidence_agent_eval import score, select


def run(data, output, manifest, *, split='dev', order_seed=2026, capacity=False):
    protocol = json.loads(manifest.read_text())
    for filename, expected in protocol['data_sha256'].items():
        if hashlib.sha256((data / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('dataset differs from frozen protocol')
    claims = list(map(json.loads, (data / f'claims_{split}.jsonl').read_text().splitlines()))
    chosen = select(claims)
    if split == 'dev':
        by_id = {c['id']: c for c in claims}
        chosen = [by_id[cid] for cid in protocol['claim_ids']]
    else:
        # Two per class: deterministic separate training smoke cases.
        chosen = chosen[:2] + chosen[8:10] + chosen[16:18]
    output.mkdir(parents=True, exist_ok=True)
    frozen = {'schema': 'constrained-agent-dev/v1', 'split': split, 'capacity_1_5b_bfloat16': capacity, 'order_seed': order_seed, 'claim_ids': [c['id'] for c in chosen],
        'prior_exposure': 'Development comparison. V1 results already observed. Not an untouched test or generalization estimate.',
        'selection': 'Same 24 frozen dev cases; train smoke uses first two per class from deterministic selection.',
        'candidate_policy': 'Forced initial search top3; model selects unread doc or verdict with one of lexical top3 sentences per read document; Abstention only after reading at least one source. Last step finish-only. At most 26 one-token letter options.',
        'scoring': 'Greedy argmax raw next-token logit among A..Z single-token IDs. Conditional softmax over legal options logged; not calibrated confidence. Option order deterministically permuted by SHA256(JSON([order_seed, claim, action])); positional sensitivity not measured on dev.',
        'baseline': 'Same model chooses finish-only using all top3 abstracts and same sentence candidates; always-NEI non-LLM baseline.',
        'data_sha256': protocol['data_sha256'],
        'code_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['src/evidence_agent.py', 'src/evidence_agent_constrained.py', 'src/evidence_agent_constrained_eval.py', 'src/evidence_agent_capacity.py']}}
    frozen_path = output / 'protocol.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text()) != frozen:
        raise ValueError('existing output protocol differs; use a new directory')
    frozen_path.write_text(json.dumps(frozen, indent=2) + '\n')
    tools = CorpusTools(list(map(json.loads, (data / 'corpus.jsonl').read_text().splitlines())))
    if capacity:
        from src.evidence_agent_capacity import CapacityQwenPolicy
        policy_class = CapacityQwenPolicy
    else:
        policy_class = ConstrainedQwenPolicy
    policy = policy_class(audit_path=output / 'selection-audit.jsonl', order_seed=order_seed)
    base_identity = policy.identity.replace(';finish_only=False', '')
    results = {'protocol': frozen, 'policy': base_identity, 'cases': []}
    for claim in chosen:
        pair = {}
        for mode in ('fixed_top3', 'agent'):
            policy.finish_only = mode == 'fixed_top3'
            policy.identity = base_identity + f';finish_only={policy.finish_only};mode={mode}'
            policy.audit_path = output / f"{mode}-{claim['id']}-policy.jsonl"
            agent = EvidenceAgent(output / 'runs.sqlite', tools, policy, max_steps=5 if mode == 'fixed_top3' else 6)
            run_id = f"{mode}:{claim['id']}"
            if mode == 'fixed_top3' and agent.load(run_id) is None:
                found = tools.search(claim['claim'])
                reads = {r['doc_id']: tools.read(r['doc_id']) for r in found}
                agent._save({'id': run_id, 'claim': claim['claim'], 'binding': agent.binding, 'steps': 4,
                    'trace': [{'action': {'tool': 'search', 'query': claim['claim']}, 'result': found}]
                    + [{'action': {'tool': 'read', 'doc_id': k}, 'result': v} for k, v in reads.items()],
                    'seen': sorted(reads), 'read': reads, 'pending': None, 'status': 'running', 'answer': None})
            started = time.perf_counter()
            state = agent.run(run_id, claim['claim'])
            measured = score(state, claim)
            gold_pairs = {(str(doc_id), sentence) for doc_id, items in claim['evidence'].items()
                          for item in items for sentence in item['sentences']}
            cited = set(map(tuple, measured['citations']))
            measured['gold_citation_hits'] = len(cited & gold_pairs)
            measured['citation_count'] = len(cited)
            measured['wall_seconds_this_invocation'] = time.perf_counter() - started
            audit = [json.loads(line) for line in policy.audit_path.read_text().splitlines()] if policy.audit_path.exists() else []
            measured['model_forward_passes'] = sum(r['kind'] == 'model_selection' for r in audit)
            pair[mode] = measured
            (output / f"{mode}-{claim['id']}.json").write_text(json.dumps(state, indent=2) + '\n')
            print(split, mode, claim['id'], measured['predicted'], measured['label_correct'], flush=True)
        pair['always_abstain'] = {'label_correct': pair['agent']['gold'] == 'NOT_ENOUGH_INFO',
                                 'evidence_complete_correct': pair['agent']['gold'] == 'NOT_ENOUGH_INFO'}
        results['cases'].append(pair)
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    n = len(chosen)
    summary = {}
    for mode in ('fixed_top3', 'agent'):
        total = {key: sum(pair[mode][key] for pair in results['cases']) for key in
                 ['label_correct', 'evidence_complete_correct', 'invalid_or_failed_steps',
                  'generation_seconds', 'model_forward_passes', 'gold_citation_hits', 'citation_count']}
        total['n'] = n
        total['gold_sentence_citation_precision'] = total['gold_citation_hits'] / total['citation_count'] if total['citation_count'] else None
        total['completed'] = sum(pair[mode]['status'] == 'completed' for pair in results['cases'])
        total['abstentions'] = sum(pair[mode]['predicted'] == 'NOT_ENOUGH_INFO' for pair in results['cases'])
        summary[mode] = total
    summary['always_abstain'] = {'n': n, 'label_correct': sum(p['always_abstain']['label_correct'] for p in results['cases'])}
    results['summary'] = summary
    results['paired_label_outcomes'] = {name: sum(bool(p['agent']['label_correct']) == a
        and bool(p['fixed_top3']['label_correct']) == b for p in results['cases'])
        for name, a, b in [('both_correct', True, True), ('agent_only', True, False),
                           ('baseline_only', False, True), ('both_wrong', False, False)]}
    (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'dev'], default='dev')
    parser.add_argument('--order-seed', type=int, default=2026)
    parser.add_argument('--capacity', action='store_true', help='Pinned official 1.5B BF16 model; unchanged action policy')
    args = parser.parse_args()
    run(args.data, args.output, args.manifest, split=args.split, order_seed=args.order_seed, capacity=args.capacity)
