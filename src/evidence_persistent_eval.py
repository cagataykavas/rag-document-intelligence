"""Frozen three-way direct/fresh-MCP/reused-MCP lifecycle comparison.

First reused-session initialization/index construction is inside the first task's
timer. One-time batch teardown is measured and added to that mode's total.
"""
import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

from src.evidence_agent import CorpusTools, EvidenceAgent
from src.evidence_agent_capacity import CapacityQwenPolicy
from src.evidence_agent_eval import score
from src.evidence_integrated_eval import recovery, write
from src.evidence_mcp_client import MCPCorpusTools
from src.evidence_mcp_persistent import PersistentMCPCorpusTools
from src.evidence_telemetry import file_telemetry


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    for filename, expected in manifest['data_sha256'].items():
        if hashlib.sha256((args.data / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('dataset differs from frozen protocol')
    if args.stage in ('compare', 'interrupt') and (output / 'runs.sqlite').exists():
        raise ValueError('fresh output directory required')
    direct = CorpusTools(list(map(json.loads, (args.data / 'corpus.jsonl').read_text().splitlines())))
    telemetry = file_telemetry(output / f'{args.stage}-spans.jsonl')
    fresh = MCPCorpusTools(args.data / 'corpus.jsonl', direct.fingerprint, telemetry=telemetry)
    reused = PersistentMCPCorpusTools(args.data / 'corpus.jsonl', direct.fingerprint, telemetry=telemetry)
    protocol = {'stage': args.stage, 'claim_ids': manifest['claim_ids'],
        'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
        'dtype': 'bfloat16', 'retrieval': 'unchanged word TF-IDF top3', 'order_seed': 2026,
        'comparison': 'Only MCP session lifecycle differs. Same model/policy/prompt/retrieval/budget. Previously exposed public dev; no answer improvement claim.',
        'mode_order': 'Direct first; fresh then reused on even case indices, reused then fresh on odd indices. Declared before execution, not randomized.',
        'timing': 'Actual agent.run wall time. Common operator corpus index and model load excluded from every mode. Lazy reused-session initialization/index included in first task. Final close measured separately and added. Concurrent host work uncontrolled.',
        'corpus_fingerprint': direct.fingerprint, 'data_sha256': manifest['data_sha256'],
        'packages': {p: version(p) for p in ['torch', 'transformers', 'mcp', 'opentelemetry-sdk', 'anyio', 'pydantic', 'scikit-learn']},
        'python': platform.python_version(),
        'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').glob('evidence*.py')}}
    write(output / f'{args.stage}-protocol.json', protocol)
    policy = CapacityQwenPolicy(audit_path=output / 'policy.jsonl')
    policy.telemetry = telemetry
    results = {'protocol': protocol, 'cases': []}
    teardown = 0.0
    try:
        if args.stage != 'compare':
            return recovery(args, policy, reused, telemetry)
        by_id = {c['id']: c for c in map(json.loads, (args.data / 'claims_dev.jsonl').read_text().splitlines())}
        for index, cid in enumerate(manifest['claim_ids']):
            claim = by_id[cid]
            transport_modes = [('fresh', fresh), ('reused', reused)]
            if index % 2:
                transport_modes.reverse()
            states, audits, measurements = {}, {}, {}
            for mode, tools in [('direct', direct)] + transport_modes:
                policy.audit_path = output / f'{mode}-{cid}-policy.jsonl'
                agent = EvidenceAgent(output / 'runs.sqlite', tools, policy, telemetry=telemetry)
                start = time.perf_counter()
                state = agent.run(f'{mode}:{cid}', claim['claim'])
                elapsed = time.perf_counter() - start
                states[mode] = state
                write(output / f'{mode}-{cid}.json', state)
                audits[mode] = list(map(json.loads, policy.audit_path.read_text().splitlines()))
                measured = score(state, claim)
                measured['wall_seconds'] = elapsed
                measured['model_forward_passes'] = sum(a['kind'] == 'model_selection' for a in audits[mode])
                measurements[mode] = measured
            def canonical(state):
                return [{k: v for k, v in e.items() if k != 'generation_seconds'} for e in state['trace']]
            checks = {mode: {
                'same_answer': states['direct']['answer'] == states[mode]['answer'],
                'same_trajectory_prompts_results': canonical(states['direct']) == canonical(states[mode]),
                'same_policy_audit': audits['direct'] == audits[mode]}
                for mode in ['fresh', 'reused']}
            results['cases'].append({'claim_id': cid, 'checks': checks, 'measurements': measurements})
            write(output / 'results.json', results)
            print(cid, checks, flush=True)
    finally:
        try:
            start = time.perf_counter()
            reused.close()
            teardown = time.perf_counter() - start
            write(output / f'{args.stage}-lifecycle.json', {
                'reused_connections_started': reused.connections_started,
                'reused_batch_close_seconds': teardown,
                'owner_thread_stopped': reused._thread is None or not reused._thread.is_alive()})
        finally:
            telemetry.shutdown()
    summary = {mode: {key: sum(c['measurements'][mode][key] for c in results['cases'])
                    for key in ['label_correct', 'evidence_complete_correct', 'invalid_or_failed_steps', 'model_forward_passes', 'wall_seconds']}
               for mode in ['direct', 'fresh', 'reused']}
    for mode in summary:
        summary[mode]['batch_close_seconds'] = teardown if mode == 'reused' else 0.0
        summary[mode]['wall_seconds_including_close'] = summary[mode]['wall_seconds'] + summary[mode]['batch_close_seconds']
    summary.update({'n': len(results['cases']),
        'all_transport_parity_checks': all(all(c['checks'][m].values()) for c in results['cases'] for m in ['fresh', 'reused'])})
    results['summary'] = summary
    write(output / 'results.json', results)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('evidence/agent-protocol.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['compare', 'interrupt', 'resume'], default='compare')
    run(parser.parse_args())
