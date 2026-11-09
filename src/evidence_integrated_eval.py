"""Frozen direct/MCP parity and two-process recovery with the actual pinned model."""
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
from src.evidence_mcp_client import MCPCorpusTools
from src.evidence_telemetry import file_telemetry


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


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
    mcp = MCPCorpusTools(args.data / 'corpus.jsonl', direct.fingerprint, telemetry=telemetry)
    protocol = {'stage': args.stage, 'claim_ids': manifest['claim_ids'],
        'model': 'Qwen/Qwen2.5-1.5B-Instruct', 'revision': '989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
        'retrieval': 'unchanged word TF-IDF top3', 'order_seed': 2026,
        'comparison': 'Transport only. Same policy, prompts, source results, budget, dtype, model. Public dev, already exposed; no new accuracy claim.',
        'corpus_fingerprint': direct.fingerprint, 'data_sha256': manifest['data_sha256'],
        'packages': {p: version(p) for p in ['torch', 'transformers', 'mcp', 'opentelemetry-sdk', 'anyio', 'pydantic', 'scikit-learn']},
        'python': platform.python_version(),
        'code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').glob('evidence*.py')}}
    write(output / f'{args.stage}-protocol.json', protocol)
    policy = CapacityQwenPolicy(audit_path=output / 'policy.jsonl')
    policy.telemetry = telemetry
    try:
        if args.stage != 'compare':
            return recovery(args, policy, mcp, telemetry)
        by_id = {c['id']: c for c in map(json.loads, (args.data / 'claims_dev.jsonl').read_text().splitlines())}
        cases = []
        for cid in manifest['claim_ids']:
            claim = by_id[cid]
            states, audits, measured = {}, {}, {}
            for mode, tools in [('direct', direct), ('mcp', mcp)]:
                policy.audit_path = output / f'{mode}-{cid}-policy.jsonl'
                start = time.perf_counter()
                state = EvidenceAgent(output / 'runs.sqlite', tools, policy, telemetry=telemetry).run(f'{mode}:{cid}', claim['claim'])
                states[mode] = state
                write(output / f'{mode}-{cid}.json', state)
                audits[mode] = list(map(json.loads, policy.audit_path.read_text().splitlines()))
                measured[mode] = score(state, claim)
                measured[mode]['wall_seconds'] = time.perf_counter() - start
                measured[mode]['model_forward_passes'] = sum(a['kind'] == 'model_selection' for a in audits[mode])
            def canonical(state):
                return [{k: v for k, v in e.items() if k != 'generation_seconds'} for e in state['trace']]
            checks = {'same_answer': states['direct']['answer'] == states['mcp']['answer'],
                'same_trajectory_prompts_results': canonical(states['direct']) == canonical(states['mcp']),
                'same_policy_audit': audits['direct'] == audits['mcp']}
            cases.append({'claim_id': cid, 'checks': checks, 'measurements': measured})
            write(output / 'results.json', {'protocol': protocol, 'cases': cases})
            print(cid, checks, flush=True)
        summary = {mode: {key: sum(c['measurements'][mode][key] for c in cases)
                    for key in ['label_correct', 'evidence_complete_correct', 'invalid_or_failed_steps', 'model_forward_passes', 'wall_seconds']}
                   for mode in ['direct', 'mcp']}
        summary.update({'n': len(cases), 'all_transport_parity_checks': all(all(c['checks'].values()) for c in cases)})
        write(output / 'results.json', {'protocol': protocol, 'summary': summary, 'cases': cases})
        print(json.dumps(summary), flush=True)
    finally:
        telemetry.shutdown()


def recovery(args, policy, tools, telemetry):
    claim = next(c for c in map(json.loads, (args.data / 'claims_train.jsonl').read_text().splitlines()) if c['id'] == 997)
    agent = EvidenceAgent(args.output / 'runs.sqlite', tools, policy, telemetry=telemetry)
    if args.stage == 'interrupt':
        def stop():
            state = agent.load('restart')
            if state['pending']['tool'] == 'read':
                write(args.output / 'before.json', state)
                raise SystemExit(75)
        agent.run('restart', claim['claim'], after_pending=stop)
        raise RuntimeError('no pending read reached')
    before = json.loads((args.output / 'before.json').read_text())
    audit_before = len(policy.audit_path.read_text().splitlines())
    after = agent.run('restart', claim['claim'])
    write(args.output / 'after.json', after)
    index = len(before['trace']) - 1
    checks = {'pending_read': before['pending']['tool'] == 'read',
        'same_raw': before['trace'][index]['raw'] == after['trace'][index]['raw'],
        'same_generation_time': before['trace'][index]['generation_seconds'] == after['trace'][index]['generation_seconds'],
        'read_completed': after['trace'][index]['result']['doc_id'] == before['pending']['doc_id'],
        'completed': after['status'] == 'completed',
        'no_repeated_policy_decision': len(policy.audit_path.read_text().splitlines()) - audit_before == after['steps'] - before['steps'],
        'same_trace_id': before['trace_context']['trace_id'] == after['trace_context']['trace_id']}
    write(args.output / 'report.json', {'checks': checks, 'passed': all(checks.values()),
        'scope': 'Two separate Python processes; controlled exit75 after persisted real-model read decision; MCP execution on resume. Read-only at-least-once replay, not side-effect exactly-once.'})
    if not all(checks.values()):
        raise RuntimeError('recovery checks failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=Path('evidence/agent-protocol.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['compare', 'interrupt', 'resume'], default='compare')
    run(parser.parse_args())
