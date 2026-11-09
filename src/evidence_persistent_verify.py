"""Audit lifecycle comparison, exported SDK spans, timing and restarted ownership."""
import argparse
import hashlib
import json
from pathlib import Path

from src.evidence_trace_verify import audit, read


def run(root):
    results = json.loads((root / 'comparison/results.json').read_text())
    lifecycle = json.loads((root / 'comparison/compare-lifecycle.json').read_text())
    spans = read([root / 'comparison/compare-spans.jsonl'])
    restart = read([root / 'restart/interrupt-spans.jsonl', root / 'restart/resume-spans.jsonl'])
    report = {'comparison': audit(spans), 'restart': audit(restart)}
    n = len(results['cases'])
    tasks = [s for s in restart if s['name'] == 'agent.task']
    forward_count = sum(c['measurements'][mode]['model_forward_passes'] for c in results['cases'] for mode in ['direct', 'fresh', 'reused'])
    by_trace = {}
    for c in results['cases']:
        state = json.loads((root / 'comparison' / f"reused-{c['claim_id']}.json").read_text())
        trace_id = '0x' + state['trace_context']['trace_id']
        by_trace[c['claim_id']] = [s for s in spans if s['context']['trace_id'] == trace_id]
    connect = [s for s in spans if s['name'] == 'mcp.connect']
    checks = {
        'all_frozen_cases_retained': [c['claim_id'] for c in results['cases']] == results['protocol']['claim_ids'] and len(set(results['protocol']['claim_ids'])) == n,
        'measured_sources_match_checkout': all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in results['protocol']['code_sha256'].items()),
        'all_three_path_parity_checks': results['summary']['all_transport_parity_checks'],
        'all_72_tasks_exported': report['comparison']['span_counts'].get('agent.task') == 3 * n,
        'actual_model_forward_count_matches': report['comparison']['span_counts'].get('model.forward') == forward_count,
        'one_reused_connection_for_batch': lifecycle['reused_connections_started'] == 1 and len(connect) == 1,
        'one_reused_batch_close': report['comparison']['span_counts'].get('mcp.close') == 1,
        'owner_thread_stopped': lifecycle['owner_thread_stopped'],
        'each_reused_task_has_two_correctly_parented_attempts': all(sum(s['name'] == 'mcp.attempt' for s in group) == 2 for group in by_trace.values()),
        'connect_counted_in_first_reused_task': len(connect) == 1 and connect[0] in by_trace[results['cases'][0]['claim_id']],
        'setup_shutdown_in_total': results['summary']['reused']['wall_seconds_including_close'] == results['summary']['reused']['wall_seconds'] + lifecycle['reused_batch_close_seconds'],
        'restart_parent_continuity': len(tasks) == 2 and tasks[1]['parent_id'] == tasks[0]['context']['span_id'],
        'restart_lifecycle_checks': json.loads((root / 'restart/report.json').read_text())['passed'],
        'one_new_connection_per_restart_process': all(json.loads((root / 'restart' / f'{stage}-lifecycle.json').read_text())['reused_connections_started'] == 1 for stage in ['interrupt', 'resume']),
        'restart_owners_stopped': all(json.loads((root / 'restart' / f'{stage}-lifecycle.json').read_text())['owner_thread_stopped'] for stage in ['interrupt', 'resume']),
    }
    serialized = json.dumps(spans + restart)
    checks['no_claim_text_in_spans'] = all(json.loads((root / 'comparison' / f"direct-{c['claim_id']}.json").read_text())['claim'] not in serialized for c in results['cases'])
    report['checks'] = checks
    report['faults'] = {p.stem: audit(read([p])) for p in (root / 'fault-spans').glob('*.jsonl') if p.read_text().strip()}
    report['passed'] = all(checks.values()) and report['comparison']['passed'] and report['restart']['passed'] and all(v['passed'] for v in report['faults'].values())
    (root / 'span-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise RuntimeError('persistent lifecycle evidence audit failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    run(parser.parse_args().directory)
