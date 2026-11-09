"""Audit local ReadableSpan exports produced by the official OTel SDK."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read(paths):
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]


def audit(spans):
    identities = {(s['context']['trace_id'], s['context']['span_id']) for s in spans}
    allowed = {'workflow.steps', 'workflow.status', 'tool.name', 'retry.attempt', 'error.type', 'model.input_tokens'}
    checks = {
        'nonempty': bool(spans),
        'unique_span_ids': len(identities) == len(spans),
        'parents_present_in_same_trace': all(not s['parent_id'] or
            (s['context']['trace_id'], s['parent_id']) in identities for s in spans),
        'positive_or_zero_durations': all(s['end_time'] >= s['start_time'] for s in spans),
        'attribute_allowlist': all(set(s['attributes']) <= allowed for s in spans),
        'no_exception_events_or_messages': all(not s['events'] and 'description' not in s['status'] for s in spans),
        'resource_allowlist': all(s['resource']['attributes'] == {'service.name': 'evidence-agent'} for s in spans),
    }
    return {'checks': checks, 'passed': all(checks.values()),
        'span_counts': dict(Counter(s['name'] for s in spans)),
        'trace_count': len({s['context']['trace_id'] for s in spans}),
        'error_spans': sum(s['status']['status_code'] == 'ERROR' for s in spans)}


def main(args):
    root = args.directory
    paired = read([root / 'paired/compare-spans.jsonl'])
    restart = read([root / 'restart/interrupt-spans.jsonl', root / 'restart/resume-spans.jsonl'])
    results = json.loads((root / 'paired/results.json').read_text())
    report = {'paired': audit(paired), 'restart': audit(restart)}
    expected_forward = sum(c['measurements'][mode]['model_forward_passes'] for c in results['cases'] for mode in ['direct', 'mcp'])
    tasks = [s for s in restart if s['name'] == 'agent.task']
    report['checks'] = {
        'all_frozen_cases_retained': len(results['cases']) == len(results['protocol']['claim_ids']),
        'model_spans_match_actual_forward_audits': report['paired']['span_counts'].get('model.forward') == expected_forward,
        'all_paired_tasks_exported': report['paired']['span_counts'].get('agent.task') == 2 * len(results['cases']),
        'restart_task_is_child_of_prior_process_task': len(tasks) == 2 and tasks[1]['parent_id'] == tasks[0]['context']['span_id'],
        'interruption_marked_error': len(tasks) == 2 and tasks[0]['status']['status_code'] == 'ERROR',
        'resume_completed': len(tasks) == 2 and tasks[1]['attributes']['workflow.status'] == 'completed',
    }
    # Public claim text must not appear in exported metadata; full policy traces
    # intentionally remain separate research artifacts and are NOT redacted.
    exported = json.dumps(paired + restart)
    report['checks']['no_claim_text_in_spans'] = all(
        json.loads((root / 'paired' / f"direct-{c['claim_id']}.json").read_text())['claim'] not in exported
        for c in results['cases'])
    report['faults'] = {p.stem: audit(read([p])) for p in (root / 'fault-spans').glob('*.jsonl') if p.read_text().strip()}
    report['passed'] = (all(report['checks'].values()) and report['paired']['passed']
                        and report['restart']['passed'] and all(v['passed'] for v in report['faults'].values()))
    (root / 'span-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise RuntimeError('span audit failed; see retained report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    main(parser.parse_args())
