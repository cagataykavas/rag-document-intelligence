"""Independent post-run audit; does not rewrite measured source or artifacts."""
import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_FAULT_COUNTS = {
    'test_disconnect_once_reconnects_then_reuses': 6,
    'test_closed_bridge_fence_does_not_dispatch': 0,
    'test_lifecycle_switch_cannot_resume_existing_checkpoint': 0,
    'test_contract_faults_not_retried[bad_schema]': 3,
    'test_close_before_start_is_safe': 0,
    'test_contract_faults_not_retried[remote_error]': 3,
    'test_bounded_failure_and_owner_cleanup[timeout]': 5,
    'test_bounded_failure_and_owner_cleanup[disconnect]': 5,
    'test_idle_disconnect_reconnects_on_next_call': 6,
    'test_reuses_one_process_and_cleans_up': 6,
    'test_contract_faults_not_retried[wrong_corpus]': 3,
    'test_snapshot_identity_survives_reuse_and_rejects_changed_reconnect': 7,
    'test_handshake_timeout_closes_children': 5,
    'test_timeout_once_reconnects_and_then_reuses': 6,
    'test_concurrent_call_and_close_are_rejected': 6,
}


def verify(root, manifest_path=Path('evidence/agent-protocol.json')):
    manifest = json.loads(manifest_path.read_text())
    comparison = root / 'comparison'
    results = json.loads((comparison / 'results.json').read_text())
    spans = [json.loads(line) for line in (comparison / 'compare-spans.jsonl').read_text().splitlines()]
    by_id = {(s['context']['trace_id'], s['context']['span_id']): s for s in spans}
    checks = {
        'external_frozen_case_order': [c['claim_id'] for c in results['cases']] == manifest['claim_ids'],
        'external_frozen_dataset_hashes': results['protocol']['data_sha256'] == manifest['data_sha256'],
        'captured_sources_unchanged': all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
            for p, digest in results['protocol']['code_sha256'].items()),
    }
    gold_path = root / 'gold-claims_dev.jsonl'
    checks['gold_source_matches_frozen_hash'] = hashlib.sha256(gold_path.read_bytes()).hexdigest() == manifest['data_sha256']['claims_dev.jsonl']
    claims = {c['id']: c for c in map(json.loads, filter(str.strip, gold_path.read_text().splitlines()))}
    faults = {p.stem: len(p.read_text().strip().splitlines()) for p in (root / 'fault-spans').glob('*.jsonl')}
    checks['exact_fault_export_set_and_span_counts'] = faults == EXPECTED_FAULT_COUNTS
    totals = {mode: {'label_correct': 0, 'evidence_complete_correct': 0,
                    'invalid_or_failed_steps': 0, 'model_forward_passes': 0}
              for mode in ['direct', 'fresh', 'reused']}
    scored_cases = []
    raw_parity, parentage, task_export = [], [], []
    for cid in manifest['claim_ids']:
        states = {mode: json.loads((comparison / f'{mode}-{cid}.json').read_text())
                  for mode in ['direct', 'fresh', 'reused']}
        audits = {mode: (comparison / f'{mode}-{cid}-policy.jsonl').read_text().splitlines()
                  for mode in states}
        canonical = {mode: [{k: v for k, v in entry.items() if k != 'generation_seconds'}
                           for entry in state['trace']] for mode, state in states.items()}
        for mode in ['fresh', 'reused']:
            raw_parity.append(states[mode]['answer'] == states['direct']['answer']
                and canonical[mode] == canonical['direct'] and audits[mode] == audits['direct'])
        for mode, state in states.items():
            claim = claims[cid]
            labels = {e['label'] for items in claim['evidence'].values() for e in items}
            gold = next(iter(labels), 'NOT_ENOUGH_INFO')
            answer = state['answer']
            correct = bool(answer and answer.get('verdict') == gold)
            cited = {(c['doc_id'], c['sentence']) for c in answer.get('citations', [])} if answer else set()
            gold_sets = [{(str(doc), sentence) for sentence in e['sentences']}
                         for doc, items in claim['evidence'].items() for e in items if e['label'] == gold]
            computed = {'label_correct': correct,
                'evidence_complete_correct': correct and (gold == 'NOT_ENOUGH_INFO' or any(g <= cited for g in gold_sets)),
                'invalid_or_failed_steps': sum('error' in entry for entry in state['trace']),
                'model_forward_passes': sum(json.loads(a).get('kind') == 'model_selection' for a in audits[mode])}
            measured = next((c['measurements'][mode] for c in results['cases'] if c['claim_id'] == cid), {})
            scored_cases.append(all(measured.get(k) == v for k, v in computed.items()))
            for key, value in computed.items():
                totals[mode][key] += value
            tid = '0x' + state['trace_context']['trace_id']
            group = [s for s in spans if s['context']['trace_id'] == tid]
            tasks = [s for s in group if s['name'] == 'agent.task']
            task_export.append(len(tasks) == 1 and tasks[0]['attributes'].get('workflow.status') == 'completed')
            attempts = [s for s in group if s['name'] == 'mcp.attempt']
            parentage.append(len(attempts) == (0 if mode == 'direct' else 2))
            for attempt in attempts:
                parent = by_id.get((tid, attempt['parent_id']))
                expected_tool = {'search_corpus': 'search', 'get_evidence': 'read'}.get(attempt['attributes']['tool.name'])
                parentage.append(bool(parent and parent['name'] == 'agent.tool'
                    and parent['attributes'].get('tool.name') == expected_tool
                    and len(tasks) == 1 and parent['parent_id'] == tasks[0]['context']['span_id']
                    and parent['start_time'] <= attempt['start_time'] <= attempt['end_time'] <= parent['end_time']))
    connects = [s for s in spans if s['name'] == 'mcp.connect']
    first = json.loads((comparison / f"reused-{manifest['claim_ids'][0]}.json").read_text())
    first_trace = '0x' + first['trace_context']['trace_id']
    first_attempts = sorted([s for s in spans if s['name'] == 'mcp.attempt'
                            and s['context']['trace_id'] == first_trace], key=lambda s: s['start_time'])
    checks['connect_immediately_parented_to_first_reused_attempt'] = bool(len(connects) == 1 and first_attempts
        and connects[0]['context']['trace_id'] == first_trace
        and connects[0]['parent_id'] == first_attempts[0]['context']['span_id']
        and first_attempts[0]['start_time'] <= connects[0]['start_time'] <= connects[0]['end_time'] <= first_attempts[0]['end_time'])
    checks['raw_scoring_matches_case_and_summary_counts'] = all(scored_cases) and all(
        results['summary'][mode].get(key) == value for mode, metrics in totals.items() for key, value in metrics.items())
    checks.update(raw_answer_prompt_trajectory_option_audit_parity=all(raw_parity),
        attempts_immediately_parented_to_matching_tool_and_task=all(parentage),
        every_raw_task_has_completed_sdk_span=all(task_export))
    return {'checks': checks, 'passed': all(checks.values()),
        'scope': 'Independent post-run audit added after measurement. Original measured source hashes unchanged. Raw parity and immediate parentage recomputed, not trusted from summary booleans.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.directory)
    (args.directory / 'independent-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)
