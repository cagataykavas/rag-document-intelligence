"""Mutation checks for the independent, raw-artifact lifecycle audit."""
import json
import shutil
from pathlib import Path

import pytest

from src.persistent_artifact_audit import verify


@pytest.fixture
def artifacts(tmp_path):
    shutil.copytree(Path('evidence/persistent/comparison'), tmp_path / 'comparison',
                    ignore=shutil.ignore_patterns('*.sqlite*'))
    shutil.copytree(Path('evidence/persistent/fault-spans'), tmp_path / 'fault-spans')
    shutil.copyfile(Path('evidence/persistent/gold-claims_dev.jsonl'), tmp_path / 'gold-claims_dev.jsonl')
    return tmp_path


def test_retained_batch_passes_independent_audit():
    assert verify(Path('evidence/persistent'))['passed']


@pytest.mark.parametrize('part', ['answer', 'trace', 'audit'])
def test_raw_changes_cannot_hide_behind_summary_booleans(artifacts, part):
    path = artifacts / 'comparison/reused-1180.json'
    state = json.loads(path.read_text())
    if part == 'answer':
        state['answer'] = {'verdict': 'tampered'}
    elif part == 'trace':
        state['trace'][0]['raw'] = 'tampered'
    else:
        (artifacts / 'comparison/reused-1180-policy.jsonl').write_text('{}\n')
    path.write_text(json.dumps(state))
    assert not verify(artifacts)['checks']['raw_answer_prompt_trajectory_option_audit_parity']


def test_self_declared_manifest_cannot_hide_missing_case(artifacts):
    path = artifacts / 'comparison/results.json'
    result = json.loads(path.read_text())
    result['cases'].pop()
    result['protocol']['claim_ids'].pop()
    path.write_text(json.dumps(result))
    assert not verify(artifacts)['checks']['external_frozen_case_order']


def test_same_trace_wrong_immediate_parent_rejected(artifacts):
    path = artifacts / 'comparison/compare-spans.jsonl'
    spans = [json.loads(line) for line in path.read_text().splitlines()]
    attempt = next(s for s in spans if s['name'] == 'mcp.attempt')
    task = next(s for s in spans if s['name'] == 'agent.task'
                and s['context']['trace_id'] == attempt['context']['trace_id'])
    attempt['parent_id'] = task['context']['span_id']
    path.write_text('\n'.join(map(json.dumps, spans)) + '\n')
    assert not verify(artifacts)['checks']['attempts_immediately_parented_to_matching_tool_and_task']


@pytest.mark.parametrize('mutation', ['remove', 'empty', 'extra'])
def test_fault_exports_cannot_disappear_or_change(artifacts, mutation):
    path = artifacts / 'fault-spans/test_disconnect_once_reconnects_then_reuses.jsonl'
    if mutation == 'remove':
        path.unlink()
    elif mutation == 'empty':
        path.write_text('')
    else:
        (path.parent / 'unexpected.jsonl').write_text('{}\n')
    assert not verify(artifacts)['checks']['exact_fault_export_set_and_span_counts']


def test_connection_wrong_immediate_parent_rejected(artifacts):
    path = artifacts / 'comparison/compare-spans.jsonl'
    spans = [json.loads(line) for line in path.read_text().splitlines()]
    connect = next(s for s in spans if s['name'] == 'mcp.connect')
    task = next(s for s in spans if s['name'] == 'agent.task'
                and s['context']['trace_id'] == connect['context']['trace_id'])
    connect['parent_id'] = task['context']['span_id']
    path.write_text('\n'.join(map(json.dumps, spans)) + '\n')
    assert not verify(artifacts)['checks']['connect_immediately_parented_to_first_reused_attempt']


@pytest.mark.parametrize('target', ['case', 'summary'])
def test_accuracy_counts_recomputed_from_gold_and_raw_answers(artifacts, target):
    path = artifacts / 'comparison/results.json'
    result = json.loads(path.read_text())
    metrics = result['summary']['reused'] if target == 'summary' else result['cases'][0]['measurements']['reused']
    metrics['label_correct'] = 999
    path.write_text(json.dumps(result))
    assert not verify(artifacts)['checks']['raw_scoring_matches_case_and_summary_counts']


def test_gold_source_change_rejected(artifacts):
    path = artifacts / 'gold-claims_dev.jsonl'
    path.write_text(path.read_text() + '\n')
    assert not verify(artifacts)['checks']['gold_source_matches_frozen_hash']
