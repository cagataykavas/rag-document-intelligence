"""Two-process local-model interruption/recovery check on one public training claim."""
import argparse
import json
from pathlib import Path

from src.evidence_agent import CorpusTools, EvidenceAgent
from src.evidence_agent_constrained import ConstrainedQwenPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['interrupt', 'resume'], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    db = args.output / 'runs.sqlite'
    if args.stage == 'interrupt' and db.exists():
        raise ValueError('interruption stage needs a fresh output directory')
    claim = next(c for c in map(json.loads, (args.data / 'claims_train.jsonl').read_text().splitlines()) if c['id'] == 997)
    corpus = CorpusTools(list(map(json.loads, (args.data / 'corpus.jsonl').read_text().splitlines())))
    audit_path = args.output / 'policy.jsonl'
    policy = ConstrainedQwenPolicy(audit_path=audit_path)
    agent = EvidenceAgent(db, corpus, policy)
    if args.stage == 'interrupt':
        def stop_after_model_selected_read():
            state = agent.load('recovery-smoke')
            if state['pending']['tool'] == 'read':
                audit = list(map(json.loads, audit_path.read_text().splitlines()))
                if not any(r['kind'] == 'model_selection' for r in audit):
                    raise RuntimeError('no real model decision before interruption')
                (args.output / 'before-restart.json').write_text(json.dumps(state, indent=2) + '\n')
                raise SystemExit(75)  # deliberate process exit after durable pending read
        agent.run('recovery-smoke', claim['claim'], after_pending=stop_after_model_selected_read)
        raise RuntimeError('did not reach a model-selected pending read')
    before = json.loads((args.output / 'before-restart.json').read_text())
    audits_before = list(map(json.loads, audit_path.read_text().splitlines()))
    result = agent.run('recovery-smoke', claim['claim'])
    audits_after = list(map(json.loads, audit_path.read_text().splitlines()))
    index = len(before['trace']) - 1
    old, resumed = before['trace'][index], result['trace'][index]
    checks = {'pending_action_was_read': before['pending']['tool'] == 'read',
        'same_persisted_raw_decision': old['raw'] == resumed['raw'],
        'same_persisted_generation_time': old['generation_seconds'] == resumed['generation_seconds'],
        'pending_read_executed': 'result' in resumed and resumed['result']['doc_id'] == before['pending']['doc_id'],
        'completed_after_restart': result['status'] == 'completed',
        'no_pending_decision_regeneration': len(audits_after) - len(audits_before) == result['steps'] - before['steps']}
    (args.output / 'after-restart.json').write_text(json.dumps(result, indent=2) + '\n')
    report = {'claim_id': 997, 'split': 'train', 'policy': policy.identity,
        'interruption': 'Separate process exited75 after real-model-selected read was committed, before tool execution.',
        'checks': checks, 'passed': all(checks.values()),
        'before_policy_calls': len(audits_before), 'after_policy_calls': len(audits_after),
        'answer_quality_claim': 'None. This checks lifecycle recovery, not answer correctness.'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise RuntimeError('recovery verification failed')


if __name__ == '__main__':
    main()
