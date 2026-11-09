"""Post-hoc descriptive diagnostics; never changes policy or excludes failed cases."""
import argparse
import hashlib
import json
from pathlib import Path

from src.evidence_agent_constrained import candidate_actions


def report(data: Path, run: Path):
    results = json.loads((run / 'results.json').read_text())
    claims = {c['id']: c for c in map(json.loads, (data / 'claims_dev.jsonl').read_text().splitlines())}
    diagnostics = []
    errors = {'schema_or_model': 0, 'tool_or_citation': 0}
    for pair in results['cases']:
        cid = pair['agent']['claim_id']
        claim = claims[cid]
        gold_docs = set(claim['evidence'])
        gold_sets = [{(str(doc_id), s) for s in e['sentences']} for doc_id, items in claim['evidence'].items() for e in items]
        baseline = json.loads((run / f'fixed_top3-{cid}.json').read_text())
        agent = json.loads((run / f'agent-{cid}.json').read_text())
        for event in agent['trace']:
            if 'error' in event:
                errors['tool_or_citation' if 'action' in event else 'schema_or_model'] += 1
        options = candidate_actions({'claim': claim['claim'], 'remaining_actions': 1,
            'history': baseline['trace'][:4]}, finish_only=True)
        candidate_sets = [{(c['doc_id'], c['sentence']) for c in action['citations']}
                          for action, _ in options if action.get('citations')]
        diagnostics.append({'claim_id': cid, 'has_gold_evidence': bool(gold_docs),
            'fixed_top3_any_gold_doc': bool(gold_docs & set(baseline['read'])),
            'agent_read_any_gold_doc': bool(gold_docs & set(agent['read'])),
            'gold_document_count': len(gold_docs),
            'fixed_top3_gold_document_count': len(gold_docs & set(baseline['read'])),
            'single_sentence_candidate_can_cover_gold_set': any(g <= c for g in gold_sets for c in candidate_sets),
            'smallest_gold_evidence_set': min(map(len, gold_sets)) if gold_sets else None})
    evidence_cases = [d for d in diagnostics if d['has_gold_evidence']]
    aggregate = {'evidence_bearing_cases': len(evidence_cases),
        'fixed_top3_any_gold_doc': sum(d['fixed_top3_any_gold_doc'] for d in evidence_cases),
        'agent_read_any_gold_doc': sum(d['agent_read_any_gold_doc'] for d in evidence_cases),
        'candidate_oracle_complete_evidence': sum(d['single_sentence_candidate_can_cover_gold_set'] for d in evidence_cases),
        'multi_sentence_only_gold_cases': sum(d['smallest_gold_evidence_set'] > 1 for d in evidence_cases),
        'agent_error_categories': errors,
        'by_class': {mode: {label: {'n': sum(p[mode]['gold'] == label for p in results['cases']),
            'label_correct': sum(p[mode]['gold'] == label and p[mode]['label_correct'] for p in results['cases']),
            'complete_evidence_correct': sum(p[mode]['gold'] == label and p[mode]['evidence_complete_correct'] for p in results['cases'])}
            for label in ['SUPPORT', 'CONTRADICT', 'NOT_ENOUGH_INFO']} for mode in ['fixed_top3', 'agent']}}
    return {'source_results_sha256': hashlib.sha256((run / 'results.json').read_bytes()).hexdigest(),
            'method': 'Post-hoc failure decomposition on all frozen cases; no filtering or policy changes. Candidate oracle is diagnostic only, never an agent input.',
            'summary': aggregate, 'cases': diagnostics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    output = report(args.data, args.run)
    (args.run / 'diagnostics.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output['summary'], indent=2))
