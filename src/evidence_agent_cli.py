"""Run/resume a read-only evidence investigation with a pinned local model."""
import argparse
import json
from pathlib import Path

from src.evidence_agent import CorpusTools, EvidenceAgent, LocalQwenPolicy
from src.evidence_agent_constrained import ConstrainedQwenPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True, help='SciFact-format corpus.jsonl')
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--claim', required=True)
    parser.add_argument('--max-steps', type=int, default=6)
    parser.add_argument('--trace-output', type=Path)
    parser.add_argument('--policy', choices=['constrained', 'free-json', 'capacity'], default='constrained')
    parser.add_argument('--audit-output', type=Path, help='Constrained option-score audit; default DB.policy.jsonl')
    args = parser.parse_args()
    tools = CorpusTools(list(map(json.loads, args.corpus.read_text().splitlines())))
    audit_path = args.audit_output or Path(str(args.db) + '.policy.jsonl')
    if args.policy == 'capacity':
        from src.evidence_agent_capacity import CapacityQwenPolicy
        policy = CapacityQwenPolicy(audit_path=audit_path)
    elif args.policy == 'constrained':
        policy = ConstrainedQwenPolicy(audit_path=audit_path)
    else:
        policy = LocalQwenPolicy()
    state = EvidenceAgent(args.db, tools, policy, max_steps=args.max_steps).run(args.run_id, args.claim)
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({key: state[key] for key in ['id', 'status', 'steps', 'answer']}, indent=2))


if __name__ == '__main__':
    main()
