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
    parser.add_argument('--transport', choices=['direct', 'mcp', 'mcp-reused'], default='direct')
    parser.add_argument('--tool-timeout', type=float, default=15.0)
    parser.add_argument('--spans-output', type=Path, help='Official OTel SDK local export; metadata only')
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
    telemetry = None
    if args.spans_output:
        from src.evidence_telemetry import file_telemetry
        telemetry = file_telemetry(args.spans_output)
        policy.telemetry = telemetry
    if args.transport in ('mcp', 'mcp-reused'):
        from src.evidence_mcp_client import MCPCorpusTools
        from src.evidence_mcp_persistent import PersistentMCPCorpusTools
        tool_class = PersistentMCPCorpusTools if args.transport == 'mcp-reused' else MCPCorpusTools
        tools = tool_class(args.corpus, tools.fingerprint,
                               timeout=args.tool_timeout, telemetry=telemetry)
    try:
        state = EvidenceAgent(args.db, tools, policy, max_steps=args.max_steps,
                              telemetry=telemetry).run(args.run_id, args.claim)
    finally:
        try:
            if args.transport in ('mcp', 'mcp-reused'):
                tools.close()
        finally:
            if telemetry:
                telemetry.shutdown()
    if args.trace_output:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({key: state[key] for key in ['id', 'status', 'steps', 'answer']}, indent=2))


if __name__ == '__main__':
    main()
