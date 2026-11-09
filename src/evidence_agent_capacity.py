"""Pinned official 1.5B/BF16 capacity check, sharing the unchanged constrained policy."""
import argparse
import json
import resource
import time
from pathlib import Path

from src.evidence_agent import CorpusTools, EvidenceAgent
from src.evidence_agent_constrained import ConstrainedQwenPolicy

MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


class CapacityQwenPolicy(ConstrainedQwenPolicy):
    def __init__(self, *, audit_path: Path, finish_only=False, threads=4, order_seed=2026):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        torch.manual_seed(2026)
        self.finish_only, self.order_seed = finish_only, order_seed
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = (f'{MODEL_ID}@{REVISION};cpu;bfloat16;threads={threads};'
                         f'constrained-options-v1;finish_only={finish_only};lexical-top3-sentences;'
                         f'require-read;order_seed={order_seed}')
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION,
            local_files_only=True, trust_remote_code=False, token=False)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION,
            local_files_only=True, trust_remote_code=False, token=False, dtype=torch.bfloat16).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'runs.sqlite').exists():
        raise ValueError('capacity resource smoke requires a fresh directory')
    started = time.perf_counter()
    claim = next(c for c in map(json.loads, (args.data / 'claims_train.jsonl').read_text().splitlines()) if c['id'] == 997)
    tools = CorpusTools(list(map(json.loads, (args.data / 'corpus.jsonl').read_text().splitlines())))
    policy = CapacityQwenPolicy(audit_path=args.output / 'policy.jsonl')
    load_seconds = time.perf_counter() - started
    state = EvidenceAgent(args.output / 'runs.sqlite', tools, policy).run('capacity-smoke', claim['claim'])
    result = {'model': MODEL_ID, 'revision': REVISION, 'dtype': str(policy.model.dtype),
        'claim_id': 997, 'split': 'train', 'model_load_and_index_seconds': load_seconds,
        'total_seconds': time.perf_counter() - started,
        'peak_rss_mib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'status': state['status'], 'steps': state['steps'], 'answer': state['answer'],
        'scope': 'One-case memory/lifecycle smoke only, not answer-accuracy evidence.'}
    (args.output / 'trace.json').write_text(json.dumps(state, indent=2) + '\n')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
