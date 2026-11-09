# Evidence agent: measured tool use, not a wrapper demo

This extension investigates a scientific claim against public SciFact abstracts.
It is a local, read-only research prototype. It does **not** provide medical
advice, execute arbitrary code, browse the web, or establish scientific truth.

Two policies intentionally share the same durable runtime:

- **V1 free JSON:** pinned Qwen2.5-0.5B-Instruct generates search/read/finish actions.
  Bad JSON, fabricated document IDs and unsupported tool names are rejected,
  consume the action budget, and are reported back to the model.
- **V2 constrained controller:** the same real model selects among application-built
  admissible actions using next-token logits. This is model-guided workflow
  control, **not** unconstrained planning or learned schema compliance. The first
  claim search is forced scaffolding, not a model decision. A finite read/answer
  menu makes schema-valid actions possible by construction.

The model is small enough for a CPU demonstration. Neither successful tool
execution nor passing infrastructure tests demonstrates answer accuracy.

## Architecture and trust boundary

`src/evidence_agent.py` provides `CorpusTools`, strict action validation,
`EvidenceAgent`, SQLite checkpoints and the optional local model adapter.
`src/evidence_agent_constrained.py` is the alternative constrained policy.

1. Search ranks the full corpus with word TF-IDF and returns three IDs/titles.
2. Read accepts only IDs actually returned by search. It returns numbered abstract
   sentences; source text is data, never an instruction or executable content.
3. Finish accepts SUPPORT, CONTRADICT, or NOT_ENOUGH_INFO. Non-abstaining answers
   must cite sentence indices from documents that were actually read. This verifies
   **membership only, not semantic entailment**.
4. Before any tool invocation, the raw model output and validated pending action
   are committed to SQLite. After the tool, its observation and state transition
   are committed together. Restart replays a pending read-only action. A crash
   between a tool and its checkpoint can repeat that read; this is at-least-once
   read-only replay, not general exactly-once side effects. A crash during model
   generation before its checkpoint can regenerate that step.
5. POSIX file locking serializes workers sharing the database on one host. A
   corpus/model/prompt/budget fingerprint prevents accidental configuration changes
   when resuming. This is not a distributed lease or authenticated API service.

Budgets cover actions (default six), generation (160 tokens in V1), and input
context (12,000 tokens). **There is no hard wall-clock deadline:** inference or an
unresponsive custom tool can block. Corpus files and the SQLite store are trusted
local inputs, not hostile uploads. Tool output size is finite for the evaluated
SciFact corpus, but the base runtime is not a general untrusted-corpus ingestion
service. The [optional MCP adapter](evidence_mcp.md) adds separate transport input
and response limits; the original measurements call tools in-process. The new explicit `--transport mcp`
mode integrates real stdio calls and collected SDK spans; see
[integrated transport and observability](integrated-mcp-observability.md).

## Constrained policy details

The policy constructs up to 26 options. At least one source must be read before abstention. Unread retrieved documents
are candidate read actions while budget remains. For each read document, the
three sentences sharing the most lowercase word tokens with the claim are
candidate citations (sentence index breaks ties). Each candidate has SUPPORT and
CONTRADICT answer alternatives; abstention has no citations. It can cite only one
sentence, which cannot cover a multi-sentence gold evidence set.

Option labels A..Z must each tokenize to exactly one token. One real model forward
pass scores their next-token logits. Argmax chooses an action; the conditional
softmax over legal labels is logged for inspection and **is not calibrated
confidence**. Options are deterministically permuted using SHA-256 of the seed (2026), claim and action. This avoids systematically placing SUPPORT first but does not eliminate positional bias. Neither
lexical sentence preselection nor forced search is presented as a model inference.

The policy audit records exact option prompts, token IDs, probabilities and chosen
actions. The runtime trace separately records all model replies, tool outputs,
errors, timings and terminal state. In constrained-run per-case JSON, the legacy
`model_calls` field counts policy invocations, including forced search;
`model_forward_passes` is the actual inference count used in summaries. Public traces contain only public benchmark
claims and abstracts. Do not publish traces from private documents without review.

## Reproduce

Run from the repository root on Linux/macOS (POSIX locking):

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-agent.txt
# Download once from the official model repository. Runtime never downloads.
.venv/bin/hf download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775
curl -fL https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz -o /tmp/scifact-data.tar.gz
mkdir -p /tmp/scifact && tar -xzf /tmp/scifact-data.tar.gz -C /tmp/scifact
.venv/bin/python -m src.evidence_agent_cli \
  --corpus /tmp/scifact/data/corpus.jsonl --db data/agent.sqlite \
  --run-id example-1 --claim 'The PRR MDA5 is a sensor of RNA virus infection.' \
  --trace-output data/example-1.json
# Default is the constrained controller. Use --policy free-json for the V1 failure-prone policy.
# Rerun that command to resume; completed runs do not make new model calls.
.venv/bin/python -m src.evidence_agent_eval --data /tmp/scifact/data \
  --manifest evidence/agent-protocol.json --output data/v1-reproduction
.venv/bin/python -m src.evidence_agent_constrained_eval --data /tmp/scifact/data \
  --manifest evidence/agent-protocol.json --output data/v2-reproduction
.venv/bin/pytest -q
.venv/bin/ruff check src app tests
```

Use a new output directory for a fresh measurement. Reusing a recovery database
resumes saved runs: per-invocation wall time then reflects replay/cache work, not a
fresh inference benchmark. Exact evaluated package versions are recorded in the
results, and a pinned model revision is required. Floating optional package ranges
above are installation guidance, not a bit-for-bit environment lock.

## Evaluation interpretation

`evidence/agent-protocol.json` was committed before V1 inference. It selects 24
public dev claims, eight per gold class, deterministically by a salted SHA-256
ordering. SciFact dev had already been used by the repository's retrieval study:
**this is not an untouched test set**. V2 is an explicitly developmental comparison
following V1 failures and separate training-claim usability checks.

The fixed baseline sees all three top retrieved abstracts and uses one model
answer decision; the agent chooses which abstracts to inspect within six actions.
They have different evidence and compute budgets. The V1 baseline still had a
search/read-capable prompt and often failed formatting; V2 makes its final menu
finish-only. The always-abstain baseline requires no language model.

All selected cases remain in denominators. Invalid/exhausted runs are incorrect,
not dropped. Label accuracy, exact gold evidence-set completeness, gold-sentence
citation precision, terminal completion, abstentions, errors, model calls and
latency are separate quantities. Evidence completeness allows extra citations;
citation precision addresses overlap with annotated gold sentences, not exhaustive
semantic support (SciFact annotations may be incomplete). Correct NEI abstention
counts as evidence-complete but is not proof that no evidence exists in the corpus.

Fault tests use explicitly scripted policies: crash before/after tool execution,
checkpoint recovery, changed-config rejection, model/tool failure, invented
citations, unauthorized shell actions and exhausted budgets. These establish
specific application invariants. They are **not real-model injection resistance,
reasoning, factuality, or clinical validation results**.

## Known next steps

- Evaluate a larger model on a newly frozen, genuinely unexposed dataset.
- Add semantic entailment checks and multi-sentence citation candidates.
- Measure candidate-order sensitivity and abstention calibration.
- Add explicit process-level deadlines and cancellation before serving users.
- The optional MCP boundary is now integrated and measured separately; see the
  [transport/observability experiment](integrated-mcp-observability.md).

## Reproduce the real-model restart smoke

The interruption must return exit code75; a normal completion is not a successful
interruption test. Use a fresh output directory:

```sh
.venv/bin/python -m src.evidence_agent_recovery_smoke --data /tmp/scifact/data \
  --output data/recovery-check --stage interrupt
# Check the previous process exited75, then start a new process:
.venv/bin/python -m src.evidence_agent_recovery_smoke --data /tmp/scifact/data \
  --output data/recovery-check --stage resume
```

The JSON report verifies pending-read replay and absence of repeated generation
for that saved decision. It makes no answer-quality claim.

## Capacity-only experiment

An official, non-gated Qwen2.5-1.5B-Instruct checkpoint is pinned at
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. It uses CPU BF16 and the **same** constrained
candidate builder, ordering seed, retrieval settings and 24 cases. Both model
size and dtype change, so this is not an isolated parameter-count ablation. No
remote model code or account credentials are needed. Download the official weights
once, then the runtime uses local files only:

```sh
.venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
.venv/bin/python -m src.evidence_agent_capacity --data /tmp/scifact/data \
  --output data/capacity-resource-smoke
.venv/bin/python -m src.evidence_agent_constrained_eval --data /tmp/scifact/data \
  --manifest evidence/agent-protocol.json --output data/capacity-comparison --capacity
```

The ordinary CLI also accepts `--policy capacity` after that download. A separate
one-case training resource smoke completed in 10.44 seconds including load/index
with peak Linux RSS of 3,677 MiB on this AVX512/BF16 host. This is not a latency
promise for other CPUs. The smoke makes no answer-accuracy claim.


The measured agent uses word-only top3 retrieval even though the parent RAG also
implements hybrid word/character retrieval. This is an explicit integration gap.
A frozen-case retrieval-only comparison is in the results report: hybrid top3
improves macro document recall from 52.08% to 60.42% on 16 evidence-bearing cases;
hybrid top10 reaches 85.42%. These are retrieval metrics, not model-answer accuracy,
and no frozen model results were rerun or relabeled.

## Public dataset attribution and trace contents

These diagnostic runs use [SciFact](https://github.com/allenai/scifact),
introduced by Wadden et al. (2020), [Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/).
SciFact claims and evidence annotations are licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); its abstracts originate
from [Semantic Scholar S2ORC](https://github.com/allenai/s2orc) and are licensed
under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/), as documented in
the [upstream dataset license](https://github.com/allenai/scifact/blob/master/LICENSE.md).

The checked-in agent prompts, read-tool outputs, and traces contain selected
public claims and abstract text, not only aggregate metrics. They are derived
benchmark artifacts: document titles and abstracts are assembled into prompts,
selected sentences are numbered for citations, and model outputs are recorded
without treating them as new ground truth. Original document IDs are retained
for attribution and audit. The source data retains its original licenses; no
private company documents or user conversations are included in these traces.
