# Integrated MCP transport and collected OpenTelemetry

This is a **transport and observability experiment**, not an answer-quality
improvement or a production deployment. The existing constrained, pinned
Qwen2.5-1.5B/BF16 policy now executes its selected read action through the official
MCP Python SDK over stdio. Forced initial search remains scaffolding, not a model
decision. The direct path remains the default.

## Architecture

- `EvidenceAgent` retains the existing host-local SQLite orchestration. It saves
  the model decision before dispatch and commits the observation after execution.
- `MCPCorpusTools` opens a real `ClientSession` and server subprocess for each
  attempt, initializes it, discovers the two permitted tools and calls
  `search_corpus` or `get_evidence`. A session-per-attempt intentionally favors
  isolation and deterministic cleanup over latency. It rebuilds the corpus index
  in each subprocess. This is not a persistent remote service.
- Each attempt has a configurable deadline (default 15 seconds, maximum 120).
  There are at most two attempts. Transport errors retry; remote tool errors,
  corpus mismatches and invalid response schemas do not. The SDK closes stdin
  and handles subprocess cleanup; cleanup grace and OS scheduling can add time.
  There is still **no hard wall-clock bound on local model inference**.
- Each response must have one text block, no more than 65,536 UTF-8 bytes, the
  expected canonical corpus fingerprint and an exact strict result schema.
  Search returns at most three distinct IDs; read must return the requested ID
  and contiguous zero-based sentence numbers. The agent additionally restricts
  reads to search-returned IDs and citations to previously read sentences.
- Corpus identity normalizes document IDs to strings and hashes the ordered
  documents' consumed fields (`doc_id`, `title`, `abstract`), excluding unused
  SciFact metadata. This fixes the earlier server/direct loader mismatch. A
  digest is binding to a trusted local corpus, **not authentication of a malicious
  server**. Only the operator configures the server executable/corpus path.
- Transport mode, timeout/retry settings, model identity, prompt, corpus and
  budget are checkpoint-bound. The new canonical identity/binding intentionally
  rejects pre-integration checkpoints; use a fresh database. No automatic
  migration or silent transport switch is performed.

Read-only replay is at-least-once: a crash after tool completion but before its
checkpoint may repeat a read. Nothing here promises exactly-once external effects,
distributed leases, authenticated remote MCP, hostile-corpus memory limits, or
semantic entailment. The SDK parses protocol messages before the application
response-size check, so that check is not a hostile-server memory sandbox.

## Actual SDK spans

`evidence_telemetry.py` creates an official SDK `TracerProvider`, explicit
`ALWAYS_ON` sampler and `SimpleSpanProcessor` with an operator-selected local
exporter. Exported records come from SDK `ReadableSpan.to_json()`, not hand-built
span-shaped events. The simple processor makes this small local evidence capture
synchronous; it is not a hosted collector or distributed ingestion measurement.

Collected spans cover task invocation, policy invocation, actual model forward
passes, tool/finish validation, each MCP attempt and checkpoint saves. Policy
invocations include forced actions; **only `model.forward` counts real constrained
model inference**. Failed attempts are ERROR even when a retry later recovers.
An exhausted task, model exception or checkpoint failure is marked ERROR.

The persisted task context lets a separate resumed process continue the same trace
as a child of the interrupted invocation. The lifecycle test uses controlled
`SystemExit(75)` after committing a real-model read decision. It does not model
power failure. Abrupt termination can lose an unfinished task span even though
completed spans have already exported and a checkpoint remains usable.

Span attributes are allowlisted operational metadata. Claim text, prompts,
responses, corpus sentences, exception messages and stack traces are excluded.
Ambient host resource attributes are not included. Full public policy/agent
research traces are separate and intentionally contain benchmark text; **they are
not privacy-redacted and must not be published for private data without review**.

## Run

Install the existing base/model requirements plus these optional packages:

```sh
pip install -r requirements-mcp.txt -r requirements-observability.txt
# Download the existing pinned official model once; runtime stays local-only.
python -m src.evidence_agent_cli --policy capacity --transport mcp \
  --corpus /tmp/scifact/data/corpus.jsonl --db data/integrated.sqlite \
  --run-id example --claim 'The PRR MDA5 is a sensor of RNA virus infection.' \
  --spans-output data/spans.jsonl --trace-output data/state.json
```

No paid APIs, credentials, collector, model download at runtime, or deployment is
required. Model cache setup remains in [evidence-agent.md](evidence-agent.md).

## Reproduce evidence

Use fresh output directories for measurements. Run from the repository root:

```sh
EVIDENCE_INTEGRATED_SPANS=evidence/integrated/fault-spans pytest -q --durations=15
python -m src.evidence_integrated_eval --data /tmp/scifact/data \
  --output evidence/integrated/paired
python -m src.evidence_integrated_eval --data /tmp/scifact/data \
  --output evidence/integrated/restart --stage interrupt
# Verify exit status 75, then start another process:
python -m src.evidence_integrated_eval --data /tmp/scifact/data \
  --output evidence/integrated/restart --stage resume
python -m src.evidence_trace_verify --directory evidence/integrated
ruff check src app tests
```

The paired experiment uses all 24 previously frozen public dev cases, exactly the
same pinned model, word-only top3 retrieval, policy, prompts, candidate ordering,
budget and scoring. Direct/MCP pairs compare full prompts, raw decisions, tool
results, final answers and option-score audits. No hybrid retrieval or prompt
change is mixed into earlier answer metrics. Every failure stays in denominators.
Latency includes subprocess startup/index building and concurrent local jobs;
fixed direct-then-MCP ordering is not a randomized isolated performance study.

See [the measured report](../evidence/integrated/REPORT.md). New framework imports
are not the objective: LangGraph has not been added. A future alternative could
replace the custom loop with graph routing and a durable pre-dispatch interrupt,
but it would need the same parity/recovery tests rather than nesting two runtimes.

## Follow-on lifecycle optimization

The fresh-per-attempt implementation and results above are preserved as a measured
baseline. Explicit `--transport mcp-reused` now provides a safely owned persistent
session; see the [lifecycle optimization](persistent-mcp-lifecycle.md) and its
separate three-way measurements. The original `mcp` option remains available.
