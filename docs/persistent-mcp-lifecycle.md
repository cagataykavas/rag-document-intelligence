# Reuse the MCP session instead of rebuilding an index per call

The [initial integrated experiment](../evidence/integrated/REPORT.md), preserved
at commit `f34d1d652431987117dde9c70364764faa224f88`, exposed a real lifecycle cost:
24 MCP tasks took 290.95 seconds versus 103.05 seconds direct. The fresh MCP path
started and indexed a server for every search/read. Correct transport parity was
not enough to call that efficient.

`PersistentMCPCorpusTools` now offers an explicit `mcp-reused` mode. It retains one
server and its fixed corpus index across a serial batch. The original `mcp` mode
is preserved as the fresh-per-attempt baseline. The model, prompt, retrieval,
candidate menu and answer checks do not change.

## Ownership and deadlines

One dedicated thread runs one async owner task. Only that task enters, uses and
exits the MCP `stdio_client`, `ClientSession`, AnyIO task groups and cancellation
scope. The synchronous bridge submits serial requests using an asyncio queue and
thread-safe futures. It never enters a task group in one task and exits it in
another. Concurrent callers and close-during-call are explicitly rejected.

The outer connection-lifetime cancel scope is entered before the SDK contexts.
Its deadline is set for initialization/the current request and disabled while
idle. On timeout or disconnect, the same owner unwinds all contexts before the
caller receives failure. The next permitted attempt creates a new connection.
There are still at most two attempts per read-only operation. Contract errors do
not retry. A separate synchronous fence detects unexpected owner/cleanup stalls,
closes admission to further requests and reports uncertainty rather than silently
continuing. OS scheduling and SDK teardown grace are not hard realtime guarantees.

Use the client as a context manager or call `close()` in `finally`. Close wakes the
idle owner, closes SDK streams/processes, joins the thread, and reports incomplete
cleanup if it cannot finish within its budget. It does not claim success while a
thread remains alive. There is no automatic resource pool, distributed scheduler,
parallel-call API, or guarantee across process forks. Each process constructs its
own client. No telemetry backend or network service is required.

The server holds a corpus snapshot for its lifetime. Editing the file does not
mutate that in-memory snapshot. A reconnect to changed content is rejected against
the original fingerprint. Changing lifecycle mode also changes checkpoint binding,
so an old fresh-session checkpoint is not silently resumed with different settings.

## Spans and fair timing

Trace context is explicitly carried to the owner for connection initialization.
Every tool attempt remains attached to its current task, even though later tasks
reuse a connection created by the first task. `mcp.connect` measures actual
initialization/discovery; `mcp.close` measures actual batch shutdown. Existing
model/policy/tool/checkpoint spans remain intact. Metadata redaction stays the same.

The new three-way experiment runs all 24 frozen public cases through direct,
fresh MCP and reused MCP. Direct runs first; fresh/reused order alternates by case
index, declared in advance. Full prompts, decisions, tool results, answers and
option-score audits must match. This is previously exposed development data.

First-call server startup and index construction are **inside** the reused mode's
task timer. Final batch shutdown is measured separately and added to that mode's
total. The shared operator-side corpus load/index and model loading are excluded
from all modes, as in the original benchmark. No setup cost is silently amortized
away. This is one local serial batch, not production throughput or a randomized
hardware benchmark; other host activity remains uncontrolled.

## Run and reproduce

Install the same optional MCP/OTel requirements as the integration guide. The CLI
can reuse a connection within one task:

```sh
python -m src.evidence_agent_cli --policy capacity --transport mcp-reused \
  --corpus /tmp/scifact/data/corpus.jsonl --db data/reused.sqlite \
  --run-id example --claim 'The PRR MDA5 is a sensor of RNA virus infection.' \
  --spans-output data/reused-spans.jsonl
```

The batch experiment and controlled two-process recovery use fresh directories:

```sh
python -m src.evidence_persistent_eval --data /tmp/scifact/data \
  --output evidence/persistent/comparison
python -m src.evidence_persistent_eval --data /tmp/scifact/data \
  --output evidence/persistent/restart --stage interrupt
# Verify exit status 75, then start another process:
python -m src.evidence_persistent_eval --data /tmp/scifact/data \
  --output evidence/persistent/restart --stage resume
EVIDENCE_PERSISTENT_SPANS=evidence/persistent/fault-spans pytest -q --durations=20
python -m src.evidence_persistent_verify --directory evidence/persistent
python -m src.persistent_artifact_audit --directory evidence/persistent
ruff check src app tests
```

Tests cover actual timeout/disconnect/reconnect, idle disconnect, initialization
timeout, nonretryable contract failures, serial admission, corpus snapshot
identity and lifecycle-bound checkpoints. They verify that the owner thread and
recorded test-server PIDs are gone after close. Scripted faults are reliability
checks, not reasoning/answer-quality evidence. Restart remains read-only
at-least-once replay; arbitrary external side effects are not exactly-once.

See [the measured lifecycle report](../evidence/persistent/REPORT.md) for results,
setup/shutdown accounting, full retained evidence and limitations.

## Checked-in evidence

Run `python tools/restore_evidence.py` before the artifact auditors above.
See [lossless packaging](evidence-packaging.md); the restored bytes match the original run exactly.
