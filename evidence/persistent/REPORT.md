# Persistent MCP lifecycle: measured transport improvement

## Result and boundaries

One clean serial batch ran the same 24 frozen public SciFact dev cases through
unchanged direct, fresh-per-operation MCP, and reused-session MCP paths. All
**24/24 cases match across all three paths** for final answers, full prompts,
actions, tool results, and option-score audits. The pinned controller is official
Qwen2.5-1.5B-Instruct revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
CPU **BF16**, four inference threads, cached offline. No new model or paid API.

| Measure | Direct | Fresh MCP | Reused MCP |
|---|---:|---:|---:|
| Completed tasks | 24 | 24 | 24 |
| Correct labels | 11/24 | 11/24 | 11/24 |
| Correct label + complete evidence | 5/24 | 5/24 | 5/24 |
| Invalid/failed steps | 0 | 0 | 0 |
| Actual model forward passes | 48 | 48 | 48 |
| Sum of agent.run wall times | 115.21 s | 314.27 s | 118.65 s |
| Additional final batch close | 0 | 0 | 0.40 s |
| Total including batch close | **115.21 s** | **314.27 s** | **119.04 s** |

Reusing the session took **62.12% less batch time (2.64× fresh/reused ratio)**
than fresh MCP in this run. It remained **3.83 s (3.32%) slower than direct**.
This improves measured end-to-end transport lifecycle overhead, not model
inference speed or answer quality. Thirteen label predictions and nineteen
complete-evidence outcomes remain wrong. Each case reads one document; this is
not a multi-hop or unconstrained-planning result.

First reused-session startup, handshake, discovery, and index construction are
inside the first task timer. The one final close is added separately; the owner
thread was verified stopped. Fresh MCP startup/index/teardown happen inside each
operation. Common operator-side corpus index and model load are excluded for
all paths. Agent construction, result serialization and scoring are outside
these timers. The earlier f34 baseline used a different, wider timing boundary,
so its 290.95/103.05-second totals are **not** used to calculate this improvement.

Direct always runs first; fresh/reused alternate order by case index, declared
before execution. No heavy tests were intentionally run alongside the timed
batch. Host activity is otherwise uncontrolled. This is one local batch without
statistical repeats, randomized ordering, hardware isolation or production
throughput claims. Frozen cases are previously exposed development data, not a
new held-out evaluation. No retrieval, model, prompt or budget tuning is bundled
into this comparison.

## Ownership and actual telemetry

One dedicated thread owns one async task and all MCP/AnyIO contexts. Connection
lifetime cancellation scopes unwind in the same task that entered them. The
bridge accepts one serial caller, rejects concurrent calls and close-during-call,
and fails closed if cleanup becomes uncertain. Read-only operations have at most
two attempts; schema/corpus/remote-tool contract errors do not retry.

The comparison retained **1,250 actual OpenTelemetry SDK spans**, including 72
tasks, 144 model forwards, 96 MCP attempts, one reused connection and one reused
close. Two-process recovery added 23 spans; fault scenarios added 61 spans across
12 nonempty exports: **1,334 spans total**. These are local SDK exporter records,
not hand-written tracing examples or a hosted collector integration.

The original `span-audit.json` checks source hashes, completion, metadata
allowlists, trace continuity, setup/close accounting and lifecycle cleanup.
`independent-audit.json` additionally recomputes parity from raw artifacts,
compares case order and data hashes against `evidence/agent-protocol.json`, and
verifies each attempt's immediate matching tool parent and task grandparent,
including temporal containment. It also verifies the connection is an immediate
child of the first reused attempt, enforces all 15 expected fault-export files
and their span counts, and recomputes per-case and summary accuracy, complete
evidence, failed steps and model-forward totals from raw answers/audits. The
retained `gold-claims_dev.jsonl` is the original public annotation file, checked
against the frozen dataset SHA256 before scoring. Mutation tests reject altered raw answers,
trajectories, option audits, self-declared shortened manifests, wrong attempt/connection parents, missing or
empty fault exports, extra fault files, altered gold data and inflated summary counts.
The independent auditor was added after measurement; all captured measured
source hashes remain unchanged. Full prompt/read artifacts intentionally include
public source text; metadata-span redaction does not redact those artifacts.

## Fault matrix and restart

| Controlled condition | Verified result |
|---|---|
| Normal search/read/search | One server reused; close idempotent; thread and test-server PID gone |
| Wrong corpus, malformed schema, remote tool error | One call; contract rejection without retry |
| Disconnect once during call | Reconnect, retry succeeds, subsequent read reuses connection |
| Disconnect on both attempts | Two calls; bounded failure; owner/process cleanup |
| Timeout on both attempts | Two 2-second deadlines; bounded failure and cleanup |
| Timeout once | Reconnect succeeds, then session reused |
| Handshake timeout | Two child processes cleaned up; no tool dispatched |
| Disconnect while idle | Next call reconnects successfully |
| Concurrent caller or close during call | Explicit rejection; active serial operation can finish |
| Corpus file changed | Existing session retains snapshot; new connection rejects changed fingerprint |
| Fresh/reused checkpoint binding changed | Resume rejected as incompatible |
| Closed bridge after uncertainty | No dispatch permitted |
| Close before startup / repeated close | Safe without creating an owner thread |

All recorded fault-server PIDs are checked absent after cleanup. The bridge's
unexpected-stall fence is tested for no dispatch once closed; deliberately
wedging the owner itself is not exercised. Scheduling and SDK teardown grace
are not hard real-time guarantees. Process-fork reuse and parallel callers are
unsupported. Tests use scripted faults, not model reasoning/factuality evidence.

Two separate Python invocations exercised a real-model persisted pending read.
The first exited with code **75**, the second resumed. All **7 recovery checks**
passed, including no regeneration of the saved decision, same trace ID and
terminal completion. Both process owners stopped, each with one connection.
This is controlled read-only at-least-once replay, not power-failure durability
or exactly-once external side effects.

## Final validation

- Final full isolated suite after all auditor hardening: **389 passed in 58.88 s**,
  including 15 persistent lifecycle cases and 13 independent artifact-audit/mutation
  tests. Earlier full runs (376 and 382 tests) and focused checks are retained.
- Repository-pinned Ruff: all checks passed.
- Original span audit and independent raw-artifact audit: all checks passed.
- Optional MCP CI includes the persistent subprocess suite. The default suite
  includes the pure artifact auditor; no model download is required for tests.
- No public push or remote CI run is claimed by this local report.

## Retained attempts and reproduction

- `comparison/` is the only complete timing batch: 24 cases, final summary and
  successful lifecycle shutdown.
- `interrupted-development-comparison/` retains the first 7-case partial run.
- `interrupted-resume-comparison/` retains 10 complete cases plus a partial 11th
  from the previous interrupted environment lifetime. Neither partial run is
  pooled into final timing or span counts.
- Prior baseline, failures and initial code remain in commit
  `f34d1d652431987117dde9c70364764faa224f88` and `evidence/integrated/`.
- Initial post-run auditor hardening exposed two mutation-test parser failures
  (missing audit kind and a blank gold-file line). Both were fixed; the 11-pass /
  2-failure log remains under `logs/initial-hardened-audit-tests.log`.
- Test runs, exit code, benchmark output and verification logs are under `logs/`.
  Runtime SQLite checkpoints remain local and ignored; JSON traces are retained.

See [lifecycle guide](../../docs/persistent-mcp-lifecycle.md) for commands and
[dataset attribution](../../docs/evidence-agent.md#public-dataset-attribution-and-trace-contents)
for SciFact CC BY 4.0 / S2ORC ODC-By 1.0 source terms. Derived artifacts retain
original document IDs and source licenses. No private user documents are included.
