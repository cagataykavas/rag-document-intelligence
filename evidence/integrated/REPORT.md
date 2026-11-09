# Integrated evidence-agent: MCP + collected OpenTelemetry

## Measured result

The real pinned Qwen2.5-1.5B-Instruct/BF16 controller now dispatches read-only
retrieval operations over official MCP stdio. Task, policy, actual model forward,
tool, retry-attempt and SQLite-checkpoint spans are produced by the official
OpenTelemetry SDK and collected by a local SDK exporter.

This is integrated executable behavior, not a standalone server or handcrafted
trace demonstration. It is not an answer-quality improvement, hosted collector,
remote deployment, unconstrained planner or multi-hop reasoning result.

## Frozen direct/MCP parity

All 24 pre-existing public SciFact dev cases remain in the experiment. This is
already-exposed development data, not a newly held-out test. Each case runs both
paths with the same pinned model revision, BF16, policy, word TF-IDF top3
retrieval, candidate ordering, prompt, six-action budget and scoring.

| Measure | Direct | MCP |
|---|---:|---:|
| Completed tasks | 24/24 | 24/24 |
| Correct labels | 11/24 (45.8%) | 11/24 (45.8%) |
| Correct label + complete gold evidence | 5/24 (20.8%) | 5/24 (20.8%) |
| Invalid/failed steps | 0 | 0 |
| Actual model forward passes | 48 | 48 |
| Measured total run time, excluding model load | 103.05 s | 290.95 s |

**24/24 pairs have identical final answers, complete prompts/actions/tool results,
and policy option-score audits.** A successful transport does not make the wrong
answers correct: 13/24 label predictions remain wrong and 19/24 lack a correct,
complete evidence outcome. Each trajectory reads one document; this does not
establish multi-hop investigation.

MCP took 187.90 seconds more in this local run. A fresh server process rebuilds the
index per call, intentionally simplifying failure isolation. Timings include
concurrent local work and fixed direct-before-MCP ordering; they are descriptive,
not a randomized performance comparison or production latency promise. No hybrid
retrieval or prompt/model tuning was folded into earlier metrics.

## Actual span evidence

The SDK exporter collected **816 spans across 48 paired task invocations**:

- 48 task spans and 144 policy-invocation spans.
- 96 real `model.forward` spans, exactly matching policy inference audit counts.
- 144 tool/finish spans, with 48 additional child MCP-attempt spans.
- 336 SQLite checkpoint-save spans.

A separate two-process restart contributed 19 spans on one continued trace.
Scripted fault tests contributed 46 spans. Total retained successful/fault/lifecycle
SDK export: **881 spans**. Span count is not throughput or business impact.

`span-audit.json` verifies unique IDs, same-trace parent linkage, duration ordering,
metadata/resource allowlists, no exception-message/events, no claim text, actual
model-count agreement and export of all task invocations. The resumed task span
is a child of the interrupted process's task span. These are real exported SDK
records, not JSON constructed to resemble traces. No hosted OTel collector was
used or verified. Full research policy/agent traces are public-text artifacts and
are **not** covered by span redaction claims.

## Recovery and transport faults

The actual 1.5B model selected a read, which the first process committed before
exiting with code 75. A second Python process resumed it through MCP and completed.
All **7 recovery checks** passed: pending read, same saved raw decision, same saved
generation time, read result committed, terminal completion, no regenerated saved
policy decision, and continued trace ID. This controlled exit tests a restart
boundary; it is not a power-failure/distributed-recovery benchmark.

Real official-SDK subprocess tests cover six transport/contract scenarios:

| Injected condition | Observed behavior |
|---|---|
| Server disconnects once during call | First attempt ERROR, second succeeds |
| Server disconnects on both calls | Exactly two calls; bounded failure |
| Server hangs during both calls | Both calls reached server; two 2-second deadlines; failure in 4.03 s including cleanup |
| Wrong corpus fingerprint | One call; rejected without retry |
| Invalid response schema | One call; rejected without retry |
| Remote tool error | One call; rejected without retry; message excluded from spans |

Additional tests cover strict document/sentence binding, duplicate IDs, oversized
responses, metadata normalization, transport-change resume rejection, model
failure, checkpoint failure, error-span status and redaction. Scripted fault tests
are infrastructure checks, **not model reasoning, injection-resistance or
factuality evidence**. Reads may replay after crash; no exactly-once side-effect
claim is made. Timeout tests cover subprocess tool requests, not inference.

## Validation and retained failures

- Final full suite: **361 passed in 25.46 s**, including 19 new integration tests.
- Ruff **0.12.11**, as pinned by the repository: all checks passed.
- Final SDK span audit: all checks passed.
- Existing direct tests remain included; optional CI now installs MCP and OTel
  dependencies and runs both standalone and integrated subprocess suites.

Failures were kept, not silently replaced:

1. The first test collection lacked inherited scikit-learn dependencies in the new
   isolated environment: 11 collection errors. Correcting that environment path
   restored the existing 342-test baseline. No product defect is inferred.
2. Initial new tests had **7 failures / 4 passes** because the ambient runtime used
   1% OTel sampling. The opt-in local provider now explicitly uses ALWAYS_ON and a
   minimal resource rather than inheriting host resource metadata.
3. The first real-model pairing exposed a corpus identity mismatch: the MCP loader
   normalized IDs and removed unused fields while direct tools hashed the raw
   input. Both completed MCP cases (1180, 343) exhausted with six rejected search
   steps each; both direct cases completed. Execution was interrupted during the
   third case, not treated as a finished 24-case result. The consumed corpus
   fields are now normalized identically. Exact failed traces/protocol and the
   interrupted third policy audit remain in `failed-initial-pairing/`.
4. Early lint runs used ambient Ruff 0.17.4 and its broader defaults, reporting 53
   then 52 issues (including inherited baseline findings). An unused new import
   was removed. The final check uses the repository-pinned 0.12.11; no broad,
   unrelated baseline formatting rewrite was performed.
5. One span audit was run before all paired cases finished and correctly failed
   completeness/parent checks for the in-progress export. That interim report is
   preserved under `logs/span-audit-before-evaluation-completion.*`. The final
   audit is from the complete export.

The initial collection/fault/lint logs and intermediate results are under `logs/`.
Runtime SQLite databases remain local and ignored; public JSON state/decision
traces, source hashes, frozen data hashes and collected spans are versioned.

## Reproduction and scope

See [the integration guide](../../docs/integrated-mcp-observability.md) for commands,
strict contracts, retry policy and privacy boundaries. `paired/compare-protocol.json`
records package versions, model/data identity and source hashes before execution.
`paired/results.json` retains all 24 scored pairs; `restart/report.json` and
`span-audit.json` retain machine-checkable outcomes.

No paid API, credential, cloud deployment, new model, new prompt, or new retrieval
configuration was used. No LangGraph adapter was added. It would be defensible as
an **alternative replacement** orchestration with conditional graph routing and a
durable pre-dispatch interrupt, but would require its own parity/restart results;
adding it as a wrapper around this loop would not demonstrate that capability.

## Defensible portfolio wording

“Integrated official MCP stdio tool execution and OpenTelemetry SDK tracing into a
checkpointed local-model evidence agent; verified identical trajectories on 24
frozen direct/MCP case pairs, collected 881 linked lifecycle/fault/runtime spans,
and tested two-process recovery plus six real subprocess failure scenarios.”

Do not replace this with an answer-accuracy improvement, production-scale
observability, multi-hop reasoning or exactly-once execution claim.
