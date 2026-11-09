# Evidence-agent experiment: results and recommendation

## Scope

Public SciFact development diagnostic, 24 predeclared cases (8 SUPPORT, 8
CONTRADICT, 8 NOT_ENOUGH_INFO). These are **not untouched test cases**: this
repository previously used SciFact dev for retrieval evaluation. V1 protocol
was frozen at commit `e0f1aa9`; V2 development code at `21eaa8e` after train-only
usability checks. Exact code hashes, dataset hashes, prompts, model outputs,
option scores and traces accompany the results.

## V1 free JSON failure baseline

All 24 fixed-top3 runs and all 24 adaptive runs exhausted without an accepted final
answer: 0/24 correct labels for both. The agent used all 144 model calls and had
136 rejected/failed steps; the remaining eight actions were valid tool operations,
not correct answers. The fixed baseline made 24 model calls. Nested JSON input
copying, truncated JSON and invented read IDs were common failure modes. V1's
baseline prompt still offered retrieval actions despite its one remaining action;
V2 explicitly corrected that design. This is failure evidence for this specific
prompt/model/protocol, not a universal statement about the model.

The original exact traces remain in `agent-run/`; no failed cases were discarded.

## V2 constrained controller

Pinned Qwen2.5-0.5B-Instruct, CPU float32, no paid API:

| Mode | Completed | Correct labels | Correct label + complete gold evidence | Gold-sentence citation precision |
|---|---:|---:|---:|---:|
| Model-guided agent | 24/24 | 5/24 (20.8%) | 2/24 (8.3%) | 3/22 (13.6%) |
| Fixed top-3 + same model | 24/24 | 7/24 (29.2%) | 2/24 (8.3%) | 2/24 (8.3%) |
| Always abstain, no model | 24/24 | 8/24 (33.3%) | 8/24 by the NEI convention | Not applicable |

Correct abstentions count as complete in the scoring convention; they do not
establish corpus-wide absence of evidence. Of the agent's two complete outcomes,
one is a gold-evidence SUPPORT answer and one is a correct NEI abstention. The
agent makes 51 real model forward passes; the fixed baseline makes 24. The agent's
forced initial search is not a model decision. In per-case constrained JSON, the
legacy `model_calls` field includes policy scaffolding; use `model_forward_passes`
for actual inference counts. Both models have zero invalid
actions because their candidate menus are valid by construction.

**The model-guided agent underperforms both baselines on label accuracy.**
Workflow completion is not a reasoning or factuality success rate.

Paired labels: both correct 4; agent only 1; fixed only 3; both wrong 16.
Per-class and per-case decomposition is in `agent-v2-dev/diagnostics.json`.

## V2 failure decomposition

Among the 16 evidence-bearing cases:

- Fixed top-3 retrieval contains a gold document in 9/16.
- The agent actually reads a gold document in 7/16.
- The lexical sentence candidate menu could cover a complete gold set in 7/16,
  even with an oracle choice. This uses gold labels only for post-hoc diagnosis,
  never in the policy.
- None of these 16 require multiple sentences in every gold evidence alternative;
  the general single-sentence citation limitation does not explain this subset's
  failures by itself.

Retrieval coverage, document choice, sentence preselection and model decisions all
need improvement. Citation membership checks cannot repair semantic mistakes.

## Train-only iteration and option sensitivity

Six distinct SciFact training claims were used for usability checks, not to claim
generalization. Initial constrained policy: agent 0/6 correct labels, with premature
abstentions. Requiring a source read and permuting options produced 3/6 correct
labels (but 0/6 evidence-complete) for seed 2026. No dev answers were used to tune
that repair.

Changing only the option-order seed to 2027 flips **5/6 agent verdicts** and 3/6
fixed-baseline verdicts. Agent label accuracy becomes 2/6; evidence completeness
remains 0/6. Both order runs are retained. This is serious positional sensitivity,
not a robust accuracy improvement. Seed 2026 was kept for the frozen development
run; no best-seed selection was made from the dev results.

## V3: one fixed capacity experiment

The official pinned Qwen2.5-1.5B-Instruct model (revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`) uses CPU BF16. The 24 cases, candidate
builder, option-order seed, retrieval and scoring are unchanged. The experiment
was frozen at `583869a` after the 0.5B findings. Both model size and dtype change;
this is a developmental capacity comparison, not an untouched-test result.

| Mode | Completed | Correct labels | Correct label + complete gold evidence | Gold-sentence citation precision |
|---|---:|---:|---:|---:|
| 1.5B model-guided agent | 24/24 | 11/24 (45.8%) | 5/24 (20.8%) | 5/21 (23.8%) |
| Fixed top-3 + same 1.5B model | 24/24 | 7/24 (29.2%) | 3/24 (12.5%) | 4/22 (18.2%) |
| Always abstain | 24/24 | 8/24 (33.3%) | 8/24 by the NEI convention | Not applicable |

The agent improves from 5 to 11 correct labels relative to the 0.5B controller on
these same cases. It beats the fixed model baseline by four labels and the trivial
abstention baseline by three. This small development sample does not establish a
reliable general accuracy gain. Five complete outcomes comprise three correct
SUPPORT evidence sets and two correct abstentions; no CONTRADICT answer has a
complete correct evidence set.

Paired with fixed retrieval: both correct 3, agent only 8, baseline only 4, both
wrong 9. The agent makes 48 real model forward passes; fixed retrieval makes 24.
Every agent trajectory reads exactly one document then finishes: the loop supports
additional reads, but **this run does not demonstrate multi-hop investigation**.
The agent reads a gold document in 8/16 evidence-bearing cases, versus 9/16 available
in fixed top-3 retrieval. The lexical candidate oracle remains 7/16. Invalid actions
remain zero by construction. Option-order sensitivity was not repeated for 1.5B;
the earlier 0.5B sensitivity result must not be assumed resolved.

At 45.8% label accuracy and 20.8% evidence-complete outcomes, the larger controller
is still not suitable for trusted scientific answers. The useful demonstrated
capability is an inspectable, recoverable, model-guided evidence workflow with
honest baselines and error attribution, not expert-level reasoning.

## Reproduction and limitations

See `../docs/evidence-agent.md`, `agent-environment.json`, each run's `protocol.json`
and `results.json`. Training traces preserve earlier policy variants; source code
hashes identify which variant generated them. Timing was collected alongside
concurrent CPU jobs and is descriptive, not an isolated latency/throughput test.

Synthetic fault/capability tests are separate from real-model correctness. The
runtime checks durable pending actions, interrupted read-only replay, serialized
same-run execution across two processes, configuration binding, bounded invalid
attempts and citation membership. The optional MCP adapter has real SDK stdio
handshake/search/read/error-recovery tests; it is not wired into the agent loop.

Recommendation for both tested capacities: use the project as an auditable agent-systems and
evaluation implementation, **not** a trusted scientific-answer agent. Do not put
production reasoning quality, prompt-injection resistance or an accuracy gain on
a CV based on these results.

## Real-model lifecycle check

`agent-recovery/report.json` records a separate public train-claim smoke, not an
accuracy benchmark. One process exits with code75 after a real Qwen-selected read
is persisted but before tool execution. A second process resumes the pending read,
finishes the workflow, and verifies the persisted model decision was not regenerated.
All six recorded recovery checks pass. Synthetic fault tests remain separately
identified; this one smoke does not demonstrate distributed recovery or arbitrary
side-effect safety.

## Retrieval integration gap: post-hoc diagnostic

The agent prototype reused the simplest existing word TF-IDF `SparseRetriever`
as a transparent shared reference backend for the agent and fixed baseline. That
was a scope choice, **not evidence that word-only retrieval was best**. The main
RAG's already implemented hybrid word/character retriever was not integrated into
the agent. This is a documented integration gap, not a new accuracy claim.

A retrieval-only comparison now uses the exact frozen 24 agent queries, the same
full corpus and identical title/abstract construction. Rankings are recorded for
all 24; metrics below use only 16 evidence-bearing cases because 8 NEI cases have no
gold documents. No model generation, prompt tuning or frozen-result changes occur.

| Backend / cutoff | Cases with any gold document | Macro document Recall@k |
|---|---:|---:|
| Measured agent: word top3 | 9/16 (56.25%) | 52.08% |
| Word top10 | 13/16 (81.25%) | 79.17% |
| Existing hybrid top3 | 10/16 (62.50%) | 60.42% |
| Existing hybrid top10 | 14/16 (87.50%) | 85.42% |

All 24 recomputed word-top3 rankings exactly match the stored fixed-baseline tool
outputs. Hybrid uses the existing 0.65 word/0.35 character weights without tuning.
Hybrid top3 recovers a gold document on one additional evidence-bearing case;
widening the cutoff contributes more coverage, but it also changes the tool menu,
context/choice budget and inference problem. Retrieval gains alone do not prove
answer gains. The measured agent remains word-top3; no selectable backend was
added in this release, and no further model run was performed.

The approximately 90% hybrid Recall@10 reported by the earlier 188-query RAG study
**does not describe this agent**. Different query subsets and cutoffs must not be
combined. Reproduction: `src.evidence_agent_retrieval_diagnostic.py`; complete
rankings, hashes and metrics: `agent-retrieval-integration.json`.

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
