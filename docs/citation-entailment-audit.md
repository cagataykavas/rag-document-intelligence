# Citation entailment release audit

`src.citation_entailment` closes the gap between a syntactically valid RAG citation and semantic support. It consumes claim-level NLI evidence produced against the exact cited chunks, then fails closed when claims are unsupported, citations are decorative, or cited evidence contradicts the answer.

Each versioned artifact binds the benchmark, generation model, prompt template, retrieval snapshot, evaluator model and evaluator revision. Claims carry content digests, a standard/critical risk tier and an exact citation set. Every citation must have one three-way entailment/neutral/contradiction verdict whose probabilities sum to one. The policy separately controls claim support, useful-citation rate, contradiction rate, critical-claim behavior, artifact freshness and resource budgets.

Reports contain exact aggregate metrics, stable finding codes and canonical SHA-256 identities for the artifact, configuration and every active policy threshold. Raw claim, chunk and benchmark identifiers are excluded. JSON parsing rejects duplicate fields, non-finite values, unknown schema fields and oversized artifacts. CLI exits are `0` for acceptance, `3` for a well-formed policy rejection and `2` for malformed evidence; `--output` uses atomic replacement.

```bash
python -m src.citation_entailment entailment-artifact.json --output entailment-report.json
```

## Trust boundaries and limitations

NLI probability is evaluator evidence, not ground truth. Thresholds require calibration on human-labelled claims for the deployment language, domain, claim length and risk tier. Evaluator bias, truncation and model/version drift can produce systematic errors. Per-citation verdicts also cannot prove support that emerges only from combining several passages.

The audit trusts the producer to evaluate the recorded claim and exact evidence digests. SHA-256 binds content but does not authenticate the evaluator. Production artifacts should be signed or MACed, retained with the retrieval snapshot, and periodically sampled for blinded human review. Passing semantic support does not prove source authority, temporal validity, answer completeness or overall factual correctness.

## Next integration step

Generate the artifact from the real query path after claim segmentation, calibrate thresholds on a held-out multilingual set, and append the accepted report digest to the response provenance record. Route critical or contradicted claims to abstention or human review before publishing the answer.
