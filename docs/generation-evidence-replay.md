# Generation evidence replay

`src/generation_evidence_replay.py` independently reconstructs an LLM generation
decision from the retained private payload, the exact post-quarantine evidence,
and the content-minimizing admission audit. It is intended for incident review,
release evidence, and offline control testing where trusting the original service
response would make the audit circular.

```python
from src.generation_evidence_replay import replay_generation_evidence

receipt = replay_generation_evidence(
    retained_model_payload,
    retained_safe_evidence,
    retained_admission_audit,
)
assert receipt["verified"] is True
```

Replay rebuilds the evidence snapshot from exact chunk IDs, sources, text bytes,
and optional PDF locators. It then reruns the strict generation policy. Accepted
answers are split into the same deterministic claims, traced to the strongest
cited source spans, and rebound to the decision digest. Rejected responses are
reproduced without inventing a claim-support binding. Any payload, evidence,
policy, or audit change produces a stable failure code.

## Authenticated records

Pass the original authentication envelope and rotation keyring to verify the HMAC
over the reconstructed audit:

```python
receipt = replay_generation_evidence(
    retained_model_payload,
    retained_safe_evidence,
    retained_admission_audit,
    authentication=retained_authentication,
    keyring={"prod-2026-q4": retired_verification_key},
)
```

The default is archival verification: signature age is not limited, while the
audit digest, named key, algorithm, purpose, and signature are still checked.
For an online control, pass `now_epoch` and `max_age_seconds` to enforce freshness.
Replay consumption is intentionally outside this offline function; online callers
that need one-time admission consumption should continue using
`verify_admission_authentication` with an `AdmissionReplayStore`.

## Receipt and privacy boundary

The `generation-evidence-replay/v1` receipt contains verification state, acceptance
state, authentication state, aggregate counts, the payload/evidence/claim/decision
digests, and a digest over the receipt itself. It does not copy answer text,
citation IDs, chunk IDs, source names, or evidence passages.

The private payload and evidence must still be retained by an authorized system if
future replay is required. The receipt cannot recover deleted inputs and does not
prove source authenticity, semantic entailment, factual correctness, or that the
chosen evidence was complete. HMAC keys authenticate the producer but do not provide
public-key non-repudiation.
