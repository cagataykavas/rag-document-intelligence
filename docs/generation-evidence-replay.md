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

## Encrypted operator CLI

`src.generation_replay_cli` provides the operator boundary for retained private
inputs. It accepts only an AES-256-GCM envelope, authenticates the envelope before
parsing its plaintext, reruns the replay verifier, and creates a new owner-only
receipt without overwriting an existing review artifact.

Decryption and admission-verification keys are loaded from separate environment
keyrings so keys never appear in command history. Each value is a padded URL-safe
base64 encoding of exactly 32 bytes:

```bash
export RAG_REPLAY_DECRYPT_KEYS='{"replay-2026-q4":"<base64-key>"}'
export RAG_ADMISSION_VERIFY_KEYS='{"prod-2026-q4":"<base64-key>"}'

python -m src.generation_replay_cli retained.bundle \
  --receipt review/receipt.json
```

The admission keyring is required only when the private bundle contains an applied
authentication envelope. Both keyrings are bounded to 16 keys and 16 KiB. In a
production deployment, inject them from a secrets manager into the short-lived
process environment and clear that environment after execution.

The encrypted file must be owned by the operator process, be a regular single-link
file, have no group or other permissions, and be at most 64 MiB. Symlinks, hard links,
unsafe permissions,
duplicate JSON fields, unknown fields, unrecognized key IDs, noncanonical evidence
rows, and invalid GCM tags fail closed. The receipt is created with mode `0600`,
synced before success, and never contains the payload, evidence text, chunk IDs,
source names, or keys.

For an explicitly approved retention workflow, the operator can request deletion:

```bash
python -m src.generation_replay_cli retained.bundle \
  --receipt review/receipt.json \
  --delete-private-bundle-after-success
```

Deletion occurs only after the receipt and its parent directory have been synced;
the bundle's parent directory is also synced after unlinking.
Immediately before unlinking, the CLI verifies that the path still names the exact
device and inode it opened. A failed verification or receipt write leaves the bundle
in place. Unlinking is not cryptographic erasure: filesystem snapshots, backups,
journals, page caches, and storage replicas can retain plaintext or ciphertext, so
their lifecycle must be governed separately.

### Bundle contract

Producers can use `seal_private_replay_bundle` or implement the same contract. The
outer JSON has exactly `schema_version`, `key_id`, `nonce_b64`, and
`ciphertext_b64`. AES-GCM additional authenticated data is the canonical JSON object
containing the outer schema version and key ID. Nonces must be 12 random bytes and
must never repeat for the same key.

The decrypted `generation-replay-private-bundle/v1` object contains exactly:

- `payload`: the original model payload;
- `evidence`: exact `chunk_id`, `source`, `text`, and `source_locator` rows;
- `expected_admission_audit`: the original content-minimizing audit;
- `authentication`: the original authentication envelope or `null`.

This CLI writes a durable standalone receipt. It does not yet append receipts to a
multi-writer review ledger or provide a retention scheduler; those controls belong
in a separately authenticated store with compare-and-set or transactional ordering.
