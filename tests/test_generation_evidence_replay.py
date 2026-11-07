from copy import deepcopy

import pytest

from src.admission_authentication import AdmissionAuthenticator
from src.claims import claim_support_sha256, trace_claim_support
from src.evidence_snapshot import build_evidence_snapshot
from src.generation_evidence_replay import (
    GenerationEvidenceReplayError,
    replay_generation_evidence,
)
from src.generation_policy import (
    CONTENT_BOUND_GENERATION_POLICY,
    GenerationPolicy,
    admit_generation,
    bind_claim_support,
)
from src.prompting import EvidenceSnippet

KEY = b"k" * 32


def evidence(text: str = "Audit records are retained for seven years."):
    return [EvidenceSnippet("policy:7", "policy.md", text)]


def payload(answer: str = "Audit records are retained for seven years."):
    return {
        "answer": answer,
        "citations": ["policy:7"],
        "claim_citations": [{"claim_index": 0, "citations": ["policy:7"]}],
        "confidence": 0.93,
        "insufficient_evidence": False,
    }


def audit_for(model_payload, snippets):
    snapshot = build_evidence_snapshot(snippets)
    admission = admit_generation(
        model_payload,
        snapshot.chunk_ids,
        CONTENT_BOUND_GENERATION_POLICY,
        evidence_snapshot_sha256=snapshot.snapshot_sha256,
    )
    if admission.accepted:
        support = trace_claim_support(
            admission.answer,
            admission.citations,
            [{"chunk_id": row.chunk_id, "text": row.text} for row in snippets],
            admission.claim_citations,
        )
        admission = bind_claim_support(admission, claim_support_sha256(support))
    return admission.to_audit_dict()


def test_replay_rebuilds_exact_content_and_returns_privacy_reduced_receipt():
    snippets = evidence()
    model_payload = payload()
    expected = audit_for(model_payload, snippets)

    receipt = replay_generation_evidence(model_payload, snippets, expected)
    rendered = repr(receipt)

    assert receipt["verified"] is True
    assert receipt["accepted"] is True
    assert receipt["authentication_verified"] is False
    assert receipt["payload_sha256"] == expected["payload_sha256"]
    assert receipt["decision_sha256"] == expected["decision_sha256"]
    assert receipt["receipt_sha256"]
    assert model_payload["answer"] not in rendered
    assert "policy:7" not in rendered
    assert "policy.md" not in rendered


@pytest.mark.parametrize(
    ("changed_payload", "changed_evidence"),
    [
        (payload("Audit records are retained for six years."), evidence()),
        (payload(), evidence("Audit records are retained for six years.")),
        (payload(), [EvidenceSnippet("policy:7", "revised.md", evidence()[0].text)]),
    ],
)
def test_replay_rejects_payload_or_evidence_substitution(
    changed_payload, changed_evidence
):
    expected = audit_for(payload(), evidence())

    with pytest.raises(GenerationEvidenceReplayError) as raised:
        replay_generation_evidence(changed_payload, changed_evidence, expected)

    assert raised.value.code == "ADMISSION_AUDIT_MISMATCH"


def test_replay_rejects_tampered_or_non_exact_audit_schema():
    expected = audit_for(payload(), evidence())
    tampered = deepcopy(expected)
    tampered["decision_sha256"] = "0" * 64
    with_extra = {**expected, "answer": "private answer"}

    with pytest.raises(GenerationEvidenceReplayError) as mismatch:
        replay_generation_evidence(payload(), evidence(), tampered)
    with pytest.raises(GenerationEvidenceReplayError) as invalid:
        replay_generation_evidence(payload(), evidence(), with_extra)

    assert mismatch.value.code == "ADMISSION_AUDIT_MISMATCH"
    assert invalid.value.code == "ADMISSION_AUDIT_INVALID"


def test_replay_reproduces_rejected_decisions_without_claim_binding():
    rejected_payload = payload()
    rejected_payload["citations"] = ["invented:1"]
    rejected_payload["claim_citations"] = [
        {"claim_index": 0, "citations": ["invented:1"]}
    ]
    expected = audit_for(rejected_payload, evidence())

    receipt = replay_generation_evidence(rejected_payload, evidence(), expected)

    assert receipt["accepted"] is False
    assert receipt["claim_support_sha256"] == expected["claim_support_sha256"]
    assert expected["claim_support_bound"] is False


def test_replay_optionally_verifies_authenticated_audit_for_archival_use():
    expected = audit_for(payload(), evidence())
    authentication = AdmissionAuthenticator(
        "archive-2026", KEY, clock=lambda: 1_700_000_000
    ).authenticate(expected, replay_id="a" * 32)

    receipt = replay_generation_evidence(
        payload(),
        evidence(),
        expected,
        authentication=authentication,
        keyring={"archive-2026": KEY},
    )

    assert receipt["authentication_verified"] is True


def test_replay_fails_closed_on_missing_keyring_or_invalid_signature():
    expected = audit_for(payload(), evidence())
    authentication = AdmissionAuthenticator("primary", KEY).authenticate(expected)

    with pytest.raises(GenerationEvidenceReplayError) as missing:
        replay_generation_evidence(
            payload(), evidence(), expected, authentication=authentication
        )
    with pytest.raises(GenerationEvidenceReplayError) as invalid:
        replay_generation_evidence(
            payload(),
            evidence(),
            expected,
            authentication=authentication,
            keyring={"primary": b"x" * 32},
        )

    assert missing.value.code == "KEYRING_REQUIRED"
    assert invalid.value.code == "AUTHENTICATION_INVALID"

    with pytest.raises(GenerationEvidenceReplayError) as malformed:
        replay_generation_evidence(
            payload(), evidence(), expected, authentication={"applied": False, "x": 1}
        )
    assert malformed.value.code == "AUTHENTICATION_INVALID"


def test_replay_rejects_invalid_evidence_and_policy_inputs():
    expected = audit_for(payload(), evidence())

    with pytest.raises(GenerationEvidenceReplayError) as invalid_evidence:
        replay_generation_evidence(payload(), [], expected)
    with pytest.raises(GenerationEvidenceReplayError) as invalid_policy:
        replay_generation_evidence(payload(), evidence(), expected, policy=object())
    changed_policy = GenerationPolicy(
        max_answer_chars=100,
        require_content_bound_evidence=True,
    )
    with pytest.raises(GenerationEvidenceReplayError) as policy_mismatch:
        replay_generation_evidence(
            payload(), evidence(), expected, policy=changed_policy
        )

    assert invalid_evidence.value.code == "EVIDENCE_INVALID"
    assert invalid_policy.value.code == "POLICY_INVALID"
    assert policy_mismatch.value.code == "ADMISSION_AUDIT_MISMATCH"
