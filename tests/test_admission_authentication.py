import base64
from pathlib import Path

import pytest

from src.admission_authentication import (
    AdmissionAuthenticationError,
    AdmissionAuthenticator,
    verify_admission_authentication,
)
from src.engine import OpenAICompatibleGenerator, RAGEngine
from src.generation_policy import admit_generation
from src.parsers import ParsedDocument
from src.store import DocumentStore

KEY = b"k" * 32
OLD_KEY = b"o" * 32
ISSUED_AT = 1_800_000_000


def valid_payload(**overrides):
    payload = {
        "answer": "Audit records are retained for seven years.",
        "citations": ["doc-123:0"],
        "confidence": 0.91,
        "insufficient_evidence": False,
    }
    payload.update(overrides)
    return payload


def valid_audit() -> dict[str, object]:
    return admit_generation(valid_payload(), {"doc-123:0"}).to_audit_dict()


def assert_error(code: str, callback) -> None:
    with pytest.raises(AdmissionAuthenticationError) as error:
        callback()
    assert error.value.code == code
    assert str(error.value) == code


def test_authenticates_exact_audit_deterministically_without_generated_content():
    signer = AdmissionAuthenticator("2026-q4-primary", KEY, clock=lambda: ISSUED_AT)
    audit = valid_audit()

    first = signer.authenticate(audit)
    second = signer.authenticate(audit)

    assert first == second
    assert first["applied"] is True
    assert first["algorithm"] == "HMAC-SHA256"
    assert valid_payload()["answer"] not in repr(first)
    assert "doc-123:0" not in repr(first)
    verify_admission_authentication(
        audit,
        first,
        {"2026-q4-primary": KEY},
        now_epoch=ISSUED_AT + 30,
    )


def test_tampered_audit_field_is_rejected_even_when_decision_digest_is_unchanged():
    signer = AdmissionAuthenticator("primary", KEY, clock=lambda: ISSUED_AT)
    audit = valid_audit()
    authentication = signer.authenticate(audit)
    audit["payload_bytes"] = int(audit["payload_bytes"]) + 1

    assert_error(
        "AUDIT_DIGEST_MISMATCH",
        lambda: verify_admission_authentication(
            audit,
            authentication,
            {"primary": KEY},
            now_epoch=ISSUED_AT,
        ),
    )


def test_tampered_decision_digest_is_rejected_before_signing():
    audit = valid_audit()
    audit["accepted"] = False

    assert_error(
        "DECISION_DIGEST_MISMATCH",
        lambda: AdmissionAuthenticator("primary", KEY).authenticate(audit),
    )


def test_tampered_signature_and_unknown_key_fail_closed():
    signer = AdmissionAuthenticator("primary", KEY, clock=lambda: ISSUED_AT)
    audit = valid_audit()
    authentication = signer.authenticate(audit)
    authentication["signature"] = "0" * 64

    assert_error(
        "SIGNATURE_INVALID",
        lambda: verify_admission_authentication(
            audit,
            authentication,
            {"primary": KEY},
            now_epoch=ISSUED_AT,
        ),
    )
    authentication = signer.authenticate(audit)
    assert_error(
        "SIGNING_KEY_UNKNOWN",
        lambda: verify_admission_authentication(
            audit,
            authentication,
            {"old": OLD_KEY},
            now_epoch=ISSUED_AT,
        ),
    )


def test_keyring_supports_rotation_without_trying_every_secret():
    authentication = AdmissionAuthenticator(
        "old", OLD_KEY, clock=lambda: ISSUED_AT
    ).authenticate(valid_audit())

    verify_admission_authentication(
        valid_audit(),
        authentication,
        {"old": OLD_KEY, "primary": KEY},
        now_epoch=ISSUED_AT,
    )


@pytest.mark.parametrize(
    ("now_epoch", "code"),
    [
        (ISSUED_AT - 6, "SIGNATURE_FROM_FUTURE"),
        (ISSUED_AT + 306, "SIGNATURE_STALE"),
    ],
)
def test_online_verification_enforces_bounded_freshness(now_epoch: int, code: str):
    audit = valid_audit()
    authentication = AdmissionAuthenticator(
        "primary", KEY, clock=lambda: ISSUED_AT
    ).authenticate(audit)

    assert_error(
        code,
        lambda: verify_admission_authentication(
            audit,
            authentication,
            {"primary": KEY},
            now_epoch=now_epoch,
        ),
    )


def test_archival_verification_can_disable_age_policy_but_not_signature_checks():
    audit = valid_audit()
    authentication = AdmissionAuthenticator(
        "primary", KEY, clock=lambda: ISSUED_AT
    ).authenticate(audit)

    verify_admission_authentication(
        audit,
        authentication,
        {"primary": KEY},
        now_epoch=ISSUED_AT + 100_000,
        max_age_seconds=None,
    )


def test_environment_configuration_is_optional_complete_and_redacted():
    assert AdmissionAuthenticator.from_env({}) is None
    encoded = base64.b64encode(KEY).decode("ascii")
    signer = AdmissionAuthenticator.from_env(
        {
            "RAG_ADMISSION_HMAC_KEY_ID": "primary",
            "RAG_ADMISSION_HMAC_KEY_B64": encoded,
        }
    )

    assert signer is not None
    assert signer.key_id == "primary"
    assert encoded not in repr(signer)
    assert "redacted" in repr(signer)
    assert_error(
        "SIGNING_CONFIG_INCOMPLETE",
        lambda: AdmissionAuthenticator.from_env(
            {"RAG_ADMISSION_HMAC_KEY_ID": "primary"}
        ),
    )


@pytest.mark.parametrize(
    ("key_id", "key"),
    [("bad key id", KEY), ("primary", b"short")],
)
def test_invalid_signing_identity_or_weak_key_is_rejected(key_id: str, key: bytes):
    assert_error(
        "KEY_ID_INVALID" if key_id == "bad key id" else "SIGNING_KEY_INVALID",
        lambda: AdmissionAuthenticator(key_id, key),
    )


def build_store(path: Path) -> tuple[DocumentStore, str]:
    store = DocumentStore(path)
    store.ingest(
        ParsedDocument(
            "policy.md",
            "text/markdown",
            "Audit records must be retained for seven years for compliance reviews.",
            {},
        ),
        words_per_chunk=30,
        overlap=5,
    )
    return store, store.chunks()[0].chunk_id


@pytest.mark.parametrize("accepted", [True, False])
def test_engine_authenticates_both_accepted_and_rejected_llm_decisions(
    tmp_path: Path, monkeypatch, accepted: bool
):
    store, safe_id = build_store(tmp_path / "rag.db")
    signer = AdmissionAuthenticator("primary", KEY, clock=lambda: ISSUED_AT)
    engine = RAGEngine(store, admission_authenticator=signer)
    citation = safe_id if accepted else "invented:0"
    monkeypatch.setattr(
        OpenAICompatibleGenerator,
        "generate",
        lambda self, prompt: valid_payload(citations=[citation]),
    )

    answer = engine.answer("How long are audit records retained?", use_llm=True)

    assert answer.generation_admission["accepted"] is accepted
    assert answer.generation_admission_authentication["applied"] is True
    verify_admission_authentication(
        answer.generation_admission,
        answer.generation_admission_authentication,
        {"primary": KEY},
        now_epoch=ISSUED_AT,
    )
