from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from src.citation_entailment import EntailmentPolicy, audit_entailment, load_artifact, main

NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def verdict(citation_id, evidence_sha, entailment=0.90, neutral=0.08, contradiction=0.02):
    return {
        "citation_id": citation_id,
        "evidence_sha256": evidence_sha,
        "entailment_probability": entailment,
        "neutral_probability": neutral,
        "contradiction_probability": contradiction,
    }


def claim(claim_id="claim-secret-1", risk="standard", citations=None, verdicts=None):
    citations = citations or ["chunk-secret-1"]
    verdicts = verdicts or [verdict("chunk-secret-1", HASH_A)]
    return {
        "claim_id": claim_id,
        "claim_sha256": HASH_B,
        "risk": risk,
        "citations": citations,
        "verdicts": verdicts,
    }


def artifact():
    return {
        "schema_version": "rag-citation-entailment/v1",
        "benchmark_id": "benchmark-secret",
        "model_id": "model-v1",
        "prompt_template_id": "prompt-v1",
        "retrieval_snapshot_sha256": HASH_C,
        "evaluator_model_id": "nli-model-v1",
        "evaluator_revision": "revision-v1",
        "created_at": "2026-09-26T11:59:00Z",
        "claims": [
            claim(),
            claim(
                "claim-secret-2",
                "critical",
                ["chunk-secret-2"],
                [verdict("chunk-secret-2", HASH_B, 0.85, 0.10, 0.05)],
            ),
        ],
    }


def codes(report):
    return {finding.code for finding in report.findings}


def test_supported_claims_are_accepted_with_content_free_evidence():
    report = audit_entailment(artifact(), evaluated_at=NOW)

    assert report.accepted is True
    assert report.metrics == {
        "claim_count": 2,
        "critical_claim_count": 1,
        "citation_count": 2,
        "supported_claim_count": 2,
        "supported_citation_count": 2,
        "contradicted_claim_count": 0,
        "claim_support_rate": 1.0,
        "citation_support_rate": 1.0,
        "contradicted_claim_rate": 0.0,
        "mean_best_entailment": 0.875,
        "max_contradiction": 0.05,
        "artifact_age_seconds": 60.0,
    }
    encoded = json.dumps(report.as_dict())
    assert "secret" not in encoded
    assert len(report.policy_sha256) == 64


def test_canonical_digest_is_independent_of_object_key_order():
    value = artifact()
    reordered = {key: value[key] for key in reversed(value)}
    reordered["claims"] = [{key: item[key] for key in reversed(item)} for item in value["claims"]]
    assert (
        audit_entailment(value, evaluated_at=NOW).artifact_sha256
        == audit_entailment(reordered, evaluated_at=NOW).artifact_sha256
    )


def test_canonical_digest_is_independent_of_claim_and_citation_order():
    value = artifact()
    value["claims"][0]["citations"].append("chunk-secret-extra")
    value["claims"][0]["verdicts"].append(verdict("chunk-secret-extra", HASH_C, 0.75, 0.20, 0.05))
    reordered = deepcopy(value)
    reordered["claims"].reverse()
    reordered["claims"][1]["citations"].reverse()
    reordered["claims"][1]["verdicts"].reverse()

    assert (
        audit_entailment(value, evaluated_at=NOW).artifact_sha256
        == audit_entailment(reordered, evaluated_at=NOW).artifact_sha256
    )


def test_policy_and_configuration_digests_bind_all_release_inputs():
    baseline = audit_entailment(artifact(), evaluated_at=NOW)
    changed_policy = audit_entailment(
        artifact(), evaluated_at=NOW, policy=EntailmentPolicy(min_claim_entailment=0.71)
    )
    changed_config = artifact()
    changed_config["evaluator_revision"] = "revision-v2"

    assert baseline.policy_sha256 != changed_policy.policy_sha256
    assert (
        baseline.configuration_sha256
        != audit_entailment(changed_config, evaluated_at=NOW).configuration_sha256
    )


def test_unsupported_claim_fails_claim_and_aggregate_gates():
    value = artifact()
    value["claims"][0]["verdicts"] = [verdict("chunk-secret-1", HASH_A, 0.20, 0.75, 0.05)]
    report = audit_entailment(value, evaluated_at=NOW)

    assert {"CLAIM_UNSUPPORTED", "CLAIM_SUPPORT_RATE", "CITATION_SUPPORT_RATE"} <= codes(report)
    assert report.metrics["supported_claim_count"] == 1


def test_decorative_neutral_citation_lowers_citation_support_rate():
    value = artifact()
    value["claims"][0]["citations"].append("chunk-secret-neutral")
    value["claims"][0]["verdicts"].append(verdict("chunk-secret-neutral", HASH_C, 0.10, 0.88, 0.02))
    report = audit_entailment(
        value, evaluated_at=NOW, policy=EntailmentPolicy(min_citation_support_rate=0.8)
    )

    assert "CITATION_SUPPORT_RATE" in codes(report)
    assert report.metrics["claim_support_rate"] == 1.0


def test_contradiction_rejects_even_when_another_citation_entails():
    value = artifact()
    value["claims"][0]["citations"].append("chunk-secret-conflict")
    value["claims"][0]["verdicts"].append(
        verdict("chunk-secret-conflict", HASH_C, 0.05, 0.05, 0.90)
    )
    report = audit_entailment(value, evaluated_at=NOW)

    assert {"CLAIM_CONTRADICTED", "CONTRADICTED_CLAIM_RATE"} <= codes(report)
    assert report.metrics["supported_claim_count"] == 2


def test_critical_claim_controls_are_explicit_and_independent():
    value = artifact()
    value["claims"][1]["verdicts"] = [verdict("chunk-secret-2", HASH_B, 0.10, 0.10, 0.80)]
    report = audit_entailment(value, evaluated_at=NOW)
    assert {"CRITICAL_CLAIM_UNSUPPORTED", "CRITICAL_CLAIM_CONTRADICTED"} <= codes(report)

    relaxed = audit_entailment(
        value,
        evaluated_at=NOW,
        policy=EntailmentPolicy(
            min_claim_support_rate=0.5,
            min_citation_support_rate=0.5,
            max_contradicted_claim_rate=0.5,
            require_all_critical_supported=False,
            require_no_critical_contradiction=False,
        ),
    )
    assert relaxed.accepted is True
    assert "CRITICAL_CLAIM_UNSUPPORTED" not in codes(relaxed)
    assert "CRITICAL_CLAIM_CONTRADICTED" not in codes(relaxed)


def test_standard_claim_diagnostics_respect_configured_rate_tolerance():
    value = artifact()
    value["claims"][0]["verdicts"] = [verdict("chunk-secret-1", HASH_A, 0.20, 0.75, 0.05)]
    report = audit_entailment(
        value,
        evaluated_at=NOW,
        policy=EntailmentPolicy(
            min_claim_support_rate=0.5,
            min_citation_support_rate=0.5,
        ),
    )

    assert report.accepted is True
    finding = next(item for item in report.findings if item.code == "CLAIM_UNSUPPORTED")
    assert finding.blocking is False
    assert finding.as_dict()["severity"] == "diagnostic"


def test_stale_and_future_artifacts_fail_closed():
    stale = artifact()
    stale["created_at"] = "2026-09-25T11:59:00Z"
    assert "STALE_ARTIFACT" in codes(
        audit_entailment(
            stale, evaluated_at=NOW, policy=EntailmentPolicy(max_artifact_age_seconds=60)
        )
    )

    future = artifact()
    future["created_at"] = "2026-09-26T12:06:00Z"
    assert "FUTURE_ARTIFACT" in codes(audit_entailment(future, evaluated_at=NOW))


@pytest.mark.parametrize(
    "payload",
    [b"", b"[]", b'{"a":1,"a":2}', b'{"a":NaN}', b"\xff", b"{"],
)
def test_strict_loader_rejects_malformed_or_ambiguous_json(payload):
    with pytest.raises(ValueError):
        load_artifact(payload)


def test_strict_loader_enforces_byte_budget():
    payload = json.dumps(artifact()).encode()
    with pytest.raises(ValueError):
        load_artifact(payload, max_bytes=len(payload) - 1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version="v2"),
        lambda value: value.update(extra=True),
        lambda value: value.update(model_id="bad id"),
        lambda value: value.update(retrieval_snapshot_sha256="ABC"),
        lambda value: value.update(created_at="2026-09-26T11:59:00"),
        lambda value: value.update(claims=[]),
        lambda value: value["claims"][0].update(extra=True),
        lambda value: value["claims"][0].update(risk="unknown"),
        lambda value: value["claims"][0].update(citations=[]),
        lambda value: value["claims"][0].update(claim_sha256="A" * 64),
        lambda value: value["claims"][0]["verdicts"][0].update(extra=True),
        lambda value: value["claims"][0]["verdicts"][0].update(entailment_probability=True),
        lambda value: value["claims"][0]["verdicts"][0].update(entailment_probability=0.8),
    ],
)
def test_rejects_malformed_artifact_contract(mutate):
    value = artifact()
    mutate(value)
    with pytest.raises(ValueError):
        audit_entailment(value, evaluated_at=NOW)


def test_rejects_duplicate_claim_citation_and_verdict_identities():
    duplicate_claim = artifact()
    duplicate_claim["claims"].append(deepcopy(duplicate_claim["claims"][0]))
    with pytest.raises(ValueError):
        audit_entailment(duplicate_claim, evaluated_at=NOW)

    duplicate_citation = artifact()
    duplicate_citation["claims"][0]["citations"].append("chunk-secret-1")
    with pytest.raises(ValueError):
        audit_entailment(duplicate_citation, evaluated_at=NOW)

    mismatch = artifact()
    mismatch["claims"][0]["verdicts"][0]["citation_id"] = "another-chunk"
    with pytest.raises(ValueError):
        audit_entailment(mismatch, evaluated_at=NOW)


def test_resource_budgets_fail_closed():
    value = artifact()
    with pytest.raises(ValueError):
        audit_entailment(value, evaluated_at=NOW, policy=EntailmentPolicy(max_claims=1))
    with pytest.raises(ValueError):
        audit_entailment(value, evaluated_at=NOW, policy=EntailmentPolicy(max_total_verdicts=1))
    value["claims"][0]["citations"].append("chunk-extra")
    value["claims"][0]["verdicts"].append(verdict("chunk-extra", HASH_C))
    with pytest.raises(ValueError):
        audit_entailment(
            value, evaluated_at=NOW, policy=EntailmentPolicy(max_citations_per_claim=1)
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_claim_entailment": -0.1},
        {"max_citation_contradiction": float("nan")},
        {"probability_sum_tolerance": 0.1},
        {"max_claims": True},
        {"require_all_critical_supported": "yes"},
    ],
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        audit_entailment(artifact(), evaluated_at=NOW, policy=EntailmentPolicy(**kwargs))


def test_evaluated_at_must_be_timezone_aware():
    with pytest.raises(ValueError):
        audit_entailment(artifact(), evaluated_at=datetime(2026, 9, 26, 12, 0))


def test_findings_are_bounded_for_large_failed_artifacts():
    value = artifact()
    value["claims"] = [
        claim(
            f"claim-{index}",
            "critical",
            [f"chunk-{index}"],
            [verdict(f"chunk-{index}", HASH_A, 0.05, 0.05, 0.90)],
        )
        for index in range(300)
    ]
    report = audit_entailment(value, evaluated_at=NOW)
    assert len(report.findings) == 256
    assert report.findings[-1].code == "FINDINGS_TRUNCATED"


def test_cli_exposes_distinct_accept_reject_and_malformed_codes(tmp_path, capsys):
    source = tmp_path / "artifact.json"
    output = tmp_path / "report.json"
    current = artifact()
    current["created_at"] = datetime.now(UTC).isoformat()
    source.write_text(json.dumps(current), encoding="utf-8")
    assert main([str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["accepted"] is True

    current["claims"][0]["verdicts"] = [verdict("chunk-secret-1", HASH_A, 0.10, 0.80, 0.10)]
    source.write_text(json.dumps(current), encoding="utf-8")
    assert main([str(source)]) == 3
    assert json.loads(capsys.readouterr().out)["accepted"] is False

    source.write_text('{"duplicate":1,"duplicate":2}', encoding="utf-8")
    assert main([str(source)]) == 2
    malformed = json.loads(capsys.readouterr().out)
    assert malformed["error_code"] == "MALFORMED_ARTIFACT"
    assert "duplicate" not in json.dumps(malformed)
