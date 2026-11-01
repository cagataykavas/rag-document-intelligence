from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.entailment_release_audit import (
    ReleaseAnchorAuditError,
    audit_retained_release_head_anchor,
)
from src.entailment_release_retention import (
    ReleaseAnchorRetentionError,
    dump_retained_release_anchor_receipt,
    load_retained_release_anchor_receipt,
)
from tests.test_entailment_release_anchor import KEY_ID, SOURCE_COMMIT
from tests.test_entailment_release_retention import (
    BUCKET,
    KMS_KEY_ARN,
    NOW,
    OWNER,
    RETAIN_UNTIL,
    Client,
    anchor_fixture,
    retain,
)

AUDIT_AT = NOW + 86_400
KEY_PREFIX = "rag/release-heads"


class Body:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0
        self.closed = False

    def read(self, size: int) -> bytes:
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class AuditClient:
    def __init__(self, *, receipt, anchor_payload: bytes) -> None:
        self.receipt = receipt
        self.anchor_payload = anchor_payload
        self.body = Body(anchor_payload)
        self.get_calls: list[dict[str, object]] = []
        self.retention_calls: list[dict[str, object]] = []

    def get_object(self, **kwargs: object) -> dict[str, object]:
        self.get_calls.append(kwargs)
        checksum = base64.b64encode(
            hashlib.sha256(self.anchor_payload).digest()
        ).decode("ascii")
        return {
            "Body": self.body,
            "VersionId": self.receipt.version_id,
            "ContentLength": len(self.anchor_payload),
            "ContentType": "application/json",
            "ChecksumSHA256": checksum,
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": KMS_KEY_ARN,
            "Metadata": {
                "anchor-sha256": self.receipt.anchor_sha256,
                "head-sha256": self.receipt.head_sha256,
                "schema-version": self.receipt.schema_version,
            },
        }

    def get_object_retention(self, **kwargs: object) -> dict[str, object]:
        self.retention_calls.append(kwargs)
        return {
            "Retention": {
                "Mode": "COMPLIANCE",
                "RetainUntilDate": datetime.fromtimestamp(
                    RETAIN_UNTIL, tz=timezone.utc
                ),
            }
        }


def audit_fixture(tmp_path):
    anchor, log_payload, public_der = anchor_fixture(tmp_path)
    writer = Client()
    receipt = retain(writer, anchor, log_payload, public_der)
    anchor_payload = writer.put_calls[0]["Body"]
    assert isinstance(anchor_payload, bytes)
    client = AuditClient(receipt=receipt, anchor_payload=anchor_payload)
    return anchor, log_payload, public_der, receipt, client


def audit(client, receipt, log_payload, public_der, **overrides):
    kwargs = {
        "public_keys": {KEY_ID: public_der},
        "expected_source_commit_sha": SOURCE_COMMIT,
        "expected_anchor_sha256": receipt.anchor_sha256,
        "expected_bucket": BUCKET,
        "expected_key_prefix": KEY_PREFIX,
        "expected_kms_key_arn": KMS_KEY_ARN,
        "expected_bucket_owner": OWNER,
        "audit_at_epoch": AUDIT_AT,
    }
    kwargs.update(overrides)
    return audit_retained_release_head_anchor(
        client,
        dump_retained_release_anchor_receipt(receipt),
        log_payload,
        **kwargs,
    )


def test_auditor_reads_exact_version_and_authenticates_complete_chain(tmp_path) -> None:
    anchor, log_payload, public_der, receipt, client = audit_fixture(tmp_path)

    result = audit(client, receipt, log_payload, public_der)

    exact_coordinate = {
        "Bucket": BUCKET,
        "Key": receipt.object_key,
        "VersionId": receipt.version_id,
        "ExpectedBucketOwner": OWNER,
    }
    assert client.get_calls == [{**exact_coordinate, "ChecksumMode": "ENABLED"}]
    assert client.retention_calls == [exact_coordinate]
    assert client.body.closed is True
    assert result.status == "passed"
    assert result.anchor_sha256 == anchor.anchor_sha256
    assert result.head_sha256 == anchor.head_sha256
    assert result.retain_until_epoch == RETAIN_UNTIL


def test_receipt_round_trip_is_canonical_and_rejects_ambiguous_json(tmp_path) -> None:
    _, _, _, receipt, _ = audit_fixture(tmp_path)
    payload = dump_retained_release_anchor_receipt(receipt)

    assert load_retained_release_anchor_receipt(payload) == receipt
    assert (
        dump_retained_release_anchor_receipt(
            load_retained_release_anchor_receipt(payload)
        )
        == payload
    )
    duplicate = payload[:-1] + b',"bucket":"other-bucket"}'
    with pytest.raises(ReleaseAnchorRetentionError, match="duplicate"):
        load_retained_release_anchor_receipt(duplicate)
    with pytest.raises(ReleaseAnchorRetentionError, match="canonical"):
        load_retained_release_anchor_receipt(b" " + payload)
    unsupported = payload.replace(b'"COMPLIANCE"', b'"GOVERNANCE"')
    with pytest.raises(ReleaseAnchorRetentionError, match="malformed"):
        load_retained_release_anchor_receipt(unsupported)
    with pytest.raises(ReleaseAnchorRetentionError, match="size"):
        load_retained_release_anchor_receipt(b"{" + b"x" * 8_192)


@pytest.mark.parametrize(
    "field",
    [
        "anchor_sha256",
        "checksum_sha256",
        "bucket",
        "object_key",
        "kms_key_id_sha256",
        "expected_bucket_owner_sha256",
        "retain_until_epoch",
    ],
)
def test_receipt_policy_substitution_fails_before_s3(tmp_path, field: str) -> None:
    _, log_payload, public_der, receipt, client = audit_fixture(tmp_path)
    replacements = {
        "anchor_sha256": "0" * 64,
        "checksum_sha256": "0" * 64,
        "bucket": "other-release-bucket",
        "object_key": f"{KEY_PREFIX}/{'0' * 64}.json",
        "kms_key_id_sha256": "0" * 64,
        "expected_bucket_owner_sha256": "0" * 64,
        "retain_until_epoch": AUDIT_AT,
    }
    candidate = replace(receipt, **{field: replacements[field]})

    with pytest.raises(ReleaseAnchorAuditError, match="receipt"):
        audit(client, candidate, log_payload, public_der)

    assert client.get_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("VersionId", "other-version"),
        ("ContentLength", 1),
        ("ContentType", "text/plain"),
        ("ChecksumSHA256", "wrong"),
        ("ServerSideEncryption", "AES256"),
        ("SSEKMSKeyId", "wrong"),
        ("DeleteMarker", True),
        ("Metadata", {}),
    ],
)
def test_object_version_metadata_mismatch_fails_closed(
    tmp_path, field: str, value: object
) -> None:
    _, log_payload, public_der, receipt, client = audit_fixture(tmp_path)
    original = client.get_object

    def get_object(**kwargs):
        response = original(**kwargs)
        response[field] = value
        return response

    client.get_object = get_object

    with pytest.raises(ReleaseAnchorAuditError, match="response is malformed"):
        audit(client, receipt, log_payload, public_der)

    assert client.body.closed is True
    assert client.retention_calls == []


def test_retained_body_must_be_canonical_authenticated_anchor(tmp_path) -> None:
    _, log_payload, public_der, receipt, client = audit_fixture(tmp_path)
    document = json.loads(client.anchor_payload)
    document["signature"] = "A" * 86
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    client.anchor_payload = payload
    client.body = Body(payload)
    receipt = replace(
        receipt,
        anchor_sha256=hashlib.sha256(payload).hexdigest(),
        checksum_sha256=hashlib.sha256(payload).hexdigest(),
        object_key=f"{KEY_PREFIX}/{hashlib.sha256(payload).hexdigest()}.json",
    )
    client.receipt = receipt

    with pytest.raises(ReleaseAnchorAuditError, match="authentication failed"):
        audit(client, receipt, log_payload, public_der)

    assert client.retention_calls == []


@pytest.mark.parametrize(
    "retention",
    [
        None,
        {"Mode": "GOVERNANCE", "RetainUntilDate": datetime.now(timezone.utc)},
        {
            "Mode": "COMPLIANCE",
            "RetainUntilDate": datetime.fromtimestamp(AUDIT_AT, tz=timezone.utc),
        },
        {"Mode": "COMPLIANCE", "RetainUntilDate": "not-a-date"},
    ],
)
def test_retention_readback_requires_live_compliance_policy(
    tmp_path, retention: object
) -> None:
    _, log_payload, public_der, receipt, client = audit_fixture(tmp_path)
    client.get_object_retention = lambda **kwargs: {"Retention": retention}

    with pytest.raises(ReleaseAnchorAuditError, match="retention"):
        audit(client, receipt, log_payload, public_der)


def test_bounded_body_is_closed_when_oversized(tmp_path) -> None:
    _, log_payload, public_der, receipt, client = audit_fixture(tmp_path)
    client.body = Body(b"x" * 4_097)

    with pytest.raises(ReleaseAnchorAuditError, match="body size"):
        audit(client, receipt, log_payload, public_der)

    assert client.body.closed is True


@pytest.mark.parametrize("operation", ["get_object", "get_object_retention"])
def test_provider_errors_do_not_copy_aws_diagnostics(tmp_path, operation: str) -> None:
    _, log_payload, public_der, receipt, client = audit_fixture(tmp_path)

    def fail(**kwargs):
        raise RuntimeError("sensitive AWS request identifiers")

    setattr(client, operation, fail)
    with pytest.raises(ReleaseAnchorAuditError) as exc:
        audit(client, receipt, log_payload, public_der)

    assert "sensitive" not in str(exc.value)
