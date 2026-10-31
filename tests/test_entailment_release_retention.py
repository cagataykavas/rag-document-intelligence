from __future__ import annotations

import base64
import hashlib
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.entailment_release_anchor import create_release_head_anchor
from src.entailment_release_retention import (
    ReleaseAnchorRetentionError,
    retain_verified_release_head_anchor,
)
from tests.test_entailment_release_anchor import (
    KEY_ID,
    SOURCE_COMMIT,
    LocalSigner,
    key_material,
    release_log_payload,
)

BUCKET = "rag-release-anchors-prod"
OWNER = "123456789012"
KMS_KEY_ARN = (
    "arn:aws:kms:eu-central-1:123456789012:key/11111111-2222-3333-4444-555555555555"
)
NOW = 1_800_000_000
RETAIN_UNTIL = NOW + 90 * 86_400


def anchor_fixture(tmp_path):
    payload = release_log_payload(tmp_path)
    private_key, public_der = key_material()
    anchor = create_release_head_anchor(
        payload,
        key_id=KEY_ID,
        source_commit_sha=SOURCE_COMMIT,
        issued_at_epoch=NOW,
        signer=LocalSigner(private_key),
    )
    return anchor, payload, public_der


class Client:
    def __init__(self) -> None:
        self.put_calls: list[dict[str, object]] = []
        self.retention_calls: list[dict[str, object]] = []

    def put_object(self, **kwargs: object) -> dict[str, object]:
        self.put_calls.append(kwargs)
        return {
            "ChecksumSHA256": kwargs["ChecksumSHA256"],
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": KMS_KEY_ARN,
            "VersionId": "3/L4kqtJlcpXroDTDmJ+rmSpXd3QBpU",
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


def retain(client, anchor, payload, public_der):
    return retain_verified_release_head_anchor(
        client,
        anchor,
        payload,
        public_keys={KEY_ID: public_der},
        expected_source_commit_sha=SOURCE_COMMIT,
        bucket=BUCKET,
        key_prefix="rag/release-heads",
        kms_key_arn=KMS_KEY_ARN,
        expected_bucket_owner=OWNER,
        now_epoch=NOW,
        retain_until_epoch=RETAIN_UNTIL,
    )


def test_retain_verified_anchor_pins_checksum_encryption_and_exact_version(
    tmp_path,
) -> None:
    anchor, payload, public_der = anchor_fixture(tmp_path)
    client = Client()

    receipt = retain(client, anchor, payload, public_der)

    anchor_payload = client.put_calls[0]["Body"]
    expected_checksum = hashlib.sha256(anchor_payload).digest()
    expected_key = f"rag/release-heads/{expected_checksum.hex()}.json"
    assert client.put_calls == [
        {
            "Bucket": BUCKET,
            "Key": expected_key,
            "Body": anchor_payload,
            "ContentType": "application/json",
            "ChecksumAlgorithm": "SHA256",
            "ChecksumSHA256": base64.b64encode(expected_checksum).decode("ascii"),
            "IfNoneMatch": "*",
            "ObjectLockMode": "COMPLIANCE",
            "ObjectLockRetainUntilDate": datetime.fromtimestamp(
                RETAIN_UNTIL, tz=timezone.utc
            ),
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": KMS_KEY_ARN,
            "ExpectedBucketOwner": OWNER,
            "Metadata": {
                "anchor-sha256": expected_checksum.hex(),
                "head-sha256": anchor.head_sha256,
                "schema-version": "entailment-release-anchor-retention/v1",
            },
        }
    ]
    assert client.retention_calls == [
        {
            "Bucket": BUCKET,
            "Key": expected_key,
            "VersionId": "3/L4kqtJlcpXroDTDmJ+rmSpXd3QBpU",
            "ExpectedBucketOwner": OWNER,
        }
    ]
    assert receipt.object_key == expected_key
    assert receipt.version_id == "3/L4kqtJlcpXroDTDmJ+rmSpXd3QBpU"
    assert receipt.anchor_sha256 == anchor.anchor_sha256
    assert receipt.checksum_sha256 == expected_checksum.hex()
    assert receipt.retention_mode == "COMPLIANCE"
    assert "11111111" not in str(receipt.to_audit_dict())


def test_invalid_anchor_is_rejected_before_s3_write(tmp_path) -> None:
    anchor, payload, public_der = anchor_fixture(tmp_path)
    client = Client()

    with pytest.raises(ReleaseAnchorRetentionError, match="verification failed"):
        retain(client, replace(anchor, head_sha256="0" * 64), payload, public_der)

    assert client.put_calls == []


@pytest.mark.parametrize(
    "response",
    [
        None,
        {},
        {
            "ChecksumSHA256": "wrong",
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": KMS_KEY_ARN,
            "VersionId": "v1",
        },
        {
            "ChecksumSHA256": "valid",
            "ServerSideEncryption": "AES256",
            "SSEKMSKeyId": KMS_KEY_ARN,
            "VersionId": "v1",
        },
        {
            "ChecksumSHA256": "valid",
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": "wrong",
            "VersionId": "v1",
        },
    ],
)
def test_malformed_s3_write_response_fails_closed(tmp_path, response: object) -> None:
    anchor, payload, public_der = anchor_fixture(tmp_path)
    client = Client()
    client.put_object = lambda **kwargs: response

    with pytest.raises(ReleaseAnchorRetentionError, match="response is malformed"):
        retain(client, anchor, payload, public_der)


@pytest.mark.parametrize(
    "retention",
    [
        None,
        {},
        {
            "Mode": "GOVERNANCE",
            "RetainUntilDate": datetime.fromtimestamp(RETAIN_UNTIL, tz=timezone.utc),
        },
        {
            "Mode": "COMPLIANCE",
            "RetainUntilDate": datetime.fromtimestamp(
                RETAIN_UNTIL - 1, tz=timezone.utc
            ),
        },
        {"Mode": "COMPLIANCE", "RetainUntilDate": "not-a-datetime"},
    ],
)
def test_retention_readback_rejects_weaker_or_malformed_policy(
    tmp_path, retention: object
) -> None:
    anchor, payload, public_der = anchor_fixture(tmp_path)
    client = Client()
    client.get_object_retention = lambda **kwargs: {"Retention": retention}

    with pytest.raises(ReleaseAnchorRetentionError, match="retention"):
        retain(client, anchor, payload, public_der)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"bucket": "192.168.1.1"}, "bucket"),
        ({"key_prefix": "../anchors"}, "policy"),
        ({"kms_key_arn": "alias/rag-release"}, "policy"),
        ({"expected_bucket_owner": "123"}, "policy"),
        ({"retain_until_epoch": NOW + 60}, "policy"),
        ({"retain_until_epoch": NOW + 11 * 366 * 86_400}, "policy"),
        (
            {"now_epoch": 253_402_300_000, "retain_until_epoch": 253_402_400_000},
            "policy",
        ),
    ],
)
def test_retention_configuration_is_bounded(
    tmp_path, overrides: dict[str, object], error: str
) -> None:
    anchor, payload, public_der = anchor_fixture(tmp_path)
    kwargs = {
        "public_keys": {KEY_ID: public_der},
        "expected_source_commit_sha": SOURCE_COMMIT,
        "bucket": BUCKET,
        "key_prefix": "rag/release-heads",
        "kms_key_arn": KMS_KEY_ARN,
        "expected_bucket_owner": OWNER,
        "now_epoch": NOW,
        "retain_until_epoch": RETAIN_UNTIL,
    }
    kwargs.update(overrides)

    with pytest.raises(ReleaseAnchorRetentionError, match=error):
        retain_verified_release_head_anchor(Client(), anchor, payload, **kwargs)


def test_provider_errors_do_not_copy_aws_diagnostics(tmp_path) -> None:
    anchor, payload, public_der = anchor_fixture(tmp_path)

    class FailingClient(Client):
        def put_object(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("sensitive bucket and AWS request details")

    with pytest.raises(
        ReleaseAnchorRetentionError, match="S3 anchor write failed"
    ) as exc:
        retain(FailingClient(), anchor, payload, public_der)

    assert "sensitive" not in str(exc.value)
