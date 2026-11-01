from __future__ import annotations

import base64
import hashlib
import hmac
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from src.entailment_release_anchor import (
    dump_release_head_anchor,
    load_release_head_anchor,
    verify_release_head_anchor,
)
from src.entailment_release_retention import (
    RETAINED_RELEASE_ANCHOR_SCHEMA_VERSION,
    load_retained_release_anchor_receipt,
)

RETAINED_RELEASE_ANCHOR_AUDIT_SCHEMA_VERSION = "entailment-release-anchor-audit/v1"
_MAX_ANCHOR_BYTES = 4_096
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}")
_OWNER_RE = re.compile(r"[0-9]{12}")
_KMS_KEY_ARN_RE = re.compile(
    r"arn:(?:aws|aws-us-gov|aws-cn):kms:[a-z0-9-]{1,32}:[0-9]{12}:"
    r"key/[A-Za-z0-9-]{1,128}"
)
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


class ReleaseAnchorAuditError(ValueError):
    """A retained release anchor could not be independently authenticated."""

    code = "release_anchor_audit_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class RetainedReleaseAnchorAudit:
    schema_version: str
    status: str
    bucket: str
    object_key: str
    version_id: str
    anchor_sha256: str
    head_sha256: str
    checksum_sha256: str
    retention_mode: str
    retain_until_epoch: int
    audited_at_epoch: int

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_bucket(bucket: str) -> bool:
    if (
        not isinstance(bucket, str)
        or _BUCKET_RE.fullmatch(bucket) is None
        or ".." in bucket
        or ".-" in bucket
        or "-." in bucket
    ):
        return False
    try:
        octets = [int(part) for part in bucket.split(".")]
    except ValueError:
        return True
    return not (len(octets) == 4 and all(0 <= part <= 255 for part in octets))


def _validate_expectations(
    *,
    expected_anchor_sha256: str,
    expected_bucket: str,
    expected_key_prefix: str,
    expected_kms_key_arn: str,
    expected_bucket_owner: str,
    audit_at_epoch: int,
) -> None:
    if (
        not isinstance(expected_anchor_sha256, str)
        or _DIGEST_RE.fullmatch(expected_anchor_sha256) is None
        or not _validate_bucket(expected_bucket)
        or not isinstance(expected_key_prefix, str)
        or _PREFIX_RE.fullmatch(expected_key_prefix) is None
        or ".." in expected_key_prefix
        or "//" in expected_key_prefix
        or expected_key_prefix.endswith("/")
        or not isinstance(expected_kms_key_arn, str)
        or _KMS_KEY_ARN_RE.fullmatch(expected_kms_key_arn) is None
        or not isinstance(expected_bucket_owner, str)
        or _OWNER_RE.fullmatch(expected_bucket_owner) is None
        or type(audit_at_epoch) is not int
        or not 1 <= audit_at_epoch <= 253_402_300_799
    ):
        raise ReleaseAnchorAuditError("audit expectations are invalid")


def _close_body(body: object) -> None:
    close = getattr(body, "close", None)
    if not callable(close):
        raise ReleaseAnchorAuditError("S3 object body is malformed")
    try:
        close()
    except Exception as exc:
        raise ReleaseAnchorAuditError("S3 object body close failed") from exc


def _read_bounded_body(body: object) -> bytes:
    read = getattr(body, "read", None)
    if not callable(read):
        raise ReleaseAnchorAuditError("S3 object body is malformed")
    chunks: list[bytes] = []
    remaining = _MAX_ANCHOR_BYTES + 1
    read_failed: Exception | None = None
    try:
        while remaining:
            try:
                chunk = read(min(1_024, remaining))
            except Exception as exc:
                read_failed = exc
                break
            if not isinstance(chunk, bytes):
                raise ReleaseAnchorAuditError("S3 object body is malformed")
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        _close_body(body)
    if read_failed is not None:
        raise ReleaseAnchorAuditError("S3 object body read failed") from read_failed
    payload = b"".join(chunks)
    if not payload or len(payload) > _MAX_ANCHOR_BYTES:
        raise ReleaseAnchorAuditError("S3 object body size is invalid")
    return payload


def _retention_epoch(response: object, *, minimum_epoch: int) -> int:
    if not isinstance(response, Mapping):
        raise ReleaseAnchorAuditError("S3 retention response is malformed")
    retention = response.get("Retention")
    if not isinstance(retention, Mapping):
        raise ReleaseAnchorAuditError("S3 retention response is malformed")
    observed_until = retention.get("RetainUntilDate")
    if (
        retention.get("Mode") != "COMPLIANCE"
        or not isinstance(observed_until, datetime)
        or observed_until.tzinfo is None
    ):
        raise ReleaseAnchorAuditError("S3 retention policy mismatch")
    observed_epoch = int(observed_until.timestamp())
    if observed_epoch < minimum_epoch:
        raise ReleaseAnchorAuditError("S3 retention policy mismatch")
    return observed_epoch


def audit_retained_release_head_anchor(
    client: Any,
    receipt_payload: bytes,
    log_payload: bytes,
    *,
    public_keys: Mapping[str, bytes],
    expected_source_commit_sha: str,
    expected_anchor_sha256: str,
    expected_bucket: str,
    expected_key_prefix: str,
    expected_kms_key_arn: str,
    expected_bucket_owner: str,
    audit_at_epoch: int,
) -> RetainedReleaseAnchorAudit:
    """Authenticate one exact retained S3 version against independent expectations."""

    _validate_expectations(
        expected_anchor_sha256=expected_anchor_sha256,
        expected_bucket=expected_bucket,
        expected_key_prefix=expected_key_prefix,
        expected_kms_key_arn=expected_kms_key_arn,
        expected_bucket_owner=expected_bucket_owner,
        audit_at_epoch=audit_at_epoch,
    )
    try:
        receipt = load_retained_release_anchor_receipt(receipt_payload)
    except Exception as exc:
        raise ReleaseAnchorAuditError("retention receipt is invalid") from exc

    expected_key = f"{expected_key_prefix}/{expected_anchor_sha256}.json"
    kms_hash = hashlib.sha256(expected_kms_key_arn.encode("utf-8")).hexdigest()
    owner_hash = hashlib.sha256(expected_bucket_owner.encode("ascii")).hexdigest()
    if (
        receipt.schema_version != RETAINED_RELEASE_ANCHOR_SCHEMA_VERSION
        or receipt.bucket != expected_bucket
        or receipt.object_key != expected_key
        or not hmac.compare_digest(receipt.anchor_sha256, expected_anchor_sha256)
        or not hmac.compare_digest(receipt.checksum_sha256, expected_anchor_sha256)
        or not hmac.compare_digest(receipt.kms_key_id_sha256, kms_hash)
        or not hmac.compare_digest(receipt.expected_bucket_owner_sha256, owner_hash)
        or receipt.retention_mode != "COMPLIANCE"
        or receipt.retain_until_epoch <= audit_at_epoch
    ):
        raise ReleaseAnchorAuditError("retention receipt does not match expectations")

    try:
        response = client.get_object(
            Bucket=receipt.bucket,
            Key=receipt.object_key,
            VersionId=receipt.version_id,
            ExpectedBucketOwner=expected_bucket_owner,
            ChecksumMode="ENABLED",
        )
    except Exception as exc:
        raise ReleaseAnchorAuditError("S3 object version lookup failed") from exc
    if not isinstance(response, Mapping):
        raise ReleaseAnchorAuditError("S3 object version response is malformed")
    payload = _read_bounded_body(response.get("Body"))
    checksum = hashlib.sha256(payload).digest()
    checksum_hex = checksum.hex()
    expected_metadata = {
        "anchor-sha256": receipt.anchor_sha256,
        "head-sha256": receipt.head_sha256,
        "schema-version": receipt.schema_version,
    }
    if (
        response.get("DeleteMarker") is True
        or response.get("VersionId") != receipt.version_id
        or type(response.get("ContentLength")) is not int
        or response.get("ContentLength") != len(payload)
        or response.get("ContentType") != "application/json"
        or response.get("ChecksumSHA256") != base64.b64encode(checksum).decode("ascii")
        or response.get("ServerSideEncryption") != "aws:kms"
        or response.get("SSEKMSKeyId") != expected_kms_key_arn
        or response.get("Metadata") != expected_metadata
        or not hmac.compare_digest(checksum_hex, receipt.checksum_sha256)
    ):
        raise ReleaseAnchorAuditError("S3 object version response is malformed")

    try:
        anchor = load_release_head_anchor(payload)
        if (
            not hmac.compare_digest(anchor.anchor_sha256, receipt.anchor_sha256)
            or not hmac.compare_digest(anchor.head_sha256, receipt.head_sha256)
            or dump_release_head_anchor(anchor) != payload
        ):
            raise ReleaseAnchorAuditError("retained anchor does not match receipt")
        verify_release_head_anchor(
            anchor,
            log_payload,
            public_keys=public_keys,
            expected_source_commit_sha=expected_source_commit_sha,
        )
    except ReleaseAnchorAuditError:
        raise
    except Exception as exc:
        raise ReleaseAnchorAuditError("retained anchor authentication failed") from exc

    try:
        retention_response = client.get_object_retention(
            Bucket=receipt.bucket,
            Key=receipt.object_key,
            VersionId=receipt.version_id,
            ExpectedBucketOwner=expected_bucket_owner,
        )
    except Exception as exc:
        raise ReleaseAnchorAuditError("S3 retention lookup failed") from exc
    observed_until = _retention_epoch(
        retention_response,
        minimum_epoch=max(receipt.retain_until_epoch, audit_at_epoch + 1),
    )
    return RetainedReleaseAnchorAudit(
        schema_version=RETAINED_RELEASE_ANCHOR_AUDIT_SCHEMA_VERSION,
        status="passed",
        bucket=receipt.bucket,
        object_key=receipt.object_key,
        version_id=receipt.version_id,
        anchor_sha256=receipt.anchor_sha256,
        head_sha256=receipt.head_sha256,
        checksum_sha256=receipt.checksum_sha256,
        retention_mode="COMPLIANCE",
        retain_until_epoch=observed_until,
        audited_at_epoch=audit_at_epoch,
    )
