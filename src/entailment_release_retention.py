from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from src.entailment_release_anchor import (
    ReleaseHeadAnchor,
    dump_release_head_anchor,
    verify_release_head_anchor,
)

RETAINED_RELEASE_ANCHOR_SCHEMA_VERSION = "entailment-release-anchor-retention/v1"
_RETENTION_MODE = "COMPLIANCE"
_MIN_RETENTION_SECONDS = 86_400
_MAX_RETENTION_SECONDS = 10 * 366 * 86_400
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_PREFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,254}")
_VERSION_ID_RE = re.compile(r"[A-Za-z0-9._=+/-]{1,1024}")
_OWNER_RE = re.compile(r"[0-9]{12}")
_KMS_KEY_ARN_RE = re.compile(
    r"arn:(?:aws|aws-us-gov|aws-cn):kms:[a-z0-9-]{1,32}:[0-9]{12}:"
    r"key/[A-Za-z0-9-]{1,128}"
)
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_OBJECT_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+=-]{0,1023}")
_MAX_RECEIPT_BYTES = 8_192
_RECEIPT_FIELDS = {
    "anchor_sha256",
    "bucket",
    "checksum_sha256",
    "expected_bucket_owner_sha256",
    "head_sha256",
    "kms_key_id_sha256",
    "object_key",
    "retain_until_epoch",
    "retention_mode",
    "schema_version",
    "version_id",
}


class ReleaseAnchorRetentionError(ValueError):
    """A signed release anchor could not be retained under the reviewed policy."""

    code = "release_anchor_retention_failed"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"{self.code}: {reason}")


@dataclass(frozen=True, slots=True)
class RetainedReleaseAnchorReceipt:
    schema_version: str
    bucket: str
    object_key: str
    version_id: str
    anchor_sha256: str
    head_sha256: str
    checksum_sha256: str
    retention_mode: str
    retain_until_epoch: int
    kms_key_id_sha256: str
    expected_bucket_owner_sha256: str

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)


def _canonical_receipt_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ReleaseAnchorRetentionError("receipt is not canonical JSON") from exc


def _reject_receipt_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseAnchorRetentionError("receipt contains a duplicate field")
        result[key] = value
    return result


def _validate_receipt_document(document: Mapping[str, object]) -> None:
    if set(document) != _RECEIPT_FIELDS:
        raise ReleaseAnchorRetentionError("receipt fields do not match the schema")
    _validate_bucket(document["bucket"])
    if (
        document["schema_version"] != RETAINED_RELEASE_ANCHOR_SCHEMA_VERSION
        or document["retention_mode"] != _RETENTION_MODE
        or not isinstance(document["object_key"], str)
        or _OBJECT_KEY_RE.fullmatch(document["object_key"]) is None
        or ".." in document["object_key"]
        or "//" in document["object_key"]
        or not isinstance(document["version_id"], str)
        or _VERSION_ID_RE.fullmatch(document["version_id"]) is None
        or type(document["retain_until_epoch"]) is not int
        or not 1 <= document["retain_until_epoch"] <= 253_402_300_799
        or any(
            not isinstance(document[field], str)
            or _DIGEST_RE.fullmatch(document[field]) is None
            for field in (
                "anchor_sha256",
                "head_sha256",
                "checksum_sha256",
                "kms_key_id_sha256",
                "expected_bucket_owner_sha256",
            )
        )
    ):
        raise ReleaseAnchorRetentionError("receipt fields are malformed")


def dump_retained_release_anchor_receipt(
    receipt: RetainedReleaseAnchorReceipt,
) -> bytes:
    """Serialize a retention receipt into its bounded canonical form."""

    if not isinstance(receipt, RetainedReleaseAnchorReceipt):
        raise TypeError("receipt must be a RetainedReleaseAnchorReceipt")
    document = receipt.to_audit_dict()
    _validate_receipt_document(document)
    payload = _canonical_receipt_json(document)
    if len(payload) > _MAX_RECEIPT_BYTES:
        raise ReleaseAnchorRetentionError("receipt exceeds the size limit")
    return payload


def load_retained_release_anchor_receipt(
    payload: bytes,
) -> RetainedReleaseAnchorReceipt:
    """Load a canonical receipt, rejecting ambiguous or oversized JSON."""

    if not isinstance(payload, bytes) or not 1 <= len(payload) <= _MAX_RECEIPT_BYTES:
        raise ReleaseAnchorRetentionError("receipt payload size is invalid")
    try:
        document = json.loads(
            payload.decode("ascii"), object_pairs_hook=_reject_receipt_duplicates
        )
    except ReleaseAnchorRetentionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseAnchorRetentionError("receipt is not valid JSON") from exc
    if not isinstance(document, Mapping):
        raise ReleaseAnchorRetentionError("receipt must be a JSON object")
    _validate_receipt_document(document)
    if _canonical_receipt_json(dict(document)) != payload:
        raise ReleaseAnchorRetentionError("receipt is not canonical JSON")
    return RetainedReleaseAnchorReceipt(**document)


def _validate_bucket(bucket: str) -> None:
    if (
        not isinstance(bucket, str)
        or _BUCKET_RE.fullmatch(bucket) is None
        or ".." in bucket
        or ".-" in bucket
        or "-." in bucket
    ):
        raise ReleaseAnchorRetentionError("S3 bucket is invalid")
    try:
        octets = [int(part) for part in bucket.split(".")]
    except ValueError:
        return
    if len(octets) == 4 and all(0 <= part <= 255 for part in octets):
        raise ReleaseAnchorRetentionError("S3 bucket is invalid")


def _validate_policy(
    *,
    bucket: str,
    key_prefix: str,
    kms_key_arn: str,
    expected_bucket_owner: str,
    now_epoch: int,
    retain_until_epoch: int,
) -> None:
    _validate_bucket(bucket)
    if (
        not isinstance(key_prefix, str)
        or _PREFIX_RE.fullmatch(key_prefix) is None
        or ".." in key_prefix
        or "//" in key_prefix
        or key_prefix.endswith("/")
        or not isinstance(kms_key_arn, str)
        or _KMS_KEY_ARN_RE.fullmatch(kms_key_arn) is None
        or not isinstance(expected_bucket_owner, str)
        or _OWNER_RE.fullmatch(expected_bucket_owner) is None
        or type(now_epoch) is not int
        or type(retain_until_epoch) is not int
        or now_epoch < 1
        or retain_until_epoch > 253_402_300_799
        or not _MIN_RETENTION_SECONDS
        <= retain_until_epoch - now_epoch
        <= _MAX_RETENTION_SECONDS
    ):
        raise ReleaseAnchorRetentionError("retention policy is invalid")


def retain_verified_release_head_anchor(
    client: Any,
    anchor: ReleaseHeadAnchor,
    log_payload: bytes,
    *,
    public_keys: Mapping[str, bytes],
    expected_source_commit_sha: str,
    bucket: str,
    key_prefix: str,
    kms_key_arn: str,
    expected_bucket_owner: str,
    now_epoch: int,
    retain_until_epoch: int,
) -> RetainedReleaseAnchorReceipt:
    """Verify and retain one signed release-head anchor as an exact locked version."""

    _validate_policy(
        bucket=bucket,
        key_prefix=key_prefix,
        kms_key_arn=kms_key_arn,
        expected_bucket_owner=expected_bucket_owner,
        now_epoch=now_epoch,
        retain_until_epoch=retain_until_epoch,
    )
    try:
        verify_release_head_anchor(
            anchor,
            log_payload,
            public_keys=public_keys,
            expected_source_commit_sha=expected_source_commit_sha,
        )
        payload = dump_release_head_anchor(anchor)
    except Exception as exc:
        raise ReleaseAnchorRetentionError("release anchor verification failed") from exc

    checksum_bytes = hashlib.sha256(payload).digest()
    checksum_base64 = base64.b64encode(checksum_bytes).decode("ascii")
    checksum_hex = checksum_bytes.hex()
    object_key = f"{key_prefix}/{checksum_hex}.json"
    retain_until = datetime.fromtimestamp(retain_until_epoch, tz=timezone.utc)
    try:
        response = client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=payload,
            ContentType="application/json",
            ChecksumAlgorithm="SHA256",
            ChecksumSHA256=checksum_base64,
            IfNoneMatch="*",
            ObjectLockMode=_RETENTION_MODE,
            ObjectLockRetainUntilDate=retain_until,
            ServerSideEncryption="aws:kms",
            SSEKMSKeyId=kms_key_arn,
            ExpectedBucketOwner=expected_bucket_owner,
            Metadata={
                "anchor-sha256": checksum_hex,
                "head-sha256": anchor.head_sha256,
                "schema-version": RETAINED_RELEASE_ANCHOR_SCHEMA_VERSION,
            },
        )
    except Exception as exc:
        raise ReleaseAnchorRetentionError("S3 anchor write failed") from exc
    if not isinstance(response, Mapping):
        raise ReleaseAnchorRetentionError("S3 anchor write response is malformed")
    version_id = response.get("VersionId")
    if (
        response.get("ChecksumSHA256") != checksum_base64
        or response.get("ServerSideEncryption") != "aws:kms"
        or response.get("SSEKMSKeyId") != kms_key_arn
        or not isinstance(version_id, str)
        or _VERSION_ID_RE.fullmatch(version_id) is None
    ):
        raise ReleaseAnchorRetentionError("S3 anchor write response is malformed")

    try:
        retention_response = client.get_object_retention(
            Bucket=bucket,
            Key=object_key,
            VersionId=version_id,
            ExpectedBucketOwner=expected_bucket_owner,
        )
    except Exception as exc:
        raise ReleaseAnchorRetentionError("S3 anchor retention lookup failed") from exc
    if not isinstance(retention_response, Mapping):
        raise ReleaseAnchorRetentionError("S3 anchor retention response is malformed")
    retention = retention_response.get("Retention")
    if not isinstance(retention, Mapping):
        raise ReleaseAnchorRetentionError("S3 anchor retention response is malformed")
    observed_until = retention.get("RetainUntilDate")
    if (
        retention.get("Mode") != _RETENTION_MODE
        or not isinstance(observed_until, datetime)
        or observed_until.tzinfo is None
        or int(observed_until.timestamp()) < retain_until_epoch
    ):
        raise ReleaseAnchorRetentionError("S3 anchor retention policy mismatch")

    return RetainedReleaseAnchorReceipt(
        schema_version=RETAINED_RELEASE_ANCHOR_SCHEMA_VERSION,
        bucket=bucket,
        object_key=object_key,
        version_id=version_id,
        anchor_sha256=anchor.anchor_sha256,
        head_sha256=anchor.head_sha256,
        checksum_sha256=checksum_hex,
        retention_mode=_RETENTION_MODE,
        retain_until_epoch=int(observed_until.timestamp()),
        kms_key_id_sha256=hashlib.sha256(kms_key_arn.encode("utf-8")).hexdigest(),
        expected_bucket_owner_sha256=hashlib.sha256(
            expected_bucket_owner.encode("ascii")
        ).hexdigest(),
    )
