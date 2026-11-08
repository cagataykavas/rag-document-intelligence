from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from src.generation_evidence_replay import (
    GenerationEvidenceReplayError,
    replay_generation_evidence,
)
from src.prompting import EvidenceSnippet

ENCRYPTED_BUNDLE_SCHEMA_VERSION = "generation-replay-encrypted-bundle/v1"
PRIVATE_BUNDLE_SCHEMA_VERSION = "generation-replay-private-bundle/v1"
DECRYPT_KEY_ENV = "RAG_REPLAY_DECRYPT_KEYS"
AUTH_KEY_ENV = "RAG_ADMISSION_VERIFY_KEYS"
MAX_ENCRYPTED_BUNDLE_BYTES = 64 * 1024 * 1024
MAX_KEYRING_BYTES = 16 * 1024
MAX_KEYRING_KEYS = 16


class ReplayCliError(ValueError):
    """Stable operator-facing failure that does not disclose private content."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReplayCliError("BUNDLE_INVALID") from exc


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayCliError("JSON_DUPLICATE_KEY")
        result[key] = value
    return result


def _load_json(payload: bytes, *, error_code: str) -> object:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ReplayCliError(error_code)
            ),
        )
    except ReplayCliError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayCliError(error_code) from exc


def _decode_key(value: object, *, error_code: str) -> bytes:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise ReplayCliError(error_code)
    try:
        key = base64.b64decode(value, altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ReplayCliError(error_code) from exc
    if len(key) != 32:
        raise ReplayCliError(error_code)
    return key


def load_keyring_from_environment(name: str) -> dict[str, bytes]:
    raw = os.environ.get(name)
    if raw is None or not raw or len(raw.encode("utf-8")) > MAX_KEYRING_BYTES:
        raise ReplayCliError("KEYRING_INVALID")
    document = _load_json(raw.encode("utf-8"), error_code="KEYRING_INVALID")
    if (
        not isinstance(document, dict)
        or not 1 <= len(document) <= MAX_KEYRING_KEYS
        or any(not isinstance(key_id, str) or not key_id for key_id in document)
    ):
        raise ReplayCliError("KEYRING_INVALID")
    return {
        key_id: _decode_key(value, error_code="KEYRING_INVALID")
        for key_id, value in document.items()
    }


def _aad(key_id: str) -> bytes:
    return _canonical_json(
        {"schema_version": ENCRYPTED_BUNDLE_SCHEMA_VERSION, "key_id": key_id}
    )


def seal_private_replay_bundle(
    private_bundle: Mapping[str, object],
    *,
    key_id: str,
    key: bytes,
    nonce: bytes | None = None,
) -> bytes:
    """Encrypt a canonical private replay bundle for an external producer."""

    if not isinstance(key_id, str) or not key_id or len(key_id.encode("utf-8")) > 128:
        raise ReplayCliError("KEY_ID_INVALID")
    if not isinstance(key, bytes) or len(key) != 32:
        raise ReplayCliError("KEY_INVALID")
    actual_nonce = os.urandom(12) if nonce is None else nonce
    if not isinstance(actual_nonce, bytes) or len(actual_nonce) != 12:
        raise ReplayCliError("NONCE_INVALID")
    plaintext = _canonical_json(dict(private_bundle))
    if len(plaintext) + 16 > MAX_ENCRYPTED_BUNDLE_BYTES:
        raise ReplayCliError("BUNDLE_TOO_LARGE")
    ciphertext = AESGCM(key).encrypt(actual_nonce, plaintext, _aad(key_id))
    return _canonical_json(
        {
            "schema_version": ENCRYPTED_BUNDLE_SCHEMA_VERSION,
            "key_id": key_id,
            "nonce_b64": base64.urlsafe_b64encode(actual_nonce).decode("ascii"),
            "ciphertext_b64": base64.urlsafe_b64encode(ciphertext).decode("ascii"),
        }
    )


def _open_private_bundle(path: Path) -> tuple[int, os.stat_result, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReplayCliError("BUNDLE_OPEN_FAILED") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
            or not 1 <= metadata.st_size <= MAX_ENCRYPTED_BUNDLE_BYTES
        ):
            raise ReplayCliError("BUNDLE_FILE_UNSAFE")
        chunks: list[bytes] = []
        remaining = MAX_ENCRYPTED_BUNDLE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after_read = os.fstat(descriptor)
        if (
            len(payload) != metadata.st_size
            or len(payload) > MAX_ENCRYPTED_BUNDLE_BYTES
            or after_read.st_dev != metadata.st_dev
            or after_read.st_ino != metadata.st_ino
            or after_read.st_size != metadata.st_size
            or after_read.st_mtime_ns != metadata.st_mtime_ns
            or after_read.st_ctime_ns != metadata.st_ctime_ns
        ):
            raise ReplayCliError("BUNDLE_FILE_CHANGED")
        return descriptor, metadata, payload
    except Exception:
        os.close(descriptor)
        raise


def _decrypt_bundle(
    envelope_payload: bytes, keyring: Mapping[str, bytes]
) -> dict[str, object]:
    envelope = _load_json(envelope_payload, error_code="BUNDLE_INVALID")
    expected_fields = {"schema_version", "key_id", "nonce_b64", "ciphertext_b64"}
    if not isinstance(envelope, dict) or set(envelope) != expected_fields:
        raise ReplayCliError("BUNDLE_INVALID")
    key_id = envelope["key_id"]
    if (
        envelope["schema_version"] != ENCRYPTED_BUNDLE_SCHEMA_VERSION
        or not isinstance(key_id, str)
        or not key_id
        or len(key_id.encode("utf-8")) > 128
        or key_id not in keyring
    ):
        raise ReplayCliError("BUNDLE_INVALID")
    try:
        nonce = base64.b64decode(envelope["nonce_b64"], altchars=b"-_", validate=True)
        ciphertext = base64.b64decode(
            envelope["ciphertext_b64"], altchars=b"-_", validate=True
        )
    except (TypeError, ValueError, binascii.Error) as exc:
        raise ReplayCliError("BUNDLE_INVALID") from exc
    if len(nonce) != 12 or not 16 <= len(ciphertext) <= MAX_ENCRYPTED_BUNDLE_BYTES:
        raise ReplayCliError("BUNDLE_INVALID")
    try:
        plaintext = AESGCM(keyring[key_id]).decrypt(nonce, ciphertext, _aad(key_id))
    except InvalidTag as exc:
        raise ReplayCliError("BUNDLE_AUTHENTICATION_FAILED") from exc
    document = _load_json(plaintext, error_code="PRIVATE_BUNDLE_INVALID")
    if not isinstance(document, dict):
        raise ReplayCliError("PRIVATE_BUNDLE_INVALID")
    return document


def _parse_private_bundle(
    document: Mapping[str, object],
) -> tuple[
    object,
    tuple[EvidenceSnippet, ...],
    Mapping[str, object],
    Mapping[str, object] | None,
]:
    expected_fields = {
        "schema_version",
        "payload",
        "evidence",
        "expected_admission_audit",
        "authentication",
    }
    if (
        set(document) != expected_fields
        or document["schema_version"] != PRIVATE_BUNDLE_SCHEMA_VERSION
    ):
        raise ReplayCliError("PRIVATE_BUNDLE_INVALID")
    evidence = document["evidence"]
    audit = document["expected_admission_audit"]
    authentication = document["authentication"]
    if (
        not isinstance(evidence, list)
        or not isinstance(audit, dict)
        or (authentication is not None and not isinstance(authentication, dict))
    ):
        raise ReplayCliError("PRIVATE_BUNDLE_INVALID")
    snippets: list[EvidenceSnippet] = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {
            "chunk_id",
            "source",
            "text",
            "source_locator",
        }:
            raise ReplayCliError("PRIVATE_BUNDLE_INVALID")
        if not all(
            isinstance(item[field], str) for field in ("chunk_id", "source", "text")
        ):
            raise ReplayCliError("PRIVATE_BUNDLE_INVALID")
        locator = item["source_locator"]
        if locator is not None and not isinstance(locator, dict):
            raise ReplayCliError("PRIVATE_BUNDLE_INVALID")
        snippets.append(
            EvidenceSnippet(item["chunk_id"], item["source"], item["text"], locator)
        )
    return document["payload"], tuple(snippets), audit, authentication


def _write_receipt(path: Path, receipt: Mapping[str, object]) -> None:
    payload = _canonical_json(dict(receipt)) + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ReplayCliError("RECEIPT_CREATE_FAILED") from exc
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    except OSError as exc:
        try:
            path.unlink()
        except OSError:
            pass
        raise ReplayCliError("RECEIPT_WRITE_FAILED") from exc
    finally:
        os.close(descriptor)
    try:
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except OSError as exc:
        raise ReplayCliError("RECEIPT_SYNC_FAILED") from exc


def _delete_exact_bundle(path: Path, descriptor: int, metadata: os.stat_result) -> None:
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            opened.st_dev != metadata.st_dev
            or opened.st_ino != metadata.st_ino
            or opened.st_size != metadata.st_size
            or opened.st_mtime_ns != metadata.st_mtime_ns
            or opened.st_ctime_ns != metadata.st_ctime_ns
            or current.st_dev != metadata.st_dev
            or current.st_ino != metadata.st_ino
            or current.st_nlink != 1
            or not stat.S_ISREG(current.st_mode)
        ):
            raise ReplayCliError("BUNDLE_DELETE_RACE")
        path.unlink()
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except ReplayCliError:
        raise
    except OSError as exc:
        raise ReplayCliError("BUNDLE_DELETE_FAILED") from exc


def run(
    bundle_path: Path,
    receipt_path: Path,
    *,
    delete_after_success: bool = False,
) -> dict[str, object]:
    decrypt_keyring = load_keyring_from_environment(DECRYPT_KEY_ENV)
    descriptor, metadata, envelope_payload = _open_private_bundle(bundle_path)
    try:
        document = _decrypt_bundle(envelope_payload, decrypt_keyring)
        payload, evidence, audit, authentication = _parse_private_bundle(document)
        auth_keyring = None
        if authentication is not None and authentication.get("applied") is not False:
            auth_keyring = load_keyring_from_environment(AUTH_KEY_ENV)
        receipt = replay_generation_evidence(
            payload,
            evidence,
            audit,
            authentication=authentication,
            keyring=auth_keyring,
        )
        _write_receipt(receipt_path, receipt)
        if delete_after_success:
            _delete_exact_bundle(bundle_path, descriptor, metadata)
        return receipt
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify an encrypted private generation-replay bundle."
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--delete-private-bundle-after-success",
        action="store_true",
        help="delete the exact private bundle only after the receipt is durable",
    )
    args = parser.parse_args(argv)
    try:
        run(
            args.bundle,
            args.receipt,
            delete_after_success=args.delete_private_bundle_after_success,
        )
    except (ReplayCliError, GenerationEvidenceReplayError) as exc:
        code = exc.code
        print(json.dumps({"verified": False, "error_code": code}), file=sys.stderr)
        return 2
    print(json.dumps({"verified": True, "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
