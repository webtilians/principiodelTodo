"""Experimental, opt-in, server-issued record provenance for INFINITO OAuth MCP.

Ed25519 proves the payload was signed by this server's key. It does not prove
the accuracy of arbitrary memory text or a source quoted inside that text.
Clients MUST pin the public key through a separately trusted channel.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Callable, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

NAME = ".infinito-experimental-provenance-ed25519"
TOOL = "recall_attested"
TTL_SECONDS = 120
MAX_RECORDS = 10
NONCE = re.compile(r"[0-9a-f]{32}\Z")
MAX_PAYLOAD_BYTES = 50000
_FIELDS = frozenset({
    "version", "issuer", "subject_account", "client_id", "source_tool",
    "nonce", "query_sha256", "issued_at", "expires_at", "records",
})
_RECORD_FIELDS = frozenset({"record_id", "record_owner", "content", "date"})


def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(encoded: str) -> bytes:
    if not isinstance(encoded, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", encoded):
        raise ValueError("malformed base64url")
    raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    if _b64(raw) != encoded:
        raise ValueError("noncanonical base64url")
    return raw


def _new_private_file(path: Path) -> None:
    """Atomically publish a complete 0600 key: safe when two server workers start."""
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(serialization.Encoding.Raw,
                            serialization.PrivateFormat.Raw,
                            serialization.NoEncryption())
    tmp = path.parent / (path.name + ".tmp-" + secrets.token_hex(12))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(tmp, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path, follow_symlinks=False)
        except FileExistsError:
            pass  # Another trusted worker published first. Inspect its key below.
    finally:
        tmp.unlink(missing_ok=True)


def _read_private_file(path: Path) -> Ed25519PrivateKey:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        st = os.fstat(stream.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077:
            raise PermissionError("Signing key must be a private regular file (0600)")
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            raise PermissionError("Signing key is not owned by the server user")
        raw = stream.read(33)
    if len(raw) != 32:
        raise ValueError("Invalid or corrupted Ed25519 signing key; refusing rotation")
    return Ed25519PrivateKey.from_private_bytes(raw)


class ServerAttestor:
    def __init__(self, root: Path, issuer: str):
        if not isinstance(issuer, str) or not issuer.startswith(("https://", "http://127.0.0.1:", "http://localhost:")):
            raise ValueError("Attested evidence requires a trusted HTTPS issuer or loopback in lab")
        root = Path(root)
        # Refuse key material in a shared or symlinked root: 0600 on the
        # private key itself does not prevent another user from swapping a
        # directory entry if that user can write the enclosing directory.
        try:
            root_stat = root.lstat()
        except FileNotFoundError:
            raise ValueError("Remote server storage root must already exist") from None
        if not stat.S_ISDIR(root_stat.st_mode) or root.is_symlink():
            raise PermissionError("Attestation key root must be a real directory")
        if root_stat.st_mode & 0o022:
            raise PermissionError("Attestation key root cannot be group/world writable")
        if hasattr(os, "getuid") and root_stat.st_uid != os.getuid():
            raise PermissionError("Attestation key root must be owned by the service user")
        self.issuer = issuer.rstrip("/")
        path = root / NAME
        try:
            self.__private = _read_private_file(path)
        except FileNotFoundError:
            _new_private_file(path)
            self.__private = _read_private_file(path)
        self.public = self.__private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.key_id = hashlib.sha256(self.public).hexdigest()

    def public_info(self) -> dict:
        return {"algorithm": "Ed25519", "issuer": self.issuer,
                "key_id": self.key_id, "public_key": _b64(self.public)}

    def attest(self, *, account: str, client_id: str, nonce: str,
               query: str, records: list[dict], now: Optional[int] = None) -> dict:
        if NONCE.fullmatch(nonce or "") is None:
            raise ValueError("Expected a 16-byte hex client challenge")
        if not isinstance(query, str) or not 0 < len(query) <= 4096:
            raise ValueError("Query must contain 1-4096 characters")
        if not isinstance(records, list) or len(records) > MAX_RECORDS:
            raise ValueError("Too many records")
        if not account or not isinstance(account, str) or not isinstance(client_id, str):
            raise ValueError("Missing server-authenticated caller identity")
        if any(not isinstance(row, dict) or set(row) != _RECORD_FIELDS
               or row["record_owner"] != account
               or any(not isinstance(v, str) for v in row.values())
               for row in records):
            raise ValueError("Refusing to sign records not owned by the authenticated account")
        issued_at = int(time.time()) if now is None else int(now)
        payload = {
            "version": 1, "issuer": self.issuer, "subject_account": account,
            "client_id": client_id, "source_tool": TOOL,
            "nonce": nonce, "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "issued_at": issued_at, "expires_at": issued_at + TTL_SECONDS,
            "records": records,
        }
        encoded = _canonical(payload)
        if len(encoded) > MAX_PAYLOAD_BYTES:
            raise ValueError("Attested response too long")
        return {"payload": payload, "key_id": self.key_id,
                "signature": _b64(self.__private.sign(encoded))}


def verify_attestation(
    envelope: object, *, pinned_public_key: bytes, expected_issuer: str,
    expected_account: str, expected_client_id: str, expected_nonce: str,
    expected_query: str, now: Optional[int] = None,
) -> dict:
    """Client-side verification. Caller supplies independently pinned key & OAuth context.

    Raises ValueError on ANY unverified proof; never claims it is a clean result.
    """
    def refuse() -> None:
        raise ValueError("UNVERIFIED: Ed25519 proof or OAuth binding failed")
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "key_id", "signature"}:
        refuse()
    payload = envelope["payload"]
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        refuse()
    try:
        if not isinstance(pinned_public_key, bytes) or len(pinned_public_key) != 32:
            refuse()
        if envelope["key_id"] != hashlib.sha256(pinned_public_key).hexdigest():
            refuse()
        sig = _unb64(envelope["signature"])
        if len(sig) != 64:
            refuse()
        data = _canonical(payload)
        if len(data) > MAX_PAYLOAD_BYTES:
            refuse()
        Ed25519PublicKey.from_public_bytes(pinned_public_key).verify(sig, data)
    except (ValueError, TypeError, InvalidSignature, OverflowError):
        refuse()
    timestamp = int(time.time()) if now is None else int(now)
    if (
        payload["version"] != 1
        or payload["issuer"] != expected_issuer.rstrip("/")
        or payload["subject_account"] != expected_account
        or payload["client_id"] != expected_client_id
        or payload["source_tool"] != TOOL
        or payload["nonce"] != expected_nonce
        or NONCE.fullmatch(expected_nonce or "") is None
        or payload["query_sha256"] != hashlib.sha256(expected_query.encode("utf-8")).hexdigest()
        or type(payload["issued_at"]) is not int
        or type(payload["expires_at"]) is not int
        or payload["issued_at"] > timestamp + 5
        or payload["expires_at"] <= timestamp
        or payload["expires_at"] - payload["issued_at"] != TTL_SECONDS
    ):
        refuse()
    records = payload["records"]
    if not isinstance(records, list) or len(records) > MAX_RECORDS:
        refuse()
    if any(not isinstance(record, dict) or set(record) != _RECORD_FIELDS
           or not all(isinstance(v, str) for v in record.values())
           or record["record_owner"] != expected_account
           for record in records):
        refuse()
    return payload
