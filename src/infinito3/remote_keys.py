"""Envelopes: how one person's request hands a key to another person who is not signed in (INFINITO Teams).

A memory opens only while one of its owner's requests is served (remote_accounts.py): its key is kept wrapped under
their password, recovery code and apps' tokens, and nothing else. That is enough for one person, but a team needs more:
when Ana adds Bea to a project at night, Bea is not there to receive the project's key, and the server must not keep it
unwrapped until she comes (docs/INFINITO_TEAMS_ADR_P0_2026-11-20.md, D2).

So each account has a key pair (X25519), made the first time Teams needs it: the private half sealed with the account's
memory key in accounts.json, next to the other wrapped keys, and the public half in the clear. Anyone's request can seal
a secret to that public half, an envelope, and only that person's own request opens it.

An envelope says what it is for (context: the organization, the space, the key's version and the person) as associated
data, so it opens nowhere else: copied to another space or handed to another person, it fails.

What this does not stop is what a memory does not stop either: someone who changes the server's running code could swap
a public key or capture keys while they are in use.
"""
import os
from typing import Optional, Tuple

MAGIC = b"ENV1"  # an envelope: magic, the sender's one-time public key (32), nonce (12), AES-GCM ciphertext and tag
_INFO = b"infinito envelope v1"
_SMALLEST = len(MAGIC) + 32 + 12 + 16


def _public_of(private) -> bytes:
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    return private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def _key(shared: bytes, sender: bytes, recipient: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=32, salt=sender + recipient, info=_INFO).derive(shared)


def new_pair() -> Tuple[bytes, bytes]:
    """A new key pair, as raw bytes: (private, public)."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

    private = X25519PrivateKey.generate()
    return private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()), _public_of(private)


def context(org_id: str, space_id: str, version: int, account: str) -> str:
    """What an envelope of a space's key is for; it opens only with these same words."""
    return f"teams-space|{org_id}|{space_id}|{int(version)}|{account}"


def seal_to(public: bytes, secret: bytes, about: str) -> bytes:
    """secret sealed so that only the holder of public's private half opens it, and only for about."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    sender = X25519PrivateKey.generate()  # used once: nobody, the sender included, can open the envelope afterwards
    shared = sender.exchange(X25519PublicKey.from_public_bytes(public))
    sender_public, nonce = _public_of(sender), os.urandom(12)
    return MAGIC + sender_public + nonce + AESGCM(_key(shared, sender_public, public)).encrypt(nonce, secret,
                                                                                               about.encode())


def open_envelope(private: bytes, envelope: bytes, about: str) -> Optional[bytes]:
    """What the envelope holds, or None if it is not for this private key, not for about, or was changed."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not isinstance(envelope, bytes) or len(envelope) < _SMALLEST or not envelope.startswith(MAGIC):
        return None
    sender_public, nonce, sealed = envelope[4:36], envelope[36:48], envelope[48:]
    me = X25519PrivateKey.from_private_bytes(private)
    try:
        shared = me.exchange(X25519PublicKey.from_public_bytes(sender_public))
        return AESGCM(_key(shared, sender_public, _public_of(me))).decrypt(nonce, sealed, about.encode())
    except (InvalidTag, ValueError):  # ValueError: a sender key that makes no secret at all
        return None
