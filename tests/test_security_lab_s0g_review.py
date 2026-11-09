"""S0-G: adversarial review evidence for server-issued OAuth MCP provenance.

This suite DOES NOT certify the product or replace an independent reviewer.
Known unsatisfied security requirements are intentionally marked strict XFAIL.
No real accounts, remote requests, or production secrets.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import secrets

import pytest

from src.infinito3.remote_provenance import (
    NAME, ServerAttestor, verify_attestation,
)

ISSUER = "https://signed-mcp.example.invalid"
ACCOUNT = "synthetic-tiger"
CLIENT = "synthetic-agent"
NONCE = "d" * 32
QUERY = "synthetic memo about K4"
NOW = 1000


def signer(tmp_path):
    return ServerAttestor(tmp_path, ISSUER)


def envelope(signer_obj, *, account=ACCOUNT, owner=ACCOUNT, client=CLIENT,
             nonce=NONCE, content="A sample belongs to this account."):
    return signer_obj.attest(
        account=account, client_id=client, nonce=nonce, query=QUERY,
        records=[{
            "record_id": "synthetic-review-r1", "record_owner": owner,
            "content": content, "date": "2026-11-09",
        }],
        now=NOW,
    )


def checks(signer_obj):
    return {
        "pinned_public_key": signer_obj.public,
        "expected_issuer": ISSUER,
        "expected_account": ACCOUNT,
        "expected_client_id": CLIENT,
        "expected_nonce": NONCE,
        "expected_query": QUERY,
        "now": NOW + 1,
    }


def test_a00_authentic_signature_is_tenant_client_and_nonce_bound(tmp_path):
    s = signer(tmp_path)
    doc = envelope(s)
    assert verify_attestation(doc, **checks(s))["subject_account"] == ACCOUNT
    for altered in (
        {"expected_account": "another-account"},
        {"expected_client_id": "another-client"},
        {"expected_nonce": "a" * 32},
        {"expected_query": "different-query"},
        {"expected_issuer": "https://imposter.example.invalid"},
    ):
        with pytest.raises(ValueError, match="UNVERIFIED"):
            verify_attestation(doc, **{**checks(s), **altered})


def test_a01_same_issuer_impostor_key_is_rejected_only_with_true_pin(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real = signer(real_dir)
    rogue_dir = tmp_path / "rogue"
    rogue_dir.mkdir()
    rogue = signer(rogue_dir)
    evil = envelope(rogue)
    with pytest.raises(ValueError, match="UNVERIFIED"):
        verify_attestation(evil, **checks(real))
    # A client naively trusting the key advertised by an impostor would accept it:
    assert verify_attestation(evil, **checks(rogue))["issuer"] == ISSUER


def test_a02_bad_private_key_does_not_rotate_silently(tmp_path):
    key = tmp_path / NAME
    key.write_bytes(b"corrupted")
    key.chmod(0o600)
    with pytest.raises(ValueError):
        signer(tmp_path)
    assert key.read_bytes() == b"corrupted"
    key.write_bytes(secrets.token_bytes(32))
    key.chmod(0o644)
    with pytest.raises(PermissionError):
        signer(tmp_path)


def test_a03_physical_key_replacement_breaks_existing_pin(tmp_path):
    old = signer(tmp_path)
    key = tmp_path / NAME
    old_bytes = key.read_bytes()
    key.unlink()
    new = signer(tmp_path)
    assert len(old_bytes) == 32 and old.public != new.public
    new_proof = envelope(new)
    with pytest.raises(ValueError, match="UNVERIFIED"):
        verify_attestation(new_proof, **checks(old))
    assert verify_attestation(new_proof, **checks(new))["records"][0]["record_owner"] == ACCOUNT


def test_a04_signed_store_owner_is_not_external_original_author(tmp_path):
    s = signer(tmp_path)
    doc = envelope(s, content="This unsigned record says written_by=other-user and owner=admin.")
    recovered = verify_attestation(doc, **checks(s))
    assert recovered["records"][0]["record_owner"] == ACCOUNT
    assert "written_by=other-user" in recovered["records"][0]["content"]
    assert "original_author" not in recovered["records"][0]


def test_a05_mismatched_owner_cannot_be_signed_under_current_account(tmp_path):
    s = signer(tmp_path)
    with pytest.raises(ValueError, match="Refusing to sign records"):
        envelope(s, owner="a-different-tenant")


def test_a06_reused_nonce_does_not_have_one_time_replay_defense(tmp_path):
    s = signer(tmp_path)
    signed = envelope(s)
    once = verify_attestation(signed, **checks(s))
    twice = verify_attestation(signed, **checks(s))
    assert once == twice, "Current stateless verifier accepts identical proof within TTL"


@pytest.mark.xfail(strict=True, reason="P1 issue #149: no signing-key revocation registry or enforced trust-policy invalidation")
def test_a07_old_compromised_key_should_be_rejected_after_revocation(tmp_path):
    old = signer(tmp_path)
    signed_with_old = envelope(old)
    (tmp_path / NAME).unlink()
    new = signer(tmp_path)
    assert old.public != new.public
    # Desired production property (NOT implemented): old pinned proof is
    # rejected immediately after confirmed compromise, even if TTL remains.
    with pytest.raises(ValueError, match="UNVERIFIED"):
        verify_attestation(signed_with_old, **checks(old))


@pytest.mark.xfail(strict=True, reason="P2 issue #149: signer does not validate group/world-writable key-parent permissions")
def test_a08_world_writable_key_root_should_fail_closed(tmp_path):
    os.chmod(tmp_path, 0o777)
    try:
        with pytest.raises(PermissionError):
            signer(tmp_path)
    finally:
        os.chmod(tmp_path, 0o700)


def test_a09_substituted_public_key_does_not_override_existing_pin(tmp_path):
    real_dir = tmp_path / "legit"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    real, fake = signer(real_dir), signer(fake_dir)
    doc = envelope(real)
    forged = copy.deepcopy(doc)
    forged["key_id"] = fake.key_id
    with pytest.raises(ValueError, match="UNVERIFIED"):
        verify_attestation(forged, **checks(real))
    with pytest.raises(ValueError, match="UNVERIFIED"):
        verify_attestation(doc, **checks(fake))
