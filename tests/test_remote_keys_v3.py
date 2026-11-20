"""Envelopes (INFINITO Teams): a key handed to a person who is not there, which only they open, and only for what it
was meant for."""
import base64
import json

import pytest

from src.infinito3 import remote_keys
from src.infinito3.remote_accounts import Accounts

pytest.importorskip("cryptography")

SECRET = bytes(range(32))
ABOUT = remote_keys.context("org1", "space1", 1, "bea")


def test_only_the_private_half_opens_what_was_sealed_to_its_public_half():
    private, public = remote_keys.new_pair()
    other_private, _ = remote_keys.new_pair()
    envelope = remote_keys.seal_to(public, SECRET, ABOUT)
    assert remote_keys.open_envelope(private, envelope, ABOUT) == SECRET
    assert remote_keys.open_envelope(other_private, envelope, ABOUT) is None
    assert SECRET not in envelope


@pytest.mark.parametrize("elsewhere", [remote_keys.context("org2", "space1", 1, "bea"),
                                       remote_keys.context("org1", "space2", 1, "bea"),
                                       remote_keys.context("org1", "space1", 2, "bea"),
                                       remote_keys.context("org1", "space1", 1, "eva")])
def test_an_envelope_opens_only_for_the_organization_space_version_and_person_it_names(elsewhere):
    private, public = remote_keys.new_pair()
    assert remote_keys.open_envelope(private, remote_keys.seal_to(public, SECRET, ABOUT), elsewhere) is None


def test_a_changed_or_made_up_envelope_opens_nothing():
    private, public = remote_keys.new_pair()
    envelope = bytearray(remote_keys.seal_to(public, SECRET, ABOUT))
    envelope[-1] ^= 1
    assert remote_keys.open_envelope(private, bytes(envelope), ABOUT) is None
    for made_up in (b"", b"ENV1", b"ENV1" + bytes(80), b"XXXX" + bytes(80), bytes(32 + 4 + 12 + 16)):
        assert remote_keys.open_envelope(private, made_up, ABOUT) is None


def test_the_same_secret_sealed_twice_looks_different_each_time():
    _, public = remote_keys.new_pair()
    assert remote_keys.seal_to(public, SECRET, ABOUT) != remote_keys.seal_to(public, SECRET, ABOUT)


@pytest.fixture
def accounts(tmp_path):
    return Accounts(tmp_path)


def test_a_key_pair_is_made_once_and_its_private_half_is_kept_only_sealed(accounts):
    key, _ = accounts.create("bea", "una contraseña larga")
    assert accounts.public_key("bea") is None  # nothing is made until Teams asks for it
    private, public = accounts.key_pair("bea", key)
    assert accounts.key_pair("bea", key) == (private, public) and accounts.public_key("bea") == public
    on_disk = accounts.file.read_bytes()
    assert private not in on_disk and base64.b64encode(private) not in on_disk
    assert json.loads(on_disk)["bea"]["pair"]["public"] == base64.b64encode(public).decode()


def test_only_the_accounts_own_key_opens_its_private_half(accounts):
    key, _ = accounts.create("bea", "una contraseña larga")
    accounts.key_pair("bea", key)
    assert accounts.key_pair("bea", bytes(32)) is None
    assert accounts.key_pair("nadie", key) is None and accounts.public_key("nadie") is None


def test_an_envelope_reaches_someone_who_is_not_signed_in_and_survives_a_new_password(accounts):
    bea_key, code = accounts.create("bea", "una contraseña larga")
    accounts.key_pair("bea", bea_key)  # Bea used Teams once; now she is away
    envelope = remote_keys.seal_to(accounts.public_key("bea"), SECRET, ABOUT)  # Ana's request, without Bea's key
    key = accounts.recover("bea", code, "otra contraseña distinta")  # the memory's key stays the same
    private, _ = accounts.key_pair("bea", key)
    assert remote_keys.open_envelope(private, envelope, ABOUT) == SECRET


def test_removing_an_account_takes_its_key_pair_with_it(accounts):
    key, _ = accounts.create("bea", "una contraseña larga")
    accounts.key_pair("bea", key)
    accounts.remove("bea")
    assert accounts.public_key("bea") is None
