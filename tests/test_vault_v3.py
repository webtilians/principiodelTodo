"""The vault: the agent names secrets, the user's screen shows them, and no value ever reaches the agent."""
import json

import pytest

pytest.importorskip("pykeepass")

from src.infinito3.vault import Credentials, Vault  # noqa: E402

MASTER = "correct horse battery staple"
SECRET = "R0uter-Casa!2026"


class FakeScreen:
    """Stands in for the user's windows: what they would type, and what they were shown."""

    def __init__(self, credentials=None, typed=None):
        self.credentials = credentials
        self.typed = typed
        self.shown = []
        self.unlock_requests = []

    def unlock(self, vault, keyfile, creating):
        self.unlock_requests.append((creating, keyfile))
        return self.credentials

    def new_secret(self, title):
        return self.typed

    def show(self, title, username, password):
        self.shown.append((title, username, password))


@pytest.fixture
def home(tmp_path):
    return tmp_path / ".infinito"


def open_vault(home, screen, **kwargs):
    vault = Vault(ui=screen, home=home, **kwargs)
    return vault, vault.unlock()


def test_the_first_unlock_creates_the_vault_and_a_later_one_opens_it(home):
    screen = FakeScreen(Credentials(MASTER))
    vault, created = open_vault(home, screen)
    assert created.startswith("Vault created") and (home / "boveda.kdbx").exists()
    assert screen.unlock_requests == [(True, None)]
    vault.lock()
    assert vault.unlock().startswith("The vault is open")
    assert screen.unlock_requests[-1] == (False, None)


def test_a_secret_is_typed_and_shown_on_the_screen_and_never_returned(home):
    screen = FakeScreen(Credentials(MASTER), typed=("admin", SECRET))
    vault, _ = open_vault(home, screen)
    replies = [vault.add("Router de casa"), vault.names(), vault.reveal("router"), vault.status()]
    assert screen.shown == [("Router de casa", "admin", SECRET)]
    assert "Router de casa" in replies[1]
    for reply in replies:
        assert SECRET not in reply and MASTER not in reply
    log = (home / "vault-log.jsonl").read_text(encoding="utf-8")
    assert SECRET not in log and MASTER not in log
    assert [json.loads(line)["action"] for line in log.splitlines()] == ["created", "added", "revealed"]


def test_the_file_only_opens_with_the_master_password(home):
    vault, _ = open_vault(home, FakeScreen(Credentials(MASTER), typed=("", SECRET)))
    vault.add("Banco")
    assert SECRET.encode() not in (home / "boveda.kdbx").read_bytes()
    wrong, reply = open_vault(home, FakeScreen(Credentials("not the password")))
    assert reply.startswith("Wrong master password")
    assert wrong.names() == "The vault is locked: call vault_unlock first."


def test_a_key_file_is_needed_too_when_the_vault_has_one(home, tmp_path):
    keyfile = tmp_path / "usb" / "infinito-llave.key"
    keyfile.parent.mkdir()
    keyfile.write_bytes(b"\x07" * 128)
    vault, _ = open_vault(home, FakeScreen(Credentials(MASTER, str(keyfile))))
    vault.lock()
    _, without_key = open_vault(home, FakeScreen(Credentials(MASTER)))
    assert without_key.startswith("Wrong master password or key file")
    screen = FakeScreen(Credentials(MASTER, str(keyfile)))
    _, with_key = open_vault(home, screen)
    assert with_key.startswith("The vault is open")
    assert screen.unlock_requests == [(False, str(keyfile))]  # the key file's place is remembered


def test_it_locks_itself_after_a_while_without_use(home):
    vault, _ = open_vault(home, FakeScreen(Credentials(MASTER)), idle_minutes=1)
    assert vault.names() == "The vault is empty."
    vault._used -= 61  # a minute and a second without use
    assert vault.names() == "The vault is locked: call vault_unlock first."


def test_closing_the_window_changes_nothing(home):
    vault, reply = open_vault(home, FakeScreen(None))
    assert "stays locked" in reply and not (home / "boveda.kdbx").exists()
    vault, _ = open_vault(home, FakeScreen(Credentials(MASTER), typed=None))
    assert vault.add("Correo") == "The user closed the window: nothing was saved."


def test_a_name_that_matches_several_entries_asks_which_one(home):
    screen = FakeScreen(Credentials(MASTER), typed=("", SECRET))
    vault, _ = open_vault(home, screen)
    vault.add("Banco personal")
    vault.add("Banco empresa")
    assert vault.reveal("banco").startswith("Several entries match")
    assert screen.shown == []


def test_remember_points_a_refused_secret_at_the_vault(tmp_path):
    from src.infinito3.mcp_server import AgentMemory

    memory = AgentMemory(str(tmp_path / "memory.db"), vault=True)
    assert "vault_add" in memory.remember("La contraseña del router es R0uter-Casa!2026")
    memory.close()
