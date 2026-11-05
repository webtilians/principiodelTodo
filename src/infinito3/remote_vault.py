"""A vault that works from any app: the agent gets a link, the person types on the server's own page.

The vault on a computer (vault.py) opens windows on that computer's screen, which ChatGPT or a phone cannot
do. So on the server vault_add and vault_reveal answer with a link. The master password and the values are
typed and shown on that page: they never pass through the agent, its model or the chat, and no tool has a
parameter for a value or returns one.

Each person's vault is a standard KeePass file (KDBX) beside their sealed memory, so a copy opens with
KeePassXC or KeePassDX, without INFINITO. It is opened here, under the master password sent from the page,
only while that request is served; the password and the values are not stored or logged. The agent sees
names only: a list of them is sealed with the memory key, so apps can read it without the master password.

What this cannot stop is what sealing cannot stop either (remote/README.md): someone who changes the
server's code can capture a master password as it arrives. So it needs a server run by someone you trust,
and it is by invitation: the server's owner lets each account in (infinito-remote vault-allow <name>).

A link is used up when the page is sent, not when it is opened, since chat apps may fetch a link first.
"""
import functools
import html
import json
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .remote_accounts import _page, _seal, _texts, _unseal
from .safety import SensitiveInformationFilter
from .types import SafetyLevel

LINK_SECONDS = 600
MAX_NAME = 80
MAX_VALUE = 4096
MIN_MASTER = 14
SHOWN_SECONDS = 60
GRANT = "vault"
KDBX_MAGIC = b"\x03\xd9\xa2\x9a"  # how every KeePass file starts
FILE = "vault.kdbx"
NAMES_FILE = "vault-names.sealed"

TOOL_DESCRIPTIONS = {
    "vault_add": "Keep a password, key or PIN in the user's vault. Give only a name for it, such as 'Router de casa', "
                 "never the value. It answers with a link: give it to the user, who types the value on that page, "
                 "so it never passes through you or the chat.",
    "vault_reveal": "Show the user a secret from their vault, by entry name. It answers with a link: give it to the "
                    "user, who sees the value on that page. You never receive it.",
    "vault_list": "The names of the entries in the user's vault. Values are never shown to you.",
}
# These tools change nothing by themselves and return no secret; only the person on the page can.
TOOL_HINTS = {"readOnlyHint": True, "openWorldHint": False}

NEEDS_ACCOUNT = "The vault needs an INFINITO account with sign-in; this memory uses a private address."
NOT_INVITED = ("The vault on this server is by invitation and this account has none, which cannot be enabled from "
               "here. Ask the user to keep the secret in their own vault or password manager.")

TEXT = {
    "es": {
        "vault_title": "Tu bóveda",
        "ask_add": "Guardar <b>{entry}</b>.",
        "ask_reveal": "Mostrar <b>{entry}</b>.",
        "check": "Te lo pide tu IA. Antes de escribir nada, comprueba que esta dirección es la de tu INFINITO.",
        "testing": "En pruebas: no guardes todavía claves críticas, como las del banco.",
        "replaces": "Ya tienes una entrada con este nombre: se sustituirá.",
        "creating": "Es la primera vez: elige una clave maestra de al menos {n} caracteres, distinta de tu contraseña "
                    "de INFINITO. Si la pierdes, nadie podrá abrir tu bóveda.",
        "master": "Clave maestra", "repeat_master": "Repítela", "username": "Usuario (opcional)",
        "value": "La clave que quieres guardar",
        "go_add": "Guardar", "go_reveal": "Mostrar",
        "saved_title": "Guardado",
        "saved": "<b>{entry}</b> ya está en tu bóveda. Puedes volver al chat: tu IA verá el nombre, nunca el valor.",
        "shown_user": "Usuario", "shown_value": "Clave",
        "shown_note": "Esta página se cierra sola en {n} segundos.",
        "wrong": "Esa clave maestra no abre tu bóveda.",
        "short_master": "La clave maestra necesita al menos {n} caracteres.",
        "differ_master": "Las dos claves maestras no coinciden.",
        "no_value": "Escribe la clave que quieres guardar.",
        "bad_value": "Esa clave es demasiado larga o tiene caracteres que no se pueden guardar.",
        "vault_expired": "Este enlace ha caducado o ya se ha usado. Pídele a tu IA uno nuevo.",
        "missing_title": "No está en tu bóveda",
        "missing": "No hay ninguna entrada llamada <b>{entry}</b>. Pídele a tu IA la lista.",
    },
    "en": {
        "vault_title": "Your vault",
        "ask_add": "Keep <b>{entry}</b>.",
        "ask_reveal": "Show <b>{entry}</b>.",
        "check": "Your AI asks for this. Before you type anything, check that this address is your INFINITO's.",
        "testing": "In testing: do not keep critical secrets yet, such as your bank's.",
        "replaces": "You already have an entry with this name: it will be replaced.",
        "creating": "First time: choose a master password of at least {n} characters, not your INFINITO password. If "
                    "you lose it, nobody can open your vault.",
        "master": "Master password", "repeat_master": "Repeat it", "username": "Username (optional)",
        "value": "The secret to keep",
        "go_add": "Keep it", "go_reveal": "Show it",
        "saved_title": "Saved",
        "saved": "<b>{entry}</b> is now in your vault. You can go back to the chat: your AI sees the name, never the "
                 "value.",
        "shown_user": "Username", "shown_value": "Secret",
        "shown_note": "This page closes by itself in {n} seconds.",
        "wrong": "That master password does not open your vault.",
        "short_master": "The master password needs at least {n} characters.",
        "differ_master": "The two master passwords differ.",
        "no_value": "Type the secret to keep.",
        "bad_value": "That secret is too long or has characters that cannot be kept.",
        "vault_expired": "This link has expired or was already used. Ask your AI for a new one.",
        "missing_title": "Not in your vault",
        "missing": "There is no entry called <b>{entry}</b>. Ask your AI for the list.",
    },
}


class ServerVault:
    """One person's KeePass file on the server, and the sealed list of its entries' names."""

    def __init__(self, folder: Path, account: str):
        self.folder = Path(folder)
        self.path = self.folder / FILE
        self.names_path = self.folder / NAMES_FILE
        self.account = account

    def exists(self) -> bool:
        return self.path.exists()

    def names(self, key: bytes) -> List[str]:
        if not self.names_path.exists():
            return []
        opened = _unseal(key, self.names_path.read_bytes(), f"vault-names:{self.account}")
        return json.loads(opened) if opened else []

    def _write_atomically(self, path: Path, write: Callable[[str], None]) -> None:
        self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        write(str(temporary))
        os.chmod(temporary, 0o600)
        temporary.replace(path)

    def _open(self, master: str):
        from pykeepass import PyKeePass
        from pykeepass.exceptions import CredentialsError

        try:
            return PyKeePass(str(self.path), password=master)
        except CredentialsError:
            return None

    @staticmethod
    def _find(db, entry: str):
        wanted = entry.casefold()
        return next((e for e in db.entries if (e.title or "").casefold() == wanted), None)

    def add(self, master: str, key: bytes, entry: str, username: str, value: str) -> bool:
        """Keep a value under a name, replacing one with that name. False when the master password is wrong."""
        from pykeepass import create_database

        self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.exists():
            db = self._open(master)
            if db is None:
                return False
        else:
            db = create_database(str(self.path.with_name(FILE + ".new")), password=master)
        found = self._find(db, entry)
        if found is not None:
            found.username, found.password = username, value
        else:
            db.add_entry(db.root_group, entry, username, value, force_creation=True)
        self._write_atomically(self.path, db.save)
        self.path.with_name(FILE + ".new").unlink(missing_ok=True)
        names = sorted(e.title for e in db.entries if e.title)
        sealed = _seal(key, json.dumps(names, ensure_ascii=False).encode(), f"vault-names:{self.account}")
        self._write_atomically(self.names_path, lambda name: Path(name).write_bytes(sealed))
        return True

    def reveal(self, master: str, entry: str) -> Optional[Tuple[str, Optional[str]]]:
        """(username, value), with None for the value when there is no such entry; None when the password is wrong."""
        db = self._open(master)
        if db is None:
            return None
        found = self._find(db, entry)
        return (found.username or "", found.password or "") if found is not None else ("", None)


def refused_name(entry: str) -> Optional[str]:
    """Why this cannot be an entry's name: names are read by the agent, so a secret must not arrive as one."""
    name = " ".join((entry or "").split())
    if not name:
        return "Not done: give the entry a name, such as 'Router de casa'."
    if len(name) > MAX_NAME:
        return f"Not done: an entry's name has at most {MAX_NAME} characters."
    if not name.isprintable():
        return "Not done: an entry's name cannot have control characters."
    if SensitiveInformationFilter().inspect(name).level in (SafetyLevel.FORBIDDEN, SafetyLevel.SENSITIVE):
        return "Not done: that looks like the secret itself. Give only a name for it, and do not repeat the value."
    return None


def answer_with_link(issuer: str, action: str, entry: str, issue: Callable[[dict], str]) -> str:
    refused = refused_name(entry)
    if refused:
        return refused
    name = " ".join(entry.split())
    token = issue({"action": action, "entry": name})
    return (f"Give the user this link and ask them to open it: {issuer}/vault?r={token}\n"
            f"It opens their vault page, for '{name}'. Whatever is secret is typed or shown there, never here. It "
            f"lasts {LINK_SECONDS // 60} minutes. Nothing changes until the user finishes on that page, so do not "
            "say it is done, and call vault_list to check.")


def list_answer(names: List[str]) -> str:
    if not names:
        return "The vault is empty."
    return "Entries in the vault (names only; values are never shown to you):\n" + "\n".join(f"- {n}" for n in names)


def _field(name: str, label: str, kind: str, complete: str, *, required: bool = True, focus: bool = False) -> str:
    return (f'<label for="{name}">{label}</label><input id="{name}" name="{name}" type="{kind}" '
            f'autocomplete="{complete}" spellcheck="false"{" required" if required else ""}'
            f'{" autofocus" if focus else ""}>')


def form_body(t: dict, token: str, action: str, entry: str, *, creating: bool, replaces: bool, error: str = "") -> str:
    e = html.escape
    fields = [_field("master", t["master"], "password", "off", focus=True)]
    notes = [f'<p class="note">{t["check"]}</p>']
    if action == "add":
        notes.append(f'<p class="note">{t["testing"]}</p>')
        if creating:
            notes.append(f'<p class="note">{t["creating"].format(n=MIN_MASTER)}</p>')
            fields.append(_field("repeat", t["repeat_master"], "password", "off"))
        elif replaces:
            notes.append(f'<p class="note">{t["replaces"]}</p>')
        fields += [_field("username", t["username"], "text", "off", required=False),
                   _field("value", t["value"], "password", "off")]
    asked = t["ask_add"] if action == "add" else t["ask_reveal"]
    return (f'<p>{asked.format(entry=e(entry))}</p>{"".join(notes)}'
            f'<form method="post" action="/vault"><input type="hidden" name="r" value="{e(token)}">{"".join(fields)}'
            + (f'<p class="error" role="alert">{e(error)}</p>' if error else "")
            + f'<button type="submit">{t["go_" + action]}</button></form>')


def shown_body(t: dict, username: str, value: str) -> str:
    e = html.escape
    user = f'<p class="note">{t["shown_user"]}</p><div class="code" style="user-select:all">{e(username)}</div>' \
        if username else ""
    return (f'{user}<p class="note">{t["shown_value"]}</p><div class="code" style="user-select:all">{e(value)}</div>'
            f'<p class="note">{t["shown_note"].format(n=SHOWN_SECONDS)}</p>')


def routes(accounts, grants, limiter) -> list:
    """The page a link opens, and the answer to it; limiter keeps one heavy password check at a time."""
    import anyio
    from starlette.routing import Route

    def texts(request) -> dict:
        base = _texts(request)
        return {**base, **TEXT[base["lang"]]}

    def expired(t: dict):
        return _page(t, t["expired_title"], f'<p>{t["vault_expired"]}</p>', status=400)

    def live(token: str) -> Optional[dict]:
        found = grants.find(GRANT, token)
        return found if found is not None and accounts.vault_allowed(found["account"]) else None

    async def page(request):
        t, token = texts(request), request.query_params.get("r", "")
        found = live(token)
        if found is None:
            return expired(t)
        action, entry = found["data"]["action"], found["data"]["entry"]
        vault = ServerVault(accounts.memory_folder(found["account"]), found["account"])
        known = entry.casefold() in {name.casefold() for name in vault.names(found["key"])}
        if action == "reveal" and not known:
            return _page(t, t["missing_title"], f'<p>{t["missing"].format(entry=html.escape(entry))}</p>')
        return _page(t, t["vault_title"], form_body(t, token, action, entry, creating=not vault.exists(), replaces=known))

    async def answer(request):
        t, form = texts(request), await request.form()
        token = str(form.get("r", ""))
        found = live(token)
        if found is None:
            return expired(t)
        account, action, entry = found["account"], found["data"]["action"], found["data"]["entry"]
        vault = ServerVault(accounts.memory_folder(account), account)
        creating = not vault.exists()
        master, value, username = str(form.get("master", "")), str(form.get("value", "")), str(form.get("username", ""))

        def again(message: str):
            known = entry.casefold() in {name.casefold() for name in vault.names(found["key"])}
            body = form_body(t, token, action, entry, creating=creating, replaces=known, error=message)
            return _page(t, t["vault_title"], body, status=400)

        tries = f"vault:{account}"  # after a few wrong master passwords, this account waits
        if accounts.waiting(tries):
            return again(t["wait"].format(n=accounts.waiting(tries)))
        if action == "add":
            if not value:
                return again(t["no_value"])
            # KeePass files are XML, which cannot hold control characters.
            if len(value) > MAX_VALUE or len(username) > MAX_VALUE or not (value + username).isprintable():
                return again(t["bad_value"])
            if creating and len(master) < MIN_MASTER:
                return again(t["short_master"].format(n=MIN_MASTER))
            if creating and master != str(form.get("repeat", "")):
                return again(t["differ_master"])
            work = functools.partial(vault.add, master, found["key"], entry, username, value)
        else:
            work = functools.partial(vault.reveal, master, entry)
        done = await anyio.to_thread.run_sync(work, limiter=limiter)
        if done is None or done is False:
            accounts._failed(tries)
            return again(t["wrong"])
        accounts._failures.pop(tries, None)
        grants.drop(token)
        if action == "add":
            return _page(t, t["saved_title"], f'<p>{t["saved"].format(entry=html.escape(entry))}</p>')
        username, shown = done
        if shown is None:
            return _page(t, t["missing_title"], f'<p>{t["missing"].format(entry=html.escape(entry))}</p>')
        response = _page(t, entry, shown_body(t, username, shown))
        response.headers["Refresh"] = f"{SHOWN_SECONDS}; url=/vault"  # no script: the browser leaves by itself
        return response

    return [Route("/vault", page, methods=["GET"]), Route("/vault", answer, methods=["POST"])]
