"""A vault that works from any app: the agent gets a link, the person types on the server's own page.

The vault on a computer (vault.py) opens windows on that computer's screen, which ChatGPT or a phone cannot
do. So on the server vault_add and vault_reveal answer with a link. The master password and the values are
typed and shown on that page: they never pass through the agent, its model or the chat, and no tool has a
parameter for a value or returns one.

Each person's vault is a standard KeePass file (KDBX) beside their sealed memory, so a copy opens with
KeePassXC or KeePassDX, without INFINITO. It is opened here, under the master password sent from the page,
only while that request is served; the password and the values are not stored or logged.

The agent knows a secret only by its name and by a note on what it is for, which the person can correct on
the page. Both are in a list sealed with the memory key, so apps can read it without the master password. A
note reaches the list only when typed on this page: the Notes of an entry made elsewhere may hold anything.

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
from typing import Callable, Dict, List, Optional, Tuple

from .remote_accounts import _page, _seal, _texts, _unseal
from .safety import SensitiveInformationFilter
from .types import SafetyLevel

LINK_SECONDS = 600
MAX_NAME = 80
MAX_NOTE = 200
MAX_VALUE = 4096
MIN_MASTER = 14
SHOWN_SECONDS = 60
GRANT = "vault"
KDBX_MAGIC = b"\x03\xd9\xa2\x9a"  # how every KeePass file starts
FILE = "vault.kdbx"
NAMES_FILE = "vault-names.sealed"

TOOL_DESCRIPTIONS = {
    "vault_add": "Keep a password, key or PIN in the user's vault. Give only a name that says what it is for, such as "
                 "'Panel de Hetzner', never the value. If you do not know what the secret is for, ask the user first: "
                 "do not invent a name. It answers with a link: give it to the user, who types the value on that page "
                 "and can correct the name and add a note, so the value never passes through you or the chat.",
    "vault_reveal": "Show the user a secret from their vault, by entry name: call vault_list first to find it. It "
                    "answers with a link: give it to the user, who sees the value on that page. You never receive it.",
    "vault_delete": "Delete an entry from the user's vault for good, by entry name: call vault_list first to find it. "
                    "It answers with a link: give it to the user, who confirms on that page with their master password. "
                    "Nothing is deleted until they do.",
    "vault_list": "The names of the entries in the user's vault, and the notes on what each is for. Values are never "
                  "shown to you.",
}
# These tools change nothing by themselves and return no secret; only the person on the page can.
TOOL_HINTS = {"readOnlyHint": True, "openWorldHint": False}

NEEDS_ACCOUNT = "The vault needs an INFINITO account with sign-in; this memory uses a private address."
NOT_INVITED = ("The vault on this server is by invitation and this account has none, which cannot be enabled from "
               "here. Ask the user to keep the secret in their own vault or password manager.")

TEXT = {
    "es": {
        "vault_title": "Tu bóveda",
        "ask_add": "Guardar una clave en tu bóveda.",
        "ask_reveal": "Mostrar <b>{entry}</b>.",
        "ask_delete": "Borrar <b>{entry}</b> de tu bóveda para siempre.",
        "delete_copies": "Las copias de seguridad la conservan, bajo la clave maestra que tenías, hasta caducar.",
        "deleted_title": "Borrada",
        "deleted": "<b>{entry}</b> ya no está en tu bóveda.",
        "go_delete": "Borrar para siempre",
        "check": "Te lo pide tu IA. Antes de escribir nada, comprueba que esta dirección es la de tu INFINITO.",
        "check_own": "Antes de escribir nada, comprueba que esta dirección es la de tu INFINITO.",
        "back_memory": "Volver a tu memoria",
        "testing": "En pruebas: no guardes todavía claves críticas, como las del banco.",
        "replaces": "Ya tienes una entrada con este nombre: se sustituirá.",
        "keep_value": "Si no escribes una clave ni un usuario, se conservan los que tiene.",
        "creating": "Es la primera vez: elige una clave maestra de al menos {n} caracteres, distinta de tu contraseña "
                    "de INFINITO. Si la pierdes, nadie podrá abrir tu bóveda.",
        "entry_name": "Nombre", "entry_note": "Para qué es (opcional)",
        "note_hint": "Tu IA leerá el nombre y esta nota, y pueden aparecer en el chat. No pongas aquí ninguna clave.",
        "master": "Clave maestra", "repeat_master": "Repítela", "username": "Usuario (opcional)",
        "value": "La clave que quieres guardar",
        "go_add": "Guardar", "go_reveal": "Mostrar",
        "saved_title": "Guardado",
        "saved": "<b>{entry}</b> ya está en tu bóveda. Puedes volver al chat: tu IA verá el nombre y la nota, nunca el "
                 "valor.",
        "shown_user": "Usuario", "shown_value": "Clave", "shown_about": "Para qué es",
        "shown_note": "Esta página se cierra sola en {n} segundos.",
        "wrong": "Esa clave maestra no abre tu bóveda.",
        "short_master": "La clave maestra necesita al menos {n} caracteres.",
        "differ_master": "Las dos claves maestras no coinciden.",
        "no_value": "Escribe la clave que quieres guardar.",
        "bad_value": "Esa clave es demasiado larga o tiene caracteres que no se pueden guardar.",
        "bad_entry": "Elige un nombre de hasta {n} caracteres que diga para qué es y no contenga la clave.",
        "bad_note": "La nota es demasiado larga o contiene una clave: tu IA la leerá.",
        "name_taken": "Ya tienes otra entrada con ese nombre. Elige otro para no sustituirla.",
        "vault_expired": "Este enlace ha caducado o ya se ha usado. Pídele a tu IA uno nuevo.",
        "missing_title": "No está en tu bóveda",
        "missing": "No hay ninguna entrada llamada <b>{entry}</b>. Pídele a tu IA la lista.",
    },
    "en": {
        "vault_title": "Your vault",
        "ask_add": "Keep a secret in your vault.",
        "ask_reveal": "Show <b>{entry}</b>.",
        "ask_delete": "Delete <b>{entry}</b> from your vault for good.",
        "delete_copies": "Backups keep it, under the master password you had, until they age out.",
        "deleted_title": "Deleted",
        "deleted": "<b>{entry}</b> is no longer in your vault.",
        "go_delete": "Delete for good",
        "check": "Your AI asks for this. Before you type anything, check that this address is your INFINITO's.",
        "check_own": "Before you type anything, check that this address is your INFINITO's.",
        "back_memory": "Back to your memory",
        "testing": "In testing: do not keep critical secrets yet, such as your bank's.",
        "replaces": "You already have an entry with this name: it will be replaced.",
        "keep_value": "If you type no secret or username, the ones it has are kept.",
        "creating": "First time: choose a master password of at least {n} characters, not your INFINITO password. If "
                    "you lose it, nobody can open your vault.",
        "entry_name": "Name", "entry_note": "What it is for (optional)",
        "note_hint": "Your AI reads the name and this note, and they may appear in the chat. Do not put a secret here.",
        "master": "Master password", "repeat_master": "Repeat it", "username": "Username (optional)",
        "value": "The secret to keep",
        "go_add": "Keep it", "go_reveal": "Show it",
        "saved_title": "Saved",
        "saved": "<b>{entry}</b> is now in your vault. You can go back to the chat: your AI sees the name and the note, "
                 "never the value.",
        "shown_user": "Username", "shown_value": "Secret", "shown_about": "What it is for",
        "shown_note": "This page closes by itself in {n} seconds.",
        "wrong": "That master password does not open your vault.",
        "short_master": "The master password needs at least {n} characters.",
        "differ_master": "The two master passwords differ.",
        "no_value": "Type the secret to keep.",
        "bad_value": "That secret is too long or has characters that cannot be kept.",
        "bad_entry": "Choose a name of up to {n} characters that says what it is for and does not hold the secret.",
        "bad_note": "The note is too long or holds a secret: your AI reads it.",
        "name_taken": "You already have another entry with that name. Choose another so it is not replaced.",
        "vault_expired": "This link has expired or was already used. Ask your AI for a new one.",
        "missing_title": "Not in your vault",
        "missing": "There is no entry called <b>{entry}</b>. Ask your AI for the list.",
    },
}


class ServerVault:
    """One person's KeePass file on the server, and the sealed list of its entries' names and notes."""

    def __init__(self, folder: Path, account: str):
        self.folder = Path(folder)
        self.path = self.folder / FILE
        self.names_path = self.folder / NAMES_FILE
        self.account = account

    def exists(self) -> bool:
        return self.path.exists()

    def listed(self, key: bytes) -> List[Dict[str, str]]:
        """Each entry's name and note, as the agent may see them."""
        if not self.names_path.exists():
            return []
        opened = _unseal(key, self.names_path.read_bytes(), f"vault-names:{self.account}")
        if not opened:
            return []
        # A list written before notes existed holds plain names.
        return [{"name": item, "note": ""} if isinstance(item, str) else item for item in json.loads(opened)]

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

    def add(self, master: str, key: bytes, entry: str, username: str, value: str, note: str = "") -> bool:
        """Keep a value under a name; an entry with that name is replaced, and keeps what is left empty.

        False when the master password is wrong; ValueError when there is nothing to keep."""
        from pykeepass import create_database

        self.folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.exists():
            db = self._open(master)
            if db is None:
                return False
        else:
            db = create_database(str(self.path.with_name(FILE + ".new")), password=master)
        found = self._find(db, entry)
        if found is None and not value:
            raise ValueError("no value")
        if found is None:
            found = db.add_entry(db.root_group, entry, username, value, notes=note, force_creation=True)
        else:
            found.notes = note
            if username:
                found.username = username
            if value:
                found.password = value
        self._write_atomically(self.path, db.save)
        self.path.with_name(FILE + ".new").unlink(missing_ok=True)
        notes = {item["name"].casefold(): item["note"] for item in self.listed(key)}
        notes[entry.casefold()] = note
        self._write_names(db, key, notes)
        return True

    def _write_names(self, db, key: bytes, notes: Dict[str, str]) -> None:
        """The list the agent reads, from the file itself. Only notes typed on the page: the Notes of an entry made
        elsewhere may hold anything."""
        items = sorted(({"name": e.title, "note": notes.get(e.title.casefold(), "")} for e in db.entries if e.title),
                       key=lambda item: item["name"].casefold())
        sealed = _seal(key, json.dumps(items, ensure_ascii=False).encode(), f"vault-names:{self.account}")
        self._write_atomically(self.names_path, lambda name: Path(name).write_bytes(sealed))

    def delete(self, master: str, key: bytes, entry: str) -> Optional[bool]:
        """Delete an entry for good, with its history, from the file and from the list; None for a wrong password."""
        db = self._open(master)
        if db is None:
            return None
        found = self._find(db, entry)
        if found is not None:
            db.delete_entry(found)
            self._write_atomically(self.path, db.save)
        self._write_names(db, key, {item["name"].casefold(): item["note"] for item in self.listed(key)})
        return True

    def reveal(self, master: str, entry: str) -> Optional[Tuple[str, Optional[str], str]]:
        """(username, value, note), with None for the value when there is no such entry; None for a wrong password."""
        db = self._open(master)
        if db is None:
            return None
        found = self._find(db, entry)
        if found is None:
            return "", None, ""
        return found.username or "", found.password or "", found.notes or ""


def _looks_secret(text: str) -> bool:
    return SensitiveInformationFilter().inspect(text).level == SafetyLevel.FORBIDDEN


def refused_name(entry: str) -> Optional[str]:
    """Why this cannot be an entry's name: names are read by the agent, so a secret must not arrive as one."""
    name = " ".join((entry or "").split())
    if not name:
        return "Not done: give the entry a name, such as 'Router de casa'."
    if len(name) > MAX_NAME:
        return f"Not done: an entry's name has at most {MAX_NAME} characters."
    if not name.isprintable():
        return "Not done: an entry's name cannot have control characters."
    if _looks_secret(name):
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
            "say it is done, and call vault_list to check: they may change the name there.")


def list_answer(items: List[Dict[str, str]]) -> str:
    if not items:
        return "The vault is empty."
    lines = [f"- {item['name']}: {item['note']}" if item["note"] else f"- {item['name']}" for item in items]
    return "Entries in the vault (names and notes only; values are never shown to you):\n" + "\n".join(lines)


def _field(name: str, label: str, kind: str, *, required: bool = True, value: str = "") -> str:
    return (f'<label for="{name}">{label}</label><input id="{name}" name="{name}" type="{kind}" autocomplete="off" '
            f'spellcheck="false" value="{html.escape(value)}"{" required" if required else ""}>')


def form_body(t: dict, token: str, action: str, entry: str, *, creating: bool, replaces: bool, name: Optional[str] = None,
              note: str = "", error: str = "", own: bool = False) -> str:
    e = html.escape
    master = _field("master", t["master"], "password")  # no autofocus: it would scroll past the address to check
    notes = [f'<p class="note">{t["check_own" if own else "check"]}</p>']
    if action == "add":
        notes.append(f'<p class="note">{t["testing"]}</p>')
        fields = [_field("name", t["entry_name"], "text", value=entry if name is None else name),
                  _field("note", t["entry_note"], "text", required=False, value=note),
                  f'<p class="hint">{t["note_hint"]}</p>', master]
        if creating:
            notes.append(f'<p class="note">{t["creating"].format(n=MIN_MASTER)}</p>')
            fields.append(_field("repeat", t["repeat_master"], "password"))
        elif replaces:
            notes.append(f'<p class="note">{t["replaces"]} {t["keep_value"]}</p>')
        fields += [_field("username", t["username"], "text", required=False),
                   _field("value", t["value"], "password", required=not replaces)]
        asked = t["ask_add"]
    else:
        fields = [master]
        asked = t["ask_" + action].format(entry=e(entry))
        if action == "delete":
            notes.append(f'<p class="note">{t["delete_copies"]}</p>')
    return (f'<p>{asked}</p>{"".join(notes)}<form method="post" action="/vault">'
            f'<input type="hidden" name="r" value="{e(token)}">{"".join(fields)}'
            + (f'<p class="error" role="alert">{e(error)}</p>' if error else "")
            + f'<button type="submit">{t["go_" + action]}</button></form>')


def shown_body(t: dict, username: str, value: str, note: str) -> str:
    e = html.escape
    about = f'<p class="note">{t["shown_about"]}</p><p>{e(note)}</p>' if note else ""
    user = f'<p class="note">{t["shown_user"]}</p><div class="code" style="user-select:all">{e(username)}</div>' \
        if username else ""
    return (f'{about}{user}<p class="note">{t["shown_value"]}</p>'
            f'<div class="code" style="user-select:all">{e(value)}</div>'
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

    def missing(t: dict, entry: str):
        return _page(t, t["missing_title"], f'<p>{t["missing"].format(entry=html.escape(entry))}</p>')

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
        known = {item["name"].casefold(): item["note"] for item in vault.listed(found["key"])}
        if action != "add" and entry.casefold() not in known:
            return missing(t, entry)
        body = form_body(t, token, action, entry, creating=not vault.exists(), replaces=entry.casefold() in known,
                         note=known.get(entry.casefold(), ""), own=found["data"].get("own", False))
        return _page(t, t["vault_title"], body)

    async def answer(request):
        t, form = texts(request), await request.form()
        token = str(form.get("r", ""))
        found = live(token)
        if found is None:
            return expired(t)
        account, action, entry = found["account"], found["data"]["action"], found["data"]["entry"]
        vault = ServerVault(accounts.memory_folder(account), account)
        creating = not vault.exists()
        known = {item["name"].casefold(): item["note"] for item in vault.listed(found["key"])}
        master, value, username = str(form.get("master", "")), str(form.get("value", "")), str(form.get("username", ""))
        name = " ".join(str(form.get("name", entry)).split())
        note = " ".join(str(form.get("note", "")).split())

        def again(message: str):
            body = form_body(t, token, action, entry, creating=creating, replaces=entry.casefold() in known, name=name,
                             note=note, error=message, own=found["data"].get("own", False))
            return _page(t, t["vault_title"], body, status=400)

        tries = f"vault:{account}"  # after a few wrong master passwords, this account waits
        if accounts.waiting(tries):
            return again(t["wait"].format(n=accounts.waiting(tries)))
        if action == "add":
            if refused_name(name) or (value and value in name):
                name = entry  # what was typed may hold the secret: it is not sent back
                return again(t["bad_entry"].format(n=MAX_NAME))
            if len(note) > MAX_NOTE or not note.isprintable() or _looks_secret(note) or (value and value in note):
                note = ""  # likewise
                return again(t["bad_note"])
            if name.casefold() != entry.casefold() and name.casefold() in known:
                return again(t["name_taken"])
            if not value and name.casefold() not in known:
                return again(t["no_value"])
            # KeePass files are XML, which cannot hold control characters.
            if len(value) > MAX_VALUE or len(username) > MAX_VALUE or not (value + username).isprintable():
                return again(t["bad_value"])
            if creating and len(master) < MIN_MASTER:
                return again(t["short_master"].format(n=MIN_MASTER))
            if creating and master != str(form.get("repeat", "")):
                return again(t["differ_master"])
            work = functools.partial(vault.add, master, found["key"], name, username, value, note)
        elif action == "delete":
            work = functools.partial(vault.delete, master, found["key"], entry)
        else:
            work = functools.partial(vault.reveal, master, entry)
        try:
            done = await anyio.to_thread.run_sync(work, limiter=limiter)
        except ValueError:
            return again(t["bad_value"])
        if done is None or done is False:
            accounts._failed(tries)
            return again(t["wrong"])
        accounts._failures.pop(tries, None)
        grants.drop(token)
        if action == "add":
            return _page(t, t["saved_title"], f'<p>{t["saved"].format(entry=html.escape(name))}</p>')
        if action == "delete":
            back = f'<p><a href="/memory">{t["back_memory"]}</a></p>' if found["data"].get("own") else ""
            return _page(t, t["deleted_title"], f'<p>{t["deleted"].format(entry=html.escape(entry))}</p>{back}')
        username, shown, about = done
        if shown is None:
            return missing(t, entry)
        response = _page(t, entry, shown_body(t, username, shown, about))
        response.headers["Refresh"] = f"{SHOWN_SECONDS}; url=/vault"  # no script: the browser leaves by itself
        return response

    return [Route("/vault", page, methods=["GET"]), Route("/vault", answer, methods=["POST"])]
