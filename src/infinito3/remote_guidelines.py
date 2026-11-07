"""The person's own guidelines for all their AIs: written once, on their memory page, and read by every app.

"Answer me in Spanish", "sign as Hilo when you use my memory", "look at my memory before answering about my work":
what someone would otherwise configure in each app, one by one. Here it is a short list, one guideline per line, that
the person edits on their memory page. It comes first in what every app reads (recall, list_memories, overview,
briefing) and in the hook that brings the memory to each message, under its own heading: these are the person's
instructions, while the memories below them stay data.

Only the person changes the list, signed in with their password. An app can propose a guideline when the person asks
for one in a chat (guideline_propose), and it waits on the page until the person accepts it with a click: otherwise a
web page or an email an agent reads could plant an instruction that every other AI would then follow. The page also
offers, one click each, what the memory already holds that reads like a guideline ("Prefiero las explicaciones en
español", "firma con Hilo").

The list is sealed with the memory's key, next to it (guidelines.sealed), and goes in the backups. Secrets are refused
in it, as in the memory.
"""
import html
import json
import os
import re
import secrets
import threading
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

FILE = "guidelines.sealed"
MAGIC = b"GDL1"
MAX_ITEMS = 20
MAX_CHARS = 400
MAX_PROPOSED = 20
SUGGESTED = 8

# The first line of what the apps read; the hook on the person's computer looks for it (remote_client.py).
HEADER = "The user's own guidelines for all their AIs, which they wrote on their INFINITO page. Follow them:"

TOOL_DESCRIPTION = ("Propose a guideline the user wants every one of their AIs to follow from now on, when they ask "
                    "for one (\"from now on answer me in Spanish\", \"always sign as Hilo\"), in their words. It waits "
                    "on their memory page until they accept it with a click; it is not a guideline before. The "
                    "guidelines they accepted come first in recall and briefing.")
TOOL_HINTS = {"title": "Propose a guideline", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
              "openWorldHint": False}
NEEDS_ACCOUNT = "Guidelines need an INFINITO account with sign-in; this memory uses a private address."
INSTRUCTIONS = """
- recall and briefing start with the user's own guidelines, which they wrote on their INFINITO page: follow them. When
  the user asks you to always do something, call guideline_propose; they accept it on that page."""

# What reads like a guideline among someone's memories: what they asked their AIs to do, or how they want answers.
_LOOKS_LIKE = re.compile(
    r"\b(cuando uses|firma|firmes|firmaras|respondeme|me respondas|responde en|lo primero que hagas|quiero que en cada|"
    r"quiero que siempre|quiero que cuando|quiero que me|prefiero las explicaciones|prefiero que|no hace falta que|"
    r"a partir de ahora|from now on|always answer|answer me|reply to me|sign as|i prefer)\b")

TEXT = {
    "es": {
        "nav_guidelines": "Pautas", "guidelines_title": "Tus pautas",
        "guidelines_note": "Lo que todas tus IAs deben tener en cuenta siempre. Les llega lo primero cada vez que "
                           "consultan tu memoria, en ChatGPT, Claude, Grok o cualquier otra. Una pauta por línea.",
        "guidelines_hint": "Por ejemplo:\nRespóndeme siempre en español.\nFirma con Hilo cuando uses mi memoria.\n"
                           "Antes de responder sobre mí o mi trabajo, consulta mi memoria.",
        "guidelines_save": "Guardar las pautas", "guidelines_saved": "Guardadas. Tus IAs las verán en su próxima consulta.",
        "guidelines_too_many": "Caben {n} pautas como mucho.", "guidelines_too_long": "Cada pauta cabe en {n} caracteres.",
        "guidelines_refused": "No se han guardado, y siguen como estaban: una línea parece una clave, y eso no va aquí.",
        "guidelines_proposed": "Propuestas por tus IAs", "guidelines_proposed_by": "propuesta por {app} el {day}",
        "guidelines_accept": "Añadir", "guidelines_discard": "Descartar",
        "guidelines_accepted": "Añadida.", "guidelines_discarded": "Descartada.",
        "guidelines_suggested": "En tu memoria hay frases que parecen pautas",
        "guidelines_suggested_note": "Añádelas con un clic si quieres que todas tus IAs las sigan.",
        "guidelines_gone": "Eso ya no está. Mira cómo están ahora tus pautas.",
    },
    "en": {
        "nav_guidelines": "Guidelines", "guidelines_title": "Your guidelines",
        "guidelines_note": "What all your AIs should always keep in mind. It reaches them first every time they look at "
                           "your memory, in ChatGPT, Claude, Grok or any other. One guideline per line.",
        "guidelines_hint": "For example:\nAlways answer me in English.\nSign as Hilo when you use my memory.\n"
                           "Before answering about me or my work, look at my memory.",
        "guidelines_save": "Save the guidelines", "guidelines_saved": "Saved. Your AIs will see them the next time they look.",
        "guidelines_too_many": "At most {n} guidelines.", "guidelines_too_long": "Each guideline fits in {n} characters.",
        "guidelines_refused": "Not saved, and they stay as they were: a line looks like a secret, and that does not go "
                              "here.",
        "guidelines_proposed": "Proposed by your AIs", "guidelines_proposed_by": "proposed by {app} on {day}",
        "guidelines_accept": "Add", "guidelines_discard": "Discard",
        "guidelines_accepted": "Added.", "guidelines_discarded": "Discarded.",
        "guidelines_suggested": "Your memory has sentences that read like guidelines",
        "guidelines_suggested_note": "Add them with a click if you want all your AIs to follow them.",
        "guidelines_gone": "That is no longer there. See how your guidelines are now.",
    },
}

_locks: Dict[str, threading.Lock] = {}
_guard = threading.Lock()


class Refused(ValueError):
    """What was asked cannot be kept; the message says why ("too_many", "too_long", "secret" or one for the agent)."""


def _lock_of(folder: Path) -> threading.Lock:
    with _guard:
        return _locks.setdefault(str(Path(folder).resolve()), threading.Lock())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty() -> dict:
    return {"items": [], "proposed": [], "changed": ""}


def _read(folder: Path, key: bytes, account: str) -> dict:
    from .remote_accounts import _unseal

    path = Path(folder) / FILE
    if not path.exists():
        return _empty()
    blob = path.read_bytes()
    plain = _unseal(key, blob[len(MAGIC):], f"guidelines:{account}") if blob.startswith(MAGIC) else None
    if plain is None:
        raise ValueError(f"{path} is not this memory's guidelines.")
    return {**_empty(), **json.loads(plain.decode("utf-8"))}


def _write(folder: Path, key: bytes, account: str, data: dict) -> None:
    from .remote_accounts import _seal

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (FILE + ".tmp")
    temporary.write_bytes(MAGIC + _seal(key, json.dumps(data, ensure_ascii=False).encode("utf-8"),
                                        f"guidelines:{account}"))
    os.chmod(temporary, 0o600)
    temporary.replace(folder / FILE)


def read(folder: Path, key: bytes, account: str) -> dict:
    with _lock_of(folder):
        return _read(folder, key, account)


def change(folder: Path, key: bytes, account: str, action) -> object:
    """Runs action on the guidelines and keeps what it changed; if it raises, nothing is written."""
    with _lock_of(folder):
        data = _read(folder, key, account)
        result = action(data)
        _write(folder, key, account, data)
        return result


def _secret(text: str) -> bool:
    from .safety import SensitiveInformationFilter
    from .types import SafetyLevel

    return SensitiveInformationFilter().inspect(text).level == SafetyLevel.FORBIDDEN


def _same(one: str, other: str) -> bool:
    def plain(text: str) -> str:
        folded = unicodedata.normalize("NFKD", text.casefold())
        return " ".join(re.findall(r"\w+", "".join(c for c in folded if not unicodedata.combining(c))))
    return plain(one) == plain(other)


def lines_of(text: str) -> List[str]:
    """What the person typed, as guidelines: one per line, without bullets, blank lines or repeats."""
    found: List[str] = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•·]|\d+[.)])\s*", "", line).strip()
        if line and not any(_same(line, kept) for kept in found):
            found.append(line)
    return found


def _checked(items: Sequence[str]) -> List[str]:
    if len(items) > MAX_ITEMS:
        raise Refused("too_many")
    if any(len(item) > MAX_CHARS for item in items):
        raise Refused("too_long")
    if any(_secret(item) for item in items):
        raise Refused("secret")
    return list(items)


# What the person does, on their page

def replace(data: dict, text: str) -> None:
    data["items"] = _checked(lines_of(text))
    data["changed"] = _now()


def add(data: dict, text: str) -> None:
    """One more guideline at the end, unless the list already says it."""
    text = " ".join((text or "").split())
    if text and not any(_same(text, item) for item in data["items"]):
        data["items"] = _checked(data["items"] + [text])
        data["changed"] = _now()


def accept(data: dict, proposal_id: str) -> None:
    found = next((p for p in data["proposed"] if p["id"] == proposal_id), None)
    if found is None:
        raise Refused("gone")
    add(data, found["text"])
    data["proposed"] = [p for p in data["proposed"] if p["id"] != proposal_id]


def discard(data: dict, proposal_id: str) -> None:
    if not any(p["id"] == proposal_id for p in data["proposed"]):
        raise Refused("gone")
    data["proposed"] = [p for p in data["proposed"] if p["id"] != proposal_id]


# What an app does

def propose(data: dict, app: Optional[str], text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        raise Refused("Not done: say the guideline, in the user's words.")
    if len(text) > MAX_CHARS:
        raise Refused(f"Not done: one guideline is at most {MAX_CHARS} characters.")
    if _secret(text):
        raise Refused("Not done: it looks like it holds a secret, and secrets never go in the guidelines. Do not repeat "
                      "it.")
    if any(_same(text, item) for item in data["items"]):
        return "already"
    if any(_same(text, p["text"]) for p in data["proposed"]):
        return "waiting"
    if len(data["proposed"]) >= MAX_PROPOSED:
        raise Refused("Not done: the user already has many guidelines waiting. Ask them to look at their page first.")
    data["proposed"].append({"id": secrets.token_hex(4), "text": text, "by": app or "an app", "at": _now()})
    return "proposed"


def block(data: dict, issuer: str) -> str:
    """What every app reads first: the person's guidelines, and how many of its proposals still wait for them."""
    lines = [HEADER] + [f"- {item}" for item in data["items"]] if data["items"] else []
    if data["proposed"]:
        waiting = (f"{len(data['proposed'])} proposed guideline{'s' if len(data['proposed']) != 1 else ''} wait for the "
                   f"user to accept at {issuer}/memory?view=guidelines; they are not guidelines yet.")
        lines = (lines or ["The user has no guidelines for their AIs yet."]) + [waiting]
    return "\n".join(lines)


def suggestions(records: Sequence, data: dict) -> List:
    """The person's memories that read like guidelines and are not in the list yet, most recent first."""
    def plain(text: str) -> str:
        folded = unicodedata.normalize("NFKD", text.casefold())
        return "".join(c for c in folded if not unicodedata.combining(c))

    found = []
    for record in reversed(list(records)):
        if len(record.content) <= MAX_CHARS and _LOOKS_LIKE.search(plain(record.content)) \
                and not any(_same(record.content, item) for item in data["items"]):
            found.append(record)
            if len(found) >= SUGGESTED:
                break
    return found


# The person's page

def guidelines_body(t: dict, data: dict, suggested: Sequence, csrf: str, hidden_fields, *, message: str = "",
                    error: str = "") -> str:
    e = html.escape
    notice = ((f'<p class="ok" role="status">{e(message)}</p>' if message else "")
              + (f'<p class="error" role="alert">{e(error)}</p>' if error else ""))
    editor = (f'<form method="post" action="/memory">{hidden_fields(action="guidelines_save", csrf=csrf)}'
              f'<label for="guidelines" class="note">{t["guidelines_note"]}</label>'
              f'<textarea id="guidelines" name="text" spellcheck="true" style="min-height:11rem" '
              f'placeholder="{e(t["guidelines_hint"])}">{e(chr(10).join(data["items"]))}</textarea>'
              f'<div class="actions"><button class="quiet" type="submit">{t["guidelines_save"]}</button></div></form>')
    proposed = ""
    if data["proposed"]:
        rows = "".join(
            f'<div class="memo"><div><div>{e(p["text"])}</div><p class="meta">'
            f'{e(t["guidelines_proposed_by"].format(app=p["by"], day=p["at"][:10]))}</p>'
            f'<form method="post" action="/memory" class="actions">{hidden_fields(csrf=csrf, id=p["id"])}'
            f'<button class="quiet" type="submit" name="action" value="guideline_accept" '
            f'aria-label="{t["guidelines_accept"]}: {e(p["text"])}">{t["guidelines_accept"]}</button>'
            f'<button class="bad-quiet" type="submit" name="action" value="guideline_discard" '
            f'aria-label="{t["guidelines_discard"]}: {e(p["text"])}">{t["guidelines_discard"]}'
            f'</button></form></div></div>' for p in data["proposed"])
        proposed = f'<h2>{t["guidelines_proposed"]}</h2>{rows}'
    from_memory = ""
    if suggested:
        rows = "".join(
            f'<div class="memo"><div><div>{e(record.content)}</div>'
            f'<form method="post" action="/memory" class="actions">{hidden_fields(csrf=csrf, id=record.id)}'
            f'<button class="quiet" type="submit" name="action" value="guideline_from_memory" '
            f'aria-label="{t["guidelines_accept"]}: {e(record.content)}">{t["guidelines_accept"]}'
            f'</button></form></div></div>' for record in suggested)
        from_memory = (f'<h2>{t["guidelines_suggested"]}</h2><p class="note">{t["guidelines_suggested_note"]}</p>'
                       f'{rows}')
    return f"{notice}{editor}{proposed}{from_memory}"
