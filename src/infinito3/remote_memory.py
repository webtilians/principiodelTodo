"""A person's own page for their memory: see it, search it, correct it and delete from it, with no AI in between.

An agent can only do what its app lets it, and ChatGPT refuses to run forget. So the memory tool memory_page answers
with a link here, where the person confirms the deletion themselves. The page is also how anyone sees what INFINITO
keeps about them, which no chat can show as plainly.

A link opens nothing by itself, because chat apps may fetch one first: it only names which memories to confirm, and
holds no key. The page always asks for the password, which is what unseals the memory. That key is then kept for a
quarter of an hour the same way an app's sign-in keeps it: wrapped under a token of which the server stores only a
hash, in a cookie that scripts cannot read and other sites cannot send. There are no scripts on the page.

Whatever a person types to correct a memory goes through the same checks as what an agent sends (remember), so a
secret cannot reach the memory this way either, and what was typed is not sent back. The same goes for the words of a
photo, which the person adds here and which is kept sealed (remote_photos.py).
"""
import functools
import hmac
import html
import secrets
import unicodedata
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence
from urllib.parse import urlencode

from . import remote_apps, remote_photos, remote_portal, remote_tasks, remote_vault
from .mcp_server import BY_THE_USER, KINDS, OPEN_KINDS, OVERVIEW_OPEN_ITEMS, STALE_DAYS, AgentMemory, _day
from .remote_accounts import _HEADERS, NAME, _STYLE as _BASE_STYLE, _page, _texts
from .types import MemoryRecord

SESSION = "memory-session"
LINK = "memory-link"
SESSION_SECONDS = 900
LINK_SECONDS = 600
PAGE_SIZE = 50
MAX_IDS = 500
COOKIE = "infinito_memory"
# Memories saved without saying what they are: mostly what someone told an app about themselves.
NO_KIND = "none"
NEURONS = KINDS + (NO_KIND,)

TOOL_DESCRIPTION = ("A link to the page where the user sees their memory and searches, corrects or deletes it "
                    "themselves. Give it memory_ids to have them confirm deleting those: use it when your app blocks "
                    "or refuses forget, as ChatGPT does, or when they want to look before deleting. Find the ids first "
                    "with list_memories or recall. Nothing is deleted until the user confirms on that page.")
# It changes nothing by itself; only the person on the page can.
TOOL_HINTS = {"title": "Memory page", "readOnlyHint": True, "openWorldHint": False}
INSTRUCTIONS = """
- If your app blocks or refuses forget, do not give up and do not say it is deleted: call memory_page with those ids and
  give the user the link, so they confirm it themselves. It is also the page to offer when they want to see what is kept."""
NEEDS_ACCOUNT = "The memory page needs an INFINITO account with sign-in; this memory uses a private address."

TEXT = {
    "es": {
        "title": "Tu memoria",
        "sign_in_note": "Escribe tu contraseña para abrirla. Antes, comprueba que esta dirección es la de tu INFINITO.",
        "asked": "Tu IA te pide que confirmes qué borrar. Lo verás aquí antes de que se borre nada.",
        "go_in": "Abrir mi memoria",
        "session_over": "Tu sesión ha terminado. Vuelve a entrar.",
        "link_expired_title": "Este enlace ha caducado",
        "link_expired": "Pídele a tu IA uno nuevo, o <a href=\"/memory\">abre tu memoria</a> y hazlo desde ahí.",
        "other_memory": "Este enlace es de otra memoria. <a href=\"/memory\">Vuelve a la tuya</a>.",
        "count": "{n} recuerdos", "count_one": "1 recuerdo", "count_of": "{n} de {total} recuerdos",
        "empty": "Tu memoria está vacía. Lo que le cuentes a tus IAs aparecerá aquí.",
        "nothing_found": "No hay nada con esa búsqueda.",
        "search": "Buscar", "search_hint": "Palabras que contenga",
        "all_projects": "Todos los clientes", "all_kinds": "Todos los tipos", "clear": "Quitar filtros",
        "kinds": {"fact": "Hecho", "decision": "Decisión", "agreement": "Acuerdo", "pending": "Pendiente",
                  "proposal": "Propuesta", "assumption": "Suposición", "to_verify": "Por verificar", "none": "Sobre ti"},
        "kind_about": {"fact": "lo que es así", "decision": "lo que se decidió", "agreement": "lo acordado con alguien",
                       "pending": "lo que falta por hacer", "proposal": "ideas sin decidir",
                       "assumption": "lo que se da por supuesto", "to_verify": "lo que hay que comprobar",
                       "none": "lo que les has contado a tus IAs"},
        "menu": "Menú",
        "source": "fuente", "by": "guardado por", "by_you": "guardado por ti", "correct": "Corregir",
        "delete_marked": "Borrar los marcados", "delete_found": "Borrar los {n} de este filtro",
        "newer": "Más recientes", "older": "Más antiguos", "sign_out": "Cerrar sesión",
        "none_marked": "No has marcado ninguno.",
        "confirm_title": "¿Borrar para siempre?",
        "confirm": "Se borrarán {n} recuerdos, con los valores anteriores de esos hechos. Desmarca los que quieras "
                   "conservar.",
        "confirm_one": "Se borrará este recuerdo, con los valores anteriores de ese hecho.",
        "confirm_copies": "Las copias de seguridad lo conservan, sellado, hasta caducar: 14 días en el servidor y 30 en "
                          "el ordenador que las descargue.",
        "go_delete": "Borrar para siempre", "cancel": "Cancelar",
        "gone_title": "No queda nada que borrar",
        "gone": "Esos recuerdos ya no están en tu memoria.",
        "deleted_title": "Borrado",
        "deleted": "Se han borrado {n} recuerdos. Tus IAs dejarán de encontrarlos.",
        "deleted_one": "Se ha borrado 1 recuerdo. Tus IAs dejarán de encontrarlo.",
        "back": "Volver a tu memoria", "back_chat": "Ya puedes volver al chat.",
        "correct_title": "Corregir un recuerdo",
        "correct_note": "La versión anterior se borra, y tus IAs verán solo esta.",
        "text": "Qué debe recordar", "go_correct": "Guardar la corrección",
        "unchanged": "No has cambiado nada.",
        "refused": "No se ha guardado, y el recuerdo sigue como estaba: parece una clave o es una pregunta, y eso no "
                   "va en la memoria.",
        "corrected": "Corregido.",
        "missing_title": "Ese recuerdo ya no está",
        "bad_form": "Esta página se ha quedado antigua. Vuelve a abrir tu memoria.",
        "vault": "Tu bóveda", "vault_note": "Solo los nombres: para ver o borrar una clave te pedirá la clave maestra.",
        "vault_show": "Ver", "vault_delete": "Borrar",
        "photo_add": "Añadir una foto", "photo_file": "La foto o el pantallazo", "photo_about": "Qué es",
        "photo_about_hint": "Por ejemplo: el cartel del refugio en el viaje a la cordillera",
        "photo_project": "Cliente o proyecto (si quieres)",
        "photo_note": "JPG, PNG, WebP o GIF, hasta 20 MB. Se guarda cifrada, y tus IAs la encuentran por lo que escribas aquí.",
        "photo_go": "Guardar la foto", "photo_saved": "Foto guardada.", "photo_title": "Tu foto",
        "photo_empty": "No has elegido ninguna foto.", "photo_too_big": "Esa foto pasa de 20 MB y no se ha guardado.",
        "photo_heic": "Las fotos HEIC aún no se pueden guardar. Pásala a JPG o haz un pantallazo de ella.",
        "photo_unknown": "Ese archivo no es una imagen JPG, PNG, WebP o GIF, y no se ha guardado.",
        "photo_full": "Tu espacio para fotos está lleno. Borra alguna antes de añadir otra.",
        "photo_refused": "No se ha guardado: lo que has escrito parece una clave o es una pregunta, y eso no va en la "
                         "memoria.",
    },
    "en": {
        "title": "Your memory",
        "sign_in_note": "Type your password to open it. First, check that this address is your INFINITO's.",
        "asked": "Your AI asks you to confirm what to delete. You will see it here before anything is deleted.",
        "go_in": "Open my memory",
        "session_over": "Your session has ended. Sign in again.",
        "link_expired_title": "This link has expired",
        "link_expired": "Ask your AI for a new one, or <a href=\"/memory\">open your memory</a> and do it from there.",
        "other_memory": "This link belongs to another memory. <a href=\"/memory\">Back to yours</a>.",
        "count": "{n} memories", "count_one": "1 memory", "count_of": "{n} of {total} memories",
        "empty": "Your memory is empty. What you tell your AIs will appear here.",
        "nothing_found": "Nothing matches that search.",
        "search": "Search", "search_hint": "Words it contains",
        "all_projects": "All clients", "all_kinds": "All kinds", "clear": "Clear filters",
        "kinds": {"fact": "Fact", "decision": "Decision", "agreement": "Agreement", "pending": "Pending",
                  "proposal": "Proposal", "assumption": "Assumption", "to_verify": "To verify", "none": "About you"},
        "kind_about": {"fact": "what is so", "decision": "what was decided", "agreement": "what was agreed with someone",
                       "pending": "what is still to do", "proposal": "ideas not decided yet",
                       "assumption": "what is taken for granted", "to_verify": "what needs checking",
                       "none": "what you told your AIs"},
        "menu": "Menu",
        "source": "source", "by": "saved by", "by_you": "saved by you", "correct": "Correct",
        "delete_marked": "Delete the marked ones", "delete_found": "Delete the {n} in this filter",
        "newer": "Newer", "older": "Older", "sign_out": "Sign out",
        "none_marked": "You marked none.",
        "confirm_title": "Delete for good?",
        "confirm": "{n} memories will be deleted, with the earlier values of those facts. Unmark any you want to keep.",
        "confirm_one": "This memory will be deleted, with the earlier values of that fact.",
        "confirm_copies": "Backups keep it, sealed, until they age out: 14 days on the server and 30 on a computer that "
                          "brings them.",
        "go_delete": "Delete for good", "cancel": "Cancel",
        "gone_title": "Nothing left to delete",
        "gone": "Those memories are no longer in your memory.",
        "deleted_title": "Deleted",
        "deleted": "{n} memories were deleted. Your AIs will no longer find them.",
        "deleted_one": "1 memory was deleted. Your AIs will no longer find it.",
        "back": "Back to your memory", "back_chat": "You can go back to the chat.",
        "correct_title": "Correct a memory",
        "correct_note": "The earlier version is deleted, and your AIs will see only this one.",
        "text": "What it should remember", "go_correct": "Save the correction",
        "unchanged": "You changed nothing.",
        "refused": "Not saved, and the memory stays as it was: it looks like a secret, or it is a question, and those "
                   "do not go in the memory.",
        "corrected": "Corrected.",
        "missing_title": "That memory is gone",
        "bad_form": "This page has gone stale. Open your memory again.",
        "vault": "Your vault", "vault_note": "Names only: seeing or deleting a secret asks for the master password.",
        "vault_show": "Show", "vault_delete": "Delete",
        "photo_add": "Add a photo", "photo_file": "The photo or screenshot", "photo_about": "What it is",
        "photo_about_hint": "For example: the sign at the refuge on the trip to the mountains",
        "photo_project": "Client or project (if you like)",
        "photo_note": "JPG, PNG, WebP or GIF, up to 20 MB. It is kept encrypted, and your AIs find it by what you write here.",
        "photo_go": "Save the photo", "photo_saved": "Photo saved.", "photo_title": "Your photo",
        "photo_empty": "You chose no photo.", "photo_too_big": "That photo is over 20 MB and was not saved.",
        "photo_heic": "HEIC photos cannot be kept yet. Turn it into a JPG or take a screenshot of it.",
        "photo_unknown": "That file is not a JPG, PNG, WebP or GIF picture, and was not saved.",
        "photo_full": "Your space for photos is full. Delete some before adding another.",
        "photo_refused": "Not saved: what you wrote looks like a secret, or it is a question, and those do not go in the "
                         "memory.",
    },
}

_STYLE = """
main { max-width: 46rem; }
h2 { font-size: 1.1rem; margin: 1.8rem 0 .3rem; }
.filters { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin-bottom: 1rem; }
.filters .wide { grid-column: 1 / -1; }
@media (max-width: 34rem) { .filters { grid-template-columns: 1fr; } .card { padding: 1.1rem; } }
.filters input[type=search], .filters select, textarea { width: 100%; padding: .55rem .7rem; font: inherit; color: inherit;
  background: transparent; border: 1px solid var(--line); border-radius: 9px; }
.filters select option { color: var(--ink); background: #101634; }
textarea { min-height: 7rem; resize: vertical; }
.filters button { margin: 0; }
.row { display: flex; align-items: center; justify-content: space-between; gap: .75rem; flex-wrap: wrap; }
.memo { display: flex; gap: .7rem; padding: .8rem 0; border-top: 1px solid var(--line); }
.memo input { margin: .3rem 0 0; flex: none; width: 1.05rem; height: 1.05rem; }
.memo label { font-weight: 400; margin: 0; overflow-wrap: anywhere; white-space: pre-wrap; }
.memo > div { min-width: 0; flex: 1; }
.meta { color: var(--soft); font-size: .86rem; margin: .2rem 0 0; overflow-wrap: anywhere; }
button.quiet { width: auto; margin: 0; padding: .4rem .8rem; font-weight: 500; color: var(--accent);
  background: transparent; border: 1px solid var(--line); }
button.bad, button.bad-quiet { width: auto; margin: 0; padding: .55rem 1rem; }
button.bad { color: #240a08; background: var(--bad); }
button.bad-quiet { font-weight: 500; color: var(--bad); background: transparent; border: 1px solid var(--line); }
.actions { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1rem; align-items: center; }
.ok { font-weight: 600; margin: 0 0 1rem; }
"""


class _Refused(Exception):
    """What was typed cannot be a memory; raised inside a sealed use, so nothing is written."""


def answer_with_link(issuer: str, memory_ids: Sequence[str], known: Sequence[str], issue) -> str:
    """What the agent tells the person; the link only says which memories, and opens nothing by itself."""
    wanted = list(dict.fromkeys(memory_ids or []))
    if not wanted:
        return (f"Give the user this link and ask them to open it: {issuer}/memory\n"
                "They sign in there with their password and can see, search, correct and delete their memories.")
    if len(wanted) > MAX_IDS:
        return f"Not done: at most {MAX_IDS} memories in one link."
    kept = set(known)
    unknown = [memory_id for memory_id in wanted if memory_id not in kept]
    if unknown:
        return f"Not done: unknown ids: {', '.join(unknown)}. Find them with list_memories or recall."
    token = issue({"ids": wanted})
    return (f"Give the user this link and ask them to open it: {issuer}/memory?r={token}\n"
            f"They sign in there with their password and confirm deleting these {len(wanted)} memories themselves. It "
            f"lasts {LINK_SECONDS // 60} minutes. Nothing is deleted until they confirm, so do not say it is done: "
            "check with recall afterwards.")


def _counted(t: dict, key: str, n: int) -> str:
    return t[f"{key}_one"] if n == 1 else t[key].format(n=n)


def _plain(text: str) -> str:
    """Without capitals or accents, so "camion" finds "camión"."""
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold()) if not unicodedata.combining(c))


def _matches(record: MemoryRecord, words: Sequence[str]) -> bool:
    where = _plain(" ".join(str(part) for part in (record.content, record.metadata.get("project") or "",
                                                   record.metadata.get("source") or "",
                                                   record.metadata.get("author") or "")))
    return all(word in where for word in words)


def _kind_of(record: MemoryRecord) -> str:
    kind = record.metadata.get("kind")
    return kind if kind in KINDS else NO_KIND


def _found(memory: AgentMemory, query: str, project: str, kind: str) -> dict:
    """Newest first, since what someone comes to fix is usually what was just said. The neurons count what the search
    and the client leave, before choosing a kind, so choosing one does not empty the others."""
    records = memory._active()
    projects = sorted({str(r.metadata["project"]) for r in records if r.metadata.get("project")}, key=str.casefold)
    words = _plain(query).split()
    matching = [r for r in reversed(memory._active(project or None)) if _matches(r, words)]
    kinds: Dict[str, int] = {}
    for record in matching:
        kinds[_kind_of(record)] = kinds.get(_kind_of(record), 0) + 1
    shown = [r for r in matching if _kind_of(r) == kind] if kind in NEURONS else matching
    return {"records": shown, "total": len(records), "projects": projects, "kinds": kinds}


def _when(record: MemoryRecord) -> datetime:
    """When it was said, always with a time zone; an imported email keeps its own date."""
    said = record.metadata.get("said_at")
    try:
        moment = datetime.fromisoformat(str(said)) if said else record.created_at
    except ValueError:
        moment = record.created_at
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _ago(t: dict, moment: datetime, now: datetime) -> str:
    hours = int((now - moment).total_seconds() // 3600)
    if hours < 1:
        return t["now"]
    if hours < 24:
        return t["hours_ago"].format(n=hours)
    return t["yesterday"] if hours < 48 else t["days_ago"].format(n=hours // 24)


def _home(t: dict, memory: AgentMemory, now: Optional[datetime] = None) -> dict:
    """What the vault home shows, all of it from this person's memory."""
    now = now or datetime.now(timezone.utc)
    records = memory._active()
    counts: Dict[str, int] = {}
    for record in records:
        counts[_kind_of(record)] = counts.get(_kind_of(record), 0) + 1
    recent = []
    for record in sorted(records, key=_when, reverse=True)[:3]:
        author = str(record.metadata.get("author") or "")
        recent.append({"id": record.id, "text": record.content, "kind": _kind_of(record),
                       "label": t["kinds"][_kind_of(record)], "ago": _ago(t, _when(record), now),
                       "photo": record.metadata.get("photo"),
                       "chip": str(record.metadata.get("project") or ("" if author == BY_THE_USER else author))})
    projects = {str(r.metadata["project"]).strip().casefold() for r in records if str(r.metadata.get("project") or "").strip()}
    return {"total": len(records), "counts": counts, "recent": recent, "projects": len(projects)}


def _projects(memory: AgentMemory, now: Optional[datetime] = None) -> List[dict]:
    """Each client with its last activity, its counts and what is still open, most recently active first."""
    now = now or datetime.now(timezone.utc)
    groups: Dict[str, List[MemoryRecord]] = {}
    for record in memory._active():
        name = str(record.metadata.get("project") or "").strip()
        if name:
            groups.setdefault(name.casefold(), []).append(record)
    found = []
    for records in sorted(groups.values(), key=lambda rs: max(_when(r) for r in rs), reverse=True):
        last = max(_when(record) for record in records)
        kinds: Dict[str, int] = {}
        for record in records:
            kinds[_kind_of(record)] = kinds.get(_kind_of(record), 0) + 1
        quiet = max((now - last).days, 0)
        still_open = [r.content for r in records if r.metadata.get("kind") in OPEN_KINDS][-OVERVIEW_OPEN_ITEMS:]
        found.append({"name": str(records[-1].metadata["project"]), "last": _day(max(records, key=_when)), "quiet": quiet,
                      "stale": quiet >= STALE_DAYS, "kinds": kinds, "open": still_open})
    return found


def _chosen(memory: AgentMemory, memory_ids: Sequence[str]) -> List[MemoryRecord]:
    wanted = set(memory_ids)
    return [record for record in reversed(memory._active()) if record.id in wanted]


def _delete(memory: AgentMemory, memory_ids: Sequence[str]) -> int:
    return memory.store.purge([record.id for record in _chosen(memory, memory_ids)])


def _correct(memory: AgentMemory, memory_id: str, text: str) -> str:
    """The new words take the place of the old ones; if they are refused nothing changes, since _Refused skips the
    write. The old memory goes first: deleting it afterwards would take the new one along, as a later value of it."""
    record = next((r for r in memory._active() if r.id == memory_id), None)
    if record is None:
        raise KeyError(memory_id)
    about = {name: record.metadata.get(name) for name in ("kind", "source", "project", "photo")}
    if text == record.content:
        return "unchanged"
    memory.store.purge([memory_id])
    said = record.metadata.get("said_at")
    answer = memory.remember(text, **about, date=str(said) if said else None, author=BY_THE_USER) \
        if any(about.values()) else memory.remember(text, author=BY_THE_USER)
    if not answer.startswith("Remembered"):
        raise _Refused()
    return "corrected"


def _meta(t: dict, record: MemoryRecord) -> str:
    e, about = html.escape, [_day(record)]
    if record.metadata.get("kind") in t["kinds"]:
        about.append(t["kinds"][record.metadata["kind"]])
    if record.metadata.get("project"):
        about.append(e(str(record.metadata["project"])))
    if record.metadata.get("source"):
        about.append(f'{t["source"]}: {e(str(record.metadata["source"]))}')
    author = record.metadata.get("author")
    if author:
        about.append(t["by_you"] if author == BY_THE_USER else f'{t["by"]} {e(str(author))}')
    return " · ".join(about)


def _item(t: dict, record: MemoryRecord, *, checked: bool = False, correct: bool = True) -> str:
    e = html.escape
    fix = f' · <a href="/memory?edit={e(record.id)}">{t["correct"]}</a>' if correct else ""
    photo = record.metadata.get("photo")
    thumb = (f'<a class="thumb" href="/memory?{urlencode({"photo": record.id})}">{remote_portal.photo_image(photo)}</a>'
             if photo else "")
    return (f'<div class="memo"><input type="checkbox" name="id" value="{e(record.id)}" id="m-{e(record.id)}"'
            f'{" checked" if checked else ""}>{thumb}<div><label for="m-{e(record.id)}">{e(record.content)}</label>'
            f'<p class="meta">{_meta(t, record)}{fix}</p></div></div>')


def _hidden(**fields: str) -> str:
    return "".join(f'<input type="hidden" name="{name}" value="{html.escape(value)}">' for name, value in fields.items())


def sign_in_body(t: dict, token: str, *, name: str = "", error: str = "") -> str:
    e = html.escape
    asked = f'<p class="note">{t["asked"]}</p>' if token else ""
    return (f'{asked}<p class="note">{t["sign_in_note"]}</p><form method="post" action="/memory">'
            + _hidden(action="signin", r=token)
            + f'<label for="name">{t["name"]}</label><input id="name" name="name" type="text" value="{e(name)}" '
              'autocomplete="username" autocapitalize="none" spellcheck="false" required>'
              f'<label for="password">{t["password"]}</label><input id="password" name="password" type="password" '
              'autocomplete="current-password" required>'
            + (f'<p class="error" role="alert">{e(error)}</p>' if error else "")
            + f'<button type="submit">{t["go_in"]}</button></form>')


def photo_form(t: dict, csrf: str) -> str:
    return (f'<details class="add"><summary>{remote_portal.icon("image")}{t["photo_add"]}</summary>'
            f'<form method="post" action="/memory" enctype="multipart/form-data">{_hidden(csrf=csrf)}'
            f'<label for="photo">{t["photo_file"]}</label><input id="photo" name="photo" type="file" required '
            f'accept="{",".join(remote_photos.TYPES.values())}">'
            f'<label for="about">{t["photo_about"]}</label><input id="about" name="about" type="text" required '
            f'maxlength="2000" placeholder="{t["photo_about_hint"]}">'
            f'<label for="client">{t["photo_project"]}</label><input id="client" name="project" type="text" maxlength="200">'
            f'<p class="hint">{t["photo_note"]}</p><button type="submit">{t["photo_go"]}</button></form></details>')


def photo_body(t: dict, record: MemoryRecord) -> str:
    return (f'<figure class="large">{remote_portal.photo_image(record.metadata["photo"], record.content)}</figure>'
            f'<p>{html.escape(record.content)}</p><p class="meta">{_meta(t, record)} · '
            f'<a href="/memory?edit={html.escape(record.id)}">{t["correct"]}</a></p>'
            f'<p class="actions"><a href="/memory?view=memories">{t["back"]}</a></p>')


def list_body(t: dict, found: dict, csrf: str, *, query: str, project: str, kind: str, page: int, message: str = "",
              error: str = "", can_add_photo: bool = False) -> str:
    e = html.escape
    records, total = found["records"], found["total"]
    filtering = bool(query or project or kind)
    pages = max(1, (len(records) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(max(1, page), pages)
    shown = records[(page - 1) * PAGE_SIZE:page * PAGE_SIZE]
    count = t["count_of"].format(n=len(records), total=total) if filtering else _counted(t, "count", total)

    def option(value: str, label: str, chosen: str) -> str:
        return f'<option value="{e(value)}"{" selected" if value == chosen else ""}>{e(label)}</option>'

    filters = (f'<form method="get" action="/memory" class="filters"><input class="wide" type="search" name="q" '
               f'value="{e(query)}" placeholder="{t["search_hint"]}" aria-label="{t["search_hint"]}">'
               f'<select name="project" aria-label="{t["all_projects"]}">{option("", t["all_projects"], project)}'
               + "".join(option(p, p, project) for p in found["projects"])
               + f'</select><select name="kind" aria-label="{t["all_kinds"]}">{option("", t["all_kinds"], kind)}'
               + "".join(option(k, t["kinds"][k], kind) for k in NEURONS)
               + f'</select><button class="wide" type="submit">{t["search"]}</button></form>')

    def link_to(number: int, label: str) -> str:
        wanted = {name: value for name, value in (("q", query), ("project", project), ("kind", kind)) if value}
        return f'<a href="/memory?{e(urlencode({**wanted, "p": number}))}">{label}</a>'

    moves = []
    if page > 1:
        moves.append(link_to(page - 1, t["newer"]))
    if page < pages:
        moves.append(link_to(page + 1, t["older"]))
    filter_fields = _hidden(q=query, project=project, kind=kind) if filtering else ""
    delete_found = (f'<button class="bad-quiet" type="submit" name="action" value="review_found">'
                    f'{t["delete_found"].format(n=len(records))}</button>') if filtering and records else ""
    if shown:
        listing = (f'<form method="post" action="/memory">{_hidden(csrf=csrf)}{filter_fields}'
                   + "".join(_item(t, record) for record in shown)
                   + f'<div class="actions"><button class="bad" type="submit" name="action" value="review">'
                     f'{t["delete_marked"]}</button>{delete_found}</div></form>')
    else:
        listing = f'<p>{t["nothing_found"] if filtering else t["empty"]}</p>'
    clear = f' · <a href="/memory">{t["clear"]}</a>' if filtering else ""
    head = f'<p class="note" style="margin:0">{count}{clear}</p>'
    notice = ((f'<p class="ok" role="status">{e(message)}</p>' if message else "")
              + (f'<p class="error" role="alert">{e(error)}</p>' if error else ""))
    add = photo_form(t, csrf) if can_add_photo and not filtering else ""
    return (f'{notice}{add}{filters}{head}{listing}'
            + (f'<p class="actions">{" · ".join(moves)}</p>' if moves else ""))


def vault_body(t: dict, items: List[Dict[str, str]], csrf: str) -> str:
    e = html.escape
    rows = []
    for item in items:
        note = f'<p class="meta">{e(item["note"])}</p>' if item.get("note") else ""
        rows.append(f'<div class="memo"><div><div>{e(item["name"])}</div>{note}</div>'
                    f'<form method="post" action="/memory" class="actions" style="margin:0">'
                    f'{_hidden(action="vault", csrf=csrf, entry=item["name"])}'
                    f'<button class="quiet" type="submit" name="what" value="reveal">{t["vault_show"]}</button>'
                    f'<button class="bad-quiet" type="submit" name="what" value="delete">{t["vault_delete"]}</button>'
                    f'</form></div>')
    return f'<p class="note">{t["vault_note"]}</p>{"".join(rows)}'


def confirm_body(t: dict, records: List[MemoryRecord], csrf: str, token: str) -> str:
    return (f'<p>{_counted(t, "confirm", len(records))}</p><p class="note">{t["confirm_copies"]}</p>'
            f'<form method="post" action="/memory">{_hidden(action="delete", csrf=csrf, r=token)}'
            + "".join(_item(t, record, checked=True, correct=False) for record in records)
            + f'<div class="actions"><button class="bad" type="submit">{t["go_delete"]}</button>'
              f'<a href="/memory">{t["cancel"]}</a></div></form>')


def correct_body(t: dict, record: MemoryRecord, csrf: str, error: str = "") -> str:
    e = html.escape
    return (f'<p class="note">{_meta(t, record)}</p><form method="post" action="/memory">'
            f'{_hidden(action="correct", csrf=csrf, id=record.id)}'
            f'<label for="text">{t["text"]}</label><textarea id="text" name="text" required spellcheck="true">'
            f'{e(record.content)}</textarea><p class="hint">{t["correct_note"]}</p>'
            + (f'<p class="error" role="alert">{e(error)}</p>' if error else "")
            + f'<div class="actions"><button class="quiet" type="submit">{t["go_correct"]}</button>'
              f'<a href="/memory">{t["cancel"]}</a></div></form>')


def set_session_cookie(response, token: str, *, secure: bool) -> None:
    response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, path="/memory", secure=secure, httponly=True,
                        samesite="strict")


def routes(accounts, grants, memories, limiter, secure: bool) -> list:
    """The page; memories opens each person's sealed memory, and limiter keeps one heavy password check at a time."""
    import anyio
    from starlette.datastructures import UploadFile
    from starlette.formparsers import MultiPartParser
    from starlette.requests import ClientDisconnect
    from starlette.responses import HTMLResponse, RedirectResponse, Response
    from starlette.routing import Route

    # A photo waits in memory until it is sealed: past this the parser would put it in a plain temporary file.
    MultiPartParser.spool_max_size = remote_photos.MAX_FORM

    # Only here: the sign-in page's form has to be able to send the browser on to the app that asked.
    policy = f"{_HEADERS['Content-Security-Policy']}; img-src 'self'; form-action 'self'; base-uri 'none'"

    def texts(request) -> dict:
        base = _texts(request)
        return {**base, **TEXT[base["lang"]], **remote_portal.TEXT[base["lang"]], **remote_apps.TEXT[base["lang"]],
                **remote_tasks.TEXT[base["lang"]]}

    def view(t: dict, title: str, body: str, status: int = 200):
        response = _page(t, title, body, status, style=_STYLE)
        response.headers["Content-Security-Policy"] = policy
        return response

    async def shown(t: dict, session: dict, title: str, body: str, status: int = 200, *, active: str = "memories",
                    query: str = "", whole: bool = False):
        """A page of the portal, for someone signed in. whole is for a page that lays itself out, as the home does."""
        pending = await use(session, lambda memory: len(memory._active(kind="pending")), writes=False)
        to_approve = sum(task["state"] in remote_tasks.WAITING for task in await queue(session))
        inner = body if whole else (f'<div class="page"><h1>{html.escape(title)}</h1><div class="sheet">{body}</div></div>')
        page_html = remote_portal.shell(t, active, session["account"], session["data"]["csrf"], inner, _hidden,
                                        pending=pending, query=query, to_approve=to_approve)
        response = HTMLResponse(
            f'<!doctype html><html lang="{t["lang"]}"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width, initial-scale=1"><title>{html.escape(title)} · INFINITO</title>'
            f'<style>{_BASE_STYLE}{_STYLE}{remote_portal.STYLE}</style></head><body class="app">{page_html}</body></html>',
            status_code=status, headers=_HEADERS)
        response.headers["Content-Security-Policy"] = policy
        return response

    async def brain_image(request):
        """The only picture: nothing of anyone's in it, so it needs no sign-in and can be kept by the browser."""
        if not remote_portal.BRAIN_FILE.exists():
            return Response(status_code=404)
        return Response(remote_portal.BRAIN_FILE.read_bytes(), media_type="image/webp",
                        headers={"Cache-Control": "public, max-age=86400", "X-Content-Type-Options": "nosniff"})

    async def photo_file(request):
        """A person's own picture, to whoever holds their session and no one else: the key to open it is in the session."""
        session = signed_in(request)
        if session is None:
            return Response(status_code=401, headers=_HEADERS)
        data = await anyio.to_thread.run_sync(remote_photos.load, accounts.memory_folder(session["account"]), session["key"],
                                              request.query_params.get("id", ""))
        if data is None:
            return Response(status_code=404, headers=_HEADERS)
        return Response(data, media_type=remote_photos.TYPES[remote_photos.format_of(data[:16])],
                        headers={**_HEADERS, "X-Content-Type-Options": "nosniff"})

    def signed_in(request) -> Optional[dict]:
        token = request.cookies.get(COOKIE, "")
        found = grants.find(SESSION, token)
        if found is None or found["key"] is None or not accounts.exists(found["account"]):
            return None
        return {**found, "token": token}

    async def use(session: dict, action, *, writes: bool):
        work = functools.partial(memories.use, session["account"], session["key"], action, writes=writes)
        return await anyio.to_thread.run_sync(work)

    async def queue(session: dict, action=None):
        """The person's work queue (remote_tasks.py): read, or changed by action, sealed with their memory's key."""
        where = (accounts.memory_folder(session["account"]), session["key"], session["account"])
        if action is None:
            return await anyio.to_thread.run_sync(lambda: remote_tasks.read(*where))
        return await anyio.to_thread.run_sync(lambda: remote_tasks.change(*where, action))

    async def tasks_page(t: dict, session: dict, **notices):
        tasks = await queue(session)
        apps = [app["name"] for app in grants.apps_of(session["account"])]
        body = remote_tasks.tasks_body(t, tasks, apps, session["data"]["csrf"], _hidden, **notices)
        return await shown(t, session, t["tasks_title"], body, active="tasks")

    async def apps_page(t: dict, session: dict, message: str = ""):
        body = remote_apps.apps_body(t, grants.apps_of(session["account"]), accounts.app_levels(session["account"]),
                                     session["data"]["csrf"], _hidden, message=message)
        return await shown(t, session, t["apps_title"], body, active="apps")

    async def on_tasks(t: dict, session: dict, form, action: str):
        """What the person decides about their tasks. Only here, signed in with their password, is a task approved."""
        task_ids = [str(value) for value in form.getlist("id")][:remote_tasks.MAX_OPEN]
        if action in ("task_approve", "task_discard") and not task_ids:
            return await tasks_page(t, session, error=t["tasks_none_marked"])
        comment = str(form.get("comment", "")).strip()
        if action == "task_reopen" and not comment:
            return await tasks_page(t, session, error=t["tasks_comment_needed"])
        assign = {task_id: str(form.get(f"for:{task_id}", "")) for task_id in task_ids if f"for:{task_id}" in form}
        steps = {
            "task_approve": (lambda tasks: remote_tasks.approve(tasks, task_ids, assign), "tasks_approved"),
            "task_discard": (lambda tasks: remote_tasks.discard(tasks, task_ids), "tasks_discarded"),
            "task_cancel": (lambda tasks: remote_tasks.discard(tasks, task_ids), "tasks_cancelled"),
            "task_close": (lambda tasks: remote_tasks.close(tasks, task_ids[0] if task_ids else ""), "tasks_closed_one"),
            "task_reopen": (lambda tasks: remote_tasks.reopen(tasks, task_ids[0] if task_ids else "", comment),
                            "tasks_reopened"),
        }
        step, said = steps[action]
        try:
            done = await queue(session, step)
        except remote_tasks.Refused as refused:
            return await tasks_page(t, session, error=t["tasks_gone"] if str(refused) == "gone" else t["tasks_refused"])
        if isinstance(done, int) and not done:
            return await tasks_page(t, session, error=t["tasks_gone"])
        return await tasks_page(t, session, message=t[said].format(n=done) if isinstance(done, int) else t[said])

    def sign_in_page(t: dict, token: str = "", *, name: str = "", error: str = "", status: int = 200):
        return view(t, t["title"], sign_in_body(t, token, name=name, error=error), status)

    def vault_items(session: dict) -> Optional[List[Dict[str, str]]]:
        account = session["account"]
        if not accounts.vault_allowed(account):
            return None
        return remote_vault.ServerVault(accounts.memory_folder(account), account).listed(session["key"])

    async def listing(t: dict, session: dict, params, **notices):
        query, project = str(params.get("q", ""))[:200], str(params.get("project", ""))[:200]
        kind = str(params.get("kind", "")) if str(params.get("kind", "")) in NEURONS else ""
        try:
            page_number = int(str(params.get("p", "1")))
        except ValueError:
            page_number = 1
        found = await use(session, lambda memory: _found(memory, query, project, kind), writes=False)
        if str(params.get("added", "")):
            notices.setdefault("message", t["photo_saved"])
        body = list_body(t, found, session["data"]["csrf"], query=query, project=project, kind=kind, page=page_number,
                         can_add_photo=accounts.vault_allowed(session["account"]), **notices)
        searching = str(params.get("view", "")) == "search"
        return await shown(t, session, t["nav_search"] if searching else t["nav_memories"], body,
                           active="search" if searching else "memories", query=query)

    async def home(t: dict, session: dict):
        data = await use(session, lambda memory: _home(t, memory), writes=False)
        data["vault"] = vault_items(session)
        body = remote_portal.dashboard(t, data, session["data"]["csrf"], _hidden)
        return await shown(t, session, t["nav_vault"], body, active="vault", whole=True)

    async def projects(t: dict, session: dict):
        found = await use(session, _projects, writes=False)
        return await shown(t, session, t["projects_title"], remote_portal.projects_body(t, found), active="projects")

    async def keys(t: dict, session: dict):
        items = vault_items(session)
        body = vault_body(t, items, session["data"]["csrf"]) if items else f'<p>{t["no_keys_yet" if items is not None else "no_keys"]}</p>'
        return await shown(t, session, t["keys"], body, active="vault")

    async def confirming(t: dict, session: dict, memory_ids: Sequence[str], token: str = ""):
        records = await use(session, lambda memory: _chosen(memory, memory_ids), writes=False)
        if not records:
            if token:
                grants.drop(token)
            return await shown(t, session, t["gone_title"], f'<p>{t["gone"]}</p><p><a href="/memory">{t["back"]}</a></p>')
        return await shown(t, session, t["confirm_title"], confirm_body(t, records, session["data"]["csrf"], token))

    async def page(request):
        t, params = texts(request), request.query_params
        token = params.get("r", "")
        link = grants.find(LINK, token) if token else None
        if token and link is None:
            return view(t, t["link_expired_title"], f'<p>{t["link_expired"]}</p>', status=400)
        session = signed_in(request)
        if session is None:  # a link alone shows nothing: chat apps may open it first, and it may reach anyone
            return sign_in_page(t, token, name=link["account"] if link else "")
        if link is not None:
            if link["account"] != session["account"]:
                return view(t, t["title"], f'<p>{t["other_memory"]}</p>', status=403)
            return await confirming(t, session, link["data"]["ids"], token)
        if params.get("edit"):
            wanted = params["edit"]
            found = await use(session, lambda memory: _chosen(memory, [wanted]), writes=False)
            if not found:
                return await shown(t, session, t["missing_title"], f'<p><a href="/memory">{t["back"]}</a></p>', status=404)
            return await shown(t, session, t["correct_title"], correct_body(t, found[0], session["data"]["csrf"]))
        if params.get("photo"):
            wanted = params["photo"]
            found = await use(session, lambda memory: _chosen(memory, [wanted]), writes=False)
            if not found or not found[0].metadata.get("photo"):
                return await shown(t, session, t["missing_title"], f'<p><a href="/memory">{t["back"]}</a></p>', status=404)
            return await shown(t, session, t["photo_title"], photo_body(t, found[0]))
        wanted = params.get("view", "")
        if wanted == "projects":
            return await projects(t, session)
        if wanted == "vault":
            return await keys(t, session)
        if wanted == "tasks":
            return await tasks_page(t, session)
        if wanted == "apps":
            return await apps_page(t, session)
        if wanted == "settings":
            return await shown(t, session, t["settings_title"], remote_portal.settings_body(t, session["account"]),
                               active="settings")
        if wanted in ("memories", "search") or any(name in params for name in ("q", "project", "kind", "p")):
            return await listing(t, session, params)
        return await home(t, session)

    async def sign_in(t: dict, form):
        token, password = str(form.get("r", "")), str(form.get("password", ""))
        name = str(form.get("name", "")).strip().lower()
        if not NAME.fullmatch(name):
            return sign_in_page(t, token, name=name, error=t["bad_password"], status=400)
        if accounts.waiting(name):
            return sign_in_page(t, token, name=name, error=t["wait"].format(n=accounts.waiting(name)), status=429)
        key = await anyio.to_thread.run_sync(accounts.open, name, password, limiter=limiter)
        if key is None:
            return sign_in_page(t, token, name=name, error=t["bad_password"], status=400)
        session = grants.issue(SESSION, client_id="", seconds=SESSION_SECONDS, account=name, key=key,
                               data={"csrf": secrets.token_urlsafe(24)})
        target = f"/memory?{urlencode({'r': token})}" if token else "/memory"
        response = RedirectResponse(target, status_code=303, headers=_HEADERS)
        set_session_cookie(response, session, secure=secure)
        return response

    async def add_photo(t: dict, request):
        """The one form with a file, and anyone can send one: nothing of it is read before knowing who sends it."""
        session = signed_in(request)
        if session is None:
            return sign_in_page(t, error=t["session_over"], status=401)
        length = request.headers.get("content-length", "")
        if not length.isdigit() or int(length) > remote_photos.MAX_FORM:
            try:
                async for _ in request.stream():  # dropped to the end: a browser still sending may not show the answer
                    pass
            except ClientDisconnect:
                pass
            return await listing(t, session, {}, error=t["photo_too_big"])
        async with request.form(max_files=1, max_fields=8) as form:
            if not hmac.compare_digest(str(form.get("csrf", "")), session["data"]["csrf"]):
                return await shown(t, session, t["title"], f'<p>{t["bad_form"]}</p><p><a href="/memory">{t["back"]}</a></p>',
                                   status=403)
            if not accounts.vault_allowed(session["account"]):
                return await listing(t, session, {}, error=t["bad_form"])
            upload = form.get("photo")
            data = await upload.read() if isinstance(upload, UploadFile) else b""
            about, project = str(form.get("about", "")), str(form.get("project", "")).strip()[:200]
        folder = accounts.memory_folder(session["account"])
        refused = remote_photos.refusal(folder, data)
        if refused:
            return await listing(t, session, {}, error=t[f"photo_{refused}"])
        saved = await use(session, lambda memory: remote_photos.keep(folder, session["key"], memory, data, about,
                                                                     project=project or None, author=BY_THE_USER), writes=True)
        if not saved.startswith("Remembered"):  # what was typed may hold a secret, so it does not come back
            return await listing(t, session, {}, error=t["photo_refused"])
        return RedirectResponse("/memory?view=memories&added=1", status_code=303, headers=_HEADERS)

    async def answer(request):
        t = texts(request)
        if request.headers.get("content-type", "").lower().startswith("multipart/"):
            return await add_photo(t, request)
        form = await request.form()
        action = str(form.get("action", ""))
        if action == "signin":
            return await sign_in(t, form)
        session = signed_in(request)
        if session is None:
            return sign_in_page(t, str(form.get("r", "")), error=t["session_over"], status=401)
        if not hmac.compare_digest(str(form.get("csrf", "")), session["data"]["csrf"]):
            return await shown(t, session, t["title"], f'<p>{t["bad_form"]}</p><p><a href="/memory">{t["back"]}</a></p>',
                               status=403)
        if action == "signout":
            grants.drop(session["token"])
            response = RedirectResponse("/memory", status_code=303, headers=_HEADERS)
            response.delete_cookie(COOKIE, path="/memory")
            return response
        if action == "review":
            memory_ids = [str(value) for value in form.getlist("id")][:MAX_IDS]
            if not memory_ids:
                return await listing(t, session, form, error=t["none_marked"])
            return await confirming(t, session, memory_ids)
        if action == "review_found":
            query, project = str(form.get("q", "")), str(form.get("project", ""))
            kind = str(form.get("kind", ""))
            found = await use(session, lambda memory: _found(memory, query, project, kind), writes=False)
            return await confirming(t, session, [record.id for record in found["records"]][:MAX_IDS])
        if action == "delete":
            memory_ids = [str(value) for value in form.getlist("id")][:MAX_IDS]
            if not memory_ids:
                return await listing(t, session, {}, error=t["none_marked"])
            deleted = await use(session, lambda memory: _delete(memory, memory_ids), writes=True)
            token = str(form.get("r", ""))
            link = grants.find(LINK, token) if token else None
            if link is not None and link["account"] == session["account"]:
                grants.drop(token)  # used up only now: chat apps may fetch a link first
            chat = f'<p class="note">{t["back_chat"]}</p>' if link is not None else ""
            return await shown(t, session, t["deleted_title"], f'<p>{_counted(t, "deleted", deleted)}</p>{chat}'
                                                               f'<p><a href="/memory">{t["back"]}</a></p>')
        if action == "correct":
            memory_id, text = str(form.get("id", "")), str(form.get("text", "")).strip()
            try:
                done = await use(session, lambda memory: _correct(memory, memory_id, text), writes=True)
            except KeyError:
                return await shown(t, session, t["missing_title"], f'<p><a href="/memory">{t["back"]}</a></p>', status=404)
            except _Refused:
                found = await use(session, lambda memory: _chosen(memory, [memory_id]), writes=False)
                # What was typed may hold a secret: the form goes back with the memory as it was.
                body = correct_body(t, found[0], session["data"]["csrf"], t["refused"])
                return await shown(t, session, t["correct_title"], body, status=400)
            return await listing(t, session, {}, message=t[done])
        if action in ("task_approve", "task_discard", "task_cancel", "task_close", "task_reopen"):
            return await on_tasks(t, session, form, action)
        if action == "app_level":
            client_id, level = str(form.get("client", "")), str(form.get("level", ""))
            if client_id not in {app["client_id"] for app in grants.apps_of(session["account"])}:
                return await apps_page(t, session)
            if str(form.get("what", "")) == "disconnect":
                grants.drop_client(session["account"], client_id)
                accounts.set_app_level(session["account"], client_id, None)
                return await apps_page(t, session, message=t["apps_disconnected"])
            if level not in remote_apps.LEVELS:
                return await apps_page(t, session)
            accounts.set_app_level(session["account"], client_id, level)
            return await apps_page(t, session, message=t["apps_saved"])
        if action == "vault":
            entry, what = str(form.get("entry", "")), str(form.get("what", ""))
            items = vault_items(session) or []
            if what not in ("reveal", "delete") or entry.casefold() not in {i["name"].casefold() for i in items}:
                return await listing(t, session, {}, error=t["bad_form"])
            link = grants.issue(remote_vault.GRANT, client_id="", seconds=remote_vault.LINK_SECONDS,
                                account=session["account"], key=session["key"],
                                data={"action": what, "entry": entry, "own": True})
            return RedirectResponse(f"/vault?{urlencode({'r': link})}", status_code=303, headers=_HEADERS)
        return await shown(t, session, t["title"], f'<p>{t["bad_form"]}</p><p><a href="/memory">{t["back"]}</a></p>', status=400)

    return [Route("/memory", page, methods=["GET"]), Route("/memory", answer, methods=["POST"]),
            Route(remote_portal.BRAIN_PATH, brain_image, methods=["GET"]),
            Route(remote_portal.PHOTO_PATH, photo_file, methods=["GET"])]
