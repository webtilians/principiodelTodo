"""What each app may do with a person's memory, chosen by the person on their memory page.

An app that works on its own, such as an agent with its own computer in the cloud or a scheduled task, should be able
to do its job and no more. So each app a person has connected gets one of three levels:

- all: everything, as before. An app that connects starts here;
- add: it reads, adds memories and works the task queue (remote_tasks.py), but never forgets and never reaches the
  vault, so what was known stays, as in a log that is only added to;
- read: it only reads.

The app is the one its sign-in names (its OAuth client), never something a model says, and the level is the
person's word, kept with their account. Only the person, signed in on their page, changes it. A refused call says so
plainly, and tells the agent not to ask another app to do it instead.
"""
from typing import Dict, Optional

LEVELS = ("all", "add", "read")
DEFAULT = "all"

READS = frozenset({"recall", "list_memories", "overview", "briefing", "photo_show", "task_list", "reflection_due",
                   "reflection_context", "reflection_list", "reflection_history", "shared_memories",
                   "cognition_decisions"})
ADDS = READS | {"remember", "remember_many", "photo_save", "task_propose", "task_take", "task_done", "task_return",
                "guideline_propose", "reflection_add", "cognition_report"}

_WHAT = {"read": "only read", "add": "read and add, but not forget or use the vault"}

TEXT = {
    "es": {
        "nav_apps": "Apps", "apps_title": "Tus apps",
        "apps_note": "Las IAs que has conectado a tu memoria, y lo que cada una puede hacer. A una que trabaja sola, "
                     "dale solo lo que necesita.",
        "apps_empty": "Aún no has conectado ninguna app. Cuando conectes una, aparecerá aquí.",
        "apps_levels": {"all": "Todo", "add": "Leer y añadir", "read": "Solo leer"},
        "apps_about": {"all": "lee, guarda, olvida y usa la bóveda",
                       "add": "lee, guarda y trabaja la cola de tareas; nunca olvida ni toca la bóveda",
                       "read": "solo lee"},
        "apps_since": "conectada el {day}", "apps_save": "Guardar", "apps_saved": "Guardado.",
        "apps_disconnect": "Desconectar", "apps_disconnected": "Desconectada. Si vuelve a conectarse, empezará con «Todo».",
        "apps_same_name": "Si ves dos con el mismo nombre, son dos conexiones de esa app: cada una tiene lo suyo.",
    },
    "en": {
        "nav_apps": "Apps", "apps_title": "Your apps",
        "apps_note": "The AIs you have connected to your memory, and what each one may do. Give one that works on its "
                     "own only what it needs.",
        "apps_empty": "You have not connected any app yet. When you do, it will appear here.",
        "apps_levels": {"all": "Everything", "add": "Read and add", "read": "Read only"},
        "apps_about": {"all": "reads, saves, forgets and uses the vault",
                       "add": "reads, saves and works the task queue; never forgets or touches the vault",
                       "read": "only reads"},
        "apps_since": "connected on {day}", "apps_save": "Save", "apps_saved": "Saved.",
        "apps_disconnect": "Disconnect", "apps_disconnected": "Disconnected. If it connects again, it starts with "
                                                              "«Everything».",
        "apps_same_name": "Two with the same name are two connections of that app: each has its own.",
    },
}


def allowed(level: Optional[str], tool: str) -> bool:
    if level == "read":
        return tool in READS
    if level == "add":
        return tool in ADDS
    return True


def refusal(level: Optional[str], tool: str, issuer: str) -> Optional[str]:
    """What the agent is told when its app may not use this tool, or None when it may."""
    if allowed(level, tool):
        return None
    return (f"Not allowed: the user lets this app {_WHAT[level]} in INFINITO. If it matters, tell them: they can change "
            f"it on their memory page, under Apps ({issuer}/memory?view=apps). Do not ask another app to do it instead.")


def device_refusal() -> str:
    """What a device's sign-in is told on /mcp (remote_device_api.py), whatever its level."""
    from .remote_device_api import PATH, SCOPE

    return (f"Not allowed: this is a device's sign-in ({SCOPE}). It only sends its events to {PATH}; it never reads "
            "or changes the memory over MCP. Do not ask another app to do it instead.")


def _device_row(t: dict, app: dict, device: dict, csrf: str, hidden_fields) -> str:
    """A device: what it is, when it was last heard from, and a switch for each capability it signed in with."""
    import html
    from datetime import datetime, timezone

    from . import remote_device_api

    e = html.escape
    d = remote_device_api.TEXT[t.get("lang", "es")]
    last = device.get("last")
    heard = (d["apps_device_last"].format(when=datetime.fromtimestamp(float(last), timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")) if last else d["apps_device_never"])
    off = set(device.get("off") or [])
    switches = []
    for name in remote_device_api.capabilities(app.get("scopes")):
        known = name in remote_device_api.CAPABILITIES
        label = d["device_caps"].get(name, name) if known else f'{name} ({d["apps_device_unknown"]})'
        checked = " checked" if name not in off else ""
        switches.append(f'<label class="meta" style="display:block"><input type="checkbox" name="on" value="{e(name)}"'
                        f'{checked}> {e(label)}</label>')
    return (f'<div class="memo"><div><div><strong>{e(app["name"])}</strong></div>'
            f'<p class="meta">{e(app["host"])} · {t["apps_since"].format(day=e(app["since"]))} · {e(heard)}</p>'
            f'<p class="meta">{d["apps_device"]}</p>'
            f'<form method="post" action="/memory" class="actions">'
            f'{hidden_fields(action="device_capabilities", csrf=csrf, client=app["client_id"])}'
            f'<fieldset style="flex:1 1 16rem;border:0;margin:0;padding:0"><legend class="meta">{d["apps_device_caps"]}'
            f'</legend>{"".join(switches)}</fieldset>'
            f'<button class="quiet" type="submit">{t["apps_save"]}</button>'
            f'<button class="bad-quiet" type="submit" name="what" value="disconnect">{t["apps_disconnect"]}</button>'
            f'</form></div></div>')


def apps_body(t: dict, apps: list, levels: Dict[str, str], csrf: str, hidden_fields, message: str = "",
              devices: Optional[Dict[str, dict]] = None) -> str:
    """One row per connected app: its name and where it signs in from, its level, and a way to disconnect it. A device
    (remote_device_api.py) has no level: it shows its last activity and a switch for each of its capabilities."""
    import html

    from .remote_device_api import is_device

    e = html.escape
    notice = f'<p class="ok" role="status">{e(message)}</p>' if message else ""
    if not apps:  # the last one may have just been disconnected: the page still says so
        return f'{notice}<p>{t["apps_empty"]}</p>'
    rows = []
    for app in apps:
        if is_device(app.get("scopes")):
            rows.append(_device_row(t, app, (devices or {}).get(app["client_id"]) or {}, csrf, hidden_fields))
            continue
        level = levels.get(app["client_id"], DEFAULT)
        options = "".join(f'<option value="{name}"{" selected" if name == level else ""}>{t["apps_levels"][name]} — '
                          f'{t["apps_about"][name]}</option>' for name in LEVELS)
        rows.append(
            f'<div class="memo"><div><div><strong>{e(app["name"])}</strong></div>'
            f'<p class="meta">{e(app["host"])} · {t["apps_since"].format(day=e(app["since"]))}</p>'
            f'<form method="post" action="/memory" class="actions">{hidden_fields(action="app_level", csrf=csrf, client=app["client_id"])}'
            f'<div class="filters" style="margin:0;flex:1 1 16rem;grid-template-columns:1fr"><select name="level" '
            f'aria-label="{e(app["name"])}">{options}</select></div>'
            f'<button class="quiet" type="submit">{t["apps_save"]}</button>'
            f'<button class="bad-quiet" type="submit" name="what" value="disconnect">{t["apps_disconnect"]}</button>'
            f'</form></div></div>')
    names = [app["name"] for app in apps]
    twice = f'<p class="note">{t["apps_same_name"]}</p>' if len(set(names)) < len(names) else ""
    return f'{notice}<p class="note">{t["apps_note"]}</p>{"".join(rows)}{twice}'
