"""INFINITO Device API v1: one way for every device to tell the memory what it saw.

A phone app, the PC app, a watch or a house sends its events to POST /api/devices/v1/events, with its own sign-in
(OAuth, as every app). The server checks each event, writes the words of the memory itself with the readers and the
sentences of the devices page (remote_devices.py), and keeps one memory per episode, at the same pace as that page.

A device asks, when it registers and when the person signs it in, for the device scope (SCOPE) and for one scope per
capability it has: CAPABILITY_PREFIX and its name, such as infinito:device:phone.motion. The sign-in page says what it
will be able to add. A token with a device scope opens only this route and never /mcp (remote.py), so it cannot read,
forget or reach the vault, whatever the app's level. Its capabilities are the ones it signed in with, and a refresh
cannot add any (the MCP SDK refuses a scope the first sign-in did not give); the person can switch each one off on their
Apps page.

Each event carries its own event_id. One already accepted is answered as it was and never saved twice, so a device can
send again whatever it is not sure arrived. The event says when it happened (observed_from, observed_to), and whether
the device measured it (observed) or the person said it (declared); the memory's own creation time is when it arrived.
Only a declared note carries free text, and the memory's filter refuses a secret in it, as from any app. An event is
data: nothing in it makes the server do anything else.

What the server keeps of each device (when it was last heard from, what the person switched off) and the ids it has
seen are sealed with the memory's key in their own file, devices.sealed, out of the backups like the route: a restore
signs every app out (remote_backup.py), so every device signs in again.
"""
import functools
import hashlib
import html
import json
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import remote_devices

SCOPE = "infinito:device"
CAPABILITY_PREFIX = SCOPE + ":"
PATH = "/api/devices/v1/events"
VERSION = 1
ORIGINS = ("observed", "declared", "inferred")
# What each capability may send: its types, and the one origin its events have. A device may declare others (pc.*, a
# watch's pulse...): until this server knows them, their events are refused, and the Apps page says so.
CAPABILITIES: Dict[str, Dict[str, object]] = {
    "phone.motion": {"types": ("episode",), "origin": "observed"},
    "phone.place": {"types": ("place",), "origin": "observed"},
    "phone.track": {"types": ("start", "point", "end"), "origin": "observed"},
    "note": {"types": ("text",), "origin": "declared"},
}
NAME = re.compile(r"[a-z][a-z0-9_.]{0,39}")
ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
REMEMBERED = re.compile(r"Remembered as (\S+?)\.")
FILE = "devices.sealed"
MAGIC = b"DEV1"
# An event id is kept this long, sealed, so a device that sends it again is answered as before. (proposal, dossier §5.4)
SEEN_SECONDS = 30 * 86400
MAX_SEEN = 20_000
MAX_BODY = 16 * 1024
# An event that happened longer ago than this is not taken: a device's queue keeps a week at most. A route point, a day,
# which is all the route keeps (remote_devices.TRACK_SECONDS).
MAX_AGE_SECONDS = 7 * 86400
# How far ahead of the server's clock a device's clock may be.
SKEW_SECONDS = 300
# A route point says the device is alive, but its last activity is written at most once a minute.
LAST_SEEN_STEP = 60
DEFAULT_DEVICE = "INFINITO device"
# The project a note may name, as the person types it in the app (INFINITO Desktop's quick capture).
MAX_PROJECT = 80

TEXT = {
    "es": {
        "asking_device": "Te lo pide: <b>{who}</b>, un dispositivo. Al entrar, podrá añadir a tu memoria {what}. "
                         "No podrá leerla, borrar nada ni abrir tu bóveda.",
        "device_nothing": "nada todavía",
        "device_and": " y ",
        "device_caps": {"phone.motion": "lo que sienta al moverse", "phone.place": "dónde está, cuando se lo pidas",
                        "phone.track": "tus trayectos con el GPS encendido",
                        "note": "las notas que le escribas o le dictes"},
        "device_what": {"phone.motion": "sensores de movimiento", "phone.place": "ubicación", "phone.track": "GPS",
                        "note": "nota"},
        "device_observed": "observado por {device} ({what})",
        "device_declared": "dicho por la persona en {device} ({what})",
        "apps_device": "Dispositivo: solo añade a tu memoria lo que mide o lo que le dices, y nunca la lee.",
        "apps_device_last": "última actividad: {when}",
        "apps_device_never": "aún no ha enviado nada",
        "apps_device_caps": "Qué puede enviar",
        "apps_device_unknown": "este servidor aún no la conoce",
        "apps_device_disconnected": "Desconectado. Si vuelve a conectarse, te pedirá otra vez permiso para lo que "
                                    "envía.",
    },
    "en": {
        "asking_device": "Asked by: <b>{who}</b>, a device. Once you sign in, it can add to your memory {what}. It "
                         "cannot read it, forget anything or open your vault.",
        "device_nothing": "nothing yet",
        "device_and": " and ",
        "device_caps": {"phone.motion": "what it feels when it moves", "phone.place": "where it is, when you ask",
                        "phone.track": "your trips with the GPS on", "note": "the notes you type or tell it"},
        "device_what": {"phone.motion": "motion sensors", "phone.place": "location", "phone.track": "GPS",
                        "note": "note"},
        "device_observed": "observed by {device} ({what})",
        "device_declared": "said by the person on {device} ({what})",
        "apps_device": "Device: it only adds to your memory what it measures or what you tell it, and never reads it.",
        "apps_device_last": "last activity: {when}",
        "apps_device_never": "has not sent anything yet",
        "apps_device_caps": "What it can send",
        "apps_device_unknown": "this server does not know it yet",
        "apps_device_disconnected": "Disconnected. If it connects again, it will ask you again for what it sends.",
    },
}


def _names(scopes) -> List[str]:
    return [scope for scope in (scopes or ()) if isinstance(scope, str)]


def is_device(scopes) -> bool:
    """Whether a sign-in is a device's: any device scope makes it one, even next to others, so asking for more never
    opens more."""
    return any(scope == SCOPE or scope.startswith(CAPABILITY_PREFIX) for scope in _names(scopes))


def capabilities(scopes) -> List[str]:
    """The capabilities a device signed in with, by name."""
    found = {scope[len(CAPABILITY_PREFIX):] for scope in _names(scopes) if scope.startswith(CAPABILITY_PREFIX)}
    return sorted(name for name in found if NAME.fullmatch(name))


def registered(client_scope: Optional[str]) -> Optional[List[str]]:
    """The device scopes an app registered with, for a sign-in that asks for no scope: without them it would get the
    whole memory's. None when it registered as a device of nothing."""
    device = [scope for scope in (client_scope or "").split() if scope == SCOPE or scope.startswith(CAPABILITY_PREFIX)]
    return device or None


def _joined(t: dict, items: List[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + t["device_and"] + items[-1]


def asking(lang: str, who: str, caps: Sequence[str]) -> str:
    """What the sign-in page says when a device asks: who it is and what it will be able to add, in HTML."""
    t = TEXT[lang]
    named = [t["device_caps"].get(name, name) for name in caps]
    return t["asking_device"].format(who=html.escape(who), what=html.escape(_joined(t, named) if named else
                                                                             t["device_nothing"]))


# The registry: one sealed file per memory, one change at a time.

_guard = threading.Lock()
_locks: Dict[str, threading.Lock] = {}


def _lock(folder: Path) -> threading.Lock:
    with _guard:
        return _locks.setdefault(str(Path(folder).resolve()), threading.Lock())


def _empty() -> dict:
    return {"devices": {}, "seen": {}}


def _read(folder: Path, key: bytes, account: str) -> dict:
    from .remote_accounts import _unseal

    path = Path(folder) / FILE
    if not path.exists():
        return _empty()
    blob = path.read_bytes()
    plain = _unseal(key, blob[len(MAGIC):], f"devices:{account}") if blob.startswith(MAGIC) else None
    if plain is None:  # a registry this key cannot open is no one's: the devices start again
        return _empty()
    data = json.loads(plain.decode("utf-8"))
    return {"devices": dict(data.get("devices") or {}), "seen": dict(data.get("seen") or {})}


def _write(folder: Path, key: bytes, account: str, data: dict) -> None:
    from .remote_accounts import _seal

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (FILE + ".tmp")
    temporary.write_bytes(MAGIC + _seal(key, json.dumps(data).encode("utf-8"), f"devices:{account}"))
    os.chmod(temporary, 0o600)
    temporary.replace(folder / FILE)


def _prune(data: dict, now: float) -> None:
    """Ids past their time go, and at most MAX_SEEN stay, the newest."""
    seen = {digest: kept for digest, kept in data["seen"].items() if kept[0] > now}
    if len(seen) > MAX_SEEN:
        seen = dict(sorted(seen.items(), key=lambda item: item[1][0])[-MAX_SEEN:])
    data["seen"] = seen


def read(folder: Path, key: bytes, account: str) -> dict:
    """The devices the memory knows (by client id: device_id, first and last activity, what is switched off)."""
    with _lock(folder):
        return _read(folder, key, account)


def switch_off(folder: Path, key: bytes, account: str, client_id: str, off: Sequence[str]) -> None:
    """What the person switched off for one device, on their Apps page: its events of those are refused."""
    with _lock(folder):
        data = _read(folder, key, account)
        entry = data["devices"].setdefault(client_id, {})
        entry["off"] = sorted({name for name in off if NAME.fullmatch(name)})
        _write(folder, key, account, data)


def forget_device(folder: Path, key: bytes, account: str, client_id: str) -> None:
    """A disconnected device goes from the registry, with the ids it sent."""
    with _lock(folder):
        data = _read(folder, key, account)
        if client_id not in data["devices"] and not any(kept[3] == client_id for kept in data["seen"].values()):
            return
        data["devices"].pop(client_id, None)
        data["seen"] = {digest: kept for digest, kept in data["seen"].items() if kept[3] != client_id}
        _write(folder, key, account, data)


def _digest(client_id: str, event_id: str) -> str:
    return hashlib.sha256(f"{client_id}\n{event_id}".encode("utf-8")).hexdigest()[:32]


# One event.

class Refused(Exception):
    def __init__(self, status: int, error: str, detail: str = "", retry: int = 0):
        super().__init__(detail or error)
        self.status, self.error, self.detail, self.retry = status, error, detail, retry


def _time(value, name: str) -> datetime:
    if not isinstance(value, str) or len(value) > 40:
        raise Refused(400, "bad_event", f"{name} must be an ISO 8601 time with its offset")
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise Refused(400, "bad_event", f"{name} must be an ISO 8601 time with its offset") from None
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise Refused(400, "bad_event", f"{name} must say its offset from UTC")
    return moment


def read_envelope(event, now: datetime) -> dict:
    """The parts every v1 event has, checked; what else it carries is ignored."""
    if not isinstance(event, dict):
        raise Refused(400, "bad_event", "the body must be one event, a JSON object")
    if type(event.get("v")) is not int or event["v"] != VERSION:
        raise Refused(400, "bad_version", f"this route takes events of version {VERSION}")
    for field in ("event_id", "device_id"):
        if not isinstance(event.get(field), str) or not ID.fullmatch(event[field]):
            raise Refused(400, "bad_event", f"{field} must be 8 to 64 letters, digits, - or _ (a UUID)")
    capability, kind, origin = event.get("capability"), event.get("type"), event.get("origin")
    if not isinstance(capability, str) or capability not in CAPABILITIES:
        raise Refused(400, "unknown_capability", f"capability must be one of {', '.join(CAPABILITIES)}")
    spec = CAPABILITIES[capability]
    if kind not in spec["types"]:
        raise Refused(400, "bad_event", f"a {capability} event's type is one of {', '.join(spec['types'])}")
    if origin not in ORIGINS:
        raise Refused(400, "bad_event", f"origin must be one of {', '.join(ORIGINS)}")
    if origin != spec["origin"]:
        raise Refused(400, "bad_origin", f"{capability} events are {spec['origin']}")
    begin, end = _time(event.get("observed_from"), "observed_from"), _time(event.get("observed_to"), "observed_to")
    if end < begin:
        raise Refused(400, "bad_time", "observed_to is before observed_from")
    if end > now + timedelta(seconds=SKEW_SECONDS):
        raise Refused(400, "bad_time", "observed_to is ahead of the server's clock")
    oldest = remote_devices.TRACK_SECONDS if capability == "phone.track" else MAX_AGE_SECONDS
    if end < now - timedelta(seconds=oldest):
        raise Refused(400, "too_old", f"a {capability} event is taken up to {oldest // 3600} hours after it happened")
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        raise Refused(400, "bad_event", "payload must be a JSON object")
    offset = -round(end.utcoffset().total_seconds() / 60)  # as the devices page has it: minutes behind UTC
    if not -840 <= offset <= 840:
        raise Refused(400, "bad_time", "observed_to has an offset no place has")
    return {"event_id": event["event_id"], "device_id": event["device_id"], "capability": capability, "type": kind,
            "origin": origin, "from": begin, "to": end, "payload": payload, "offset": offset}


def _source(lang: str, event: dict, device: str) -> str:
    t = TEXT[lang]
    template = t["device_declared"] if event["origin"] == "declared" else t["device_observed"]
    return template.format(device=device, what=t["device_what"][event["capability"]])


def _saved(said: str) -> str:
    """The id of the memory just kept, or a refusal: the memory says why it did not keep it."""
    found = REMEMBERED.match(said)
    if not found:
        raise Refused(400, "not_saved", said)
    return found.group(1)


def _act(event: dict, *, folder: Path, key: bytes, account: str, device: str, lang: str, pace, use: Callable,
         now: float) -> Tuple[dict, bool]:
    """What the event does, and whether its id is kept: a route point is not, the route itself knows it."""
    capability, kind, payload = event["capability"], event["type"], event["payload"]
    begin, end = event["from"], event["to"]
    said_at = end.isoformat()
    project: Optional[str] = None

    def remember(text: str, kind_: Optional[str]) -> Callable:
        """Kept as the device's: its name is the author, and the source and the origin say how it knew."""
        return lambda memory: memory.remember(text, kind_, _source(lang, event, device), project, said_at, device,
                                              origin=event["origin"])

    if capability == "phone.track":
        where = (folder, key, account)
        if kind == "point":
            good, place = remote_devices.read_place({name: payload.get(name, "") for name in ("lat", "lon", "accuracy")},
                                                    decimals=remote_devices.TRACK_DECIMALS)
            if not good or place is None:
                raise Refused(400, "bad_payload", "a point has lat, lon and accuracy (metres)")
            if not pace.point.allow(account):
                raise Refused(429, "too_fast", "route points come at most one every few seconds",
                              remote_devices.POINT_GAP_SECONDS)
            # Never ahead of the server's clock: the devices page adds its points at the server's time.
            at = min(end.timestamp(), now)
            added = remote_devices.track_change(*where, lambda track: remote_devices.add_point_at(track, place, at),
                                                now=now)
            return ({"saved": True} if added else {"saved": True, "duplicate": True}), False
        if kind == "start":
            remote_devices.track_change(*where, lambda track: remote_devices.start_trip(track, begin.timestamp()),
                                        now=now)
            return {"saved": True}, True
        if not pace.trip.allow(account):  # end
            raise Refused(429, "too_fast", "too many trips in an hour", 1)
        trip = remote_devices.track_change(*where, lambda track: remote_devices.finish_trip(track, end.timestamp()),
                                           now=now)
        if trip is None:
            return {"saved": False, "reason": "no_trip"}, True
        text = remote_devices.trip_sentence(lang, trip, event["offset"])
        memory_id = _saved(use(remember(text, "fact")))
        return {"saved": True, "text": text, "memory_id": memory_id}, True
    if capability == "note":
        from .mcp_server import KINDS, MAX_STATEMENT

        text, kind_ = payload.get("text"), payload.get("kind")
        if not isinstance(text, str) or not text.strip() or len(text.strip()) > MAX_STATEMENT:
            raise Refused(400, "bad_payload", f"a note has text, 1 to {MAX_STATEMENT} characters")
        if kind_ is not None and kind_ not in KINDS:
            raise Refused(400, "bad_payload", f"a note's kind is one of {', '.join(KINDS)}")
        named = payload.get("project")
        if named is not None and (not isinstance(named, str) or not named.strip() or len(named.strip()) > MAX_PROJECT
                                  or not named.strip().isprintable()):
            raise Refused(400, "bad_payload", f"a note's project is a name, 1 to {MAX_PROJECT} characters")
        project = named.strip() if named else None
        text, throttle, gap = text.strip(), pace.note, 1
    else:
        place = {name: payload[name] for name in ("lat", "lon", "accuracy") if name in payload}
        if capability == "phone.motion":
            found = remote_devices.read_event({"shakes": payload.get("shakes", ""),
                                               "seconds": (end - begin).total_seconds(),
                                               "position": payload.get("position", "unknown"),
                                               "offset": event["offset"], **place})
            text = found and remote_devices.sentence(lang, found, end.astimezone(timezone.utc))
        else:  # phone.place
            found = remote_devices.read_where({**place, "offset": event["offset"]})
            text = found and remote_devices.place_sentence(lang, found, end.astimezone(timezone.utc))
        if not text:
            raise Refused(400, "bad_payload", f"that {capability} event is not one the server can read")
        kind_, throttle, gap = "fact", pace.event, remote_devices.MIN_GAP_SECONDS
    if not throttle.allow(account):
        raise Refused(429, "too_fast", "too many events in a row", gap)
    memory_id = _saved(use(remember(text, kind_)))
    return {"saved": True, "text": text, "memory_id": memory_id}, True


def handle(*, folder: Path, key: bytes, account: str, client_id: str, device: str, scopes, raw: bytes, lang: str,
           pace, use: Callable, now: Optional[float] = None) -> Tuple[int, dict, int]:
    """One event, from a device signed in to this memory: the status, the answer and, for 429, the seconds to wait."""
    moment = time.time() if now is None else now
    try:
        try:
            event = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise Refused(400, "bad_event", "the body must be one event, a JSON object") from None
        event = read_envelope(event, datetime.fromtimestamp(moment, timezone.utc))
        if event["capability"] not in capabilities(scopes):
            raise Refused(403, "not_declared", "this device did not sign in with this capability")
        with _lock(folder):
            data = _read(folder, key, account)
            _prune(data, moment)
            digest = _digest(client_id, event["event_id"])
            if digest in data["seen"]:
                _, memory_id, saved, _ = data["seen"][digest]
                return 200, {"saved": saved, "duplicate": True, "memory_id": memory_id}, 0
            entry = data["devices"].get(client_id) or {}
            if event["capability"] in entry.get("off", []):
                raise Refused(403, "switched_off", "the person switched this capability off on their Apps page")
            answer, kept = _act(event, folder=folder, key=key, account=account, device=device, lang=lang, pace=pace,
                                use=use, now=moment)
            fresh = client_id not in data["devices"] or entry.get("device_id") != event["device_id"]
            if kept or fresh or moment - float(entry.get("last") or 0) >= LAST_SEEN_STEP:
                data["devices"][client_id] = {**entry, "device_id": event["device_id"],
                                              "first": entry.get("first") or moment, "last": moment,
                                              "off": entry.get("off", [])}
                if kept:
                    data["seen"][digest] = [moment + SEEN_SECONDS, answer.get("memory_id"), answer["saved"], client_id]
                _write(folder, key, account, data)
            return 200, answer, 0
    except Refused as refused:
        return refused.status, {"error": refused.error, "detail": refused.detail}, refused.retry


def routes(accounts, grants, memories, pace) -> list:
    """The route devices send to; memories opens each person's sealed memory (remote.py's SealedMemories)."""
    import anyio
    from starlette.requests import ClientDisconnect
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from .remote_accounts import _app_name, _texts

    def answer(data: dict, status: int = 200, headers: Optional[dict] = None):
        return JSONResponse(data, status_code=status, headers={"Cache-Control": "no-store", **(headers or {})})

    async def events(request):
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        found = grants.find("access", token) if token else None
        if not found or found["key"] is None or not accounts.exists(found["account"]):
            # The device refreshes its sign-in when it reads this, as an app does on /mcp.
            return answer({"error": "invalid_token"}, 401, {"WWW-Authenticate": 'Bearer error="invalid_token"'})
        scopes = (found["data"] or {}).get("scopes") or []
        if not is_device(scopes):
            return answer({"error": "insufficient_scope", "detail": f"only a device's sign-in ({SCOPE}) sends events"},
                          403, {"WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{SCOPE}"'})
        raw = bytearray()
        try:
            async for chunk in request.stream():
                raw += chunk
                if len(raw) > MAX_BODY:
                    return answer({"error": "too_large", "detail": f"one event is at most {MAX_BODY} bytes"}, 413)
        except ClientDisconnect:
            return answer({"error": "bad_event"}, 400)
        account, key = found["account"], found["key"]
        work = functools.partial(
            handle, folder=accounts.memory_folder(account), key=key, account=account, client_id=found["client_id"],
            device=_app_name(grants.client(found["client_id"])) or DEFAULT_DEVICE, scopes=scopes, raw=bytes(raw),
            lang=_texts(request)["lang"], pace=pace,
            use=lambda action: memories.use(account, key, action, writes=True))
        status, body, retry = await anyio.to_thread.run_sync(work)
        return answer(body, status, {"Retry-After": str(retry)} if status == 429 and retry else None)

    return [Route(PATH, events, methods=["POST"])]
