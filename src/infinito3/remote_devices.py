"""The phone as a sense: its motion sensors tell the memory what happened to it, with no one typing a word.

A page of the portal (/memory?view=devices), opened on a phone by its owner, listens to the accelerometer
(DeviceMotionEvent) and to the orientation (DeviceOrientationEvent) in the browser. No reading leaves the phone: the
page only counts, for each episode of movement, how many times it was shaken, for how long it moved and how it lay
once it came to rest, and sends those three facts when the phone is still again.

Where the phone is (the Geolocation API) is sent only when the person asks: with the button that saves where they
are, or while they keep the GPS on. A memory keeps coordinates rounded to about ten metres with their accuracy: no map
service is asked what is there, so the place never leaves INFINITO, and the AI that reads the memory can tell where
those coordinates are.

With the GPS on, the page sends a point every few seconds. The points are the person's route, not memories: they are
sealed with the memory's key in their own file (track.sealed), kept for TRACK_SECONDS and left out of the backups, so
the route of a day is gone the next. The page can draw the last minutes of it, with no map under it. When the GPS is
turned off, one memory says what the trip was: how far, how long, from where to where.

The server checks them and writes the words of the memory itself, so a page cannot put anything else in the memory
this way, and each episode is one memory, not a stream of readings. The memory says it came from the phone's sensors
(its author), so an AI can tell what the person said from what a device saw.

The page is the one in the portal with a script: its own, inline, allowed by a nonce for that response only, and it
can only talk back to this server (connect-src 'self'). It sends with the session and the form token of the page,
like any other form of the portal.
"""
import html
import json
import math
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")

# Who saved these memories: not the person and not an AI, but what the phone's sensors saw.
AUTHOR = "phone sensors"
POSITIONS = ("face_up", "face_down", "upright", "on_side", "tilted", "unknown")
MAX_SHAKES = 50
MAX_SECONDS = 300
# One memory per episode, and episodes cannot follow each other faster than a person can move a phone.
MIN_GAP_SECONDS = 3
MAX_PER_HOUR = 60
RECENT_SHOWN = 8
# Four decimals of a degree are about eleven metres: as close as a phone's GPS is sure of, and no closer.
DECIMALS = 4
# A fix less sure than this says nothing of where the phone was.
MAX_ACCURACY = 100_000
# The route: sealed in its own file, a day long at most, and never in a backup (remote_backup.py does not list it).
TRACK_FILE = "track.sealed"
TRACK_MAGIC = b"TRK1"
TRACK_SECONDS = 24 * 3600
TRACK_MAX_POINTS = 5000
# Five decimals are about a metre: what a drawing of a walk needs. The route is sealed and kept only a day.
TRACK_DECIMALS = 5
TRACK_WINDOWS = (300, 900, 3600)
POINT_GAP_SECONDS = 2
TRIPS_PER_HOUR = 30
# A trip shorter than this is the phone staying where it was: a GPS standing still wanders a few metres.
STAYED_METRES = 25
# Everything the page's script may ask of the server.
ACTIONS = ("device_event", "device_place", "device_track_start", "device_track_point", "device_track_end",
           "device_track_get", "device_track_clear")

TEXT = {
    "es": {
        "nav_devices": "Dispositivos", "devices_title": "Tus dispositivos",
        "devices_intro": "Abre esta página en tu teléfono y conéctalo. Cuando lo muevas, INFINITO guardará lo que ha "
                         "pasado como un recuerdo: cuántas veces lo agitaste, cuánto se movió, cómo quedó al dejarlo "
                         "quieto, dónde estaba y, si activas el GPS, por dónde ha ido. Después, tu IA podrá "
                         "contártelo sin que escribas nada.",
        "devices_connect": "Conectar este teléfono", "devices_connected": "Teléfono conectado",
        "devices_idle": "Todavía no está conectado.",
        "devices_listening": "Escuchando. Agita el teléfono y déjalo quieto.",
        "devices_moving": "Moviéndose…", "devices_saving": "Guardando lo que ha pasado…",
        "devices_saved": "Guardado en tu memoria. Escuchando de nuevo.",
        "devices_motion": "Movimiento", "devices_shakes": "Sacudidas", "devices_position": "Posición",
        "devices_location": "Ubicación", "devices_place": "Guardar dónde estoy",
        "devices_gps_on": "Activar GPS", "devices_gps_off": "Apagar GPS",
        "devices_gps_tracking": "GPS activado: tu recorrido se va guardando mientras esta página siga abierta.",
        "devices_gps_stopped": "GPS apagado.",
        "devices_gps_trip_saved": "GPS apagado. El trayecto se ha guardado en tu memoria.",
        "devices_route_title": "Tu recorrido", "devices_route": "Ver mi recorrido",
        "devices_route_window": "Tiempo del recorrido",
        "devices_route_windows": {"300": "últimos 5 minutos", "900": "últimos 15 minutos", "3600": "última hora"},
        "devices_route_label": "Tu recorrido dibujado, sin mapa de fondo",
        "devices_route_empty": "No hay recorrido en ese tiempo. Activa el GPS y muévete.",
        "devices_route_caption": "{dist} en {dur} · {n} puntos",
        "devices_route_note": "Sin mapa de fondo: tu recorrido no se envía a ningún servicio de mapas.",
        "devices_route_clear": "Borrar mi recorrido", "devices_route_cleared": "Recorrido borrado.",
        "devices_bad_window": "Ese tiempo de recorrido no existe.",
        "devices_track_source": "GPS del teléfono (página Dispositivos de INFINITO)",
        "devices_locating": "Buscando dónde estás…", "devices_place_saved": "Ubicación guardada en tu memoria.",
        "devices_no_gps": "Este navegador no da la ubicación.",
        "devices_gps_denied": "No se ha dado permiso para usar la ubicación.",
        "devices_gps_failed": "No se ha podido saber dónde estás. Comprueba que la ubicación del teléfono está activada.",
        "devices_bad_place": "Esa ubicación no se ha entendido y no se ha guardado.",
        "devices_place_source": "ubicación del teléfono (página Dispositivos de INFINITO)",
        "devices_saved_title": "Lo último que ha sentido tu teléfono",
        "devices_none": "Aún no hay nada guardado desde un teléfono.",
        "devices_privacy": "Las lecturas de movimiento no salen del teléfono: solo se envía un resumen de cada "
                           "movimiento cuando vuelve a estar quieto. La ubicación solo se envía cuando la pides: al "
                           "pulsar «Guardar dónde estoy» o mientras el GPS esté activado. Con el GPS activado, tu "
                           "recorrido se guarda cifrado solo 24 horas y fuera de las copias de seguridad; al apagarlo, "
                           "queda en tu memoria un recuerdo del trayecto (distancia, tiempo, inicio y fin), que verán "
                           "las IAs que conectes. Todo funciona mientras esta página siga abierta, con la pantalla "
                           "encendida, y tu sesión dure (15 minutos desde que entraste).",
        "devices_no_sensor": "Este navegador no da acceso al acelerómetro.",
        "devices_needs_https": "El navegador solo da los sensores en una página segura (https).",
        "devices_denied": "No se ha dado permiso para usar los sensores de movimiento.",
        "devices_no_readings": "No llega ninguna lectura del sensor. Abre esta página en tu teléfono.",
        "devices_offline": "No se ha podido enviar. Comprueba la conexión.",
        "devices_session_over": "Tu sesión ha terminado. Vuelve a entrar para seguir guardando.",
        "devices_bad_event": "Ese movimiento no se ha entendido y no se ha guardado.",
        "devices_too_soon": "Demasiados movimientos seguidos: espera unos segundos.",
        "devices_not_saved": "No se ha podido guardar en tu memoria.",
        "devices_source": "sensores de movimiento del teléfono (página Dispositivos de INFINITO)",
        "devices_positions": {"face_up": "boca arriba", "face_down": "boca abajo", "upright": "de pie",
                              "on_side": "de lado", "tilted": "inclinado", "unknown": "—"},
    },
    "en": {
        "nav_devices": "Devices", "devices_title": "Your devices",
        "devices_intro": "Open this page on your phone and connect it. When you move it, INFINITO keeps what happened as "
                         "a memory: how many times you shook it, how long it moved, how it lay once you put it down and, "
                         "where it was and, if you turn the GPS on, where it went. Later, your AI can tell you, "
                         "without you typing anything.",
        "devices_connect": "Connect this phone", "devices_connected": "Phone connected",
        "devices_idle": "Not connected yet.",
        "devices_listening": "Listening. Shake the phone and put it down.",
        "devices_moving": "Moving…", "devices_saving": "Saving what happened…",
        "devices_saved": "Saved in your memory. Listening again.",
        "devices_motion": "Motion", "devices_shakes": "Shakes", "devices_position": "Position",
        "devices_location": "Location", "devices_place": "Save where I am",
        "devices_gps_on": "Turn GPS on", "devices_gps_off": "Turn GPS off",
        "devices_gps_tracking": "GPS on: your route is kept while this page stays open.",
        "devices_gps_stopped": "GPS off.",
        "devices_gps_trip_saved": "GPS off. The trip was saved in your memory.",
        "devices_route_title": "Your route", "devices_route": "Show my route",
        "devices_route_window": "Time of the route",
        "devices_route_windows": {"300": "last 5 minutes", "900": "last 15 minutes", "3600": "last hour"},
        "devices_route_label": "Your route drawn, with no map under it",
        "devices_route_empty": "No route in that time. Turn the GPS on and move.",
        "devices_route_caption": "{dist} in {dur} · {n} points",
        "devices_route_note": "No map under it: your route is not sent to any map service.",
        "devices_route_clear": "Delete my route", "devices_route_cleared": "Route deleted.",
        "devices_bad_window": "There is no such time for a route.",
        "devices_track_source": "the phone's GPS (INFINITO Devices page)",
        "devices_locating": "Finding where you are…", "devices_place_saved": "Location saved in your memory.",
        "devices_no_gps": "This browser gives no location.",
        "devices_gps_denied": "Permission to use the location was not given.",
        "devices_gps_failed": "Where you are could not be found. Check that the phone's location is on.",
        "devices_bad_place": "That location was not understood and was not saved.",
        "devices_place_source": "the phone's location (INFINITO Devices page)",
        "devices_saved_title": "The latest your phone has felt",
        "devices_none": "Nothing has been saved from a phone yet.",
        "devices_privacy": "Motion readings never leave the phone: only a summary of each movement is sent, once it "
                           "is still again. The location is sent only when you ask: when you press «Save where I am» or "
                           "while the GPS is on. With the GPS on, your route is kept encrypted for 24 hours only and "
                           "out of the backups; when you turn it off, a memory of the trip (distance, time, start and "
                           "end) stays in your memory, and the AIs you connect will see it. All of it works while this "
                           "page stays open, with the screen on, and your session lasts (15 minutes from when you "
                           "signed in).",
        "devices_no_sensor": "This browser gives no access to the accelerometer.",
        "devices_needs_https": "Browsers only give the sensors to a secure page (https).",
        "devices_denied": "Permission to use the motion sensors was not given.",
        "devices_no_readings": "No sensor readings arrive. Open this page on your phone.",
        "devices_offline": "It could not be sent. Check the connection.",
        "devices_session_over": "Your session has ended. Sign in again to keep saving.",
        "devices_bad_event": "That movement was not understood and was not saved.",
        "devices_too_soon": "Too many movements in a row: wait a few seconds.",
        "devices_not_saved": "It could not be saved in your memory.",
        "devices_source": "the phone's motion sensors (INFINITO Devices page)",
        "devices_positions": {"face_up": "face up", "face_down": "face down", "upright": "upright",
                              "on_side": "on its side", "tilted": "tilted", "unknown": "—"},
    },
}

# What the page's script shows, in the page's language: the names it needs, nothing more.
_SCRIPT_TEXTS = ("devices_connected", "devices_listening", "devices_moving", "devices_saving", "devices_saved",
                 "devices_no_sensor", "devices_needs_https", "devices_denied", "devices_no_readings", "devices_offline",
                 "devices_session_over", "devices_positions", "devices_locating", "devices_place_saved",
                 "devices_no_gps", "devices_gps_denied", "devices_gps_failed", "devices_gps_on", "devices_gps_off",
                 "devices_gps_tracking", "devices_gps_stopped", "devices_gps_trip_saved", "devices_route_empty",
                 "devices_route_caption", "devices_route_cleared")


def read_offset(form) -> Optional[int]:
    """The phone's time zone as JavaScript gives it (minutes behind UTC), or None if it is not one."""
    try:
        offset = int(str(form.get("offset", "0")))
    except ValueError:
        return None
    return offset if -840 <= offset <= 840 else None


def read_place(form, decimals: int = DECIMALS) -> Tuple[bool, Optional[Dict[str, float]]]:
    """Where the phone was, from lat, lon and accuracy (metres): (True, None) if the form names no place, (True, place)
    rounded to decimals if it names a good one, and (False, None) if it names a wrong one."""
    raw = [str(form.get(name, "")).strip() for name in ("lat", "lon", "accuracy")]
    if not any(raw):
        return True, None
    try:
        lat, lon, accuracy = (float(value) for value in raw)
    except ValueError:
        return False, None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and 0 <= accuracy <= MAX_ACCURACY):  # NaN fails these too
        return False, None
    return True, {"lat": round(lat, decimals), "lon": round(lon, decimals), "accuracy": max(1, round(accuracy))}


def read_event(form) -> Optional[Dict[str, object]]:
    """The three facts of an episode from the page's form, and where it was if the form says so; None if any is
    missing or out of bounds."""
    try:
        shakes = int(str(form.get("shakes", "")))
        seconds = float(str(form.get("seconds", "")))
    except ValueError:
        return None
    offset, (good_place, place) = read_offset(form), read_place(form)
    position = str(form.get("position", "unknown"))
    if offset is None or not good_place or not (0 <= shakes <= MAX_SHAKES and 0 <= seconds <= MAX_SECONDS):
        return None
    if position not in POSITIONS:
        return None
    if shakes == 0 and seconds < 1:  # a nudge is not an experience
        return None
    return {"shakes": shakes, "seconds": max(1, round(seconds)), "position": position, "offset": offset,
            "place": place}


def read_where(form) -> Optional[Dict[str, object]]:
    """Where the phone is, sent by the button that saves it; None if there is no good place in the form."""
    offset, (good_place, place) = read_offset(form), read_place(form)
    if offset is None or not good_place or place is None:
        return None
    return {"place": place, "offset": offset}


def _when(lang: str, now: datetime, offset: int) -> str:
    """The time the person saw on the phone (offset is the phone's, as JavaScript gives it: minutes behind UTC). It has
    its seconds: two episodes alike in the same minute are still two memories, as the memory joins memories with the
    same words."""
    local = now - timedelta(minutes=offset)
    return f"el {local:%d/%m/%Y} a las {local:%H:%M:%S}" if lang == "es" else f"on {local:%Y-%m-%d} at {local:%H:%M:%S}"


def _at(lang: str, place: Dict[str, float]) -> str:
    """The coordinates, with how sure the phone was of them."""
    metres = int(place["accuracy"])
    sure = f"{metres} m" if metres < 1000 else f"{round(metres / 1000)} km"
    coordinates = f"{place['lat']:.{DECIMALS}f}, {place['lon']:.{DECIMALS}f}"
    if lang == "es":
        return f"en las coordenadas {coordinates} (precisión de unos {sure})"
    return f"at coordinates {coordinates} (accurate to about {sure})"


def sentence(lang: str, event: Dict[str, object], now: datetime) -> str:
    """The words of the memory of an episode, written here and never taken from the page."""
    shakes, seconds, position = int(event["shakes"]), int(event["seconds"]), str(event["position"])
    lay = TEXT[lang]["devices_positions"][position] if position != "unknown" else ""
    place = event.get("place")
    at = f" {_at(lang, place)}" if place else ""
    when = _when(lang, now, int(event["offset"]))
    if lang == "es":
        lasted = f"{seconds} segundo{'s' if seconds != 1 else ''}"
        moved = (f"se agitó {shakes} {'vez' if shakes == 1 else 'veces'} en {lasted}" if shakes
                 else f"se movió durante {lasted} sin sacudidas")
        rest = f"quedó quieto, {lay}" if lay else "quedó quieto"
        rest = f"{rest},{at}" if place and lay else f"{rest}{at}"
        return f"Mi teléfono {moved} y después {rest}, {when}."
    lasted = f"{seconds} second{'s' if seconds != 1 else ''}"
    moved = (f"was shaken {shakes} time{'s' if shakes != 1 else ''} in {lasted}" if shakes
             else f"moved for {lasted} without being shaken")
    rest = f"came to rest {lay}" if lay else "came to rest"
    return f"My phone {moved} and then {rest}{at}, {when}."


def place_sentence(lang: str, where: Dict[str, object], now: datetime) -> str:
    """The words of the memory of where the phone was, when the person asked to save it."""
    when = _when(lang, now, int(where["offset"]))
    if lang == "es":
        return f"Mi teléfono estaba {_at('es', where['place'])} {when}."
    return f"My phone was {_at('en', where['place'])} {when}."


class Throttle:
    """At most one call every min_gap seconds and per_hour an hour for each memory, kept by this process. Without them,
    it is the pace of episodes: MIN_GAP_SECONDS and MAX_PER_HOUR."""

    def __init__(self, clock=time.monotonic, min_gap: Optional[float] = None, per_hour: Optional[int] = None):
        self.clock = clock
        self.min_gap, self.per_hour = min_gap, per_hour
        self.times: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def allow(self, account: str) -> bool:
        now = self.clock()
        gap = MIN_GAP_SECONDS if self.min_gap is None else self.min_gap
        most = MAX_PER_HOUR if self.per_hour is None else self.per_hour
        with self.lock:
            kept = [moment for moment in self.times.get(account, []) if now - moment < 3600]
            if (kept and now - kept[-1] < gap) or len(kept) >= most:
                self.times[account] = kept
                return False
            self.times[account] = kept + [now]
            return True


_track_guard = threading.Lock()
_track_locks: Dict[str, threading.Lock] = {}


def _track_lock(folder: Path) -> threading.Lock:
    with _track_guard:
        return _track_locks.setdefault(str(Path(folder).resolve()), threading.Lock())


def _empty_track() -> dict:
    return {"started": None, "points": []}


def _read_track(folder: Path, key: bytes, account: str) -> dict:
    from .remote_accounts import _unseal

    path = Path(folder) / TRACK_FILE
    if not path.exists():
        return _empty_track()
    blob = path.read_bytes()
    plain = _unseal(key, blob[len(TRACK_MAGIC):], f"track:{account}") if blob.startswith(TRACK_MAGIC) else None
    # A route this key cannot open is no one's to keep: it starts again, as it would the next day.
    return json.loads(plain.decode("utf-8")) if plain is not None else _empty_track()


def _write_track(folder: Path, key: bytes, account: str, track: dict) -> None:
    from .remote_accounts import _seal

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (TRACK_FILE + ".tmp")
    temporary.write_bytes(TRACK_MAGIC + _seal(key, json.dumps(track).encode("utf-8"), f"track:{account}"))
    os.chmod(temporary, 0o600)
    temporary.replace(folder / TRACK_FILE)


def _prune(track: dict, now: float) -> None:
    """What is older than a day goes, and at most the last TRACK_MAX_POINTS stay."""
    cutoff = now - TRACK_SECONDS
    track["points"] = [point for point in track["points"] if point[0] >= cutoff][-TRACK_MAX_POINTS:]
    if track.get("started") is not None and track["started"] < cutoff:
        track["started"] = None


def track_read(folder: Path, key: bytes, account: str, now: Optional[float] = None) -> dict:
    with _track_lock(folder):
        track = _read_track(folder, key, account)
    _prune(track, time.time() if now is None else now)
    return track


def track_change(folder: Path, key: bytes, account: str, action: Callable[[dict], T], now: Optional[float] = None) -> T:
    """Runs action on the route and keeps it, one change at a time per memory."""
    with _track_lock(folder):
        track = _read_track(folder, key, account)
        _prune(track, time.time() if now is None else now)
        result = action(track)
        _write_track(folder, key, account, track)
        return result


def track_clear(folder: Path) -> None:
    with _track_lock(folder):
        path = Path(folder) / TRACK_FILE
        if path.exists():
            path.unlink()


def add_point(track: dict, place: Dict[str, float], now: float) -> None:
    track["points"].append([round(now, 1), place["lat"], place["lon"], place["accuracy"]])


def start_trip(track: dict, now: float) -> None:
    track["started"] = now


def points_since(track: dict, seconds: int, now: float) -> List[list]:
    return [point for point in track["points"] if point[0] >= now - seconds]


def metres(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """How far apart two (lat, lon) are on the ground, in metres."""
    rad = math.pi / 180
    d_lat, d_lon = (b[0] - a[0]) * rad, (b[1] - a[1]) * rad
    h = math.sin(d_lat / 2) ** 2 + math.cos(a[0] * rad) * math.cos(b[0] * rad) * math.sin(d_lon / 2) ** 2
    return 2 * 6_371_000 * math.asin(min(1.0, math.sqrt(h)))


def travelled(points: List[list]) -> float:
    """How far the route went. A step counts once it is farther than the fix is sure of (and than 5 m) from the last
    place counted, so a phone standing still does not add up the wandering of its GPS."""
    total, kept = 0.0, None
    for point in points:
        here = (point[1], point[2])
        if kept is None:
            kept = here
            continue
        step = metres(kept, here)
        if step > max(5, point[3]):
            total, kept = total + step, here
    return total


def finish_trip(track: dict, now: float) -> Optional[Dict[str, object]]:
    """The trip since the GPS was turned on, or None if it was not on or sent no point. The first point may come a
    moment before the start does, as the page sends both at once."""
    started, track["started"] = track.get("started"), None
    if started is None:
        return None
    points = [point for point in track["points"] if point[0] >= started - 10]
    if not points:
        return None
    return {"started": started, "ended": now, "first": points[0], "last": points[-1], "count": len(points),
            "metres": travelled(points)}


def _length(lang: str, metres_: float) -> str:
    if metres_ < 1000:
        return f"{round(metres_)} m"
    kilometres = f"{metres_ / 1000:.1f}"
    return f"{kilometres.replace('.', ',') if lang == 'es' else kilometres} km"


def _lasting(lang: str, seconds: float) -> str:
    seconds = max(1, round(seconds))
    if seconds < 60:
        return f"{seconds} segundo{'s' if seconds != 1 else ''}" if lang == "es" else f"{seconds} second{'s' if seconds != 1 else ''}"
    if seconds < 3600:
        minutes = round(seconds / 60)
        return f"{minutes} minuto{'s' if minutes != 1 else ''}" if lang == "es" else f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, minutes = seconds // 3600, round((seconds % 3600) / 60)
    return f"{hours} h y {minutes} min" if lang == "es" else f"{hours} h {minutes} min"


def trip_sentence(lang: str, trip: Dict[str, object], offset: int) -> str:
    """The memory of a trip with the GPS on, in the words of this server, at the times the person saw."""
    begin = datetime.fromtimestamp(float(trip["started"]), timezone.utc) - timedelta(minutes=offset)
    end = datetime.fromtimestamp(float(trip["ended"]), timezone.utc) - timedelta(minutes=offset)
    first, last = trip["first"], trip["last"]

    def place(point) -> Dict[str, float]:
        return {"lat": round(point[1], DECIMALS), "lon": round(point[2], DECIMALS), "accuracy": point[3]}

    def plain(point) -> str:
        return f"{round(point[1], DECIMALS):.{DECIMALS}f}, {round(point[2], DECIMALS):.{DECIMALS}f}"

    lasted, far = _lasting(lang, float(trip["ended"]) - float(trip["started"])), float(trip["metres"])
    if lang == "es":
        when = f"el {begin:%d/%m/%Y} entre las {begin:%H:%M:%S} y las {end:%H:%M:%S}"
        if far < STAYED_METRES:
            return f"Con el GPS activado, mi teléfono se quedó {_at('es', place(last))} durante {lasted}, {when}."
        return (f"Con el GPS activado, mi teléfono recorrió {_length('es', far)} en {lasted}, desde las coordenadas "
                f"{plain(first)} hasta las coordenadas {plain(last)}, {when}.")
    when = f"on {begin:%Y-%m-%d} between {begin:%H:%M:%S} and {end:%H:%M:%S}"
    if far < STAYED_METRES:
        return f"With the GPS on, my phone stayed {_at('en', place(last))} for {lasted}, {when}."
    return (f"With the GPS on, my phone travelled {_length('en', far)} in {lasted}, from coordinates {plain(first)} to "
            f"coordinates {plain(last)}, {when}.")


def recent(memory, limit: int = RECENT_SHOWN) -> List[str]:
    """The words of the latest memories the phone's sensors saved, newest first."""
    found = [record for record in memory._active() if record.metadata.get("author") == AUTHOR]
    found.sort(key=lambda record: record.created_at, reverse=True)
    return [record.content for record in found[:limit]]


def policy(base: str, nonce: str) -> str:
    """The portal's policy, plus this page's own script and the server it may send to."""
    return f"{base}; script-src 'nonce-{nonce}'; connect-src 'self'"


def body(t: dict, saved: List[str], csrf: str, nonce: str) -> str:
    """The page: buttons to connect, to turn the GPS on and to save where the phone is; what the sensors read now; the
    route of the last minutes, drawn; and what has been kept."""
    e = html.escape
    words = {name: t[name] for name in _SCRIPT_TEXTS}
    rows = "".join(f'<li>{e(text)}</li>' for text in saved)
    listing = f'<ul class="felt" id="dev-log">{rows}</ul>' if saved else (
        f'<ul class="felt" id="dev-log"></ul><p class="note" id="dev-none">{t["devices_none"]}</p>')
    windows = "".join(f'<option value="{seconds}">{e(t["devices_route_windows"][str(seconds)])}</option>'
                      for seconds in TRACK_WINDOWS)
    return (f'<div id="devices" data-csrf="{e(csrf)}" data-texts="{e(json.dumps(words, ensure_ascii=False))}">'
            f'<p>{t["devices_intro"]}</p>'
            f'<div class="actions"><button type="button" class="connect" id="dev-connect">{t["devices_connect"]}</button>'
            f'<button type="button" class="place" id="dev-gps" aria-pressed="false">{t["devices_gps_on"]}</button>'
            f'<button type="button" class="place" id="dev-place">{t["devices_place"]}</button></div>'
            f'<p class="note" id="dev-status" role="status" aria-live="polite">{t["devices_idle"]}</p>'
            f'<dl class="senses"><div><dt>{t["devices_motion"]}</dt><dd id="dev-motion">—</dd></div>'
            f'<div><dt>{t["devices_shakes"]}</dt><dd id="dev-shakes">0</dd></div>'
            f'<div><dt>{t["devices_position"]}</dt><dd id="dev-position">—</dd></div>'
            f'<div class="wide"><dt>{t["devices_location"]}</dt><dd id="dev-location">—</dd></div></dl>'
            f'<h2>{t["devices_route_title"]}</h2>'
            f'<div class="trail"><button type="button" class="place" id="dev-route">{t["devices_route"]}</button>'
            f'<select id="dev-window" aria-label="{e(t["devices_route_window"])}">{windows}</select>'
            f'<button type="button" class="bad-quiet" id="dev-route-clear">{t["devices_route_clear"]}</button></div>'
            f'<div class="route" id="dev-route-box" hidden><svg id="dev-route-svg" viewBox="0 0 320 220" role="img" '
            f'aria-label="{e(t["devices_route_label"])}"></svg><p class="note" id="dev-route-caption"></p>'
            f'<p class="note">{t["devices_route_note"]}</p></div>'
            f'<h2>{t["devices_saved_title"]}</h2>{listing}'
            f'<p class="note">{t["devices_privacy"]}</p></div>'
            f'<script nonce="{e(nonce)}">{SCRIPT}</script>')


STYLE = """
button.connect { width: auto; margin: 0; padding: .7rem 1.3rem; }
button.connect:disabled { filter: grayscale(.6); cursor: default; }
button.place { width: auto; margin: 0; padding: .7rem 1.1rem; font-weight: 500; color: var(--accent);
  background: transparent; border: 1px solid var(--line); }
button.place[aria-pressed=true] { color: #071022; font-weight: 600; background: linear-gradient(90deg, #34d399, var(--accent));
  border-color: transparent; }
.senses { display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin: 1.2rem 0 0; }
.senses div { padding: .7rem .8rem; border: 1px solid var(--line); border-radius: 12px; }
.senses dt { color: var(--soft); font-size: .8rem; }
.senses dd { margin: .2rem 0 0; font-size: 1.35rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.senses .wide { grid-column: 1 / -1; }
.senses .wide dd { font-size: 1.05rem; overflow-wrap: anywhere; }
@media (max-width: 30rem) { .senses { grid-template-columns: 1fr 1fr; } .senses div:nth-child(3) { grid-column: 1 / -1; } }
.trail { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin-top: .6rem; }
.trail select { width: auto; padding: .62rem .7rem; font: inherit; color: inherit; background: transparent;
  border: 1px solid var(--line); border-radius: 9px; }
.trail select option { color: var(--ink); background: #101634; }
.route { margin: 1rem 0 0; padding: .6rem; border: 1px solid var(--line); border-radius: 12px;
  background: rgba(5, 10, 30, .55); }
.route svg { display: block; width: 100%; height: auto; }
.route .grid { stroke: rgba(130, 160, 255, .13); stroke-width: 1; }
.route .path { fill: none; stroke: var(--accent); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
.route .sure { fill: rgba(167, 139, 250, .14); stroke: rgba(167, 139, 250, .45); stroke-width: 1; }
.route .start { fill: #34d399; }
.route .end { fill: var(--accent-2); stroke: #fff; stroke-width: 2; }
.route .bar { stroke: var(--soft); stroke-width: 2; }
.route .label { fill: var(--soft); font-size: 11px; }
ul.felt { list-style: none; margin: .4rem 0 0; padding: 0; }
ul.felt li { padding: .7rem 0; border-top: 1px solid var(--line); overflow-wrap: anywhere; }
ul.felt li.new { color: var(--accent); }
"""

# Counting shakes: within a burst of strong movement, each change of direction along the burst's own axis is a turn,
# and two turns are one shake (an isolated flick has one or two, so it counts as one). A burst ends after a short
# quiet; an episode ends when the phone has been still for REST_MS, and only then is it sent.
# The GPS: while it is on, a point goes at most every POINT_GAP_MS, and only if the phone moved POINT_MOVE_M or
# POINT_IDLE_MS went by; fixes less sure than POINT_ACCURACY_M are shown but not sent. Closing the page ends the trip.
SCRIPT = r"""
(function () {
  'use strict';
  var root = document.getElementById('devices');
  if (!root) { return; }
  var t = JSON.parse(root.getAttribute('data-texts'));
  var csrf = root.getAttribute('data-csrf');
  var lang = document.documentElement.lang || 'es';
  function el(id) { return document.getElementById(id); }
  var button = el('dev-connect'), statusOut = el('dev-status'), motionOut = el('dev-motion'),
      shakesOut = el('dev-shakes'), positionOut = el('dev-position'), log = el('dev-log'),
      placeButton = el('dev-place'), gpsButton = el('dev-gps'), locationOut = el('dev-location'),
      routeButton = el('dev-route'), routeWindow = el('dev-window'), routeClear = el('dev-route-clear'),
      routeBox = el('dev-route-box'), routeSvg = el('dev-route-svg'), routeCaption = el('dev-route-caption');

  var MOVE = 1.5, STILL = 0.6, HIGH = 11, LOW = 4, REST_MS = 1500, BURST_GAP_MS = 300, PEAK_GAP_MS = 80;
  var POINT_GAP_MS = 5000, POINT_MOVE_M = 3, POINT_IDLE_MS = 30000, POINT_ACCURACY_M = 100, FIX_FRESH_MS = 120000;
  var listening = false, readings = 0, gravity = null, episode = null, orientation = null, stopped = false;
  var fix = null, watching = null, lastSent = null, awake = false;

  function say(text) { statusOut.textContent = text; }

  function metres(a, b) {
    var rad = Math.PI / 180, dLat = (b.lat - a.lat) * rad, dLon = (b.lon - a.lon) * rad;
    var h = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(a.lat * rad) * Math.cos(b.lat * rad) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
    return 2 * 6371000 * Math.asin(Math.min(1, Math.sqrt(h)));
  }

  function gpsIdle() {
    if (watching !== null) { navigator.geolocation.clearWatch(watching); watching = null; }
    gpsButton.setAttribute('aria-pressed', 'false');
    gpsButton.textContent = t.devices_gps_on;
  }

  function post(action, data) {
    var form = new URLSearchParams();
    form.set('action', action);
    form.set('csrf', csrf);
    Object.keys(data).forEach(function (name) { form.set(name, String(data[name])); });
    return fetch('/memory', {method: 'POST', body: form, credentials: 'same-origin', headers: {'Accept': 'application/json'}})
      .then(function (response) {
        return response.json().then(function (answer) { return {status: response.status, answer: answer}; });
      })
      .then(function (result) {
        if (result.status === 401) { stopped = true; gpsIdle(); say(t.devices_session_over); }
        return result;
      });
  }

  function remembered(text) {
    var none = el('dev-none');
    if (none) { none.remove(); }
    var item = document.createElement('li');
    item.className = 'new';
    item.textContent = text;
    log.insertBefore(item, log.firstChild);
  }

  function send(action, data, done) {
    if (stopped) { say(t.devices_session_over); return; }
    say(t.devices_saving);
    post(action, data).then(function (result) {
      if (result.status === 200 && result.answer.saved) { remembered(result.answer.text); say(done); }
      else if (result.status !== 401) { say(result.answer.error || t.devices_offline); }
    }).catch(function () { say(t.devices_offline); });
  }

  function linear(event) {
    var a = event.acceleration;
    if (a && typeof a.x === 'number') { return [a.x, a.y || 0, a.z || 0]; }
    var g = event.accelerationIncludingGravity;
    if (!g || typeof g.x !== 'number') { return null; }
    var v = [g.x, g.y || 0, g.z || 0];
    if (!gravity) { gravity = v.slice(); }
    for (var i = 0; i < 3; i++) { gravity[i] = 0.9 * gravity[i] + 0.1 * v[i]; }
    return [v[0] - gravity[0], v[1] - gravity[1], v[2] - gravity[2]];
  }

  function burstShakes(burst) {
    return burst && burst.peaks ? Math.max(1, Math.floor((burst.turns + 1) / 2)) : 0;
  }

  function position() {
    if (!orientation || typeof orientation.beta !== 'number') { return 'unknown'; }
    var b = Math.abs(orientation.beta), g = Math.abs(orientation.gamma || 0);
    if (b < 25 && g < 25) { return 'face_up'; }
    if (b > 155 && g < 25) { return 'face_down'; }
    if (b >= 55 && b <= 125) { return 'upright'; }
    if (g > 55) { return 'on_side'; }
    return 'tilted';
  }

  function onMotion(event) {
    if (stopped) { return; }
    var v = linear(event);
    if (!v) { return; }
    readings++;
    var now = Date.now();
    var size = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    motionOut.textContent = size.toFixed(1);
    if (!episode) {
      if (size < MOVE) { return; }
      episode = {start: now, quietSince: null, shakes: 0, burst: null};
      say(t.devices_moving);
    }
    var burst = episode.burst;
    if (size > HIGH) {
      if (!burst) {
        burst = episode.burst = {axis: [v[0] / size, v[1] / size, v[2] / size], sign: 1, armed: false,
                                 turns: 0, peaks: 1, lastPeak: now, quietSince: null};
      } else {
        var along = v[0] * burst.axis[0] + v[1] * burst.axis[1] + v[2] * burst.axis[2];
        if (burst.armed && Math.abs(along) > HIGH && now - burst.lastPeak > PEAK_GAP_MS) {
          var sign = along > 0 ? 1 : -1;
          if (sign !== burst.sign) { burst.turns++; }
          burst.sign = sign; burst.peaks++; burst.armed = false; burst.lastPeak = now;
        }
      }
    }
    if (burst) {
      var across = Math.abs(v[0] * burst.axis[0] + v[1] * burst.axis[1] + v[2] * burst.axis[2]);
      if (across < LOW) { burst.armed = true; }
      if (size < LOW) {
        if (burst.quietSince === null) { burst.quietSince = now; }
        else if (now - burst.quietSince > BURST_GAP_MS) { episode.shakes += burstShakes(burst); episode.burst = null; }
      } else { burst.quietSince = null; }
    }
    shakesOut.textContent = String(episode.shakes + burstShakes(episode.burst));
    if (size < STILL) {
      if (episode.quietSince === null) { episode.quietSince = now; }
      else if (now - episode.quietSince >= REST_MS) { finish(); }
    } else { episode.quietSince = null; }
  }

  function finish() {
    var done = episode;
    episode = null;
    done.shakes += burstShakes(done.burst);
    var seconds = Math.min(300, Math.max(0, (done.quietSince - done.start) / 1000));
    if (done.shakes === 0 && seconds < 1) { say(watching !== null ? t.devices_gps_tracking : t.devices_listening); return; }
    var data = {shakes: Math.min(50, done.shakes), seconds: seconds.toFixed(1), position: position(),
                offset: new Date().getTimezoneOffset()};
    if (watching !== null && fix && Date.now() - fix.at < FIX_FRESH_MS) {
      data.lat = fix.lat.toFixed(5);
      data.lon = fix.lon.toFixed(5);
      data.accuracy = Math.round(fix.accuracy);
    }
    send('device_event', data, t.devices_saved);
  }

  function keepAwake() {
    if (awake || !navigator.wakeLock) { return; }
    awake = true;
    var ask = function () { navigator.wakeLock.request('screen').catch(function () {}); };
    ask();
    document.addEventListener('visibilitychange', function () {
      if (document.visibilityState === 'visible' && (listening || watching !== null)) { ask(); }
    });
  }

  function start() {
    if (listening) { return; }
    listening = true;
    window.addEventListener('devicemotion', onMotion);
    window.addEventListener('deviceorientation', function (event) {
      orientation = {beta: event.beta, gamma: event.gamma};
      positionOut.textContent = t.devices_positions[position()];
    });
    button.disabled = true;
    button.textContent = t.devices_connected;
    say(t.devices_listening);
    keepAwake();
    setTimeout(function () { if (!readings) { say(t.devices_no_readings); } }, 3000);
  }

  button.addEventListener('click', function () {
    if (!window.isSecureContext) { say(t.devices_needs_https); return; }
    if (typeof window.DeviceMotionEvent === 'undefined') { say(t.devices_no_sensor); return; }
    var asks = [];
    if (typeof DeviceMotionEvent.requestPermission === 'function') { asks.push(DeviceMotionEvent.requestPermission()); }
    if (typeof window.DeviceOrientationEvent !== 'undefined' &&
        typeof DeviceOrientationEvent.requestPermission === 'function') {
      asks.push(DeviceOrientationEvent.requestPermission().catch(function () { return 'denied'; }));
    }
    Promise.all(asks).then(function (answers) {
      if (answers.length && answers[0] !== 'granted') { say(t.devices_denied); return; }
      start();
    }).catch(function () { say(t.devices_denied); });
  });

  function showFix(found) {
    fix = {lat: found.coords.latitude, lon: found.coords.longitude, accuracy: found.coords.accuracy, at: Date.now()};
    locationOut.textContent = fix.lat.toFixed(5) + ', ' + fix.lon.toFixed(5) + ' ±' + Math.round(fix.accuracy) + ' m';
  }

  function gpsFailed(error) { say(error && error.code === 1 ? t.devices_gps_denied : t.devices_gps_failed); }

  placeButton.addEventListener('click', function () {
    if (!navigator.geolocation) { say(t.devices_no_gps); return; }
    say(t.devices_locating);
    navigator.geolocation.getCurrentPosition(function (found) {
      showFix(found);
      send('device_place', {lat: fix.lat.toFixed(5), lon: fix.lon.toFixed(5), accuracy: Math.round(fix.accuracy),
                            offset: new Date().getTimezoneOffset()}, t.devices_place_saved);
    }, gpsFailed, {enableHighAccuracy: true, timeout: 20000, maximumAge: 0});
  });

  function onFix(found) {
    showFix(found);
    if (stopped || watching === null || fix.accuracy > POINT_ACCURACY_M) { return; }
    var now = Date.now();
    if (lastSent && (now - lastSent.at < POINT_GAP_MS ||
                     (metres(lastSent, fix) < POINT_MOVE_M && now - lastSent.at < POINT_IDLE_MS))) { return; }
    lastSent = {lat: fix.lat, lon: fix.lon, at: now};
    post('device_track_point', {lat: fix.lat.toFixed(6), lon: fix.lon.toFixed(6), accuracy: Math.round(fix.accuracy)})
      .catch(function () {});
  }

  function gpsOn() {
    if (stopped) { say(t.devices_session_over); return; }
    if (!navigator.geolocation) { say(t.devices_no_gps); return; }
    lastSent = null;
    watching = navigator.geolocation.watchPosition(onFix, function (error) {
      if (error.code === 1) { gpsIdle(); gpsFailed(error); }
      else if (!fix) { gpsFailed(error); }
    }, {enableHighAccuracy: true, maximumAge: 5000, timeout: 30000});
    gpsButton.setAttribute('aria-pressed', 'true');
    gpsButton.textContent = t.devices_gps_off;
    say(t.devices_gps_tracking);
    keepAwake();
    post('device_track_start', {}).catch(function () { say(t.devices_offline); });
  }

  function gpsOff() {
    gpsIdle();
    if (stopped) { return; }
    say(t.devices_saving);
    post('device_track_end', {offset: new Date().getTimezoneOffset()}).then(function (result) {
      if (result.status === 200 && result.answer.saved) { remembered(result.answer.text); say(t.devices_gps_trip_saved); }
      else if (result.status !== 401) { say(result.answer.error || t.devices_gps_stopped); }
    }).catch(function () { say(t.devices_offline); });
  }

  gpsButton.addEventListener('click', function () { if (watching !== null) { gpsOff(); } else { gpsOn(); } });

  window.addEventListener('pagehide', function () {
    if (watching === null || stopped || !navigator.sendBeacon) { return; }
    var form = new URLSearchParams();
    form.set('action', 'device_track_end');
    form.set('csrf', csrf);
    form.set('offset', String(new Date().getTimezoneOffset()));
    navigator.sendBeacon('/memory', form);
  });

  var SVG = 'http://www.w3.org/2000/svg';
  function shape(name, attributes) {
    var node = document.createElementNS(SVG, name);
    Object.keys(attributes).forEach(function (key) { node.setAttribute(key, String(attributes[key])); });
    routeSvg.appendChild(node);
    return node;
  }

  function length(m) {
    return m < 1000 ? Math.round(m) + ' m' : (m / 1000).toLocaleString(lang, {maximumFractionDigits: 1}) + ' km';
  }

  function lasting(s) {
    if (s < 60) { return Math.round(s) + ' s'; }
    if (s < 3600) { return Math.round(s / 60) + ' min'; }
    return Math.floor(s / 3600) + ' h ' + Math.round((s % 3600) / 60) + ' min';
  }

  function travelled(points) {  // a phone standing still does not add up the wandering of its GPS
    var total = 0, kept = null;
    points.forEach(function (p) {
      var here = {lat: p[1], lon: p[2]};
      if (!kept) { kept = here; return; }
      var step = metres(kept, here);
      if (step > Math.max(5, p[3])) { total += step; kept = here; }
    });
    return total;
  }

  function draw(points) {
    while (routeSvg.firstChild) { routeSvg.removeChild(routeSvg.firstChild); }
    routeBox.hidden = false;
    if (!points.length) { routeCaption.textContent = t.devices_route_empty; return; }
    var W = 320, H = 220, PAD = 24, rad = Math.PI / 180;
    var lat0 = 0, lon0 = 0;
    points.forEach(function (p) { lat0 += p[1] / points.length; lon0 += p[2] / points.length; });
    var xy = points.map(function (p) { return [(p[2] - lon0) * Math.cos(lat0 * rad) * 111320, (p[1] - lat0) * 110540]; });
    var xs = xy.map(function (q) { return q[0]; }), ys = xy.map(function (q) { return q[1]; });
    var minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
    var scale = Math.min((W - 2 * PAD) / Math.max(maxX - minX, 40), (H - 2 * PAD) / Math.max(maxY - minY, 40));
    var cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    function at(q) { return [W / 2 + (q[0] - cx) * scale, H / 2 - (q[1] - cy) * scale]; }
    var bar = 5;
    [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000].forEach(function (s) {
      if (s * scale <= W * 0.3) { bar = s; }
    });
    var left = cx - W / 2 / scale, right = cx + W / 2 / scale, low = cy - H / 2 / scale, high = cy + H / 2 / scale;
    for (var gx = Math.ceil(left / bar) * bar; gx <= right; gx += bar) {
      var x = W / 2 + (gx - cx) * scale;
      shape('line', {x1: x.toFixed(1), y1: 0, x2: x.toFixed(1), y2: H, 'class': 'grid'});
    }
    for (var gy = Math.ceil(low / bar) * bar; gy <= high; gy += bar) {
      var y = H / 2 - (gy - cy) * scale;
      shape('line', {x1: 0, y1: y.toFixed(1), x2: W, y2: y.toFixed(1), 'class': 'grid'});
    }
    shape('polyline', {points: xy.map(function (q) { var p = at(q); return p[0].toFixed(1) + ',' + p[1].toFixed(1); }).join(' '),
                       'class': 'path'});
    var first = at(xy[0]), last = at(xy[xy.length - 1]);
    var sure = Math.min(points[points.length - 1][3] * scale, 60);
    if (sure > 7) { shape('circle', {cx: last[0].toFixed(1), cy: last[1].toFixed(1), r: sure.toFixed(1), 'class': 'sure'}); }
    shape('circle', {cx: first[0].toFixed(1), cy: first[1].toFixed(1), r: 5, 'class': 'start'});
    shape('circle', {cx: last[0].toFixed(1), cy: last[1].toFixed(1), r: 6.5, 'class': 'end'});
    shape('line', {x1: PAD, y1: H - 10, x2: (PAD + bar * scale).toFixed(1), y2: H - 10, 'class': 'bar'});
    shape('text', {x: PAD, y: H - 15, 'class': 'label'}).textContent = length(bar);
    shape('text', {x: W - 16, y: 18, 'class': 'label', 'text-anchor': 'middle'}).textContent = 'N ↑';
    routeCaption.textContent = t.devices_route_caption.replace('{dist}', length(travelled(points)))
      .replace('{dur}', lasting(points[points.length - 1][0] - points[0][0])).replace('{n}', String(points.length));
  }

  routeButton.addEventListener('click', function () {
    if (stopped) { say(t.devices_session_over); return; }
    post('device_track_get', {seconds: routeWindow.value}).then(function (result) {
      if (result.status === 200) { draw(result.answer.points); }
      else if (result.status !== 401) { say(result.answer.error || t.devices_offline); }
    }).catch(function () { say(t.devices_offline); });
  });

  routeClear.addEventListener('click', function () {
    if (stopped) { say(t.devices_session_over); return; }
    post('device_track_clear', {}).then(function (result) {
      if (result.status === 200) { draw([]); say(t.devices_route_cleared); }
    }).catch(function () { say(t.devices_offline); });
  });
})();
"""
