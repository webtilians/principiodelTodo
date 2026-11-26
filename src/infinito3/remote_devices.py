"""The phone as a sense: its motion sensors tell the memory what happened to it, with no one typing a word.

A page of the portal (/memory?view=devices), opened on a phone by its owner, listens to the accelerometer
(DeviceMotionEvent) and to the orientation (DeviceOrientationEvent) in the browser. No reading leaves the phone: the
page only counts, for each episode of movement, how many times it was shaken, for how long it moved and how it lay
once it came to rest, and sends those three facts when the phone is still again.

Where the phone is (the Geolocation API) is sent only when the person asks: with the button that saves where they
are, or, if they tick the box, with each episode. It goes as coordinates rounded to about ten metres with their
accuracy, and the memory keeps them as they are: no map service is asked what is there, so the place never leaves
INFINITO, and the AI that reads the memory can tell where those coordinates are.

The server checks them and writes the words of the memory itself, so a page cannot put anything else in the memory
this way, and each episode is one memory, not a stream of readings. The memory says it came from the phone's sensors
(its author), so an AI can tell what the person said from what a device saw.

The page is the one in the portal with a script: its own, inline, allowed by a nonce for that response only, and it
can only talk back to this server (connect-src 'self'). It sends with the session and the form token of the page,
like any other form of the portal.
"""
import html
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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

TEXT = {
    "es": {
        "nav_devices": "Dispositivos", "devices_title": "Tus dispositivos",
        "devices_intro": "Abre esta página en tu teléfono y conéctalo. Cuando lo muevas, INFINITO guardará lo que ha "
                         "pasado como un recuerdo: cuántas veces lo agitaste, cuánto se movió, cómo quedó al dejarlo "
                         "quieto y, si quieres, dónde estaba. Después, tu IA podrá contártelo sin que escribas nada.",
        "devices_connect": "Conectar este teléfono", "devices_connected": "Teléfono conectado",
        "devices_idle": "Todavía no está conectado.",
        "devices_listening": "Escuchando. Agita el teléfono y déjalo quieto.",
        "devices_moving": "Moviéndose…", "devices_saving": "Guardando lo que ha pasado…",
        "devices_saved": "Guardado en tu memoria. Escuchando de nuevo.",
        "devices_motion": "Movimiento", "devices_shakes": "Sacudidas", "devices_position": "Posición",
        "devices_location": "Ubicación", "devices_place": "Guardar dónde estoy",
        "devices_with_place": "Añadir también la ubicación a cada movimiento",
        "devices_locating": "Buscando dónde estás…", "devices_place_saved": "Ubicación guardada en tu memoria.",
        "devices_no_gps": "Este navegador no da la ubicación.",
        "devices_gps_denied": "No se ha dado permiso para usar la ubicación.",
        "devices_gps_failed": "No se ha podido saber dónde estás. Comprueba que la ubicación del teléfono está activada.",
        "devices_bad_place": "Esa ubicación no se ha entendido y no se ha guardado.",
        "devices_place_source": "ubicación del teléfono (página Dispositivos de INFINITO)",
        "devices_saved_title": "Lo último que ha sentido tu teléfono",
        "devices_none": "Aún no hay nada guardado desde un teléfono.",
        "devices_privacy": "Las lecturas de movimiento no salen del teléfono: solo se envía un resumen de cada "
                           "movimiento cuando vuelve a estar quieto. La ubicación solo se envía cuando la pides, con el "
                           "botón o con la casilla, y se guarda en tu memoria cifrada como cualquier otro recuerdo: la "
                           "verán las IAs que conectes. Funciona mientras esta página siga abierta y tu sesión dure "
                           "(15 minutos desde que entraste).",
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
                         "if you like, where it was. Later, your AI can tell you, without you typing anything.",
        "devices_connect": "Connect this phone", "devices_connected": "Phone connected",
        "devices_idle": "Not connected yet.",
        "devices_listening": "Listening. Shake the phone and put it down.",
        "devices_moving": "Moving…", "devices_saving": "Saving what happened…",
        "devices_saved": "Saved in your memory. Listening again.",
        "devices_motion": "Motion", "devices_shakes": "Shakes", "devices_position": "Position",
        "devices_location": "Location", "devices_place": "Save where I am",
        "devices_with_place": "Also add the location to each movement",
        "devices_locating": "Finding where you are…", "devices_place_saved": "Location saved in your memory.",
        "devices_no_gps": "This browser gives no location.",
        "devices_gps_denied": "Permission to use the location was not given.",
        "devices_gps_failed": "Where you are could not be found. Check that the phone's location is on.",
        "devices_bad_place": "That location was not understood and was not saved.",
        "devices_place_source": "the phone's location (INFINITO Devices page)",
        "devices_saved_title": "The latest your phone has felt",
        "devices_none": "Nothing has been saved from a phone yet.",
        "devices_privacy": "Motion readings never leave the phone: only a summary of each movement is sent, once it "
                           "is still again. The location is sent only when you ask, with the button or the box, and it "
                           "is kept in your encrypted memory like any other memory: the AIs you connect will see it. It "
                           "works while this page stays open and your session lasts (15 minutes from when you signed in).",
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
                 "devices_no_gps", "devices_gps_denied", "devices_gps_failed")


def _offset(form) -> Optional[int]:
    """The phone's time zone as JavaScript gives it (minutes behind UTC), or None if it is not one."""
    try:
        offset = int(str(form.get("offset", "0")))
    except ValueError:
        return None
    return offset if -840 <= offset <= 840 else None


def read_place(form) -> Tuple[bool, Optional[Dict[str, float]]]:
    """Where the phone was, from lat, lon and accuracy (metres): (True, None) if the form names no place, (True, place)
    rounded to DECIMALS if it names a good one, and (False, None) if it names a wrong one."""
    raw = [str(form.get(name, "")).strip() for name in ("lat", "lon", "accuracy")]
    if not any(raw):
        return True, None
    try:
        lat, lon, accuracy = (float(value) for value in raw)
    except ValueError:
        return False, None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and 0 <= accuracy <= MAX_ACCURACY):  # NaN fails these too
        return False, None
    return True, {"lat": round(lat, DECIMALS), "lon": round(lon, DECIMALS), "accuracy": max(1, round(accuracy))}


def read_event(form) -> Optional[Dict[str, object]]:
    """The three facts of an episode from the page's form, and where it was if the form says so; None if any is
    missing or out of bounds."""
    try:
        shakes = int(str(form.get("shakes", "")))
        seconds = float(str(form.get("seconds", "")))
    except ValueError:
        return None
    offset, (good_place, place) = _offset(form), read_place(form)
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
    offset, (good_place, place) = _offset(form), read_place(form)
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
    """At most one episode every MIN_GAP_SECONDS and MAX_PER_HOUR an hour for each memory, kept by this process."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.times: Dict[str, List[float]] = {}
        self.lock = threading.Lock()

    def allow(self, account: str) -> bool:
        now = self.clock()
        with self.lock:
            kept = [moment for moment in self.times.get(account, []) if now - moment < 3600]
            if (kept and now - kept[-1] < MIN_GAP_SECONDS) or len(kept) >= MAX_PER_HOUR:
                self.times[account] = kept
                return False
            self.times[account] = kept + [now]
            return True


def recent(memory, limit: int = RECENT_SHOWN) -> List[str]:
    """The words of the latest memories the phone's sensors saved, newest first."""
    found = [record for record in memory._active() if record.metadata.get("author") == AUTHOR]
    found.sort(key=lambda record: record.created_at, reverse=True)
    return [record.content for record in found[:limit]]


def policy(base: str, nonce: str) -> str:
    """The portal's policy, plus this page's own script and the server it may send to."""
    return f"{base}; script-src 'nonce-{nonce}'; connect-src 'self'"


def body(t: dict, saved: List[str], csrf: str, nonce: str) -> str:
    """The page: a button to connect, one to save where the phone is, what the sensors read now, and what has been
    kept."""
    e = html.escape
    words = {name: t[name] for name in _SCRIPT_TEXTS}
    rows = "".join(f'<li>{e(text)}</li>' for text in saved)
    listing = f'<ul class="felt" id="dev-log">{rows}</ul>' if saved else (
        f'<ul class="felt" id="dev-log"></ul><p class="note" id="dev-none">{t["devices_none"]}</p>')
    return (f'<div id="devices" data-csrf="{e(csrf)}" data-texts="{e(json.dumps(words, ensure_ascii=False))}">'
            f'<p>{t["devices_intro"]}</p>'
            f'<div class="actions"><button type="button" class="connect" id="dev-connect">{t["devices_connect"]}</button>'
            f'<button type="button" class="place" id="dev-place">{t["devices_place"]}</button></div>'
            f'<label class="opt"><input type="checkbox" id="dev-with-place"> {t["devices_with_place"]}</label>'
            f'<p class="note" id="dev-status" role="status" aria-live="polite">{t["devices_idle"]}</p>'
            f'<dl class="senses"><div><dt>{t["devices_motion"]}</dt><dd id="dev-motion">—</dd></div>'
            f'<div><dt>{t["devices_shakes"]}</dt><dd id="dev-shakes">0</dd></div>'
            f'<div><dt>{t["devices_position"]}</dt><dd id="dev-position">—</dd></div>'
            f'<div class="wide"><dt>{t["devices_location"]}</dt><dd id="dev-location">—</dd></div></dl>'
            f'<h2>{t["devices_saved_title"]}</h2>{listing}'
            f'<p class="note">{t["devices_privacy"]}</p></div>'
            f'<script nonce="{e(nonce)}">{SCRIPT}</script>')


STYLE = """
button.connect { width: auto; margin: 0; padding: .7rem 1.3rem; }
button.connect:disabled { filter: grayscale(.6); cursor: default; }
button.place { width: auto; margin: 0; padding: .7rem 1.1rem; font-weight: 500; color: var(--accent);
  background: transparent; border: 1px solid var(--line); }
label.opt { display: flex; gap: .55rem; align-items: center; font-weight: 400; margin: .9rem 0 0; }
label.opt input { width: 1.1rem; height: 1.1rem; margin: 0; flex: none; }
.senses { display: grid; grid-template-columns: repeat(3, 1fr); gap: .6rem; margin: 1.2rem 0 0; }
.senses div { padding: .7rem .8rem; border: 1px solid var(--line); border-radius: 12px; }
.senses dt { color: var(--soft); font-size: .8rem; }
.senses dd { margin: .2rem 0 0; font-size: 1.35rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.senses .wide { grid-column: 1 / -1; }
.senses .wide dd { font-size: 1.05rem; overflow-wrap: anywhere; }
@media (max-width: 30rem) { .senses { grid-template-columns: 1fr 1fr; } .senses div:nth-child(3) { grid-column: 1 / -1; } }
ul.felt { list-style: none; margin: .4rem 0 0; padding: 0; }
ul.felt li { padding: .7rem 0; border-top: 1px solid var(--line); overflow-wrap: anywhere; }
ul.felt li.new { color: var(--accent); }
"""

# Counting shakes: within a burst of strong movement, each change of direction along the burst's own axis is a turn,
# and two turns are one shake (an isolated flick has one or two, so it counts as one). A burst ends after a short
# quiet; an episode ends when the phone has been still for REST_MS, and only then is it sent.
SCRIPT = r"""
(function () {
  'use strict';
  var root = document.getElementById('devices');
  if (!root) { return; }
  var t = JSON.parse(root.getAttribute('data-texts'));
  var csrf = root.getAttribute('data-csrf');
  function el(id) { return document.getElementById(id); }
  var button = el('dev-connect'), statusOut = el('dev-status'), motionOut = el('dev-motion'),
      shakesOut = el('dev-shakes'), positionOut = el('dev-position'), log = el('dev-log'),
      placeButton = el('dev-place'), withPlace = el('dev-with-place'), locationOut = el('dev-location');

  var MOVE = 1.5, STILL = 0.6, HIGH = 11, LOW = 4, REST_MS = 1500, BURST_GAP_MS = 300, PEAK_GAP_MS = 80;
  var listening = false, readings = 0, gravity = null, episode = null, orientation = null, stopped = false;
  var fix = null, watching = null;

  function say(text) { statusOut.textContent = text; }

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
    if (done.shakes === 0 && seconds < 1) { say(t.devices_listening); return; }
    var data = {shakes: Math.min(50, done.shakes), seconds: seconds.toFixed(1), position: position(),
                offset: new Date().getTimezoneOffset()};
    if (withPlace.checked && fix && Date.now() - fix.at < 120000) {
      data.lat = fix.lat.toFixed(5);
      data.lon = fix.lon.toFixed(5);
      data.accuracy = Math.round(fix.accuracy);
    }
    send('device_event', data, t.devices_saved);
  }

  function send(action, data, done) {
    if (stopped) { say(t.devices_session_over); return; }
    var form = new URLSearchParams();
    form.set('action', action);
    form.set('csrf', csrf);
    Object.keys(data).forEach(function (name) { form.set(name, String(data[name])); });
    say(t.devices_saving);
    fetch('/memory', {method: 'POST', body: form, credentials: 'same-origin', headers: {'Accept': 'application/json'}})
      .then(function (response) {
        return response.json().then(function (answer) { return {status: response.status, answer: answer}; });
      })
      .then(function (result) {
        if (result.status === 200 && result.answer.saved) {
          var none = el('dev-none');
          if (none) { none.remove(); }
          var item = document.createElement('li');
          item.className = 'new';
          item.textContent = result.answer.text;
          log.insertBefore(item, log.firstChild);
          say(done);
        } else if (result.status === 401) {
          stopped = true;
          say(t.devices_session_over);
        } else {
          say(result.answer.error || t.devices_offline);
        }
      })
      .catch(function () { say(t.devices_offline); });
  }

  function keepAwake() {
    if (!navigator.wakeLock) { return; }
    var ask = function () { navigator.wakeLock.request('screen').catch(function () {}); };
    ask();
    document.addEventListener('visibilitychange', function () {
      if (document.visibilityState === 'visible' && listening) { ask(); }
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

  function showFix(found) {
    fix = {lat: found.coords.latitude, lon: found.coords.longitude, accuracy: found.coords.accuracy, at: Date.now()};
    locationOut.textContent = fix.lat.toFixed(4) + ', ' + fix.lon.toFixed(4) + ' \u00b1' + Math.round(fix.accuracy) + ' m';
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

  withPlace.addEventListener('change', function () {
    if (!withPlace.checked) {
      if (watching !== null) { navigator.geolocation.clearWatch(watching); watching = null; }
      return;
    }
    if (!navigator.geolocation) { withPlace.checked = false; say(t.devices_no_gps); return; }
    watching = navigator.geolocation.watchPosition(showFix, function (error) {
      if (error.code === 1) {
        withPlace.checked = false;
        navigator.geolocation.clearWatch(watching);
        watching = null;
        gpsFailed(error);
      } else if (!fix) { gpsFailed(error); }
    }, {enableHighAccuracy: true, maximumAge: 10000, timeout: 30000});
  });

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
})();
"""
