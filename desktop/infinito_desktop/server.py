"""Everything this app says to the server, in one place and from a closed list: whether it answers, signing in as a
device, renewing that sign-in and ending it, and since phase 2 the notes of the quick capture, as Device API events.
Nothing else leaves the PC; a test fails if another module reaches the network or this one reaches another path.

The sign-in is the one every app on this PC shares (src/infinito3/remote_client.py), with this app's name and the
device scope (src/infinito3/remote_device_api.py): the sign-in page says it is a device and what it may add, and its
token opens no part of the memory, not even reading it.
"""
import json
import socket
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Callable, Optional, Tuple

from . import NAME

DEVICE_SCOPE = "infinito:device"
NOTE = "note"
# What the app may send, asked for when it signs in: since phase 2, the notes of the quick capture. A capability added
# later needs a new sign-in, which the sign-in page shows the person.
SCOPE = f"{DEVICE_SCOPE} {DEVICE_SCOPE}:{NOTE}"
EVENTS = "/api/devices/v1/events"
PATHS = ("/.well-known/oauth-authorization-server", "/register", "/authorize", "/token", "/revoke", EVENTS)
TIMEOUT = 10


def client_name() -> str:
    return f"{NAME} en {socket.gethostname()}"


def _allowed(address: str, url: str) -> str:
    """The URL, if it is one of the server's paths this app may use; anything else is refused before it is sent."""
    if not url.startswith(address.rstrip("/") + "/") or urllib.parse.urlparse(url).path not in PATHS:
        raise PermissionError(f"{NAME} does not call {urllib.parse.urlparse(url).path}")
    return url


def answers(address: str) -> Optional[int]:
    """The HTTP status of the server's OAuth metadata, which says it is up and signing people in."""
    url = _allowed(address, f"{address.rstrip('/')}/.well-known/oauth-authorization-server")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def sign_in(address: str, open_page: Callable[[str], object] = webbrowser.open, wait_seconds: float = 300) -> dict:
    from infinito3 import remote_client

    session = remote_client.sign_in(address, client_name=client_name(), scope=SCOPE, open_page=open_page,
                                    wait_seconds=wait_seconds)
    session["scope"] = SCOPE  # what it signed in for: an older session, without notes, asks to sign in again
    return session


def may_send(session: Optional[dict], capability: str = NOTE) -> bool:
    """Whether a session signed in for a capability (the server checks it again on every event)."""
    return bool(session) and f"{DEVICE_SCOPE}:{capability}" in str(session.get("scope") or "").split()


class Expired(Exception):
    """The server no longer takes this session (it ran out, or the person disconnected the app)."""


def refresh(session: dict) -> dict:
    from infinito3 import remote_client

    _allowed(session["server"], session["token_endpoint"])
    try:
        return remote_client.refresh(session, TIMEOUT)
    except urllib.error.HTTPError as error:
        if error.code in (400, 401):
            raise Expired() from None
        raise


def send(session: dict, event: dict) -> Tuple[int, dict]:
    """One event to the Device API: its status and the server's answer. A network failure raises OSError, and the
    event waits in the queue (outbox.py)."""
    url = _allowed(session["server"], f"{session['server'].rstrip('/')}{EVENTS}")
    request = urllib.request.Request(url, json.dumps(event).encode("utf-8"), {
        "Content-Type": "application/json", "Accept": "application/json", "Accept-Language": "es-ES",
        "Authorization": f"Bearer {session['access_token']}"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            body = json.loads(error.read() or b"{}")
        except ValueError:
            body = {}
        if error.code == 429:
            body.setdefault("retry_after", error.headers.get("Retry-After"))
        return error.code, body if isinstance(body, dict) else {}


def revoke(session: dict) -> None:
    """Ends the session on the server too, so a copy of it opens nothing; offline, the file going is enough."""
    url = _allowed(session["server"], f"{session['server'].rstrip('/')}/revoke")
    for token, hint in ((session.get("refresh_token"), "refresh_token"), (session.get("access_token"), "access_token")):
        if not token:
            continue
        # A public client has no secret, but the MCP SDK's revocation form requires the field: it goes empty.
        body = urllib.parse.urlencode({"token": token, "token_type_hint": hint, "client_id": session["client_id"],
                                       "client_secret": ""})
        try:
            urllib.request.urlopen(urllib.request.Request(url, body.encode(), {
                "Content-Type": "application/x-www-form-urlencoded"}), timeout=TIMEOUT).close()
        except (OSError, urllib.error.URLError):
            pass
