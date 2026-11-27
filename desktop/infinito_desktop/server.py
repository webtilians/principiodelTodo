"""Everything this app says to the server, in one place and from a closed list: whether it answers, signing in as a
device, renewing that sign-in and ending it. Nothing else leaves the PC in phase 1; a test fails if another module
reaches the network or this one reaches another path.

The sign-in is the one every app on this PC shares (src/infinito3/remote_client.py), with this app's name and the
device scope (src/infinito3/remote_device_api.py): the sign-in page says it is a device, and its token opens no part
of the memory, not even reading it.
"""
import socket
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from typing import Callable, Optional

from . import NAME

DEVICE_SCOPE = "infinito:device"
# In phase 1 the app sends nothing, so it asks for no capability; each later phase asks for its own (a new sign-in).
SCOPE = DEVICE_SCOPE
PATHS = ("/.well-known/oauth-authorization-server", "/register", "/authorize", "/token", "/revoke")
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

    return remote_client.sign_in(address, client_name=client_name(), scope=SCOPE, open_page=open_page,
                                 wait_seconds=wait_seconds)


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
