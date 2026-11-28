"""This app's sign-in, sealed with DPAPI in its own folder: never in plain text, in a log, on the panel or in --check.

The refresh token rotates with each use, so two renewals at once would leave the app signed out: one at a time, here
by a lock and across processes by the app running only once (app.py).
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from . import dpapi, server

FILE = "session.dpapi"


class Store:
    def __init__(self, folder: Path, protect: Callable[[bytes], bytes] = dpapi.protect,
                 unprotect: Callable[[bytes], bytes] = dpapi.unprotect):
        self.path = Path(folder) / FILE
        self.protect, self.unprotect = protect, unprotect
        self.lock = threading.Lock()

    def load(self) -> Optional[dict]:
        try:
            return json.loads(self.unprotect(self.path.read_bytes()).decode("utf-8"))
        except (OSError, ValueError):
            return None  # none, or one this Windows user cannot open: as if never signed in

    def save(self, session: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_bytes(self.protect(json.dumps(session).encode("utf-8")))
        temporary.replace(self.path)

    def signed_in(self) -> bool:
        return self.path.exists()

    def server(self) -> Optional[str]:
        session = self.load()
        return session.get("server") if session else None

    def may_send_notes(self) -> bool:
        return server.may_send(self.load())

    def sign_in(self, address: str, open_page=None) -> None:
        """A new sign-in; the one before it, if any, is ended on the server, so the Apps page lists this one only."""
        kept = server.sign_in(address, **({"open_page": open_page} if open_page else {}))
        with self.lock:
            old = self.load()
            self.save(kept)
        if old:
            server.revoke(old)

    def _fresh(self) -> Tuple[str, Optional[dict]]:
        """The session with a live access token, renewed if needed: ok, none, old, expired or offline. Under the lock."""
        session = self.load()
        if not session:
            return "none", None
        if not (session.get("access_token") and session.get("expires_at", 0) > time.time() + 300):
            try:
                session = server.refresh(session)
                self.save(session)
            except server.Expired:
                return "expired", None
            except (OSError, ValueError, KeyError):
                return "offline", None
        return ("ok" if server.may_send(session) else "old"), session

    def renew(self) -> str:
        """ok, none (never signed in), old (signed in before it could send notes), expired or offline."""
        with self.lock:
            return self._fresh()[0]

    def send(self, event: dict) -> Tuple[int, dict]:
        """One event with a live token: (status, answer). No session, an old one or no network raise OSError, so the
        event stays in the queue; a session the server no longer takes is a 401."""
        with self.lock:
            outcome, session = self._fresh()
        if outcome == "expired":
            return 401, {"error": "invalid_token"}
        if session is None or outcome == "old":
            raise OSError(outcome)
        return server.send(session, event)

    def sign_out(self) -> None:
        """Ends the session on the server and here: the sealed file goes, with whatever the app kept."""
        with self.lock:
            session = self.load()
            if session:
                server.revoke(session)
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
