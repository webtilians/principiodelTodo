"""This app's sign-in, sealed with DPAPI in its own folder: never in plain text, in a log, on the panel or in --check.

The refresh token rotates with each use, so two renewals at once would leave the app signed out: one at a time, here
by a lock and across processes by the app running only once (app.py).
"""
import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

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

    def sign_in(self, address: str, open_page=None) -> None:
        kept = server.sign_in(address, **({"open_page": open_page} if open_page else {}))
        with self.lock:
            self.save(kept)

    def renew(self) -> str:
        """ok, none (never signed in), expired (the server no longer takes it) or offline."""
        with self.lock:
            session = self.load()
            if not session:
                return "none"
            if session.get("access_token") and session.get("expires_at", 0) > time.time() + 300:
                return "ok"
            try:
                self.save(server.refresh(session))
            except server.Expired:
                return "expired"
            except (OSError, ValueError, KeyError):
                return "offline"
            return "ok"

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
