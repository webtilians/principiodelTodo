"""What waits to be sent, sealed with DPAPI like the session: at most MAX_ITEMS and MAX_DAYS, then it goes.

Each note keeps its event_id, so sending it again never makes a second memory (the server answers «duplicate»); a
note that waits is retried with a growing wait, and «Cerrar sesión» empties the queue. Only notes that passed the secret
filter (capture.py) ever get here, and never a raw reading.
"""
import json
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from . import dpapi

FILE = "queue.dpapi"
MAX_ITEMS = 200  # (proposal, P-11)
MAX_DAYS = 7  # (proposal, P-11): as long as the server takes an event after it happened
FIRST_WAIT, LONGEST_WAIT = 60, 3600


class Full(Exception):
    pass


class Queue:
    def __init__(self, folder: Path, protect: Callable[[bytes], bytes] = dpapi.protect,
                 unprotect: Callable[[bytes], bytes] = dpapi.unprotect):
        self.path = Path(folder) / FILE
        self.protect, self.unprotect = protect, unprotect
        self.lock = threading.Lock()

    def _read(self) -> List[dict]:
        try:
            items = json.loads(self.unprotect(self.path.read_bytes()).decode("utf-8"))
        except (OSError, ValueError):
            return []
        return items if isinstance(items, list) else []

    def _write(self, items: List[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_bytes(self.protect(json.dumps(items).encode("utf-8")))
        temporary.replace(self.path)

    def _kept(self, items: List[dict], now: float) -> List[dict]:
        return [item for item in items if now - item["added"] < MAX_DAYS * 86400]

    def add(self, event: dict, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        with self.lock:
            items = self._kept(self._read(), now)
            if len(items) >= MAX_ITEMS:
                raise Full()
            items.append({"event": event, "added": now, "tries": 0, "next": now})
            self._write(items)

    def items(self, now: Optional[float] = None) -> List[dict]:
        now = time.time() if now is None else now
        with self.lock:
            return self._kept(self._read(), now)

    def due(self, now: Optional[float] = None) -> List[dict]:
        now = time.time() if now is None else now
        return [item for item in self.items(now) if item["next"] <= now]

    def done(self, event_id: str) -> None:
        with self.lock:
            self._write([item for item in self._read() if item["event"]["event_id"] != event_id])

    def later(self, event_id: str, now: Optional[float] = None, wait: Optional[float] = None) -> None:
        """Tried and not sent: again after a wait that doubles each time, from a minute to an hour."""
        now = time.time() if now is None else now
        with self.lock:
            items = self._read()
            for item in items:
                if item["event"]["event_id"] == event_id:
                    item["tries"] += 1
                    item["next"] = now + (wait if wait else min(LONGEST_WAIT, FIRST_WAIT * 2 ** (item["tries"] - 1)))
            self._write(items)

    def clear(self) -> None:
        with self.lock:
            self.path.unlink(missing_ok=True)
