"""One look at everything: it gathers what each check reads (files, the scheduled tasks, the server) and asks each one.

It only reads, and each piece fails on its own: whatever goes wrong in one is that check's «no lo sé», never the app's
end. The engine is read through its own summary (cognition/status.py), as `python -m cognition status --json` does.
"""
import ctypes
import os
import subprocess
from datetime import datetime
from typing import List, Optional

from . import checks, places, server

NO_WINDOW = 0x08000000


def _engine_summary() -> Optional[dict]:
    from cognition import config as settings
    from cognition.status import summary

    folder = places.cognition_home()
    return summary(folder, settings.load(folder)) if folder.is_dir() else None


def _schtasks() -> Optional[str]:
    if os.name != "nt":
        return None
    done = subprocess.run(["schtasks", "/query", "/fo", "csv", "/v"], capture_output=True, timeout=60,
                          creationflags=NO_WINDOW)
    if done.returncode != 0:
        return None
    return done.stdout.decode(f"cp{ctypes.windll.kernel32.GetOEMCP()}", errors="replace")


def everything(store, address: str = places.SERVER, now: Optional[datetime] = None) -> List[checks.Check]:
    now = now or datetime.now().astimezone()
    page = f"{address}/memory?view=cognition"
    looks = [
        ("session", lambda: checks.session(store.renew(), address)),
        ("server", lambda: checks.server(lambda: server.answers(address), address)),
        ("hooks", lambda: checks.hooks(places.claude_settings(), places.hooks_session())),
        ("engine", lambda: checks.engine(_engine_summary(), now, page)),
        ("backups", lambda: checks.backups(places.backups(), now)),
        ("tasks", lambda: checks.tasks(None if (text := _schtasks()) is None else checks.scheduled_tasks(text))),
    ]
    found = []
    for key, look in looks:
        try:
            found.append(look())
        except Exception as error:  # noqa: BLE001  # one check that breaks says so and the others go on
            found.append(checks.Check(key, checks.UNKNOWN, key, f"No he podido comprobarlo ({type(error).__name__})."))
    return found
