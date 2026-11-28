"""Where the cognitive engine stands, as data: what `python -m cognition status --json` prints, and what INFINITO Desktop
(desktop/infinito_desktop/checks.py) reads instead of reading the engine's files another way.

It never opens session.json: whether the engine is signed in is whether that file is there.
"""
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from .cycle import Engine
from .diary import check, proposals


def summary(folder: Path, conf: dict, today: Optional[date] = None) -> Optional[dict]:
    """The engine's state, or None when it has no folder on this computer (it never ran here)."""
    folder = Path(folder)
    if not folder.is_dir():
        return None
    live = bool(conf.get("live_authorized"))
    engine = Engine(folder, conf, None, lambda role, chosen: None, live=live)
    today = today or datetime.now().date()
    paused = folder / "PAUSED"
    since = None
    if paused.exists():
        try:
            since = paused.read_text(encoding="utf-8").strip() or None
        except OSError:
            since = None
    end = engine.last_review()
    review = None
    if end:
        judged = engine._judged(end)
        review = {"at": end.get("at"), "outcome": end.get("outcome"), "judged": judged,
                  "problem": None if judged else engine.supervisor_problem(end.get("run"))}
    return {
        "signed_in": (folder / "session.json").exists(),
        "server": conf.get("server"),
        "live": live,
        "paused": paused.exists(),
        "paused_since": since,
        "last_review": review,
        "spent_today_eur": round(engine.ledger.spent(today), 4),
        "cap_today_eur": round(engine.ledger.cap(today), 4),
        "proposals_waiting": sum(p["status"] == "pending" for p in proposals(engine.diary)),
        "diary_problems": check(engine.diary),
    }
