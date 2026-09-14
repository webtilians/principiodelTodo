import re
from datetime import datetime, timedelta
from typing import List, Optional

from .types import Goal


class SimpleGoalEngine:
    """Small, deterministic temporal parser for the first INFINITO 3.0 milestone.

    It fixes the legacy ordering bug where "pasado mañana" matched "mañana"
    first. A production temporal parser can replace this class later without
    changing CognitiveEngine.
    """

    _REMINDER_MARKERS = (
        "recuérdame", "recuerdame", "avísame", "avisame", "no olvides",
        "tengo cita", "tengo reunión", "tengo reunion", "tengo que",
    )
    _INTERROGATIVE_PREFIXES = (
        "qué ", "que ", "cuándo ", "cuando ", "dónde ", "donde ",
        "cómo ", "como ", "cuál ", "cual ", "cuáles ", "cuales ",
    )

    def __init__(self, now_fn=datetime.now):
        self._now_fn = now_fn
        self._goals: List[Goal] = []

    def ingest(self, text: str) -> List[Goal]:
        normalized = " ".join(text.lower().split())
        if self._is_interrogative(normalized):
            return []
        if not self._looks_like_goal(normalized):
            return []

        due_at = self._parse_due_at(normalized)
        goal = Goal(description=text, due_at=due_at, metadata={"source": "text"})
        self._goals.append(goal)
        return [goal]

    def due(self) -> List[Goal]:
        now = self._now_fn()
        return [g for g in self._goals if not g.completed and g.due_at and g.due_at <= now]

    def all(self) -> List[Goal]:
        return list(self._goals)

    def _is_interrogative(self, text: str) -> bool:
        stripped = text.lstrip("¿").strip()
        return "?" in text and any(
            stripped.startswith(prefix) for prefix in self._INTERROGATIVE_PREFIXES
        )

    def _looks_like_goal(self, text: str) -> bool:
        has_marker = any(marker in text for marker in self._REMINDER_MARKERS)
        has_future_reference = any(
            marker in text for marker in ("hoy", "mañana", "pasado mañana", "semana que viene")
        )
        return has_marker or has_future_reference

    def _parse_due_at(self, text: str) -> Optional[datetime]:
        now = self._now_fn()

        # Most specific phrases first.
        if "pasado mañana" in text:
            target = now + timedelta(days=2)
        elif "mañana" in text:
            target = now + timedelta(days=1)
        elif "semana que viene" in text or "próxima semana" in text or "proxima semana" in text:
            target = now + timedelta(days=7)
        elif "hoy" in text:
            target = now
        else:
            return None

        parsed_time = self._parse_clock_time(text)
        if parsed_time is None:
            return target

        hour, minute = parsed_time
        return target.replace(hour=hour, minute=minute, second=0, microsecond=0)

    @staticmethod
    def _parse_clock_time(text: str):
        match = re.search(r"(?:a las|a la)\s+(\d{1,2})(?::(\d{2}))?", text)
        if not match:
            return None
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return hour, minute
        return None
