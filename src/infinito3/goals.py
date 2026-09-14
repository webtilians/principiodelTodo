import re
import unicodedata
from datetime import datetime, timedelta
from typing import List, Optional, Set

from .types import Goal


class SimpleGoalEngine:
    """Small, deterministic temporal parser for the first INFINITO 3.0 milestone.

    It fixes the legacy ordering bug where "pasado mañana" matched "mañana"
    first and keeps goal lifecycle semantics deliberately conservative: a time
    reference is not a goal by itself, overdue does not mean completed, and
    completion/cancellation requires an explicit user signal.
    """

    _REMINDER_MARKERS = (
        "recuérdame", "recuerdame", "avísame", "avisame", "no olvides",
        "tengo cita", "tengo reunión", "tengo reunion", "tengo que",
        "debo ", "necesito ", "i have to", "i need to",
    )
    _REMINDER_REQUEST_MARKERS = (
        "recuérdame", "recuerdame", "avísame", "avisame", "no olvides",
        "recordarme", "remind me",
    )
    _INTERROGATIVE_PREFIXES = (
        "qué ", "que ", "cuándo ", "cuando ", "dónde ", "donde ",
        "cómo ", "como ", "cuál ", "cual ", "cuáles ", "cuales ",
        "what ", "when ", "where ", "how ", "which ",
    )
    _INTERROGATIVE_RE = re.compile(
        r"(?:^|[¿?]\s*)"
        r"(?:qué|que|cuándo|cuando|dónde|donde|cómo|como|cuál|cual|cuáles|cuales|"
        r"what|when|where|how|which)\b",
        re.I,
    )
    _EXPLICIT_COMPLETION_RE = re.compile(
        r"(?:deja\s+de\s+considerarl[oa]\s+pendiente|"
        r"marc(?:a|alo|ala|arlo|arla)?\s+como\s+(?:hech[oa]|completad[oa])|"
        r"(?:ya\s+)?(?:esta|está)\s+(?:hech[oa]|terminad[oa]|completad[oa])|"
        r"(?:he|hemos)\s+(?:hecho|terminado|completado))",
        re.I,
    )
    _CANCELLATION_RE = re.compile(
        r"\b(?:cancela|cancelar|cancelado|cancelada|anula|anular|anulado|anulada)\b",
        re.I,
    )
    _STOP_WORDS = {
        "a", "al", "de", "del", "el", "la", "los", "las", "un", "una", "unos", "unas",
        "y", "o", "que", "tengo", "debo", "necesito", "mañana", "manana", "hoy", "pasado",
        "semana", "las", "la", "con", "por", "para", "ya", "he", "hemos", "considerarlo",
        "considerarla", "pendiente", "como", "i", "to", "the", "a", "an", "and", "have", "need",
        "tomorrow", "today", "next", "week",
    }

    def __init__(self, now_fn=datetime.now):
        self._now_fn = now_fn
        self._goals: List[Goal] = []

    def ingest(self, text: str) -> List[Goal]:
        normalized = " ".join(text.lower().split())
        if self._is_interrogative(normalized):
            return []

        lifecycle = self._lifecycle_intent(normalized)
        if lifecycle is not None:
            self._apply_lifecycle_update(text, lifecycle)
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
        if "?" not in text:
            return False

        # A polite reminder request can itself be phrased as a question. Those
        # are commands, not information-seeking probes, and should still create
        # a goal.
        if any(marker in text for marker in self._REMINDER_REQUEST_MARKERS):
            return False

        stripped = text.lstrip("¿").strip()
        if any(stripped.startswith(prefix) for prefix in self._INTERROGATIVE_PREFIXES):
            return True

        # Long-horizon conversations often prefix the actual question with
        # temporal framing: "Hoy es 17... ¿qué tengo pendiente mañana?".
        return bool(self._INTERROGATIVE_RE.search(text))

    def _looks_like_goal(self, text: str) -> bool:
        # A temporal word alone is not intention. This avoids turning narrative
        # statements such as "Hoy he leído..." into open goals. Explicit
        # reminder requests share the same intent signal whether declarative or
        # politely phrased as a question.
        if any(marker in text for marker in self._REMINDER_MARKERS + self._REMINDER_REQUEST_MARKERS):
            return True

        normalized = self._normalize(text)
        temporal = bool(
            re.search(
                r"\b(?:manana|pasado manana|semana que viene|proxima semana|tomorrow|next week)\b",
                normalized,
            )
        )
        planned_action = bool(
            re.search(
                r"\b(?:voy a|quiero|hare|tendre|me toca|quedo|i will|i am going to|i'm going to)\b",
                normalized,
            )
        )
        return temporal and planned_action

    def _lifecycle_intent(self, text: str) -> Optional[str]:
        normalized = self._normalize(text)
        if self._CANCELLATION_RE.search(normalized):
            return "cancelled"
        if self._EXPLICIT_COMPLETION_RE.search(normalized):
            return "completed"

        # "Ya recogí el paquete" is useful evidence only if it can be anchored
        # to an existing open goal. The overlap threshold in the matcher keeps
        # a generic "ya..." statement from completing an unrelated task.
        if normalized.startswith("ya "):
            return "completed"
        return None

    def _apply_lifecycle_update(self, text: str, lifecycle: str) -> Optional[Goal]:
        open_goals = [goal for goal in self._goals if not goal.completed]
        if not open_goals:
            return None

        message_terms = self._goal_terms(text)
        if not message_terms:
            return None

        ranked = []
        for goal in open_goals:
            goal_terms = self._goal_terms(goal.description)
            overlap = len(message_terms & goal_terms)
            coverage = overlap / max(1, min(len(message_terms), len(goal_terms)))
            ranked.append((overlap, coverage, goal))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

        best_overlap, best_coverage, best = ranked[0]
        second_overlap = ranked[1][0] if len(ranked) > 1 else -1
        if best_overlap < 1:
            return None
        if best_overlap == 1 and best_coverage < 0.34:
            return None
        if best_overlap == second_overlap and best_overlap < 2:
            return None

        best.completed = True
        best.metadata["lifecycle"] = lifecycle
        best.metadata["lifecycle_source"] = text
        best.metadata["lifecycle_at"] = self._now_fn().isoformat()
        return best

    @classmethod
    def _goal_terms(cls, text: str) -> Set[str]:
        normalized = cls._normalize(text)
        terms: Set[str] = set()
        for token in re.findall(r"[a-z0-9]+", normalized):
            if token in cls._STOP_WORDS or len(token) <= 2 or token.isdigit():
                continue
            terms.add(token[:5] if len(token) >= 5 else token)
        return terms

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(
            char
            for char in unicodedata.normalize("NFKD", text.lower())
            if not unicodedata.combining(char)
        )

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
