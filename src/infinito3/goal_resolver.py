import math
import re
import unicodedata
from datetime import datetime
from typing import Optional, Sequence

from .cognitive_events import CognitiveEvent, CognitiveEventType
from .types import Goal


class StateAwareGoalResolver:
    """Resolve lifecycle events against open goals using state, time and semantics.

    The resolver never mutates goals. It combines lexical/entity overlap, embedding
    similarity against both canonical and original goal text, explicit old due
    times from semantic extraction, and proximity to the event clock. A target is
    returned only when evidence is sufficient; ambiguous flat matches fail closed.
    """

    _STOP = {
        "the", "a", "an", "to", "i", "it", "that", "this", "my", "me", "do", "did",
        "is", "was", "be", "already", "done", "mark", "task", "cancel", "reschedule",
        "el", "la", "los", "las", "un", "una", "de", "del", "al", "que", "ya", "he",
        "como", "para", "por", "con", "tengo", "marca", "hecho", "hecha", "cancela",
        "hoy", "today", "tomorrow", "manana", "mañana", "next", "proximo", "próximo",
        "lunes", "martes", "miercoles", "miércoles", "jueves", "viernes", "sabado", "sábado", "domingo",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    }

    def resolve(self, event: CognitiveEvent, goals: Sequence[Goal], *, embedding_provider=None) -> Optional[Goal]:
        open_goals = [goal for goal in goals if not goal.completed]
        if not open_goals:
            return None

        query = " ".join(part for part in (event.value or "", event.source_text or "") if part).strip()
        query_terms = self._terms(query)
        query_embedding = self._safe_embed(embedding_provider, query)
        previous_due = self._metadata_datetime(event.metadata.get("previous_due_at"))

        ranked = []
        for goal in open_goals:
            source = str(goal.metadata.get("source_text") or "")
            candidate_text = " ".join(part for part in (goal.description, source) if part)
            goal_terms = self._terms(candidate_text)
            overlap = len(query_terms & goal_terms)
            lexical = overlap / max(1, len(query_terms | goal_terms))
            entity_overlap = overlap / max(1, min(len(query_terms), len(goal_terms)))

            semantic = 0.0
            if query_embedding is not None:
                vectors = [
                    self._safe_embed(embedding_provider, goal.description),
                    self._safe_embed(embedding_provider, source) if source else None,
                ]
                semantic = max(
                    [max(0.0, self._cosine(query_embedding, vector)) for vector in vectors if vector] or [0.0]
                )

            temporal = self._temporal_score(event, goal, previous_due)
            score = 0.38 * semantic + 0.28 * lexical + 0.22 * entity_overlap + temporal
            ranked.append((score, semantic, lexical, entity_overlap, temporal, goal))

        ranked.sort(key=lambda row: row[0], reverse=True)
        best = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -1.0
        score, semantic, lexical, entity_overlap, temporal, goal = best
        margin = score - second_score

        # Strong explicit identity is enough even when several goals share generic words.
        if entity_overlap >= 0.50 and lexical > 0.0:
            return goal
        # An explicit previous due time plus some textual/semantic evidence is strong.
        if temporal >= 0.24 and (lexical > 0.0 or semantic >= 0.30):
            return goal
        # Cross-language semantic resolution must clear both an absolute score and margin.
        if semantic >= 0.50 and score >= 0.30 and margin >= 0.018:
            return goal
        # Ordinary mixed evidence path.
        if score >= 0.34 and margin >= 0.025 and (lexical > 0.0 or semantic >= 0.38):
            return goal

        # Pronoun-heavy completion/cancellation can still be resolved by a uniquely
        # due-now goal; do not use this for reschedules because the new due date is
        # not evidence for the old target.
        if event.type in (CognitiveEventType.COMPLETE_GOAL, CognitiveEventType.CANCEL_GOAL):
            if temporal >= 0.20 and margin >= 0.06:
                return goal
        return None

    def _temporal_score(self, event: CognitiveEvent, goal: Goal, previous_due: Optional[datetime]) -> float:
        due = goal.due_at
        if due is None:
            return 0.0
        score = 0.0
        if previous_due is not None:
            seconds = abs((due - previous_due).total_seconds())
            if seconds <= 60:
                score += 0.32
            elif due.date() == previous_due.date():
                score += 0.24

        source = self._normalize(event.source_text or "")
        weekday_names = {
            0: ("lunes", "monday"), 1: ("martes", "tuesday"), 2: ("miercoles", "wednesday"),
            3: ("jueves", "thursday"), 4: ("viernes", "friday"), 5: ("sabado", "saturday"),
            6: ("domingo", "sunday"),
        }
        if any(name in source for name in weekday_names[due.weekday()]):
            score += 0.10
        if re.search(rf"\b{due.hour}(?::0?{due.minute})?\b", source):
            score += 0.08

        if event.type in (CognitiveEventType.COMPLETE_GOAL, CognitiveEventType.CANCEL_GOAL):
            hours = abs((due - event.occurred_at).total_seconds()) / 3600.0
            if hours <= 12:
                score += 0.20
            elif hours <= 36:
                score += 0.14
            elif hours <= 96:
                score += 0.07
        return min(score, 0.42)

    @classmethod
    def _terms(cls, text: str):
        return {
            token
            for token in re.findall(r"[a-z0-9áéíóúüñ]+", cls._normalize(text))
            if len(token) > 2 and token not in cls._STOP
        }

    @staticmethod
    def _metadata_datetime(value) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", ""))
        except ValueError:
            return None

    @staticmethod
    def _safe_embed(provider, text: str):
        if provider is None or not text:
            return None
        try:
            return list(provider.embed(text))
        except Exception:
            return None

    @staticmethod
    def _cosine(a, b) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(
            "".join(
                char for char in unicodedata.normalize("NFKD", text.lower())
                if not unicodedata.combining(char)
            ).split()
        )
