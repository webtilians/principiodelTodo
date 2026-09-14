import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .temporal import parse_explicit_date, parse_weekday_date


class CognitiveEventType(Enum):
    ASSERT_FACT = "assert_fact"
    REPLACE_FACT = "replace_fact"
    RETRACT_FACT = "retract_fact"
    ASSERT_PREFERENCE = "assert_preference"
    RETRACT_PREFERENCE = "retract_preference"
    CREATE_GOAL = "create_goal"
    COMPLETE_GOAL = "complete_goal"
    CANCEL_GOAL = "cancel_goal"
    RESCHEDULE_GOAL = "reschedule_goal"
    STORE_NOTE = "store_note"


@dataclass(frozen=True)
class CognitiveEvent:
    type: CognitiveEventType
    source_text: str
    subject: str = "user"
    predicate: Optional[str] = None
    value: Optional[str] = None
    previous_value: Optional[str] = None
    due_at: Optional[datetime] = None
    confidence: float = 1.0
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class RuleBasedCognitiveEventExtractor:
    """Transparent bilingual baseline for state-changing cognitive events.

    The extractor is intentionally operation-oriented. It does not decide what
    is relevant to the current question and it does not retrieve memory. It only
    converts explicit user statements into typed state transitions. Values are
    captured from the text rather than drawn from domain dictionaries.
    """

    _QUESTION_RE = re.compile(r"[?¿]")
    _WS_RE = re.compile(r"\s+")

    _FACT_PATTERNS: Tuple[Tuple[re.Pattern, str, bool], ...] = (
        (re.compile(r"\b(?:me llamo|mi nombre es)\s+([^,.!?;]+)", re.I), "name", True),
        (re.compile(r"\b(?:my name is)\s+([^,.!?;]+)", re.I), "name", True),
        (re.compile(r"\b(?:vivo en)\s+([^,.!?;]+)", re.I), "location", True),
        (re.compile(r"\b(?:i live in)\s+([^,.!?;]+)", re.I), "location", True),
        (re.compile(r"\bmi bici(?: principal)?(?: ahora)? es(?: una| un)?\s+([^,.!?;]+)", re.I), "bike", True),
        (re.compile(r"\bmy (?:main )?bike(?: now)? is(?: a| an)?\s+([^,.!?;]+)", re.I), "bike", True),
        (re.compile(r"\bmi color favorito es\s+([^,.!?;]+)", re.I), "favorite_color", True),
        (re.compile(r"\bmy favorite colou?r is(?: now)?\s+([^,.!?;]+)", re.I), "favorite_color", True),
        (re.compile(r"\bestoy estudiando\s+([^,.!?;]+)", re.I), "studying_language", True),
        (re.compile(r"\bi(?:'m| am) studying\s+([^,.!?;]+)", re.I), "studying_language", True),
        (re.compile(r"\btrabajo como\s+([^,.!?;]+)", re.I), "occupation", True),
        (re.compile(r"\bi work as(?: an?| the)?\s+([^,.!?;]+)", re.I), "occupation", True),
        (re.compile(r"\bmi perro se llama\s+([^,.!?;]+)", re.I), "pet_name", True),
        (re.compile(r"\bmy dog(?:'s name is| is called)\s+([^,.!?;]+)", re.I), "pet_name", True),
    )

    _PREFERENCE_PATTERNS: Tuple[re.Pattern, ...] = (
        re.compile(r"\bme gusta\s+([^,.!?;]+)", re.I),
        re.compile(r"\bme encanta\s+([^,.!?;]+)", re.I),
        re.compile(r"\bi like\s+([^,.!?;]+)", re.I),
        re.compile(r"\bi love\s+([^,.!?;]+)", re.I),
    )

    _GOAL_CREATE_MARKERS = (
        "tengo que", "tengo cita", "tengo dentista", "tengo reunion", "tengo reunión",
        "he quedado", "debo ", "necesito ", "me toca ",
        "i have to", "i need to", "i have an appointment", "i've arranged", "i have arranged",
    )

    _COMPLETE_MARKERS = (
        "marca ese compromiso como hecho", "marcalo como hecho", "márcalo como hecho",
        "marca esa tarea como hecha", "ya he ", "ya fui ", "ya comi ", "ya comí ",
        "already ", "mark that task done", "mark it done", "mark it complete",
        "is done", "it's done", "completed it", "finished it",
    )
    _CANCEL_MARKERS = (
        "cancela", "cancelar", "cancel ", "cancelled", "canceled", "anula", "anular",
        "ya no lo tengo", "no necesito hacerlo", "don't need to do it anymore",
    )
    _RESCHEDULE_MARKERS = (
        "lo han movido", "lo movieron", "se ha movido", "ya no es el", "reschedule", "rescheduled",
        "moved to", "moved from",
    )

    def __init__(self, now_fn=datetime.now):
        self._now_fn = now_fn

    def extract(self, text: str) -> List[CognitiveEvent]:
        now = self._now_fn()
        stripped = " ".join(text.strip().split())
        if not stripped:
            return []

        normalized = self._normalize(stripped)
        events: List[CognitiveEvent] = []

        # Information-seeking questions do not mutate cognitive state. Polite
        # reminder requests remain commands and are handled by the goal parser.
        if self._is_information_question(stripped, normalized):
            return []

        events.extend(self._extract_profile_replacements(stripped, normalized, now))
        events.extend(self._extract_preference_retractions(stripped, normalized, now))
        events.extend(self._extract_fact_retractions(stripped, normalized, now))
        events.extend(self._extract_facts(stripped, normalized, now, events))
        events.extend(self._extract_preferences(stripped, normalized, now, events))
        events.extend(self._extract_goal_events(stripped, normalized, now))
        events.extend(self._extract_notes(stripped, normalized, now))

        # Deduplicate equivalent events produced by overlapping generic patterns.
        seen = set()
        unique: List[CognitiveEvent] = []
        for event in events:
            key = (
                event.type.value,
                event.subject,
                event.predicate,
                self._normalize(event.value or ""),
                self._normalize(event.previous_value or ""),
                event.due_at.isoformat() if event.due_at else None,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(event)
        return unique

    def _extract_profile_replacements(
        self, text: str, normalized: str, now: datetime
    ) -> List[CognitiveEvent]:
        events: List[CognitiveEvent] = []

        patterns = (
            (re.compile(r"\bi moved to\s+([^,.!?;]+)", re.I), "location"),
            (re.compile(r"\bi(?:'ve| have) moved to\s+([^,.!?;]+)", re.I), "location"),
            (re.compile(r"\b(?:my current city is)\s+([^,.!?;]+)", re.I), "location"),
            (re.compile(r"\b(?:ahora vivo en|me he mudado a)\s+([^,.!?;]+)", re.I), "location"),
            (re.compile(r"\bmi bici(?: principal)? ahora es(?: una| un)?\s+([^,.!?;]+)", re.I), "bike"),
            (re.compile(r"\bmy favorite colou?r is now\s+([^,.!?;]+)", re.I), "favorite_color"),
        )
        for pattern, predicate in patterns:
            match = pattern.search(text)
            if match:
                value = self._clean_value(match.group(1))
                if value:
                    events.append(
                        CognitiveEvent(
                            CognitiveEventType.REPLACE_FACT,
                            text,
                            predicate=predicate,
                            value=value,
                            occurred_at=now,
                            metadata={"exclusive": True, "extractor": "rule_v1"},
                        )
                    )

        rename = re.search(
            r"\b(?:from now on,?\s*)?call me\s+([^,.!?;]+?)(?:\s+instead of\s+([^,.!?;]+))?(?:[.!]|$)",
            text,
            re.I,
        )
        if rename:
            events.append(
                CognitiveEvent(
                    CognitiveEventType.REPLACE_FACT,
                    text,
                    predicate="name",
                    value=self._clean_value(rename.group(1)),
                    previous_value=self._clean_value(rename.group(2)) if rename.group(2) else None,
                    occurred_at=now,
                    metadata={"exclusive": True, "extractor": "rule_v1"},
                )
            )

        language_change = re.search(
            r"\bi no longer study\s+([^;,.!?]+)[;,.]?\s*(?:i(?:'m| am) studying)\s+([^;,.!?]+)",
            text,
            re.I,
        )
        if language_change:
            old_value = self._clean_value(language_change.group(1))
            new_value = self._clean_value(language_change.group(2).replace(" now", ""))
            events.append(
                CognitiveEvent(
                    CognitiveEventType.REPLACE_FACT,
                    text,
                    predicate="studying_language",
                    value=new_value,
                    previous_value=old_value,
                    occurred_at=now,
                    metadata={"exclusive": True, "extractor": "rule_v1"},
                )
            )

        return events

    def _extract_fact_retractions(
        self, text: str, normalized: str, now: datetime
    ) -> List[CognitiveEvent]:
        events: List[CognitiveEvent] = []
        patterns = (
            (re.compile(r"\bi no longer study\s+([^;,.!?]+)", re.I), "studying_language"),
            (re.compile(r"\bya no estudio\s+([^;,.!?]+)", re.I), "studying_language"),
        )
        for pattern, predicate in patterns:
            match = pattern.search(text)
            if match:
                events.append(
                    CognitiveEvent(
                        CognitiveEventType.RETRACT_FACT,
                        text,
                        predicate=predicate,
                        value=self._clean_value(match.group(1)),
                        occurred_at=now,
                        metadata={"extractor": "rule_v1"},
                    )
                )
        return events

    def _extract_facts(
        self,
        text: str,
        normalized: str,
        now: datetime,
        existing: Sequence[CognitiveEvent],
    ) -> List[CognitiveEvent]:
        events: List[CognitiveEvent] = []
        already = {(e.predicate, self._normalize(e.value or "")) for e in existing}
        replacement_predicates = {
            e.predicate for e in existing if e.type == CognitiveEventType.REPLACE_FACT
        }
        for pattern, predicate, exclusive in self._FACT_PATTERNS:
            match = pattern.search(text)
            if not match or predicate in replacement_predicates:
                continue
            value = self._clean_value(match.group(1))
            key = (predicate, self._normalize(value))
            if value and key not in already:
                events.append(
                    CognitiveEvent(
                        CognitiveEventType.ASSERT_FACT,
                        text,
                        predicate=predicate,
                        value=value,
                        occurred_at=now,
                        metadata={"exclusive": exclusive, "extractor": "rule_v1"},
                    )
                )
        return events

    def _extract_preference_retractions(
        self, text: str, normalized: str, now: datetime
    ) -> List[CognitiveEvent]:
        events: List[CognitiveEvent] = []
        patterns = (
            re.compile(r"\b(?:ya no me gusta|he dejado)\s+([^,.!?;]+)", re.I),
            re.compile(r"\bi (?:don't|do not) (?:like|enjoy)\s+([^,.!?;]+?)(?:\s+anymore)?(?:[.!]|$)", re.I),
            re.compile(r"\bi stopped\s+(?:playing|doing|drinking|eating|practicing|practising)\s+([^,.!?;]+)", re.I),
            re.compile(r"\bi (?:don't|do not) drink\s+([^,.!?;]+?)\s+anymore", re.I),
        )
        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            value = self._normalize_preference_value(match.group(1))
            if value:
                events.append(
                    CognitiveEvent(
                        CognitiveEventType.RETRACT_PREFERENCE,
                        text,
                        predicate="likes",
                        value=value,
                        occurred_at=now,
                        metadata={"extractor": "rule_v1"},
                    )
                )

        # Common natural construction: "He dejado el kayak; ya no me gusta."
        stopped = re.search(r"\bhe dejado\s+(?:el |la |los |las )?([^;,.!?]+)", text, re.I)
        if stopped:
            value = self._normalize_preference_value(stopped.group(1))
            if value:
                events.append(
                    CognitiveEvent(
                        CognitiveEventType.RETRACT_PREFERENCE,
                        text,
                        predicate="likes",
                        value=value,
                        occurred_at=now,
                        metadata={"extractor": "rule_v1"},
                    )
                )
        return events

    def _extract_preferences(
        self,
        text: str,
        normalized: str,
        now: datetime,
        existing: Sequence[CognitiveEvent],
    ) -> List[CognitiveEvent]:
        if any(e.type == CognitiveEventType.RETRACT_PREFERENCE for e in existing):
            # Do not turn the negated clause inside the same sentence back into
            # a positive preference.
            return []
        events: List[CognitiveEvent] = []
        for pattern in self._PREFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                value = self._normalize_preference_value(match.group(1))
                if value:
                    events.append(
                        CognitiveEvent(
                            CognitiveEventType.ASSERT_PREFERENCE,
                            text,
                            predicate="likes",
                            value=value,
                            occurred_at=now,
                            metadata={"exclusive": False, "extractor": "rule_v1"},
                        )
                    )
        # "Ahora también me gusta..." is already covered by the generic pattern.
        return events

    def _extract_goal_events(
        self, text: str, normalized: str, now: datetime
    ) -> List[CognitiveEvent]:
        events: List[CognitiveEvent] = []

        is_reschedule = any(marker in normalized for marker in self._RESCHEDULE_MARKERS)
        is_cancel = any(marker in normalized for marker in self._CANCEL_MARKERS)
        is_complete = any(marker in normalized for marker in self._COMPLETE_MARKERS)

        if is_reschedule:
            target = self._goal_target(text, normalized, mode="reschedule")
            due_at = self._parse_due_at(text, now, prefer_last_weekday=True)
            if target:
                events.append(
                    CognitiveEvent(
                        CognitiveEventType.RESCHEDULE_GOAL,
                        text,
                        predicate="goal",
                        value=target,
                        due_at=due_at,
                        occurred_at=now,
                        metadata={"extractor": "rule_v1"},
                    )
                )
            return events

        if is_cancel:
            target = self._goal_target(text, normalized, mode="cancel")
            events.append(
                CognitiveEvent(
                    CognitiveEventType.CANCEL_GOAL,
                    text,
                    predicate="goal",
                    value=target or text,
                    occurred_at=now,
                    metadata={"extractor": "rule_v1"},
                )
            )
            return events

        if is_complete:
            target = self._goal_target(text, normalized, mode="complete")
            events.append(
                CognitiveEvent(
                    CognitiveEventType.COMPLETE_GOAL,
                    text,
                    predicate="goal",
                    value=target or text,
                    occurred_at=now,
                    metadata={"extractor": "rule_v1"},
                )
            )
            return events

        if any(marker in normalized for marker in self._GOAL_CREATE_MARKERS):
            due_at = self._parse_due_at(text, now)
            events.append(
                CognitiveEvent(
                    CognitiveEventType.CREATE_GOAL,
                    text,
                    predicate="goal",
                    value=self._canonical_goal_description(text),
                    due_at=due_at,
                    occurred_at=now,
                    metadata={"extractor": "rule_v1", "source_text": text},
                )
            )
        return events

    def _extract_notes(self, text: str, normalized: str, now: datetime) -> List[CognitiveEvent]:
        match = re.search(r"\bmi frase de prueba es:\s*(.+)$", text, re.I)
        if not match:
            match = re.search(r"\bmy test phrase is:\s*(.+)$", text, re.I)
        if not match:
            return []
        value = self._clean_value(match.group(1))
        return [
            CognitiveEvent(
                CognitiveEventType.STORE_NOTE,
                text,
                predicate="test_phrase",
                value=value,
                occurred_at=now,
                metadata={"extractor": "rule_v1", "instruction_like_data": True},
            )
        ]

    def _parse_due_at(
        self, text: str, now: datetime, *, prefer_last_weekday: bool = False
    ) -> Optional[datetime]:
        normalized = self._normalize(text)
        explicit = parse_explicit_date(text, now)
        target_date = explicit

        weekday_matches = list(
            re.finditer(
                r"\b(lunes|martes|miercoles|jueves|viernes|sabado|domingo|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                normalized,
            )
        )
        if target_date is None and weekday_matches:
            chosen = weekday_matches[-1] if prefer_last_weekday else weekday_matches[0]
            token = chosen.group(1)
            parsed = parse_weekday_date(token, now)
            if parsed is not None:
                target_date = parsed
                prefix = normalized[max(0, chosen.start() - 12): chosen.start()]
                suffix = normalized[chosen.end(): chosen.end() + 14]
                explicit_next = "next " in prefix or "que viene" in suffix or "proximo" in prefix
                if explicit_next and target_date <= now.date() + timedelta(days=6):
                    # "next Monday" / "lunes que viene" is the following-week
                    # occurrence, not an accidental same-day match.
                    if target_date <= now.date() or now.weekday() == target_date.weekday():
                        target_date = target_date + timedelta(days=7)

        if target_date is None:
            if "pasado manana" in normalized or "day after tomorrow" in normalized:
                target_date = (now + timedelta(days=2)).date()
            elif "manana" in normalized or "tomorrow" in normalized:
                target_date = (now + timedelta(days=1)).date()
            elif "hoy" in normalized or "today" in normalized:
                target_date = now.date()

        if target_date is None:
            return None

        time_match = re.search(
            r"(?:a las|a la|at)\s+(\d{1,2})(?::(\d{2}))?",
            normalized,
        )
        hour = int(time_match.group(1)) if time_match else now.hour
        minute = int(time_match.group(2) or 0) if time_match else now.minute
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            hour, minute = now.hour, now.minute
        return datetime.combine(target_date, datetime.min.time()).replace(hour=hour, minute=minute)

    @classmethod
    def _goal_target(cls, text: str, normalized: str, *, mode: str) -> str:
        # Strip lifecycle/calendar scaffolding while preserving the user's own
        # noun phrase. Matching can later be semantic, so no domain translation
        # table is required here.
        cleaned = normalized
        scaffolding = (
            "mark that task done", "mark it done", "mark it complete", "already",
            "cancela", "cancel", "cancelar", "anula", "anular", "ya", "he", "fui",
            "marca", "marcalo", "como hecho", "como hecha", "lo han movido", "ya no es",
            "moved to", "moved from", "reschedule", "rescheduled", "this morning",
            "i", "the", "task", "today", "hoy",
        )
        for marker in scaffolding:
            cleaned = cleaned.replace(marker, " ")
        cleaned = re.sub(
            r"\b(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            " ",
            cleaned,
        )
        cleaned = re.sub(r"\b\d{1,2}(?::\d{2})?\b", " ", cleaned)
        cleaned = " ".join(token for token in cleaned.split() if len(token) > 2)
        return cleaned[:180].strip() or text[:180].strip()

    @classmethod
    def _canonical_goal_description(cls, text: str) -> str:
        return " ".join(text.strip().split())

    @classmethod
    def _normalize_preference_value(cls, value: str) -> str:
        cleaned = cls._clean_value(value)
        cleaned = re.sub(r"^(?:el|la|los|las|un|una|the)\s+", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+(?:anymore|ahora)$", "", cleaned, flags=re.I)
        # Preference activities often contain a light verb. Removing it makes
        # positive and negative variants converge on the same state key.
        cleaned = re.sub(
            r"^(?:hacer|practicar|jugar al|jugar a|salir en|tomar|comer|escuchar|cuidar|dibujar con|tallar|playing|doing|drinking|eating|practicing|practising)\s+",
            "",
            cleaned,
            flags=re.I,
        )
        return cleaned.strip()

    @classmethod
    def _clean_value(cls, value: Optional[str]) -> str:
        if not value:
            return ""
        cleaned = " ".join(value.strip(" .,:;!?\"'\t\n").split())
        return cleaned

    @classmethod
    def _normalize(cls, text: str) -> str:
        deaccented = "".join(
            char
            for char in unicodedata.normalize("NFKD", text.lower())
            if not unicodedata.combining(char)
        )
        return cls._WS_RE.sub(" ", deaccented).strip()

    @classmethod
    def _is_information_question(cls, text: str, normalized: str) -> bool:
        if not cls._QUESTION_RE.search(text):
            return False
        if any(marker in normalized for marker in ("recuerdame", "avisame", "remind me")):
            return False
        return True
