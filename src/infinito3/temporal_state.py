import math
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .cognitive_events import CognitiveEvent, CognitiveEventType
from .types import Goal, MemoryKind, MemoryRecord


@dataclass
class TemporalFactVersion:
    subject: str
    predicate: str
    value: str
    valid_from: datetime
    valid_to: Optional[datetime] = None
    active: bool = True
    retracted: bool = False
    source_event_id: Optional[str] = None
    memory_id: Optional[str] = None
    supersedes_version_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class TemporalGoalVersion:
    goal_id: str
    description: str
    status: str
    changed_at: datetime
    due_at: Optional[datetime] = None
    source_event_id: Optional[str] = None
    previous_due_at: Optional[datetime] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class TemporalStateTransition:
    event_id: str
    event_type: str
    memory_ids: List[str] = field(default_factory=list)
    goal_ids: List[str] = field(default_factory=list)
    closed_version_ids: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class TemporalCognitiveState:
    """Authoritative event-sourced state for facts, preferences and goals.

    Current state is a projection of an append-only cognitive-event stream.
    Replacements and retractions close old versions rather than destroying them,
    so current and historical questions share the same lineage.
    """

    _HISTORY_MARKERS = (
        "antes de", "antes del", "antes que", "anterior", "anteriormente",
        "usaba antes", "vivia antes", "vivía antes", "llamabas antes",
        "ya no me gusta", "he dicho explicitamente que ya no", "historial",
        "before ", "previous", "previously", "used to", "no longer like",
        "stopped ", "historical", "history",
    )

    def __init__(self, now_fn=datetime.now):
        self._now_fn = now_fn
        self._events: List[CognitiveEvent] = []
        self._facts: Dict[Tuple[str, str], List[TemporalFactVersion]] = {}
        self._goal_history: List[TemporalGoalVersion] = []

    def apply(self, events: Sequence[CognitiveEvent], *, memory_store=None, goal_engine=None) -> List[TemporalStateTransition]:
        transitions: List[TemporalStateTransition] = []
        for event in events:
            self._events.append(event)
            transition = TemporalStateTransition(event.id, event.type.value)
            if event.type in (CognitiveEventType.ASSERT_FACT, CognitiveEventType.REPLACE_FACT, CognitiveEventType.ASSERT_PREFERENCE):
                self._apply_fact_event(event, transition, memory_store)
            elif event.type in (CognitiveEventType.RETRACT_FACT, CognitiveEventType.RETRACT_PREFERENCE):
                self._apply_retraction(event, transition, memory_store)
            elif event.type == CognitiveEventType.STORE_NOTE:
                self._apply_note(event, transition, memory_store)
            elif event.type == CognitiveEventType.CREATE_GOAL:
                self._apply_goal_create(event, transition, goal_engine)
            elif event.type in (CognitiveEventType.COMPLETE_GOAL, CognitiveEventType.CANCEL_GOAL, CognitiveEventType.RESCHEDULE_GOAL):
                self._apply_goal_lifecycle(event, transition, goal_engine, memory_store=memory_store)
            transitions.append(transition)
        return transitions

    def current_fact(self, predicate: str, *, subject: str = "user") -> Optional[TemporalFactVersion]:
        active = [v for v in self._facts.get((subject, predicate), []) if v.active and not v.retracted]
        return active[-1] if active else None

    def current_values(self, predicate: str, *, subject: str = "user") -> List[str]:
        return [v.value for v in self._facts.get((subject, predicate), []) if v.active and not v.retracted]

    def fact_history(self, predicate: str, *, subject: str = "user") -> List[TemporalFactVersion]:
        return list(self._facts.get((subject, predicate), []))

    def events(self) -> List[CognitiveEvent]:
        return list(self._events)

    def goal_history(self) -> List[TemporalGoalVersion]:
        return list(self._goal_history)

    @classmethod
    def wants_history(cls, query: str) -> bool:
        normalized = cls._normalize(query)
        return any(marker in normalized for marker in cls._HISTORY_MARKERS)

    def _apply_fact_event(self, event, transition, memory_store) -> None:
        if not event.predicate or not event.value:
            transition.notes.append("ignored_fact_without_predicate_or_value")
            return

        key = (event.subject, event.predicate)
        versions = self._facts.setdefault(key, [])
        exclusive = bool(event.metadata.get("exclusive")) or event.type == CognitiveEventType.REPLACE_FACT
        previous_active = [v for v in versions if v.active and not v.retracted]

        if exclusive:
            for version in previous_active:
                if self._same_value(version.value, event.value):
                    transition.notes.append("equivalent_current_fact")
                    return
                version.active = False
                version.valid_to = event.occurred_at
                transition.closed_version_ids.append(version.id)

        supersedes = previous_active[-1].id if exclusive and previous_active else None
        version = TemporalFactVersion(
            subject=event.subject,
            predicate=event.predicate,
            value=event.value,
            valid_from=event.occurred_at,
            source_event_id=event.id,
            supersedes_version_id=supersedes,
        )
        versions.append(version)

        if memory_store is not None:
            # Initial assertions preserve the user's exact wording for audit and
            # backward compatibility. Replacements are canonicalized so a phrase
            # such as "call me Dani instead of Diego" cannot keep the stale value
            # inside the active memory text.
            if event.type in (CognitiveEventType.ASSERT_FACT, CognitiveEventType.ASSERT_PREFERENCE):
                content = event.source_text
            else:
                content = self._canonical_fact_content(event.predicate, event.value)
            record = MemoryRecord(
                content=content,
                kind=MemoryKind.USER_MODEL,
                importance=0.92 if exclusive else 0.82,
                confidence=event.confidence,
                fact_subject=event.subject,
                fact_predicate=event.predicate,
                fact_value=self._normalize_value(event.value),
                metadata={
                    "cognitive_event_id": event.id,
                    "cognitive_event_type": event.type.value,
                    "source_text": event.source_text,
                    "fact_exclusive": exclusive,
                    "temporal_valid_from": event.occurred_at.isoformat(),
                    "temporal_supersedes_version_id": supersedes,
                },
            )
            stored = memory_store.add(record)
            version.memory_id = getattr(stored, "id", None)
            if version.memory_id:
                transition.memory_ids.append(version.memory_id)

    def _apply_retraction(self, event, transition, memory_store) -> None:
        if not event.predicate or not event.value:
            transition.notes.append("ignored_retraction_without_target")
            return
        versions = self._facts.setdefault((event.subject, event.predicate), [])
        candidates = [
            v for v in versions
            if v.active and not v.retracted and self._values_match(v.value, event.value)
        ]
        if not candidates:
            transition.notes.append("retraction_target_not_in_temporal_state")
        for version in candidates:
            version.active = False
            version.retracted = True
            version.valid_to = event.occurred_at
            transition.closed_version_ids.append(version.id)

        retractor = getattr(memory_store, "retract_fact", None) if memory_store is not None else None
        if callable(retractor):
            affected = retractor(
                event.subject,
                event.predicate,
                self._normalize_value(event.value),
                reason=event.type.value,
                source_text=event.source_text,
                at=event.occurred_at,
            )
            transition.memory_ids.extend(list(affected or []))

    def _apply_note(self, event, transition, memory_store) -> None:
        if memory_store is None or not event.value:
            return
        record = MemoryRecord(
            content=event.value,
            kind=MemoryKind.SEMANTIC,
            importance=0.86,
            confidence=event.confidence,
            fact_subject=event.subject,
            fact_predicate=event.predicate or "note",
            fact_value=self._normalize_value(event.value),
            metadata={
                "cognitive_event_id": event.id,
                "cognitive_event_type": event.type.value,
                "source_text": event.source_text,
                "instruction_like_data": bool(event.metadata.get("instruction_like_data")),
                "fact_exclusive": False,
            },
        )
        stored = memory_store.add(record)
        if getattr(stored, "id", None):
            transition.memory_ids.append(stored.id)

    def _apply_goal_create(self, event, transition, goal_engine) -> None:
        if goal_engine is None:
            return
        creator = getattr(goal_engine, "add_structured_goal", None)
        description = event.value or event.source_text
        if callable(creator):
            goal = creator(
                description,
                due_at=event.due_at,
                metadata={
                    "cognitive_event_id": event.id,
                    "source_text": event.source_text,
                    "structured": True,
                },
            )
        else:
            created = goal_engine.ingest(event.source_text)
            goal = created[0] if created else None
        if goal is None:
            transition.notes.append("goal_projection_failed")
            return
        transition.goal_ids.append(goal.id)
        self._goal_history.append(
            TemporalGoalVersion(goal.id, goal.description, "open", event.occurred_at, due_at=goal.due_at, source_event_id=event.id)
        )

    def _apply_goal_lifecycle(self, event, transition, goal_engine, *, memory_store=None) -> None:
        if goal_engine is None:
            return
        goals = [goal for goal in goal_engine.all() if not goal.completed]
        if not goals:
            transition.notes.append("no_open_goal_for_lifecycle_event")
            return
        target = self._match_goal(event, goals, memory_store=memory_store)
        if target is None:
            transition.notes.append("goal_lifecycle_target_not_found")
            return

        previous_due = target.due_at
        if event.type == CognitiveEventType.RESCHEDULE_GOAL:
            if event.due_at is None:
                transition.notes.append("reschedule_without_due_at")
                return
            target.due_at = event.due_at
            target.description = self._canonical_goal_label(event.value or target.description)
            target.metadata["lifecycle"] = "rescheduled"
            target.metadata["reschedule_source"] = event.source_text
            target.metadata["rescheduled_at"] = event.occurred_at.isoformat()
            status = "rescheduled"
        else:
            target.completed = True
            status = "cancelled" if event.type == CognitiveEventType.CANCEL_GOAL else "completed"
            target.metadata["lifecycle"] = status
            target.metadata["lifecycle_source"] = event.source_text
            target.metadata["lifecycle_at"] = event.occurred_at.isoformat()

        transition.goal_ids.append(target.id)
        self._goal_history.append(
            TemporalGoalVersion(
                target.id,
                target.description,
                status,
                event.occurred_at,
                due_at=target.due_at,
                previous_due_at=previous_due,
                source_event_id=event.id,
            )
        )

    def _match_goal(self, event: CognitiveEvent, goals: Sequence[Goal], *, memory_store=None) -> Optional[Goal]:
        query = event.value or event.source_text
        query_terms = self._content_terms(query)
        embedder = getattr(memory_store, "embedding_provider", None)
        query_embedding = self._safe_embed(embedder, query)
        ranked = []
        for goal in goals:
            goal_terms = self._content_terms(goal.description)
            overlap = len(query_terms & goal_terms)
            lexical = overlap / max(1, len(query_terms | goal_terms))
            semantic = 0.0
            if query_embedding is not None:
                goal_embedding = self._safe_embed(embedder, goal.description)
                if goal_embedding is not None:
                    semantic = max(0.0, self._cosine(query_embedding, goal_embedding))
            due_bonus = 0.12 if event.due_at and goal.due_at and event.due_at.date() == goal.due_at.date() else 0.0
            ranked.append((0.50 * semantic + 0.38 * lexical + due_bonus, semantic, lexical, goal))
        ranked.sort(key=lambda row: row[0], reverse=True)
        if not ranked:
            return None
        best_score, best_semantic, best_lexical, best = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -1.0
        if best_lexical <= 0.0 and best_semantic < 0.38:
            return None
        if best_score - second_score < 0.025 and best_lexical == 0.0:
            return None
        return best

    @classmethod
    def _canonical_fact_content(cls, predicate: str, value: str) -> str:
        templates = {
            "name": "My name is {value}.",
            "location": "I live in {value}.",
            "bike": "My bike is {value}.",
            "favorite_color": "My favorite color is {value}.",
            "studying_language": "I am studying {value}.",
            "occupation": "I work as {value}.",
            "pet_name": "My dog's name is {value}.",
            "likes": "I like {value}.",
            "prefers": "I prefer {value}.",
        }
        return templates.get(predicate, "{predicate}: {value}.").format(predicate=predicate, value=value)

    @staticmethod
    def _canonical_goal_label(value: str) -> str:
        return " ".join(value.strip().split())[:180]

    @classmethod
    def _content_terms(cls, text: str):
        normalized = cls._normalize(text)
        stop = {
            "the", "a", "an", "to", "i", "it", "that", "this", "my", "me", "do", "did",
            "el", "la", "los", "las", "un", "una", "de", "del", "al", "que", "ya", "he",
            "como", "para", "por", "con", "tengo", "task", "done", "marca", "mark", "cancel",
        }
        return {token for token in re.findall(r"[a-z0-9]+", normalized) if len(token) > 2 and token not in stop}

    @classmethod
    def _values_match(cls, left: str, right: str) -> bool:
        a, b = cls._normalize_value(left), cls._normalize_value(right)
        if a == b:
            return True
        if a in b or b in a:
            return min(len(a), len(b)) >= 4
        a_terms, b_terms = cls._content_terms(a), cls._content_terms(b)
        if not a_terms or not b_terms:
            return False
        return len(a_terms & b_terms) / max(1, min(len(a_terms), len(b_terms))) >= 0.6

    @classmethod
    def _same_value(cls, left: str, right: str) -> bool:
        return cls._normalize_value(left) == cls._normalize_value(right)

    @staticmethod
    def _normalize_value(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9áéíóúüñ]+", value.lower()))

    @staticmethod
    def _safe_embed(embedder, text: str):
        if embedder is None:
            return None
        try:
            return list(embedder.embed(text))
        except Exception:
            return None

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
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
