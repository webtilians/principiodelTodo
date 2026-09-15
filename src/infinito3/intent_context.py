"""Opt-in integration of the shared intent contract; legacy builders stay intact."""
import re
from dataclasses import asdict
from datetime import datetime

from .cognitive_events import CognitiveEvent, CognitiveEventType
from .context_intent import detect_history_cue, resolve_context_intent
from .preference_context import PreferenceStateContextBuilder
from .semantic_event_extractor import SemanticCognitiveEventExtractor
from .types import ContextItem, ContextSource


class IntentContextBuilder(PreferenceStateContextBuilder):
    _GOAL_GENERIC_WORDS = (
        "commitment task goal appointment calendar agenda status completed complete closed done finished "
        "cancelled canceled resolved still open pending outstanding remaining unfinished have any is was "
        "the my our did whether or yet what which show list compromiso tarea objetivo cita calendario agenda "
        "estado completado cerrado terminado cancelado sigue abierto pendiente"
    )

    def resolve_intent(self, query):
        return resolve_context_intent(query, self._now_fn())

    @classmethod
    def _is_preference_query(cls, query):
        return resolve_context_intent(query).mode == "preferences"

    @classmethod
    def _is_historical_preference_query(cls, query):
        return resolve_context_intent(query).historical

    @classmethod
    def _is_recency_preference_query(cls, query):
        return resolve_context_intent(query).recent

    @classmethod
    def _is_history_query(cls, query):
        return resolve_context_intent(query).historical

    @classmethod
    def _asks_for_goals(cls, query):
        return resolve_context_intent(query).mode == "goals"

    @classmethod
    def _requested_core_facts(cls, query):
        return set(resolve_context_intent(query).predicates)

    @classmethod
    def _self_contained_math(cls, query):
        return not resolve_context_intent(query).retrieve

    def build(self, query, **kwargs):
        intent = self.resolve_intent(query)
        if not intent.retrieve:
            kwargs.update(memory_candidates=[], recent_turns=[])
        packet = super().build(query, **kwargs)
        data = asdict(intent)
        data["predicates"] = sorted(intent.predicates)
        data["window"] = [x.isoformat() for x in intent.window] if intent.window else None
        packet.diagnostics["context_intent"] = data
        return packet

    def _filter_goals_for_temporal_intent(self, query, items):
        intent = self.resolve_intent(query)
        selected = list(items)
        if intent.goal_status == "closed":
            selected = [item for item in selected if item.metadata.get("closed_goal")]
        elif intent.goal_status == "open" and not intent.goal_status_check:
            selected = [item for item in selected if not item.metadata.get("closed_goal")]
        if intent.window:
            start, end = intent.window
            selected = [
                item for item in selected if item.metadata.get("due_at")
                and start <= datetime.fromisoformat(item.metadata["due_at"]) < end
            ]
            return selected
        if intent.goal_status is not None:
            return selected
        return super()._filter_goals_for_temporal_intent(query, selected)

    def _build_pools(self, query, candidates, recent_turns):
        pools = super()._build_pools(query, candidates, recent_turns)
        intent = self.resolve_intent(query)
        include_closed = intent.goal_status in ("closed", "any") or intent.goal_status_check
        if intent.mode == "goals" and include_closed:
            # Closed state is positive evidence. Match arbitrary goal descriptions
            # using only remaining content words; if none remain, the user asked
            # for the closed cohort as a whole.
            terms = self._content_words(query) - self._content_words(self._GOAL_GENERIC_WORDS)
            for goal in self.goal_engine.all():
                if not goal.completed:
                    continue
                if terms and not terms & self._content_words(goal.description):
                    continue
                content = f"[GOAL STATE] {goal.description}; status={goal.metadata.get('lifecycle', 'completed')}"
                pools[ContextSource.GOAL].append(ContextItem(
                    source=ContextSource.GOAL, content=content, score=0.99,
                    estimated_tokens=self.token_estimator.estimate(content), goal_id=goal.id,
                    metadata={
                        "closed_goal": True,
                        "due_at": goal.due_at.isoformat() if goal.due_at else None,
                    },
                ))
        return pools

    def _select_preference_items(self, query, items):
        # Historical operator wording (lost interest / no longer enjoy / etc.) is
        # structured intent evidence. Use it before any semantic fallback so the
        # reranker is not asked to infer a relation that the query states directly.
        if self._is_historical_preference_query(query) and len(items) > 1:
            intent = self.resolve_intent(query)
            if intent.history_cue and intent.history_cue != "predecessor":
                cue_matches = []
                for item in items:
                    evidence = str(item.metadata.get("retraction_source_text") or item.content)
                    if detect_history_cue(evidence) == intent.history_cue:
                        cue_matches.append(item)
                if len(cue_matches) == 1:
                    cue_matches[0].metadata["preference_history_cue_match"] = intent.history_cue
                    return cue_matches
                if cue_matches:
                    items = cue_matches
            matched = self._historical_evidence_match(query, items)
            if matched:
                return matched

        # A successful empty semantic classification is authoritative, not a
        # request to fall back to unrelated embedding neighbours.
        if self.reranker is None or len(items) <= 1:
            return super()._select_preference_items(query, items)
        result = self.reranker.rerank(query, items)
        self._record_preference_reranker_event(result, len(items), kind="preference_membership")
        if result.success:
            selected = set(map(str, result.selected_ids))
            return [item for item in items if str(item.memory_id) in selected]
        return []  # explicit abstention on classifier failure


class IntentEventExtractor(SemanticCognitiveEventExtractor):
    """Semantic extractor with deterministic disambiguation for intent-bearing mutations.

    The V7 mixed-state audit exposed two lexical collisions in the older rule
    layer: first-person residence moves were mistaken for goal reschedules, and
    lifecycle target cleanup removed short scaffolding tokens inside real words.
    Literal note commands also need to remain data even when their payload itself
    contains words such as ``prior``. These cases are resolved before semantic
    fallback and without entity dictionaries.
    """

    _LOCATION_MOVE_PATTERNS = (
        re.compile(r"\bi(?:'ve| have)? moved from\s+([^,.!?;]+?)\s+to\s+([^,.!?;]+)", re.I),
        re.compile(r"\bme he mudado de\s+([^,.!?;]+?)\s+a\s+([^,.!?;]+)", re.I),
    )
    _LITERAL_NOTE_PATTERNS = (
        re.compile(
            r"\b(?:store|save|keep|remember)\s+(?:this|the following)\s+"
            r"(?:literal\s+)?(?:verification|test)\s+phrase(?:\s+as\s+data)?\s*:\s*(.+)$",
            re.I,
        ),
        re.compile(
            r"\b(?:guarda|conserva|recuerda)\s+(?:esta|la siguiente)\s+frase\s+"
            r"(?:literal\s+)?de\s+(?:verificacion|prueba)(?:\s+como\s+datos?)?\s*:\s*(.+)$",
            re.I,
        ),
    )
    _GOAL_SCAFFOLDING = (
        "mark that task done", "mark it done", "mark it complete", "already",
        "cancela", "cancel", "cancelar", "anula", "anular", "ya", "he", "fui",
        "marca", "marcalo", "como hecho", "como hecha", "lo han movido", "ya no es",
        "moved to", "moved from", "reschedule", "rescheduled", "this morning",
        "i", "the", "task", "today", "hoy",
    )
    _CALENDAR_WORD_RE = re.compile(
        r"\b(?:lunes|martes|miercoles|jueves|viernes|sabado|domingo|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|"
        r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
        re.I,
    )

    def extract(self, text):
        stripped = " ".join(text.strip().split())
        if not stripped:
            return []

        note = self._literal_note_event(stripped)
        if note is not None:
            return [note]

        move = self._location_move_event(stripped)
        if move is not None:
            return [move]

        intent = resolve_context_intent(stripped, self._now_fn())
        if not intent.retrieve:
            self._stats.skipped_non_mutating_requests += 1
            return []
        return super().extract(stripped)

    def _literal_note_event(self, text):
        for pattern in self._LITERAL_NOTE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            value = self._clean_value(match.group(1))
            if not value:
                return None
            return CognitiveEvent(
                CognitiveEventType.STORE_NOTE,
                text,
                predicate="literal_verification_phrase",
                value=value,
                occurred_at=self._now_fn(),
                metadata={"extractor": "intent_rule_v3", "instruction_like_data": True},
            )
        return None

    def _location_move_event(self, text):
        for pattern in self._LOCATION_MOVE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            previous_value = self._clean_value(match.group(1))
            value = self._clean_value(match.group(2))
            if not value:
                return None
            return CognitiveEvent(
                CognitiveEventType.REPLACE_FACT,
                text,
                predicate="location",
                value=value,
                previous_value=previous_value or None,
                occurred_at=self._now_fn(),
                metadata={"exclusive": True, "extractor": "intent_rule_v3"},
            )
        return None

    @classmethod
    def _goal_target(cls, text, normalized, *, mode):
        # Lifecycle scaffolding must be removed as lexical units. Plain
        # ``str.replace`` corrupts real words (e.g. the token ``i`` inside
        # ``light``), which V7 caught in a rescheduled goal description.
        cleaned = normalized
        for marker in sorted(cls._GOAL_SCAFFOLDING, key=len, reverse=True):
            cleaned = re.sub(
                rf"(?<!\w){re.escape(marker)}(?!\w)",
                " ",
                cleaned,
                flags=re.I,
            )
        cleaned = cls._CALENDAR_WORD_RE.sub(" ", cleaned)
        cleaned = re.sub(r"\b\d{1,2}(?::\d{2})?\b", " ", cleaned)
        cleaned = " ".join(token for token in cleaned.split() if len(token) > 2)
        return cleaned[:180].strip() or text[:180].strip()
