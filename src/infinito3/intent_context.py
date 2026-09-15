"""Opt-in integration of the shared intent contract; legacy builders stay intact."""
from dataclasses import asdict
from datetime import datetime

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
        elif intent.goal_status == "open":
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
        if intent.mode == "goals" and intent.goal_status in ("closed", "any"):
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
    def extract(self, text):
        intent = resolve_context_intent(text, self._now_fn())
        if not intent.retrieve:
            self._stats.skipped_non_mutating_requests += 1
            return []
        return super().extract(text)
