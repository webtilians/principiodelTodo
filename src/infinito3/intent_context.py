"""Opt-in integration of the shared intent contract; legacy builders stay intact."""
from dataclasses import asdict
from datetime import datetime

from .context_intent import resolve_context_intent
from .preference_context import PreferenceStateContextBuilder
from .semantic_event_extractor import SemanticCognitiveEventExtractor
from .types import ContextItem, ContextSource


class IntentContextBuilder(PreferenceStateContextBuilder):
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
        if intent.window:
            start, end = intent.window
            return [item for item in items if item.metadata.get("due_at")
                    and start <= datetime.fromisoformat(item.metadata["due_at"]) < end]
        return super()._filter_goals_for_temporal_intent(query, items)

    def _build_pools(self, query, candidates, recent_turns):
        pools = super()._build_pools(query, candidates, recent_turns)
        intent = self.resolve_intent(query)
        if intent.mode == "goals":
            # Positive closure evidence prevents equating an empty active list
            # with an inaccessible or unknown calendar. Scope to matching goals.
            terms = self._content_words(query) - self._content_words("commitment task goal still open have any")
            for goal in self.goal_engine.all():
                if not goal.completed or not terms & self._content_words(goal.description):
                    continue
                content = f"[GOAL STATE] {goal.description}; status={goal.metadata.get('lifecycle', 'completed')}"
                pools[ContextSource.GOAL].append(ContextItem(
                    source=ContextSource.GOAL, content=content, score=0.99,
                    estimated_tokens=self.token_estimator.estimate(content), goal_id=goal.id,
                    metadata={"closed_goal": True, "due_at": goal.due_at.isoformat() if goal.due_at else None}))
        return pools

    def _select_preference_items(self, query, items):
        # A successful empty semantic classification is authoritative, not a
        # request to fall back to unrelated embedding neighbours.
        if self.reranker is None or len(items) <= 1:
            return super()._select_preference_items(query, items)
        if self._is_historical_preference_query(query):
            matched = self._historical_evidence_match(query, items)
            if matched: return matched
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
