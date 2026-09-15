from dataclasses import replace
from typing import List, Sequence

from .structured_temporal_context import StructuredTemporalContextBuilder
from .types import ContextItem, ContextSource, MemoryStatus


class ExplicitStateContextBuilder(StructuredTemporalContextBuilder):
    """Render negative and historical state as evidence instead of silence.

    Empty active state is information: when a user explicitly asks about goals,
    an empty goal set is rendered as such rather than leaving the answer model to
    infer meaning from an absent section.  Likewise, explicit retractions are
    recoverable from the temporal store even when semantic search of the old
    value would miss them after a language change.
    """

    def _build_pools(self, query, candidates, recent_turns):
        pools = super()._build_pools(query, candidates, recent_turns)

        if self._asks_for_goals(query) and not pools[ContextSource.GOAL]:
            pools[ContextSource.GOAL].append(self._empty_goal_item())

        if self._is_retraction_history_query(query):
            explicit = self._explicit_retraction_items(query)
            if explicit:
                explicit_ids = {item.memory_id for item in explicit}
                for source in (ContextSource.USER_MODEL, ContextSource.MEMORY):
                    pools[source] = [
                        item for item in pools[source]
                        if item.memory_id not in explicit_ids
                    ]
                pools[ContextSource.USER_MODEL].extend(explicit)

        for source in pools:
            pools[source].sort(key=lambda item: item.score, reverse=True)
        return pools

    def _prune_memory_pool(self, query: str, items: Sequence[ContextItem]) -> List[ContextItem]:
        if self._is_retraction_history_query(query):
            explicit = [
                item for item in items
                if item.metadata.get("temporal_relation") == "explicit_retraction"
            ]
            if explicit:
                return explicit
        return super()._prune_memory_pool(query, items)

    def _filter_goals_for_temporal_intent(self, query: str, items: List[ContextItem]) -> List[ContextItem]:
        empty = [item for item in items if item.metadata.get("structured_empty_state")]
        real = [item for item in items if not item.metadata.get("structured_empty_state")]
        if not real and empty:
            return empty
        selected = super()._filter_goals_for_temporal_intent(query, real)
        if selected:
            return selected
        if self._asks_for_goals(query):
            return [self._empty_goal_item()]
        return selected

    def _explicit_retraction_items(self, query: str) -> List[ContextItem]:
        records = [
            record
            for record in self._all_with_inactive()
            if record.status == MemoryStatus.SUPERSEDED
            and record.fact_predicate in {"likes", "prefers"}
            and record.id
            and str(record.metadata.get("retraction_source_text") or "").strip()
        ]
        if not records:
            return []

        query_words = self._content_words(query)
        ranked = []
        for record in records:
            source_text = str(record.metadata.get("retraction_source_text") or "")
            source_words = self._content_words(source_text)
            overlap = len(query_words & source_words)
            union = len(query_words | source_words) or 1
            lexical = overlap / union
            ranked.append((lexical, record))
        ranked.sort(key=lambda pair: (pair[0], pair[1].updated_at), reverse=True)

        # If wording clearly identifies one retraction, keep only that evidence.
        # Otherwise expose a small recent set; the model can reason over the
        # explicit user retraction text without guessing from vector proximity.
        best = ranked[0][0]
        second = ranked[1][0] if len(ranked) > 1 else -1.0
        if best > 0.0 and best - second >= 0.05:
            selected_records = [ranked[0][1]]
        else:
            selected_records = [record for _, record in ranked[:4]]

        items = []
        for record in selected_records:
            source_text = str(record.metadata.get("retraction_source_text") or "")
            value = record.fact_value or record.content
            metadata = dict(record.metadata)
            metadata["temporal_relation"] = "explicit_retraction"
            evidence = replace(
                record,
                content=(
                    f"[TEMPORAL RETRACTION] predicate={record.fact_predicate}; "
                    f"value={value}; user_retraction={source_text}"
                ),
                status=MemoryStatus.ACTIVE,
                importance=max(record.importance, 0.99),
                metadata=metadata,
            )
            item = self._memory_item(query, evidence, rank=0, total=1)
            item.score = max(item.score, 0.99)
            item.metadata["temporal_relation"] = "explicit_retraction"
            item.metadata["retraction_source_text"] = source_text
            items.append(item)
        return items

    def _empty_goal_item(self) -> ContextItem:
        content = "No active goals or commitments."
        return ContextItem(
            source=ContextSource.GOAL,
            content=content,
            score=1.0,
            estimated_tokens=self.token_estimator.estimate(content),
            metadata={"structured_empty_state": True, "state": "no_active_goals"},
        )
