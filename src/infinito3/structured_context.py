import re

from .temporal_context import TemporalSemanticContextBuilder
from .types import ContextSource, MemoryKind


class StructuredTemporalContextBuilder(TemporalSemanticContextBuilder):
    """Temporal context builder that renders state semantics explicitly."""

    @classmethod
    def _asks_for_goals(cls, query: str) -> bool:
        if super()._asks_for_goals(query):
            return True
        q = cls._normalized(query)
        return bool(
            re.search(
                r"\b(compromiso\w*|cita\w*|agenda|appointment\w*|commitment\w*|schedule\w*)\b",
                q,
            )
            or any(
                marker in q
                for marker in (
                    "que tengo hoy", "qué tengo hoy", "que tengo esta semana", "qué tengo esta semana",
                    "what do i have today", "what do i have this week", "what is still open",
                    "que sigue abierto", "qué sigue abierto",
                )
            )
        )

    def _historical_candidates(self, query, candidates):
        if any(
            record.metadata.get("structured_state")
            and record.metadata.get("temporal_relation") == "immediately_previous"
            for record in candidates
        ):
            return list(candidates)
        return super()._historical_candidates(query, candidates)

    def _build_pools(self, query, candidates, recent_turns):
        pools = super()._build_pools(query, candidates, recent_turns)
        historical_predicates = {
            record.fact_predicate
            for record in candidates
            if record.fact_predicate
            and record.metadata.get("structured_state")
            and record.metadata.get("temporal_relation") == "immediately_previous"
        }
        if historical_predicates:
            for source in (ContextSource.USER_MODEL, ContextSource.MEMORY):
                pools[source] = [
                    item
                    for item in pools[source]
                    if item.metadata.get("fact_predicate") not in historical_predicates
                    or item.metadata.get("temporal_relation") == "immediately_previous"
                ]
        return pools

    def _memory_item(self, query, memory, rank, total):
        item = super()._memory_item(query, memory, rank, total)
        relation = memory.metadata.get("temporal_relation")
        structured = bool(memory.metadata.get("structured_state"))
        predicate = memory.fact_predicate
        value = memory.metadata.get("structured_display_value") or memory.fact_value

        if relation == "immediately_previous" and predicate and value:
            before = memory.metadata.get("temporal_before_value") or "unknown"
            item.content = self._sanitize(
                f"TEMPORAL FACT | predicate={predicate} | value={value} | "
                f"relation=immediately_previous | before={before}"
            )
            item.metadata["fact_predicate"] = predicate
            item.metadata["temporal_relation"] = relation
            item.metadata["temporal_before_value"] = before
            item.metadata["structured_state"] = True
            item.score = max(item.score, 0.99)
            item.estimated_tokens = self.token_estimator.estimate(item.content)
            return item

        if structured and predicate and value:
            label = "CURRENT USER STATE" if memory.kind == MemoryKind.USER_MODEL else "STORED STRUCTURED DATA"
            item.content = self._sanitize(
                f"{label} | predicate={predicate} | value={value} | relation=current"
            )
            item.metadata["fact_predicate"] = predicate
            item.metadata["temporal_relation"] = "current"
            item.metadata["structured_state"] = True
            item.score = max(item.score, 0.98)
            item.estimated_tokens = self.token_estimator.estimate(item.content)
        return item
