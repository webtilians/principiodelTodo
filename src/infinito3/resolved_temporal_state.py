from typing import Dict, Optional

from .semantic_temporal_state import SemanticTemporalCognitiveState
from .types import ContextItem, ContextSource, MemoryStatus


class ResolvedSemanticTemporalState(SemanticTemporalCognitiveState):
    """Temporal reducer with a conservative second-stage retraction resolver.

    Exact/lexical/embedding store matching remains authoritative and cheap.  Only
    if that path cannot identify a target do we ask an injected semantic
    membership resolver to choose among ACTIVE facts of the same predicate.
    Exactly one selected id is required; ambiguity fails closed.
    """

    def __init__(self, *args, retraction_reranker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.retraction_reranker = retraction_reranker
        self._retraction_stats: Dict[str, int] = {
            "calls": 0,
            "successes": 0,
            "resolved": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

    def retraction_resolution_stats(self) -> Dict[str, int]:
        return dict(self._retraction_stats)

    def _apply_retraction(self, event, transition, memory_store) -> None:
        retractor = getattr(memory_store, "retract_fact", None) if memory_store is not None else None
        affected = []
        if callable(retractor):
            affected = list(
                retractor(
                    event.subject,
                    event.predicate,
                    self._normalize_value(event.value or ""),
                    reason=event.type.value,
                    source_text=event.source_text,
                    at=event.occurred_at,
                )
                or []
            )

        if not affected:
            affected = self._resolve_retraction_target(event, memory_store)

        if affected:
            affected_set = set(affected)
            versions = self._facts.setdefault((event.subject, event.predicate), [])
            for version in versions:
                if version.active and version.memory_id in affected_set:
                    version.active = False
                    version.retracted = True
                    version.valid_to = event.occurred_at
                    transition.closed_version_ids.append(version.id)
            transition.memory_ids.extend(affected)
            return

        # Preserve the transparent base behavior (including diagnostic notes)
        # when no resolver can identify one target safely.
        super(SemanticTemporalCognitiveState, self)._apply_retraction(event, transition, memory_store)

    def _resolve_retraction_target(self, event, memory_store) -> list:
        if self.retraction_reranker is None or memory_store is None:
            return []
        closer = getattr(memory_store, "retract_memory_id", None)
        getter = getattr(memory_store, "all", None)
        if not callable(closer) or not callable(getter):
            return []

        candidates = [
            record
            for record in getter()
            if record.status == MemoryStatus.ACTIVE
            and record.fact_subject == event.subject
            and record.fact_predicate == event.predicate
            and record.id
        ]
        if not candidates:
            return []

        items = [
            ContextItem(
                source=ContextSource.USER_MODEL,
                content=record.content,
                score=0.0,
                estimated_tokens=0,
                memory_id=record.id,
                metadata={
                    "fact_predicate": record.fact_predicate,
                    "fact_value": record.fact_value,
                },
            )
            for record in candidates
        ]
        self._retraction_stats["calls"] += 1
        result = self.retraction_reranker.rerank(event.source_text, items)
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            self._retraction_stats[key] += int(result.usage.get(key) or 0)
        if not result.success:
            return []
        self._retraction_stats["successes"] += 1
        if len(result.selected_ids) != 1:
            return []

        affected = list(
            closer(
                result.selected_ids[0],
                reason=event.type.value,
                source_text=event.source_text,
                at=event.occurred_at,
                resolver="llm_semantic_membership",
            )
            or []
        )
        if affected:
            self._retraction_stats["resolved"] += 1
        return affected
