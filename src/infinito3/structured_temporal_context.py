from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from .semantic_interpreter import StateQueryPlan
from .temporal_context import TemporalSemanticContextBuilder
from .types import ContextItem, ContextSource, MemoryKind, MemoryRecord, MemoryStatus


class StructuredTemporalContextBuilder(TemporalSemanticContextBuilder):
    """Temporal context builder with direct structured-state retrieval.

    Semantic/vector retrieval remains useful for open-ended memories and
    preferences. Explicit slot questions bypass top-k uncertainty: requested
    predicates are read directly from temporal state / structured memory. History
    is rendered with its relationship, not as an ambiguous present-tense fact.
    """

    def __init__(self, *args, temporal_state=None, query_analyzer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_state = temporal_state
        self.query_analyzer = query_analyzer
        self._active_plan = StateQueryPlan()

    def build(self, query: str, *, memory_candidates: Optional[Sequence[MemoryRecord]] = None,
              recent_turns=None, max_tokens: int = 1200, candidate_k: int = 20):
        self._active_plan = self.query_analyzer.analyze(query) if self.query_analyzer is not None else StateQueryPlan()
        candidates = list(memory_candidates) if memory_candidates is not None else list(
            self.memory_store.search(query, top_k=candidate_k)
        )
        candidates = self._inject_structured_candidates(query, candidates, self._active_plan)
        packet = super().build(query, memory_candidates=candidates, recent_turns=recent_turns,
                               max_tokens=max_tokens, candidate_k=candidate_k)
        packet.diagnostics["structured_state"] = {
            "predicates": list(self._active_plan.predicates),
            "history": [
                {"predicate": item.predicate, "before_value": item.before_value}
                for item in self._active_plan.history
            ],
            "asks_goals": self._active_plan.asks_goals,
            "query_confidence": self._active_plan.confidence,
        }
        if self.query_analyzer is not None:
            packet.diagnostics["state_query_analyzer_usage"] = self.query_analyzer.usage_summary()
        return packet

    def _inject_structured_candidates(self, query: str, candidates: Sequence[MemoryRecord],
                                      plan: StateQueryPlan) -> List[MemoryRecord]:
        records = self._all_records()
        by_id = {record.id: record for record in records}
        ordered: List[MemoryRecord] = []
        seen = set()

        for history_request in plan.history:
            record, display_value = self._history_record(
                history_request.predicate, history_request.before_value, records, by_id
            )
            if record is not None:
                rendered = self._historical_render(
                    record, history_request.before_value, display_value=display_value
                )
                metadata = dict(record.metadata)
                metadata.update({
                    "structured_state_selected": True,
                    "structured_history_selected": True,
                    "temporal_relation": "immediately_previous" if history_request.before_value else "historical",
                    "before_value": history_request.before_value,
                })
                structured = replace(record, content=rendered, status=MemoryStatus.ACTIVE,
                                     importance=max(record.importance, 0.99), metadata=metadata)
                ordered.append(structured)
                seen.add(record.id)

        historical_predicates = {request.predicate for request in plan.history}
        for predicate in plan.predicates:
            if predicate in historical_predicates:
                continue
            record, display_value = self._current_record(predicate, records)
            if record is None or record.id in seen:
                continue
            metadata = dict(record.metadata)
            metadata.update({"structured_state_selected": True, "temporal_relation": "current"})
            value = display_value or record.fact_value or record.content
            structured = replace(
                record,
                content=f"CURRENT FACT | predicate={predicate} | value={value}",
                importance=max(record.importance, 0.99),
                metadata=metadata,
            )
            ordered.append(structured)
            seen.add(record.id)

        for record in candidates:
            if record.id in seen:
                continue
            if record.fact_predicate in historical_predicates and record.status == MemoryStatus.ACTIVE:
                continue
            ordered.append(record)
            seen.add(record.id)
        return ordered

    def _apply_precision_filter(self, query: str, pools: Dict[ContextSource, List[ContextItem]]) -> Tuple[Dict[ContextSource, List[ContextItem]], int]:
        originals = {source: list(items) for source, items in pools.items()}
        filtered, _ = super()._apply_precision_filter(query, pools)
        if not self._self_contained_math(query):
            for source in (ContextSource.USER_MODEL, ContextSource.MEMORY):
                existing = {item.memory_id for item in filtered[source]}
                for item in originals[source]:
                    if not item.metadata.get("structured_state_selected") or item.memory_id in existing:
                        continue
                    filtered[source].append(item)
                    existing.add(item.memory_id)
            if self._active_plan.asks_goals:
                filtered[ContextSource.GOAL] = self._filter_goals_for_temporal_intent(
                    query, list(originals[ContextSource.GOAL])
                )
        for source in filtered:
            filtered[source].sort(key=lambda item: item.score, reverse=True)
        before = sum(len(items) for items in originals.values())
        after = sum(len(items) for items in filtered.values())
        return filtered, max(0, before - after)

    def _current_record(self, predicate: str, records: Sequence[MemoryRecord]):
        if self.temporal_state is not None:
            current = self.temporal_state.current_fact(predicate)
            if current is not None and current.memory_id:
                for record in records:
                    if record.id == current.memory_id:
                        return record, current.value
        candidates = [record for record in records
                      if record.status == MemoryStatus.ACTIVE and record.fact_predicate == predicate]
        record = max(candidates, key=lambda r: (r.updated_at, r.created_at), default=None)
        return record, None

    def _history_record(self, predicate: str, before_value: Optional[str], records, by_id):
        if self.temporal_state is not None:
            versions = self.temporal_state.fact_history(predicate)
            if before_value:
                target = next((v for v in reversed(versions) if self._same_value(v.value, before_value)), None)
                if target is not None and target.supersedes_version_id:
                    predecessor = next((v for v in versions if v.id == target.supersedes_version_id), None)
                    if predecessor is not None and predecessor.memory_id:
                        return by_id.get(predecessor.memory_id), predecessor.value
            inactive = [version for version in versions if not version.active and version.memory_id]
            if inactive:
                return by_id.get(inactive[-1].memory_id), inactive[-1].value
        candidates = [record for record in records
                      if record.fact_predicate == predicate and record.status != MemoryStatus.ACTIVE]
        record = max(candidates, key=lambda r: (r.updated_at, r.created_at), default=None)
        return record, None

    @staticmethod
    def _historical_render(record: MemoryRecord, before_value: Optional[str], *, display_value: Optional[str] = None) -> str:
        value = display_value or record.fact_value or record.content
        if before_value:
            return (f"TEMPORAL FACT | predicate={record.fact_predicate} | value={value} | "
                    f"relation=immediately_previous | before={before_value} | status=historical")
        return f"TEMPORAL FACT | predicate={record.fact_predicate} | value={value} | status=historical"

    def _all_records(self) -> List[MemoryRecord]:
        getter = getattr(self.memory_store, "all", None)
        if getter is None:
            return []
        try:
            return list(getter(include_inactive=True))
        except TypeError:
            return list(getter())

    @staticmethod
    def _same_value(left: str, right: str) -> bool:
        normalize = lambda value: " ".join(str(value).lower().strip().split())
        return normalize(left) == normalize(right)
