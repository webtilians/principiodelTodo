import re
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from .semantic_context import SemanticCohortContextBuilder
from .types import ContextItem, ContextSource, MemoryRecord, MemoryStatus


class TemporalSemanticContextBuilder(SemanticCohortContextBuilder):
    """Semantic Context Builder with explicit temporal-lineage support."""

    def build(self, query: str, *, memory_candidates: Optional[Sequence[MemoryRecord]] = None,
              recent_turns=None, max_tokens: int = 1200, candidate_k: int = 20):
        candidates = list(memory_candidates) if memory_candidates is not None else None
        if candidates is not None and self._is_history_query(query):
            candidates = self._historical_candidates(query, candidates)
        return super().build(query, memory_candidates=candidates, recent_turns=recent_turns,
                             max_tokens=max_tokens, candidate_k=candidate_k)

    @classmethod
    def _requested_core_facts(cls, query: str) -> set:
        """Extend structured intent beyond the original five profile facets.

        These are predicate/query forms, not domain values. The same mechanism
        therefore works for any language value, occupation, pet name, etc.
        """
        requested = set(super()._requested_core_facts(query))
        q = " ".join(cls._normalized(query).split())

        if re.search(r"\b(call me|called me|should you call me|llamabas|me llamaban)\b", q):
            requested.add("name")
        if re.search(r"\b(vivia|vivi|lived|used to live)\b", q) and re.search(r"\b(donde|where|ciudad|city)\b", q):
            requested.add("location")
        if (
            re.search(r"\b(idioma|language|lengua)\b", q)
            and re.search(r"\b(estudi\w*|study\w*|learning|aprend\w*)\b", q)
        ):
            requested.add("studying_language")
        if re.search(r"\b(profesion|profesion|trabajo|occupation|job|work)\b", q) and re.search(r"\b(mi|my|soy|i)\b", q):
            requested.add("occupation")
        if re.search(r"\b(perro|dog|mascota|pet)\b", q) and re.search(r"\b(nombre|name|llama|called)\b", q):
            requested.add("pet_name")
        return requested

    def _apply_precision_filter(self, query: str, pools: Dict[ContextSource, List[ContextItem]]) -> Tuple[Dict[ContextSource, List[ContextItem]], int]:
        originals = {source: list(items) for source, items in pools.items()}
        filtered, _ = super()._apply_precision_filter(query, pools)

        requested = self._requested_core_facts(query)
        asks_goals = self._asks_for_goals(query)
        if requested and not asks_goals:
            query_words = self._content_words(query)
            facet_words = self._content_words(self._FACET_HINT_TEXT)
            additional_words = query_words - facet_words
            for source in (ContextSource.USER_MODEL, ContextSource.MEMORY):
                existing_ids = {item.memory_id for item in filtered[source]}
                for item in originals[source]:
                    if item.memory_id in existing_ids:
                        continue
                    predicate = str(item.metadata.get("fact_predicate") or "")
                    if not predicate:
                        continue
                    if predicate in requested or additional_words & self._content_words(item.content):
                        item.metadata["mixed_structured_fact"] = True
                        filtered[source].append(item)
                        existing_ids.add(item.memory_id)
                filtered[source].sort(key=lambda item: item.score, reverse=True)

        before = sum(len(items) for items in originals.values())
        after = sum(len(items) for items in filtered.values())
        return filtered, max(0, before - after)

    def _historical_candidates(self, query: str, candidates: Sequence[MemoryRecord]) -> List[MemoryRecord]:
        all_records = self._all_with_inactive()
        by_id = {record.id: record for record in all_records}
        query_norm = self._normalized(query)
        direct_predecessors = []
        target_predicates = set()

        for record in all_records:
            if record.status != MemoryStatus.ACTIVE or not record.supersedes_id:
                continue
            value = self._normalized(record.fact_value or "")
            if value and value in query_norm:
                predecessor = by_id.get(record.supersedes_id)
                if predecessor is not None:
                    direct_predecessors.append(self._as_historical_evidence(predecessor))
                    if record.fact_predicate:
                        target_predicates.add(record.fact_predicate)

        if direct_predecessors:
            rest = [
                self._as_historical_evidence(record)
                for record in candidates
                if record.fact_predicate not in target_predicates
            ]
            seen = set()
            ordered = []
            for record in direct_predecessors + rest:
                if record.id in seen:
                    continue
                seen.add(record.id)
                ordered.append(record)
            return ordered

        return [self._as_historical_evidence(record) for record in candidates]

    def _all_with_inactive(self) -> List[MemoryRecord]:
        getter = getattr(self.memory_store, "all", None)
        if getter is None:
            return []
        try:
            return list(getter(include_inactive=True))
        except TypeError:
            return list(getter())

    @staticmethod
    def _as_historical_evidence(record: MemoryRecord) -> MemoryRecord:
        if record.status == MemoryStatus.ACTIVE:
            return record
        metadata = dict(record.metadata)
        metadata["historical_original_status"] = record.status.value
        return replace(record, status=MemoryStatus.ACTIVE, importance=max(record.importance, 0.96), metadata=metadata)

    @classmethod
    def _is_history_query(cls, query: str) -> bool:
        q = cls._normalized(query)
        markers = (
            "antes de", "antes del", "anterior", "anteriormente", "usaba antes",
            "vivia antes", "llamabas antes", "ya no me gusta", "he dicho explicitamente que ya no",
            "before ", "previous", "previously", "used to", "no longer like", "historical",
        )
        return any(marker in q for marker in markers)
