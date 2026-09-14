import re
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from .semantic_context import SemanticCohortContextBuilder
from .temporal import parse_weekday_range
from .types import ContextItem, ContextSource, MemoryKind, MemoryRecord, MemoryStatus


class TemporalSemanticContextBuilder(SemanticCohortContextBuilder):
    """Semantic Context Builder with structured predicates and temporal lineage."""

    def build(self, query: str, *, memory_candidates: Optional[Sequence[MemoryRecord]] = None,
              recent_turns=None, max_tokens: int = 1200, candidate_k: int = 20):
        candidates = list(memory_candidates) if memory_candidates is not None else None
        if candidates is not None and self._is_history_query(query):
            candidates = self._historical_candidates(query, candidates)
        return super().build(query, memory_candidates=candidates, recent_turns=recent_turns,
                             max_tokens=max_tokens, candidate_k=candidate_k)

    @classmethod
    def _requested_core_facts(cls, query: str) -> set:
        requested = set(super()._requested_core_facts(query))
        q = " ".join(cls._normalized(query).split())

        # A structured profile query should be resolved by predicate, not by
        # accidental lexical similarity between the question and stored values.
        profile_scope = bool(
            re.search(
                r"\b(?:mi|mis|my|current|currently|actual|actuales|ahora|recuerda|remember|profile|perfil)\b",
                q,
            )
        )
        if profile_scope:
            if re.search(r"\b(?:nombre|name)\b", q):
                requested.add("name")
            if re.search(r"\b(?:ciudad|city|location|ubicacion|residencia|residence|home)\b", q):
                requested.add("location")
            if re.search(r"\b(?:bici|bicicleta|bike|bicycle|montura|ride)\b", q):
                requested.add("bike")
            if re.search(r"\b(?:idioma|language|lengua)\b", q):
                requested.add("studying_language")
            if re.search(r"\b(?:profesion|profession|occupation|job|trabajo|empleo|work)\b", q):
                requested.add("occupation")
            if re.search(r"\b(?:mascota|pet|perro|dog|gato|cat|loro|parrot)\b", q):
                requested.add("pet_name")
            if re.search(r"\b(?:color)\b", q) and re.search(r"\b(?:favorit|favorite|favourite)\w*\b", q):
                requested.add("favorite_color")

        if re.search(r"\b(call me|called me|should you call me|llamabas|me llamaban)\b", q):
            requested.add("name")
        if re.search(r"\b(?:como me llamo|como prefiero que me llames|nombre preferido|preferred name)\b", q):
            requested.add("name")
        if re.search(r"\b(vivia|vivi|lived|used to live)\b", q) and re.search(r"\b(donde|where|ciudad|city)\b", q):
            requested.add("location")
        if re.search(r"\b(idioma|language|lengua)\b", q) and re.search(r"\b(estudi\w*|study\w*|learning|aprend\w*)\b", q):
            requested.add("studying_language")
        if re.search(r"\b(profesion|trabajo|occupation|job|work|empleo)\b", q) and re.search(r"\b(mi|my|soy|i|actual|current)\b", q):
            requested.add("occupation")
        if re.search(r"\b(?:a que me dedico|what do i do for work|what is my profession)\b", q):
            requested.add("occupation")
        if re.search(r"\b(perro|dog|mascota|pet|gato|cat|loro|parrot)\b", q) and re.search(r"\b(nombre|name|llama|called)\b", q):
            requested.add("pet_name")
        if "frase de prueba" in q or "test phrase" in q or "verification phrase" in q or "frase de verificacion" in q:
            requested.add("test_phrase")
        return requested

    @classmethod
    def _asks_for_goals(cls, query: str) -> bool:
        if super()._asks_for_goals(query):
            return True
        q = " ".join(cls._normalized(query).split())

        # Generic commitment vocabulary.  These are relation words, not domain
        # values, so they generalize to arbitrary appointments/tasks.
        if re.search(
            r"\b(?:commitment\w*|compromiso\w*|appointment\w*|cita\w*|pending|pendiente\w*|to-do\w*)\b",
            q,
        ):
            return True

        # Elliptical follow-ups are common after a list of commitments.
        if re.search(
            r"\b(?:what|which|que)\b.{0,28}\b(?:remain\w*|remaining|left|open|queda\w*|abiert\w*)\b",
            q,
        ):
            return True
        if re.search(r"\b(?:still pending|still open|sigue\w* pendiente\w*|sigue\w* abiert\w*)\b", q):
            return True
        return False

    def _build_pools(self, query, candidates, recent_turns):
        pools = super()._build_pools(query, candidates, recent_turns)
        requested = self._requested_core_facts(query)

        # Structured predicates are authoritative retrieval keys. If semantic
        # retrieval misses a requested field because its value shares no words
        # with the query, recover every active requested slot directly.
        if requested:
            seen = {
                item.memory_id
                for source in (ContextSource.USER_MODEL, ContextSource.MEMORY)
                for item in pools[source]
                if item.memory_id
            }
            active = [r for r in self._all_with_inactive() if r.status == MemoryStatus.ACTIVE]
            for record in active:
                if not record.id or record.id in seen or record.fact_predicate not in requested:
                    continue
                item = self._memory_item(query, record, rank=0, total=1)
                source = ContextSource.USER_MODEL if record.kind == MemoryKind.USER_MODEL else ContextSource.MEMORY
                item.metadata["predicate_fallback"] = True
                item.metadata["structured_state_retrieval"] = True
                pools[source].append(item)
                seen.add(record.id)

        # Historical predecessor evidence must not be polluted by the current
        # version of the same predicate.
        if self._is_history_query(query):
            historical_ids = {
                record.id
                for record in candidates
                if record.metadata.get("historical_original_status")
            }
            historical_predicates = {
                record.fact_predicate
                for record in candidates
                if record.metadata.get("historical_original_status") and record.fact_predicate
            }
            if historical_predicates:
                for source in (ContextSource.USER_MODEL, ContextSource.MEMORY):
                    pools[source] = [
                        item for item in pools[source]
                        if item.metadata.get("fact_predicate") not in historical_predicates
                        or item.memory_id in historical_ids
                    ]

        for source in pools:
            pools[source].sort(key=lambda item: item.score, reverse=True)
        return pools

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

    def _filter_goals_for_temporal_intent(self, query: str, items: List[ContextItem]) -> List[ContextItem]:
        """Extend calendar filtering with explicit ranges and before-weekday bounds."""
        now = self._now_fn()
        focus = self._question_focus(query)
        q = " ".join(self._normalized(focus).split())

        weekday_range = parse_weekday_range(focus, now)
        if weekday_range is not None:
            start_date, end_date = weekday_range
            return [
                item for item in items
                if self._due_date(item) is not None and start_date <= self._due_date(item) <= end_date
            ]

        # "before Monday" / "antes del lunes" means a strict upper bound, not
        # the Monday itself.  Resolve the weekday with the shared parser in the
        # superclass path, then compare dates here.
        before_match = re.search(
            r"\b(?:before|antes de(?:l)?)\s+(lunes|monday|martes|tuesday|miercoles|wednesday|jueves|thursday|viernes|friday|sabado|saturday|domingo|sunday)\b",
            q,
        )
        if before_match:
            from .temporal import parse_weekday_date
            target = parse_weekday_date(before_match.group(1), now)
            if target is not None:
                return [
                    item for item in items
                    if self._due_date(item) is not None and self._due_date(item) < target
                ]

        return super()._filter_goals_for_temporal_intent(query, items)

    @staticmethod
    def _due_date(item: ContextItem):
        from datetime import datetime
        raw_due = item.metadata.get("due_at")
        if not raw_due:
            return None
        try:
            return datetime.fromisoformat(str(raw_due)).date()
        except (TypeError, ValueError):
            return None

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
                    direct_predecessors.append(
                        self._as_historical_evidence(predecessor, relation="immediately_previous")
                    )
                    if record.fact_predicate:
                        target_predicates.add(record.fact_predicate)
        if direct_predecessors:
            rest = [
                self._as_historical_evidence(record)
                for record in candidates
                if record.fact_predicate not in target_predicates
            ]
            seen, ordered = set(), []
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
    def _as_historical_evidence(record: MemoryRecord, relation: Optional[str] = None) -> MemoryRecord:
        metadata = dict(record.metadata)
        if record.status != MemoryStatus.ACTIVE:
            metadata["historical_original_status"] = record.status.value
        if relation:
            metadata["temporal_relation"] = relation
            predicate = record.fact_predicate or "fact"
            value = record.fact_value or record.content
            content = (
                f"[TEMPORAL HISTORY] relation={relation}; predicate={predicate}; "
                f"value={value}. Original evidence: {record.content}"
            )
        else:
            content = record.content
        return replace(
            record,
            content=content,
            status=MemoryStatus.ACTIVE,
            importance=max(record.importance, 0.96),
            metadata=metadata,
        )

    @classmethod
    def _is_history_query(cls, query: str) -> bool:
        q = cls._normalized(query)
        markers = (
            "antes de", "antes del", "anterior", "anteriormente", "usaba antes",
            "vivia antes", "llamabas antes", "ya no", "he dicho explicitamente que ya no",
            "before ", "previous", "previously", "used to", "no longer", "historical",
            "lost interest", "stopped ", "dropped ",
        )
        return any(marker in q for marker in markers)
