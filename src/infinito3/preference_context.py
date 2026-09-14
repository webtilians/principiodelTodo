import re
from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from .generalized_context_builder import GeneralizedContextBuilder
from .temporal_context import TemporalSemanticContextBuilder
from .types import ContextItem, ContextPacket, ContextSource, MemoryRecord, MemoryStatus


class PreferenceStateContextBuilder(TemporalSemanticContextBuilder):
    """Resolve preference questions from explicit current/retracted state.

    Preference facets are a poor fit for ordinary top-k retrieval because all
    facts share one structured predicate (`likes`) while the requested facet can
    be arbitrary. For preference questions this builder therefore starts from
    the complete preference state, reconciles durable retraction tombstones, and
    uses the existing semantic membership reranker over that bounded cohort.

    No activity/domain vocabulary is encoded here.
    """

    _PREFERENCE_TERMS = (
        "hobby", "hobbies", "activity", "activities", "interest", "interests",
        "preference", "preferences", "aficion", "aficiones", "actividad", "actividades",
        "preferencia", "preferencias", "me gusta", "gustan", "enjoy", "enjoying",
    )
    _REQUEST_START = re.compile(
        r"^(?:what|which|name|tell me|do i|did i|cu[aá]l|cu[aá]les|qu[eé]|dime|nombra|recuerda)\b",
        re.I,
    )
    _RECENCY_MARKERS = (
        "newer", "new", "recent", "recently added", "after dropping", "after i dropped",
        "nueva", "nuevo", "reciente", "despues de dejar", "después de dejar",
        "despues de abandonar", "después de abandonar",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preference_query_active = False

    def build(self, query: str, *, memory_candidates: Optional[Sequence[MemoryRecord]] = None,
              recent_turns=None, max_tokens: int = 1200, candidate_k: int = 20) -> ContextPacket:
        if not self._is_preference_query(query):
            return super().build(
                query,
                memory_candidates=memory_candidates,
                recent_turns=recent_turns,
                max_tokens=max_tokens,
                candidate_k=candidate_k,
            )

        self._preference_query_active = True
        self._reranker_events = []
        try:
            candidates = self._preference_state_candidates(query)
            packet = GeneralizedContextBuilder.build(
                self,
                query,
                memory_candidates=candidates,
                recent_turns=recent_turns,
                max_tokens=max_tokens,
                candidate_k=max(candidate_k, len(candidates)),
            )
            packet.diagnostics["preference_state"] = {
                "candidate_count": len(candidates),
                "historical": self._is_historical_preference_query(query),
                "recency_constrained": self._is_recency_preference_query(query),
            }
            if self._reranker_events:
                packet.diagnostics["semantic_reranker"] = self._summarize_reranker_events()
            return packet
        finally:
            self._preference_query_active = False

    @classmethod
    def _is_preference_query(cls, query: str) -> bool:
        q = " ".join(cls._normalized(query).split())
        if not ("?" in query or "¿" in query or cls._REQUEST_START.search(query.strip())):
            return False
        return any(term in q for term in cls._PREFERENCE_TERMS)

    @classmethod
    def _is_historical_preference_query(cls, query: str) -> bool:
        q = " ".join(cls._normalized(query).split())
        return any(
            marker in q
            for marker in (
                "no longer", "lost interest", "stopped", "dropped", "retracted", "used to",
                "ya no", "deje", "dejé", "abandone", "abandoné", "histor", "antes me gust",
            )
        )

    @classmethod
    def _is_recency_preference_query(cls, query: str) -> bool:
        q = " ".join(cls._normalized(query).split())
        return any(marker in q for marker in cls._RECENCY_MARKERS)

    def _preference_state_candidates(self, query: str) -> List[MemoryRecord]:
        records = self._all_with_inactive()
        active = [
            record for record in records
            if record.status == MemoryStatus.ACTIVE and record.fact_predicate == "likes"
        ]
        tombstones = [
            record for record in records
            if record.status == MemoryStatus.ACTIVE
            and bool(record.metadata.get("retraction_tombstone"))
            and record.metadata.get("retracted_predicate") == "likes"
        ]
        by_id = {record.id: record for record in records}
        resolved = self._resolve_preference_tombstones(tombstones, active, by_id)

        if self._is_historical_preference_query(query):
            historical = []
            for tombstone in tombstones:
                target = resolved.get(tombstone.id)
                if target is not None:
                    historical.append(self._historical_preference_record(target, tombstone))
                else:
                    historical.append(self._unresolved_retraction_record(tombstone))
            return historical

        suppressed_ids = {
            target.id for target in resolved.values()
            if target is not None and target.status == MemoryStatus.ACTIVE
        }
        current = [record for record in active if record.id not in suppressed_ids]

        if self._is_recency_preference_query(query) and tombstones:
            cutoff = max(self._retraction_time(tombstone) for tombstone in tombstones)
            current = [record for record in current if self._valid_from(record) >= cutoff]
        return current

    def _resolve_preference_tombstones(
        self,
        tombstones: Sequence[MemoryRecord],
        active: Sequence[MemoryRecord],
        by_id: Dict[str, MemoryRecord],
    ) -> Dict[str, Optional[MemoryRecord]]:
        result: Dict[str, Optional[MemoryRecord]] = {}
        for tombstone in tombstones:
            resolved_ids = list(tombstone.metadata.get("resolved_memory_ids") or [])
            resolved_record = next((by_id.get(str(memory_id)) for memory_id in resolved_ids if by_id.get(str(memory_id))), None)
            if resolved_record is not None:
                result[tombstone.id] = resolved_record
                continue

            target_value = self._normalized(tombstone.fact_value or "")
            lexical = [
                record for record in active
                if self._normalized(record.fact_value or "") == target_value
                or (
                    target_value
                    and self._normalized(record.fact_value or "")
                    and (
                        target_value in self._normalized(record.fact_value or "")
                        or self._normalized(record.fact_value or "") in target_value
                    )
                )
            ]
            if len(lexical) == 1:
                result[tombstone.id] = lexical[0]
                continue

            result[tombstone.id] = self._semantic_retraction_target(tombstone, active)
        return result

    def _semantic_retraction_target(
        self, tombstone: MemoryRecord, active: Sequence[MemoryRecord]
    ) -> Optional[MemoryRecord]:
        if self.reranker is None or not active:
            return None
        items = [self._memory_item("preference retraction target", record, rank=index, total=len(active))
                 for index, record in enumerate(active)]
        query = (
            "Select only the remembered preference that denotes the same activity or item as "
            f"this retracted preference: {tombstone.fact_value or tombstone.content}"
        )
        reranked = self.reranker.rerank(query, items)
        self._record_preference_reranker_event(reranked, len(items), kind="retraction_target")
        if not reranked.success or len(reranked.selected_ids) != 1:
            return None
        selected = str(reranked.selected_ids[0])
        return next((record for record in active if str(record.id) == selected), None)

    def _apply_precision_filter(self, query, pools):
        if not self._preference_query_active:
            return super()._apply_precision_filter(query, pools)

        before = sum(len(items) for items in pools.values())
        preference_items = [
            item
            for source in (ContextSource.USER_MODEL, ContextSource.MEMORY)
            for item in pools[source]
            if item.metadata.get("fact_predicate") == "likes"
            or item.metadata.get("preference_history")
        ]
        selected = self._select_preference_items(query, preference_items)
        filtered = {source: [] for source in pools}
        for item in selected:
            filtered[item.source].append(item)
        return filtered, max(0, before - len(selected))

    def _select_preference_items(self, query: str, items: Sequence[ContextItem]) -> List[ContextItem]:
        if not items:
            return []
        if len(items) == 1:
            return list(items)

        # Historical questions often quote the user's retraction language rather
        # than the original activity name ("no longer appeals", "lost interest",
        # "isn't my thing anymore").  Tombstones preserve that exact evidence.
        # When one candidate has a clear lexical match on at least two meaningful
        # terms, that evidence is more authoritative than semantic facet ranking
        # and avoids asking a model to choose among several equally retracted facts.
        if self._is_historical_preference_query(query):
            evidence_match = self._historical_evidence_match(query, items)
            if evidence_match:
                evidence_match[0].metadata["preference_history_evidence_match"] = True
                return evidence_match

        if self.reranker is not None:
            reranked = self.reranker.rerank(query, items)
            self._record_preference_reranker_event(reranked, len(items), kind="preference_membership")
            if reranked.success and reranked.selected_ids:
                selected_ids = {str(memory_id) for memory_id in reranked.selected_ids}
                selected = [item for item in items if str(item.memory_id) in selected_ids]
                for item in selected:
                    item.metadata["preference_state_selected"] = True
                return selected

        scorer = getattr(self.memory_store, "semantic_scores", None)
        if scorer is None:
            return list(items[:1])
        focus = self._semantic_focus_query(query)
        semantic = scorer(focus, [item.memory_id for item in items if item.memory_id])
        ordered = sorted(items, key=lambda item: semantic.get(str(item.memory_id), -1.0), reverse=True)
        if not ordered:
            return []
        if self._query_allows_multiple(query):
            top = semantic.get(str(ordered[0].memory_id), 0.0)
            floor = max(0.16, top * 0.62, top - 0.22)
            return [item for item in ordered if semantic.get(str(item.memory_id), 0.0) >= floor]
        return ordered[:1]

    @classmethod
    def _historical_evidence_match(
        cls, query: str, items: Sequence[ContextItem]
    ) -> List[ContextItem]:
        query_terms = cls._content_words(query)
        if not query_terms:
            return []

        ranked = []
        for item in items:
            evidence = str(item.metadata.get("retraction_source_text") or item.content)
            overlap = query_terms & cls._content_words(evidence)
            ranked.append((len(overlap), item.score, item))
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        if not ranked or ranked[0][0] < 2:
            return []
        second_overlap = ranked[1][0] if len(ranked) > 1 else 0
        if ranked[0][0] <= second_overlap:
            return []
        return [ranked[0][2]]

    def _record_preference_reranker_event(self, reranked, candidate_count: int, *, kind: str) -> None:
        self._reranker_events.append({
            "success": bool(reranked.success),
            "candidate_count": int(candidate_count),
            "embedding_selected_count": 0,
            "selected_count": len(reranked.selected_ids),
            "provider": reranked.provider,
            "model": reranked.model,
            "input_tokens": int(reranked.usage.get("input_tokens") or 0),
            "output_tokens": int(reranked.usage.get("output_tokens") or 0),
            "total_tokens": int(reranked.usage.get("total_tokens") or 0),
            "error": reranked.error,
            "kind": kind,
        })

    @staticmethod
    def _historical_preference_record(target: MemoryRecord, tombstone: MemoryRecord) -> MemoryRecord:
        metadata = dict(target.metadata)
        evidence = str(tombstone.metadata.get("retraction_source_text") or tombstone.content)
        metadata.update({
            "preference_history": True,
            "preference_status": "retracted",
            "retraction_source_text": evidence,
            "fact_value": f"{target.fact_value or target.content}; retraction evidence: {evidence}",
        })
        return replace(
            target,
            status=MemoryStatus.ACTIVE,
            content=(
                f"[PREFERENCE HISTORY] status=retracted; value={target.fact_value or target.content}; "
                f"retraction_evidence={evidence}"
            ),
            importance=max(target.importance, 0.99),
            metadata=metadata,
        )

    @staticmethod
    def _unresolved_retraction_record(tombstone: MemoryRecord) -> MemoryRecord:
        metadata = dict(tombstone.metadata)
        metadata.update({
            "preference_history": True,
            "preference_status": "retracted",
            "fact_predicate": "likes",
            "fact_value": tombstone.content,
        })
        return replace(
            tombstone,
            fact_predicate="likes",
            status=MemoryStatus.ACTIVE,
            importance=max(tombstone.importance, 0.99),
            metadata=metadata,
        )

    @staticmethod
    def _retraction_time(record: MemoryRecord) -> datetime:
        raw = record.metadata.get("retraction_at")
        if raw:
            try:
                return datetime.fromisoformat(str(raw))
            except ValueError:
                pass
        return record.updated_at

    @staticmethod
    def _valid_from(record: MemoryRecord) -> datetime:
        raw = record.metadata.get("temporal_valid_from")
        if raw:
            try:
                return datetime.fromisoformat(str(raw))
            except ValueError:
                pass
        return record.created_at
