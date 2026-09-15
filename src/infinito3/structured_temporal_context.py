import re
from dataclasses import replace
from datetime import datetime
from typing import Dict, List, Sequence

from .temporal_context import TemporalSemanticContextBuilder
from .types import ContextItem, ContextSource, MemoryStatus


class StructuredTemporalContextBuilder(TemporalSemanticContextBuilder):
    """Temporal context with explicit state-oriented query handling.

    This layer stays domain-neutral: it recognizes relations such as preference,
    recency, retraction, arithmetic, and calendar scope.  Category membership
    (creative, outdoor, radio-related, etc.) remains delegated to embeddings and
    the injectable semantic membership reranker.
    """

    _PREFERENCE_QUERY_RE = re.compile(
        r"\b(?:preference\w*|preferencia\w*|hobb(?:y|ies)|aficion\w*|activity|activities|"
        r"actividad\w*|interest\w*|interes\w*|enjoy\w*|disfrut\w*|fond of|taken up)\b",
        re.I,
    )
    _RECENT_AFTER_RETRACTION_RE = re.compile(
        r"\b(?:newer|new|recent|recently|nuevo\w*|reciente\w*)\b.*\b(?:after|despues|tras|"
        r"drop\w*|stopp\w*|retract\w*|dejar\w*)\b|"
        r"\b(?:after|despues|tras)\b.*\b(?:drop\w*|stopp\w*|retract\w*|dejar\w*)\b",
        re.I,
    )
    _RETRACTION_HISTORY_RE = re.compile(
        r"\b(?:no longer|lost interest|stopped liking|doesn't appeal|does not appeal|"
        r"ya no|deje de|dejé de|perdi el interes|perdí el interés|retracted|retir(?:e|é))\b",
        re.I,
    )

    @classmethod
    def _self_contained_math(cls, query: str) -> bool:
        if super()._self_contained_math(query):
            return True
        q = cls._normalized(query)
        word_operation = re.search(
            r"\b\d+\s+(?:mas|menos|por|entre|dividido\s+entre|multiplied\s+by|times|plus|minus|divided\s+by)\s+\d+\b",
            q,
        )
        return bool(word_operation and not re.search(r"\b(mi|mis|my|nuestro|nuestra)\b", q))

    @classmethod
    def _query_allows_multiple(cls, query: str) -> bool:
        if super()._query_allows_multiple(query):
            return True
        q = " ".join(cls._normalized(query).split())
        return bool(
            re.search(
                r"\b(?:activities|hobbies|interests|preferences|actividades|aficiones|intereses|preferencias)\b",
                q,
            )
        )

    def _build_pools(self, query, candidates, recent_turns):
        pools = super()._build_pools(query, candidates, recent_turns)
        if self._is_preference_query(query) and not self._is_retraction_history_query(query):
            all_records = self._all_with_inactive()
            active = [
                record for record in all_records
                if record.status == MemoryStatus.ACTIVE
                and record.fact_predicate in {"likes", "prefers"}
                and record.id
            ]
            if self._is_recent_after_retraction_query(query):
                latest_retraction = self._latest_retraction_recorded_at(all_records)
                if latest_retraction is not None:
                    active = [record for record in active if record.created_at > latest_retraction]

            allowed_ids = {record.id for record in active}
            if self._is_recent_after_retraction_query(query):
                for source in (ContextSource.USER_MODEL, ContextSource.MEMORY):
                    pools[source] = [
                        item for item in pools[source]
                        if item.metadata.get("fact_predicate") not in {"likes", "prefers"}
                        or item.memory_id in allowed_ids
                    ]

            seen = {
                item.memory_id
                for source in (ContextSource.USER_MODEL, ContextSource.MEMORY)
                for item in pools[source]
                if item.memory_id
            }
            for record in active:
                if record.id in seen:
                    continue
                item = self._memory_item(query, record, rank=0, total=1)
                item.metadata["structured_preference_retrieval"] = True
                source = ContextSource.USER_MODEL if record.kind.value == "user_model" else ContextSource.MEMORY
                pools[source].append(item)
                seen.add(record.id)

        for source in pools:
            pools[source].sort(key=lambda item: item.score, reverse=True)
        return pools

    def _prune_memory_pool(self, query: str, items: Sequence[ContextItem]) -> List[ContextItem]:
        if not items:
            return []
        preference_items = [
            item for item in items
            if item.metadata.get("fact_predicate") in {"likes", "prefers"} and item.memory_id
        ]
        if (
            self.reranker is not None
            and self._is_preference_query(query)
            and not self._query_allows_multiple(query)
            and len(preference_items) >= 2
        ):
            reranked = self.reranker.rerank(self._membership_focus(query), preference_items)
            self._reranker_events.append(
                {
                    "success": bool(reranked.success),
                    "candidate_count": len(preference_items),
                    "embedding_selected_count": 0,
                    "selected_count": len(reranked.selected_ids),
                    "provider": reranked.provider,
                    "model": reranked.model,
                    "input_tokens": int(reranked.usage.get("input_tokens") or 0),
                    "output_tokens": int(reranked.usage.get("output_tokens") or 0),
                    "total_tokens": int(reranked.usage.get("total_tokens") or 0),
                    "error": reranked.error,
                }
            )
            if reranked.success and reranked.selected_ids:
                selected_ids = set(reranked.selected_ids)
                selected = [item for item in preference_items if str(item.memory_id) in selected_ids]
                for item in selected:
                    item.metadata["semantic_reranker_selected"] = True
                    item.metadata["singular_preference_query"] = True
                return selected
        return super()._prune_memory_pool(query, items)

    def _filter_goals_for_temporal_intent(self, query: str, items: List[ContextItem]) -> List[ContextItem]:
        focus = self._question_focus(query)
        q = " ".join(self._normalized(focus).split())
        now = self._now_fn()
        if re.search(r"\b(?:today|hoy)\b", q):
            selected = [item for item in items if self._due_date(item) == now.date()]
            if re.search(r"\b(?:this morning|esta manana)\b", q):
                selected = [item for item in selected if self._due_hour(item) is not None and self._due_hour(item) < 12]
            return selected
        return super()._filter_goals_for_temporal_intent(query, items)

    def _historical_candidates(self, query: str, candidates):
        records = super()._historical_candidates(query, candidates)
        if not self._is_retraction_history_query(query):
            return records
        rendered = []
        for record in records:
            source = str(record.metadata.get("retraction_source_text") or "").strip()
            if not source:
                rendered.append(record)
                continue
            metadata = dict(record.metadata)
            metadata["temporal_relation"] = "explicit_retraction"
            value = record.fact_value or record.content
            rendered.append(
                replace(
                    record,
                    content=(
                        f"[TEMPORAL RETRACTION] predicate={record.fact_predicate or 'fact'}; "
                        f"value={value}; user_retraction={source}"
                    ),
                    status=MemoryStatus.ACTIVE,
                    importance=max(record.importance, 0.98),
                    metadata=metadata,
                )
            )
        return rendered

    @classmethod
    def _is_preference_query(cls, query: str) -> bool:
        return bool(cls._PREFERENCE_QUERY_RE.search(cls._normalized(query)))

    @classmethod
    def _is_recent_after_retraction_query(cls, query: str) -> bool:
        return bool(cls._RECENT_AFTER_RETRACTION_RE.search(cls._normalized(query)))

    @classmethod
    def _is_retraction_history_query(cls, query: str) -> bool:
        return bool(cls._RETRACTION_HISTORY_RE.search(cls._normalized(query)))

    @staticmethod
    def _latest_retraction_recorded_at(records):
        moments = []
        for record in records:
            raw = record.metadata.get("retraction_recorded_at")
            if not raw:
                continue
            try:
                moments.append(datetime.fromisoformat(str(raw)))
            except (TypeError, ValueError):
                continue
        return max(moments) if moments else None

    @classmethod
    def _membership_focus(cls, query: str) -> str:
        focus = cls._semantic_focus_query(query)
        focus = re.sub(
            r"\b(?:newer|new|recent|recently|after|before|dropping|dropped|old|current|currently|"
            r"still|added|add|nuevo\w*|reciente\w*|despues|tras|actual\w*)\b",
            " ",
            focus,
            flags=re.I,
        )
        return " ".join(focus.split()) or query

    @staticmethod
    def _due_hour(item: ContextItem):
        raw = item.metadata.get("due_at")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw)).hour
        except (TypeError, ValueError):
            return None
