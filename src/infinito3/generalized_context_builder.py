from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from .context_builder import BalancedContextBuilder
from .types import ContextItem, ContextSource


class GeneralizedContextBuilder(BalancedContextBuilder):
    """Context selection without domain-specific vocabularies.

    The policy relies on signals INFINITO already has: structured fact
    predicates, semantic retrieval rank, and explicit query intent. It does not
    encode lists of music genres, foods, or other domain values.
    """

    _GENERIC_MULTI_MARKERS = (
        "todo lo",
        "toda la",
        "todos ",
        "todas ",
        "enumera",
        "enumerame",
        "menciona todos",
        "menciona todas",
        "incluye todo",
        "incluye todos",
        "incluye todas",
        "lista todos",
        "lista todas",
        "all ",
        "everything",
        "list all",
        "mention all",
        "include all",
        "every ",
    )

    _FACET_HINT_TEXT = (
        "nombre name llamo ciudad city vivo live ubicacion location "
        "bici bicicleta bike bicycle modelo model uso use edad age anos old "
        "color favorito favorite colour pendiente tarea objetivo goal task agenda"
    )

    @classmethod
    def _query_allows_multiple(cls, query: str) -> bool:
        if super()._query_allows_multiple(query):
            return True
        normalized = " ".join(cls._normalized(query).split())
        return any(marker in normalized for marker in cls._GENERIC_MULTI_MARKERS)

    def _apply_precision_filter(
        self,
        query: str,
        pools: Dict[ContextSource, List[ContextItem]],
    ) -> Tuple[Dict[ContextSource, List[ContextItem]], int]:
        filtered = {source: list(items) for source, items in pools.items()}
        before = sum(len(items) for items in filtered.values())

        # A self-contained request should not receive unrelated personal state.
        if self._self_contained_math(query):
            return {source: [] for source in filtered}, before

        requested_facts = self._requested_core_facts(query)
        asks_goals = self._asks_for_goals(query)
        query_words = self._content_words(query)
        facet_words = self._content_words(self._FACET_HINT_TEXT)
        additional_words = query_words - facet_words

        # Keep signatures from every goal so an excluded stale goal cannot leak
        # back through its duplicate episodic memory.
        all_goal_signatures = {
            self._content_signature(item.content.split(" | due=", 1)[0])
            for item in filtered[ContextSource.GOAL]
        }
        goal_items = [
            item
            for item in filtered[ContextSource.GOAL]
            if asks_goals
            or bool(
                additional_words
                & self._content_words(item.content.split(" | due=", 1)[0])
            )
        ]
        if asks_goals:
            goal_items = self._filter_goals_for_temporal_intent(query, goal_items)
        filtered[ContextSource.GOAL] = goal_items

        # Goal descriptions can also be stored as episodic memories. Keep one
        # authoritative copy instead of paying for duplicate evidence.
        if all_goal_signatures:
            for source in (ContextSource.USER_MODEL, ContextSource.MEMORY):
                filtered[source] = [
                    item
                    for item in filtered[source]
                    if self._content_signature(item.content) not in all_goal_signatures
                ]

        for source in (ContextSource.USER_MODEL, ContextSource.MEMORY):
            items = filtered[source]

            if requested_facts or asks_goals:
                eligible: List[ContextItem] = []
                for item in items:
                    matching_facts = {
                        fact
                        for fact in requested_facts
                        if self._matches_fact(item, fact)
                    }
                    if matching_facts:
                        item.metadata["requested_facts"] = sorted(matching_facts)
                        eligible.append(item)
                        continue

                    # Episodic memory may still support the non-facet part of a
                    # compound request. Unrequested profile facts are excluded.
                    if source == ContextSource.MEMORY and additional_words:
                        if additional_words & self._content_words(item.content):
                            eligible.append(item)

                filtered[source] = eligible
            else:
                # Semantic retrieval has already ranked the memories. For a
                # multi-value request, keep a coherent cohort of the top
                # structured predicate; for a singular request keep dominant
                # evidence only. This works for arbitrary topics, not just a
                # hand-maintained vocabulary.
                filtered[source] = self._prune_memory_pool(query, items)

        after = sum(len(items) for items in filtered.values())
        return filtered, max(0, before - after)

    def _filter_goals_for_temporal_intent(
        self,
        query: str,
        items: List[ContextItem],
    ) -> List[ContextItem]:
        """Apply only temporal constraints explicitly present in the query.

        Overdue goals remain valid for broad questions such as "what is still
        pending?". They are excluded only when the user asks for a future
        window (tomorrow, day after tomorrow, upcoming/future). This avoids
        silently redefining every overdue task as completed.
        """
        q = " ".join(self._normalized(query).split())
        now = self._now_fn()

        day_after = any(marker in q for marker in ("pasado manana", "day after tomorrow"))
        tomorrow = not day_after and any(marker in q for marker in ("manana", "tomorrow"))
        future_only = bool(
            any(
                marker in q
                for marker in (
                    "futuro", "futura", "futuros", "futuras", "future",
                    "upcoming", "proximo", "proxima", "proximos", "proximas",
                )
            )
        )
        # "Hoy es 18... ¿qué tarea futura...?" uses today only as framing.
        # Explicit future intent therefore takes precedence over the word hoy.
        today = (
            not day_after
            and not tomorrow
            and not future_only
            and any(marker in q for marker in ("hoy", "today"))
        )

        if not (day_after or tomorrow or today or future_only):
            return items

        if day_after:
            target_date = (now + timedelta(days=2)).date()
        elif tomorrow:
            target_date = (now + timedelta(days=1)).date()
        elif future_only:
            target_date = None
        else:
            target_date = now.date()

        selected: List[ContextItem] = []
        for item in items:
            raw_due = item.metadata.get("due_at")
            if not raw_due:
                continue
            try:
                due_at = datetime.fromisoformat(str(raw_due))
            except (TypeError, ValueError):
                continue

            if target_date is not None:
                if due_at.date() == target_date and due_at >= now:
                    selected.append(item)
            elif due_at >= now:
                selected.append(item)

        return selected
