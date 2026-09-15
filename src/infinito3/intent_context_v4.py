"""Post-V9 deterministic context candidate."""
from datetime import datetime

from .intent_context import IntentContextBuilder, IntentEventExtractor
from .preference_ordering import membership_query
from .types import MemoryStatus


class DeterministicIntentContextBuilder(IntentContextBuilder):
    def _preference_state_candidates(self, query):
        intent = self.resolve_intent(query)
        if intent.ordering != "latest":
            return super()._preference_state_candidates(query)
        records = self._all_with_inactive()
        active = [record for record in records if record.status == MemoryStatus.ACTIVE and record.fact_predicate == "likes"]
        tombstones = [record for record in records if record.status == MemoryStatus.ACTIVE and bool(record.metadata.get("retraction_tombstone")) and record.metadata.get("retracted_predicate") == "likes"]
        by_id = {record.id: record for record in records}
        resolved = self._resolve_preference_tombstones(tombstones, active, by_id)
        suppressed_ids = {target.id for target in resolved.values() if target is not None and target.status == MemoryStatus.ACTIVE}
        return [record for record in active if record.id not in suppressed_ids]

    def _select_preference_items(self, query, items):
        intent = self.resolve_intent(query)
        semantic_query = membership_query(query) if intent.ordering == "latest" else query
        selected = super()._select_preference_items(semantic_query, items)
        if intent.ordering != "latest" or len(selected) <= 1:
            if intent.ordering == "latest" and selected:
                selected[0].metadata["structured_preference_order"] = "latest"
            return selected
        by_id = {record.id: record for record in self._all_with_inactive()}
        selected.sort(key=lambda item: self._valid_from(by_id[item.memory_id]) if item.memory_id in by_id else datetime.min, reverse=True)
        selected[0].metadata["structured_preference_order"] = "latest"
        return selected[:1]


__all__ = ["DeterministicIntentContextBuilder", "IntentEventExtractor"]
