"""Post-V9 deterministic context candidate."""
import re
from datetime import datetime

from .intent_context import IntentContextBuilder, IntentEventExtractor
from .types import MemoryStatus


class DeterministicIntentContextBuilder(IntentContextBuilder):
    def _preference_state_candidates(self, query):
        intent = self.resolve_intent(query)
        if intent.ordering != "latest":
            return super()._preference_state_candidates(query)

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
        suppressed_ids = {
            target.id for target in resolved.values()
            if target is not None and target.status == MemoryStatus.ACTIVE
        }
        return [record for record in active if record.id not in suppressed_ids]

    @staticmethod
    def _semantic_membership_query(query):
        focused = re.sub(
            r"\b(?:most recently|most recent|latest|newest|recently)\b",
            " ", query, flags=re.I,
        )
        focused = re.sub(
            r"\b(?:did i|have i)\s+(?:add(?:ed)?|take(?:n)? up|start(?:ed)?|begin|began)\b",
            " ", focused, flags=re.I,
        )
        return " ".join(focused.split()) or query

    def _latest_selected_item(self, items):
        if len(items) <= 1:
            if items:
                items[0].metadata["structured_preference_order"] = "latest"
            return list(items)
        by_id = {record.id: record for record in self._all_with_inactive()}
        ranked = sorted(
            items,
            key=lambda item: self._valid_from(by_id[item.memory_id])
            if item.memory_id in by_id else datetime.min,
            reverse=True,
        )
        ranked[0].metadata["structured_preference_order"] = "latest"
        return ranked[:1]


__all__ = ["DeterministicIntentContextBuilder", "IntentEventExtractor"]
