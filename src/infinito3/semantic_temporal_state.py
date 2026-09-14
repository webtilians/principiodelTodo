from .temporal_state import TemporalCognitiveState
from .types import MemoryKind, MemoryRecord


class SemanticTemporalCognitiveState(TemporalCognitiveState):
    """Temporal reducer that reconciles semantic retractions with durable lineage.

    A retraction is itself persistent evidence. Even when the current store cannot
    confidently resolve a bilingual/paraphrased target, we keep a tombstone for
    the retraction event. Read-time preference-state resolution can then reconcile
    that tombstone against active values without silently forgetting the user's
    explicit negative update.
    """

    def _apply_retraction(self, event, transition, memory_store) -> None:
        affected = []
        retractor = getattr(memory_store, "retract_fact", None) if memory_store is not None else None
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
                self._store_retraction_tombstone(event, transition, memory_store, resolved_ids=affected)
                return

        # In-memory stores and unresolved semantic targets retain the transparent
        # lexical behavior of the base reducer. The tombstone is written after
        # the base attempt so it cannot be mistaken for the target being closed.
        super()._apply_retraction(event, transition, memory_store)
        self._store_retraction_tombstone(event, transition, memory_store, resolved_ids=[])

    def _store_retraction_tombstone(self, event, transition, memory_store, *, resolved_ids) -> None:
        if memory_store is None or not event.predicate or not event.value:
            return
        add = getattr(memory_store, "add", None)
        if not callable(add):
            return

        predicate = f"retracted:{event.predicate}"
        normalized_value = self._normalize_value(event.value)
        record = MemoryRecord(
            content=(
                f"[RETRACTION EVENT] predicate={event.predicate}; value={event.value}; "
                f"evidence={event.source_text}"
            ),
            kind=MemoryKind.SEMANTIC,
            importance=0.9,
            confidence=event.confidence,
            fact_subject=event.subject,
            fact_predicate=predicate,
            fact_value=normalized_value,
            created_at=event.occurred_at,
            updated_at=event.occurred_at,
            metadata={
                "cognitive_event_id": event.id,
                "cognitive_event_type": event.type.value,
                "retraction_tombstone": True,
                "retracted_predicate": event.predicate,
                "retraction_source_text": event.source_text,
                "retraction_at": event.occurred_at.isoformat(),
                "resolved_memory_ids": list(resolved_ids),
                "resolved_at_write": bool(resolved_ids),
            },
        )
        stored = add(record)
        memory_id = getattr(stored, "id", None)
        if memory_id:
            transition.memory_ids.append(memory_id)
