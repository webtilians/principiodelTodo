from .cognitive_events import CognitiveEventType
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

    def _apply_goal_lifecycle(self, event, transition, goal_engine, *, memory_store=None) -> None:
        """Treat rescheduling as a time mutation, not a goal-identity rewrite.

        ``Goal.description`` remains the original auditable identity. A successful
        reschedule also stores a ``canonical_label`` derived from the lifecycle
        target; context rendering uses that label together with the authoritative
        ``due_at``. This prevents an old weekday/time embedded in the original
        sentence from contradicting the new structured due date.
        """
        if event.type != CognitiveEventType.RESCHEDULE_GOAL or goal_engine is None:
            return super()._apply_goal_lifecycle(
                event, transition, goal_engine, memory_store=memory_store
            )

        original_descriptions = {
            goal.id: goal.description
            for goal in goal_engine.all()
            if not goal.completed
        }
        history_start = len(self._goal_history)
        super()._apply_goal_lifecycle(
            event, transition, goal_engine, memory_store=memory_store
        )
        if not transition.goal_ids:
            return

        goals_by_id = {goal.id: goal for goal in goal_engine.all()}
        for goal_id in transition.goal_ids:
            original = original_descriptions.get(goal_id)
            goal = goals_by_id.get(goal_id)
            if original is None or goal is None:
                continue

            candidate = self._canonical_goal_label(event.value or "")
            candidate_terms = self._content_terms(candidate)
            original_terms = self._content_terms(original)
            if candidate and candidate_terms and candidate_terms & original_terms:
                canonical_label = candidate
            else:
                canonical_label = self._canonical_goal_label(original)

            goal.description = original
            goal.metadata["canonical_label"] = canonical_label
            goal.metadata["canonical_due_at"] = goal.due_at.isoformat() if goal.due_at else None
            for version in self._goal_history[history_start:]:
                if version.goal_id == goal_id:
                    version.description = canonical_label

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
