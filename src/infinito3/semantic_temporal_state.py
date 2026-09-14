from .temporal_state import TemporalCognitiveState


class SemanticTemporalCognitiveState(TemporalCognitiveState):
    """Temporal reducer that reconciles semantic store retractions with lineage."""

    def _apply_retraction(self, event, transition, memory_store) -> None:
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
                # The store may resolve an ontology/predicate drift by unique
                # value identity (for example drinks=kombucha -> likes=kombucha).
                # Close whichever temporal version owns the affected memory id,
                # rather than assuming it lives under event.predicate.
                for (subject, _predicate), versions in self._facts.items():
                    if subject != event.subject:
                        continue
                    for version in versions:
                        if version.active and version.memory_id in affected_set:
                            version.active = False
                            version.retracted = True
                            version.valid_to = event.occurred_at
                            transition.closed_version_ids.append(version.id)
                transition.memory_ids.extend(affected)
                return

        # In-memory stores and unresolved semantic targets retain the transparent
        # lexical behavior of the base reducer.
        super()._apply_retraction(event, transition, memory_store)
