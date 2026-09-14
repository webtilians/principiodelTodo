from datetime import datetime
from typing import Optional, Sequence

from .semantic_temporal_state import SemanticTemporalCognitiveState
from .types import Goal


class SemanticGoalTemporalState(SemanticTemporalCognitiveState):
    """Temporal state with lifecycle resolution driven by target semantics + time.

    Event extraction is responsible for reducing free language to a concise goal
    target. Resolution then combines lexical evidence, embedding similarity and
    explicit previous/current due dates. Time is supporting evidence, never proof
    that a goal is complete.
    """

    def _match_goal(self, event, goals: Sequence[Goal], *, memory_store=None) -> Optional[Goal]:
        if not goals:
            return None

        query = event.value or event.source_text
        query_terms = self._content_terms(query)
        embedder = getattr(memory_store, "embedding_provider", None)
        query_embedding = self._safe_embed(embedder, query)
        previous_due = self._parse_metadata_datetime(event.metadata.get("previous_due_at"))

        ranked = []
        for goal in goals:
            goal_terms = self._content_terms(goal.description)
            overlap = len(query_terms & goal_terms)
            lexical = overlap / max(1, len(query_terms | goal_terms))

            semantic = 0.0
            if query_embedding is not None:
                goal_embedding = self._safe_embed(embedder, goal.description)
                if goal_embedding is not None:
                    semantic = max(0.0, self._cosine(query_embedding, goal_embedding))

            temporal = 0.0
            if previous_due is not None and goal.due_at is not None:
                if self._same_slot(previous_due, goal.due_at):
                    temporal += 0.42
                elif previous_due.date() == goal.due_at.date():
                    temporal += 0.24
            elif event.due_at is not None and goal.due_at is not None:
                # For completion/cancellation the referenced target may still
                # name the current due date. For reschedule the new due date is
                # intentionally weak evidence because the stored goal still has
                # the old date.
                if event.type.value != "reschedule_goal" and self._same_slot(event.due_at, goal.due_at):
                    temporal += 0.24

            source_terms = self._content_terms(event.source_text)
            source_overlap = len(source_terms & goal_terms) / max(1, len(source_terms | goal_terms))

            score = 0.50 * semantic + 0.28 * lexical + 0.12 * source_overlap + temporal
            ranked.append((score, semantic, lexical, temporal, source_overlap, goal))

        ranked.sort(key=lambda row: row[0], reverse=True)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        best_score, best_semantic, best_lexical, best_temporal, best_source, goal = best
        second_score = second[0] if second else -1.0

        # Strong old-time evidence may resolve an otherwise terse lifecycle
        # utterance. Otherwise demand semantic/lexical support and a clear lead.
        has_evidence = (
            best_temporal >= 0.40
            or best_semantic >= 0.42
            or best_lexical >= 0.10
            or best_source >= 0.12
        )
        if not has_evidence:
            return None
        if second is not None and best_score - second_score < 0.035:
            if best_temporal < 0.40 and best_lexical < 0.18:
                return None
        return goal

    @staticmethod
    def _parse_metadata_datetime(value):
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None

    @staticmethod
    def _same_slot(left: datetime, right: datetime) -> bool:
        return (
            left.date() == right.date()
            and left.hour == right.hour
            and left.minute == right.minute
        )
