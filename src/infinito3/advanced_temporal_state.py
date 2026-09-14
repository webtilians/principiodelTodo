from .goal_resolver import StateAwareGoalResolver
from .semantic_temporal_state import SemanticTemporalCognitiveState


class ResolvedTemporalCognitiveState(SemanticTemporalCognitiveState):
    """Semantic temporal reducer with explicit state-aware goal resolution."""

    def __init__(self, now_fn, *, goal_resolver=None):
        super().__init__(now_fn=now_fn)
        self.goal_resolver = goal_resolver or StateAwareGoalResolver()

    def _match_goal(self, event, goals, *, memory_store=None):
        provider = getattr(memory_store, "embedding_provider", None) if memory_store is not None else None
        return self.goal_resolver.resolve(event, goals, embedding_provider=provider)
