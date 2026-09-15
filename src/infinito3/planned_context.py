from .cognitive_query import build_cognitive_query_plan
from .intent_context_v4 import DeterministicIntentContextBuilder, IntentEventExtractor


class PlannedIntentContextBuilder(DeterministicIntentContextBuilder):
    def resolve_plan(self, query):
        return build_cognitive_query_plan(query, self._now_fn())

    def resolve_intent(self, query):
        return self.resolve_plan(query).intent

    def build(self, query, **kwargs):
        packet = super().build(query, **kwargs)
        packet.diagnostics["cognitive_query_plan"] = self.resolve_plan(query).to_dict()
        return packet


__all__ = ["PlannedIntentContextBuilder", "IntentEventExtractor"]
