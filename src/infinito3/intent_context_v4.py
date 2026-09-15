"""Post-V9 deterministic context candidate."""

from .intent_context import IntentContextBuilder, IntentEventExtractor


class DeterministicIntentContextBuilder(IntentContextBuilder):
    pass


__all__ = ["DeterministicIntentContextBuilder", "IntentEventExtractor"]
