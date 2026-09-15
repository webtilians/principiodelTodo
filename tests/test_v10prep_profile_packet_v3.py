from src.infinito3.intent_context_v4 import DeterministicIntentContextBuilder
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.types import MemoryKind, MemoryRecord


def test_profile_packet_contains_each_requested_slot():
    query = "Read back my name, home city, bicycle, language and occupation."
    store = InMemoryMemoryStore()
    values = (("name", "Noor"), ("location", "Turin"), ("bike", "Norco Sight"), ("studying_language", "Danish"), ("occupation", "paper conservator"))
    for predicate, value in values:
        store.add(MemoryRecord(f"{predicate}: {value}", MemoryKind.USER_MODEL, .95, fact_predicate=predicate, fact_value=value))
    packet = DeterministicIntentContextBuilder(store, TemporalGoalEngine()).build(query, memory_candidates=[], max_tokens=600)
    assert all(value in packet.rendered for _, value in values)
