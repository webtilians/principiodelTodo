from src.infinito3.generalized_context_builder import GeneralizedContextBuilder
from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord


def _builder():
    return GeneralizedContextBuilder(InMemoryMemoryStore(), SimpleGoalEngine())


def _user_fact(text, predicate, value):
    return MemoryRecord(
        content=text,
        kind=MemoryKind.USER_MODEL,
        importance=0.7,
        confidence=1.0,
        fact_subject="user",
        fact_predicate=predicate,
        fact_value=value,
    )


def test_generic_multi_value_query_keeps_same_predicate_without_domain_dictionary():
    builder = _builder()
    candidates = [
        _user_fact("Me gusta la astrofotografía.", "likes", "la astrofotografía"),
        _user_fact("Me gusta la escalada.", "likes", "la escalada"),
        _user_fact("Vivo en Cádiz.", "location", "cadiz"),
    ]

    packet = builder.build(
        "¿Qué aficiones me gustan? Incluye todo lo que te he contado sobre ellas.",
        memory_candidates=candidates,
    )

    assert "astrofotografía" in packet.rendered
    assert "escalada" in packet.rendered
    assert "Cádiz" not in packet.rendered


def test_multiple_requested_core_facts_are_preserved_independently():
    builder = _builder()
    candidates = [
        _user_fact("Me llamo Irene.", "name", "irene"),
        _user_fact("Vivo en Burgos.", "location", "burgos"),
        _user_fact("Me gusta el café.", "likes", "el cafe"),
    ]

    packet = builder.build(
        "¿Cómo me llamo y en qué ciudad vivo?",
        memory_candidates=candidates,
    )

    assert "Irene" in packet.rendered
    assert "Burgos" in packet.rendered
    assert "café" not in packet.rendered


def test_irrelevant_urgent_goal_is_not_injected_into_profile_query():
    goals = SimpleGoalEngine()
    goals.ingest("Mañana tengo que comprar pan a las 8.")
    builder = GeneralizedContextBuilder(InMemoryMemoryStore(), goals)
    candidates = [
        _user_fact("Mi bici es una Canyon Sender.", "bike", "canyon sender"),
    ]

    packet = builder.build("¿Qué bici uso?", memory_candidates=candidates)

    assert "Canyon Sender" in packet.rendered
    assert "comprar pan" not in packet.rendered


def test_self_contained_math_drops_personal_context():
    builder = _builder()
    candidates = [
        _user_fact("Me llamo Alba.", "name", "alba"),
        _user_fact("Me gusta el blues.", "likes", "el blues"),
    ]

    packet = builder.build("¿Cuánto es 2 + 2?", memory_candidates=candidates)

    assert packet.items == []
    assert "Alba" not in packet.rendered
    assert "blues" not in packet.rendered
