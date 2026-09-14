from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.semantic_context import (
    SemanticCohortContextBuilder,
    SemanticScoringSQLiteMemoryStore,
)
from src.infinito3.types import MemoryKind, MemoryRecord


class FakeSemanticEmbeddingProvider:
    def embed(self, text):
        normalized = text.lower()
        if "aire libre" in normalized:
            return [1.0, 0.0, 0.0]
        if "preferencias" in normalized:
            return [0.58, 0.58, 0.58]
        if "senderismo" in normalized:
            return [0.98, 0.05, 0.0]
        if "escalada" in normalized:
            return [0.94, 0.10, 0.0]
        if "kayak" in normalized:
            return [0.90, 0.12, 0.02]
        if "ajedrez" in normalized:
            return [0.20, 0.96, 0.0]
        if "jazz" in normalized:
            return [0.12, 0.98, 0.0]
        if "curry" in normalized:
            return [0.08, 0.12, 0.98]
        return [0.33, 0.33, 0.33]


def _store():
    return SemanticScoringSQLiteMemoryStore(
        path=":memory:",
        embedding_provider=FakeSemanticEmbeddingProvider(),
    )


def _add_like(store, text):
    store.add(
        MemoryRecord(
            content=text,
            kind=MemoryKind.USER_MODEL,
            importance=0.7,
            confidence=1.0,
        )
    )


def test_semantic_cohort_removes_same_predicate_distractors_without_domain_dictionary():
    store = _store()
    for text in (
        "Me gusta hacer senderismo.",
        "Me gusta la escalada.",
        "Me gusta navegar en kayak.",
        "Me gusta jugar al ajedrez.",
        "Me gusta escuchar jazz.",
        "Me gusta cocinar curry.",
    ):
        _add_like(store, text)

    builder = SemanticCohortContextBuilder(store, SimpleGoalEngine())
    packet = builder.build(
        "¿Qué actividades al aire libre te he dicho que me gustan? Incluye todas las que recuerdes.",
        memory_candidates=store.all(),
    )

    assert "senderismo" in packet.rendered
    assert "escalada" in packet.rendered
    assert "kayak" in packet.rendered
    assert "ajedrez" not in packet.rendered
    assert "jazz" not in packet.rendered
    assert "curry" not in packet.rendered
    assert all("semantic_focus_score" in item.metadata for item in packet.items)


def test_semantic_cohort_keeps_heterogeneous_values_for_genuinely_broad_query():
    store = _store()
    for text in (
        "Me gusta hacer senderismo.",
        "Me gusta escuchar jazz.",
        "Me gusta cocinar curry.",
    ):
        _add_like(store, text)

    builder = SemanticCohortContextBuilder(store, SimpleGoalEngine())
    packet = builder.build(
        "Cuéntame todas mis preferencias que recuerdes.",
        memory_candidates=store.all(),
    )

    assert "senderismo" in packet.rendered
    assert "jazz" in packet.rendered
    assert "curry" in packet.rendered
