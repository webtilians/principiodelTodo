from datetime import datetime, timedelta

from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.preference_context import PreferenceStateContextBuilder
from src.infinito3.semantic_context import SemanticScoringSQLiteMemoryStore
from src.infinito3.semantic_reranker import SemanticRerankResult
from src.infinito3.types import MemoryKind, MemoryRecord


class FlatEmbeddingProvider:
    def embed(self, text):
        return [0.5, 0.5, 0.5]


class PreferenceReranker:
    def rerank(self, query, candidates):
        q = query.lower()
        selected = []
        for item in candidates:
            content = item.content.lower()
            keep = False
            if "same activity or item" in q and "birdwatch" in q:
                keep = "aves" in content
            elif "outdoor" in q:
                keep = "senderismo nocturno" in content
            elif "drink" in q:
                keep = "rooibos" in content
            elif "no longer appeals" in q or "retracted" in q:
                keep = "aves" in content or "birdwatch" in content
            elif "newer creative" in q or "after dropping" in q:
                keep = "urban sketching" in content
            else:
                keep = True
            if keep and item.memory_id:
                selected.append(str(item.memory_id))
        return SemanticRerankResult(
            selected_ids=selected,
            success=True,
            provider="fake",
            model="preference-test",
            usage={"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
        )


def _store():
    return SemanticScoringSQLiteMemoryStore(
        path=":memory:",
        embedding_provider=FlatEmbeddingProvider(),
    )


def _add_preference(store, text, value, at):
    return store.add(
        MemoryRecord(
            content=text,
            kind=MemoryKind.USER_MODEL,
            importance=0.82,
            confidence=0.95,
            fact_subject="user",
            fact_predicate="likes",
            fact_value=value,
            created_at=at,
            updated_at=at,
            metadata={"temporal_valid_from": at.isoformat()},
        )
    )


def _add_tombstone(store, value, evidence, at, resolved_ids=()):
    return store.add(
        MemoryRecord(
            content=f"[RETRACTION EVENT] predicate=likes; value={value}; evidence={evidence}",
            kind=MemoryKind.SEMANTIC,
            importance=0.9,
            confidence=0.96,
            fact_subject="user",
            fact_predicate="retracted:likes",
            fact_value=value,
            created_at=at,
            updated_at=at,
            metadata={
                "retraction_tombstone": True,
                "retracted_predicate": "likes",
                "retraction_source_text": evidence,
                "retraction_at": at.isoformat(),
                "resolved_memory_ids": list(resolved_ids),
                "resolved_at_write": bool(resolved_ids),
            },
        )
    )


def _builder(store):
    return PreferenceStateContextBuilder(
        store,
        SimpleGoalEngine(),
        reranker=PreferenceReranker(),
    )


def test_current_preference_state_suppresses_lexical_and_semantic_retractions():
    store = _store()
    t0 = datetime(2026, 11, 3, 10, 0)
    paddle = _add_preference(store, "Últimamente disfruto mucho del paddle surf.", "paddle surf", t0)
    _add_preference(store, "Me he aficionado a observar aves.", "observar aves", t0 + timedelta(minutes=1))
    _add_preference(store, "Ahora también disfruto del senderismo nocturno.", "senderismo nocturno", t0 + timedelta(hours=3))
    _add_tombstone(store, "paddle surf", "I've lost interest in paddle surf.", t0 + timedelta(hours=2), [paddle.id])
    _add_tombstone(store, "birdwatching", "Birdwatching no longer appeals to me.", t0 + timedelta(hours=2, minutes=1))

    packet = _builder(store).build("What outdoor activities do I currently enjoy?")

    assert "senderismo nocturno" in packet.rendered
    assert "paddle surf" not in packet.rendered
    assert "observar aves" not in packet.rendered
    assert packet.diagnostics["preference_state"]["historical"] is False


def test_current_drink_preferences_exclude_retracted_item():
    store = _store()
    t0 = datetime(2026, 11, 3, 10, 0)
    cold = _add_preference(store, "Brewing cold brew has become a hobby of mine.", "cold brew", t0)
    _add_preference(store, "Me gusta beber rooibos.", "rooibos", t0 + timedelta(minutes=1))
    _add_tombstone(store, "cold brew", "Cold brew isn't my thing anymore.", t0 + timedelta(hours=2), [cold.id])

    packet = _builder(store).build("Which drinks are still among my preferences?")

    assert "rooibos" in packet.rendered
    assert "cold brew" not in packet.rendered.lower()


def test_historical_preference_query_returns_retraction_evidence():
    store = _store()
    t0 = datetime(2026, 11, 3, 10, 0)
    _add_preference(store, "Me he aficionado a observar aves.", "observar aves", t0)
    _add_tombstone(
        store,
        "birdwatching",
        "Birdwatching no longer appeals to me.",
        t0 + timedelta(hours=2),
    )

    packet = _builder(store).build("What activity did I explicitly say no longer appeals to me?")

    assert "observar aves" in packet.rendered
    assert "Birdwatching no longer appeals to me" in packet.rendered
    assert "status=retracted" in packet.rendered
    assert packet.diagnostics["preference_state"]["historical"] is True


def test_recency_preference_query_uses_retraction_cutoff():
    store = _store()
    t0 = datetime(2026, 11, 3, 10, 0)
    _add_preference(store, "I've gotten really into woodworking lately.", "woodworking", t0)
    cold = _add_preference(store, "Brewing cold brew has become a hobby of mine.", "cold brew", t0 + timedelta(minutes=1))
    _add_tombstone(store, "cold brew", "Cold brew isn't my thing anymore.", t0 + timedelta(hours=2), [cold.id])
    _add_preference(store, "Recently I've started enjoying urban sketching.", "urban sketching", t0 + timedelta(hours=3))

    packet = _builder(store).build("What newer creative activity did I add after dropping some old preferences?")

    assert "urban sketching" in packet.rendered
    assert "woodworking" not in packet.rendered
    assert packet.diagnostics["preference_state"]["recency_constrained"] is True


def test_non_preference_math_query_does_not_activate_preference_state_mode():
    store = _store()
    _add_preference(store, "Me gusta el senderismo nocturno.", "senderismo nocturno", datetime(2026, 11, 3, 10, 0))

    packet = _builder(store).build("Dime solo cuánto es 8 por 12.")

    assert PreferenceStateContextBuilder._is_preference_query("Dime solo cuánto es 8 por 12.") is False
    assert "preference_state" not in packet.diagnostics
