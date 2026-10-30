from src.infinito3.engine import CognitiveEngine
from src.infinito3.persistent_memory import (
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SQLiteCognitiveMemoryStore,
)
from src.infinito3.types import MemoryKind, MemoryRecord, MemoryStatus


def test_sqlite_memory_survives_restart(tmp_path):
    db_path = tmp_path / "memory.db"
    first = SQLiteCognitiveMemoryStore(str(db_path))
    stored = first.add(
        MemoryRecord("Me gusta el descenso en bicicleta", MemoryKind.USER_MODEL, 0.9)
    )
    first.close()

    second = SQLiteCognitiveMemoryStore(str(db_path))
    loaded = second.get(stored.id)

    assert loaded is not None
    assert loaded.content == "Me gusta el descenso en bicicleta"
    assert loaded.status == MemoryStatus.ACTIVE
    second.close()


def test_repeated_fact_reinforces_instead_of_duplicating():
    store = SQLiteCognitiveMemoryStore(":memory:")
    first = store.add(
        MemoryRecord(
            "Me gusta descenso",
            MemoryKind.USER_MODEL,
            0.8,
            confidence=0.60,
        )
    )
    second = store.add(
        MemoryRecord(
            "Me encanta descenso",
            MemoryKind.USER_MODEL,
            0.85,
            confidence=0.60,
        )
    )

    assert first.id == second.id
    assert len(store.all()) == 1
    assert second.confidence > first.confidence
    assert second.metadata["reinforcement_count"] == 1
    store.close()


def test_exclusive_fact_creates_supersession_lineage():
    store = SQLiteCognitiveMemoryStore(":memory:")
    old = store.add(MemoryRecord("Me llamo Enrique", MemoryKind.USER_MODEL, 0.95))
    new = store.add(MemoryRecord("Me llamo Carlos", MemoryKind.USER_MODEL, 0.95))

    active = store.all()
    history = store.history_for_fact("user", "name")

    assert len(active) == 1
    assert active[0].id == new.id
    assert new.supersedes_id == old.id
    assert [record.status for record in history] == [
        MemoryStatus.SUPERSEDED,
        MemoryStatus.ACTIVE,
    ]
    store.close()


def test_multi_value_preferences_do_not_contradict_each_other():
    store = SQLiteCognitiveMemoryStore(":memory:")
    store.add(MemoryRecord("Me gusta el descenso", MemoryKind.USER_MODEL, 0.9))
    store.add(MemoryRecord("Me gusta el café", MemoryKind.USER_MODEL, 0.8))

    assert len(store.all()) == 2
    assert all(record.status == MemoryStatus.ACTIVE for record in store.all())
    store.close()


def test_hybrid_search_returns_relevant_memory_and_tracks_access():
    store = SQLiteCognitiveMemoryStore(":memory:", HashEmbeddingProvider(dimensions=64))
    relevant = store.add(
        MemoryRecord("Me gusta entrenar descenso en bicicleta", MemoryKind.USER_MODEL, 0.9)
    )
    store.add(
        MemoryRecord("Mi color favorito es negro", MemoryKind.USER_MODEL, 0.8)
    )

    results = store.search("qué bicicleta y descenso me gusta", top_k=1)

    assert len(results) == 1
    assert results[0].id == relevant.id
    assert results[0].access_count == 1
    assert results[0].last_accessed_at is not None
    store.close()


def test_soft_forgetting_is_reversible():
    store = SQLiteCognitiveMemoryStore(":memory:")
    weak = store.add(
        MemoryRecord(
            "Dato episódico poco importante",
            MemoryKind.EPISODIC,
            0.05,
            confidence=0.10,
        )
    )

    report = store.forget(min_retention=0.40)

    assert report.forgotten == 1
    assert store.all() == []
    assert store.get(weak.id).status == MemoryStatus.FORGOTTEN
    assert store.restore(weak.id) is True
    assert store.get(weak.id).status == MemoryStatus.ACTIVE
    store.close()


def test_high_importance_memory_is_protected_from_forgetting():
    store = SQLiteCognitiveMemoryStore(":memory:")
    durable = store.add(
        MemoryRecord("Mi nombre es Enrique", MemoryKind.USER_MODEL, 0.95, confidence=0.9)
    )

    report = store.forget(min_retention=0.99, protect_importance=0.85)

    assert report.forgotten == 0
    assert store.get(durable.id).status == MemoryStatus.ACTIVE
    store.close()


def test_persistent_cognitive_engine_uses_sqlite_backend(tmp_path):
    db_path = tmp_path / "engine_memory.db"
    engine = CognitiveEngine.persistent(str(db_path))

    decision = engine.process("Me gusta entrenar descenso en bicicleta")
    assert decision.stored is True
    memory_id = decision.stored_memory_id
    engine.memory_store.close()

    reopened = SQLiteCognitiveMemoryStore(str(db_path))
    assert reopened.get(memory_id) is not None
    reopened.close()


def test_openai_embedding_adapter_accepts_injected_client():
    class Embedding:
        embedding = [0.1, 0.2, 0.3]

    class Response:
        data = [Embedding()]

    class Embeddings:
        def create(self, input, model):
            assert input == ["hola"]
            assert model == "test-model"
            return Response()

    class Client:
        embeddings = Embeddings()

    provider = OpenAIEmbeddingProvider(Client(), model="test-model")

    assert provider.embed("hola") == [0.1, 0.2, 0.3]
