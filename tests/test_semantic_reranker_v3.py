from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.semantic_context import SemanticCohortContextBuilder, SemanticScoringSQLiteMemoryStore
from src.infinito3.semantic_reranker import LLMSemanticMembershipReranker, SemanticRerankResult
from src.infinito3.types import ContextItem, ContextSource, LLMResponse, MemoryKind, MemoryRecord


class FakeAdapter:
    def __init__(self, text):
        self.text = text
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return LLMResponse(
            text=self.text,
            provider="fake",
            model="fake-reranker",
            usage={"input_tokens": 23, "output_tokens": 7, "total_tokens": 30},
        )


class FlatEmbeddingProvider:
    def embed(self, text):
        # Deliberately unhelpful embeddings: the reranker must make the final
        # membership decision rather than relying on an embedding threshold.
        return [0.5, 0.5, 0.5]


class CreativeFacetReranker:
    def rerank(self, query, candidates):
        selected = [
            str(item.memory_id)
            for item in candidates
            if any(
                marker in item.content.lower()
                for marker in ("acuarelas", "cerámica", "fotografía nocturna")
            )
        ]
        return SemanticRerankResult(
            selected_ids=selected,
            success=True,
            provider="fake",
            model="facet-test",
            usage={"input_tokens": 40, "output_tokens": 10, "total_tokens": 50},
        )


def test_llm_semantic_membership_reranker_parses_compact_indices_and_usage():
    adapter = FakeAdapter('{"selected":[0,2,99]}')
    reranker = LLMSemanticMembershipReranker(adapter)
    candidates = [
        ContextItem(ContextSource.USER_MODEL, "A", 1.0, 1, memory_id="uuid-a"),
        ContextItem(ContextSource.USER_MODEL, "B", 0.9, 1, memory_id="uuid-b"),
        ContextItem(ContextSource.USER_MODEL, "C", 0.8, 1, memory_id="uuid-c"),
    ]

    result = reranker.rerank("which are creative?", candidates)

    assert result.success is True
    assert result.selected_ids == ["uuid-a", "uuid-c"]
    assert result.usage["total_tokens"] == 30
    request = adapter.requests[0]
    assert request.metadata["infinito_semantic_reranker"] is True
    assert request.metadata["compact_indices"] is True
    assert '"i":0' in request.messages[-1].content
    assert "uuid-a" not in request.messages[-1].content


def test_semantic_builder_uses_reranker_and_exposes_cost_diagnostics():
    store = SemanticScoringSQLiteMemoryStore(
        path=":memory:",
        embedding_provider=FlatEmbeddingProvider(),
    )
    for text in (
        "Me gusta pintar con acuarelas.",
        "Me gusta hacer cerámica.",
        "Me gusta la fotografía nocturna.",
        "Me gusta jugar al ajedrez.",
        "Me gusta cocinar curry.",
    ):
        store.add(
            MemoryRecord(
                content=text,
                kind=MemoryKind.USER_MODEL,
                importance=0.7,
            )
        )

    builder = SemanticCohortContextBuilder(
        store,
        SimpleGoalEngine(),
        reranker=CreativeFacetReranker(),
    )
    packet = builder.build(
        "¿Qué aficiones creativas o artísticas te he dicho que me gustan? Incluye todas.",
        memory_candidates=store.all(),
    )

    assert "acuarelas" in packet.rendered
    assert "cerámica" in packet.rendered
    assert "fotografía nocturna" in packet.rendered
    assert "ajedrez" not in packet.rendered
    assert "curry" not in packet.rendered
    stats = packet.diagnostics["semantic_reranker"]
    assert stats["calls"] == 1
    assert stats["successful_calls"] == 1
    assert stats["total_tokens"] == 50
    assert stats["candidate_count"] == 5
    assert stats["selected_count"] == 3
