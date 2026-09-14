from src.infinito3.semantic_context import SemanticCohortContextBuilder
from src.infinito3.types import ContextItem, ContextSource


def _items(count):
    return [
        ContextItem(
            source=ContextSource.USER_MODEL,
            content=f"fact {index}",
            score=1.0 - index * 0.01,
            estimated_tokens=2,
            memory_id=f"m{index}",
            metadata={"fact_predicate": "likes"},
        )
        for index in range(count)
    ]


def test_reranker_gate_escalates_when_rejected_competitors_outnumber_selected():
    candidates = _items(11)
    selected = candidates[:2]
    semantic = {item.memory_id: 0.5 - index * 0.01 for index, item in enumerate(candidates)}

    assert SemanticCohortContextBuilder._should_rerank(candidates, selected, semantic) is True


def test_reranker_gate_skips_balanced_small_cohort():
    candidates = _items(4)
    selected = candidates[:2]
    semantic = {"m0": 0.61, "m1": 0.55, "m2": 0.31, "m3": 0.20}

    assert SemanticCohortContextBuilder._should_rerank(candidates, selected, semantic) is False


def test_reranker_gate_escalates_when_embeddings_are_flat():
    candidates = _items(5)
    semantic = {item.memory_id: 0.40 for item in candidates}

    assert SemanticCohortContextBuilder._should_rerank(candidates, candidates, semantic) is True


def test_reranker_gate_skips_full_cohort_when_embeddings_have_structure():
    candidates = _items(5)
    semantic = {"m0": 0.70, "m1": 0.62, "m2": 0.51, "m3": 0.42, "m4": 0.28}

    assert SemanticCohortContextBuilder._should_rerank(candidates, candidates, semantic) is False
