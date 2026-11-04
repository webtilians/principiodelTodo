"""Finding memories by meaning: what recall hands the agent when it cannot hand everything."""
import os

import pytest

from src.infinito3.mcp_server import SHOW_ALL_UP_TO, AgentMemory


class LatestFirst:
    """A stand-in for the model: the later a memory, the closer its meaning."""

    def rank(self, query, texts):
        return list(range(len(texts)))[::-1]


@pytest.fixture
def memory(tmp_path):
    agent_memory = AgentMemory(str(tmp_path / "memory.db"), ranker=LatestFirst())
    for n in range(30):
        agent_memory.remember(f"Mi proyecto número {n} usa la librería {n} para procesar datos.")
    yield agent_memory
    agent_memory.close()


def test_a_limit_hands_over_only_the_closest_in_meaning_oldest_first(memory):
    lines = [line for line in memory.recall("lo que sea", limit=5).splitlines() if line.startswith("[")]
    assert [line.split('"')[1].split()[3] for line in lines] == ["25", "26", "27", "28", "29"]


def test_without_a_limit_every_memory_comes_while_there_are_few(memory):
    assert SHOW_ALL_UP_TO >= 30
    assert sum(line.startswith("[") for line in memory.recall("lo que sea").splitlines()) == 30


@pytest.mark.skipif(not os.environ.get("INFINITO_EMBEDDINGS"), reason="needs the model (INFINITO_EMBEDDINGS)")
def test_the_model_finds_answers_that_share_no_word_with_the_question():
    from src.infinito3.semantic import ranker_from

    ranker = ranker_from(os.environ["INFINITO_EMBEDDINGS"])
    memories = [f"Mi proyecto número {n} usa la librería {n} para procesar datos." for n in range(40)] + [
        "Ahora mismo mis canciones favoritas son Iris de Goo Goo Dolls y Duvet de bôa.",
        "Tengo una Renault Trafic de 2003 que uso como furgoneta.",
        "Mi perra se llama Nena.",
    ]
    for question, answer in (("¿Qué música me gusta?", "canciones"), ("¿Qué coche tengo?", "Trafic"),
                             ("¿Tengo mascota?", "Nena"), ("What's my dog called?", "Nena")):
        top = [memories[i] for i in ranker.rank(question, memories)[:5]]
        assert any(answer in text for text in top), question
