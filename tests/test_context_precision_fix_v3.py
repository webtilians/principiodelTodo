import pytest

from src.infinito3.engine import CognitiveEngine
from src.infinito3.precision_cases import precision_evaluation_suite
from src.infinito3.types import ContextSource


@pytest.mark.parametrize('scenario', precision_evaluation_suite(), ids=lambda s: s.name)
def test_precision_selection_with_all_stored_candidates(scenario):
    """Isolate the builder from retrieval: include even irrelevant candidates."""
    engine = CognitiveEngine.persistent(':memory:')
    try:
        for text in scenario.setup_turns:
            engine.process(text)
        packet = engine.context_builder.build(scenario.probe,
            memory_candidates=engine.memory_store.all(), max_tokens=scenario.context_budget_tokens)
        rendered = packet.rendered.lower()
        for phrase in scenario.expectation.context_contains:
            assert phrase.lower() in rendered
        for phrase in scenario.expectation.context_excludes:
            assert phrase.lower() not in rendered
        assert packet.estimated_tokens <= scenario.context_budget_tokens
    finally:
        engine.memory_store.close()


def test_irrelevant_goal_cannot_return_through_memory_channel():
    engine = CognitiveEngine.persistent(':memory:')
    try:
        engine.process('Mañana tengo que comprar una bici a las 10.')
        engine.process('Mi bici es una Trek Session.')
        packet = engine.context_builder.build('¿Qué bici uso?', memory_candidates=engine.memory_store.all())
        assert 'Trek Session' in packet.rendered
        assert 'comprar' not in packet.rendered
        assert all(item.source != ContextSource.GOAL for item in packet.items)
    finally:
        engine.memory_store.close()


def test_arithmetic_using_personal_facts_keeps_requested_memory():
    engine = CognitiveEngine.persistent(':memory:')
    try:
        engine.process('Tengo 37 años.')
        packet = engine.context_builder.build('Suma mi edad al resultado de 2 + 3.',
                                             memory_candidates=engine.memory_store.all())
        assert '37' in packet.rendered
    finally:
        engine.memory_store.close()
