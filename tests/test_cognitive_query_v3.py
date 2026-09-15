from datetime import datetime

from src.infinito3.cognitive_query import CognitiveQueryOperator, build_cognitive_query_plan
from src.infinito3.context_intent import resolve_context_intent
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.planned_context import PlannedIntentContextBuilder
from src.infinito3.temporal_goals import TemporalGoalEngine


def _ops(plan):
    return [step.operator for step in plan.steps]


def test_plan_preserves_context_intent_contract():
    now = datetime(2027, 3, 4, 10, 0)
    queries = (
        "Read back my name, city, bicycle, language and occupation.",
        "Which commitment is due this morning?",
        "Which hobby did I add most recently?",
        "What city preceded my current home?",
        "Quote my stored test phrase.",
        "What is 44 times 3?",
    )
    for query in queries:
        assert build_cognitive_query_plan(query, now).intent == resolve_context_intent(query, now)


def test_profile_plan_decomposes_requested_slots():
    plan = build_cognitive_query_plan(
        "Read back my name, home city, bicycle, language and occupation."
    )
    step = plan.first(CognitiveQueryOperator.READ_FACTS)
    assert step is not None
    assert set(step.predicates) == {
        "name", "location", "bike", "studying_language", "occupation",
    }


def test_temporal_and_history_operators_are_explicit():
    now = datetime(2027, 3, 6, 11, 15)
    morning = build_cognitive_query_plan("Which commitment is due this morning?", now)
    assert CognitiveQueryOperator.READ_GOALS in _ops(morning)
    window = morning.first(CognitiveQueryOperator.FILTER_WINDOW)
    assert window is not None
    assert window.value == "morning"
    assert window.window == (datetime(2027, 3, 6, 5), datetime(2027, 3, 6, 13))

    predecessor = build_cognitive_query_plan("What city preceded my current home?", now)
    assert predecessor.has(CognitiveQueryOperator.READ_HISTORY)
    assert predecessor.has(CognitiveQueryOperator.PREDECESSOR)
    assert predecessor.first(CognitiveQueryOperator.PREDECESSOR).predicates == ("location",)


def test_latest_preference_separates_membership_from_ordering():
    plan = build_cognitive_query_plan("Which craft did I add most recently?")
    assert _ops(plan).index(CognitiveQueryOperator.SEMANTIC_MEMBERSHIP) < _ops(plan).index(CognitiveQueryOperator.ORDER_LATEST)
    assert plan.has(CognitiveQueryOperator.READ_PREFERENCES)


def test_literal_and_standalone_are_typed():
    literal = build_cognitive_query_plan("Quote my stored test phrase.")
    assert literal.has(CognitiveQueryOperator.LITERAL_READ)
    assert literal.has(CognitiveQueryOperator.READ_FACTS)

    standalone = build_cognitive_query_plan("What is 44 times 3?")
    assert standalone.steps[0].operator == CognitiveQueryOperator.NO_RETRIEVAL
    assert standalone.retrieve is False


def test_planned_builder_publishes_auditable_plan_without_changing_intent():
    now = datetime(2027, 3, 6, 11, 15)
    builder = PlannedIntentContextBuilder(
        InMemoryMemoryStore(), TemporalGoalEngine(now_fn=lambda: now), now_fn=lambda: now
    )
    query = "Which commitment is due this morning?"
    packet = builder.build(query)
    assert builder.resolve_intent(query) == resolve_context_intent(query, now)
    plan = packet.diagnostics["cognitive_query_plan"]
    assert plan["version"] == "cognitive_query_plan_v1"
    assert plan["intent"]["window"] == [
        "2027-03-06T05:00:00", "2027-03-06T13:00:00",
    ]
