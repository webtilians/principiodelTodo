from datetime import datetime

import pytest

from src.infinito3.context_intent import detect_history_cue, resolve_context_intent
from src.infinito3.intent_context import IntentContextBuilder
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.types import MemoryKind, MemoryRecord, MemoryStatus


def _fact(predicate, value):
    return MemoryRecord(
        content=f"{predicate}={value}",
        kind=MemoryKind.USER_MODEL,
        importance=.95,
        fact_predicate=predicate,
        fact_value=value,
    )


def test_recall_composes_multiple_profile_predicates():
    intent = resolve_context_intent(
        "Recall my name, home city, bicycle, language and profession."
    )
    assert intent.version == "context_intent_v2"
    assert intent.mode == "facts"
    assert intent.predicates == frozenset({
        "name", "location", "bike", "studying_language", "occupation"
    })

    store = InMemoryMemoryStore()
    values = {
        "name": "Mara",
        "location": "Turin",
        "bike": "Kona Process",
        "studying_language": "Latvian",
        "occupation": "book conservator",
    }
    for predicate, value in values.items():
        store.add(_fact(predicate, value))
    packet = IntentContextBuilder(store, TemporalGoalEngine()).build(
        "Recall my name, home city, bicycle, language and profession.",
        memory_candidates=[],
    )
    for value in values.values():
        assert value in packet.rendered


@pytest.mark.parametrize("query", [
    "Which city preceded Rotterdam as my home?",
    "What was my former home before Rotterdam?",
    "Which city came before Rotterdam?",
    "What was my prior city to Rotterdam?",
])
def test_predecessor_language_is_historical(query):
    intent = resolve_context_intent(query)
    assert intent.historical
    assert intent.history_cue == "predecessor"
    assert "location" in intent.predicates


def test_predecessor_query_uses_temporal_lineage():
    store = InMemoryMemoryStore()
    old = store.add(MemoryRecord(
        "I lived in Porto", MemoryKind.USER_MODEL, .9,
        status=MemoryStatus.SUPERSEDED,
        fact_predicate="location", fact_value="Porto",
    ))
    current = store.add(MemoryRecord(
        "I live in Rotterdam", MemoryKind.USER_MODEL, .95,
        supersedes_id=old.id,
        fact_predicate="location", fact_value="Rotterdam",
    ))
    builder = IntentContextBuilder(store, TemporalGoalEngine())
    packet = builder.build(
        "Which city preceded Rotterdam as my home?",
        memory_candidates=[current],
    )
    assert "Porto" in packet.rendered
    assert "immediately_previous" in packet.rendered
    assert "Rotterdam" not in packet.rendered


def test_show_calendar_range_is_a_goal_query():
    now = datetime(2027, 3, 1, 8)  # Monday
    goals = TemporalGoalEngine(now_fn=lambda: now)
    for day in range(2, 8):
        goals.add_structured_goal(
            f"errand-{day}", due_at=datetime(2027, 3, day, 10)
        )
    builder = IntentContextBuilder(InMemoryMemoryStore(), goals, now_fn=lambda: now)
    packet = builder.build("Show my calendar from Wednesday through Saturday.")
    assert packet.diagnostics["context_intent"]["mode"] == "goals"
    assert len(packet.items) == 4
    assert "errand-3" in packet.rendered
    assert "errand-6" in packet.rendered
    assert "errand-2" not in packet.rendered
    assert "errand-7" not in packet.rendered


def test_lifecycle_question_can_retrieve_closed_goal_without_goal_noun():
    goals = TemporalGoalEngine()
    goal = goals.add_structured_goal("parcel handoff")
    goal.completed = True
    goal.metadata["lifecycle"] = "completed"
    builder = IntentContextBuilder(InMemoryMemoryStore(), goals)
    query = "Is the parcel handoff completed or still open?"
    intent = builder.resolve_intent(query)
    assert intent.mode == "goals"
    assert intent.goal_status == "any"
    packet = builder.build(query)
    assert "parcel handoff" in packet.rendered
    assert "status=completed" in packet.rendered


def test_closed_cohort_excludes_open_goals_when_requested():
    goals = TemporalGoalEngine()
    closed = goals.add_structured_goal("return library key")
    closed.completed = True
    closed.metadata["lifecycle"] = "completed"
    goals.add_structured_goal("collect framing order")
    builder = IntentContextBuilder(InMemoryMemoryStore(), goals)
    packet = builder.build("List my completed commitments.")
    assert "return library key" in packet.rendered
    assert "collect framing order" not in packet.rendered


def test_literal_note_aliases_share_one_retrieval_family():
    intent = resolve_context_intent(
        "What literal verification phrase did I ask you to remember?"
    )
    assert {
        "test_phrase", "verification_phrase", "literal_test_phrase",
        "literal_verification_phrase",
    } <= set(intent.predicates)

    store = InMemoryMemoryStore()
    store.add(MemoryRecord(
        "amber lighthouse 73", MemoryKind.SEMANTIC, .95,
        fact_predicate="literal_test_phrase",
        fact_value="amber lighthouse 73",
    ))
    packet = IntentContextBuilder(store, TemporalGoalEngine()).build(
        "What verification phrase did I ask you to remember?",
        memory_candidates=[],
    )
    assert "amber lighthouse 73" in packet.rendered


@pytest.mark.parametrize(("text", "expected"), [
    ("I lost interest in rowing.", "lost_interest"),
    ("Ceramics has lost its appeal.", "lost_appeal"),
    ("I no longer enjoy fencing.", "no_longer_enjoy"),
    ("I stopped collecting stamps.", "stopped"),
])
def test_history_cue_is_entity_independent(text, expected):
    assert detect_history_cue(text) == expected


def test_historical_preference_operator_selects_matching_retraction_evidence():
    store = InMemoryMemoryStore()
    pottery = store.add(MemoryRecord(
        "I enjoy wheel throwing", MemoryKind.USER_MODEL, .9,
        fact_predicate="likes", fact_value="wheel throwing",
        status=MemoryStatus.FORGOTTEN,
    ))
    fencing = store.add(MemoryRecord(
        "I enjoy fencing", MemoryKind.USER_MODEL, .9,
        fact_predicate="likes", fact_value="fencing",
        status=MemoryStatus.FORGOTTEN,
    ))
    store.add(MemoryRecord(
        "Retraction of wheel throwing", MemoryKind.SEMANTIC, .95,
        fact_predicate="retracted:likes", fact_value="wheel throwing",
        metadata={
            "retraction_tombstone": True,
            "retracted_predicate": "likes",
            "resolved_memory_ids": [pottery.id],
            "retraction_source_text": "Wheel throwing has lost its appeal for me.",
        },
    ))
    store.add(MemoryRecord(
        "Retraction of fencing", MemoryKind.SEMANTIC, .95,
        fact_predicate="retracted:likes", fact_value="fencing",
        metadata={
            "retraction_tombstone": True,
            "retracted_predicate": "likes",
            "resolved_memory_ids": [fencing.id],
            "retraction_source_text": "I no longer enjoy fencing.",
        },
    ))
    builder = IntentContextBuilder(store, TemporalGoalEngine())
    packet = builder.build("Which pastime did I say had lost its appeal?")
    assert "wheel throwing" in packet.rendered
    assert "fencing" not in packet.rendered
    assert packet.items[0].metadata["preference_history_cue_match"] == "lost_appeal"


def test_standalone_safety_boundary_is_preserved_in_v2():
    assert not resolve_context_intent("Return only 33 plus 9.").retrieve
    assert not resolve_context_intent("Explain diffraction briefly.").retrieve
    assert resolve_context_intent("What is my rent plus 9?").retrieve
    assert resolve_context_intent("Remember this: return only 33 plus 9.").retrieve
