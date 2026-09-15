from datetime import datetime

from src.infinito3.cognitive_events import CognitiveEvent, CognitiveEventType
from src.infinito3.intent_context import IntentContextBuilder
from src.infinito3.persistent_memory import HashEmbeddingProvider
from src.infinito3.semantic_temporal_state import SemanticTemporalCognitiveState
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.temporal_memory import TemporalAwareSQLiteMemoryStore


def _clock(value):
    return lambda: value


def _event(kind, source, *, value, due_at=None, occurred_at):
    return CognitiveEvent(
        kind,
        source,
        predicate="goal",
        value=value,
        due_at=due_at,
        occurred_at=occurred_at,
    )


def test_reschedule_removes_stale_schedule_from_live_goal_description():
    now = datetime(2027, 8, 2, 9)
    goals = TemporalGoalEngine(now_fn=_clock(now))
    state = SemanticTemporalCognitiveState(now_fn=_clock(now))
    original = "return the borrowed lighting rig Thursday at 17:30"
    state.apply([
        _event(
            CognitiveEventType.CREATE_GOAL,
            "I need to return the borrowed lighting rig Thursday at 17:30.",
            value=original,
            due_at=datetime(2027, 8, 5, 17, 30),
            occurred_at=now,
        )
    ], goal_engine=goals)
    state.apply([
        _event(
            CognitiveEventType.RESCHEDULE_GOAL,
            "The lighting rig return has moved to Saturday at 11:20.",
            value="lighting rig return",
            due_at=datetime(2027, 8, 7, 11, 20),
            occurred_at=datetime(2027, 8, 3, 9),
        )
    ], goal_engine=goals)

    goal = goals.all()[0]
    assert goal.description == "lighting rig return"
    assert goal.due_at == datetime(2027, 8, 7, 11, 20)
    assert goal.metadata["audit_description"] == original
    assert goal.metadata["canonical_label"] == "lighting rig return"
    assert goal.metadata["canonical_due_at"] == "2027-08-07T11:20:00"
    assert state.goal_history()[-1].description == "lighting rig return"
    assert state.goal_history()[-1].previous_due_at == datetime(2027, 8, 5, 17, 30)


def test_context_uses_new_due_date_without_old_weekday_or_clock():
    now = datetime(2027, 8, 2, 9)
    store = TemporalAwareSQLiteMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    goals = TemporalGoalEngine(now_fn=_clock(now))
    state = SemanticTemporalCognitiveState(now_fn=_clock(now))
    state.apply([
        _event(
            CognitiveEventType.CREATE_GOAL,
            "I need to return the borrowed lighting rig Thursday at 17:30.",
            value="return the borrowed lighting rig Thursday at 17:30",
            due_at=datetime(2027, 8, 5, 17, 30),
            occurred_at=now,
        ),
        _event(
            CognitiveEventType.RESCHEDULE_GOAL,
            "Reschedule the lighting rig return to Saturday at 11:20.",
            value="lighting rig return",
            due_at=datetime(2027, 8, 7, 11, 20),
            occurred_at=datetime(2027, 8, 3, 9),
        ),
    ], memory_store=store, goal_engine=goals)

    packet = IntentContextBuilder(store, goals, now_fn=_clock(datetime(2027, 8, 7, 8))).build(
        "Which commitment is due this morning?",
        memory_candidates=[],
        max_tokens=300,
    )
    assert "lighting rig return" in packet.rendered
    assert "2027-08-07T11:20" in packet.rendered
    assert "Thursday" not in packet.rendered
    assert "17:30" not in packet.rendered
    store.close()


def test_completion_after_reschedule_keeps_canonical_goal_identity():
    now = datetime(2027, 8, 2, 9)
    store = TemporalAwareSQLiteMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    goals = TemporalGoalEngine(now_fn=_clock(now))
    state = SemanticTemporalCognitiveState(now_fn=_clock(now))
    state.apply([
        _event(
            CognitiveEventType.CREATE_GOAL,
            "On Thursday at 17:30 I must return the borrowed lighting rig.",
            value="return the borrowed lighting rig Thursday at 17:30",
            due_at=datetime(2027, 8, 5, 17, 30),
            occurred_at=now,
        ),
        _event(
            CognitiveEventType.RESCHEDULE_GOAL,
            "Move the lighting rig return to Saturday at 11:20.",
            value="lighting rig return",
            due_at=datetime(2027, 8, 7, 11, 20),
            occurred_at=datetime(2027, 8, 3, 9),
        ),
        _event(
            CognitiveEventType.COMPLETE_GOAL,
            "The lighting rig has been returned; mark it complete.",
            value="lighting rig",
            occurred_at=datetime(2027, 8, 7, 12),
        ),
    ], memory_store=store, goal_engine=goals)

    goal = goals.all()[0]
    assert goal.completed is True
    assert goal.description == "lighting rig return"
    packet = IntentContextBuilder(store, goals, now_fn=_clock(datetime(2027, 8, 7, 12))).build(
        "Is the lighting rig return completed or still open?",
        memory_candidates=[],
        max_tokens=300,
    )
    assert "lighting rig return" in packet.rendered
    assert "status=completed" in packet.rendered
    assert "Thursday" not in packet.rendered
    assert "17:30" not in packet.rendered
    store.close()


def test_reschedule_without_embedded_old_schedule_preserves_original_identity_text():
    now = datetime(2027, 8, 2, 9)
    goals = TemporalGoalEngine(now_fn=_clock(now))
    state = SemanticTemporalCognitiveState(now_fn=_clock(now))
    original = "return the borrowed camera tripod"
    state.apply([
        _event(
            CognitiveEventType.CREATE_GOAL,
            "I must return the borrowed camera tripod.",
            value=original,
            due_at=datetime(2027, 8, 5, 10),
            occurred_at=now,
        ),
        _event(
            CognitiveEventType.RESCHEDULE_GOAL,
            "Reschedule the camera tripod return to Saturday at 16:10.",
            value="camera tripod return",
            due_at=datetime(2027, 8, 7, 16, 10),
            occurred_at=datetime(2027, 8, 3, 9),
        ),
    ], goal_engine=goals)
    goal = goals.all()[0]
    assert goal.description == original
    assert "audit_description" not in goal.metadata
    assert goal.metadata["canonical_label"] == "camera tripod return"
