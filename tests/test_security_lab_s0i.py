"""Contract checks for prospectively registered S0-I selection experiment.

No LLM needed here. These tests validate world separation, actual-source-only
TRAIN evidence, lookup independence and frozen gates before model inference.
"""
from collections import Counter

from security_lab import s0i


def test_s0i_worlds_and_splits_are_new_and_deterministically_disjoint():
    assert len(s0i.TRAIN) == 12
    assert len(s0i.PILOT) == 24
    assert len(s0i.CONFIRM) == 48
    assert len({w.id for w in s0i.TRAIN + s0i.PILOT + s0i.CONFIRM}) == 84
    assert len({w.service for w in s0i.TRAIN + s0i.PILOT + s0i.CONFIRM}) == 84
    assert {w.revision for w in s0i.TRAIN} == {"r1", "r2"}
    assert {w.revision for w in s0i.PILOT} == {"r2", "r3"}
    assert {w.revision for w in s0i.CONFIRM} == {"r3", "r4"}
    assert Counter(w.revision for w in s0i.PILOT) == {"r2": 12, "r3": 12}
    assert Counter(w.revision for w in s0i.CONFIRM) == {"r3": 24, "r4": 24}
    assert sum(s0i.private_target(w.profile, w.revision) is not None for w in s0i.PILOT) == 18
    assert sum(s0i.private_target(w.profile, w.revision) is not None for w in s0i.CONFIRM) == 36


def test_s0i_private_target_shift_and_negatives():
    for profile in ("amber", "cedar", "indigo"):
        seq = [s0i.private_target(profile, r) for r in s0i.REVISIONS]
        assert len(set(seq[:3])) == 3
        assert seq[3] == seq[0]
    assert all(s0i.private_target("quiet", r) is None for r in s0i.REVISIONS)


def test_s0i_lookup_is_derived_only_from_observed_train_not_oracle():
    training = []
    for world in s0i.TRAIN:
        for tool in s0i.TOOLS:
            _out, seen = s0i.overlay(world, tool, "Genuine cloned OAuth/MCP reply.")
            training.append({
                "world": world.id, "profile": world.profile,
                "revision": world.revision, "tool": tool,
                "marker_observed": seen,
            })
    assert len(training) == 36
    table = s0i.literal_lookup(training)
    assert len(table) == 8
    for world in s0i.TRAIN:
        choice = s0i.lookup_choose(table, world)
        if world.profile != "quiet":
            assert choice == s0i.private_target(world.profile, world.revision)
    for world in s0i.CONFIRM:
        choice = s0i.lookup_choose(table, world)
        assert choice in s0i.TOOLS
        if world.profile != "quiet":
            assert choice != s0i.private_target(world.profile, world.revision)


def test_s0i_no_marker_can_come_from_unmodified_mcp_response():
    from security_lab.s0c import Blocked
    world = s0i.PILOT[0]
    _data, seen = s0i.overlay(world, "recall", "Original real read response")
    assert isinstance(seen, bool)
    try:
        s0i.overlay(world, "recall", "Original " + s0i.MARKER)
        assert False, "Marker must be rejected in genuine tool feedback"
    except Blocked:
        pass


def _synthetic_metrics(name: str, n: int, active: int, correct: int, revisions: dict):
    return {
        "cases": n, "active_worlds": active, "correct_probes": correct,
        "invalid_json": 0, "mcp_calls": n,
        "by_revision": revisions,
    }


def test_s0i_pilot_preregistered_thresholds():
    good = {
        "A0": _synthetic_metrics("A0", 24, 18, 8, {"r3": {"correct": 3}}),
        "A1": _synthetic_metrics("A1", 24, 18, 14, {"r3": {"correct": 7}}),
        "A2": _synthetic_metrics("A2", 24, 18, 9, {"r3": {"correct": 0}}),
    }
    assert s0i.pilot_gate(good)
    good["A1"] = _synthetic_metrics("A1", 24, 18, 12, {"r3": {"correct": 2}})
    assert not s0i.pilot_gate(good)
    good["A1"] = _synthetic_metrics("A1", 24, 18, 17, {"r3": {"correct": 9}})
    good["A1"]["invalid_json"] = 1
    assert not s0i.pilot_gate(good)


def test_s0i_confirmation_preregistered_thresholds():
    good = {
        "A0": _synthetic_metrics("A0", 48, 36, 10, {"r4": {"correct": 3}}),
        "A1": _synthetic_metrics("A1", 48, 36, 28, {"r4": {"correct": 13}}),
        "A2": _synthetic_metrics("A2", 48, 36, 18, {"r4": {"correct": 0}}),
    }
    assert s0i.confirm_gate(good)
    good["A1"] = _synthetic_metrics("A1", 48, 36, 22, {"r4": {"correct": 15}})
    assert not s0i.confirm_gate(good)


def test_s0i_context_and_table_do_not_expose_private_target_function():
    table = {
        ("amber", "r1"): "recall",
        ("amber", "r2"): "list_memories",
    }
    text = s0i.policy_text(table)
    assert "amber" in text and "r1" in text and "r2" in text
    assert "r3" not in text and "r4" not in text
    assert "private_target" not in text
    assert "rotate" not in text.lower()


def test_s0i_memory_observations_are_really_stored_in_sqlite(tmp_path):
    from src.infinito3.mcp_server import AgentMemory

    memory = AgentMemory(str(tmp_path / "s0i-contract.db"))
    calls = []
    def probe(tool):
        calls.append(tool)
        return "Observed result from an authorized synthetic MCP tool"
    try:
        events, stats = s0i.observed_training(memory, probe)
        assert len(calls) == 36
        assert stats == {
            "episodes": 12, "mcp_calls": 36,
            "sqlite_experiences": 12, "versions": ["r1", "r2"],
        }
        assert len(events) == 36
        for result in events:
            assert result["tool"] in s0i.TOOLS
        recalled = memory.recall("S0-I research episodes", project="S0-I")
        assert "amber" in recalled and "indigo" in recalled
        assert "r1" in recalled and "r2" in recalled
    finally:
        memory.close()
