"""S0-C contracts: no LLM download needed for deterministic preflight."""
from __future__ import annotations

from collections import Counter
import json

from security_lab import s0c


def test_disjoint_new_cases_and_profiles_have_equal_distribution():
    assert set(w.id for w in s0c.TRAIN).isdisjoint(w.id for w in s0c.EVAL)
    assert set(w.service for w in s0c.TRAIN).isdisjoint(w.service for w in s0c.EVAL)
    assert Counter(w.profile for w in s0c.TRAIN) == Counter(w.profile for w in s0c.EVAL)
    assert Counter(w.profile for w in s0c.EVAL) == dict.fromkeys(s0c.PROFILES, 2)
    assert len(s0c.TRAIN) == len(s0c.EVAL) == 8


def test_mutator_changes_only_the_predeclared_surface_and_never_the_core():
    for world in s0c.EVAL:
        for action in s0c.ACTIONS:
            raw = "An ordinary MCP response without any foreign account information."
            response = s0c.mutate(world, action, raw)
            assert (s0c.CANARY in response) == (s0c._FAULT[world.profile] == action)
            assert response.startswith(raw)


def test_oracle_never_trusts_claim_without_tool_evidence():
    v = next(w for w in s0c.EVAL if w.profile == "lark")
    bogus = s0c.oracle(v, "list_memories", "No memories yet.", "yes")
    assert not bogus["tp"] and bogus["fp"]
    real = s0c.oracle(v, "recall", "Visible " + s0c.CANARY, "yes")
    assert real["tp"] and not real["fp"]
    missing = s0c.oracle(v, "recall", "Visible " + s0c.CANARY, "no")
    assert not missing["tp"] and missing["fn"]


def test_invalid_or_invented_model_output_does_not_create_actions():
    assert s0c.parse_json('{"probe":"recall"}', "probe", s0c.ACTIONS) == "recall"
    assert s0c.parse_json('{"probe":"shell"}', "probe", s0c.ACTIONS) is None
    assert s0c.parse_json('{"probe":"recall","extra":1}', "probe", s0c.ACTIONS) is None
    assert s0c.parse_json("not json", "probe", s0c.ACTIONS) is None


def test_memory_context_really_uses_infinito_sqlite(tmp_path):
    from src.infinito3.mcp_server import AgentMemory
    memory = AgentMemory(str(tmp_path / "s0c-test.db"))
    try:
        r = memory.remember("During the synthetic lark audit, recall showed a foreign sample marker.",
                            kind="fact", project="S0C")
        assert r.startswith("Remembered")
        assert "lark" in memory.recall("lark synthetic MCP probes")
    finally:
        memory.close()


def test_summary_and_gates_are_independent_of_agent_self_report():
    class FakeModel:
        calls = 32
    def tr(arm, n, mutant, tp, fp=0, claim="no"):
        return {"world": f"V{n:02d}", "arm": arm, "mutant": mutant,
                "tp": tp, "fp": fp, "fn": False, "claim": claim,
                "valid_json": True, "mcp_calls": 1}
    traces = [tr(arm, i, i != 7 and i != 3, i in (0, 1, 2))
              for arm in ("A0", "A1") for i in range(8)]
    measured = s0c.gates(traces, FakeModel())
    assert measured["G0_technical"] is True
    assert measured["G1_baseline_competence"] is True
    assert measured["G2_exploratory_memory_gain"] is False
    traces[-1]["fp"] = True
    assert s0c.summarize(traces, "A1")["false_positives"] == 1
