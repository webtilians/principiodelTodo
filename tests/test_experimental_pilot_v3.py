"""Contracts for the synthetic preflight; these tests make NO LLM claims."""
from experimental_pilot.pilot import (
    PROJECT,
    compose_control,
    evaluate_actions,
    export_packets,
    observe_episode,
    retrieve,
    run_preflight,
    seed_experiences,
    verified_edges,
)
from src.infinito3.persistent_memory import SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord


def test_composes_new_path_across_two_independently_verified_episodes():
    store = SQLiteCognitiveMemoryStore(":memory:")
    try:
        seed_experiences(store)
        records = retrieve(store)
        assert len(records) == 3
        edges = verified_edges(records)
        assert len(edges) == 4
        plan = compose_control(edges, "fresh", "delivered")
        assert [e["tool"] for e in plan] == ["ingest", "normalize", "tag", "deliver"]
        assert len({e["episode_id"] for e in plan}) == 2
        result = evaluate_actions([e["tool"] for e in plan] + ["verify"])
        assert result["success"] is True
        assert result["final_state"] == "delivered"
    finally:
        store.close()


def test_does_not_promote_unverified_or_foreign_source_even_if_claims_success():
    store = SQLiteCognitiveMemoryStore(":memory:")
    try:
        seed_experiences(store)
        fake = MemoryRecord(
            "Synthetic fake transitions",
            MemoryKind.EPISODIC,
            0.9,
            metadata={
                "project": PROJECT,
                "origin": "synthetic",
                "source": "untrusted-import",
                "verified": True,
                "episode_id": "forged",
                "events": [{"tool": "deliver", "before": "fresh", "after": "delivered", "ok": True}],
            },
        )
        store.add(fake, alone=True)
        assert "forged" not in {e["episode_id"] for e in verified_edges(store.all())}
        assert "unverified-C" not in {e["episode_id"] for e in verified_edges(store.all())}
    finally:
        store.close()


def test_referee_requires_successful_final_verify_and_budget():
    assert not evaluate_actions(["ingest", "normalize", "tag", "deliver"])["success"]
    assert not evaluate_actions(["deliver", "verify"])["success"]
    assert not evaluate_actions(["ingest", "verify", "normalize", "tag", "deliver"])["success"]
    assert evaluate_actions(["ingest"] * 7)["reason"] == "action_budget_exceeded"
    assert not evaluate_actions(["ingest", "normalize", "bad", "tag", "deliver", "verify"])["success"]


def test_memory_is_only_local_to_requested_store():
    first, second = SQLiteCognitiveMemoryStore(":memory:"), SQLiteCognitiveMemoryStore(":memory:")
    try:
        observe_episode(first, "A", "fresh", ["ingest"], "imported")
        assert len(first.all()) == 1
        assert second.all() == []
    finally:
        first.close()
        second.close()


def test_export_does_not_supply_complete_solution_and_is_not_model_run():
    store = SQLiteCognitiveMemoryStore(":memory:")
    try:
        seed_experiences(store)
        data = export_packets(store)
        assert data["task"]["origin"] == "synthetic-only"
        assert data["variants"]["A_no_memory"] == []
        assert len(data["variants"]["B_episodes"]) == 2
        assert len(data["variants"]["C_verified_effects"]) == 4
        assert "unverified-C" not in str(data["variants"])
        assert "no model has been run" in data["note"]
    finally:
        store.close()


def test_deterministic_preflight_has_no_model_calls():
    report = run_preflight()
    assert report["status"] == "PASS"
    assert report["model_calls"] == 0
    assert report["real_users"] == 0
    assert report["real_mcp_calls"] == 0
    assert report["unverified_promoted"] is False
    assert report["cross_episode_composition"] is True
