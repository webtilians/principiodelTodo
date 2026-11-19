"""DSE-1 contracts; deterministic fake-free checks are NOT real-model results."""
import json

from observatory.dse1_verified_experiences import (
    ALLOWED, ARMS, HELDOUT, TRAIN, World, execute, evidence_from_observations,
    load_observations, parse_plan, prompt, raw_context, visible_episode,
)

def test_preregistered_split_no_composed_train():
    assert len(TRAIN) == 7
    assert len(HELDOUT) == 6
    assert ARMS == ("A", "B", "C")
    assert [(c.public_bit, c.private_bit) for c in HELDOUT] == [
        (0, 0), (1, 1), (0, 1), (1, 0), (0, 0), (1, 1)]
    assert all(c.source == "hidden" and c.invert for c in HELDOUT)
    for case, seq in TRAIN:
        assert not ("op_s" in seq and "op_x" in seq)
        assert "_hidden" not in json.dumps(visible_episode(case, seq))

def test_world_success_requires_real_verified_correct_sequence():
    c = HELDOUT[0]
    w = World(c)
    assert w.step("op_r") == {"error": "precondition"}
    assert w.step("verify") == {"verified": False}
    events, passed = execute(c, ["op_s", "op_r", "op_x", "verify"])
    assert passed
    assert events[-1]["response"] == {"verified": True}
    assert not execute(c, ["op_s", "op_r", "verify"])[1]

def test_sqlite_retrieves_only_observed_episodes():
    records = load_observations()
    assert len(records) == 7
    assert set().union(*(set(x) for x in records)) == {
        "source_id", "service", "source", "invert", "public_bit",
        "events", "verified_success",
    }
    assert all("_hidden" not in json.dumps(x) for x in records)
    raw = raw_context(records)
    summary = evidence_from_observations(records)
    assert "op_s enables op_r" in summary
    assert "[T2,T3] op_x flips" in summary
    assert "T6" in raw
    assert "private_bit" not in raw
    assert "private_bit" not in summary

def test_engineered_summaries_reject_corrupt_evidence():
    records = load_observations()
    records[2]["events"][1]["response"]["value"] = 0  # impossible flip
    try:
        evidence_from_observations(records)
    except ValueError as exc:
        assert "Flip" in str(exc)
    else:
        raise AssertionError("unverified effect was accepted")

def test_strict_parser_and_prompt_no_oracle():
    assert parse_plan('{"plan":["op_s","op_r","op_x","verify"]}') == [
        "op_s", "op_r", "op_x", "verify"]
    for raw in (
        "op_s", "nothing", '{"plan":[]}',
        '{"plan":["op_s","op_r"]}',
        '{"plan":["op_s","op_r","op_x","verify"],"oracle":1}',
        '{"plan":["op_s","op_r","op_x","op_j","op_j","verify"]}',
        '{"plan":["delete","verify"]}',
    ):
        assert parse_plan(raw) is None
    for c in HELDOUT:
        text = prompt(c, "")
        assert "private_bit" not in text and "_hidden_bit" not in text
        assert "GOAL" in text
        assert all(action in text for action in ALLOWED)
