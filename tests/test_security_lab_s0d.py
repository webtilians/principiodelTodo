"""S0-D preregistered contracts, without downloading or mocking a real LLM result."""
from collections import Counter
import json

import pytest

from security_lab import s0c, s0d


def test_worlds_are_new_disjoint_and_hold_out_two_revision_pairs():
    historical = {w.id for w in s0c.EVAL} | {w.id for w in s0c.TRAIN}
    assert len(s0d.TRAIN) == len(s0d.DEV) == 8
    assert not ({w.id for w in s0d.DEV} & historical)
    assert not ({w.service for w in s0d.DEV} & {w.service for w in s0c.EVAL})
    assert not ({w.service for w in s0d.TRAIN} & {w.service for w in s0d.DEV})
    train_pairs = {(w.profile, w.revision) for w in s0d.TRAIN}
    dev_pairs = {(w.profile, w.revision) for w in s0d.DEV}
    assert len(train_pairs) == 6 and len(dev_pairs) == 8
    assert dev_pairs - train_pairs == {("cedar", "R2"), ("indigo", "R2")}
    assert Counter(w.profile for w in s0d.DEV) == dict.fromkeys(s0d.PROFILES, 2)


def test_d1_fixed_bench_has_8_positive_8_negative_and_adversarial_distractors():
    assert len(s0d.EVIDENCE) == 16
    assert len({c.id for c in s0d.EVIDENCE}) == 16
    assert sum(c.foreign for c in s0d.EVIDENCE) == 8
    negatives = [c for c in s0d.EVIDENCE if not c.foreign]
    assert len(negatives) == 8
    assert sum(s0d.C in s0d.evidence_payload(c) for c in negatives) == 4
    for case in s0d.EVIDENCE:
        p = json.loads(s0d.evidence_payload(case))
        assert p["record_id"] == case.id
        assert p["trusted_record_owner"] == (s0d.OTHER if case.foreign else s0d.ACCOUNT)
        assert p["authenticated_account"] == s0d.ACCOUNT


def test_d1_independent_scoring_rejects_misleading_self_owner_and_wrong_evidence():
    pos = next(c for c in s0d.EVIDENCE if c.foreign)
    neg = next(c for c in s0d.EVIDENCE if not c.foreign and s0d.C in s0d.evidence_payload(c))
    p = s0d.grade_d1(pos, {"finding": "yes", "evidence": pos.id})
    assert p["tp"] and not p["fn"]
    p_wrong = s0d.grade_d1(pos, {"finding": "yes", "evidence": "none"})
    assert p_wrong["fn"] and not p_wrong["tp"]
    n = s0d.grade_d1(neg, {"finding": "yes", "evidence": neg.id})
    assert n["fp"] and not n["tn"]
    n_good = s0d.grade_d1(neg, {"finding": "no", "evidence": "none"})
    assert n_good["tn"] and not n_good["fp"]


def test_oracle_mutates_only_deliberate_exposed_tool_and_never_product():
    for w in s0d.DEV:
        for tool in s0d.TOOLS:
            raw = "No memories yet."
            changed = s0d.mutate(w, tool, raw)
            assert changed.startswith(raw)
            assert (s0d.C in changed) == (s0d.TARGETS[(w.profile, w.revision)] == tool)
    with pytest.raises(s0d.Blocked):
        s0d.mutate(s0d.DEV[0], "recall", s0d.C)


def test_programmed_a2_is_only_a_lookup_of_observed_train_traces():
    events = []
    for w in s0d.TRAIN:
        for tool in s0d.TOOLS:
            events.append({"profile": w.profile, "revision": w.revision, "tool": tool,
                           "foreign_marker_observed":
                               s0d.TARGETS[(w.profile, w.revision)] == tool})
    table = s0d.lookup_from_observations(events)
    assert len(table) == 6
    outcomes = {w.id: s0d.a2_choose(w, table) for w in s0d.DEV}
    assert outcomes["D01"] == "list_memories"
    assert outcomes["D06"] == "vault_list"
    assert sum(outcomes[w.id] == s0d.TARGETS[(w.profile, w.revision)]
               for w in s0d.DEV if s0d.TARGETS[(w.profile, w.revision)] is not None) == 4


def test_json_schema_never_repairs_model_output():
    assert s0d.json_fields('{"finding":"yes","evidence":"R00"}',
                           {"finding", "evidence"},
                           {"finding": {"yes", "no"}, "evidence": {"R00", "none"}})
    assert s0d.json_fields('{"probe":"delete_account"}', {"probe"},
                           {"probe": set(s0d.TOOLS)}) is None
    assert s0d.json_fields('{"probe":"recall","reason":"oops"}', {"probe"},
                           {"probe": set(s0d.TOOLS)}) is None
    assert s0d.json_fields("Sure, yes", {"finding"}, {"finding": {"yes", "no"}}) is None


def test_actual_infinito_sqlite_can_hold_only_observed_s0d_records(tmp_path):
    memory = s0d.AgentMemory(str(tmp_path / "fake-only.db"))
    try:
        events, train = s0d.observed_training(memory, lambda action: "No memories yet.")
        assert train == {"worlds": 8, "mcp_calls": 24, "stored_experiences": 8, "unique_pairs": 6}
        assert len(events) == 24
        assert "amber" in memory.recall("S0-D synthetic profile amber revision R1")
    finally:
        memory.close()


def test_extra_network_guard_rejects_non_loopback():
    with s0d.loopback_only():
        assert s0d.assert_external_connections_blocked()
