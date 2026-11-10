"""S0-I-R runtime-only sharding tests; synthetic fake results are NOT science.

Reject missing worlds, repeated arms, falsified A2 lookup, wrong pinned GGUF,
unequal actual Qwen prompt tokens and inconsistent replicated TRAIN histories.
"""
import copy
import json

import pytest

from security_lab import s0i, s0i_parallel as parallel
from security_lab.s0c import MODEL_REPO, MODEL_FILE, MODEL_REVISION, MODEL_SHA256


def fake_training_events():
    result = []
    for w in s0i.TRAIN:
        for tool in s0i.TOOLS:
            result.append({
                "world": w.id, "profile": w.profile,
                "revision": w.revision, "tool": tool,
                "marker_observed": tool == s0i.private_target(w.profile, w.revision),
            })
    return result


def fake_shard(stage, index):
    worlds = parallel.expected_worlds(stage, index)
    observed = fake_training_events()
    lookup = s0i.literal_lookup(observed)
    rows = []
    for w in worlds:
        for arm in parallel.ARMS:
            chosen = (
                s0i.private_target(w.profile, w.revision) or "recall"
                if arm == "A1" else
                s0i.lookup_choose(lookup, w) if arm == "A2" else "recall"
            )
            found = s0i.private_target(w.profile, w.revision) == chosen and w.profile != "quiet"
            rows.append({
                "world": w.id, "profile": w.profile, "revision": w.revision,
                "arm": arm, "selected": chosen, "qwen_suggestion": chosen,
                "valid_json": True, "target_present": w.profile != "quiet",
                "correct_probe": bool(found), "marker_observed": bool(found),
                "mcp_calls": 1, "prompt_tokens": 1000, "completion_tokens": 8,
                "token_budget_matched": True,
                "retrieved_real_sqlite": True,
            })
    return {
        "study": parallel.STUDY, "status": "COMPLETED_SHARD",
        "stage": stage, "shard_index": index,
        "shard_count": parallel.SHARDS[stage],
        "original_world_ids": [w.id for w in worlds],
        "model": {
            "real": True, "repo": MODEL_REPO, "file": MODEL_FILE,
            "revision": MODEL_REVISION, "weights_sha256": MODEL_SHA256,
            "backend": "llama-cpp-python", "backend_version": "0.3.36",
            "temperature": 0.0, "seed": 0,
            "calls": len(rows), "prompt_tokens": len(rows) * 1000,
            "completion_tokens": len(rows) * 8,
        },
        "train": parallel.EXPECTED_TRAIN,
        "train_events": observed,
        "genuine_oauth_mcp_calls": 36 + len(rows),
        "external_connections_blocked": True,
        "traces": rows,
    }


def files_for(stage, tmp_path):
    paths = []
    for index in range(parallel.SHARDS[stage]):
        path = tmp_path / f"shard-{index}" / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fake_shard(stage, index)), encoding="utf-8")
        paths.append(path)
    return paths


def test_identical_frozen_world_partition_no_overlap_and_total_cases():
    for stage, worlds in parallel.STAGES.items():
        shards = [parallel.expected_worlds(stage, i) for i in range(parallel.SHARDS[stage])]
        flattened = [w.id for part in shards for w in part]
        assert len(flattened) == len(worlds)
        assert set(flattened) == {w.id for w in worlds}
        assert len(flattened) == len(set(flattened))
        assert len(set(len(shard) for shard in shards)) == 1
    assert [len(parallel.expected_worlds("pilot", i)) for i in range(3)] == [8] * 3
    assert [len(parallel.expected_worlds("confirm", i)) for i in range(8)] == [6] * 8


@pytest.mark.parametrize("stage", ["pilot", "confirm"])
def test_aggregation_applies_original_gates_and_original_world_order(tmp_path, stage):
    result = parallel.aggregate(stage, files_for(stage, tmp_path))
    assert result["status"] == "COMPLETED"
    assert result["gate"]
    assert result["worlds"] == len(parallel.STAGES[stage])
    assert result["model_calls"] == 3 * len(parallel.STAGES[stage])
    assert result["genuine_mcp_eval_calls"] == result["model_calls"]
    assert result["train_replay_mcp_calls"] == 36 * parallel.SHARDS[stage]
    assert [(x["world"], x["arm"]) for x in result["traces"]] == [
        (w.id, a) for w in parallel.STAGES[stage] for a in parallel.ARMS
    ]
    assert result["metrics"]["A1"]["correct_probes"] > result["metrics"]["A2"]["correct_probes"]
    # The acceptance decision must use the frozen S0-I functions.
    if stage == "pilot":
        assert result["gate"] == s0i.pilot_gate(result["metrics"])
    else:
        assert result["gate"] == s0i.confirm_gate(result["metrics"])


def test_missing_or_duplicate_shard_fails_closed(tmp_path):
    paths = files_for("pilot", tmp_path)
    with pytest.raises(ValueError, match="Missing"):
        parallel.aggregate("pilot", paths[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        parallel.aggregate("pilot", [paths[0], paths[1], paths[1]])


@pytest.mark.parametrize("alteration", ["world", "arm", "owner", "hash", "mcp", "unmatched", "train", "a2", "mismatch", "result"])
def test_tampered_shard_is_rejected(tmp_path, alteration):
    shard = fake_shard("pilot", 0)
    if alteration == "world":
        shard["traces"][0]["world"] = "INJECTED"
    elif alteration == "arm":
        shard["traces"][0]["arm"] = "A9"
    elif alteration == "owner":
        shard["traces"][0]["profile"] = "injected"
    elif alteration == "hash":
        shard["model"]["weights_sha256"] = "0" * 64
    elif alteration == "mcp":
        shard["genuine_oauth_mcp_calls"] -= 1
    elif alteration == "unmatched":
        shard["traces"][0]["prompt_tokens"] = 250
        shard["model"]["prompt_tokens"] -= 750
    elif alteration == "train":
        shard["train_events"][0]["marker_observed"] = not shard["train_events"][0]["marker_observed"]
    elif alteration == "a2":
        row = next(r for r in shard["traces"] if r["arm"] == "A2")
        row["selected"] = next(t for t in s0i.TOOLS if t != row["selected"])
    elif alteration == "mismatch":
        row = next(r for r in shard["traces"] if r["arm"] == "A1")
        row["qwen_suggestion"] = "vault_list" if row["selected"] != "vault_list" else "recall"
    elif alteration == "result":
        shard["traces"][0]["correct_probe"] = not shard["traces"][0]["correct_probe"]
    with pytest.raises(ValueError):
        parallel.validate_shard(shard, "pilot")


def test_aggregate_science_can_be_no_go_even_if_tech_correct(tmp_path):
    paths = files_for("pilot", tmp_path)
    for path in paths:
        report = json.loads(path.read_text())
        for row in report["traces"]:
            if row["arm"] == "A1":
                row["selected"] = "recall"
                row["qwen_suggestion"] = "recall"
                found = (
                    s0i.private_target(row["profile"], row["revision"]) == "recall"
                )
                row["marker_observed"] = bool(found)
                row["correct_probe"] = bool(found)
        path.write_text(json.dumps(report), encoding="utf-8")
    answer = parallel.aggregate("pilot", paths)
    assert answer["status"] == "COMPLETED"
    assert answer["gate"] is False
    assert "NO-GO" in answer["decision"]


def test_bad_stage_and_index_refused():
    for stage, index in (("pilot", 3), ("confirm", 8), ("not-a-stage", 0), ("pilot", -1)):
        with pytest.raises(ValueError):
            parallel.expected_worlds(stage, index)
