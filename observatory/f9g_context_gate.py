"""Paired native development trajectories; observer geometry never reaches writer.

This is a representational gate, not an autonomous policy success evaluation.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX
from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9_external_minigrid import seed_for
from observatory.f9g_context_memory import ContextF4, CAPACITY, DIM


def route(env, target, direction):
    """Test-fixture BFS only: identical geometry-driven actions for both members."""
    base = env.unwrapped
    start = (*map(int, base.agent_pos), int(base.agent_dir))
    queue = collections.deque([(start, [])])
    seen = {start}
    exits = {tuple(base.success_pos), tuple(base.failure_pos)}
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))
    while queue:
        (x, y, d), actions = queue.popleft()
        if (x, y) == target and d == direction:
            return actions
        choices = [(0, (x, y, (d-1) % 4)), (1, (x, y, (d+1) % 4))]
        dx, dy = dirs[d]
        cell = base.grid.get(x+dx, y+dy)
        if (x+dx, y+dy) not in exits and (cell is None or cell.can_overlap()):
            choices.append((2, (x+dx, y+dy, d)))
        for action, state in choices:
            if state not in seen:
                seen.add(state)
                queue.append((state, actions+[action]))
    raise AssertionError("fixture route unreachable")


def paired(size, seed):
    envs = [build_env(size), build_env(size)]
    writers = [{m: ContextF4(m) for m in ("f4", "zero", "erase", "swap")}
               for _ in range(2)]
    trace, first, both_seen = [], None, [set(), set()]
    try:
        raws = [e.reset(seed=seed)[0] for e in envs]
        labels = get_world_labels(envs[0])
        raws[1], _ = swap_cue(envs[1], get_world_labels(envs[1]))
        # Full-grid equality except the intervened cue, before any transition.
        ga, gb = [e.unwrapped.grid.encode() for e in envs]
        changed = np.any(ga != gb, axis=-1)
        assert int(changed.sum()) == 1
        assert changed[1, size//2-1]
        assert envs[0].action_space.n == envs[1].action_space.n == 7

        def record(stage, action=None):
            nonlocal first
            outputs = [{m: w.encode(raws[j]) for m, w in writers[j].items()}
                       for j in range(2)]
            raw_equal = np.array_equal(*raws)
            if not raw_equal and first is None:
                first = len(trace)
            readouts = [o["f4"][148:] for o in outputs]
            different = not np.array_equal(*readouts)
            if first is not None:
                assert different, "cue distinction lost before/at fork"
            for j in range(2):
                for m in outputs[j]:
                    assert outputs[j][m].shape == (DIM,)
                    np.testing.assert_array_equal(outputs[j][m][:148], raws[j])
                assert not outputs[j]["zero"][148:].any()
                assert not outputs[j]["erase"][148:].any()
                assert writers[j]["zero"].store.order == []
                assert writers[j]["erase"].store.saved == writers[j]["f4"].store.saved
                expected = readouts[j].reshape(CAPACITY, 8).copy()
                expected[:, [0, 1]] = expected[:, [1, 0]]
                np.testing.assert_array_equal(outputs[j]["swap"][148:], expected.ravel())
                for obj in np.rint(raws[j][:147].reshape(-1, 3)[:, 0] *
                                   max(OBJECT_TO_IDX.values())).astype(int):
                    if obj in (OBJECT_TO_IDX["key"], OBJECT_TO_IDX["ball"]):
                        both_seen[j].add(int(obj))
            trace.append(dict(stage=stage, action=action, raw_equal=raw_equal,
                              readout_different=different,
                              occupancy=[len(w["f4"].store.order) for w in writers],
                              events_written=[w["f4"].serial for w in writers],
                              readouts=[r.tolist() for r in readouts]))

        def step(action, stage):
            nonlocal raws
            transitions = [e.step(action) for e in envs]
            assert all(not t[2] and not t[3] for t in transitions)
            assert transitions[0][1:4] == transitions[1][1:4]
            assert tuple(envs[0].unwrapped.agent_pos) == tuple(envs[1].unwrapped.agent_pos)
            assert envs[0].unwrapped.agent_dir == envs[1].unwrapped.agent_dir
            raws = [t[0] for t in transitions]
            record(stage, action)

        record("reset")
        for target, direction, stage in [
            ((2, size//2), 3, "inspect"),
            ((labels["success_position"][0], size//2), 0, "approach_fork"),
        ]:
            for action in route(envs[0], target, direction):
                step(action, stage)
        record("fork")
        # Inspect all directions using only native turns, including both exit objects.
        for _ in range(4):
            step(0, "fork_scan")
        assert first is not None, "fixture never observed cue"
        assert all(len(s) == 2 for s in both_seen), "both key and ball must be observed"
        assert trace[-1]["raw_equal"], "final view must hide the intervened cue"
        assert trace[-1]["readout_different"]
        assert all(w["f4"].serial <= CAPACITY for w in writers), "fixture exceeds declared horizon"
        return dict(size=size, seed=seed, first_cue_observation=first,
                    both_objects_seen=True, gate_pass=True, trace=trace)
    finally:
        for env in envs:
            env.close()


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = [paired(size, seed_for("development", index, episode))
               for size in (7, 13) for index in range(4) for episode in range(10)]
    traces = output / "paired_traces.json"
    traces.write_text(json.dumps(records), encoding="utf-8")
    report = dict(gate_pass=all(r["gate_pass"] for r in records),
                  heldout_opened=False, training_steps=0, native_actions=7,
                  capacity=CAPACITY, input_dimension=DIM,
                  source_sha=os.environ.get("GITHUB_SHA", "local"),
                  dependencies={p: importlib.metadata.version(p)
                                for p in ("numpy", "gymnasium", "minigrid")},
                  trace_sha256=hashlib.sha256(traces.read_bytes()).hexdigest(),
                  by_size={str(size): dict(pairs=sum(r["size"] == size for r in records),
                            passed=sum(r["gate_pass"] for r in records if r["size"] == size),
                            max_events=max(max(t["events_written"])
                                           for r in records if r["size"] == size for t in r["trace"]))
                           for size in (7, 13)})
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    assert report["gate_pass"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="observatory_results/f9g")
    run(parser.parse_args().output)
