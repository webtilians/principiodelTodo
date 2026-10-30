"""F9-B post-training BEHAVIOR audit (not heldout, never retrains).

Load 100k and 300k frozen development checkpoints. Trace actions and branch
choices on existing development seeds, intervene on the start-room cue while
holding everything else constant, then zero-shot transfer to Memory S13.
The hidden simulator labels are for *measurement only*, never policy inputs.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

from observatory.f9_external_minigrid import file_sha, seed_for
from observatory.f9b_causal_memory import load_locked

TASKS = {7: "MiniGrid-MemoryS7-v0", 13: "MiniGrid-MemoryS13-v0"}
ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def build_env(size: int):
    """Same partial observation normalization as the original F9-A baselines."""
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
    from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

    if size not in TASKS:
        raise ValueError("Only S7 and S13 memory environments are audited")
    obj_max = float(max(OBJECT_TO_IDX.values()))
    color_max = float(max(COLOR_TO_IDX.values()))

    class StandardPartialView(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(148,), dtype=np.float32)

        def observation(self, obs):
            image = obs["image"].astype(np.float32).copy()
            if image.shape != (7, 7, 3):
                raise ValueError("Observation must be partial 7x7 symbolic view")
            image[:, :, 0] /= obj_max
            image[:, :, 1] /= color_max
            image[:, :, 2] /= 3.
            compass = np.array([float(obs["direction"]) / 3.], dtype=np.float32)
            return np.concatenate((image.reshape(-1), compass)).astype(np.float32)

    return StandardPartialView(gym.make(TASKS[size]))


def get_world_labels(env):
    """Observer-only simulator labels; absolutely never passed to the policy."""
    base = env.unwrapped
    mid = base.height // 2
    cue = base.grid.get(1, mid - 1)
    if cue is None or cue.type not in ("key", "ball"):
        raise AssertionError("Unexpected MiniGrid Memory initial cue")
    upper_y = mid - 2
    lower_y = mid + 2
    end_x = tuple(base.success_pos)[0]
    upper = base.grid.get(end_x, upper_y)
    lower = base.grid.get(end_x, lower_y)
    if not upper or not lower:
        raise AssertionError("End-of-hall objects missing")
    return {
        "cue": cue.type,
        "start_position": [int(x) for x in base.agent_pos],
        "start_direction": int(base.agent_dir),
        "upper_object": upper.type,
        "lower_object": lower.type,
        "correct_side": "upper" if base.success_pos[1] < mid else "lower",
        "success_position": [int(x) for x in base.success_pos],
        "failure_position": [int(x) for x in base.failure_pos],
    }


def swap_cue(env, original_labels):
    """Controlled intervention: change only start object and corresponding goal.

    All walls, starting position, candidate exit objects, RNG seed, compass,
    and policies are held fixed. The new reward condition is logically
    consistent with the changed cue; this is *not* a heldout test.
    """
    from minigrid.core.world_object import Ball, Key
    base = env.unwrapped
    pos = (1, base.height // 2 - 1)
    old = base.grid.get(*pos)
    if old.type not in ("key", "ball"):
        raise AssertionError("Wrong cue object")
    base.grid.set(*pos, Ball("green") if old.type == "key" else Key("green"))
    base.success_pos, base.failure_pos = (
        tuple(base.failure_pos), tuple(base.success_pos))
    new = get_world_labels(env)
    assert new["cue"] != original_labels["cue"]
    assert new["correct_side"] != original_labels["correct_side"]
    for name in ("start_position", "start_direction", "upper_object", "lower_object"):
        assert new[name] == original_labels[name]
    # Apply the existing observation wrapper to the *new* visible raw grid.
    obs = env.observation(base.gen_obs())
    return obs, new


def rollout(model, recurrent: bool, size: int, episode_seed: int,
            swap: bool = False):
    import numpy as np
    env = build_env(size)
    try:
        obs, _ = env.reset(seed=episode_seed)
        labels = get_world_labels(env)
        before = obs.copy()
        if swap:
            obs, labels = swap_cue(env, labels)
        initial_visible_changed = not np.array_equal(obs, before)
        # Explicitly instrument every action, including first at the fork.
        counts = [0] * len(ACTION_NAMES)
        trajectory = []
        memory = None
        first = True
        total = 0.
        done = False
        success = False
        while not done:
            if recurrent:
                action, memory = model.predict(
                    obs, state=memory,
                    episode_start=np.array([first], dtype=bool),
                    deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            if action < 0 or action >= len(ACTION_NAMES):
                raise ValueError("Invalid MiniGrid action")
            old_position = [int(x) for x in env.unwrapped.agent_pos]
            counts[action] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            new_position = [int(x) for x in env.unwrapped.agent_pos]
            total += float(reward)
            success |= reward > 0
            trajectory.append({
                "step": len(trajectory) + 1,
                "position_before": old_position,
                "action": action,
                "position_after": new_position,
                "reward": float(reward),
            })
            first = False
            done = bool(terminated or truncated)
        end = [int(x) for x in env.unwrapped.agent_pos]
        if end == labels["success_position"]:
            side = labels["correct_side"]
            terminal_reason = "correct_exit"
        elif end == labels["failure_position"]:
            side = ("lower" if labels["correct_side"] == "upper" else "upper")
            terminal_reason = "wrong_exit"
        elif truncated:
            side = "none"
            terminal_reason = "timeout"
        else:
            side = "none"
            terminal_reason = "other_termination"
        return {
            "episode_seed": episode_seed, "environment": TASKS[size],
            "intervention": "cue_swapped" if swap else "original",
            "cue": labels["cue"], "correct_side": labels["correct_side"],
            "upper_object": labels["upper_object"],
            "lower_object": labels["lower_object"],
            "starting_position": labels["start_position"],
            "initial_observation_changed_by_swap": initial_visible_changed,
            "chosen_side": side, "success": int(success),
            "return": total, "length": len(trajectory),
            "terminal_reason": terminal_reason,
            "action_counts": dict(zip(ACTION_NAMES, counts)),
            "trajectory": trajectory,
        }
    finally:
        env.close()


def check_manifest(folder: Path):
    # This enforces SAME F9-B source, deps and checkpoint hashes as when
    # models were frozen, rather than loading arbitrary untrusted ZIPs.
    manifest = load_locked(folder, "f9b_baseline_development")
    if manifest.get("task") != "memory" or manifest["config"]["seeds"] != 1:
        raise ValueError("Audit currently requires exactly one trained seed")
    if manifest.get("heldout_opened") is not False:
        raise ValueError("Auditing must not use any opened heldout material")
    if manifest.get("eligible_for_heldout", False):
        raise ValueError("Do not use this exploratory audit for heldout selection")
    return manifest


def evaluate_models(roots: dict[str, Path], size: int, episodes: int):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    if not 1 <= episodes <= 100:
        raise ValueError("Only 1..100 developmental episodes per size permitted")
    rows = {}
    meta = {}
    for budget, directory in roots.items():
        manifest = check_manifest(directory)
        meta[budget] = {
            "manifest_sha256": file_sha(directory / "frozen_manifest.json"),
            "timesteps": manifest["config"]["timesteps"],
            "source_sha256": manifest["source_sha256"],
            "models": {},
        }
        for name, recurrent, cls in (
            ("ppo", False, PPO), ("lstm", True, RecurrentPPO)
        ):
            details = manifest["models"][f"{name}_0"]
            checkpoint = directory / "checkpoints" / details["model_path"]
            model = cls.load(checkpoint, device="cpu")
            meta[budget]["models"][name] = {
                "model_sha256": details["model_sha256"],
                "parameters": details["parameters"],
            }
            original, changed = [], []
            for i in range(episodes):
                seed = seed_for("development", 0, i)
                original.append(rollout(model, recurrent, size, seed))
                changed.append(rollout(model, recurrent, size, seed, swap=True))
                if (i + 1) % 10 == 0:
                    print("PROGRESS",budget,name,"S"+str(size),
                          i + 1,"/",episodes,flush=True)
            rows[f"{budget}:{name}"] = {"original": original, "cue_swapped": changed}
    return meta, rows


def reduce_rows(rows: dict):
    import numpy as np
    summary = {}
    for policy, records in rows.items():
        original, swapped = records["original"], records["cue_swapped"]
        n = len(original)
        pairs = list(zip(original, swapped))
        for a, b in pairs:
            assert a["episode_seed"] == b["episode_seed"]
            assert a["upper_object"] == b["upper_object"]
            assert a["lower_object"] == b["lower_object"]
            assert a["starting_position"] == b["starting_position"]
            assert a["correct_side"] != b["correct_side"]
        def rate(items):
            return sum(e["success"] for e in items)/len(items)
        summary[policy] = {
            "episodes": n,
            "original_success_rate": rate(original),
            "cue_swapped_success_rate": rate(swapped),
            "original_mean_return": statistics.mean(x["return"] for x in original),
            "original_mean_length": statistics.mean(x["length"] for x in original),
            "swap_changed_initial_partial_observation": sum(
                x["initial_observation_changed_by_swap"] for x in swapped),
            "chosen_side_original": dict(collections.Counter(
                x["chosen_side"] for x in original)),
            "chosen_side_swapped": dict(collections.Counter(
                x["chosen_side"] for x in swapped)),
            "chosen_side_changed_in_pairs": sum(
                a["chosen_side"] != b["chosen_side"] for a, b in pairs),
            "matched_correct_side_original": sum(
                a["chosen_side"] == a["correct_side"] for a in original),
            "matched_correct_side_swapped": sum(
                a["chosen_side"] == a["correct_side"] for a in swapped),
            "successful_original_by_cue": {
                cue: {"successes": sum(x["success"] for x in original if x["cue"] == cue),
                      "episodes": sum(x["cue"] == cue for x in original)}
                for cue in ("key", "ball")
            },
            "original_action_counts": dict(collections.Counter({
                name:sum(e["action_counts"][name] for e in original)
                for name in ACTION_NAMES})),
            "swapped_action_counts": dict(collections.Counter({
                name:sum(e["action_counts"][name] for e in swapped)
                for name in ACTION_NAMES})),
            "first_action_changed_by_swap": sum(
                a["trajectory"][0]["action"] != b["trajectory"][0]["action"]
                for a, b in pairs),
            "same_full_action_sequence_after_swap": sum(
                [t["action"] for t in a["trajectory"]] ==
                [t["action"] for t in b["trajectory"]]
                for a, b in pairs),
            "changed_side_when_initial_observation_differs": sum(
                a["chosen_side"] != b["chosen_side"] for a, b in pairs
                if b["initial_observation_changed_by_swap"]),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--100k", dest="baseline100", type=Path, required=True)
    parser.add_argument("--300k", dest="baseline300", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(7, 13), required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.baseline100.resolve() == args.baseline300.resolve():
        parser.error("100k and 300k must have distinct checkpoints")
    import torch
    torch.set_num_threads(1)
    manifests, records = evaluate_models(
        {"100k": args.baseline100, "300k": args.baseline300},
        args.size, args.episodes)
    if (manifests["100k"]["timesteps"] != 100_000
            or manifests["300k"]["timesteps"] != 300_000):
        raise ValueError("Expected actual 100k and 300k frozen training budgets")
    summary = reduce_rows(records)
    report = {
        "experiment": "F9-B behavioral audit of PPO/LSTM, zero new training",
        "phase": "development_diagnostic",
        "environment": TASKS[args.size], "size": args.size,
        "episodes_per_policy_per_condition": args.episodes,
        "seed_partition": "development, index 0; no heldout opened",
        "source_sha256": file_sha(Path(__file__)),
        "checkpoint_provenance": manifests,
        "summary": summary,
        "limitations": [
            "Counterfactual swaps initial cue and its true target; it keeps exit objects, geometry, start and RNG fixed.",
            "If the cue is not visible under the policy trajectory, no changed decision is expected even for an ideal memory mechanism.",
            "S13 is zero-shot transfer across corridor lengths, not test-heldout and not a trained S13 baseline.",
            "Changing the correct target changes reward, not the agent's inputs; hidden simulator state used ONLY for diagnostic labels.",
            "One training seed and paired development episodes: no evidence of statistical or scientific superiority.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    with (args.output / "trajectories.jsonl").open("w", encoding="utf8") as out:
        for policy, variants in records.items():
            for condition, episodes in variants.items():
                for episode in episodes:
                    out.write(json.dumps({
                        "policy": policy, "condition": condition, **episode,
                    },ensure_ascii=False) + "\n")
    print("F9-B_BEHAVIOR_SUMMARY", json.dumps(summary,ensure_ascii=False),flush=True)
    print("REPORT", args.output / "report.json",flush=True)


if __name__ == "__main__":
    main()
