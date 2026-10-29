"""F9-C controlled memory assay using real MiniGrid Memory images, PPO and LSTM.

This is intentionally NOT full MiniGrid navigation: the camera is placed at
the visible start cue and then the split; the policy's only scored decision
is upper/lower. It isolates whether past-only information affects decisions.
No hidden cue or target labels are supplied to the model.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

from observatory.f9_external_minigrid import ensure_dependencies, file_sha, seed_for
from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue

MODES = ("ppo", "lstm")
TRAIN_STEPS = (20_000, 100_000, 300_000)


class CueForkEnv:
    """Gym-compatible diagnostic with native external MiniGrid observations."""

    def __init__(self, size=7, *, reverse_cue=False):
        import gymnasium as gym
        from gymnasium import spaces
        if size not in (7, 13):
            raise ValueError("Only MiniGrid Memory S7 and S13")
        self.size = size
        self.reverse_cue = reverse_cue
        self.world = build_env(size)
        self.observation_space = self.world.observation_space
        self.action_space = spaces.Discrete(2)
        self.phase = 0
        self.labels = None
        self._current_obs = None

    def reset(self, *, seed=None, options=None):
        import numpy as np
        self.world.reset(seed=seed)
        original_labels = get_world_labels(self.world)
        if self.reverse_cue:
            _unused, labels = swap_cue(self.world, original_labels)
        else:
            labels = original_labels
        self.labels = labels
        self.phase = 0
        base = self.world.unwrapped
        # Authentic cue at (1,mid-1); north-facing agent at (2,mid).
        # This controlled viewpoint is checked, not silently assumed visible.
        base.agent_pos = np.array((2, self.size // 2))
        base.agent_dir = 3
        observation = self.world.observation(base.gen_obs())
        self._current_obs = observation
        return observation, {"diagnostic_stage": "cue"}

    def step(self, action):
        import numpy as np
        if int(action) not in (0, 1):
            raise ValueError("Two-choice diagnostic actions only")
        if self.phase == 0:
            self.phase = 1
            base = self.world.unwrapped
            # At the real fork, face NORTH: the upper candidate object must
            # appear in the observation. The lower candidate and old cue are
            # behind the camera, so matching still requires initial memory.
            base.agent_pos = np.array((self.labels["success_position"][0],
                                       self.size // 2))
            base.agent_dir = 3
            self._current_obs = self.world.observation(base.gen_obs())
            return self._current_obs, 0., False, False, {
                "diagnostic_stage": "decision"}
        if self.phase != 1:
            raise RuntimeError("Reset before stepping after episode terminates")
        self.phase = 2
        chosen = "upper" if int(action) == 0 else "lower"
        reward = float(chosen == self.labels["correct_side"])
        return self._current_obs, reward, True, False, {
            "diagnostic_stage": "finished",
            "choice": chosen,
            "correct": bool(reward),
        }

    def close(self):
        self.world.close()


def make_env(size=7, reverse_cue=False):
    import gymnasium as gym

    class Wrapped(gym.Env):
        metadata = {}
        def __init__(self):
            self.inner = CueForkEnv(size, reverse_cue=reverse_cue)
            self.observation_space = self.inner.observation_space
            self.action_space = self.inner.action_space

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.inner.reset(seed=seed, options=options)

        def step(self, action):
            return self.inner.step(action)

        def close(self):
            self.inner.close()

    return Wrapped()


def compare_pair(seed, size=7):
    """Assert cue visibility and absence at decision, before training."""
    import numpy as np
    frames = []
    labels = []
    for reverse in (False, True):
        env = make_env(size, reverse_cue=reverse)
        try:
            cue, _ = env.reset(seed=seed)
            fork, _, terminated, _, _ = env.step(0)
            assert not terminated
            frames.append((cue, fork))
            labels.append(env.inner.labels)
        finally:
            env.close()
    seen_change = not np.array_equal(frames[0][0], frames[1][0])
    fork_equal = np.array_equal(frames[0][1], frames[1][1])
    if not seen_change or not fork_equal:
        raise AssertionError(
            f"Invalid memory diagnostic seed {seed}: cue_visible={seen_change} fork_equal={fork_equal}")
    if labels[0]["correct_side"] == labels[1]["correct_side"]:
        raise AssertionError("Cue intervention failed to flip the correct target")
    from minigrid.core.constants import OBJECT_TO_IDX
    object_indices = np.rint(frames[0][1][:147].reshape(7,7,3)[:,:,0] *
                            max(OBJECT_TO_IDX.values())).astype(int)
    upper_visible = int(OBJECT_TO_IDX[labels[0]["upper_object"]]) in object_indices
    if not upper_visible:
        raise AssertionError(f"Fork candidate is NOT visible in seed {seed}: invalid diagnostic")
    return {"initial_cue_changed": seen_change,
            "fork_observation_identical": fork_equal,
            "upper_candidate_visible": upper_visible}


def train(variant, steps, index, out):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    if variant not in MODES:
        raise ValueError(variant)
    env = make_env(7)
    seed = seed_for("train", index)
    common = {
        "env": env, "seed": seed, "device": "cpu",
        "n_steps": 128, "batch_size": 128, "n_epochs": 4,
        "gamma": .99, "gae_lambda": .95,
        "learning_rate": 3e-4, "ent_coef": .01, "verbose": 0,
    }
    model = (RecurrentPPO(
        "MlpLstmPolicy", **common,
        policy_kwargs={"lstm_hidden_size": 64,
                       "n_lstm_layers": 1, "net_arch": dict(pi=[64], vf=[64])})
        if variant == "lstm" else
        PPO("MlpPolicy", **common,
            policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])}))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"f9c_memory_s7_{variant}_seed{index}.zip"
    started = time.monotonic()
    try:
        model.learn(total_timesteps=steps)
        model.save(path)
        return {"file": path.name, "sha256": file_sha(path),
                "parameters": sum(x.numel() for x in model.policy.parameters()),
                "steps": int(model.num_timesteps),
                "training_seconds": round(time.monotonic() - started, 3)}
    finally:
        model.get_env().close()


def rollout(model, recurrent, size, seed, reverse=False):
    import numpy as np
    env = make_env(size, reverse_cue=reverse)
    state = None
    first = True
    history = []
    try:
        observation, _ = env.reset(seed=seed)
        truth = dict(env.inner.labels)
        done = False
        while not done:
            if recurrent:
                action, state = model.predict(
                    observation, state=state,
                    episode_start=np.array([first], dtype=bool),
                    deterministic=True)
            else:
                action, _ = model.predict(observation, deterministic=True)
            action = int(action)
            history.append(action)
            observation, reward, terminated, truncated, _ = env.step(action)
            first = False
            done = terminated or truncated
        assert len(history) == 2
        return {"seed": seed, "cue": truth["cue"],
                "correct_side": truth["correct_side"],
                "chosen_side": "upper" if history[-1] == 0 else "lower",
                "cue_action": history[0], "decision_action": history[-1],
                "success": int(reward > 0)}
    finally:
        env.close()


def evaluate_checkpoint(checkpoint: Path, recurrent, size, episodes, index=0):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    cls = RecurrentPPO if recurrent else PPO
    model = cls.load(checkpoint, device="cpu")
    observations = []
    for j in range(episodes):
        seed = seed_for("development", index, j)
        pair_check = compare_pair(seed, size)
        a = rollout(model, recurrent, size, seed, False)
        b = rollout(model, recurrent, size, seed, True)
        observations.append({"original": a, "counterfactual": b})
    original = [x["original"] for x in observations]
    altered = [x["counterfactual"] for x in observations]
    return {
        "episodes": episodes,
        "original_success_rate": statistics.mean(x["success"] for x in original),
        "counterfactual_success_rate": statistics.mean(x["success"] for x in altered),
        "decision_flips": sum(
            a["decision_action"] != b["decision_action"]
            for a, b in zip(original, altered)),
        "correct_on_both": sum(a["success"] and b["success"]
                               for a, b in zip(original, altered)),
        "original_cue_counts": dict(collections.Counter(x["cue"] for x in original)),
        "original_branch_counts": dict(collections.Counter(
            x["chosen_side"] for x in original)),
        "counterfactual_branch_counts": dict(collections.Counter(
            x["chosen_side"] for x in altered)),
        "original_success_by_cue": {
            cue: {"successes": sum(x["success"] for x in original if x["cue"] == cue),
                  "episodes": sum(x["cue"] == cue for x in original)}
            for cue in ("ball", "key")
        },
        "episode_pairs": observations,
    }


def run(budget, seeds, episodes, out):
    import torch
    torch.set_num_threads(1)
    deps = ensure_dependencies()
    for size in (7, 13):
        for k in range(20):
            compare_pair(seed_for("development", 0, k), size)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    results = []
    manifest = {}
    for i in range(seeds):
        row = {}
        for variant in MODES:
            details = train(variant, budget, i, checkpoints)
            manifest[f"{variant}_{i}"] = details
            cp = checkpoints / details["file"]
            row[variant] = {
                "S7": evaluate_checkpoint(cp, variant == "lstm", 7, episodes, index=i),
                "S13": evaluate_checkpoint(cp, variant == "lstm", 13, min(30,episodes), index=i),
            }
            for size, data in row[variant].items():
                print("F9C",variant,"seed",i,size,
                      "success",data["original_success_rate"],
                      "counterfactual",data["counterfactual_success_rate"],
                      "flips",data["decision_flips"],"/",data["episodes"],flush=True)
        results.append(row)
    report = {
        "experiment": "F9-C controlled externally sourced cue-retention-choice diagnostic",
        "phase": "development_diagnostic",
        "not_original_navigation_benchmark": True,
        "cue_exposure": "forced camera viewpoint from native MiniGrid room",
        "fork_navigation": "teleport to native fork, choose upper/lower (2 actions)",
        "heldout_opened": False,
        "training_budget_per_policy": budget,
        "training_seeds": seeds,
        "development_episodes_s7": episodes,
        "development_episodes_s13": min(30,episodes),
        "source_sha256": file_sha(Path(__file__)),
        "dependencies": deps,
        "checkpoint_manifest": manifest,
        "per_seed": results,
        "limitations": (
            "Two-frame memory assay using authentic MiniGrid view and target mapping, "
            "NOT real 7-action navigation. The policy is teleported between views. "
            "A good result does not prove navigation/generalization or INFINITO advantage. "
            "One training seed is exploratory. Reserved heldout is never opened."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "frozen_manifest.json").write_text(json.dumps({
        "stage": "f9c_development_diagnostic",
        "models": {key:{"model_path": value["file"],
                         "model_sha256": value["sha256"],
                         "parameters": value["parameters"],
                         "training_seconds": value["training_seconds"]}
                   for key,value in manifest.items()},
        "source_sha256": report["source_sha256"],
        "config": {"seeds": seeds, "timesteps": budget, "eval_episodes": episodes},
        "heldout_opened": False,
    }, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timesteps not in TRAIN_STEPS or not 1 <= args.seeds <= 5:
        parser.error("Use 20k/100k/300k steps and 1..5 training seeds")
    if not 10 <= args.eval_episodes <= 200:
        parser.error("10..200 development episodes")
    result = run(args.timesteps, args.seeds, args.eval_episodes, args.output)
    print("F9C_RESULT_S7",json.dumps({
        v: {"success":result["per_seed"][0][v]["S7"]["original_success_rate"],
            "cue_swap_flips":result["per_seed"][0][v]["S7"]["decision_flips"]}
        for v in MODES}),flush=True)


if __name__ == "__main__":
    main()
