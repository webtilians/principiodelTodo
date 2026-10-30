"""F9-B: genuine F4 Store/FIFO as a MiniGrid memory-input mechanism.

Sequential baseline gate -> controlled ablations -> sealed heldout. Does NOT
integrate F4 LearnedRelevance or F5 predictive memory (no valid key-query
feedback exists in MiniGrid Memory). These limitations are deliberate.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

from observatory.f9_external_minigrid import (
    ensure_dependencies, file_sha, grouped_bootstrap, make_env as base_env,
    seed_for,
)
from observatory.selective_memory import Event, FIFO

TASK = "memory"
STORED_SLOTS = 3
VARIANTS = ("ppo", "lstm", "ppo_f4", "lstm_f4", "lstm_zero", "lstm_f4_corrupt")
BASELINES = ("ppo", "lstm")
TRAINABLE_ABLATIONS = ("ppo_f4", "lstm_f4", "lstm_zero")
ALL_TRAINED = BASELINES + TRAINABLE_ABLATIONS
COMPETENT_SUCCESS = 0.70
MIN_RANDOM_MARGIN = 0.15
MIN_GATE_SEEDS = 4
MIN_HELDOUT_SEEDS = 5
MIN_HELDOUT_EPISODES = 500
COMPETENCE_LADDER = (20_000, 100_000, 300_000)


def make_env(mode: str = "none"):
    """All memory features originate exclusively from the original observation.

    zero: extra 12 channels are exactly zero, preserving D's input width.
    corrupt: paired intervention on a D checkpoint, altering colors in stored
             snapshots but leaving the current partial observation unchanged.
    """
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
    from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

    env = base_env(TASK)  # 148 features: current egocentric 7x7x3 + direction
    if mode == "none":
        return env
    if mode not in ("f4", "zero", "corrupt"):
        raise ValueError("Unknown memory adapter mode")

    ncolors = len(COLOR_TO_IDX)
    max_object = max(OBJECT_TO_IDX.values())
    max_color = max(COLOR_TO_IDX.values())
    ignored = {OBJECT_TO_IDX[name] for name in
               ("unseen", "empty", "wall", "floor", "agent")
               if name in OBJECT_TO_IDX}

    class F4EpisodicStore(gym.ObservationWrapper):
        def __init__(self, wrapped):
            super().__init__(wrapped)
            self.mode = mode
            self.selector = FIFO(STORED_SLOTS, seed=0)
            self.store = self.selector.begin()
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(148 + STORED_SLOTS * 4,),
                dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            self.store = self.selector.begin()
            return super().reset(seed=seed, options=options)

        def observation(self, observation):
            image = observation[:147].reshape(7, 7, 3)
            if self.mode != "zero":
                for pixel in image.reshape(-1, 3):
                    object_id = int(np.rint(pixel[0] * max_object))
                    color_id = int(np.rint(pixel[1] * max_color))
                    state_id = int(np.rint(pixel[2] * 3))
                    if object_id in ignored:
                        continue
                    if not 0 <= color_id < ncolors:
                        continue
                    # F4 Event is a visible-symbol key-value, not a future label.
                    identity = object_id * ncolors + color_id
                    self.selector.consider(
                        self.store, Event(key=identity, payload=state_id))
            memory = np.zeros((STORED_SLOTS, 4), dtype=np.float32)
            if self.mode != "zero":
                for slot, key in enumerate(self.store.order[:STORED_SLOTS]):
                    obj_id, color_id = divmod(key, ncolors)
                    if self.mode == "corrupt":
                        # Structured negative control: remove true stored color
                        # but retain object type and the same nonzero occupancy.
                        color_id = (color_id + 1) % ncolors
                    memory[slot, :] = (
                        obj_id / max_object, color_id / max_color,
                        self.store.saved[key] / 3, 1.0)
            combined = np.concatenate([observation, memory.flatten()]).astype(np.float32)
            assert self.observation_space.contains(combined)
            return combined

    return F4EpisodicStore(env)


def spec(variant):
    if variant not in VARIANTS and variant != "random":
        raise ValueError(variant)
    recurrent = variant.startswith("lstm")
    if variant == "lstm_f4_corrupt":
        return "lstm_f4", True, "corrupt"
    if variant.endswith("_f4"):
        return variant, recurrent, "f4"
    if variant == "lstm_zero":
        return variant, True, "zero"
    return variant, recurrent, "none"


def learn(variant, index, timesteps, models_dir):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    canonical, recurrent, mode = spec(variant)
    if canonical == "random" or canonical == "lstm_f4_corrupt":
        raise ValueError("Cannot train control-only policy")
    env = make_env(mode)
    seed = seed_for("train", index)
    env.reset(seed=seed)
    kwargs = dict(
        env=env, learning_rate=3e-4,
        n_steps=64, batch_size=64, n_epochs=2,
        gamma=.99, gae_lambda=.95,
        ent_coef=0.0, verbose=0, seed=seed, device="cpu")
    if recurrent:
        model = RecurrentPPO(
            "MlpLstmPolicy", **kwargs,
            policy_kwargs={"lstm_hidden_size": 64, "n_lstm_layers": 1,
                           "net_arch": dict(pi=[64], vf=[64])})
    else:
        model = PPO(
            "MlpPolicy", **kwargs,
            policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])})
    started = time.monotonic()
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.learn(total_timesteps=timesteps)
        path = models_dir / f"memory_{canonical}_s{index}.zip"
        model.save(path)
        return {
            "model_path": path.name,
            "model_sha256": file_sha(path),
            "parameters": int(sum(p.numel() for p in model.policy.parameters())),
            "training_seconds": round(time.monotonic() - started, 3),
            "training_timesteps_requested": timesteps,
            "training_timesteps_actual": int(model.num_timesteps),
        }
    finally:
        model.get_env().close()


def evaluate(variant, index, episodes, split, path=None):
    import numpy as np
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO

    if split not in ("development", "heldout"):
        raise ValueError("Only non-training evaluation seeds allowed")
    canonical, recurrent, mode = spec(variant)
    model = None
    if canonical != "random":
        if path is None:
            raise ValueError(f"Missing frozen checkpoint for {variant}")
        model = (RecurrentPPO if recurrent else PPO).load(path, device="cpu")
    env = make_env(mode)
    observations = []
    try:
        for ep in range(episodes):
            seed = seed_for(split, index, ep)
            obs, _ = env.reset(seed=seed)
            rng = random.Random(seed * 31 + 19)
            done = False
            first = True
            state = None
            reward_total = 0.
            steps = 0
            success = False
            while not done:
                if canonical == "random":
                    action = rng.randrange(env.action_space.n)
                elif recurrent:
                    action, state = model.predict(
                        obs, state=state,
                        episode_start=np.array([first], dtype=bool),
                        deterministic=True)
                    action = int(action)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                success = success or float(reward) > 0
                reward_total += float(reward)
                steps += 1
                first = False
                done = bool(terminated or truncated)
            observations.append({
                "episode_seed": seed, "success": int(success),
                "return": reward_total, "length": steps})
    finally:
        env.close()
    return {
        "success_rate": statistics.mean(x["success"] for x in observations),
        "mean_return": statistics.mean(x["return"] for x in observations),
        "mean_length": statistics.mean(x["length"] for x in observations),
        "episodes": observations,
    }


def summarize(rows, variants):
    result = {"policies": {
        v: {k: statistics.mean(r[v][k] for r in rows)
            for k in ("success_rate", "mean_return", "mean_length")}
        for v in variants},
        "paired": {},
    }
    for challenger, baseline in [
            ("lstm", "ppo"),
            ("ppo_f4", "ppo"),
            ("lstm_f4", "lstm"),
            ("lstm_f4", "lstm_zero"),
            ("lstm_f4", "lstm_f4_corrupt")]:
        if challenger in variants and baseline in variants:
            result["paired"][f"{challenger}_minus_{baseline}"] = grouped_bootstrap(
                rows, challenger, baseline)
    return result


def dependency_fingerprint():
    import minigrid, stable_baselines3, sb3_contrib, torch, numpy, gymnasium
    return {
        "minigrid": minigrid.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "gymnasium": gymnasium.__version__,
        "python": sys.version.split()[0],
    }


def load_locked(folder, stage):
    manifest_file = folder / "frozen_manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("stage") != stage:
        raise ValueError(f"Expected {stage} manifest")
    if manifest.get("source_sha256") != file_sha(Path(__file__)):
        raise ValueError("F9-B code changed since model freeze")
    if manifest.get("dependencies") != dependency_fingerprint():
        raise ValueError("F9-B dependencies changed since model freeze")
    for name, details in manifest["models"].items():
        path = folder / "checkpoints" / details["model_path"]
        if not path.is_file() or file_sha(path) != details["model_sha256"]:
            raise ValueError(f"Missing/changed checkpoint {name}")
    return manifest


def competence(rows):
    if not rows or "lstm" not in rows[0] or "random" not in rows[0]:
        return False, {"reason": "Missing recurrent baseline or random diagnostic"}
    lstm = statistics.mean(r["lstm"]["success_rate"] for r in rows)
    random_score = statistics.mean(r["random"]["success_rate"] for r in rows)
    enough = len(rows) >= MIN_GATE_SEEDS
    eligible = enough and lstm >= COMPETENT_SUCCESS and (
        lstm - random_score) >= MIN_RANDOM_MARGIN
    return eligible, {
        "training_seeds": len(rows),
        "lstm_success": lstm,
        "random_success": random_score,
        "threshold_lstm": COMPETENT_SUCCESS,
        "min_random_margin": MIN_RANDOM_MARGIN,
        "ready": eligible,
        "reason": ("Strong baseline demonstrated on development"
                   if eligible else "INCONCLUSIVE: recurrent baseline not yet competent"),
    }


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def baselines(seeds, timesteps, episodes, out):
    ensure_dependencies()
    out.mkdir(parents=True, exist_ok=True)
    by_seed = []
    models = {}
    for index in range(seeds):
        row = {"random": evaluate("random", index, episodes, "development")}
        for variant in BASELINES:
            trained = learn(variant, index, timesteps, out / "checkpoints")
            models[f"{variant}_{index}"] = trained
            row[variant] = evaluate(
                variant, index, episodes, "development",
                out / "checkpoints" / trained["model_path"])
            row[variant]["training"] = trained
            print(f"BASELINE {variant} seed={index} success={row[variant]['success_rate']:.3f}", flush=True)
        by_seed.append(row)
    ready, gate = competence(by_seed)
    manifest = {
        "stage": "f9b_baseline_development",
        "source_sha256": file_sha(Path(__file__)),
        "dependencies": dependency_fingerprint(),
        "task": TASK, "models": models,
        "config": {"seeds": seeds, "timesteps": timesteps, "eval_episodes": episodes},
        "gate": gate, "heldout_opened": False,
    }
    report = {
        "experiment": "F9-B external Memory baseline competence — NOT an INFINITO improvement",
        "phase": "development", "stage": "baselines", "task": TASK,
        "summary": summarize(by_seed, ("random", *BASELINES)),
        "gate": gate, "per_seed": by_seed,
        "interpretation": ("Baselines eligible to proceed to ablation"
                           if ready else "Inconclusive: increase development training budget"),
        "limitations": "Only original partial MiniGrid views; no INFINITO memory tested yet.",
    }
    write_json(out / "frozen_manifest.json", manifest)
    write_json(out / "report.json", report)
    return report


def ablations(baseline_dir, out, *, smoke=False):
    ensure_dependencies()
    baseline_manifest = load_locked(baseline_dir, "f9b_baseline_development")
    baseline_report = json.loads((baseline_dir / "report.json").read_text(encoding="utf-8"))
    if baseline_report.get("phase") != "development":
        raise ValueError("Baseline report is not development")
    cfg = baseline_manifest["config"]
    if not baseline_manifest["gate"]["ready"] and not smoke:
        raise ValueError("Stop: LSTM baseline not competent; cannot begin causal ablations.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    models = dict(baseline_manifest["models"])
    for details in models.values():
        name = details["model_path"]
        shutil.copy2(baseline_dir / "checkpoints" / name, out / "checkpoints" / name)
    rows = []
    for index in range(cfg["seeds"]):
        # Re-evaluate all conditions with identical development seeds.
        row = {}
        for variant in ("random", *BASELINES):
            path = (None if variant == "random" else
                    out / "checkpoints" / models[f"{variant}_{index}"]["model_path"])
            row[variant] = evaluate(variant, index, cfg["eval_episodes"],
                                    "development", path)
        for variant in TRAINABLE_ABLATIONS:
            trained = learn(variant, index, cfg["timesteps"], out / "checkpoints")
            models[f"{variant}_{index}"] = trained
            row[variant] = evaluate(
                variant, index, cfg["eval_episodes"], "development",
                out / "checkpoints" / trained["model_path"])
            row[variant]["training"] = trained
            print(f"ABLATION {variant} seed={index} success={row[variant]['success_rate']:.3f}", flush=True)
        row["lstm_f4_corrupt"] = evaluate(
            "lstm_f4_corrupt", index, cfg["eval_episodes"], "development",
            out / "checkpoints" / models[f"lstm_f4_{index}"]["model_path"])
        rows.append(row)
    gate, gate_meta = competence(rows)
    complete = len(rows) >= MIN_HELDOUT_SEEDS and gate and not smoke
    manifest = {
        "stage": "f9b_ablations_development",
        "source_sha256": file_sha(Path(__file__)),
        "dependencies": dependency_fingerprint(),
        "task": TASK, "models": models, "config": cfg,
        "gate": gate_meta, "eligible_for_heldout": complete,
        "heldout_opened": False,
        "baseline_manifest_sha256": file_sha(baseline_dir / "frozen_manifest.json"),
    }
    report = {
        "experiment": "F9-B external Memory: actual F4 Store/FIFO and causal intervention",
        "phase": "development", "stage": "ablations", "task": TASK,
        "summary": summarize(rows, VARIANTS), "per_seed": rows,
        "gate": gate_meta, "eligible_for_heldout": complete,
        "limitations": (
            "F4 Store/FIFO integrated, NOT F4 LearnedRelevance nor F5. "
            "Input shapes/parameter counts differ; D vs E matches feature width "
            "but does not prove compute parity. Corruption may be OOD. "
            "Do not infer generalization until the sealed holdout is opened once."
        ),
    }
    write_json(out / "frozen_manifest.json", manifest)
    write_json(out / "report.json", report)
    return report


def heldout(frozen_dir, out, episodes):
    ensure_dependencies()
    m = load_locked(frozen_dir, "f9b_ablations_development")
    if (not m.get("eligible_for_heldout")
            or m["config"]["seeds"] < MIN_HELDOUT_SEEDS
            or episodes < MIN_HELDOUT_EPISODES):
        raise ValueError("Holdout requires 5 competent baselines and >=500 episodes/seed.")
    if out.resolve() == frozen_dir.resolve():
        raise ValueError("Holdout output must differ from frozen model directory")
    # Open once, before exposing any reserved score.
    marker = frozen_dir / ".heldout_opened"
    with marker.open("x", encoding="utf-8") as lock:
        lock.write(f"source={file_sha(frozen_dir / 'frozen_manifest.json')}\n")
    rows = []
    models = m["models"]
    for index in range(m["config"]["seeds"]):
        row = {}
        for variant in VARIANTS:
            canonical, _, _ = spec(variant)
            checkpoint = (None if canonical == "random" else
                          frozen_dir / "checkpoints" /
                          models[f"{canonical}_{index}"]["model_path"])
            row[variant] = evaluate(variant, index, episodes, "heldout", checkpoint)
        rows.append(row)
    report = {
        "experiment": "F9-B sealed one-time MiniGrid Memory holdout",
        "phase": "heldout", "stage": "holdout", "task": TASK,
        "source_manifest_sha256": file_sha(frozen_dir / "frozen_manifest.json"),
        "episodes_per_training_seed": episodes,
        "summary": summarize(rows, VARIANTS), "per_seed": rows,
        "limitations": (
            "Same family of MiniGrid Memory levels with fresh seeds, not "
            "generalization to unrelated tasks. F4 Store/FIFO only; causal "
            "color corruption may produce distribution shift."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="F9-B: causal F4 memory in MiniGrid")
    parser.add_argument("--phase", choices=("baselines", "ablations", "heldout"),
                        required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "baselines":
        if not 1 <= args.seeds <= 10 or not 128 <= args.timesteps <= 1_000_000:
            parser.error("baselines: seeds 1..10, timesteps 128..1000000")
        if not 1 <= args.eval_episodes <= 1000:
            parser.error("eval episodes 1..1000")
        if not args.smoke and args.timesteps not in COMPETENCE_LADDER:
            parser.error("Use preregistered development budgets 20k, 100k or 300k")
        report = baselines(args.seeds, args.timesteps, args.eval_episodes, args.output)
    elif args.phase == "ablations":
        if args.baselines is None:
            parser.error("ablations requires --baselines")
        report = ablations(args.baselines, args.output, smoke=args.smoke)
    else:
        if args.frozen is None:
            parser.error("heldout requires --frozen")
        report = heldout(args.frozen, args.output, args.eval_episodes)
    print("GATE", json.dumps(report.get("gate", {}), ensure_ascii=False), flush=True)
    print("SUMMARY", json.dumps(report["summary"], ensure_ascii=False), flush=True)
    print("REPORT", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
