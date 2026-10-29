"""F9-A: *external* MiniGrid evaluation, real PPO and RecurrentPPO baselines.

No INFINITO-specific module is integrated yet. Never make claims about memory
benefits for INFINITO from these *external* baselines. Optional heavy deps
are imported only when running the benchmark, never by lab portal import.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

TASKS = {
    "memory": "MiniGrid-MemoryS7-v0",
    "doorkey": "MiniGrid-DoorKey-5x5-v0",
    "dynamic": "MiniGrid-Dynamic-Obstacles-5x5-v0",
}
ALGORITHMS = ("random", "ppo", "recurrent_ppo")
TRAIN_BASE = 1_000
DEVELOPMENT_BASE = 100_000
HOLDOUT_BASE = 900_000


def seed_for(split: str, train_index: int, episode: int = 0) -> int:
    """No overlap in preassigned train/development/heldout ranges."""
    if not 0 <= train_index < 50 or not 0 <= episode < 10_000:
        raise ValueError("seed index out of range")
    base = {"train": TRAIN_BASE, "development": DEVELOPMENT_BASE,
            "heldout": HOLDOUT_BASE}[split]
    return base + train_index * 10_000 + episode


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dependencies() -> dict:
    try:
        import gymnasium
        import minigrid
        import stable_baselines3
        import sb3_contrib
        import torch
        import numpy
    except ImportError as error:
        raise RuntimeError(
            "F9 requires its separate heavy environment: "
            "pip install -r observatory/requirements_f9.txt") from error
    torch.set_num_threads(1)
    return {
        "python": platform.python_version(),
        "gymnasium": gymnasium.__version__,
        "minigrid": minigrid.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": getattr(sb3_contrib, "__version__", "unknown"),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
    }


def make_env(task: str):
    """Same egocentric image + direction for ALL agents; no full map access."""
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
    from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

    if task not in TASKS:
        raise ValueError(f"Unknown external MiniGrid task: {task}")
    obs_max = float(max(OBJECT_TO_IDX.values()))
    color_max = float(max(COLOR_TO_IDX.values()))

    class EqualPartialObservation(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(7 * 7 * 3 + 1,),
                dtype=np.float32)

        def observation(self, obs):
            image = obs["image"].astype(np.float32).copy()
            assert image.shape == (7, 7, 3), image.shape
            image[:, :, 0] /= obs_max
            image[:, :, 1] /= color_max
            image[:, :, 2] /= 3.0
            direction = float(obs["direction"]) / 3.0
            return np.concatenate(
                (image.reshape(-1), np.array([direction], dtype=np.float32))
            ).astype(np.float32)

    return EqualPartialObservation(gym.make(TASKS[task]))


def train_policy(task: str, algorithm: str, train_index: int,
                 timesteps: int, out: Path) -> dict:
    """Train real baselines using the same exact transition budget."""
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    if algorithm not in ("ppo", "recurrent_ppo"):
        raise ValueError(algorithm)
    env = make_env(task)
    seed = seed_for("train", train_index)
    env.reset(seed=seed)
    common = dict(
        env=env, learning_rate=3e-4, n_steps=64, batch_size=64,
        n_epochs=2, gamma=.99, gae_lambda=.95,
        ent_coef=.0, verbose=0, seed=seed, device="cpu",
    )
    t0 = time.monotonic()
    if algorithm == "ppo":
        model = PPO("MlpPolicy", **common, policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64])})
    else:
        model = RecurrentPPO("MlpLstmPolicy", **common, policy_kwargs={
            "lstm_hidden_size": 64, "n_lstm_layers": 1,
            "net_arch": dict(pi=[64], vf=[64])})
    try:
        model.learn(total_timesteps=timesteps)
        model_path = out / f"{task}_{algorithm}_s{train_index}.zip"
        model.save(model_path)
        return {
            "model_path": model_path.name, "model_sha256": file_sha(model_path),
            "parameters": sum(x.numel() for x in model.policy.parameters()),
            "training_seconds": round(time.monotonic() - t0, 3),
            "training_timesteps_requested": timesteps,
            "training_timesteps_actual": int(model.num_timesteps),
        }
    finally:
        model.get_env().close()


def evaluate(task: str, algorithm: str, train_index: int, n: int,
             split: str, checkpoint: Path | None = None) -> dict:
    import numpy as np
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    if split not in ("development", "heldout"):
        raise ValueError("evaluation split must not be a training split")
    model = None
    if algorithm == "ppo":
        model = PPO.load(checkpoint, device="cpu")
    if algorithm == "recurrent_ppo":
        model = RecurrentPPO.load(checkpoint, device="cpu")
    env = make_env(task)
    records = []
    try:
        for i in range(n):
            episode_seed = seed_for(split, train_index, i)
            obs, _ = env.reset(seed=episode_seed)
            rng = random.Random(episode_seed * 31 + 19)
            done = False
            reward_total = 0.0
            length = 0
            success = False
            memory = None
            first = True
            while not done:
                if algorithm == "random":
                    action = rng.randrange(env.action_space.n)
                elif algorithm == "recurrent_ppo":
                    action, memory = model.predict(
                        obs, state=memory,
                        episode_start=np.array([first], dtype=bool),
                        deterministic=True)
                    action = int(action)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                reward_total += float(reward)
                success = success or reward > 0
                length += 1
                first = False
                done = bool(terminated or truncated)
            records.append({
                "episode_seed": episode_seed, "success": int(success),
                "return": reward_total, "length": length,
            })
    finally:
        env.close()
    return {
        "success_rate": statistics.mean(e["success"] for e in records),
        "mean_return": statistics.mean(e["return"] for e in records),
        "mean_length": statistics.mean(e["length"] for e in records),
        "episodes": records,
    }


def grouped_bootstrap(rows: list, challenger: str, reference: str):
    """Resample training seeds, not episodes; absent for tiny technical smokes."""
    differences = [r[challenger]["success_rate"] -
                   r[reference]["success_rate"] for r in rows]
    if len(differences) < 4:
        return {"mean_difference": statistics.mean(differences),
                "bootstrap_95": None, "independent_training_seeds": len(rows)}
    rng = random.Random(91231)
    estimates = sorted(
        statistics.mean(rng.choices(differences, k=len(differences)))
        for _ in range(2000))
    return {"mean_difference": statistics.mean(differences),
            "bootstrap_95": [estimates[49], estimates[1949]],
            "independent_training_seeds": len(rows)}


def aggregate(per_seed, active):
    summary = {
        algorithm: {
            key: statistics.mean(r[algorithm][key] for r in per_seed)
            for key in ("success_rate", "mean_return", "mean_length")
        } for algorithm in active
    }
    comparisons = {}
    if "ppo" in active and "recurrent_ppo" in active:
        comparisons["recurrent_ppo_minus_ppo"] = grouped_bootstrap(
            per_seed, "recurrent_ppo", "ppo")
    return {"policies": summary, "paired": comparisons}


def run_development(task: str, seeds: int, timesteps: int,
                    eval_episodes: int, out: Path):
    deps = ensure_dependencies()
    out.mkdir(parents=True, exist_ok=True)
    models = out / "checkpoints"
    models.mkdir(exist_ok=True)
    by_seed = []
    manifest_models = {}
    for i in range(seeds):
        metrics = {}
        for algorithm in ALGORITHMS:
            checkpoint = None
            if algorithm != "random":
                train = train_policy(task, algorithm, i, timesteps, models)
                checkpoint = models / train["model_path"]
                manifest_models[f"{algorithm}_{i}"] = train
                print(f"TRAIN {task} {algorithm} seed={i} steps={train['training_timesteps_actual']}", flush=True)
            result = evaluate(task, algorithm, i, eval_episodes, "development", checkpoint)
            result["training"] = train if algorithm != "random" else None
            metrics[algorithm] = result
            print(f"DEV {task} {algorithm} seed={i} success={result['success_rate']:.4f}",flush=True)
        by_seed.append(metrics)
    fingerprint = file_sha(Path(__file__))
    manifest = {
        "stage": "development_complete", "task": task,
        "source_sha256": fingerprint, "dependencies": deps,
        "config": {"seeds": seeds, "timesteps": timesteps,
                   "eval_episodes": eval_episodes},
        "models": manifest_models,
        "heldout_opened": False,
    }
    (out / "frozen_manifest.json").write_text(json.dumps(manifest, indent=2))
    report = {
        "experiment": "F9-A independent MiniGrid baseline PPO vs LSTM (no INFINITO integration)",
        "phase": "development", "task": task, "env_id": TASKS[task],
        "training_seeds": seeds, "timesteps_requested": timesteps,
        "eval_episodes": eval_episodes, "seed_partition": {
            "train": TRAIN_BASE, "development": DEVELOPMENT_BASE,
            "heldout_reserved": HOLDOUT_BASE},
        "versions": deps, "summary": aggregate(by_seed, ALGORITHMS),
        "per_seed": by_seed,
        "limitations": (
            "External models only, not INFINITO. Training sample budgets equal "
            "but model parameter counts differ. Sparse rewards and tiny smokes "
            "may produce zero success. This is DEVELOPMENT evaluation only; "
            "never interpret it as an independent heldout generalization test."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def run_heldout(checkpoints: Path, eval_episodes: int, out: Path):
    """Load only frozen checkpoints. Any changed code/model fails closed."""
    deps = ensure_dependencies()
    manifest_path = checkpoints / "frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["stage"] != "development_complete":
        raise ValueError("invalid checkpoint manifest")
    if manifest["source_sha256"] != file_sha(Path(__file__)):
        raise ValueError("code changed since checkpoint freeze; cannot open holdout")
    if manifest["dependencies"] != deps:
        raise ValueError("dependencies changed since checkpoint freeze")
    task = manifest["task"]
    seeds = manifest["config"]["seeds"]
    by_seed = []
    for i in range(seeds):
        row = {}
        for algorithm in ALGORITHMS:
            checkpoint = None
            if algorithm != "random":
                frozen = manifest["models"][f"{algorithm}_{i}"]
                checkpoint = checkpoints / "checkpoints" / frozen["model_path"]
                if file_sha(checkpoint) != frozen["model_sha256"]:
                    raise ValueError("checkpoint checksum mismatch")
            row[algorithm] = evaluate(task, algorithm, i, eval_episodes,
                                      "heldout", checkpoint)
        by_seed.append(row)
    report = {
        "experiment": "F9-A heldout (frozen PPO/LSTM checkpoints)",
        "phase": "heldout", "task": task,
        "checkpoint_manifest_sha256": file_sha(manifest_path),
        "eval_episodes": eval_episodes,
        "summary": aggregate(by_seed, ALGORITHMS),
        "per_seed": by_seed,
        "limitations": (
            "Same MiniGrid environment family, unseen RNG seeds; not yet "
            "transfer to an unseen task type or proof of INFINITO improvement."
        )
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="F9 external MiniGrid, real PPO and LSTM baselines")
    p.add_argument("--task", choices=tuple(TASKS), default="memory")
    p.add_argument("--phase", choices=("development", "heldout"), default="development")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=256)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--checkpoints", type=Path)
    p.add_argument("--output", type=Path, default=Path("observatory_results/f9_external"))
    args = p.parse_args()
    if not 1 <= args.seeds <= 10 or not 128 <= args.timesteps <= 1_000_000:
        p.error("seeds 1..10 and timesteps 128..1,000,000 required")
    if not 1 <= args.eval_episodes <= 1000:
        p.error("eval-episodes 1..1000 required")
    if args.phase == "heldout":
        if args.checkpoints is None:
            p.error("heldout requires --checkpoints pointing to frozen development output")
        result = run_heldout(args.checkpoints, args.eval_episodes, args.output)
    else:
        result = run_development(args.task, args.seeds, args.timesteps,
                                 args.eval_episodes, args.output)
    print("RESULT", json.dumps(result["summary"], ensure_ascii=False), flush=True)
    print("REPORT", str(args.output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
