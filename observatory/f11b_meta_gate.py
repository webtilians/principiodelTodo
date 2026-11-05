"""INFINITO F11-B: learn when to reuse A3 options or replan with A2.

Only NEW F11-B TRAIN seeds are used; F11-A DEV/CONFIRM remain sealed.
Meta-selector learns from *one actually executed arm* per training episode.
Global selector weights freeze BEFORE disjoint TRAIN-heldout evaluation.
No privileged simulator role/phase/seed reaches the learned decision.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from collections import deque
from pathlib import Path
from time import perf_counter

from observatory.f11a_train_v2 import (
    ACTIONS, HORIZON, World, OptionController, episode, react,
    success, total_reward,
)
from observatory.f11a_verify_v3 import (
    BoundedOptions, TemporalPlanner, synthetic_bytes,
)

STUDY = "F11-B"
BASE_SEED = 62_100_000
BASE_WORLDS = 256
META_WORLDS = 128
HELDOUT_WORLDS = 128
BASE_EPISODES = 30
SEQUENCE_EPISODES = 40
TRAIN_EPSILON = 0.50  # neutral RANDOMIZED ARM, not gate choice
EVAL_EPSILON = 0.08
FEATURES = 11
ARMS = ("A2", "A3", "heuristic", "learned", "learned_nohist")


def world_seed(replica: int, index: int) -> int:
    if not (0 <= replica < 4 and 0 <= index < 512):
        raise ValueError("F11-B only reserves TRAIN worlds r0..3, i0..511")
    return BASE_SEED + 10_000 * replica + index


def flip_at(world: World) -> int:
    # Analyzer-only change point. NEVER passed to the gate or policy.
    return 14 + random.Random(world.seed + 71937).randrange(11)


def features_for(history: "Feedback", arm: str, availability: bool, use_history=True):
    other = "A3" if arm == "A2" else "A2"
    own = history.results[arm]
    alt = history.results[other]
    if not use_history:
        own = ()
        alt = ()
    def score(seq):
        return sum(seq) / len(seq) if seq else 0.5
    def latest(seq):
        return seq[-1] if seq else 0.5
    a = score(own)
    b = score(alt)
    return (
        1.0,
        float(availability),
        a - .5,
        b - .5,
        latest(own) - .5,
        latest(alt) - .5,
        min(len(own), 5) / 5.,
        min(len(alt), 5) / 5.,
        a - b,
        min(len(own) + len(alt), 10) / 10.,
        float(arm == "A3"),
    )


class Feedback:
    def __init__(self):
        self.results = {arm: deque(maxlen=5) for arm in ("A2", "A3")}
        self.choices = []
        self.switches = 0

    def add(self, arm: str, won: bool):
        if self.choices and arm != self.choices[-1]:
            self.switches += 1
        self.choices.append(arm)
        self.results[arm].append(float(won))


class LogisticGate:
    """Contextual bandit: fit binary public success from each chosen arm only."""

    def __init__(self):
        self.weights = {"A2": [0.] * FEATURES, "A3": [0.] * FEATURES}
        self.updates = 0
        self.predictions = 0

    @staticmethod
    def sigmoid(z):
        z = max(-25, min(25, z))
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, arm: str, x) -> float:
        self.predictions += 1
        return self.sigmoid(sum(a * b for a, b in zip(self.weights[arm], x)))

    def update(self, arm: str, x, observed_success: bool):
        """SGD performed in META-FIT only, after the real chosen arm returns."""
        p = self.predict(arm, x)
        rate = 0.08 / (1.0 + self.updates / 20_000.)
        self.weights[arm] = [
            w + rate * ((float(observed_success) - p) * xi - 0.0005 * w)
            for w, xi in zip(self.weights[arm], x)
        ]
        self.updates += 1

    def choose(self, history: Feedback, available: bool, rng: random.Random,
               *, no_history: bool = False, epsilon: float = EVAL_EPSILON):
        # No phase, seed, trial counter or privileged objective labels available.
        xs = {arm: features_for(history, arm, available, use_history=not no_history)
              for arm in ("A2", "A3")}
        scores = {arm: self.predict(arm, xs[arm]) for arm in xs}
        if rng.random() < epsilon:
            return rng.choice(("A2", "A3"))
        return max(("A2", "A3"), key=lambda arm: (scores[arm], arm == "A2"))

    def snapshot(self):
        return {"weights": self.weights, "updates": self.updates,
                "feature_count": FEATURES, "learner": "bandit_SGD_observed_reward_only"}


def fixed_heuristic(history: Feedback):
    """Strong, untrained baseline: retry alternative after a failure."""
    if not history.choices:
        return "A2"
    arm = history.choices[-1]
    if history.results[arm] and history.results[arm][-1] == 0.0:
        return "A3" if arm == "A2" else "A2"
    return arm


def policy_for(arm: str, options: BoundedOptions, effects: TemporalPlanner):
    if arm == "A2":
        return effects.choose
    if arm == "A3":
        return OptionController(options, effects).choose
    raise ValueError("No permitted policy")


def fit_base(replica: int, worlds: int, episodes: int, logfile=None):
    options = BoundedOptions()
    effects = TemporalPlanner()
    h = hashlib.sha256()
    wins = actions = 0
    for idx in range(worlds):
        world = World.generate(world_seed(replica, idx))
        for j in range(episodes):
            phase = int(j >= 15)
            steps, _ = episode(world, react, world.seed * 100 + j, phase)
            # Same neutral observable experiences for both learned components.
            options.observe(steps)
            effects.observe(steps, j)
            record = {
                "split": "F11B_BASE_TRAIN", "seed_for_audit": world.seed,
                "episode_for_audit": j, "steps": steps,
            }
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            h.update(line.encode())
            if logfile:
                logfile.write(line)
            wins += int(success(steps))
            actions += len(steps)
    return options, effects, {
        "base_fit_worlds": worlds, "base_fit_episodes": worlds * episodes,
        "base_fit_wins": wins, "base_fit_actions": actions,
        "base_event_sha256": h.hexdigest(),
        "base_learned_options": len(options.support),
    }


def train_gate(replica: int, options, effects, worlds: int,
               episodes: int, logfile=None):
    gate = LogisticGate()
    h = hashlib.sha256()
    successes = {"A2": 0, "A3": 0}
    exposures = {"A2": 0, "A3": 0}
    for idx in range(BASE_WORLDS, BASE_WORLDS + worlds):
        world = World.generate(world_seed(replica, idx))
        cutoff = flip_at(world)
        history = Feedback()
        behavior = random.Random(world.seed * 103 + 31)
        for j in range(episodes):
            # Only the chosen arm is observed. The counterfactual is NOT run.
            arm = behavior.choice(("A2", "A3"))
            start = world.reset(0).observe()
            candidate = bool(options.by_start.get(start))
            x = features_for(history, arm, candidate)
            steps, _ = episode(world, policy_for(arm, options, effects),
                               world.seed * 1000 + j, int(j >= cutoff))
            win = success(steps)
            gate.update(arm, x, win)
            history.add(arm, win)
            exposures[arm] += 1
            successes[arm] += int(win)
            record = {
                "split": "F11B_META_FIT", "world_seed_for_audit": world.seed,
                "episode_for_audit": j,
                "phase_for_audit_only": int(j >= cutoff),
                "arm_chosen_by_randomized_behavior": arm,
                "public_feature_vector": x,
                "observed_success": win, "observed_reward": total_reward(steps),
            }
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            h.update(line.encode())
            if logfile:
                logfile.write(line)
    return gate, {
        "gate_fit_worlds": worlds, "gate_fit_episodes": worlds * episodes,
        "gate_updates": gate.updates,
        "chosen_exposures": exposures, "chosen_successes": successes,
        "meta_event_sha256": h.hexdigest(),
        "counterfactual_arm_executions_for_learning": 0,
    }


def evaluate(replica: int, options, effects, gate: LogisticGate,
             worlds: int, episodes: int, out=None):
    records = []
    results = {
        name: {"episodes": 0, "wins": 0, "return_sum": 0.,
               "old_wins": 0, "old_episodes": 0,
               "new_wins": 0, "new_episodes": 0,
               "early_new_wins": 0, "early_new_episodes": 0,
               "switches": 0, "option_usage": 0,
               "action_steps": 0}
        for name in ARMS
    }
    frozen_weights = json.dumps(gate.weights, sort_keys=True)
    started = perf_counter()
    for idx in range(BASE_WORLDS + META_WORLDS, BASE_WORLDS + META_WORLDS + worlds):
        world = World.generate(world_seed(replica, idx))
        cutoff = flip_at(world)
        initial_state = world.reset(0).observe()
        available = bool(options.by_start.get(initial_state))
        for label in ARMS:
            history = Feedback()
            chooser = random.Random(world.seed * 13 + 7)
            for j in range(episodes):
                if label == "A2":
                    arm = "A2"
                elif label == "A3":
                    arm = "A3"
                elif label == "heuristic":
                    arm = fixed_heuristic(history)
                else:
                    arm = gate.choose(history, available, chooser,
                                      no_history=(label == "learned_nohist"))
                # Same exogenous randomness per world & episode across arms.
                policy = policy_for(arm, options, effects)
                steps, _ = episode(world, policy, world.seed * 1000 + j,
                                   phase=int(j >= cutoff))
                observed = success(steps)
                history.add(arm, observed)
                cell = results[label]
                cell["episodes"] += 1
                cell["wins"] += int(observed)
                cell["return_sum"] += total_reward(steps)
                cell["action_steps"] += len(steps)
                period = "old" if j < cutoff else "new"
                cell[period + "_wins"] += int(observed)
                cell[period + "_episodes"] += 1
                if cutoff <= j < cutoff + 5:
                    cell["early_new_wins"] += int(observed)
                    cell["early_new_episodes"] += 1
                if arm == "A3":
                    cell["option_usage"] += 1
                records.append({
                    "world_seed_for_audit": world.seed,
                    "cutoff_for_audit": cutoff,
                    "episode_for_audit": j,
                    "period_for_audit": period,
                    "condition": label,
                    "arm_selected": arm,
                    "success": observed,
                    "public_reward": total_reward(steps),
                    "action_steps": len(steps),
                })
            cell["switches"] += history.switches
    measured = perf_counter() - started
    if json.dumps(gate.weights, sort_keys=True) != frozen_weights:
        raise RuntimeError("Global gate weights changed on sealed TRAIN heldout")
    if out:
        out.joinpath("train_heldout_per_episode.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8")
    return results, measured


def run(replica: int, base_worlds=BASE_WORLDS,
        meta_worlds=META_WORLDS, heldout_worlds=HELDOUT_WORLDS,
        base_episodes=BASE_EPISODES, sequence_episodes=SEQUENCE_EPISODES,
        out: Path | None = None):
    if not (0 <= replica < 4 and 1 <= base_worlds <= BASE_WORLDS
            and 1 <= meta_worlds <= META_WORLDS
            and 1 <= heldout_worlds <= HELDOUT_WORLDS
            and 1 <= base_episodes <= BASE_EPISODES
            and 2 <= sequence_episodes <= SEQUENCE_EPISODES):
        raise ValueError("F11-B: outside TRAIN-only preregistered bounds")
    if out:
        out.mkdir(parents=True, exist_ok=True)
    with gzip.open(out / "base_events.jsonl.gz", "wt", encoding="utf-8") if out else _NullWriter() as logfile:
        options, effects, base = fit_base(replica, base_worlds, base_episodes, logfile if out else None)
    with gzip.open(out / "meta_events.jsonl.gz", "wt", encoding="utf-8") if out else _NullWriter() as logfile:
        gate, meta = train_gate(replica, options, effects, meta_worlds, sequence_episodes,
                                logfile if out else None)
    gate_before = gate.snapshot()
    results, seconds = evaluate(replica, options, effects, gate, heldout_worlds,
                                sequence_episodes, out)
    gate_after = gate.snapshot()
    if gate_before["weights"] != gate_after["weights"]:
        raise RuntimeError("Evaluation updated learned global gate weights")
    option_bytes = synthetic_bytes(options.snapshot())
    effects_bytes = synthetic_bytes(effects.snapshot())
    gate_bytes = synthetic_bytes(gate_before)
    summary = {
        "study": STUDY, "status": "TRAIN_HELDOUT_DIAGNOSTIC_ONLY",
        "replica": replica,
        "base_fit_seed_start": world_seed(replica, 0),
        "base_fit_seed_end": world_seed(replica, base_worlds - 1),
        "gate_fit_seed_start": world_seed(replica, BASE_WORLDS),
        "gate_fit_seed_end": world_seed(replica, BASE_WORLDS + meta_worlds - 1),
        "heldout_seed_start": world_seed(replica, BASE_WORLDS + META_WORLDS),
        "heldout_seed_end": world_seed(replica, BASE_WORLDS + META_WORLDS + heldout_worlds - 1),
        "base": base, "meta": meta,
        "train_heldout_results": results,
        "evaluation_elapsed_seconds": seconds,
        "model_bytes": {
            "a2_effects": effects_bytes, "a3_options": option_bytes,
            "gate": gate_bytes, "total_adaptive": effects_bytes + option_bytes + gate_bytes,
        },
        "global_gate_weights_frozen_before_heldout": True,
        "heldout_was_excluded_from_base_and_meta_fit": True,
        "meta_fit_uses_only_chosen_arm_observed_outcomes": True,
        "dev_opened": False, "confirm_opened": False,
    }
    if out:
        out.joinpath("frozen_gate.json").write_text(json.dumps(gate_before, sort_keys=True,
                                                              indent=2), encoding="utf-8")
        out.joinpath("summary.json").write_text(json.dumps(summary, sort_keys=True,
                                                           indent=2), encoding="utf-8")
    return summary


class _NullWriter:
    def __enter__(self): return None
    def __exit__(self, *args): return False


def main():
    parser = argparse.ArgumentParser(description="F11-B: train meta-selector then holdout TRAIN-only")
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--base-worlds", type=int, default=BASE_WORLDS)
    parser.add_argument("--meta-worlds", type=int, default=META_WORLDS)
    parser.add_argument("--heldout-worlds", type=int, default=HELDOUT_WORLDS)
    parser.add_argument("--base-episodes", type=int, default=BASE_EPISODES)
    parser.add_argument("--sequence-episodes", type=int, default=SEQUENCE_EPISODES)
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    result = run(a.replica, a.base_worlds, a.meta_worlds, a.heldout_worlds,
                 a.base_episodes, a.sequence_episodes, a.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
