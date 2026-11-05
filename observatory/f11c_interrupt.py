"""F11-C: learn an in-episode interruption rule from observed reward.

TRAIN-only experiment, disjoint fresh 63_100_000 worlds. The gate sees PUBLIC
observations and a previously successful A3 option. It never receives phase,
seed, private effect queue, or counterfactual outcomes of unexecuted actions.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from observatory.f11a_train_v2 import ACTIONS, World, OptionController, episode, react, success
from observatory.f11a_verify_v3 import BoundedOptions, TemporalPlanner, synthetic_bytes

STUDY = "F11-C"
TRAIN_BASE = 63_100_000
BASE_WORLDS = 256
GATE_WORLDS = 128
HELDOUT_WORLDS = 128
BASE_EPISODES = 30
GATE_EPISODES = 30
HORIZON = 12
FEATURES = 8
ARMS = ("P", "C", "G", "H", "L", "L_no_surprise", "A4")


def study_seed(replica: int, index: int) -> int:
    if not (0 <= replica < 4 and 0 <= index < 512):
        raise ValueError("Only F11-C TRAIN seeds (63_100_000 series) are permitted")
    return TRAIN_BASE + 10_000 * replica + index


def change_step(world: World) -> int:
    return 3 + random.Random(world.seed + 81039).randrange(5)


def gate_features(obs: tuple, expected: tuple, last_reward: float, step: int,
                  streak: int, remaining: int, ablate=False) -> tuple[float, ...]:
    diffs = [int(obs[j] != expected[j]) for j in range(6)]
    # Only prior observed feedback, public-state differences, option progress.
    values = (
        1.0,
        float(any(diffs)),
        sum(diffs[:5]) / 5.,
        float(diffs[5]),
        float(last_reward < -0.06),
        step / HORIZON,
        min(streak, 3) / 3.,
        min(remaining, HORIZON) / HORIZON,
    )
    if ablate:
        x = list(values)
        for j in (1, 2, 3, 4, 6):
            x[j] = 0.
        return tuple(x)
    return values


class LearnedInterrupt:
    """On-policy REINFORCE on the chosen action's final PUBLIC outcome only."""

    def __init__(self):
        self.w = [0.] * FEATURES
        self.baseline = .0
        self.updates = 0
        self.trajectories = 0

    @staticmethod
    def sigmoid(v):
        return 1 / (1 + math.exp(-max(-15, min(15, v))))

    def probability(self, features):
        z = sum(w * x for w, x in zip(self.w, features))
        return min(.95, max(.05, self.sigmoid(z)))

    def decision(self, features, rng, *, train=False):
        p = self.probability(features)
        if train:
            return bool(rng.random() < p), p
        return p >= .5, p

    def learn(self, decisions, observed_reward):
        if not decisions:
            return
        baseline_before = self.baseline
        # Discount observed total reward into [-1,1]; no counterfactual label.
        ret = max(-1., min(1., observed_reward))
        advantage = ret - baseline_before
        self.baseline = .98 * baseline_before + .02 * ret
        lr = .045 / (1. + self.trajectories / 3000.)
        for features, action, predicted_probability in decisions:
            for i, x in enumerate(features):
                self.w[i] += lr * advantage * (float(action) - predicted_probability) * x
        self.w = [max(-8., min(8., x)) for x in self.w]
        self.updates += len(decisions)
        self.trajectories += 1

    def snapshot(self):
        return {"weights": list(self.w), "baseline": self.baseline,
                "updates": self.updates, "trajectories": self.trajectories,
                "algorithm": "REINFORCE_observed_episodic_return_only"}


def fit_bases(replica: int, worlds: int, episodes: int, file=None):
    options, planner = BoundedOptions(), TemporalPlanner()
    wins, actions = 0, 0
    digest = hashlib.sha256()
    for i in range(worlds):
        world = World.generate(study_seed(replica, i))
        for j in range(episodes):
            phase = int(j >= 15)
            steps, _ = episode(world, react, world.seed * 100 + j, phase)
            options.observe(steps)
            planner.observe(steps, j)
            wins += int(success(steps))
            actions += len(steps)
            record = {"partition": "F11C_BASE_FIT",
                      "world_seed_for_audit": world.seed,
                      "episode_for_audit": j, "steps": steps}
            line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(line.encode())
            if file:
                file.write(line)
    return options, planner, {"base_fit_worlds": worlds,
                               "base_fit_episodes": worlds * episodes,
                               "base_fit_wins": wins, "base_fit_actions": actions,
                               "base_event_sha256": digest.hexdigest(),
                               "option_count": len(options.support)}


def simulate(world: World, planner: TemporalPlanner, options: BoundedOptions,
             selector: LearnedInterrupt | None, method: str, scenario: str,
             noise_seed: int, *, train=False) -> tuple[dict, list]:
    """Single actual episode, optional hidden mid-episode rule mutation.

    The simulator's phase update is confined HERE, outside the policy/gate.
    No counterfactual episode is run during selector training.
    """
    if method not in ARMS:
        raise ValueError("Unknown comparison arm")
    if scenario not in ("stable", "flip"):
        raise ValueError("Unknown perturbation")
    simulator = world.reset(phase=0)
    rng = random.Random(noise_seed)
    initial_obs = simulator.observe()
    trace = options.start(initial_obs) if method != "P" and method != "A4" else None
    had_option = trace is not None
    switched = (method == "P" or trace is None)
    option_at = 0
    previous_reward = 0.0
    streak = 0
    decisions = []
    false_positive = 0
    interrupt_count = 0
    interrupted_at = None
    reused_actions = 0
    predictions = 0
    steps = []
    cutoff = change_step(world) if scenario == "flip" else None
    for step in range(HORIZON):
        # This mutation is private to simulation. Actor inputs do not include it.
        if cutoff is not None and step == cutoff:
            simulator.phase = 1
        obs = simulator.observe()
        if method == "A4":
            # Privileged *ceiling only*, never part of selector training.
            role = world.roles(simulator.phase)
            _, goal, power, route, stock, pending_count = obs
            if pending_count:
                chosen = "E"
            elif not power:
                chosen = role["charge"]
            elif not stock:
                chosen = role["gather"]
            elif route != goal:
                chosen = role["route"]
            else:
                chosen = role["deliver"]
        elif switched or trace is None or option_at >= len(trace):
            chosen = planner.choose(obs, rng)
        else:
            expected_obs, scheduled_action = trace[option_at]
            diff = sum(obs[i] != expected_obs[i] for i in range(6))
            streak = streak + 1 if diff else 0
            feat = gate_features(obs, expected_obs, previous_reward, step, streak,
                                 len(trace) - option_at,
                                 ablate=(method == "L_no_surprise"))
            abort = False
            if method == "G":
                abort = diff > 0
            elif method == "H":
                abort = any(obs[i] != expected_obs[i] for i in range(2, 5))
            elif method in ("L", "L_no_surprise"):
                if selector is None:
                    raise ValueError("Trained gate is required for learned arms")
                abort, probability = selector.decision(feat, rng, train=train)
                predictions += 1
                if train:
                    decisions.append((feat, abort, probability))
            if abort:
                switched = True
                interrupt_count += 1
                interrupted_at = step
                false_positive += int(cutoff is None or step < cutoff)
                chosen = planner.choose(obs, rng)
            else:
                chosen = scheduled_action
                option_at += 1
                reused_actions += 1
        after, reward, done = simulator.step(chosen, rng)
        steps.append({"before": obs, "action": chosen, "after": after,
                      "reward": reward, "done": done})
        previous_reward = reward
        if done:
            break
    reward_sum = sum(row["reward"] for row in steps)
    record = {"success": bool(steps[-1]["done"]), "reward": reward_sum,
              "actions": len(steps), "had_option": had_option,
              "interrupted": bool(interrupt_count), "interrupt_at": interrupted_at,
              "false_positive_interrupt": false_positive,
              "reused_actions": reused_actions, "selector_decisions": predictions,
              "change_reached": bool(cutoff is not None and len(steps) > cutoff)}
    # Delayed outcome is the REAL episode reward, not the branch not taken.
    if train and selector is not None:
        selector.learn(decisions, reward_sum)
    return record, steps


def train_gate(replica, options, planner, gate, worlds, episodes, file=None):
    digest = hashlib.sha256()
    observed = {"episodes": 0, "successes": 0, "steps": 0, "options_selected": 0}
    for idx in range(BASE_WORLDS, BASE_WORLDS + worlds):
        world = World.generate(study_seed(replica, idx))
        for j in range(episodes):
            scenario = "flip" if (j + idx) % 2 else "stable"
            record, steps = simulate(world, planner, options, gate, "L", scenario,
                                     noise_seed=world.seed * 1000 + j,
                                     train=True)
            observed["episodes"] += 1
            observed["successes"] += int(record["success"])
            observed["steps"] += record["actions"]
            observed["options_selected"] += int(record["had_option"])
            row = {"partition": "F11C_GATE_FIT",
                   "world_seed_for_audit": world.seed,
                   "scenario_for_audit": scenario, "episode_for_audit": j,
                   "observed": record, "steps": steps}
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(line.encode())
            if file:
                file.write(line)
    observed["gate_event_sha256"] = digest.hexdigest()
    return observed


def heldout(replica, options, planner, selector, worlds, out=None):
    stats = {scenario: {arm: {"episodes": 0, "successes": 0, "return_sum": 0.,
                             "action_steps": 0, "had_option": 0,
                             "interruptions": 0, "false_positive_interruptions": 0,
                             "change_reached": 0, "reused_actions": 0,
                             "selector_decisions": 0}
                        for arm in ARMS} for scenario in ("stable", "flip")}
    log = []
    weights = json.dumps(selector.snapshot(), sort_keys=True)
    for idx in range(BASE_WORLDS + GATE_WORLDS,
                     BASE_WORLDS + GATE_WORLDS + worlds):
        world = World.generate(study_seed(replica, idx))
        for scenario in ("stable", "flip"):
            for arm in ARMS:
                rec, steps = simulate(world, planner, options, selector, arm, scenario,
                                      noise_seed=world.seed * 1000 + (41 if scenario == "stable" else 77))
                score = stats[scenario][arm]
                score["episodes"] += 1
                score["successes"] += int(rec["success"])
                score["return_sum"] += rec["reward"]
                score["action_steps"] += rec["actions"]
                score["had_option"] += int(rec["had_option"])
                score["interruptions"] += int(rec["interrupted"])
                score["false_positive_interruptions"] += rec["false_positive_interrupt"]
                score["change_reached"] += int(rec["change_reached"])
                score["reused_actions"] += rec["reused_actions"]
                score["selector_decisions"] += rec["selector_decisions"]
                log.append({"world_seed_for_audit": world.seed,
                            "change_step_for_audit": change_step(world) if scenario == "flip" else None,
                            "scenario_for_audit": scenario, "arm": arm,
                            "outcome": rec, "trajectory": steps})
    if weights != json.dumps(selector.snapshot(), sort_keys=True):
        raise RuntimeError("F11-C HELDOUT improperly updated learned weights")
    if out:
        out.joinpath("heldout_per_world.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in log),
            encoding="utf-8")
    return stats


def run(replica, base_worlds=BASE_WORLDS, gate_worlds=GATE_WORLDS,
        heldout_worlds=HELDOUT_WORLDS, base_episodes=BASE_EPISODES,
        gate_episodes=GATE_EPISODES, out: Path | None = None):
    if not (0 <= replica < 4 and 1 <= base_worlds <= BASE_WORLDS
            and 1 <= gate_worlds <= GATE_WORLDS
            and 1 <= heldout_worlds <= HELDOUT_WORLDS
            and 1 <= base_episodes <= BASE_EPISODES
            and 1 <= gate_episodes <= GATE_EPISODES):
        raise ValueError("F11-C only permits its preregistered TRAIN bounds")
    if out:
        out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base_file = gzip.open(out / "base_events.jsonl.gz", "wt", encoding="utf-8") if out else None
    try:
        options, planner, base = fit_bases(replica, base_worlds, base_episodes, base_file)
    finally:
        if base_file:
            base_file.close()
    gate = LearnedInterrupt()
    gate_file = gzip.open(out / "gate_events.jsonl.gz", "wt", encoding="utf-8") if out else None
    try:
        training = train_gate(replica, options, planner, gate, gate_worlds,
                              gate_episodes, gate_file)
    finally:
        if gate_file:
            gate_file.close()
    frozen_gate = gate.snapshot()
    results = heldout(replica, options, planner, gate, heldout_worlds, out=out)
    timespent = time.perf_counter() - started
    a2_bytes = synthetic_bytes(planner.snapshot())
    options_bytes = synthetic_bytes(options.snapshot())
    gate_bytes = synthetic_bytes(frozen_gate)
    summary = {
        "study": STUDY, "status": "TRAIN_HELDOUT_NOT_CONFIRMATION",
        "replica": replica,
        "base_fit": base, "gate_fit": training,
        "base_fit_start": study_seed(replica, 0),
        "base_fit_end": study_seed(replica, base_worlds - 1),
        "gate_fit_start": study_seed(replica, BASE_WORLDS),
        "gate_fit_end": study_seed(replica, BASE_WORLDS + gate_worlds - 1),
        "heldout_start": study_seed(replica, BASE_WORLDS + GATE_WORLDS),
        "heldout_end": study_seed(replica, BASE_WORLDS + GATE_WORLDS + heldout_worlds - 1),
        "heldout": results, "frozen_selector": frozen_gate,
        "bytes": {"a2": a2_bytes, "options": options_bytes, "gate": gate_bytes,
                  "adaptive_total": a2_bytes + options_bytes + gate_bytes},
        "seconds_total_workflow": timespent,
        "dev_opened": False, "confirm_opened": False,
        "shared_base_exploration": True,
        "global_weights_frozen_during_heldout": True,
    }
    if out:
        out.joinpath("frozen_model.json").write_text(
            json.dumps({"selector": frozen_gate, "planner": planner.snapshot(),
                        "options": options.snapshot(),
                        "status": "frozen_before_heldout"}, sort_keys=True), encoding="utf-8")
        out.joinpath("summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description="F11-C TRAIN-only learned policy interruption")
    p.add_argument("--replica", type=int, required=True)
    p.add_argument("--base-worlds", type=int, default=BASE_WORLDS)
    p.add_argument("--gate-worlds", type=int, default=GATE_WORLDS)
    p.add_argument("--heldout-worlds", type=int, default=HELDOUT_WORLDS)
    p.add_argument("--base-episodes", type=int, default=BASE_EPISODES)
    p.add_argument("--gate-episodes", type=int, default=GATE_EPISODES)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.replica, a.base_worlds, a.gate_worlds,
                         a.heldout_worlds, a.base_episodes,
                         a.gate_episodes, a.out), sort_keys=True))


if __name__ == "__main__":
    main()
