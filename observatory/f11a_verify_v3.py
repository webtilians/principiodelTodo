"""F11-A v3 verification: neutral-collector TRAIN fit vs DISJOINT TRAIN heldout.

NO DEV, NO CONFIRM. Imports the unchanged synthetic physics from F11-A v2.
All learning uses public episode steps only, never World.roles() or hidden queue.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from observatory.f11a_train_v2 import (
    ACTIONS, CHANGE_AT, HORIZON, MAX_EPISODES, Options, OptionController,
    World, episode, react, success, total_reward, train_seed, privileged,
)

FIT_WORLDS = 384
HOLDOUT_WORLDS = 128
RECALL_K = 12
# Temporal model sees the public observation schema, not simulator roles.
RESOURCE_FEATURES = (2, 4, 3)
LATENT_EFFECT_LAGS = (1, 2, 3)
ARMS = ("A0", "A1", "A2", "A3", "A3_ablated", "A4")


class BalancedEpisodic:
    """RAG baseline: retrieves literal successful episode suffixes, not Q values.

    Time-balanced candidate allocation prevents automatic exclusion of old
    successes. Each action is copied from a raw, traceable (episode, step).
    """

    def __init__(self):
        self.episodes = []
        self.index = defaultdict(list)
        self.queries = 0
        self.candidates_examined = 0

    def observe(self, steps):
        key = len(self.episodes)
        self.episodes.append((steps, bool(success(steps))))
        for j, row in enumerate(steps):
            self.index[tuple(row["before"])].append((key, j))

    def choose(self, obs, rng):
        self.queries += 1
        entries = self.index.get(tuple(obs), ())
        if not entries:
            return rng.choice(ACTIONS)
        # 4 oldest, 4 central, 4 newest successful exact-state memories.
        positives = [e for e in entries if self.episodes[e[0]][1]]
        pool = positives if positives else list(entries)
        n = len(pool)
        candidates = {pool[min(n - 1, k * (n - 1) // (RECALL_K - 1))]
                      for k in range(RECALL_K)}
        self.candidates_examined += len(candidates)
        chosen = min(candidates, key=lambda e: (
            not self.episodes[e[0]][1],
            len(self.episodes[e[0]][0]) - e[1],
            -e[0],
        ))
        return self.episodes[chosen[0]][0][chosen[1]]["action"]

    def snapshot(self):
        return {"episodic_entries": len(self.episodes),
                "retrieval_K": RECALL_K,
                "queries": self.queries,
                "candidates_examined": self.candidates_examined}


class TemporalPlanner:
    """A2 schema-aware MODEL-based control from only publicly visible effects.

    Learns: which opaque actions precede a PUBLIC feature change (with lags),
    action leading to observed success, terminal-state prototypes and a
    low-harm wait action. It DOES NOT receive the simulator's private mapping,
    phase, delay, queue contents, evaluator reward, or an expert action plan.
    The greedy best-progress target planner is a *schema-guided baseline*,
    not evidence of emergent planning or policy invention.
    """

    def __init__(self):
        self.effects = defaultdict(Counter)  # (mode, action) -> public feature
        self.terminals = defaultdict(Counter)  # mode -> action
        self.prototypes = defaultdict(Counter)  # (mode, goal) -> (power,route,stock)
        self.waits = defaultdict(Counter)  # mode -> candidate actions
        self.events = 0
        self.calls = 0
        self.attributions = 0

    def observe(self, steps, episode_number=0):
        weight = 1 + min(episode_number, 29) / 10.0
        for i, row in enumerate(steps):
            before, after = row["before"], row["after"]
            mode = before[0]
            if row["done"]:
                self.terminals[mode][row["action"]] += weight
                self.prototypes[(mode, before[1])][tuple(before[j] for j in RESOURCE_FEATURES)] += weight
            if before[-1] > 0 and after[-1] < before[-1] and row["reward"] >= -0.06:
                self.waits[mode][row["action"]] += weight
            diffs = [pos for pos in RESOURCE_FEATURES if before[pos] != after[pos]]
            if row["done"]:
                diffs = []  # terminal consumes resources; not a delayed gain
            for pos in diffs:
                for lag in LATENT_EFFECT_LAGS:
                    j = i - lag
                    if j < 0:
                        continue
                    earlier = steps[j]
                    # Delayed action likely produced an observable queue rise.
                    q_before, q_after = earlier["before"][-1], earlier["after"][-1]
                    visible_enqueue = q_after > q_before
                    if not visible_enqueue:
                        continue
                    action = earlier["action"]
                    self.effects[(mode, action)][pos] += weight / lag
                    self.attributions += 1
            self.events += 1

    def _prototype(self, obs):
        candidates = self.prototypes.get((obs[0], obs[1]))
        if not candidates:
            return None
        return candidates.most_common(1)[0][0]

    def _effect_action(self, obs, pos):
        scores = {action: self.effects[(obs[0], action)][pos] for action in ACTIONS}
        best = max(scores.values())
        if best <= 0:
            return None
        return min(action for action, score in scores.items() if score == best)

    def choose(self, obs, rng):
        self.calls += 1
        if obs[-1] > 0:
            waits = self.waits.get(obs[0], {})
            if waits:
                return max(sorted(waits), key=lambda a: waits[a])
        target = self._prototype(obs)
        if target is None:
            return rng.choice(ACTIONS)
        for field, desired in zip(RESOURCE_FEATURES, target):
            if obs[field] == desired:
                continue
            action = self._effect_action(obs, field)
            if action is not None:
                return action
        terminal = self.terminals.get(obs[0], {})
        if terminal:
            return max(sorted(terminal), key=lambda a: terminal[a])
        return rng.choice(ACTIONS)

    def snapshot(self):
        return {"events": self.events, "attributions": self.attributions,
                "calls": self.calls, "model_effects": [
                    [mode, action, sorted(dict(scores).items())]
                    for (mode, action), scores in sorted(self.effects.items())],
                "terminal_actions": {str(k): dict(v) for k, v in self.terminals.items()},
                "public_success_prototypes": [
                    [mode, goal, [[list(v), n] for v, n in counter.most_common()]]
                    for (mode, goal), counter in sorted(self.prototypes.items())],
                "wait_actions": {str(k): dict(v) for k, v in self.waits.items()}}


class BoundedOptions(Options):
    """A3 must not become an unbounded lookup of raw training-world episodes."""

    def observe(self, steps):
        if not success(steps):
            return
        trace = tuple((tuple(x["before"]), x["action"]) for x in steps)
        start = trace[0][0]
        bucket = self.by_start[start]
        if trace in bucket:
            self.support[trace] += 1
        elif len(bucket) < 4:
            bucket.add(trace)
            self.support[trace] = 1
        else:
            weakest = max(bucket, key=lambda t: (len(t), -self.support[t], t))
            if len(trace) < len(weakest):
                bucket.remove(weakest)
                del self.support[weakest]
                self.contradicted.pop(weakest, None)
                bucket.add(trace)
                self.support[trace] = 1


def synthetic_bytes(value):
    return len(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def train(replica, fit_worlds=FIT_WORLDS, holdout_worlds=HOLDOUT_WORLDS,
          episodes=MAX_EPISODES, out: Path | None = None):
    if not (0 <= replica < 4 and 1 <= fit_worlds <= FIT_WORLDS
            and 1 <= holdout_worlds <= HOLDOUT_WORLDS
            and 1 <= episodes <= MAX_EPISODES):
        raise ValueError("F11-A v3 is TRAIN-only: fixed FIT and HELDOUT seed bounds")
    # Fixed partition boundary, even for small smoke tests.
    assert FIT_WORLDS + HOLDOUT_WORLDS == 512
    bank, effects, options = BalancedEpisodic(), TemporalPlanner(), BoundedOptions()
    digest = hashlib.sha256()
    explored = won = actions = 0
    log = None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        log = gzip.open(out / "fit_events.jsonl.gz", "wt", encoding="utf-8")
    try:
        for i in range(fit_worlds):
            world = World.generate(train_seed(replica, i))
            for j in range(episodes):
                phase = int(j >= CHANGE_AT)
                steps, _ = episode(world, react, world.seed * 100 + j, phase)
                row = {"split": "TRAIN_FIT", "world_seed": world.seed, "episode": j,
                       "source": "independent_random", "steps": steps}
                line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                digest.update(line.encode())
                if log:
                    log.write(line)
                bank.observe(steps)
                effects.observe(steps, j)
                options.observe(steps)
                explored += 1
                won += int(success(steps))
                actions += len(steps)
    finally:
        if log:
            log.close()

    # FROZEN initial model checkpoint. Never fit on the reserved TRAIN-heldout.
    mem_a1 = bank.snapshot()
    mem_a2 = effects.snapshot()
    mem_a3 = options.snapshot()
    opts_bytes = synthetic_bytes(mem_a3)
    a2_bytes = synthetic_bytes(mem_a2)
    summary = {
        "experiment": "F11-A", "study": "V3_A3_VERIFICATION",
        "status": "TRAIN_HELDOUT_DIAGNOSTIC_NOT_DEV_NOT_CONFIRM",
        "replica": replica,
        "fit_seed_first": train_seed(replica, 0),
        "fit_seed_last": train_seed(replica, fit_worlds - 1),
        "holdout_seed_first": train_seed(replica, FIT_WORLDS),
        "holdout_seed_last": train_seed(replica, FIT_WORLDS + holdout_worlds - 1),
        "fit_episodes": explored, "fit_actions": actions, "fit_wins": won,
        "neutral_collector_fraction": 1.0, "actor_collector_fraction": 0.0,
        "fit_event_sha256": digest.hexdigest(),
        "fit_episodic_entries": mem_a1["episodic_entries"],
        "fit_public_temporal_attributions": mem_a2["attributions"],
        "fit_options": len(options.support),
        "approx_a1_bytes": synthetic_bytes(mem_a1) + actions * 100,
        "approx_a2_bytes": a2_bytes,
        "approx_a3_bytes_including_a2": a2_bytes + opts_bytes,
        "approx_a3_to_a2_memory_ratio": (a2_bytes + opts_bytes) / max(1, a2_bytes),
        "worlds_disjoint": True, "dev_opened": False, "confirm_opened": False,
    }
    stats = {phase: {arm: {"won": 0, "episodes": 0, "reward_sum": 0.0,
                          "actions_sum": 0, "option_reuse": 0, "option_aborts": 0}
                     for arm in ARMS} for phase in ("0", "1")}
    audit_rows = []
    # Test on held-out TRAIN worlds. No model fitting on these results.
    for i in range(FIT_WORLDS, FIT_WORLDS + holdout_worlds):
        world = World.generate(train_seed(replica, i))
        for phase in (0, 1):
            seed = world.seed * 100 + 91 + phase
            a3 = OptionController(options, effects)
            ablate = OptionController(options, effects, enabled=False)
            arms = {"A0": react, "A1": bank.choose, "A2": effects.choose,
                    "A3": a3.choose, "A3_ablated": ablate.choose,
                    "A4": privileged(world, phase)}
            for name, policy in arms.items():
                rows, _ = episode(world, policy, seed, phase)
                cell = stats[str(phase)][name]
                cell["won"] += int(success(rows))
                cell["episodes"] += 1
                cell["reward_sum"] += total_reward(rows)
                cell["actions_sum"] += len(rows)
                if name == "A3":
                    cell["option_reuse"] += a3.reused
                    cell["option_aborts"] += int(a3.aborted)
                audit_rows.append({"world_seed": world.seed, "phase": phase,
                                   "arm": name, "success": success(rows),
                                   "reward": total_reward(rows),
                                   "actions": len(rows)})
    summary["train_heldout_results"] = stats
    summary["retrieval_queries"] = bank.queries
    summary["retrieval_candidates"] = bank.candidates_examined
    summary["effects_planning_calls"] = effects.calls
    if out:
        (out / "fit_checkpoint.json").write_text(
            json.dumps({"episodic": mem_a1, "effects": mem_a2, "options": mem_a3,
                        "fit_event_sha256": digest.hexdigest(),
                        "frozen_before_heldout": True}, sort_keys=True), encoding="utf-8")
        (out / "heldout_results.jsonl").write_text(
            "".join(json.dumps(x, sort_keys=True) + "\n" for x in audit_rows),
            encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True),
                                           encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description="F11-A v3 neutral-collector TRAIN verification only")
    p.add_argument("--replica", type=int, required=True)
    p.add_argument("--fit-worlds", type=int, default=FIT_WORLDS)
    p.add_argument("--holdout-worlds", type=int, default=HOLDOUT_WORLDS)
    p.add_argument("--episodes", type=int, default=MAX_EPISODES)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(train(a.replica, a.fit_worlds, a.holdout_worlds, a.episodes, a.out),
                     sort_keys=True))


if __name__ == "__main__":
    main()
