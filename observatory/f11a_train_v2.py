"""F11-A TRAIN v2: delayed outcomes, unannounced rule changes, genuine episodic retrieval.

Synthetic TRAIN only. Sealed DEV/CONFIRM not generated, inspected or imported.
The learner receives only public observations and logged transition feedback.
No external tools, user data, real MCP memory or privileged oracle labels.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ACTIONS = ("A", "B", "C", "D", "E", "F")
TRAIN_BASE = 61_100_000
MAX_WORLDS = 512
MAX_EPISODES = 30
HORIZON = 12
PROBES_PER_WORLD = 2
CHANGE_AT = 15
PLAN_DEPTH = 5
RECALL_K = 8
ROLES = ("charge", "gather", "route", "deliver")


def train_seed(replica: int, index: int) -> int:
    if not 0 <= replica < 4 or not 0 <= index < MAX_WORLDS:
        raise ValueError("Only reserved F11-A TRAIN seeds, r0..3 x 0..511")
    return TRAIN_BASE + replica * 10_000 + index


@dataclass(frozen=True)
class World:
    seed: int
    mode: int
    goal: int
    initial_route: int
    delay: int
    slip: float

    @classmethod
    def generate(cls, seed: int) -> "World":
        rng = random.Random(seed)
        return cls(seed, rng.randrange(4), rng.randrange(2),
                   rng.randrange(2), rng.choice((1, 2)),
                   rng.choice((0.0, 0.05)))

    def roles(self, phase: int) -> dict[str, str]:
        """Private simulator mapping. NEVER passed to learning methods."""
        labels = ("A", "B", "C", "D")
        rotation = (self.mode + (2 if phase else 0)) % 4
        return {role: labels[(i + rotation) % 4]
                for i, role in enumerate(ROLES)}

    def reset(self, phase: int = 0) -> "Simulator":
        return Simulator(self, phase)


class Simulator:
    """Hidden queue and hidden rule phase; only observe() is given to the actor."""

    def __init__(self, world: World, phase: int):
        self.world = world
        self.phase = phase
        self.tick = 0
        self.power = 0
        self.stock = 0
        self.route = world.initial_route
        self.pending: list[tuple[int, str]] = []
        self.completed_effects = 0

    def observe(self) -> tuple[int, int, int, int, int, int]:
        w = self.world
        return (w.mode, w.goal, self.power, self.route, self.stock,
                min(len(self.pending), 2))

    def step(self, action: str, rng: random.Random) -> tuple[tuple, float, bool]:
        if action not in ACTIONS:
            raise ValueError("Unauthorized action")
        self.tick += 1
        reward, done = -0.025, False
        roles = self.world.roles(self.phase)
        selected = next((kind for kind, label in roles.items() if label == action),
                        "wait" if action == "E" else "hazard")
        if rng.random() < self.world.slip:
            reward = -0.065
        elif selected in {"charge", "gather", "route"}:
            # The effect materializes one or two whole actions later, and the
            # agent does NOT see which operation the queue contains.
            if len(self.pending) >= 2:
                reward = -0.09
            else:
                self.pending.append((self.tick + self.world.delay, selected))
        elif selected == "deliver":
            if self.power and self.stock and self.route == self.world.goal:
                reward, done = 1.0, True
                self.power = 0
                self.stock = 0
            else:
                reward = -0.13
        elif selected == "hazard":
            reward = -0.12
            self.stock = 0
        else:
            reward = -0.035

        ready = [kind for due, kind in self.pending if due <= self.tick]
        self.pending = [(due, kind) for due, kind in self.pending if due > self.tick]
        for effect in ready:
            if effect == "charge":
                self.power = 1
            elif effect == "gather":
                self.stock = 1
            elif effect == "route":
                self.route = 1 - self.route
        self.completed_effects += len(ready)
        return self.observe(), reward, done


def episode(world: World, policy, seed: int, phase: int) -> tuple[list[dict], int]:
    simulator = world.reset(phase)
    rng = random.Random(seed)
    steps = []
    for _ in range(HORIZON):
        before = simulator.observe()
        action = policy(before, rng)
        after, reward, done = simulator.step(action, rng)
        steps.append({"before": before, "action": action, "after": after,
                      "reward": reward, "done": done})
        if done:
            break
    return steps, simulator.completed_effects


def success(steps: list[dict]) -> bool:
    return bool(steps and steps[-1]["done"])


def total_reward(steps: list[dict]) -> float:
    return sum(x["reward"] for x in steps)


def react(obs, rng) -> str:
    return rng.choice(ACTIONS)


class Episodic:
    """A1: store FULL historical episodes and retrieve a matching raw episode.

    There are no learned per-state action scores, counters or cached policy.
    The index only accelerates access to raw (episode,step) pairs. A decision
    copies the *next observed action* of one retrieved episode.
    """

    def __init__(self) -> None:
        self.episodes: list[tuple[list[dict], bool]] = []
        self.postings: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        self.queries = 0

    def observe(self, steps: list[dict]) -> None:
        index = len(self.episodes)
        self.episodes.append((steps, success(steps)))
        for pos, row in enumerate(steps):
            self.postings[tuple(row["before"])].append((index, pos))

    def choose(self, obs, rng) -> str:
        self.queries += 1
        matches = self.postings.get(tuple(obs), ())
        if not matches:
            return rng.choice(ACTIONS)
        # Retrieve up to K recent instances. Compare complete episodes rather
        # than averaging actions; prefer successful and shorter solutions.
        recent = list(matches[-RECALL_K:])
        # Reserve an equal number of retrieval slots for positive examples,
        # without constructing or caching a policy/value table.
        positive = [item for item in reversed(matches)
                    if self.episodes[item[0]][1]][:RECALL_K]
        candidates = list(dict.fromkeys(recent + positive))
        best = min(candidates, key=lambda item: (
            not self.episodes[item[0]][1],
            len(self.episodes[item[0]][0]) - item[1],
            -item[0],
        ))
        steps, _ = self.episodes[best[0]]
        return steps[best[1]]["action"]

    def snapshot(self) -> dict:
        return {"episodes": len(self.episodes), "steps": sum(len(e) for e, _ in self.episodes),
                "retrieval_k": RECALL_K, "queries": self.queries,
                "source": "train_events.jsonl.gz"}


class Effects:
    """A2: public transition model with bounded expected-value planning."""

    def __init__(self) -> None:
        self.transitions: dict[tuple, Counter] = defaultdict(Counter)
        self.steps = 0
        self.queries = 0

    def observe(self, steps: list[dict]) -> None:
        for step in steps:
            key = (tuple(step["before"]), step["action"])
            result = (tuple(step["after"]), step["reward"], step["done"])
            self.transitions[key][result] += 1
            self.steps += 1
        self.value.cache_clear()

    @lru_cache(maxsize=40_000)
    def value(self, state: tuple, depth: int) -> float:
        if depth == 0:
            return 0.0
        return max(self.action_value(state, a, depth) for a in ACTIONS)

    def action_value(self, state: tuple, action: str, depth: int) -> float:
        counts = self.transitions.get((state, action))
        if not counts:
            return -0.065
        return sum(count * (reward + (0 if done or depth <= 1 else
                                     0.95 * self.value(next_state, depth - 1)))
                   for (next_state, reward, done), count in counts.items()) / sum(counts.values())

    def choose(self, obs, rng) -> str:
        self.queries += 1
        scores = {action: self.action_value(tuple(obs), action, PLAN_DEPTH)
                  for action in ACTIONS}
        high = max(scores.values())
        return rng.choice(sorted(a for a, score in scores.items()
                                 if abs(score - high) < 1e-10))

    def snapshot(self) -> list[dict]:
        return [{"state": list(state), "action": action,
                 "outcomes": [{"after": list(s), "reward": reward, "done": done, "n": n}
                              for (s, reward, done), n in sorted(counts.items())]}
                for (state, action), counts in sorted(self.transitions.items())]


class Options:
    """A3: whole successful trajectories reused conditionally, not expert rules."""

    def __init__(self) -> None:
        self.support: Counter = Counter()
        self.contradicted: Counter = Counter()
        self.by_start: dict[tuple, set] = defaultdict(set)
        self.failures_from_self = 0

    def observe(self, steps: list[dict]) -> None:
        if not success(steps):
            return
        trace = tuple((tuple(x["before"]), x["action"]) for x in steps)
        self.support[trace] += 1
        self.by_start[trace[0][0]].add(trace)

    def start(self, obs: tuple):
        matches = list(self.by_start.get(tuple(obs), ()))
        if not matches:
            return None
        # Evidence-driven reliability, with failures after the rule flip
        # downweighting obsolete procedures.
        return min(matches, key=lambda tr: (
            -(self.support[tr] - 3 * self.contradicted[tr]),
            len(tr), tr))

    def feedback(self, trace: tuple | None, passed: bool) -> None:
        if trace is None or trace not in self.support:
            return
        if not passed:
            self.contradicted[trace] += 1
            self.failures_from_self += 1
        else:
            self.support[trace] += 1

    def snapshot(self) -> list[dict]:
        return [{"trace": [[list(obs), action] for obs, action in trace],
                 "support": n, "contradicted": self.contradicted[trace]}
                for trace, n in sorted(self.support.items())]


class OptionController:
    def __init__(self, options: Options, effects: Effects, *, enabled: bool = True):
        self.options, self.effects, self.enabled = options, effects, enabled
        self.trace = None
        self.selected_trace = None
        self.cursor = 0
        self.reused = 0
        self.aborted = False

    def choose(self, obs, rng) -> str:
        if not self.enabled:
            return self.effects.choose(obs, rng)
        if self.trace is None and self.cursor == 0:
            self.trace = self.options.start(tuple(obs))
            self.selected_trace = self.trace
        if self.trace is not None:
            if self.cursor < len(self.trace) and self.trace[self.cursor][0] == tuple(obs):
                action = self.trace[self.cursor][1]
                self.cursor += 1
                self.reused += 1
                return action
            self.aborted = self.cursor < len(self.trace)
            self.trace = None
        return self.effects.choose(obs, rng)


def privileged(world: World, phase: int):
    """A4 bounded heuristic; only allowed in TRAIN diagnostic probes."""
    roles = world.roles(phase)

    def policy(obs, rng):
        _, goal, energy, route, stock, pending = obs
        if pending:
            return "E"
        if not energy:
            return roles["charge"]
        if not stock:
            return roles["gather"]
        if route != goal:
            return roles["route"]
        return roles["deliver"]

    return policy


def train(replica: int, worlds: int = MAX_WORLDS, episodes: int = MAX_EPISODES,
          probes: int = PROBES_PER_WORLD, out: Path | None = None) -> dict:
    if not 0 <= replica < 4 or not 1 <= worlds <= MAX_WORLDS or not 1 <= episodes <= MAX_EPISODES:
        raise ValueError("F11-A TRAIN bounds exceeded")
    if not 0 <= probes <= PROBES_PER_WORLD:
        raise ValueError("TRAIN probe budget exceeded")
    if worlds > 1 and train_seed(replica, worlds-1) >= 61_200_000:
        raise ValueError("DEV seeds must never be opened")
    bank, effects, options = Episodic(), Effects(), Options()
    training = {"episodes": 0, "actions": 0, "wins": 0,
                "delayed_effects_resolved": 0, "phase0_wins": 0, "phase1_wins": 0,
                "policy_collector_episodes": 0, "option_feedback_failures": 0}
    outcome = {str(p): {a: {"won": 0, "episodes": 0, "reward": 0.0, "actions": 0}
                        for a in ("A0", "A1", "A2", "A3", "A3_no_options", "A4")}
               for p in (0, 1)}
    event_hash = hashlib.sha256()
    file = None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        file = gzip.open(out / "train_events.jsonl.gz", "wt", encoding="utf-8")
    try:
        for i in range(worlds):
            world = World.generate(train_seed(replica, i))
            for j in range(episodes):
                phase = int(j >= CHANGE_AT)
                controlled = j >= 5 and j % 5 == 0
                controller = OptionController(options, effects)
                chosen = controller.choose if controlled else react
                steps, resolved = episode(world, chosen, world.seed * 100 + j, phase)
                if controlled:
                    options.feedback(controller.selected_trace, success(steps))
                    training["policy_collector_episodes"] += 1
                record = {"split": "TRAIN", "world_seed": world.seed, "episode": j,
                          "phase_for_audit_only": phase, "source": "options" if controlled else "random",
                          "steps": steps}
                line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                event_hash.update(line.encode())
                if file:
                    file.write(line)
                # Exact same public experience for all learning arms.
                bank.observe(steps)
                effects.observe(steps)
                options.observe(steps)
                training["episodes"] += 1
                training["actions"] += len(steps)
                training["wins"] += int(success(steps))
                training["delayed_effects_resolved"] += resolved
                training["phase%d_wins" % phase] += int(success(steps))
        training["option_feedback_failures"] = options.failures_from_self
    finally:
        if file is not None:
            file.close()

    reuse = aborts = 0
    for i in range(worlds):
        world = World.generate(train_seed(replica, i))
        for j in range(probes):
            phase = j % 2
            rng_seed = world.seed * 100 + MAX_EPISODES + j
            controller = OptionController(options, effects)
            no_options = OptionController(options, effects, enabled=False)
            arms = {"A0": react, "A1": bank.choose, "A2": effects.choose,
                    "A3": controller.choose, "A3_no_options": no_options.choose,
                    "A4": privileged(world, phase)}
            for label, policy in arms.items():
                steps, _ = episode(world, policy, rng_seed, phase)
                cell = outcome[str(phase)][label]
                cell["won"] += int(success(steps))
                cell["episodes"] += 1
                cell["reward"] += total_reward(steps)
                cell["actions"] += len(steps)
            reuse += controller.reused
            aborts += int(controller.aborted)

    summary = {
        "experiment": "F11-A", "version": "TRAIN_V2",
        "status": "TRAIN_ONLY_NO_SCIENTIFIC_PASS",
        "replica": replica,
        "train_seed_first": train_seed(replica, 0),
        "train_seed_last": train_seed(replica, worlds - 1),
        "worlds": worlds, "episodes_per_world": episodes,
        "training": training, "probe_results_by_rule_phase": outcome,
        "unique_learned_options": len(options.support),
        "options_reused_steps_in_train_probes": reuse,
        "guard_aborts_in_train_probes": aborts,
        "shared_event_sha256": event_hash.hexdigest(),
        "effects_observed_steps": effects.steps,
        "episodic": bank.snapshot(),
        "effects_queries": effects.queries,
        "probe_worlds_are_training_worlds": True,
        "seeds_dev_opened": False, "seeds_confirm_opened": False,
        "all_arms_saw_identical_training_experience": True,
    }
    if out is not None:
        checkpoint = {"experiment": "F11-A", "version": "TRAIN_V2",
                      "status": "TRAIN_ONLY", "event_sha256": event_hash.hexdigest(),
                      "options": options.snapshot(), "effects": effects.snapshot(),
                      "a1": bank.snapshot(),
                      "episodic_source": "train_events.jsonl.gz"}
        (out / "train_summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
        (out / "checkpoint.json").write_text(
            json.dumps(checkpoint, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="F11-A delayed and changing-rule TRAIN v2 only")
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--worlds", type=int, default=MAX_WORLDS)
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--probes", type=int, default=PROBES_PER_WORLD)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = train(args.replica, args.worlds, args.episodes, args.probes, args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
