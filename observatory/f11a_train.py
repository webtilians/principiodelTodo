"""F11-A training-only pilot: shared exploration, learned effects and reusable policy options.

Scientific status: TRAIN ONLY. No DEV/CONFIRM seeds or claims of transfer.
The agent sees ONLY observation, action and feedback. No evaluator/oracle labels
are passed into the learners. The entire study uses synthetic stdlib environments.
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

ACTIONS = ("A", "B", "C", "D")
TRAIN_BASE = 61_100_000
MAX_WORLDS = 512
MAX_EPISODES = 30
HORIZON = 6
PROBES_PER_WORLD = 2


def train_seed(replica: int, index: int) -> int:
    """Only the preregistered TRAIN range is available in this module."""
    if not (0 <= replica < 4 and 0 <= index < MAX_WORLDS):
        raise ValueError("TRAIN requires replica 0..3 and world index 0..511")
    return TRAIN_BASE + replica * 10_000 + index


@dataclass(frozen=True)
class World:
    seed: int
    mode: int
    goal: int
    initial_route: int
    slip: float

    @classmethod
    def generate(cls, seed: int) -> "World":
        rng = random.Random(seed)
        return cls(seed, rng.randrange(3), rng.randrange(2),
                   rng.randrange(2), rng.choice((0.0, 0.1)))

    def reset(self) -> tuple[int, int, int, int]:
        return self.mode, self.goal, 0, self.initial_route

    def _roles(self) -> dict[str, str]:
        # Simulator-only mechanism. NEVER surfaced to training or evaluable actors.
        # Each public mode maps arbitrary opaque action labels to different effects.
        if self.mode == 0:
            return {"charge": "A", "route": "B", "deliver": "D"}
        if self.mode == 1:
            return {"charge": "B", "route": "C", "deliver": "D"}
        return {"charge": "C", "route": "A", "deliver": "D"}

    def step(self, obs: tuple[int, int, int, int], action: str,
             rng: random.Random) -> tuple[tuple[int, int, int, int], float, bool]:
        if action not in ACTIONS:
            raise ValueError("Action outside the public permitted catalog")
        mode, goal, power, route = obs
        if mode != self.mode or goal != self.goal:
            raise ValueError("Observation does not belong to this public task")
        if rng.random() < self.slip:
            return obs, -0.04, False
        roles = self._roles()
        if action == roles["charge"]:
            power = 1
        elif action == roles["route"]:
            route = 1 - route
        elif action == roles["deliver"]:
            if power and route == goal:
                return (mode, goal, power, route), 0.98, True
            return obs, -0.08, False
        return (mode, goal, power, route), -0.02, False


def episode(world: World, policy, seed: int) -> list[dict]:
    rng = random.Random(seed)
    obs = world.reset()
    steps = []
    for _ in range(HORIZON):
        action = policy(obs, rng)
        after, reward, done = world.step(obs, action, rng)
        steps.append({"before": obs, "action": action, "after": after,
                      "reward": reward, "done": done})
        obs = after
        if done:
            break
    return steps


def won(steps: list[dict]) -> bool:
    return bool(steps and steps[-1]["done"])


def random_policy(obs, rng) -> str:
    return rng.choice(ACTIONS)


class Episodic:
    """A1: retrieve observed actions from episodes; no persistent general rules."""

    def __init__(self) -> None:
        self.by_state: dict[tuple, Counter] = defaultdict(Counter)
        self.count = 0

    def observe(self, steps: list[dict]) -> None:
        self.count += 1
        success = won(steps)
        for i, step in enumerate(steps):
            key = tuple(step["before"])
            # Retrieval scores depend on observed outcomes, never evaluator truth.
            self.by_state[key][step["action"]] += (2 if success else 0.25) / (1 + i)

    def choose(self, obs, rng) -> str:
        matches = self.by_state.get(obs)
        if not matches:
            return rng.choice(ACTIONS)
        best = max(matches.values())
        return rng.choice(sorted(a for a, score in matches.items() if score == best))


class Effects:
    """A2: estimate public transitions, then plan; no cached action policy."""

    def __init__(self) -> None:
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        self.observed_steps = 0

    def observe(self, steps: list[dict]) -> None:
        for step in steps:
            key = (tuple(step["before"]), step["action"])
            result = (tuple(step["after"]), step["reward"], step["done"])
            self.counts[key][result] += 1
            self.observed_steps += 1
        self.value.cache_clear()

    @lru_cache(maxsize=None)
    def value(self, obs: tuple, depth: int) -> float:
        if depth <= 0:
            return 0.0
        return max(self.action_value(obs, action, depth) for action in ACTIONS)

    def action_value(self, obs: tuple, action: str, depth: int) -> float:
        found = self.counts.get((obs, action))
        if not found:
            return -0.06  # unknown action: no privileged default outcome
        total = sum(found.values())
        return sum(n * (reward + (0.0 if done else 0.95 * self.value(next_obs, depth - 1)))
                   for (next_obs, reward, done), n in found.items()) / total

    def choose(self, obs, rng, remaining: int = HORIZON) -> str:
        scores = {a: self.action_value(obs, a, remaining) for a in ACTIONS}
        top = max(scores.values())
        return rng.choice(sorted(a for a in ACTIONS if abs(scores[a] - top) < 1e-10))


class Options:
    """A3: induce guarded multi-action options only from successful experiences.

    The initiation condition is the first observed state. An option is aborted
    whenever subsequent observed states diverge from its recorded guards.
    No expert-written rule, plan, or environment transition is passed in.
    """

    def __init__(self) -> None:
        self.support: Counter = Counter()
        self.failed: Counter = Counter()

    def observe(self, steps: list[dict]) -> None:
        if not won(steps):
            return
        trace = tuple((tuple(s["before"]), s["action"]) for s in steps)
        self.support[trace] += 1

    def start(self, obs: tuple):
        matches = [(trace, n) for trace, n in self.support.items()
                   if trace and trace[0][0] == obs]
        if not matches:
            return None
        # Pick reliable short procedures; all traces originate in observed success.
        matches.sort(key=lambda row: (-(row[1] - self.failed[row[0]]),
                                      len(row[0]), row[0]))
        return matches[0][0]

    def update(self, trace: tuple, succeeded: bool) -> None:
        """Online contradiction response for later adaptation experiments."""
        if trace in self.support:
            if succeeded:
                self.support[trace] += 1
            else:
                self.failed[trace] += 1

    def snapshot(self) -> list[dict]:
        return [
            {"trace": [[list(obs), action] for obs, action in trace],
             "support": count, "failures": self.failed[trace]}
            for trace, count in sorted(self.support.items())
        ]


class OptionController:
    """Uses learned guarded options; otherwise falls back to the same A2 model."""

    def __init__(self, options: Options, effects: Effects):
        self.options = options
        self.effects = effects
        self.current = None
        self.cursor = 0
        self.reused = False
        self.aborted = False

    def choose(self, obs, rng) -> str:
        if self.cursor == 0 and self.current is None:
            self.current = self.options.start(obs)
        if self.current is not None:
            if self.cursor < len(self.current) and self.current[self.cursor][0] == obs:
                action = self.current[self.cursor][1]
                self.cursor += 1
                self.reused = True
                return action
            self.aborted = True
            self.current = None
        return self.effects.choose(obs, rng, max(1, HORIZON - self.cursor))


def privileged_oracle(world: World):
    """A4 diagnostic ceiling; never used in episode collection or learner input."""
    roles = world._roles()

    def choose(obs, rng):
        _, goal, power, route = obs
        if not power:
            return roles["charge"]
        if route != goal:
            return roles["route"]
        return roles["deliver"]

    return choose


def effect_snapshot(effects: Effects) -> list[dict]:
    return [
        {"state": list(state), "action": action,
         "outcomes": [{"next": list(next_state), "reward": reward,
                       "done": done, "count": n}
                      for (next_state, reward, done), n in sorted(outcomes.items())]}
        for (state, action), outcomes in sorted(effects.counts.items())
    ]


def train(replica: int, worlds: int = MAX_WORLDS,
          episodes: int = MAX_EPISODES, probes: int = PROBES_PER_WORLD,
          out: Path | None = None) -> dict:
    if not (1 <= worlds <= MAX_WORLDS and 1 <= episodes <= MAX_EPISODES
            and 0 <= probes <= 4):
        raise ValueError("Training bounds exceed preregistration")
    if not 0 <= replica <= 3:
        raise ValueError("Replica must be 0..3")
    bank, effects, options = Episodic(), Effects(), Options()
    counters = {a: {"won": 0, "episodes": 0, "actions": 0} for a in
                ("A0", "A1", "A2", "A3", "A4")}
    training_actions = training_wins = training_episodes = 0
    event_hash = hashlib.sha256()
    file = None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        file = gzip.open(out / "train_events.jsonl.gz", mode="wt", encoding="utf-8")
    try:
        for i in range(worlds):
            world = World.generate(train_seed(replica, i))
            for j in range(episodes):
                # One shared public experience log is supplied to A1/A2/A3.
                steps = episode(world, random_policy, world.seed * 100 + j)
                row = {"split": "TRAIN", "world_seed": world.seed, "episode": j,
                       "steps": steps}
                line = json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
                event_hash.update(line.encode())
                if file is not None:
                    file.write(line)
                bank.observe(steps)
                effects.observe(steps)
                options.observe(steps)
                training_wins += won(steps)
                training_episodes += 1
                training_actions += len(steps)
    finally:
        if file is not None:
            file.close()

    reused = aborted = 0
    # In-sample TRAIN diagnostics only; same worlds, never DEV/CONFIRM.
    for i in range(worlds):
        world = World.generate(train_seed(replica, i))
        for j in range(probes):
            seed = world.seed * 100 + MAX_EPISODES + j
            controller = OptionController(options, effects)
            arms = {
                "A0": random_policy,
                "A1": bank.choose,
                "A2": effects.choose,
                "A3": controller.choose,
                "A4": privileged_oracle(world),
            }
            for label, policy in arms.items():
                steps = episode(world, policy, seed)
                counters[label]["won"] += int(won(steps))
                counters[label]["episodes"] += 1
                counters[label]["actions"] += len(steps)
            reused += int(controller.reused)
            aborted += int(controller.aborted)

    summary = {
        "status": "TRAIN_ONLY_NOT_SCIENTIFIC_RESULT",
        "experiment": "F11-A", "replica": replica,
        "phase": "TRAIN", "train_seed_first": train_seed(replica, 0),
        "train_seed_last": train_seed(replica, worlds - 1),
        "worlds": worlds, "episodes_per_world": episodes,
        "training_episodes": training_episodes,
        "training_actions": training_actions,
        "exploration_wins": training_wins,
        "shared_event_sha256": event_hash.hexdigest(),
        "effects_observed_steps": effects.observed_steps,
        "learned_option_count": len(options.support),
        "learned_option_reuse": reused,
        "learned_option_guard_aborts": aborted,
        "probe_worlds_are_training_worlds": True,
        "probe_results": counters,
        "dev_opened": False, "confirm_opened": False,
    }
    if out is not None:
        checkpoint = {
            "experiment": "F11-A", "replica": replica,
            "status": "TRAIN_ONLY", "shared_event_sha256": event_hash.hexdigest(),
            "effects": effect_snapshot(effects),
            "learned_options": options.snapshot(),
            # A1 needs episodic records to resume; the full source is train_events.
            "episodic_source": "train_events.jsonl.gz",
        }
        (out / "checkpoint.json").write_text(
            json.dumps(checkpoint, sort_keys=True, indent=2), encoding="utf-8"
        )
        (out / "train_summary.json").write_text(
            json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="F11-A training only (never DEV/CONFIRM)")
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--worlds", type=int, default=MAX_WORLDS)
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--probes", type=int, default=PROBES_PER_WORLD)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = train(args.replica, args.worlds, args.episodes, args.probes, args.out)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
