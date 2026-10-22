"""F4: Learn which events merit scarce memory using *past* query feedback.

Controlled research benchmark. No consciousness metric, no oracle queries in
the memory selection API, no rewards derived from Phi. Pure Python 3.10+.
This is a separate key-value controller, NOT yet wired into V5.1's soft slots.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

KEYS = 8
CAPACITY = 3
PHASE_LENGTH = 120
TRAIN_PHASES = 3
TEST_LENGTH = 160
FREQUENT_MASS = 0.86


@dataclass(frozen=True)
class Event:
    key: int
    payload: int


@dataclass(frozen=True)
class Episode:
    events: Tuple[Event, ...]
    # Hidden until AFTER each policy has made all its storage decisions:
    future_query_key: int
    fallback_guess: int


class Store:
    """One memory budget, used by ALL strategies with the same episode."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.saved: Dict[int, int] = {}
        self.order: List[int] = []

    def put(self, event: Event, evict_key: Optional[int] = None) -> None:
        if event.key in self.saved:
            self.saved[event.key] = event.payload
            return
        if len(self.saved) >= self.capacity:
            if evict_key is None or evict_key not in self.saved:
                raise ValueError("full memory: explicitly choose a key to evict")
            del self.saved[evict_key]
            self.order.remove(evict_key)
        self.saved[event.key] = event.payload
        self.order.append(event.key)

    def recall(self, key: int) -> Optional[int]:
        return self.saved.get(key)


class Selector:
    """Selection interface deliberately never receives the future query."""
    def __init__(self, capacity: int, seed: int):
        self.capacity = capacity
        self.rng = random.Random(seed)

    def begin(self) -> Store:
        return Store(self.capacity)

    def consider(self, store: Store, event: Event) -> None:
        raise NotImplementedError

    def feedback(self, queried_key: int) -> None:
        """Called ONLY after memory decisions and the answer."""


class FIFO(Selector):
    def consider(self, store: Store, event: Event) -> None:
        store.put(event, store.order[0] if len(store.saved) == store.capacity else None)


class RandomReservoir(Selector):
    def begin(self) -> Store:
        self._seen = 0
        return super().begin()

    def consider(self, store: Store, event: Event) -> None:
        self._seen += 1
        if len(store.saved) < store.capacity:
            store.put(event)
        elif self.rng.randrange(self._seen) < store.capacity:
            evict = self.rng.choice(store.order)
            store.put(event, evict)


class LearnedRelevance(Selector):
    """Online discounted query-demand model and prospective write/evict gate.

    Learns from identities of *past* requested keys, never the current query.
    All key weights start equal. The controller uses only visible event key,
    running history of previous queries, and its memory's contents.
    """
    def __init__(self, capacity: int, seed: int, decay: float = 0.97):
        super().__init__(capacity, seed)
        if not 0 < decay < 1:
            raise ValueError("0 < decay < 1 required")
        self.decay = decay
        self.weights = [1.0 for _ in range(KEYS)]
        self.feedback_count = 0
        self.frozen = False

    def _priority(self, key: int) -> float:
        return self.weights[key]

    def consider(self, store: Store, event: Event) -> None:
        if event.key in store.saved:
            store.put(event)
        elif len(store.saved) < store.capacity:
            store.put(event)
        else:
            weakest = min(self._priority(key) for key in store.order)
            options = [key for key in store.order if abs(self._priority(key) - weakest) < 1e-12]
            evict = self.rng.choice(options)
            if self._priority(event.key) > weakest or (
                abs(self._priority(event.key) - weakest) < 1e-12 and self.rng.random() < .5
            ):
                store.put(event, evict)

    def feedback(self, queried_key: int) -> None:
        if self.frozen:
            return
        self.weights = [max(1e-8, w * self.decay) for w in self.weights]
        self.weights[queried_key] += 1.0
        self.feedback_count += 1

    def freeze(self) -> None:
        self.frozen = True

    def predicted_top(self) -> List[int]:
        return sorted(range(KEYS), key=lambda key: (-self.weights[key], key))[:CAPACITY]


def build_profile(seed: int) -> List[Tuple[int, int]]:
    rng = random.Random(seed * 7919 + 41)
    order = list(range(KEYS))
    rng.shuffle(order)
    return [(order[0], order[1]), (order[2], order[3]), (order[4], order[5])]


def make_episode(seed: int, index: int, relevant: Tuple[int, int], uniform: bool) -> Episode:
    # These are generated by an environment-only RNG. The selector does NOT
    # see the future query, relevance profile, or whether this is a test item.
    rng = random.Random(seed * 10000019 + index * 8191 + (789 if uniform else 13))
    keys = list(range(KEYS))
    rng.shuffle(keys)
    events = tuple(Event(key, rng.randrange(2)) for key in keys)
    if uniform:
        requested = rng.randrange(KEYS)
    elif rng.random() < FREQUENT_MASS:
        requested = rng.choice(relevant)
    else:
        requested = rng.choice([key for key in range(KEYS) if key not in relevant])
    return Episode(events=events, future_query_key=requested, fallback_guess=rng.randrange(2))


def attempt(selector: Selector, ep: Episode) -> dict:
    memory = selector.begin()
    for event in ep.events:
        selector.consider(memory, event)
        assert len(memory.saved) <= CAPACITY
    # Boundary: query becomes available to the experiment ONLY now.
    target = next(event.payload for event in ep.events if event.key == ep.future_query_key)
    prediction = memory.recall(ep.future_query_key)
    hit = prediction is not None
    if not hit:
        prediction = ep.fallback_guess
    out = {"hit": int(hit), "correct": int(prediction == target),
           "stored": len(memory.saved)}
    selector.feedback(ep.future_query_key)
    return out


def evaluate(seed: int, *, uniform: bool = False,
             phase_length: int = PHASE_LENGTH,
             test_length: int = TEST_LENGTH) -> dict:
    profile = build_profile(seed)
    choices: Dict[str, Selector] = {
        "adaptive": LearnedRelevance(CAPACITY, seed * 7 + 1),
        "static_first_regime": LearnedRelevance(CAPACITY, seed * 7 + 2),
        "fifo": FIFO(CAPACITY, seed * 7 + 3),
        "reservoir": RandomReservoir(CAPACITY, seed * 7 + 4),
    }
    trace = []
    for episode in range(phase_length * TRAIN_PHASES + test_length):
        holdout = episode >= phase_length * TRAIN_PHASES
        phase = min(episode // phase_length, TRAIN_PHASES - 1)
        if episode == phase_length:
            choices["static_first_regime"].freeze()
        if holdout and episode == phase_length * TRAIN_PHASES:
            choices["adaptive"].freeze()
        observed = make_episode(seed, episode, profile[phase], uniform)
        result = {name: attempt(strategy, observed)
                  for name, strategy in choices.items()}
        trace.append({"phase": phase, "holdout": holdout, "step": episode,
                      "query_key": observed.future_query_key,
                      "result": result})
    segments = {}
    for phase in range(TRAIN_PHASES):
        points = [p for p in trace if p["phase"] == phase and not p["holdout"]]
        segments["regime_" + str(phase)] = {
            name: {
                "hit": statistics.mean(p["result"][name]["hit"] for p in points),
                "first_25_hit": statistics.mean(p["result"][name]["hit"] for p in points[:25]),
                "last_25_hit": statistics.mean(p["result"][name]["hit"] for p in points[-25:]),
            } for name in choices
        }
    heldout = [p for p in trace if p["holdout"]]
    segments["heldout_frozen"] = {
        name: {
            "hit": statistics.mean(p["result"][name]["hit"] for p in heldout),
            "accuracy": statistics.mean(p["result"][name]["correct"] for p in heldout),
        } for name in choices
    }
    return {"seed": seed,
            "hidden_regime_pairs": [list(pair) for pair in profile],
            "learned_top_keys_at_end": choices["adaptive"].predicted_top(),
            "uniform_query_control": uniform,
            "segments": segments,
            "trace": trace}


def cluster_bootstrap_difference(results, challenger, baseline,
                                 iterations: int = 2000, seed: int = 71) -> dict:
    """Paired uncertainty resampling whole seeds; do not treat episodes as iid."""
    diffs = [
        r["segments"]["heldout_frozen"][challenger]["hit"]
        - r["segments"]["heldout_frozen"][baseline]["hit"] for r in results
    ]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(diffs, k=len(diffs)))
                   for _ in range(iterations))
    return {"mean": statistics.mean(diffs), "ci95": [draws[49], draws[1949]],
            "unit": "hit_rate_points", "independent_units": len(diffs),
            "note": "exploratory seed-cluster bootstrap; not causal proof of cognition"}


def summarize(results, *, uniform: bool) -> dict:
    strategies = ("adaptive", "static_first_regime", "fifo", "reservoir")
    report = {
        "condition": "uniform_negative_control" if uniform else "hidden_drifting_query_demand",
        "episodes_per_seed": PHASE_LENGTH * TRAIN_PHASES + TEST_LENGTH,
        "heldout_is_frozen": True,
        "strategies": {},
        "regime_learning": {},
        "paired_differences": {},
    }
    for s in strategies:
        report["strategies"][s] = {
            "heldout_hit": statistics.mean(r["segments"]["heldout_frozen"][s]["hit"] for r in results),
            "heldout_accuracy": statistics.mean(r["segments"]["heldout_frozen"][s]["accuracy"] for r in results),
        }
        report["regime_learning"][s] = {
            "phase_%s_first25" % phase: statistics.mean(r["segments"]["regime_%s" % phase][s]["first_25_hit"] for r in results)
            for phase in range(TRAIN_PHASES)
        } | {
            "phase_%s_last25" % phase: statistics.mean(r["segments"]["regime_%s" % phase][s]["last_25_hit"] for r in results)
            for phase in range(TRAIN_PHASES)
        }
    for baseline in strategies[1:]:
        report["paired_differences"]["adaptive_minus_" + baseline] = cluster_bootstrap_difference(
            results, "adaptive", baseline)
    return report


def main():
    p = argparse.ArgumentParser(description="F4 blind-horizon selective memory evaluation")
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--output", type=Path, default=Path("observatory_results/selective_memory"))
    args = p.parse_args()
    if not 4 <= args.seeds <= 250:
        raise ValueError("seeds must be 4..250")
    args.output.mkdir(parents=True, exist_ok=True)
    combined = {}
    for uniform in (False, True):
        results = [evaluate(i + 1 + (10000 if uniform else 0), uniform=uniform)
                   for i in range(args.seeds)]
        label = "uniform_negative_control" if uniform else "drifting_relevance"
        summary = summarize(results, uniform=uniform)
        combined[label] = summary
        with (args.output / (label + "_per_seed.jsonl")).open("w") as file:
            for run in results:
                file.write(json.dumps(run, ensure_ascii=False) + "\n")
        print("CASE", label, "HOLDOUT", {
            k: round(v["heldout_hit"], 4) for k, v in summary["strategies"].items()
        }, flush=True)
    (args.output / "report.json").write_text(
        json.dumps({
            "experiment": "F4 selective-write learning; online query feedback, no future access",
            "controller": "discounted categorical query demand; no hand-picked relevant keys",
            "conditions": combined,
            "seeds": args.seeds,
            "capacity": CAPACITY, "keys": KEYS,
            "limits": (
                "Standalone symbolic key-value controller, NOT integrated into "
                "V5.1 neural memory. Predictable demand for key IDs, not inferred "
                "semantic significance or subjective experience. "
                "Future query revealed only after the storage decision; "
                "heldout adaptation frozen; negative uniform control included."
            ),
        }, ensure_ascii=False, indent=2))
    print("REPORT", str(args.output / "report.json"), flush=True)


if __name__ == "__main__":
    main()
