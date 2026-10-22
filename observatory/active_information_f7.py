"""F7: active sensing by learned value of information.

This is a compact POMDP with one OPTIONAL information-gathering action
followed by a terminal action. Everything evaluative is predeclared in
docs/INFINITO_F7_ACTIVE_INFORMATION_PROTOCOL.md.

Only the environment knows latent z and the sensor reliabilities. Every
agent sees the same potential observations, but receives a sensor signal
only if it chooses to pay to inspect. Feedback z arrives after decision.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

PROBE_COST = 0.12
TRAIN_N = 240
TEST_N = 600
TRAIN_SEEDS = 24
CUE_ACCURACIES = (0.60, 0.82)
INFORMATIVE_ACCURACY = 0.89
STAKE_VALUES = (0.2, 1.0)


@dataclass(frozen=True)
class WorldRules:
    informative_source: int
    reliabilities: tuple[float, float]
    cue_accuracies: tuple[float, float] = CUE_ACCURACIES


@dataclass(frozen=True)
class Episode:
    latent: int     # Not passed to decision function.
    cue: int
    difficulty: int
    stakes: float
    signals: tuple[int, int]  # Not passed unless a probe is purchased.


def world_rules(seed: int, *, unlearnable: bool, swapped: bool = False) -> WorldRules:
    informative = random.Random(seed * 1307 + 89).randrange(2)
    if swapped:
        informative = 1 - informative
    reliabilities = ([0.5, 0.5] if unlearnable else
                     [INFORMATIVE_ACCURACY if i == informative else 0.5 for i in range(2)])
    return WorldRules(informative, tuple(reliabilities))


def episodes(seed: int, n: int, *, unlearnable: bool = False, change_at=None):
    """Generate hidden world in advance: no policy can alter its RNG draws."""
    source = random.Random(seed * 90709 + (71 if unlearnable else 37))
    output = []
    for t in range(n):
        rule = world_rules(seed, unlearnable=unlearnable,
                           swapped=change_at is not None and t >= change_at)
        latent = source.randrange(2)
        difficulty = source.randrange(2)
        stakes = STAKE_VALUES[source.randrange(2)]
        cue = latent if source.random() < rule.cue_accuracies[difficulty] else 1 - latent
        signals = tuple(
            latent if source.random() < reliability else 1 - latent
            for reliability in rule.reliabilities
        )
        output.append(Episode(latent, cue, difficulty, stakes, signals))
    return output


class BayesianCalibration:
    """Beta-smoothed binary signal-reliability learner.

    Learns correctness AFTER the terminal feedback. At decision time,
    reliability is estimated only from previous episodes. Counts are
    updated online for sensors that were ACTUALLY inspected.
    """
    def __init__(self):
        self.cue_correct = [2.0, 2.0]
        self.cue_wrong = [2.0, 2.0]
        self.sensor_correct = [2.0, 2.0]
        self.sensor_wrong = [2.0, 2.0]
        self.sensor_reads = [0, 0]

    def cue_accuracy(self, difficulty):
        return self.cue_correct[difficulty] / (
            self.cue_correct[difficulty] + self.cue_wrong[difficulty])

    def sensor_accuracy(self, source):
        return self.sensor_correct[source] / (
            self.sensor_correct[source] + self.sensor_wrong[source])

    def observe(self, episode: Episode, inspected: int | None):
        correct = self.cue_correct if episode.cue == episode.latent else self.cue_wrong
        correct[episode.difficulty] += 1
        if inspected is not None:
            correct = (self.sensor_correct if episode.signals[inspected] == episode.latent
                       else self.sensor_wrong)
            correct[inspected] += 1
            self.sensor_reads[inspected] += 1


def calibrated_model(seed: int, calibration: list[Episode]) -> BayesianCalibration:
    """Same offline random inspection episodes for ALL experimental arms."""
    model = BayesianCalibration()
    rng = random.Random(seed * 2003 + 99)
    for ep in calibration:
        inspected = rng.randrange(2)
        model.observe(ep, inspected)
    return model


def posterior(cue: int, difficulty: int, model: BayesianCalibration) -> float:
    q = model.cue_accuracy(difficulty)
    return q if cue else 1.0 - q


def assimilate(p: float, observation: int, reliability: float) -> float:
    true_like = reliability if observation else (1 - reliability)
    false_like = 1 - reliability if observation else reliability
    numerator = p * true_like
    denominator = numerator + (1 - p) * false_like
    return numerator / denominator if denominator > 0 else p


def expected_terminal_value(p: float, stake: float) -> float:
    return stake * (2 * max(p, 1 - p) - 1)


def expected_inspection_value(p: float, stake: float, reliability: float) -> float:
    """Enumerate unobserved possible sensor outcomes before buying them."""
    expect = 0.0
    for observed in (0, 1):
        likelihood = (p * (reliability if observed else 1 - reliability)
                      + (1 - p) * (1 - reliability if observed else reliability))
        q = assimilate(p, observed, reliability)
        expect += likelihood * expected_terminal_value(q, stake)
    return expect - PROBE_COST


def choose_probe(name: str, model: BayesianCalibration, ep: Episode,
                 rng: random.Random, oracle: WorldRules | None = None) -> int | None:
    if name == "never":
        return None
    if name == "random":
        return rng.randrange(2) if rng.random() < 0.5 else None
    if oracle:
        cue = oracle.cue_accuracies[ep.difficulty]
        p = cue if ep.cue else 1 - cue
        reliabilities = oracle.reliabilities
    else:
        p = posterior(ep.cue, ep.difficulty, model)
        reliabilities = (model.sensor_accuracy(0), model.sensor_accuracy(1))
    if name == "always":
        return max(range(2), key=lambda index: (abs(reliabilities[index] - 0.5), -index))
    if name == "noisy_tv":
        return min(range(2), key=lambda index: (abs(reliabilities[index] - 0.5), index))
    if name in ("voi_adaptive", "voi_frozen", "oracle_voi"):
        options = [expected_inspection_value(p, ep.stakes, r)
                   for r in reliabilities]
        best = max(range(2), key=lambda index: options[index])
        if options[best] > expected_terminal_value(p, ep.stakes) + 1e-10:
            return best
        return None
    raise ValueError("unknown policy: " + name)


def act(ep: Episode, model: BayesianCalibration, inspected: int | None,
        oracle: WorldRules | None = None):
    if oracle:
        q = oracle.cue_accuracies[ep.difficulty]
        p = q if ep.cue else 1.0 - q
        if inspected is not None:
            p = assimilate(p, ep.signals[inspected], oracle.reliabilities[inspected])
    else:
        p = posterior(ep.cue, ep.difficulty, model)
        if inspected is not None:
            p = assimilate(p, ep.signals[inspected], model.sensor_accuracy(inspected))
    return int(p >= 0.5)


def evaluate(seed: int, *, unlearnable: bool = False, change: bool = False,
             train_n: int = TRAIN_N, test_n: int = TEST_N) -> dict:
    calibration = episodes(seed, train_n, unlearnable=unlearnable)
    test = episodes(seed + 1171, test_n, unlearnable=unlearnable,
                    change_at=test_n // 2 if change else None)
    # Source assignment MUST match between calibration and test:
    # use same source world rules (seed) but independent draws.
    # We generate evaluation traces with a per-episode independent RNG,
    # maintaining seed for the hidden physical sensor identity.
    test = episodes_with_independent_noise(
        seed, test_n, unlearnable=unlearnable,
        change_at=test_n // 2 if change else None)
    original = calibrated_model(seed, calibration)
    names = ("voi_adaptive", "voi_frozen", "never", "random",
             "always", "noisy_tv", "oracle_voi")
    outcomes = {}
    for name in names:
        # Each arm starts at identical learned beliefs. Replaying calibration
        # avoids copying mutable model state and makes the protocol auditable.
        model = calibrated_model(seed, calibration)
        rng = random.Random(seed * 1103 + 133 + names.index(name))
        reward, accurate, inspections, noises = [], [], [], []
        for t, ep in enumerate(test):
            rules = world_rules(seed, unlearnable=unlearnable,
                                swapped=change and t >= test_n // 2)
            oracle = rules if name == "oracle_voi" else None
            sensor = choose_probe(name, model, ep, rng, oracle=oracle)
            selected = act(ep, model, sensor, oracle=oracle)
            correct = int(selected == ep.latent)
            reward.append(ep.stakes * (2 * correct - 1) -
                          (PROBE_COST if sensor is not None else 0.0))
            accurate.append(correct)
            inspections.append(int(sensor is not None))
            noises.append(int(sensor is not None and
                              rules.reliabilities[sensor] == 0.5))
            # No test-time updates for the frozen and nonadaptive arms.
            if name == "voi_adaptive":
                model.observe(ep, sensor)
        split = test_n // 2
        def segment(lo, hi):
            return {
                "reward": statistics.mean(reward[lo:hi]),
                "correct": statistics.mean(accurate[lo:hi]),
                "inspection_rate": statistics.mean(inspections[lo:hi]),
                "noise_rate": statistics.mean(noises[lo:hi]),
            }
        outcomes[name] = {
            "all": segment(0, test_n),
            "first_half": segment(0, split),
            "second_half": segment(split, test_n),
            "after_first_50": segment(split, min(split + 50, test_n)),
            "after_last_100": segment(max(split, test_n - 100), test_n),
        }
    return {
        "seed": seed, "informative_source": world_rules(seed, unlearnable=unlearnable).informative_source,
        "condition": ("unlearnable" if unlearnable else
                      "switch" if change else "stable"),
        "calibration_reads_per_source": calibrated_model(seed, calibration).sensor_reads,
        "outcomes": outcomes,
    }


def episodes_with_independent_noise(seed: int, n: int, *,
                                    unlearnable: bool, change_at: int | None):
    """Keep latent mechanism fixed, change only RNG for heldout observations."""
    source = random.Random(seed * 90709 + 4000021 + (71 if unlearnable else 37))
    output = []
    for t in range(n):
        rule = world_rules(seed, unlearnable=unlearnable,
                           swapped=change_at is not None and t >= change_at)
        latent = source.randrange(2)
        difficulty = source.randrange(2)
        stakes = STAKE_VALUES[source.randrange(2)]
        cue = latent if source.random() < rule.cue_accuracies[difficulty] else 1 - latent
        signals = tuple(latent if source.random() < q else 1 - latent
                        for q in rule.reliabilities)
        output.append(Episode(latent, cue, difficulty, stakes, signals))
    return output


def bootstrap_differences(results: list[dict], challenger: str, baseline: str,
                          segment: str, iterations: int = 2000):
    delta = [r["outcomes"][challenger][segment]["reward"] -
             r["outcomes"][baseline][segment]["reward"] for r in results]
    rng = random.Random(119)
    means = sorted(statistics.mean(rng.choices(delta, k=len(delta)))
                   for _ in range(iterations))
    return {
        "difference": statistics.mean(delta),
        "cluster_bootstrap95": [means[49], means[1949]],
        "n_independent_seeds": len(delta),
    }


def summarize(results: list[dict], *, condition: str):
    policies = results[0]["outcomes"]
    summaries = {}
    for name in policies:
        summaries[name] = {
            segment: {
                metric: statistics.mean(r["outcomes"][name][segment][metric]
                                        for r in results)
                for metric in ("reward", "correct", "inspection_rate", "noise_rate")
            } for segment in ("all", "first_half", "second_half",
                              "after_first_50", "after_last_100")
        }
    paired = {}
    for baseline in ("never", "random", "always", "noisy_tv", "voi_frozen"):
        paired[baseline] = bootstrap_differences(
            results, "voi_adaptive", baseline,
            "all" if condition != "switch" else "second_half")
    return {"condition": condition, "seeds": len(results),
            "policies": summaries, "voi_adaptive_minus": paired}


def run(seeds: int = TRAIN_SEEDS, train_n: int = TRAIN_N, test_n: int = TEST_N):
    answer = {
        "design": "F7 learned value-of-information under finite sensing cost",
        "protocol": {"seeds": seeds, "train_episodes_per_seed": train_n,
                     "test_episodes_per_seed": test_n, "probe_cost": PROBE_COST,
                     "fixed_seed_groups": "different heldout draws, same world mechanism",
                     "primary": "stable: adaptive > never, random, always",
                     "test_shift": "switch: adaptive should learn new sensor > frozen",
                     "negative": "unlearnable sensor: no gain vs never"},
        "conditions": {},
        "per_seed": {},
        "limitations": (
            "Small binary task; source signal has known form, but source precision "
            "is learned from history. Bayesian-controller family and costs "
            "are preprogrammed. Feedback reveals latent after terminal decision. "
            "Test-time learning differs by policy, as documented. "
            "This is not intrinsic curiosity, consciousness, nor integrated V5.1."
        ),
    }
    for condition in ("stable", "switch", "unlearnable"):
        data = [evaluate(seed + 1 + (20000 if condition == "unlearnable" else 0),
                         unlearnable=condition == "unlearnable",
                         change=condition == "switch",
                         train_n=train_n, test_n=test_n)
                for seed in range(seeds)]
        answer["conditions"][condition] = summarize(data, condition=condition)
        answer["per_seed"][condition] = data
        print("CONDITION", condition, "REWARDS", {
            p: round(v["all"]["reward"], 4)
            for p, v in answer["conditions"][condition]["policies"].items()
        }, flush=True)
    return answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=TRAIN_SEEDS)
    p.add_argument("--train", type=int, default=TRAIN_N)
    p.add_argument("--test", type=int, default=TEST_N)
    p.add_argument("--output", type=Path,
                   default=Path("observatory_results/active_information_f7"))
    args = p.parse_args()
    if not (4 <= args.seeds <= 100 and 40 <= args.train <= 5000
            and 100 <= args.test <= 5000 and args.test % 2 == 0):
        raise ValueError("invalid experiment dimensions")
    report = run(args.seeds, args.train, args.test)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print("WROTE", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
