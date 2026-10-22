"""F7b exploratory error-triggered *recalibration*, leaving F7 frozen.

The agent inspects when VOI is positive, then notices when actual probe
reliability persistently fails to match its pre-feedback expectations.
It may trigger an expensive fixed-length re-identification process.
This intentionally hand-built response to surprise is NOT spontaneous curiosity.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
from pathlib import Path

from observatory.active_information_f7 import (
    PROBE_COST, BayesianCalibration, act, calibrated_model, choose_probe,
    episodes, episodes_with_independent_noise, evaluate as original_evaluate,
    bootstrap_differences, world_rules, public_view,
)

WINDOW = 24
MIN_OBSERVATIONS = 18
SURPRISE_THRESHOLD = 0.22
RECALIBRATION_PROBES = 40


def test_recalibration(seed: int, *, unlearnable=False, change=False,
                       train_n=240, test_n=600):
    train = episodes(seed, train_n, unlearnable=unlearnable)
    test = episodes_with_independent_noise(
        seed, test_n, unlearnable=unlearnable,
        change_at=test_n // 2 if change else None)
    model = calibrated_model(seed, train)
    rng = random.Random(seed * 1103 + 133)
    error_hist = [collections.deque(maxlen=WINDOW),
                  collections.deque(maxlen=WINDOW)]
    forced_left = 0
    next_forced_source = 0
    reward, correct, inspections, noise, forced, detections = [], [], [], [], [], []
    triggers = []
    for t, ep in enumerate(test):
        exploring = forced_left > 0
        if exploring:
            sensor = next_forced_source
            next_forced_source = 1 - next_forced_source
            forced_left -= 1
        else:
            sensor = choose_probe("voi_adaptive", model, public_view(ep), rng)
        expected = model.sensor_accuracy(sensor) if sensor is not None else None
        action = act(public_view(ep), model, sensor,
                     ep.signals[sensor] if sensor is not None else None)
        success = int(action == ep.latent)
        rule = world_rules(seed, unlearnable=unlearnable,
                           swapped=change and t >= test_n // 2)
        reward.append(ep.stakes * (2 * success - 1) -
                      (PROBE_COST if sensor is not None else 0))
        correct.append(success)
        inspections.append(int(sensor is not None))
        noise.append(int(sensor is not None and rule.reliabilities[sensor] == .5))
        forced.append(int(exploring))
        # The hidden z is exposed to the learner only here: after acting.
        model.observe(ep, sensor)
        new_detection = 0
        if sensor is not None and not exploring:
            observed_correct = int(ep.signals[sensor] == ep.latent)
            error_hist[sensor].append(expected - observed_correct)
            if (len(error_hist[sensor]) >= MIN_OBSERVATIONS and
                    statistics.mean(error_hist[sensor]) > SURPRISE_THRESHOLD):
                new_detection = 1
                triggers.append(t)
                # Forget stale evidence for both sensors, but leave free-cue
                # calibration intact. This is an explicit intervention.
                model.sensor_correct = [2.0, 2.0]
                model.sensor_wrong = [2.0, 2.0]
                model.sensor_reads = [0, 0]
                error_hist = [collections.deque(maxlen=WINDOW),
                              collections.deque(maxlen=WINDOW)]
                forced_left = RECALIBRATION_PROBES
                next_forced_source = 0
        detections.append(new_detection)
    split = test_n // 2

    def segment(start: int, end: int):
        def avg(xs):
            return statistics.mean(xs[start:end])
        return {
            "reward": avg(reward), "correct": avg(correct),
            "inspection_rate": avg(inspections), "noise_rate": avg(noise),
            "forced_rate": avg(forced), "detector_rate": avg(detections),
        }

    return {
        "seed": seed, "condition": "unlearnable" if unlearnable else "switch" if change else "stable",
        "detection_steps": triggers,
        "metrics": {
            "all": segment(0, test_n),
            "first_half": segment(0, split),
            "second_half": segment(split, test_n),
            "after_first_50": segment(split, split + 50),
            "after_last_100": segment(test_n - 100, test_n),
        },
    }


def paired_bootstrap(delta, iterations=2000, seed=344):
    rng = random.Random(seed)
    b = sorted(statistics.mean(rng.choices(delta, k=len(delta)))
               for _ in range(iterations))
    return {
        "mean_delta_reward": statistics.mean(delta),
        "cluster_bootstrap95": [b[49], b[1949]],
        "n_independent_seeds": len(delta)
    }


def run(seeds=24, train_n=240, test_n=600):
    report = {
        "experiment": "F7b post-F7 exploratory surprise-triggered sensor re-identification",
        "detector": {
            "window": WINDOW, "min_observations": MIN_OBSERVATIONS,
            "threshold_mean_expected_minus_observed": SURPRISE_THRESHOLD,
            "forced_probes": RECALIBRATION_PROBES,
        },
        "conditions": {}, "per_seed": {},
        "disclaimer": (
            "The change detector and forced exploration are engineered. "
            "They do not emerge spontaneously. F7b is exploratory, after F7 "
            "results were inspected. Baselines and cases are exactly F7."
        ),
    }
    for condition in ("stable", "switch", "unlearnable"):
        unlearnable = condition == "unlearnable"
        change = condition == "switch"
        seeds_for_case = [
            i + 1 + (20000 if unlearnable else 0) for i in range(seeds)]
        ours = [test_recalibration(seed, unlearnable=unlearnable, change=change,
                                   train_n=train_n, test_n=test_n)
                for seed in seeds_for_case]
        baselines = [original_evaluate(seed, unlearnable=unlearnable, change=change,
                                       train_n=train_n, test_n=test_n)
                     for seed in seeds_for_case]
        segments = ("all", "first_half", "second_half",
                    "after_first_50", "after_last_100")
        results = {
            "n_seeds": seeds,
            "mean_detector_triggers": statistics.mean(
                len(x["detection_steps"]) for x in ours),
            "mean_reward": {
                seg: statistics.mean(x["metrics"][seg]["reward"] for x in ours)
                for seg in segments
            },
            "mean_forced_rate": statistics.mean(
                x["metrics"]["all"]["forced_rate"] for x in ours),
            "comparison": {},
        }
        for seg in segments:
            results["comparison"][seg] = {}
            for baseline in ("voi_adaptive", "voi_frozen", "never", "always", "oracle_voi"):
                diffs = [
                    our["metrics"][seg]["reward"] -
                    ref["outcomes"][baseline][seg]["reward"]
                    for our, ref in zip(ours, baselines)
                ]
                results["comparison"][seg][baseline] = paired_bootstrap(diffs)
        report["conditions"][condition] = results
        report["per_seed"][condition] = ours
        print("CONDITION", condition,
              "adaptive-recalibration", round(results["mean_reward"]["all"], 5),
              "late", round(results["mean_reward"]["after_last_100"], 5),
              "triggers", round(results["mean_detector_triggers"], 3), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=24)
    p.add_argument("--train", type=int, default=240)
    p.add_argument("--test", type=int, default=600)
    p.add_argument("--output", type=Path,
                   default=Path("observatory_results/recalibration_f7b"))
    a = p.parse_args()
    if not (4 <= a.seeds <= 100 and 40 <= a.train <= 5000
            and 100 <= a.test <= 5000 and a.test % 2 == 0):
        raise ValueError("invalid dimensions")
    answer = run(a.seeds, a.train, a.test)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "report.json").write_text(
        json.dumps(answer, ensure_ascii=False, indent=2))
    print("REPORT", a.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
