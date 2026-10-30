"""F5: Does prospective risk-based retention improve prediction of consequences?

Inspired by EcoDispatch's immediate-cost plus future-coverage *potential*.
Self-contained synthetic world; never edits historical INFINITO or EcoDispatch.
Predictions use ONLY events actually retained, and weights fitted on past data.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_EVENTS = 12
N_CONTEXTS = 4
MEMORY_SLOTS = 3
POLICIES = ("future_lambda1", "future_lambda025", "lambda0_novelty",
            "global_blind", "fifo", "reservoir", "oracle_diagnostic",
            "full_diagnostic")


@dataclass(frozen=True)
class Dataset:
    contexts: np.ndarray
    features: np.ndarray
    order: np.ndarray
    targets: np.ndarray


def world_weights(seed: int) -> np.ndarray:
    """Causal drivers randomized per seed and independently across contexts."""
    rng = np.random.default_rng(seed * 137 + 17)
    weights = rng.choice((-1, 1), size=(N_CONTEXTS, N_EVENTS)).astype(float) * 0.035
    ids = rng.permutation(N_EVENTS)
    for c in range(N_CONTEXTS):
        main = ids[c * MEMORY_SLOTS:(c + 1) * MEMORY_SLOTS]
        weights[c, main] = np.asarray([1.75, -1.30, 0.85]) * rng.choice(
            (-1, 1), size=MEMORY_SLOTS)
    return weights


def make_data(rng: np.random.Generator, weights: np.ndarray, n: int,
              *, kind: str = "linear") -> Dataset:
    contexts = rng.integers(0, N_CONTEXTS, size=n)
    features = rng.normal(0, 1, size=(n, N_EVENTS))
    orders = np.stack([rng.permutation(N_EVENTS) for _ in range(n)])
    if kind == "null":
        noise_sd = math.sqrt(float(np.mean(np.sum(weights**2, axis=1))) + .35**2)
        targets = rng.normal(0, noise_sd, size=n)
    else:
        targets = np.einsum("ij,ij->i", weights[contexts], features) + rng.normal(
            0, .35, size=n)
        if kind == "nonlinear":
            # Nonlinear interaction is not in the learner's model class.
            targets = targets + .6 * features[:, 0] * features[:, 1]
    return Dataset(contexts, features, orders, targets)


def fit_predictor(training: Dataset, *, ridge: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Only PRE-experiment labeled training episodes are seen here."""
    learned = np.empty((N_CONTEXTS, N_EVENTS), dtype=float)
    for c in range(N_CONTEXTS):
        x = training.features[training.contexts == c]
        y = training.targets[training.contexts == c]
        if not len(x):
            raise ValueError("context has no calibration episodes")
        learned[c] = np.linalg.solve(x.T @ x + ridge * np.eye(N_EVENTS), x.T @ y)
    x = training.features
    pooled = np.linalg.solve(x.T @ x + ridge * np.eye(N_EVENTS),
                             x.T @ training.targets)
    return learned, pooled


class AdmissionController:
    """Actual online API: decide using ONE observed event at a time.

    The controller cannot see future events, targets, full episode tensors, or
    the selection of true drivers (except the explicitly labeled oracle).
    """
    def __init__(self, policy: str, context: int, learned: np.ndarray,
                 pooled: np.ndarray, rng: np.random.Generator,
                 *, truth: np.ndarray | None = None):
        self.policy = policy
        self.rng = rng
        self.store: dict[int, float] = {}
        self.seen = 0
        if policy == "oracle_diagnostic":
            if truth is None:
                raise ValueError("oracle requires simulator-only coefficients")
            self.coefficients, self.lam = truth[context], 1.0
        elif policy == "global_blind":
            self.coefficients, self.lam = pooled, 1.0
        elif policy == "future_lambda1":
            self.coefficients, self.lam = learned[context], 1.0
        elif policy == "future_lambda025":
            self.coefficients, self.lam = learned[context], 0.25
        elif policy == "lambda0_novelty":
            self.coefficients, self.lam = learned[context], 0.0
        elif policy in ("fifo", "reservoir"):
            self.coefficients, self.lam = np.zeros(N_EVENTS), 0.0
        else:
            raise ValueError(policy)

    def _priority(self, key: int, value: float) -> float:
        # EcoDispatch-like immediate novelty + expected loss reduction.
        return 0.04 * abs(value) + self.lam * (
            float(self.coefficients[key]) * value) ** 2

    def observe(self, key: int, value: float) -> None:
        """Only (key, value), past training and current memory are accessible."""
        self.seen += 1
        if self.policy == "fifo":
            if len(self.store) == MEMORY_SLOTS:
                del self.store[next(iter(self.store))]
            self.store[key] = value
            return
        if self.policy == "reservoir":
            if len(self.store) < MEMORY_SLOTS:
                self.store[key] = value
            elif int(self.rng.integers(self.seen)) < MEMORY_SLOTS:
                remove = int(self.rng.choice(tuple(self.store)))
                del self.store[remove]
                self.store[key] = value
            return
        if len(self.store) < MEMORY_SLOTS:
            self.store[key] = value
        else:
            least = min(self.store,
                        key=lambda k: (self._priority(k, self.store[k]), k))
            if self._priority(key, value) > self._priority(least, self.store[least]):
                del self.store[least]
                self.store[key] = value
        assert len(self.store) <= MEMORY_SLOTS

    def remembered(self) -> dict[int, float]:
        return dict(self.store)


def select_memory(order: np.ndarray, x: np.ndarray, context: int,
                  weights: np.ndarray, global_weights: np.ndarray,
                  rng: np.random.Generator, policy: str,
                  *, true_weights: np.ndarray | None = None) -> tuple[int, ...]:
    controller = AdmissionController(policy, context, weights, global_weights,
                                     rng, truth=true_weights)
    for j in order:
        controller.observe(int(j), float(x[j]))
    assert len(controller.store) == MEMORY_SLOTS
    return tuple(controller.remembered())


def run_condition(test: Dataset, weights: np.ndarray, global_weights: np.ndarray,
                  truth: np.ndarray, *, seed: int) -> dict:
    """Same predictions, budget and test instances for each admissible policy."""
    stats = {p: {"sq": [], "abs": [], "sign": [], "retention": []} for p in POLICIES}
    for i, (c0, values, order, y0) in enumerate(
        zip(test.contexts, test.features, test.order, test.targets)
    ):
        context, target = int(c0), float(y0)
        for pi, policy in enumerate(POLICIES):
            if policy == "full_diagnostic":
                selected = tuple(range(N_EVENTS))
            else:
                rng = np.random.default_rng(seed * 1000003 + i * 37 + pi)
                selected = select_memory(order, values, context, weights,
                                         global_weights, rng, policy,
                                         true_weights=truth)
            # The representation is reconstructed ONLY from retained memories.
            remembered = np.zeros(N_EVENTS, dtype=float)
            for j in selected:
                remembered[j] = values[j]
            prediction = float(np.dot(weights[context], remembered))
            error = prediction - target
            s = stats[policy]
            s["sq"].append(error**2)
            s["abs"].append(abs(error))
            s["sign"].append(float((prediction >= 0) == (target >= 0)))
            all_energy = float(np.sum((truth[context] * values)**2)) + 1e-12
            keep_energy = float(np.sum(
                (truth[context, list(selected)] * remembered[list(selected)])**2))
            s["retention"].append(keep_energy / all_energy)
    return {name: {
        "mse": float(np.mean(v["sq"])),
        "mae": float(np.mean(v["abs"])),
        "sign_accuracy": float(np.mean(v["sign"])),
        "true_weighted_coverage": float(np.mean(v["retention"])),
    } for name, v in stats.items()}


def evaluate_one(seed: int, train_n: int, test_n: int) -> dict:
    true_weights = world_weights(seed)
    train_rng = np.random.default_rng(seed * 1217 + 1)
    training = make_data(train_rng, true_weights, train_n)
    learned, pooled = fit_predictor(training)
    # Real negative control: train its predictor ALSO on independent noise-only
    # history, rather than training on signal and testing on a random regime.
    null_training = make_data(np.random.default_rng(seed * 1217 + 100001),
                              true_weights, train_n, kind="null")
    null_learned, null_pooled = fit_predictor(null_training)
    scenarios = {}
    for scenario_index, scenario in enumerate(
        ("stable", "shift", "null", "nonlinear")
    ):
        if scenario == "shift":
            scenario_truth = np.roll(true_weights, shift=1, axis=0)
        else:
            scenario_truth = true_weights
        test_rng = np.random.default_rng(
            seed * 314159 + 400 + scenario_index * 1033)
        kind = ("null" if scenario == "null" else
                ("nonlinear" if scenario == "nonlinear" else "linear"))
        test = make_data(test_rng, scenario_truth, test_n, kind=kind)
        model_w, pooled_w = ((null_learned, null_pooled) if scenario == "null"
                             else (learned, pooled))
        scenarios[scenario] = run_condition(
            test, model_w, pooled_w, scenario_truth,
            seed=seed + scenario_index * 4001)
    return {
        "seed": seed,
        "scenarios": scenarios,
        "estimation_relative_error": float(
            np.linalg.norm(learned - true_weights) / np.linalg.norm(true_weights)),
    }


def paired_bootstrap(seed_reports: list[dict], scenario: str,
                     challenger: str, baseline: str,
                     n: int = 1500) -> dict:
    differences = np.array([
        report["scenarios"][scenario][challenger]["mse"]
        - report["scenarios"][scenario][baseline]["mse"]
        for report in seed_reports
    ])
    rng = np.random.default_rng(712)
    samples = rng.choice(differences, size=(n, len(differences)),
                         replace=True).mean(axis=1)
    return {
        "delta_mse": float(np.mean(differences)),
        "bootstrap_95pct": [float(q) for q in np.quantile(samples, [.025, .975])],
        "independent_seed_count": len(differences),
        "note": "Exploratory paired seed bootstrap; not real-world guarantee",
    }


def summarize(reports: list[dict]) -> dict:
    result = {}
    for scen in ("stable", "shift", "null", "nonlinear"):
        summary = {p: {
            metric: float(np.mean([
                r["scenarios"][scen][p][metric] for r in reports
            ]))
            for metric in ("mse", "mae", "sign_accuracy",
                           "true_weighted_coverage")
        } for p in POLICIES}
        comparisons = {
            name: paired_bootstrap(reports, scen, "future_lambda1", name)
            for name in ("fifo", "reservoir", "lambda0_novelty",
                         "global_blind", "future_lambda025")
        }
        result[scen] = {
            "policies": summary, "comparisons_vs_lambda1": comparisons,
        }
    result["predeclared_success"] = {
        "stable_mse_better_than_fifo_lambda0_and_global": all(
            result["stable"]["comparisons_vs_lambda1"][name]
            ["bootstrap_95pct"][1] < 0
            for name in ("fifo", "lambda0_novelty", "global_blind")
        ),
        "no_positive_gain_in_null_vs_fifo": (
            result["null"]["comparisons_vs_lambda1"]["fifo"]
            ["bootstrap_95pct"][1] >= 0
        ),
    }
    return result


def run(seeds: int = 20, train_n: int = 960, test_n: int = 240) -> dict:
    if seeds < 4 or train_n < 96 or test_n < 24:
        raise ValueError("not enough independent seeds/train/test observations")
    reps = [evaluate_one(s + 100, train_n, test_n) for s in range(seeds)]
    return {
        "protocol": "preregistered F5 EcoDispatch-style prospective memory",
        "seeds": seeds, "train_per_seed": train_n,
        "test_per_scenario_per_seed": test_n,
        "memory_slots": MEMORY_SLOTS,
        "event_types": N_EVENTS, "contexts": N_CONTEXTS,
        "causal_model": (
            "linear contextual coefficients + observation noise; "
            "nonlinear misspecification test"
        ),
        "training_visibility": (
            "full past labeled episodes; only selected <=3 events at future inference"
        ),
        "conditions": summarize(reps),
        "per_seed": reps,
        "limitations": (
            "Synthetic world with contextual linear signal; future potential "
            "approximated by diagonal additive terms. Does not demonstrate "
            "autonomous motive, emergence, or consciousness."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--train", type=int, default=960)
    p.add_argument("--test", type=int, default=240)
    p.add_argument(
        "--output", type=Path,
        default=Path("observatory_results/f5_predictive_memory"))
    args = p.parse_args()
    report = run(args.seeds, args.train, args.test)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf8")
    for scenario in ("stable", "shift", "null", "nonlinear"):
        vals = report["conditions"][scenario]["policies"]
        print("SCENARIO", scenario, "MSE",
              json.dumps({k: round(v["mse"], 4) for k, v in vals.items()}),
              flush=True)
    print("SUCCESS_CRITERIA",
          json.dumps(report["conditions"]["predeclared_success"]), flush=True)


if __name__ == "__main__":
    main()
