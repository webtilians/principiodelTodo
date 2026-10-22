"""INFINITO F6: model-based homeostatic control in a hidden-dynamics world.

Agents know the utility of states but learn action consequences from random
training interactions. Inspired by EcoDispatch's future-coverage potential.
No biological feeling, consciousness, curiosity, or original V5.1 edit.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTIONS = ("wait", "drink", "rest", "forage")
ACTION_COUNT = len(ACTIONS)
FEATURES = 11


@dataclass(frozen=True)
class Parameters:
    drink_effect: float
    rest_effect: float
    forage_effect: float
    heat_drift: float
    fatigue_drift: float


def params_for_seed(seed: int, shift: bool = False) -> Parameters:
    gen = np.random.default_rng(seed * 3011 + 43)
    return Parameters(
        drink_effect=float(gen.uniform(.32, .42) * (.58 if shift else 1)),
        rest_effect=float(gen.uniform(.25, .35) * (.65 if shift else 1)),
        forage_effect=float(gen.uniform(.36, .48) * (.55 if shift else 1)),
        heat_drift=float(gen.uniform(.035, .055) * (1.5 if shift else 1)),
        fatigue_drift=float(gen.uniform(.024, .035)),
    )


class World:
    """State: thirst, fatigue, water, heat. Rules are never passed to an agent."""
    def __init__(self, seed: int, *, shift: bool = False, horizon: int = 48):
        self.rng = np.random.default_rng(seed)
        self.params = params_for_seed(seed // 1000, shift=shift)
        self.horizon = horizon
        self.t = 0
        self.state = np.array([
            self.rng.uniform(.16, .53), self.rng.uniform(.12, .52),
            self.rng.uniform(.28, .79), float(self.rng.integers(0, 2))
        ], dtype=float)

    def step(self, action: int):
        if action not in range(ACTION_COUNT):
            raise ValueError("invalid action")
        p = self.params
        thirst, fatigue, water, heat = self.state
        if action == 1:
            if water >= .21:
                thirst -= p.drink_effect
                water -= .26
            fatigue += .018
        elif action == 2:
            fatigue -= p.rest_effect
            thirst += .012
        elif action == 3:
            water += p.forage_effect
            fatigue += .14
            thirst += .055
        thirst += .025 + heat * p.heat_drift + self.rng.normal(0, .008)
        fatigue += p.fatigue_drift + self.rng.normal(0, .008)
        next_heat = heat if self.rng.random() > .11 else 1. - heat
        self.state = np.array([
            np.clip(thirst, 0, 1.35), np.clip(fatigue, 0, 1.35),
            np.clip(water, 0, 1), next_heat
        ], dtype=float)
        self.t += 1
        return self.state.copy(), float(utility(self.state[:3], np.asarray(action))), self.t >= self.horizon


def utility(next_three, actions):
    """Explicitly programmed homeostatic reward; no phenomenal or intrinsic needs."""
    x, a = np.asarray(next_three), np.asarray(actions)
    return (-2.5 * x[..., 0]**2 - 1.7 * x[..., 1]**2
            - .08 * (1 - x[..., 2])**2 - .035 * (a == 3)
            - .012 * (a == 2) - .009 * (a == 1))


def basis(obs):
    x = np.asarray(obs, dtype=float)
    d, f, w, h = (x[..., i] for i in range(4))
    return np.stack((np.ones_like(d), d, f, w, h, d*d, f*f, w*w,
                     (w < .21).astype(float), d*h, f*h), axis=-1)


class LearnedWorldModel:
    """Ridge regression per action, on transitions, without future action labels."""
    def __init__(self, ridge: float = .04):
        self.gram = np.repeat((np.eye(FEATURES) * ridge)[None, ...], ACTION_COUNT, axis=0)
        self.cross = np.zeros((ACTION_COUNT, FEATURES, 3))
        self.weights = np.zeros((ACTION_COUNT, FEATURES, 3))
        self.counts = np.zeros(ACTION_COUNT, dtype=int)

    def fit(self, transitions):
        for obs, action, reward, next_obs in transitions:
            feat = basis(obs)
            self.gram[action] += np.outer(feat, feat)
            self.cross[action] += np.outer(feat, next_obs[:3])
            self.counts[action] += 1
        for a in range(ACTION_COUNT):
            self.weights[a] = np.linalg.solve(self.gram[a], self.cross[a])
        return self

    def predict(self, observations, action):
        x = np.asarray(observations, dtype=float)
        val = np.clip(basis(x) @ self.weights[action],
                      [0, 0, 0], [1.35, 1.35, 1])
        return np.concatenate((val, x[..., 3:4]), axis=-1)


class Planner:
    """Receding-horizon MPC enumerates 4^H action sequences at every decision."""
    def __init__(self, model, horizon=3, discount=.92, *, shuffled_model=False):
        self.model, self.horizon, self.discount = model, horizon, discount
        self.choices = np.asarray(list(itertools.product(range(ACTION_COUNT),
                                                     repeat=horizon)), dtype=int)
        self.mapping = [2, 3, 0, 1] if shuffled_model else list(range(ACTION_COUNT))

    def act(self, observation):
        choices = self.choices
        states = np.repeat(np.asarray(observation)[None, :], len(choices), axis=0)
        scores = np.zeros(len(choices), dtype=float)
        for t in range(self.horizon):
            acts = choices[:, t]
            for a in range(ACTION_COUNT):
                mask = acts == a
                after = self.model.predict(states[mask], self.mapping[a])
                states[mask] = after
                scores[mask] += self.discount**t * utility(
                    after[:, :3], np.full(mask.sum(), a))
        return int(choices[int(np.argmax(scores)), 0])


class TabularQ:
    """Model-free Q-learning baseline from the exact same random transitions."""
    def __init__(self, gamma=.93):
        self.gamma = gamma
        self.q = np.zeros((6, 6, 6, 2, ACTION_COUNT))

    def index(self, state):
        return tuple(int(min(5, max(0, np.floor(float(z) * 6 / hi))))
                     for z, hi in zip(state[:3], (1.35, 1.35, 1))) + (int(state[3]),)

    def fit(self, transitions, passes=8, alpha=.22, seed=11):
        rng = random.Random(seed)
        order = list(range(len(transitions)))
        for _ in range(passes):
            rng.shuffle(order)
            for i in order:
                obs, action, reward, next_obs = transitions[i]
                s, nxt = self.index(obs), self.index(next_obs)
                target = reward + self.gamma * self.q[nxt].max()
                self.q[s][action] += alpha * (target - self.q[s][action])
        return self

    def act(self, obs):
        return int(np.argmax(self.q[self.index(obs)]))


def collect_transitions(seed, episodes=110, horizon=48):
    transitions = []
    for episode in range(episodes):
        world = World(seed * 1000 + episode + 100, horizon=horizon)
        rng = random.Random(seed * 99991 + episode)
        for t in range(horizon):
            obs = world.state.copy()
            action = rng.randrange(ACTION_COUNT)
            next_obs, reward, done = world.step(action)
            transitions.append((obs, action, reward, next_obs))
            if done:
                break
    return transitions


def reflex(obs):
    """A strong human-coded policy, explicitly granted domain insight."""
    thirst, fatigue, water, _ = obs
    if thirst > .37 and water >= .21:
        return 1
    if fatigue > .39:
        return 2
    if water < .3:
        return 3
    return 0


def test_agents(seed, models, *, shift=False, episodes=12, horizon=48):
    policies = {
        "model_based_3": Planner(models["world"], horizon=3),
        "model_based_1": Planner(models["world"], horizon=1),
        "shuffled_dynamics_3": Planner(models["world"], horizon=3, shuffled_model=True),
        "model_free_Q": models["q"],
        "reflex_oracle_rule": None,
        "random": None,
    }
    totals = {name: [] for name in policies}
    thirst_fail = {name: [] for name in policies}
    for episode in range(episodes):
        env_seed = (seed + 50000) * 1000 + episode
        for name, policy in policies.items():
            world = World(env_seed, shift=shift, horizon=horizon)
            rng = random.Random(env_seed + 902)
            rewards, bad = [], 0
            for t in range(horizon):
                obs = world.state.copy()
                action = (rng.randrange(ACTION_COUNT) if name == "random"
                          else reflex(obs) if policy is None else policy.act(obs))
                nxt, reward, done = world.step(action)
                rewards.append(reward)
                bad += int(nxt[0] > .85)
            totals[name].append(float(np.mean(rewards)))
            thirst_fail[name].append(bad / horizon)
    return {
        "seed": seed, "shift": shift,
        "policies": {
            name: {
                "reward_by_episode": totals[name],
                "mean_reward": float(np.mean(totals[name])),
                "high_thirst_fraction": float(np.mean(thirst_fail[name])),
            } for name in policies
        }
    }


def model_validation(model, seed, samples=200):
    env = World((seed + 91991) * 1000 + 1)
    rng = random.Random(seed * 171 + 4)
    squared = []
    for _ in range(samples):
        obs = env.state.copy()
        action = rng.randrange(ACTION_COUNT)
        nxt, _, _ = env.step(action)
        predicted = model.predict(obs[None, :], action)[0]
        squared.extend((predicted[:3] - nxt[:3])**2)
    return float(np.sqrt(np.mean(squared)))


def seed_bootstrap(samples, iterations=2000, seed=839):
    x = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(x, size=(iterations, len(x)), replace=True), axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def summarize(results):
    names = tuple(results[0]["policies"])
    output = {"n_seeds": len(results), "policies": {}, "paired_vs_model_based_3": {}}
    for name in names:
        vals = [r["policies"][name]["mean_reward"] for r in results]
        output["policies"][name] = {
            "mean_reward_per_step": float(np.mean(vals)),
            "mean_cost_per_step": float(-np.mean(vals)),
            "high_thirst_fraction": float(np.mean([
                r["policies"][name]["high_thirst_fraction"] for r in results])),
        }
    for name in names:
        if name == "model_based_3":
            continue
        differences = [
            r["policies"]["model_based_3"]["mean_reward"]
            - r["policies"][name]["mean_reward"] for r in results
        ]
        output["paired_vs_model_based_3"][name] = {
            "mean_reward_gain": float(np.mean(differences)),
            "seed_cluster_bootstrap_95": seed_bootstrap(differences),
            "number_of_independent_units": len(results),
        }
    return output


def run(seeds=8, train_episodes=110, test_episodes=12, horizon=48):
    report = {
        "protocol": {
            "seeds": seeds, "train_episodes": train_episodes,
            "test_episodes": test_episodes, "horizon": horizon,
            "model_fit": "linear ridge per action on equal random-policy trajectories",
            "model_free_fit": "offline tabular Q trained on SAME transitions",
            "primary": "model_based_3 vs model_based_1 and shuffled_dynamics_3",
            "calibration_to_test": "different environment seeds; frozen agent models",
        },
        "per_seed": [],
        "limitations": (
            "Synthetic physiology and programmed reward; learned transition model, "
            "not subjective state, self-originated curiosity or awareness. "
            "Tabular Q is an exploratory baseline with fixed discretization."
        ),
    }
    in_dist, shifted = [], []
    for i in range(seeds):
        training = collect_transitions(i + 3, episodes=train_episodes, horizon=horizon)
        world = LearnedWorldModel().fit(training)
        q = TabularQ().fit(training, seed=i + 5)
        models = {"world": world, "q": q}
        calibration = model_validation(world, i + 30)
        normal = test_agents(i + 401, models, episodes=test_episodes, horizon=horizon)
        drift = test_agents(i + 401, models, shift=True,
                            episodes=test_episodes, horizon=horizon)
        in_dist.append(normal)
        shifted.append(drift)
        report["per_seed"].append({
            "seed": i, "prediction_rmse": calibration,
            "in_distribution": normal, "shifted_mechanisms": drift,
        })
        print("SEED", i, "rmse", round(calibration, 4),
              "mean_rewards", {
                  k: round(v["mean_reward"], 3)
                  for k, v in normal["policies"].items()}, flush=True)
    report["in_distribution"] = summarize(in_dist)
    report["shifted_mechanisms"] = summarize(shifted)
    report["model_prediction_rmse_mean"] = float(
        np.mean([x["prediction_rmse"] for x in report["per_seed"]]))
    return report


def main():
    p = argparse.ArgumentParser(description="F6 homeostasis: learned model plus planning")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--train-episodes", type=int, default=110)
    p.add_argument("--test-episodes", type=int, default=12)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--output", type=Path,
                   default=Path("observatory_results/homeostasis_f6"))
    a = p.parse_args()
    if not (2 <= a.seeds <= 60 and 10 <= a.train_episodes <= 400
            and 3 <= a.test_episodes <= 40 and 16 <= a.horizon <= 100):
        raise ValueError("parameter sizes out of range")
    report = run(a.seeds, a.train_episodes, a.test_episodes, a.horizon)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print("SUMMARY", json.dumps({
        key: report[key] for key in (
            "in_distribution", "shifted_mechanisms", "model_prediction_rmse_mean")
    }), flush=True)


if __name__ == "__main__":
    main()
