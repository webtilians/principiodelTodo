"""F8: predictive homeostasis plus learned, costly active sensing.

Controlled *model-based RL / POMDP* benchmark; not a claim of consciousness.
All agents receive the SAME offline training, physical world, noise stream,
and evaluation budget. Hidden regime and unpurchased sensor values remain
in the environment; the evaluator alone knows them.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from pathlib import Path

ACTIONS = ("esperar", "beber", "descansar", "buscar_oeste", "buscar_este")
N_ACT = len(ACTIONS)
HORIZON_PLAN = 2
SENSOR_COST = 0.045
PROBE_RELIABILITY = 0.94
CUE_RELIABILITY = 0.57
HAZARD = 0.035
POLICIES = ("voi", "never", "always", "random_probe",
            "voi_no_memory", "noisy_tv", "reflex", "oracle")


def clamp(x: float, low=0., high=1.0) -> float:
    return max(low, min(high, x))


def rules(seed: int):
    """Simulator-only episode physics; calibrated on exactly this seed."""
    r = random.Random(307 * seed + 29)
    return {
        "drink": r.uniform(.32, .39),
        "rest": r.uniform(.26, .35),
        "abundant": r.uniform(.44, .57),
        "scarce": r.uniform(.035, .085),
        "heat_drift": r.uniform(.033, .049),
        "good_sensor": r.randrange(2),
    }


def terminal_utility(state, action: int) -> float:
    thirst, fatigue, water, _ = state
    return (-2.5*thirst**2 - 1.7*fatigue**2 -
            .11*(1-water)**2 -
            (.032 if action in (3, 4) else .012 if action == 2
             else .009 if action == 1 else 0.0))


def environment_transition(state, action: int, mode: int, config: dict,
                           noise=(0., 0., False)):
    """The truth: hidden from world-model and policy. Noise is exogenous."""
    thirst, fatigue, water, heat = state
    if action == 1 and water >= .22:
        thirst -= config["drink"]
        water -= .25
    elif action == 2:
        fatigue -= config["rest"]
        thirst += .015
    elif action in (3, 4):
        matched = ((action == 3 and mode == 0) or (action == 4 and mode == 1))
        water += config["abundant"] if matched else config["scarce"]
        fatigue += .095
        thirst += .045
    thirst += .028 + heat*config["heat_drift"] + noise[0]
    fatigue += .029 + noise[1]
    return (clamp(thirst, 0, 1.35), clamp(fatigue, 0, 1.35),
            clamp(water), (1 - heat) if noise[2] else heat)


class LearnedDynamics:
    """Tabular-in-physical-context conditional mean transitions from data only.

    No true dynamics constants, privileged mode, or future test outcomes enter
    the planning model. The mode labels exist ONLY during random calibration.
    """
    def __init__(self):
        self.deltas = {}
        self.counts = {}

    @staticmethod
    def context(state):
        # Availability matters only to drinking: model cannot drink without water.
        return int(state[3]), int(state[2] >= .22)

    def fit(self, seed: int, n: int):
        rng = random.Random(seed*4241 + 107)
        config = rules(seed)
        buckets = {}
        for i in range(n):
            # Stratified calibration: every (mode, action, heat, availability)
            # has equal initial representation; no stochastic empty cells.
            combos = list(itertools.product(range(2),range(N_ACT),range(2),range(2)))
            mode, action, heat, available = combos[i % len(combos)]
            water = rng.uniform(.28, .43) if available else rng.uniform(.06, .18)
            state = (rng.uniform(.26, .57), rng.uniform(.26, .57), water, heat)
            noise = (rng.gauss(0, .007), rng.gauss(0, .007), False)
            after = environment_transition(state, action, mode, config, noise)
            key = (mode, action, heat, available)
            delta = tuple(after[j] - state[j] for j in range(3))
            buckets.setdefault(key, []).append(delta)
        for key, values in buckets.items():
            self.deltas[key] = tuple(statistics.mean(z[j] for z in values)
                                     for j in range(3))
            self.counts[key] = len(values)
        missing = [key for key in itertools.product(range(2), range(N_ACT),
                                                    range(2), range(2))
                   if key not in self.deltas]
        if missing:
            raise ValueError("insufficient calibration coverage: " + repr(missing[:4]))
        return self

    def predict(self, state, action: int, mode: int):
        heat, avail = self.context(state)
        delta = self.deltas[(mode, action, heat, avail)]
        return (*[clamp(state[i] + delta[i], 0, 1.35 if i < 2 else 1)
                  for i in range(3)], state[3])


class LearnedSensors:
    def __init__(self):
        self.cue_right = 2.
        self.cue_wrong = 2.
        self.right = [2., 2.]
        self.wrong = [2., 2.]

    def fit(self, seed: int, n: int, null=False):
        r = random.Random(seed*1033 + (911 if null else 17))
        good = rules(seed)["good_sensor"]
        for i in range(n):
            mode = r.randrange(2)
            cue = mode if r.random() < (.5 if null else CUE_RELIABILITY) else 1-mode
            self.cue_right += cue == mode
            self.cue_wrong += cue != mode
            sensor = i % 2
            accuracy = .5 if null or sensor != good else PROBE_RELIABILITY
            observed = mode if r.random() < accuracy else 1-mode
            self.right[sensor] += observed == mode
            self.wrong[sensor] += observed != mode
        return self

    @property
    def qcue(self):
        return self.cue_right/(self.cue_right + self.cue_wrong)

    def qsensor(self, sensor):
        return self.right[sensor]/(self.right[sensor]+self.wrong[sensor])


def bayes(p: float, signal: int, reliability: float):
    """P(mode==1) updated by a binary observation."""
    probability = reliability if signal else 1-reliability
    opposite = 1-reliability if signal else reliability
    denom = p*probability+(1-p)*opposite
    return clamp(p*probability/denom) if denom else p


def make_cases(seed: int, scenario: str, episodes: int, horizon: int):
    """Pre-sample hidden modes, cues, potential sensors, and transition noises.

    Policies get only observations they are allowed to see, never this object.
    No action affects the random number stream or when the world changes.
    """
    good = rules(seed)["good_sensor"]
    r = random.Random(seed*7919 + 19 + (97 if scenario == "null" else 0))
    cases = []
    for episode in range(episodes):
        start_mode = r.randrange(2)
        change_at = (r.randrange(horizon//3, 2*horizon//3)
                     if scenario == "switch" else horizon + 1)
        initial = (r.uniform(.16,.67), r.uniform(.15,.63),
                   r.uniform(.18,.66), r.randrange(2))
        stream = []
        for t in range(horizon):
            mode = start_mode if t < change_at else 1-start_mode
            qcue = .5 if scenario == "null" else CUE_RELIABILITY
            cue = mode if r.random() < qcue else 1-mode
            sensors = tuple(
                mode if r.random() < (PROBE_RELIABILITY if scenario != "null" and s == good else .5)
                else 1-mode for s in range(2))
            noises = (r.gauss(0,.008), r.gauss(0,.008), r.random() < .09)
            stream.append((mode, cue, sensors, noises))
        cases.append((initial, stream))
    return cases


def planned_values(state, model):
    """Mode-conditioned scores for SAME candidate future action sequence.

    A valid POMDP approximate rollout: at decision time unknown mode is
    integrated out; no policy may condition an unobserved future action on it.
    """
    scores = []
    for first in range(N_ACT):
        for second in range(N_ACT):
            mode_rewards = []
            for mode in (0,1):
                after = model.predict(state, first, mode)
                later = model.predict(after, second, mode)
                val = terminal_utility(after,first) + .93*terminal_utility(later,second)
                mode_rewards.append(val)
            scores.append((first, mode_rewards[0], mode_rewards[1]))
    return scores


def best_action(scores, p):
    action, value = None, -float("inf")
    for first, s0, s1 in scores:
        guess = (1-p)*s0 + p*s1
        if guess > value:
            action, value = first, guess
    return action, value


def choose_information(scores, prior, sensors: LearnedSensors):
    """Acquire a sensor only if prospective expected value exceeds its price."""
    _, no_info = best_action(scores, prior)
    candidates = []
    for sensor in (0, 1):
        quality = sensors.qsensor(sensor)
        expected = 0.
        for signal in (0,1):
            probability = ((prior*quality+(1-prior)*(1-quality))
                           if signal else
                           (prior*(1-quality)+(1-prior)*quality))
            posterior = bayes(prior, signal, quality)
            _, utility = best_action(scores, posterior)
            expected += probability*utility
        candidates.append(expected - no_info - SENSOR_COST)
    sensor = max(range(2), key=lambda i: candidates[i])
    return sensor if candidates[sensor] > 1e-10 else None


def reflex_action(state):
    thirst, fatigue, water, heat = state
    if thirst > .34 and water >= .22:
        return 1
    if water < .36:
        return 3  # designer-favoured west: robust baseline, not oracle
    if fatigue > .38:
        return 2
    return 0


def evaluate_policy(seed: int, scenario: str, name: str, cases,
                    model: LearnedDynamics, sensors: LearnedSensors):
    rewards, errors, used, worthless, thirsty, briers = [], [], 0, 0, 0, []
    counts = {key: 0 for key in ACTIONS}
    num_steps = 0
    rng = random.Random(seed*10009 + POLICIES.index(name)*857 + 39)
    truth_config = rules(seed)
    good = truth_config["good_sensor"]
    for state0, stream in cases:
        state = state0
        p = .5
        for mode, cue, potential_signals, noise in stream:
            num_steps += 1
            if name in ("voi_no_memory","reflex"):
                p = .5
            else:
                p = (1-HAZARD)*p + HAZARD*.5
            p = bayes(p, cue, sensors.qcue)
            scores = planned_values(state, model)
            probe = None
            if name == "oracle":
                p = float(mode)  # explicit simulator-only upper bound
            elif name == "voi":
                probe = choose_information(scores, p, sensors)
            elif name == "voi_no_memory":
                probe = choose_information(scores, p, sensors)
            elif name == "always":
                probe = max(range(2),key=lambda s:abs(sensors.qsensor(s)-.5))
            elif name == "random_probe":
                probe = rng.randrange(2) if rng.random() < .5 else None
            elif name == "noisy_tv":
                probe = min(range(2),key=lambda s:abs(sensors.qsensor(s)-.5))
            if probe is not None:
                p = bayes(p, potential_signals[probe], sensors.qsensor(probe))
                used += 1
                worthless += int(scenario == "null" or probe != good)
            briers.append((p-mode)**2)  # evaluator only; not training
            action = reflex_action(state) if name == "reflex" else best_action(scores,p)[0]
            counts[ACTIONS[action]] += 1
            before = state
            predicted = tuple((1-p)*model.predict(before,action,0)[i] +
                              p*model.predict(before,action,1)[i] for i in range(3))
            state = environment_transition(before,action,mode,truth_config,noise)
            reward = terminal_utility(state,action) - (SENSOR_COST if probe is not None else 0.)
            rewards.append(reward)
            thirsty += int(state[0] > .85)
            errors.append(sum((predicted[j]-state[j])**2 for j in range(3))/3)
            # Infer hidden regime via transition likelihood, without hidden mode.
            # Skip when action uninformative: only route yield carries evidence.
            if name != "voi_no_memory" and name != "oracle" and action in (3,4):
                x0 = model.predict(before,action,0)[2]
                x1 = model.predict(before,action,1)[2]
                observed = state[2]
                like0 = math.exp(-min(200.,(observed-x0)**2 / (2*.075**2)))
                like1 = math.exp(-min(200.,(observed-x1)**2 / (2*.075**2)))
                denom = p*like1 + (1-p)*like0
                if denom > 1e-80:
                    p = clamp(p*like1/denom)
        assert sum(counts.values()) == num_steps
    return {
        "return_mean": statistics.mean(rewards),
        "mean_prediction_mse": statistics.mean(errors),
        "mean_regime_brier": statistics.mean(briers),
        "probe_rate": used/num_steps,
        "noise_probe_rate": worthless/num_steps,
        "high_thirst_rate": thirsty/num_steps,
        "actions": counts,
    }


def uncertainty_ci(differences, seed=559, draws=1500):
    r = random.Random(seed)
    means = sorted(statistics.mean(r.choices(differences, k=len(differences)))
                   for _ in range(draws))
    lo=int(.025*draws);hi=min(draws-1,int(.975*draws))
    return [means[lo],means[hi]]


def run(seeds=12, train=1800, test_episodes=10, horizon=42, seed_offset=1000):
    if not 4<=seeds<=120 or not 300<=train<=15000 or not 3<=test_episodes<=40 or not 18<=horizon<=80:
        raise ValueError("invalid size")
    results = {"design": "F8 predictive control + active information with partially hidden physical regime",
               "protocol": {"seeds":seeds,"train_samples":train,
                            "test_episodes_per_scenario":test_episodes,
                            "horizon":horizon,"seed_offset":seed_offset,
                            "planning_horizon":HORIZON_PLAN,"probe_cost":SENSOR_COST,
                            "training": "offline randomly sampled labelled regimes; identical for all policies",
                            "controls":["stable","switch","null"]},
               "scenarios": {}, "per_seed": [],
               "limitations": ("Synthetic partially observable world; learned empirical transitions and "
                               "sensor reliabilities, but known two-regime hypothesis class and programmed "
                               "utility. F8 not original V5.1 or emergent subjective awareness. "
                               "F7b-specific unseen sensor reassignment is not tested here.") }
    data = {scenario: [] for scenario in ("stable","switch","null")}
    for k in range(seeds):
        seed=seed_offset+k*17
        model=LearnedDynamics().fit(seed,train)
        row={"seed":seed,"scenarios":{}}
        for scenario in data:
            learned_sensors=LearnedSensors().fit(seed,train//2,null=scenario=="null")
            scenarios=make_cases(seed,scenario,test_episodes,horizon)
            # Same hidden physics and source identity in calibration and test;
            # observation noise is independently generated by make_cases.
            policies={name:evaluate_policy(seed,scenario,name,scenarios,model,learned_sensors)
                      for name in POLICIES}
            row["scenarios"][scenario]=policies
            data[scenario].append(policies)
        results["per_seed"].append(row)
        print("SEED",seed,"VOI",{
            s:round(row["scenarios"][s]["voi"]["return_mean"],5) for s in data
        },flush=True)
    for scenario,runs in data.items():
        aggregate={name:{metric:statistics.mean(r[name][metric] for r in runs)
                         for metric in ("return_mean","mean_prediction_mse",
                                        "probe_rate","noise_probe_rate","high_thirst_rate")}
                   for name in POLICIES}
        comparisons={}
        for baseline in POLICIES:
            if baseline=="voi":continue
            delta=[r["voi"]["return_mean"]-r[baseline]["return_mean"] for r in runs]
            comparisons[baseline]={"mean":statistics.mean(delta),
                                   "bootstrap95":uncertainty_ci(delta),
                                   "independent_seeds":seeds}
        results["scenarios"][scenario]={"policies":aggregate,"voi_minus":comparisons}
        print("SCENARIO",scenario,{name:round(vals["return_mean"],5)
                                    for name,vals in aggregate.items()},flush=True)
    stable=results["scenarios"]["stable"]["voi_minus"]
    null=results["scenarios"]["null"]
    results["checks"]={
        "stable_better_than_never_and_random": all(stable[k]["bootstrap95"][0]>0
                                                    for k in ("never","random_probe")),
        "null_does_not_buy_uninformative_sensors":null["policies"]["voi"]["probe_rate"] < .05,
        "note":"Criteria are diagnostics, not automatic assertions of success; report failures."
    }
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seeds",type=int,default=12)
    p.add_argument("--train",type=int,default=1800)
    p.add_argument("--test-episodes",type=int,default=10)
    p.add_argument("--horizon",type=int,default=42)
    p.add_argument("--seed-offset",type=int,default=1000)
    p.add_argument("--output",type=Path,
                   default=Path("observatory_results/f8_active_world"))
    args=p.parse_args()
    report=run(args.seeds,args.train,args.test_episodes,args.horizon,args.seed_offset)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("CHECKS",json.dumps(report["checks"]),flush=True)
    print("REPORT",args.output/"report.json",flush=True)


if __name__=="__main__":
    main()
