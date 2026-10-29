"""F9-D: native seven-action navigation, behavior cloning then genuine PPO.

Teacher may inspect geometry/labels to generate TRAINING actions. Actor and
critic receive only the public partial image and compass. No teacher at test.
Development only; sealed holdout has no entry point.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from observatory.f9_external_minigrid import seed_for, file_sha, ensure_dependencies
from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue

BASE_SHA = "bb0508bea461f82c5ea067def25fad70fb40efe8"


def checked_seed(split, index, episode):
    # Restrict indices: historical seed_for permits overlapping ranges at >=10.
    if split not in ("train", "development") or not 0 <= index < 4:
        raise ValueError("Only four development experiments; no holdout access")
    return seed_for(split, index, episode)


class TrainingWorlds(gym.Wrapper):
    """Explicit finite train partition, paired cues; never random implicit resets."""
    def __init__(self, index, worlds=600):
        super().__init__(build_env(7))
        self.index, self.worlds, self.counter = index, worlds, 0

    def reset(self, *, seed=None, options=None):
        j = self.counter // 2 % self.worlds
        reverse = bool(self.counter % 2)
        self.counter += 1
        obs, _ = self.env.reset(seed=checked_seed("train", self.index, j))
        if reverse:
            obs, _ = swap_cue(self.env, get_world_labels(self.env))
        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        return obs, reward, terminated, truncated, {}


def cue_visible(env):
    """Observer-only visibility, verified by a same-pose image intervention."""
    base = env.unwrapped
    before = env.observation(base.gen_obs())
    labels = get_world_labels(env)
    changed, _ = swap_cue(env, labels)
    restored, _ = swap_cue(env, get_world_labels(env))
    assert np.array_equal(before, restored)
    return not np.array_equal(before, changed)


def route(env, target, direction=None):
    """Privileged BFS teacher. Returns native actions, never moves the camera."""
    base = env.unwrapped
    start = (*map(int, base.agent_pos), int(base.agent_dir))
    forbidden = {tuple(base.success_pos), tuple(base.failure_pos)} - {tuple(target)}
    queue = collections.deque([(start, [])])
    seen = {start}
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    while queue:
        (x, y, d), actions = queue.popleft()
        if (x, y) == tuple(target) and (direction is None or d == direction):
            return actions
        choices = [(0, (x, y, (d - 1) % 4)), (1, (x, y, (d + 1) % 4))]
        dx, dy = directions[d]
        pos = (x + dx, y + dy)
        cell = base.grid.get(*pos)
        if pos not in forbidden and (cell is None or cell.can_overlap()):
            choices.append((2, (*pos, d)))
        for action, state in choices:
            if state not in seen:
                seen.add(state)
                queue.append((state, actions + [action]))
    raise AssertionError("Teacher target unreachable")


def demonstration(seed, size=7, reverse=False):
    env = build_env(size)
    x, y, visibility = [], [], []
    try:
        obs, _ = env.reset(seed=seed)
        if reverse:
            obs, _ = swap_cue(env, get_world_labels(env))
        labels = get_world_labels(env)
        mid = size // 2
        # Both cue variants take identical actions until the fork.
        for target, direction in [((2, mid), 3),
                                  ((labels["success_position"][0], mid), 3),
                                  (tuple(labels["success_position"]), None)]:
            for action in route(env, target, direction):
                x.append(obs.copy())
                y.append(action)
                visibility.append(cue_visible(env))
                obs, reward, done, truncated, _ = env.step(action)
                if (done or truncated) and tuple(env.unwrapped.agent_pos) != tuple(labels["success_position"]):
                    raise AssertionError("Teacher terminated prematurely")
            if target == (2, mid):
                assert cue_visible(env), "Teacher must actually observe cue"
        assert done and reward > 0 and any(visibility)
        return np.array(x), np.array(y), visibility
    finally:
        env.close()


def make_model(recurrent, index, worlds):
    from stable_baselines3 import PPO
    from sb3_contrib import RecurrentPPO
    kwargs = dict(env=TrainingWorlds(index, worlds), seed=checked_seed("train", index, 0),
                  device="cpu", n_steps=128, batch_size=128, n_epochs=4,
                  learning_rate=3e-4, gamma=.99, gae_lambda=.95, ent_coef=.01)
    if recurrent:
        return RecurrentPPO("MlpLstmPolicy", **kwargs, policy_kwargs={
            "lstm_hidden_size": 64, "net_arch": dict(pi=[64], vf=[64])})
    return PPO("MlpPolicy", **kwargs, policy_kwargs={
        "net_arch": dict(pi=[64, 64], vf=[64, 64])})


def clone(model, recurrent, dataset, epochs, index):
    """Full sequences, fresh recurrent state per demonstration; padding masked."""
    policy = model.policy
    rng = np.random.default_rng(checked_seed("train", index, 0))
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    history = []
    for epoch in range(epochs):
        losses = []
        for start in range(0, len(dataset), 64):
            # Permute once per epoch below, not independently per batch.
            if start == 0:
                order = rng.permutation(len(dataset))
            batch = [dataset[j] for j in order[start:start + 64]]
            n, length = len(batch), max(len(x) for x, _, _ in batch)
            obs = np.zeros((n, length, 148), dtype=np.float32)
            acts = np.zeros((n, length), dtype=np.int64)
            mask = np.zeros((n, length), dtype=np.float32)
            for b, (x, y, _) in enumerate(batch):
                obs[b, :len(x)], acts[b, :len(y)], mask[b, :len(y)] = x, y, 1
            inputs = torch.tensor(obs.reshape(-1, 148))
            if recurrent:
                state = (torch.zeros(1, n, 64), torch.zeros(1, n, 64))
                # Initial states already zero; no mid-episode resets.
                dist, _ = policy.get_distribution(inputs, state, torch.zeros(n * length))
            else:
                dist = policy.get_distribution(inputs)
            logp = dist.log_prob(torch.tensor(acts.reshape(-1)))
            weights = torch.tensor(mask.reshape(-1))
            loss = -(logp * weights).sum() / weights.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print("BC", recurrent, index, epoch + 1, history[-1], flush=True)
    return history


def actor_action(model, obs, state, first, recurrent):
    """Deliberately tiny boundary: no env, info, reward, label or teacher input."""
    if recurrent:
        action, state = model.predict(obs.copy(), state=state,
                                     episode_start=np.array([first]), deterministic=True)
    else:
        action, state = model.predict(obs.copy(), deterministic=True)
    return int(action), state


def rollout(model, recurrent, size, seed, reverse=False, control="normal"):
    env = build_env(size)
    shadow = build_env(size)
    trace, state, total, first = [], None, 0., True
    rng = np.random.default_rng(seed)
    try:
        obs, _ = env.reset(seed=seed)
        shadow.reset(seed=seed)
        swap_cue(shadow, get_world_labels(shadow))
        if reverse:
            obs, _ = swap_cue(env, get_world_labels(env))
            shadow.reset(seed=seed)
        labels = get_world_labels(env)
        while True:
            visible = cue_visible(env)
            actor_obs = obs.copy()
            if control == "mask_cue":
                # Diagnostic observer intervention only. Mask the same cells in
                # both counterfactuals, using the same-pose alternate image.
                alt = shadow.observation(shadow.unwrapped.gen_obs())
                changed = np.any(actor_obs[:147].reshape(7, 7, 3) !=
                                 alt[:147].reshape(7, 7, 3), axis=-1)
                actor_obs[:147].reshape(7, 7, 3)[changed] = 0
            if model is None:
                action = int(rng.integers(7))
            else:
                action, state = actor_action(model, actor_obs, state,
                                            first or control == "reset_memory", recurrent)
            before = list(map(int, env.unwrapped.agent_pos))
            obs, reward, done, truncated, _ = env.step(action)
            # Identical action replay: shadow reward/termination never reaches actor.
            shadow.step(action)
            total += float(reward)
            trace.append(dict(action=action, position=before, cue_visible=visible,
                              observation_sha256=hashlib.sha256(actor_obs.tobytes()).hexdigest(),
                              reward=float(reward)))
            first = False
            if done or truncated:
                break
        end = tuple(env.unwrapped.agent_pos)
        side = ("upper" if end[1] < size // 2 else "lower") if end in (
            tuple(labels["success_position"]), tuple(labels["failure_position"])) else "none"
        return dict(seed=seed, reverse=reverse, cue=labels["cue"],
                    correct_side=labels["correct_side"], side=side, success=int(total > 0),
                    native_return=total, length=len(trace), timeout=bool(truncated),
                    cue_seen=any(t["cue_visible"] for t in trace), trace=trace)
    finally:
        env.close()
        shadow.close()


def evaluate(model, recurrent, index, episodes, out, name, controls=True):
    summaries = {}
    for size in (7, 13):
        for control in (["normal", "mask_cue"] + (["reset_memory"] if recurrent else [])
                        if controls else ["normal"]):
            pairs = []
            for j in range(episodes):
                seed = checked_seed("development", index, j)
                pair = [rollout(model, recurrent, size, seed, reverse, control)
                        for reverse in (False, True)]
                assert pair[0]["correct_side"] != pair[1]["correct_side"]
                pairs.append(pair)
            records = [r for pair in pairs for r in pair]
            summary = dict(episodes_per_condition=episodes,
                original_success=float(np.mean([p[0]["success"] for p in pairs])),
                swapped_success=float(np.mean([p[1]["success"] for p in pairs])),
                both_correct=sum(a["success"] and b["success"] for a, b in pairs),
                valid_side_flips=sum(a["side"] != b["side"] and
                                    a["side"] != "none" and b["side"] != "none" for a, b in pairs),
                cue_seen_rate=float(np.mean([r["cue_seen"] for r in records])),
                mean_native_return=float(np.mean([r["native_return"] for r in records])),
                timeout_rate=float(np.mean([r["timeout"] for r in records])),
                success_by_cue={cue: dict(successes=sum(r["success"] for r in records if r["cue"] == cue),
                                         episodes=sum(r["cue"] == cue for r in records))
                                for cue in ("key", "ball")})
            summaries[f"S{size}/{control}"] = summary
            with (out / "trajectories.jsonl").open("a") as f:
                for pair in pairs:
                    f.write(json.dumps(dict(model=name, size=size, control=control, pair=pair)) + "\n")
            print("NAV_RESULT", index, name, size, control, json.dumps(summary), flush=True)
    return summaries


def run(args):
    deps = ensure_dependencies()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "trajectories.jsonl").exists():
        raise ValueError("Use a fresh output directory")
    dataset = [demonstration(checked_seed("train", args.index, j), reverse=reverse)
               for j in range(args.worlds) for reverse in (False, True)]
    report = dict(experiment="F9-D autonomous native navigation", base_sha=BASE_SHA,
                  index=args.index, heldout_opened=False, shaping=False,
                  native_actions=7, phase="development", dependencies=deps,
                  config=vars(args) | {"output": str(out)}, models={},
                  teacher="Privileged geometry/target for training action labels only",
                  f4_status="Gated on four competent recurrent seeds; no F4 claim",
                  source_sha256=file_sha(Path(__file__)))
    report["models"]["random"] = evaluate(None, False, args.index, args.episodes, out, "random", False)
    for recurrent in (False, True):
        for bc in (False, True):
            name = ("lstm" if recurrent else "ppo") + ("_bc" if bc else "_scratch")
            model = make_model(recurrent, args.index, args.worlds)
            details = dict(parameters=sum(p.numel() for p in model.policy.parameters()))
            if bc:
                details["bc_loss"] = clone(model, recurrent, dataset, args.epochs, args.index)
                details["before_rl"] = evaluate(model, recurrent, args.index, args.episodes,
                                               out, name + "_before_rl")
            model.learn(total_timesteps=args.timesteps)
            details["actual_rl_steps"] = model.num_timesteps
            checkpoint = out / (name + ".zip")
            model.save(checkpoint)
            details["checkpoint_sha256"] = file_sha(checkpoint)
            details["after_rl"] = evaluate(model, recurrent, args.index, args.episodes, out, name)
            report["models"][name] = details
            model.get_env().close()
            (out / "report.json").write_text(json.dumps(report, indent=2))
    print("NAV_REPORT", json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, choices=range(4), required=True)
    p.add_argument("--worlds", type=int, default=600)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--timesteps", type=int, default=20000)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not (1 <= args.worlds <= 5000 and 1 <= args.epochs <= 100 and
            128 <= args.timesteps <= 300000 and 1 <= args.episodes <= 100):
        p.error("Bounded development budget required")
    run(args)


if __name__ == "__main__":
    main()
