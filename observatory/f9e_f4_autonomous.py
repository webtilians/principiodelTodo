"""F9-E: F4 Store/FIFO versus matched recurrent agents in native MiniGrid.

Exploratory only. Seven native actions, public partial observations; teacher
uses privileged route labels TRAIN only. Never opens heldout. F4 here is real
FIFO/Store, not LearnedRelevance or F5.
"""
from __future__ import annotations
import argparse
import collections
import json
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

from observatory.f9_external_minigrid import ensure_dependencies, file_sha
from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9d_autonomous_navigation import (
    TrainingWorlds, checked_seed, cue_visible, demonstration, actor_action,
)
from observatory.selective_memory import Event, FIFO

SLOTS = 3
MODES = ('raw', 'zero', 'f4')


class VisibleF4:
    """Retain seen object/color identities, in first-seen FIFO slot order.

    Does not inspect cue position, success target, world geometry or labels.
    One can counterfactually erase or swap identity in READOUT alone.
    """
    def __init__(self, mode='f4'):
        if mode not in ('f4', 'zero', 'erase', 'swap_identity'):
            raise ValueError(mode)
        self.mode = mode
        self.selector = FIFO(SLOTS, seed=0)
        self.ncolors = len(COLOR_TO_IDX)
        self.objmax = max(OBJECT_TO_IDX.values())
        self.colmax = max(COLOR_TO_IDX.values())
        self.key = OBJECT_TO_IDX['key']
        self.ball = OBJECT_TO_IDX['ball']
        self.reset()

    def reset(self):
        self.store = self.selector.begin()

    def readout(self):
        slots = np.zeros((SLOTS, 4), dtype=np.float32)
        if self.mode not in ('zero', 'erase'):
            for index, identity in enumerate(self.store.order[:SLOTS]):
                kind, color = divmod(identity, self.ncolors)
                if self.mode == 'swap_identity':
                    kind = self.ball if kind == self.key else self.key
                slots[index] = (float(kind == self.key), float(kind == self.ball),
                                color / self.colmax, 1.)
        return slots.reshape(-1)

    def encode(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (148,):
            raise AssertionError('F4 reads only original 148-dimensional public observation')
        if self.mode != 'zero':
            for pixel in raw[:147].reshape(-1, 3):
                obj = int(np.rint(pixel[0] * self.objmax))
                col = int(np.rint(pixel[1] * self.colmax))
                if obj in (self.key, self.ball) and 0 <= col < self.ncolors:
                    self.selector.consider(self.store, Event(obj*self.ncolors+col, 0))
        return np.concatenate([raw, self.readout()]).astype(np.float32)


class MemoryObservation(gym.ObservationWrapper):
    def __init__(self, env, mode):
        super().__init__(env)
        self.writer = VisibleF4(mode)
        self.observation_space = spaces.Box(0., 1., shape=(160,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.writer.reset()
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        return self.writer.encode(obs)


def augment_dataset(dataset, mode):
    if mode == 'raw':
        return dataset
    converted = []
    for observations, actions, seen in dataset:
        writer = VisibleF4(mode)
        converted.append((np.stack([writer.encode(o) for o in observations]),
                          actions, seen))
    return converted


def make_model(mode, index, worlds):
    from sb3_contrib import RecurrentPPO
    env = TrainingWorlds(index, worlds)
    if mode != 'raw':
        env = MemoryObservation(env, mode)
    return RecurrentPPO('MlpLstmPolicy', env=env,
                        seed=checked_seed('train', index, 0), device='cpu',
                        n_steps=128, batch_size=128, n_epochs=4,
                        learning_rate=3e-4, gamma=.99, gae_lambda=.95,
                        ent_coef=.01, verbose=0,
                        policy_kwargs={'lstm_hidden_size': 64,
                                       'net_arch': dict(pi=[64], vf=[64])})


def clone(model, dataset, epochs, index):
    """F9-D full-sequence BC, adapted for matched 160-wide F4 and zero models."""
    policy = model.policy
    rng = np.random.default_rng(checked_seed('train', index, 0))
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    dim = model.observation_space.shape[0]
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(dataset))
        losses = []
        for first in range(0, len(dataset), 64):
            batch = [dataset[j] for j in order[first:first+64]]
            n, length = len(batch), max(len(obs) for obs, _, _ in batch)
            xs = np.zeros((n, length, dim), np.float32)
            ys = np.zeros((n, length), np.int64)
            mask = np.zeros((n, length), np.float32)
            for i, (obs, actions, _) in enumerate(batch):
                if obs.shape[1] != dim:
                    raise AssertionError('Incorrect BC feature dimension')
                xs[i, :len(obs)] = obs
                ys[i, :len(actions)] = actions
                mask[i, :len(actions)] = 1.
            distribution, _ = policy.get_distribution(
                torch.tensor(xs.reshape(-1, dim)),
                (torch.zeros(1,n,64), torch.zeros(1,n,64)),
                torch.zeros(n*length))
            logp = distribution.log_prob(torch.tensor(ys.reshape(-1)))
            weights = torch.tensor(mask.reshape(-1))
            loss = -(logp*weights).sum()/weights.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), .5)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if epoch == 0 or (epoch+1) % 10 == 0 or epoch == epochs-1:
            print('F9E_BC', index, dim, epoch+1, round(history[-1],5), flush=True)
    return history


def rollout(model, mode, size, seed, reverse=False, control='normal'):
    if mode not in MODES or control not in ('normal','mask_cue','erase','swap_identity'):
        raise ValueError((mode,control))
    if control in ('erase','swap_identity') and mode != 'f4':
        raise ValueError('Intervention needs a trained F4 checkpoint')
    env = build_env(size)
    shadow = build_env(size) if control == 'mask_cue' else None
    memory_mode = {'erase':'erase', 'swap_identity':'swap_identity'}.get(control,mode)
    writer = None if mode == 'raw' else VisibleF4(memory_mode)
    state, first, trace, total = None, True, [], 0.
    try:
        raw, _ = env.reset(seed=seed)
        if reverse:
            raw, _ = swap_cue(env, get_world_labels(env))
        labels = get_world_labels(env)
        if shadow is not None:
            shadow.reset(seed=seed)
            if not reverse:
                swap_cue(shadow, get_world_labels(shadow))

        def actor_obs(raw):
            current = raw.copy()
            if shadow is not None:
                alt = shadow.observation(shadow.unwrapped.gen_obs())
                mask = np.any(current[:147].reshape(7,7,3) !=
                              alt[:147].reshape(7,7,3), axis=-1)
                current[:147].reshape(7,7,3)[mask] = 0.
            return current if writer is None else writer.encode(current)

        observation = actor_obs(raw)
        while True:
            seen = cue_visible(env)  # observer probes RAW env, never writes F4
            before = tuple(map(int, env.unwrapped.agent_pos))
            action, state = actor_action(model, observation, state, first, True)
            if not 0 <= action < 7:
                raise AssertionError('Non-native action')
            raw, reward, done, truncated, _ = env.step(action)
            if shadow is not None:
                shadow.step(action)
            total += float(reward)
            trace.append({'action':action,'position':list(before),
                          'cue_visible':bool(seen),
                          'memory_occupancy':0 if writer is None else len(writer.store.order),
                          'memory_readout':[] if writer is None else writer.readout().tolist()})
            first = False
            if done or truncated:
                break
            observation = actor_obs(raw)
        end = tuple(map(int,env.unwrapped.agent_pos))
        if end == tuple(labels['success_position']):
            side = labels['correct_side']
        elif end == tuple(labels['failure_position']):
            side = 'lower' if labels['correct_side']=='upper' else 'upper'
        else:
            side = 'none'
        return {'seed':seed,'reverse':reverse,'cue':labels['cue'],
                'correct_side':labels['correct_side'],'side':side,
                'success':int(total>0),'native_return':total,
                'timeout':bool(truncated),'length':len(trace),
                'cue_seen':any(x['cue_visible'] for x in trace),
                'memory_written':any(x['memory_occupancy']>0 for x in trace),
                'trace':trace}
    finally:
        env.close()
        if shadow is not None:
            shadow.close()


def evaluate(model, mode, index, size, episodes, control, out):
    pairs=[]
    for j in range(episodes):
        seed=checked_seed('development', index, j)
        a=rollout(model,mode,size,seed,False,control)
        b=rollout(model,mode,size,seed,True,control)
        if a['correct_side'] == b['correct_side']:
            raise AssertionError('Invalid paired intervention')
        pairs.append([a,b])
        with out.open('a') as stream:
            stream.write(json.dumps({'mode':mode,'control':control,'size':size,
                                     'index':index,'pair':[a,b]})+'\n')
    flat=[item for pair in pairs for item in pair]
    return {
        'pairs':episodes,
        'original_success':statistics.mean(pair[0]['success'] for pair in pairs),
        'swapped_success':statistics.mean(pair[1]['success'] for pair in pairs),
        'combined_success':statistics.mean(x['success'] for x in flat),
        'both_correct':sum(a['success'] and b['success'] for a,b in pairs),
        'valid_side_flips':sum(a['side'] != b['side'] and
                               a['side']!='none' and b['side']!='none'
                               for a,b in pairs),
        'cue_seen_rate':statistics.mean(x['cue_seen'] for x in flat),
        'memory_written_rate':statistics.mean(x['memory_written'] for x in flat),
        'timeout_rate':statistics.mean(x['timeout'] for x in flat),
        'mean_length':statistics.mean(x['length'] for x in flat),
        'exit_counts':dict(collections.Counter(x['side'] for x in flat)),
    }


def run(args):
    torch.set_num_threads(1)
    deps = ensure_dependencies()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    trace_out=out/'trajectories.jsonl'
    if trace_out.exists():
        raise FileExistsError('Must choose a fresh experiment output folder')
    demonstrations=[demonstration(checked_seed('train',args.index,j),reverse=rev)
                    for j in range(args.worlds) for rev in (False,True)]
    result={'experiment':'F9-E genuine F4 Store/FIFO native navigation',
            'f4_scope':'FIFO Store Event; NOT LearnedRelevance or F5',
            'heldout_opened':False,'not_confirmatory':True,'shaping':False,
            'native_actions':7,'seed_index':args.index,
            'source_sha256':file_sha(Path(__file__)),'dependencies':deps,
            'config':{'worlds':args.worlds,'epochs':args.epochs,
                      'rl_steps_requested':args.timesteps,'evaluation_pairs':args.episodes},
            'models':{}}
    for mode in MODES:
        data=augment_dataset(demonstrations,mode)
        model=make_model(mode,args.index,args.worlds)
        metrics={'input_width':model.observation_space.shape[0],
                 'parameters':int(sum(p.numel() for p in model.policy.parameters()))}
        metrics['bc_loss']=clone(model,data,args.epochs,args.index)
        pre=out/f'{mode}_pre_rl.zip'
        model.save(pre)
        metrics['pre_rl_sha256']=file_sha(pre)
        metrics['before_rl']={'S7/normal':evaluate(model,mode,args.index,7,args.episodes,'normal',trace_out)}
        model.learn(total_timesteps=args.timesteps)
        metrics['rl_steps_actual']=int(model.num_timesteps)
        post=out/f'{mode}_post_rl.zip'
        model.save(post)
        metrics['post_rl_sha256']=file_sha(post)
        metrics['after_rl']={}
        for size in (7,13):
            controls=('normal','mask_cue','erase','swap_identity') if mode=='f4' else ('normal','mask_cue')
            for control in controls:
                stats=evaluate(model,mode,args.index,size,args.episodes,control,trace_out)
                metrics['after_rl'][f'S{size}/{control}']=stats
                print('F9E_RESULT',args.index,mode,'S'+str(size),control,
                      json.dumps(stats),flush=True)
        result['models'][mode]=metrics
        (out/'report.json').write_text(json.dumps(result,indent=2))
        model.get_env().close()
    print('F9E_FINISHED',args.index,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=int,choices=range(4),required=True)
    p.add_argument('--worlds',type=int,default=600)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--timesteps',type=int,default=20000)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not (1<=args.worlds<=600 and 1<=args.epochs<=40 and
            128<=args.timesteps<=20000 and 1<=args.episodes<=100):
        p.error('Development budget must be bounded at F9-D budget')
    run(args)

if __name__=='__main__':
    main()
