"""New F10-G worlds. No previous experiment imports or confirmation API."""
import hashlib
import numpy as np

KEYS, EVENTS, FEATURES = 24, 32, 29
VARIANTS = ('U_before', 'U_after', 'S_before', 'S_after')
BASE = {'train': 18100000, 'dev': 18200000, 'ood': 18400000}


def seed_for(split, replica, j):
    if split not in BASE or replica not in range(4):
        raise ValueError('Only preregistered TRAIN/DEV/OOD')
    if not 0 <= j < (512 if split == 'train' else 128):
        raise ValueError('World outside preregistered range')
    return BASE[split] + 10000 * replica + j


def world(split, replica, j):
    seed = seed_for(split, replica, j)
    rng = np.random.default_rng(seed)
    k = (8, 12)[j % 2] if split == 'ood' else (3, 5)[j % 2]
    query = int(rng.integers(KEYS))
    inversion = int(rng.integers(2))
    keys = rng.choice([i for i in range(KEYS) if i != query], EVENTS)
    keys[:k] = query
    values = rng.integers(2, size=EVENTS)
    values[1] = 1 - values[0]
    signs = rng.integers(2, size=EVENTS)
    signs[:k] = 0
    signs[0] = 1
    times = rng.permutation(EVENTS).astype(np.float32) / EVENTS
    sources = rng.integers(2, size=EVENTS)
    permutation = rng.permutation(EVENTS)
    lo, hi = sorted(times[:2])
    cases = []
    for variant in VARIANTS:
        sig, moment = signs.copy(), times.copy()
        sig[1] = variant.startswith('S')
        moment[0], moment[1] = (hi, lo) if variant.endswith('before') else (lo, hi)
        memory = np.zeros((EVENTS, FEATURES), np.float32)
        memory[np.arange(EVENTS), keys] = 1
        memory[:, KEYS] = values * 2 - 1
        memory[:, KEYS+1] = sig
        memory[:, KEYS+2] = moment
        memory[np.arange(EVENTS), KEYS+3+sources] = 1
        memory = memory[permutation]
        valid = memory[:, query] > 0
        winner = int(np.argmax(np.where(valid & (memory[:, KEYS+1] > 0), memory[:, KEYS+2], -1)))
        history = np.zeros((EVENTS+1, FEATURES+1), np.float32)
        history[:EVENTS, :FEATURES] = memory
        history[-1, query] = 1
        history[-1, -1] = inversion
        truth = int(memory[winner, KEYS] > 0) ^ inversion
        cases.append(dict(seed=seed, variant=variant, k=k, history=history,
                          memory=memory, winner=winner, truth=truth,
                          hash=hashlib.sha256(history.tobytes()).hexdigest()))
    return cases


def dataset(split, replica):
    return [c for j in range(512 if split == 'train' else 128)
            for c in world(split, replica, j)]
