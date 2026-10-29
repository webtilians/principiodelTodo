"""F9-G observation-only contextual FIFO. No environment or label API."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from observatory.selective_memory import Event, FIFO

CAPACITY = 64  # Explicit extension from F9-E's three identity slots.
FIELDS = 8
DIM = 148 + CAPACITY * FIELDS


class ContextF4:
    """Each visible key/ball occurrence is a new Event, with packed integer payload.

    Keys are local occurrence serials; payload packs observation tick, image x/y,
    color and identity. Store/FIFO are unmodified. No persistent side cache.
    Readout: key, ball, color, ego-x, ego-y, relative age, order, occupied.
    """
    def __init__(self, mode="f4", capacity=CAPACITY):
        if mode not in ("f4", "zero", "erase", "swap"):
            raise ValueError(mode)
        if not 1 <= capacity <= CAPACITY:
            raise ValueError(capacity)
        self.mode, self.capacity = mode, capacity
        self.selector = FIFO(capacity, seed=0)
        self.reset()

    def reset(self):
        self.store = self.selector.begin()
        self.tick = -1
        self.serial = 0

    def readout(self):
        result = np.zeros((self.capacity, FIELDS), np.float32)
        if self.mode in ("zero", "erase"):
            return result.ravel()
        for rank, key in enumerate(self.store.order):
            packed = self.store.recall(key)
            kind = packed % 2
            packed //= 2
            color = packed % 6
            packed //= 6
            y = packed % 7
            packed //= 7
            x = packed % 7
            observed = packed // 7
            if self.mode == "swap":
                kind = 1 - kind
            age = self.tick - observed
            result[rank] = (1-kind, kind, color/5, x/6, y/6,
                            age/(age+1), rank/max(1, self.capacity-1), 1)
        return result.ravel()

    def encode(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (148,) or not np.isfinite(raw).all():
            raise ValueError("Only the public 148-dimensional observation is accepted")
        self.tick += 1
        if self.mode != "zero":
            frame = raw[:147].reshape(7, 7, 3)
            for x in range(7):
                for y in range(7):
                    obj = int(np.rint(frame[x, y, 0] * max(OBJECT_TO_IDX.values())))
                    color = int(np.rint(frame[x, y, 1] * max(COLOR_TO_IDX.values())))
                    if obj in (OBJECT_TO_IDX["key"], OBJECT_TO_IDX["ball"]) and 0 <= color < 6:
                        kind = int(obj == OBJECT_TO_IDX["ball"])
                        payload = ((((self.tick*7+x)*7+y)*6+color)*2+kind)
                        self.selector.consider(self.store, Event(self.serial, payload))
                        self.serial += 1
        return np.concatenate((raw, self.readout()))


class ContextObservation(gym.ObservationWrapper):
    def __init__(self, env, mode="f4"):
        super().__init__(env)
        self.writer = ContextF4(mode)
        self.observation_space = spaces.Box(0, 1, shape=(DIM,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.writer.reset()
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        return self.writer.encode(obs)
