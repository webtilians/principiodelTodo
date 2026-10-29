"""F9-I bounded event retention; selector receives observations only."""
import collections
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from observatory.selective_memory import Event, FIFO, RandomReservoir, Selector
from observatory.f9g_context_memory import ContextF4, DIM, CAPACITY, FIELDS


def unpack(payload):
    kind = payload % 2
    payload //= 2
    color = payload % 6
    payload //= 6
    y = payload % 7
    payload //= 7
    x = payload % 7
    return payload // 7, x, y, color, kind


def context(payload):
    _, x, y, color, _ = unpack(payload)
    return x, y, color


class ContextDiversity(Selector):
    """Preassigned rule: discard newest duplicate context; else oldest.
    
    Identity (key versus ball) is NOT a relevance feature. No persistent
    side cache: priority is computed from at most capacity+1 candidates.
    This is a hand-specified contextual rule, NOT LearnedRelevance.
    """
    def consider(self, store, event):
        if len(store.order) < self.capacity:
            store.put(event)
            return
        candidates = [Event(k, store.saved[k]) for k in store.order] + [event]
        counts = collections.Counter(context(e.payload) for e in candidates)
        duplicated = [e for e in candidates if counts[context(e.payload)] > 1]
        # Prefer context coverage, retaining earliest evidence within a context.
        victim = max(duplicated, key=lambda e: e.key) if duplicated else candidates[0]
        if victim.key != event.key:
            store.put(event, victim.key)


class SelectiveF4(ContextF4):
    def __init__(self, mode="fifo", capacity=64, seed=0):
        if mode not in ("fifo", "context", "random", "zero"):
            raise ValueError(mode)
        if not 1 <= capacity <= CAPACITY:
            raise ValueError(capacity)
        self.mode, self.capacity = mode, capacity
        cls = {"fifo": FIFO, "context": ContextDiversity,
               "random": RandomReservoir, "zero": FIFO}[mode]
        self.selector = cls(capacity, seed)
        self.reset()

    def reset(self):
        super().reset()
        self.new_events = []

    def readout(self):
        # Always 660 inputs, including when the physical store has 8 slots.
        small = super().readout()
        return np.pad(small, (0, (CAPACITY-self.capacity)*FIELDS))

    def encode(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (148,) or not np.isfinite(raw).all():
            raise ValueError("Only public observations accepted")
        self.tick += 1
        self.new_events = []
        if self.mode != "zero":
            frame = raw[:147].reshape(7, 7, 3)
            for x in range(7):
                for y in range(7):
                    obj = int(np.rint(frame[x,y,0]*max(OBJECT_TO_IDX.values())))
                    color = int(np.rint(frame[x,y,1]*max(COLOR_TO_IDX.values())))
                    if obj in (OBJECT_TO_IDX["key"],OBJECT_TO_IDX["ball"]) and 0 <= color < 6:
                        kind = int(obj == OBJECT_TO_IDX["ball"])
                        payload = ((((self.tick*7+x)*7+y)*6+color)*2+kind)
                        event = Event(self.serial,payload)
                        self.selector.consider(self.store,event)
                        self.new_events.append(event)
                        self.serial += 1
        assert len(self.store.order) <= self.capacity
        return np.concatenate((raw,self.readout()))


def corrupt_one(observation, memory, event_id):
    """Observer-only intervention: exactly one stored event's identity channels.
    
    No reinsertion if evicted, no changes to other slots, payload, age, order,
    raw pixels, actor state or selector. Called after cue leaves view.
    """
    out = observation.copy()
    if event_id in memory.store.saved:
        offset = 148 + FIELDS*memory.store.order.index(event_id)
        out[offset:offset+2] = out[offset:offset+2][::-1]
    return out


class SelectiveObservation(gym.ObservationWrapper):
    def __init__(self, env, mode="fifo", capacity=64, seed=0):
        super().__init__(env)
        self.writer = SelectiveF4(mode,capacity,seed)
        self.observation_space = spaces.Box(0,1,shape=(DIM,),dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.writer.reset()
        return super().reset(seed=seed,options=options)

    def observation(self,obs):
        return self.writer.encode(obs)
