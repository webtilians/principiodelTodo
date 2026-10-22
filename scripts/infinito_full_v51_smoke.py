"""Smoke run of the *complete V5.1 trainer* with every original component loaded."""
import argparse
import contextlib
import io
import json
import random
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import torch
from scripts.infinito_replay_probes import load_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    opts = parser.parse_args()
    seed = 11
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    mod = load_version("v51")
    # Hidden dimension 64 retains all four causal modules; never claims original 512 run.
    args = argparse.Namespace(
        lr=0.001, seed=seed, batch_size=4, input_dim=257,
        hidden_dim=64, attention_heads=8, memory_slots=16,
        text_mode=False, input_text=None,
    )
    trainer = mod.InfinitoV51ConsciousnessBreakthrough(args)
    # Original method appends a metric key not initialized in constructor.
    trainer.metrics_history.setdefault("biological_plausibility", deque(maxlen=2000))
    # The historical trainer adds FactDecoder as optimizer param group AFTER
    # instantiating CosineAnnealingWarmRestarts. Sync the schedule base LR.
    if len(trainer.scheduler.base_lrs) != len(trainer.optimizer.param_groups):
        trainer.scheduler.base_lrs = [group["lr"] for group in trainer.optimizer.param_groups]
        trainer.scheduler._last_lr = list(trainer.scheduler.base_lrs)
    for i in range(1, opts.steps + 1):
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            metrics = trainer.train_step(i)
        print(json.dumps({
            "step": i,
            "metrics": {k: float(v) for k, v in metrics.items()
                        if isinstance(v, (int, float)) and np.isfinite(v)},
            "log_tail": buffer.getvalue()[-1000:],
            "hidden_state_norm": float(trainer.model.visual_module.h_state.norm()),
            "memory_norm": float(trainer.model.memory.memory.norm())
        }), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
