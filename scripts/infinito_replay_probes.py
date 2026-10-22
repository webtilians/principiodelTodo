"""Bounded historical INFINITO probes, *not* complete training reproductions.

The historical implementations are imported unchanged. V1 comes from the
exact first public release (2025-09-16); legacy scripts are in master/archive.
Every probe is run in a separate subprocess with a strict timeout.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ORIGINAL_V1 = "14860c32b32b470cc478dd1806059b9fd2c46ce2"
VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v51")
SEEDS = (11, 29, 47)
MODES = ("reference", "counterfactual")


def module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, BASE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_version(version):
    if version == "v1":
        old = subprocess.run(
            ["git", "show", ORIGINAL_V1 + ":infinito.py"],
            cwd=BASE, check=True, capture_output=True, text=True,
        ).stdout
        # The historic script runs a full experiment on import. Only remove
        # its unguarded CLI block; retain the original model definition.
        marker = "# Run Infinito"
        if marker not in old:
            raise RuntimeError("historical V1 marker missing")
        source = old.split(marker)[0]
        scope = {"__name__": "infinito_v1_historical", "__file__": "infinito.py"}
        exec(compile(source, "historical/infinito.py", "exec"), scope)
        return scope
    if version == "v2":
        sys.path.insert(0, str(BASE / "archive"))
        return module_from_path("archive/infinito_v2_advanced.py", "infinito_v2_historical")
    if version == "v3":
        return module_from_path("archive/infinito_v3_fixed.py", "infinito_v3_historical")
    if version == "v4":
        return module_from_path("src/infinito_grok.py", "infinito_v4_historical")
    if version == "v5":
        return module_from_path("src/infinito_v5_revolutionary.py", "infinito_v5_historical")
    if version == "v51":
        return module_from_path("src/infinito_v5_1_consciousness.py", "infinito_v51_historical")
    raise ValueError(version)


def as_float(x):
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().mean().item())
    return float(np.asarray(x).mean())


def inspect_v1(mod, seed, counterfactual):
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    obj = mod["PrincipioTodoRecursivo"](size=16, max_depth=24)
    initial = obj._input_bin()
    current = initial.clone()
    rows = []
    for step in range(24):
        phi = obj._one_recursion(current if counterfactual else initial)
        current = phi.detach()
        vals = phi.detach().cpu().numpy()
        log = obj.complexity_log[-1]
        rows.append({"step": step, "metrics": {
            "field_std": float(vals.std()),
            "field_mean": float(vals.mean()),
            "clusters_reported": float(log["clusters"]),
            "entropy_reported": float(log["entropy"]),
            "loss_reported": float(log["loss"]),
        }})
    return rows, "V1 model + optimizer, 24 original _one_recursion calls; counterfactual passes prior output"


def inspect_v2(mod, seed, counterfactual):
    import numpy as np
    import torch
    from types import SimpleNamespace
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Keep original model forward and simulator step without starting
    # the expensive constructor's long training loop.
    network = mod.AdvancedNN_Middleware(channels=8, grid_size=32).eval()
    rules = [torch.randn(3, 3) * 0.12 for _ in range(16)]
    initial = (torch.rand(1, 1, 32, 32) > 0.91).float()
    current = initial.clone()
    proxy = SimpleNamespace(last_consciousness_level=0.0)
    rows = []
    with torch.no_grad():
        for step in range(24):
            phi = mod.InfinitoV2._enhanced_sim_step(
                proxy, current if counterfactual else initial, rules, learning_rate=0.008)
            current = phi.clone()
            predicted_laws, score = network(phi)
            rows.append({"step": step, "metrics": {
                "field_std": as_float(phi.std()),
                "field_mean": as_float(phi.mean()),
                "untrained_score": as_float(score),
                "predicted_law_abs": as_float(predicted_laws.abs()),
            }})
    return rows, "V2 original dynamics kernel + forward untrained network; NOT complete V2 evolutionary simulation"


def inspect_v3(mod, seed, counterfactual):
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    model = mod.StabilizedConsciousnessNN(input_size=128, hidden_size=32).eval()
    history = torch.zeros(1, 32)
    rows = []
    with torch.no_grad():
        for step in range(24):
            # Same model and input for both conditions, fixed by seed;
            # probe whether hidden_state argument matters.
            x = torch.randn(1, 128) * 0.1 + math.sin(step / 7.0) * 0.2
            score, history_new = model(x, None if counterfactual else history)
            history = history_new.detach()
            rows.append({"step": step, "metrics": {
                "untrained_score": as_float(score),
                "hidden_std": as_float(history.std()),
            }})
    return rows, "V3 original NN forward; test prior-hidden ablation, no training"


def inspect_v4(mod, seed, counterfactual):
    import torch
    torch.manual_seed(seed)
    rows = []
    for step in range(24):
        # V4 model's fast Phi *approximation* explicitly depends on state
        # variance and mean absolute connection strength. Permuting the
        # connectivity is a negative control with the same magnitude.
        state = torch.randn(16)
        matrix = torch.randn(16, 16) * 0.2
        permuted = matrix[torch.randperm(16)][:, torch.randperm(16)]
        graph = permuted if counterfactual else matrix
        value = mod.calculate_phi_approximation(state, graph)
        rows.append({"step": step, "metrics": {
            "phi_proxy": as_float(value),
            "connectivity_abs_mean": as_float(graph.abs()),
        }})
    return rows, "V4 original phi approximation; permuted connectivity with identical mean absolute strength"


def measure_phi_calculator(mod, seed, counterfactual, version):
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    calc = (mod.PhiCalculatorV5(hidden_dim=16) if version == "v5"
            else mod.EnhancedPhiCalculatorV51(hidden_dim=16)).eval()
    rows = []
    with torch.no_grad():
        for step in range(24):
            states = [torch.randn(2, 5, 16) * 0.3 for _ in range(4)]
            weights = torch.softmax(torch.randn(2, 2, 5, 5), dim=-1)
            # Column permutation changes who attends to whom, while
            # preserving each row's sum and mean attention weight.
            permuted = weights[..., torch.randperm(5)]
            previous_numpy = np.random.get_state()
            previous_torch = torch.get_rng_state()
            if version == "v5":
                phi, info = calc(*states, permuted if counterfactual else weights)
            else:
                consciousness = torch.tensor([0.4, 0.7])
                phi, info = calc(
                    *states, permuted if counterfactual else weights, consciousness)
            np.random.set_state(previous_numpy)
            torch.set_rng_state(previous_torch)
            rows.append({"step": step, "metrics": {
                "phi_proxy": as_float(phi),
                "attention_mean": as_float(weights),
            }})
    return rows, ("%s original PhiCalculator only; attention-column permutation null; "
                  "NOT full end-to-end model" % version)


def inspect_v5(mod, seed, counterfactual):
    return measure_phi_calculator(mod, seed, counterfactual, "v5")


def inspect_v51(mod, seed, counterfactual):
    return measure_phi_calculator(mod, seed, counterfactual, "v51")


def probe(version, seed, mode):
    import numpy as np
    import torch
    os.environ["MPLBACKEND"] = "Agg"
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    mod = load_version(version)
    handler = {
        "v1": inspect_v1, "v2": inspect_v2, "v3": inspect_v3,
        "v4": inspect_v4, "v5": inspect_v5, "v51": inspect_v51,
    }[version]
    if version == "v1":
        return handler(mod, seed, mode == "counterfactual")
    return handler(mod, seed, mode == "counterfactual")


def run_all(output, timeout):
    results = []
    observations = []
    for version in VERSIONS:
        for seed in SEEDS:
            for mode in MODES:
                run_id = "%s_%s_s%d" % (version, mode, seed)
                command = [sys.executable, str(Path(__file__).resolve()),
                           "--single", version, str(seed), mode]
                try:
                    process = subprocess.run(
                        command, capture_output=True, text=True,
                        cwd=BASE, timeout=timeout,
                        env=dict(os.environ, SDL_AUDIODRIVER="dummy",
                                 SDL_VIDEODRIVER="dummy", MPLBACKEND="Agg"),
                    )
                    if process.returncode:
                        results.append({
                            "run_id": run_id, "status": "error",
                            "error": process.stderr[-1200:],
                            "stdout": process.stdout[-500:],
                        })
                        continue
                    payload = json.loads(process.stdout.strip().splitlines()[-1])
                    rows = payload["observations"]
                    observations.extend({
                        "run_id": run_id, "step": row["step"],
                        "metrics": row["metrics"],
                    } for row in rows)
                    results.append({
                        "run_id": run_id, "status": "ok",
                        "n_observations": len(rows), "scope": payload["scope"],
                    })
                except subprocess.TimeoutExpired:
                    results.append({"run_id": run_id, "status": "timeout"})
                except Exception as exc:
                    results.append({"run_id": run_id, "status": "error",
                                    "error": repr(exc)})
                print(run_id, results[-1]["status"], flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({
        "versions": VERSIONS, "seeds": SEEDS, "modes": MODES,
        "timeout_per_probe_seconds": timeout,
        "historical_v1_ref": ORIGINAL_V1, "experiment": "bounded_components",
        "results": results, "warning": "Different component metrics are not cross-version comparable."
    }, indent=2), encoding="utf-8")
    with (output / "raw.jsonl").open("w", encoding="utf-8") as out:
        for point in observations:
            out.write(json.dumps(point, sort_keys=True) + "\n")
    # Import the observer from the same repository; blind before analyzing.
    sys.path.insert(0, str(BASE))
    from observatory.blinding import blind_runs
    from observatory.core import analyze, load_jsonl, save_json
    runs = load_jsonl(output / "raw.jsonl")
    if runs:
        blind_runs(runs, output_path=output / "blind.jsonl",
                   key_file=output / "PRIVATE_mapping.json")
        report = analyze(load_jsonl(output / "blind.jsonl"),
                         window=6, permutations=199, seed=23)
        save_json(output / "observatory_report.json", report)
    print(json.dumps({
        "successful": sum(x["status"] == "ok" for x in results),
        "total": len(results), "observations": len(observations),
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", nargs=3, metavar=("VERSION", "SEED", "MODE"))
    parser.add_argument("--output", type=Path, default=Path("observatory_results/historical"))
    parser.add_argument("--timeout", type=int, default=55)
    args = parser.parse_args()
    if args.single:
        version, seed_text, mode = args.single
        if version not in VERSIONS or mode not in MODES:
            raise ValueError("invalid version/mode")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rows, scope = probe(version, int(seed_text), mode)
            print(json.dumps({"observations": rows, "scope": scope},
                             allow_nan=False))
        except Exception:
            traceback.print_exc()
            sys.exit(1)
    else:
        run_all(args.output, args.timeout)


if __name__ == "__main__":
    main()
