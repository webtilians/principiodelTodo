"""F9-X: frozen F9-V read-head mechanistic attribution on entirely fresh worlds.

No head, actor or F4 training. Off-line observer labels are never model inputs.
The entire intervention uses genuine old F9-P/F4 histories and F9-V READ/SKIP.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from observatory.f9_external_minigrid import ensure_dependencies
from observatory.f9q_train_cue_probes import validate_sources
from observatory.f9v_lexicographic_causal_f4 import one_seed_cases, head_inputs
from observatory.f9v_frozen_confirm import (
    validate_original_implementation, load_dev_models, FROZEN_WEIGHTS
)

BASE = 9_500_000
WORLDS = 32
GROUPS = {"public": slice(0, 148), "H": slice(148, 212), "C": slice(212, 276)}
PATCHES = ("cue_H", "cue_C", "cue_HC", "history_HC")
REGIMES = ("history_f4_normal", "history_f4_postcue_off")


def seed_for(index: int, j: int) -> int:
    if index not in range(4) or not 0 <= j < WORLDS:
        raise ValueError("Only new F9-X attribution worlds 9500000+ are permitted")
    return BASE + 10_000 * index + j


def vector(case):
    return np.concatenate((
        np.asarray(case["public"], dtype=np.float32),
        np.asarray(case["hc"], dtype=np.float32),
    ))


def patched_vector(recipient, donor, patch):
    a, b = vector(recipient), vector(donor)
    if patch in ("cue_H",):
        a[GROUPS["H"]] = b[GROUPS["H"]]
    elif patch in ("cue_C",):
        a[GROUPS["C"]] = b[GROUPS["C"]]
    elif patch in ("cue_HC", "history_HC"):
        a[148:] = b[148:]
    else:
        raise ValueError(patch)
    assert np.array_equal(a[:148], vector(recipient)[:148])
    return a


def score(head, x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(np.asarray(x, dtype=np.float32))
    with torch.no_grad():
        logit = float(head.net(x).squeeze().item())
    return {"logit": logit, "prob": float(torch.sigmoid(torch.tensor(logit))),
            "read": int(logit >= 0.0)}


def choice_metrics(cases, reads):
    assert len(cases) == len(reads)
    selected = [c for c, read in zip(cases, reads) if read]
    chosen = [c["read_branch"] if read else c["skip_branch"]
              for c, read in zip(cases, reads)]
    return {
        "cases": len(cases),
        "reads": len(selected),
        "useful_read": sum(c["effect"] == 1 for c in selected),
        "harmful_read": sum(c["effect"] == -1 for c in selected),
        "neutral_read": sum(c["effect"] == 0 for c in selected),
        "positive_available": sum(c["effect"] == 1 for c in cases),
        "first_correct": sum(v["first_arm_correct"] for v in chosen),
    }


def sensitivity(head, value, baseline, ig_steps=32):
    x = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    logit = head.net(x).squeeze()
    grad = torch.autograd.grad(logit, x)[0].detach()
    x0 = torch.tensor(baseline, dtype=torch.float32)
    dx = x.detach() - x0
    avg = torch.zeros_like(x0)
    for step in range(1, ig_steps + 1):
        xi = (x0 + (step / ig_steps) * dx).detach().requires_grad_(True)
        yi = head.net(xi).squeeze()
        avg += torch.autograd.grad(yi, xi)[0].detach() / ig_steps
    ig = dx * avg
    return {
        "grad_abs": {g: float(grad[s].abs().sum()) for g, s in GROUPS.items()},
        "ig_abs": {g: float(ig[s].abs().sum()) for g, s in GROUPS.items()},
        "ig_signed": {g: float(ig[s].sum()) for g, s in GROUPS.items()},
    }


def summarize_cases(cases, rows, patch_rows):
    by_condition = {}
    keys = sorted({(c["cue"], c["regime"]) for c in cases})
    for cue, regime in keys:
        idx = [i for i, c in enumerate(cases)
               if (c["cue"], c["regime"]) == (cue, regime)]
        subset = [cases[i] for i in idx]
        by_condition[f"{cue}|{regime}"] = {
            "lex_hc": choice_metrics(subset, [rows[i]["lex_hc"]["read"] for i in idx]),
            "oracle_label_positives": sum(c["effect"] == 1 for c in subset),
            "lex_hc_mean_prob": float(np.mean([rows[i]["lex_hc"]["prob"] for i in idx])),
        }
    scores = {}
    for method in ("lex_hc", "lex_public", "key_normal", "always", "never"):
        scores[method] = choice_metrics(cases, [r[method]["read"] for r in rows])
    patch_summary = {}
    for method in PATCHES:
        eligible = [p for p in patch_rows if p["patch"] == method]
        ix = [p["recipient_index"] for p in eligible]
        if eligible:
            patch_summary[method] = {
                **choice_metrics([cases[i] for i in ix], [p["prediction"]["read"] for p in eligible]),
                "eligible_states": len(eligible),
                "flipped_decisions": sum(rows[i]["lex_hc"]["read"] != p["prediction"]["read"]
                                        for i, p in zip(ix, eligible)),
                "mean_abs_logit_delta": float(np.mean(
                    [abs(rows[i]["lex_hc"]["logit"] - p["prediction"]["logit"])
                     for i, p in zip(ix, eligible)])),
            }
    gradients = {
        metric: {group: float(np.mean([r["sensitivity"][metric][group] for r in rows]))
                 for group in GROUPS}
        for metric in ("grad_abs", "ig_abs", "ig_signed")
    }
    pairs = []
    for start in range(0, len(cases), 4):
        assert cases[start]["seed"] == cases[start + 3]["seed"]
        for i, j in ((start, start + 1), (start + 2, start + 3)):
            a, b = cases[i], cases[j]
            assert a["regime"] == b["regime"] and a["reverse"] != b["reverse"]
            assert np.array_equal(a["public"], b["public"]), "Cue pair must have matched public"
            pairs.append({
                "seed": a["seed"], "regime": a["regime"],
                "read_disagreement": int(rows[i]["lex_hc"]["read"] != rows[j]["lex_hc"]["read"]),
                "oracle_disagreement": int(a["effect"] != b["effect"]),
                "both_D_positive": int(a["effect"] == 1 and b["effect"] == 1),
                "either_D_positive": int(a["effect"] == 1 or b["effect"] == 1),
                "different_logit": abs(rows[i]["lex_hc"]["logit"] - rows[j]["lex_hc"]["logit"]),
            })
    return {
        "by_condition": by_condition, "strategies": scores,
        "patches": patch_summary, "gradient_sensitivity": gradients,
        "matched_cue_pairs": {
            "n": len(pairs),
            "different_READ": sum(p["read_disagreement"] for p in pairs),
            "D_different": sum(p["oracle_disagreement"] for p in pairs),
            "mean_abs_logit_difference": float(np.mean([p["different_logit"] for p in pairs])),
            "by_regime": {
                reg: {"pairs": sum(p["regime"] == reg for p in pairs),
                      "different_READ": sum(p["read_disagreement"] for p in pairs
                                            if p["regime"] == reg)}
                for reg in REGIMES
            },
        },
        "history_patches_ineligible_public_mismatch": 2 * WORLDS * 2 -
            sum(p["patch"] == "history_HC" for p in patch_rows),
    }


def execute(args):
    if args.index not in range(4):
        raise ValueError("Unregistered F9-P actor")
    torch.set_num_threads(1)
    validate_original_implementation()
    ensure_dependencies()
    paths, shas, _ = validate_sources(args.bc_source, args.ppo_source, args.index)
    from sb3_contrib import RecurrentPPO
    actor = RecurrentPPO.load(str(paths["ppo_sham"]), device="cpu")
    actor.policy.set_training_mode(False)
    for p in actor.policy.parameters():
        p.requires_grad_(False)
    heads, old = load_dev_models(args.weights_source, args.index)
    assert shas["ppo_sham"] == old["source_shas"]["ppo_sham"]
    hc, public = heads["lex_hc"], heads["lex_public"]
    baseline = head_inputs(old["train_cases"], use_hc=True).mean(dim=0).numpy()
    assert baseline.shape == (276,)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for j in range(WORLDS):
        world = one_seed_cases(actor, args.index, seed_for(args.index, j))
        assert len(world) == 4
        assert all(v["public_fork_equal"] for v in world)
        cases.extend(world)
        print("F9X_NEW_WORLD", args.index, j, flush=True)
    rows = []
    for c in cases:
        read_hc = score(hc, vector(c))
        read_public = score(public, head_inputs([c], use_hc=False)[0])
        shortcut = int(c["cue"] == "key" and c["regime"] == "history_f4_normal")
        rows.append({
            "lex_hc": read_hc, "lex_public": read_public,
            "key_normal": {"read": shortcut},
            "always": {"read": 1}, "never": {"read": 0},
            "sensitivity": sensitivity(hc, vector(c), baseline),
        })
    patches = []
    def apply(i, j, name):
        pred = score(hc, patched_vector(cases[i], cases[j], name))
        patches.append({"recipient_index": i, "donor_index": j,
                        "patch": name, "prediction": pred})
        return pred
    for start in range(0, len(cases), 4):
        for i, j in ((start, start+1), (start+1, start),
                     (start+2, start+3), (start+3, start+2)):
            for patch in ("cue_H", "cue_C", "cue_HC"):
                got = apply(i, j, patch)
                if patch == "cue_HC":
                    assert abs(got["logit"] - rows[j]["lex_hc"]["logit"]) < 2e-5
        for i, j in ((start, start+2), (start+2, start),
                     (start+1, start+3), (start+3, start+1)):
            if np.array_equal(cases[i]["public"], cases[j]["public"]):
                apply(i, j, "history_HC")
    summary = summarize_cases(cases, rows, patches)
    report = {
        "experiment": "F9-X frozen-head information attribution",
        "stage": "EXPLORATORY_NEW_WORLDS",
        "index": args.index, "original_dev_run": 41345497955,
        "frozen_heads_sha256": FROZEN_WEIGHTS[args.index],
        "actor_sha256": shas["ppo_sham"],
        "worlds": WORLDS, "states": len(cases),
        "world_seeds": sorted({c["seed"] for c in cases}),
        "summary": summary,
        "cases": cases, "predictions": rows, "patches": patches,
        "original_f9v_8300000_reused": False,
        "f9w_9300000_used": False, "future_f9x_9600000_used": False,
        "retrained_actor": False, "retrained_head": False, "retrained_F4": False,
    }
    assert len(cases) == WORLDS * 4
    assert summary["matched_cue_pairs"]["n"] == WORLDS * 2
    assert all(9_500_000 <= s < 9_540_000 for s in report["world_seeds"])
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("F9X_FROZEN_ATTRIBUTION_SUMMARY", args.index,
          json.dumps(summary, ensure_ascii=False), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--bc-source", type=Path, required=True)
    p.add_argument("--ppo-source", type=Path, required=True)
    p.add_argument("--weights-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    execute(p.parse_args())


if __name__ == "__main__":
    main()
