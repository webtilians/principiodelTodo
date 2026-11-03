"""F10-N-C: ORIGINAL F10-N evaluation, originals loaded from original runs.

No train(), observe(), optimizer, new update, or selection by observed
confirmation score. Pre-CONFIRM gate checks all original checkpoint SHA and
regenerates exact original DEV+OOD prediction rows BEFORE instantiating any
26300000+ world. The ONLY injection to original evaluator is the data source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from observatory import f10n_worlds as original_world
from observatory import f10n_experiment as original_eval
from observatory.f10m_experiment import frozen_original_models
from observatory.f10n_confirm_worlds import (
    new_confirm_cases,old_ood_regression_cases
)

ORIGINAL_F10N_RUN=41714113859
ORIGINAL_F10N_CODE="6d3e7bd8753f3ad5d1d4a6e44208866bac036298"
ORIGINAL_F10L_RUN=41685499939
MEMORIES=("reward_table","learned_effects_uniform","learned_effects_directed")
MODES=("normal","shuffle","x_disabled","public_randomized")
ARMS=original_eval.ARM_NAMES

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_original(rep,nroot,lroot,g0_path):
    assert rep in range(4)
    train=json.loads((nroot/"training.json").read_text())
    g0=json.loads(g0_path.read_text())
    assert train["replica"]==rep
    assert train["status"]=="MEMORIES_SHA_FROZEN_BEFORE_DEV_OOD"
    assert train["train_heldout_srx_count"]==0
    assert train["heldout_program"]=="SRX"
    assert train["unseen_memory_flip1_never_trained"]
    assert train["confirm_26300000_sealed"]
    assert train["epochs"]==16
    assert train["episodes_per_condition"]==98304
    assert train["train_program_catalog_size"]==84
    assert train["original_sha_verified_after_training"]
    assert set(train["memory_files"])==set(MEMORIES)
    assert set(train["experience_log_files"])=={"uniform","directed"}
    assert train["original_skills"]["original_run_id"]==ORIGINAL_F10L_RUN
    assert train["original_skills"]["all_5_original_checkpoint_sha"]==(
        g0["replicas"][str(rep)]["model_sha256"])
    memory_shas={}
    for name,info in train["memory_files"].items():
        f=nroot/info["file"]
        assert sha(f)==info["sha256"],"Frozen original memory SHA changed"
        memory_shas[name]=info["sha256"]
        data=json.loads(f.read_text())
        assert data["episodes"]==98304
    log_shas={}
    for name,info in train["experience_log_files"].items():
        f=nroot/info["file"]
        assert sha(f)==info["sha256"],"Original experience log SHA changed"
        log_shas[name]=info["sha256"]
    selector,xor,lut,skills=frozen_original_models(rep,lroot,g0)
    assert lut==(1,0)
    assert skills["all_5_original_checkpoint_sha"]==(
        train["original_skills"]["all_5_original_checkpoint_sha"])
    for name,expected in skills["all_5_original_checkpoint_sha"].items():
        assert sha(lroot/(name+".pt"))==expected
    assert len(train["train_case_shas"])==3072
    assert len(set(train["train_case_shas"]))==3072
    expected_rows=nroot/"rows.json"
    assert expected_rows.exists()
    return train,selector,lut,skills,{
        "original_training_json_sha256":sha(nroot/"training.json"),
        "original_rows_json_sha256":sha(expected_rows),
        "memory_sha256":memory_shas,"trial_log_sha256":log_shas,
        "f10l_original_sha256":skills["all_5_original_checkpoint_sha"],
    }

def identical_original_worlds(rep):
    old=original_world.dataset("ood",rep)
    copied=old_ood_regression_cases(rep)
    assert len(old)==len(copied)==512
    for expected,actual in zip(old,copied):
        assert set(expected)==set(actual)
        for key in expected:
            if isinstance(expected[key],np.ndarray):
                assert expected[key].dtype==actual[key].dtype
                assert np.array_equal(expected[key],actual[key])
                assert expected[key].tobytes()==actual[key].tobytes()
            else:
                assert expected[key]==actual[key],(rep,key)
    print("F10NC_ORIGINAL_OOD_WORLDS_BIT_EXACT",rep,512,flush=True)

def _original_evaluate_in_temp(rep,train,selector,lut,nroot,confirm_data=None):
    """Run EXACT ORIGINAL original_eval.evaluate in fresh temporary folder."""
    with tempfile.TemporaryDirectory(prefix=f"f10nc_regression_rep{rep}_") as t:
        out=Path(t)
        for info in train["memory_files"].values():
            shutil.copyfile(nroot/info["file"],out/info["file"])
            assert sha(out/info["file"])==info["sha256"]
        if confirm_data is None:
            # Original unpatched DEV/OOD; never generate confirmation here.
            original_eval.evaluate(rep,out,train,selector,lut)
        else:
            assert len(confirm_data)==512
            def data_source(split,replica):
                assert replica==rep and split in ("dev","ood")
                return (original_world.dataset("dev",rep)
                        if split=="dev" else confirm_data)
            with patch.object(original_eval,"dataset",side_effect=data_source):
                original_eval.evaluate(rep,out,train,selector,lut)
        for info in train["memory_files"].values():
            assert sha(out/info["file"])==info["sha256"]
        return json.loads((out/"rows.json").read_text())

def run(rep,nroot,lroot,g0_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train,selector,lut,skills,proof=verify_original(rep,nroot,lroot,g0_path)
    before_n={p.name:sha(p) for p in nroot.iterdir() if p.is_file()}
    before_l={p.name:sha(p) for p in lroot.iterdir() if p.is_file()}
    # Both checks operate ONLY on the already-open F10-N OOD 26400000+.
    identical_original_worlds(rep)
    reproduced=_original_evaluate_in_temp(rep,train,selector,lut,nroot)
    reference=json.loads((nroot/"rows.json").read_text())
    assert reproduced==reference, (
        "Original DEV/OOD rows not bit-identical; prohibit opening 26300000+")
    assert len(reproduced)==4*(768+512)
    assert {x["split"] for x in reproduced}=={"dev","ood"}
    print("F10NC_FROZEN_ORIGINAL_EVALUATOR_BIT_EXACT",rep,len(reproduced),
          flush=True)

    # Confirm BOTH ORIGINAL artifact trees still byte identical.
    assert {p.name:sha(p) for p in nroot.iterdir() if p.is_file()}==before_n
    assert {p.name:sha(p) for p in lroot.iterdir() if p.is_file()}==before_l

    # FIRST and ONLY new confirmation dataset generation for this replica.
    fresh=new_confirm_cases(rep)
    assert len(fresh)==512
    assert len({x["case_sha"] for x in fresh})==512
    assert {x["seed"] for x in fresh}=={
        26_300_000+10000*rep+j for j in range(128)}
    assert not ({x["case_sha"] for x in fresh}&{
        x["case_sha"] for x in original_world.dataset("ood",rep)})
    # Identical, unchanged ORIGINAL evaluation code. Only OOD dataset input
    # is substituted with never-before-opened CONFIRM worlds.
    all_rows=_original_evaluate_in_temp(
        rep,train,selector,lut,nroot,confirm_data=fresh)
    selected=[dict(x,split="confirm") for x in all_rows if x["split"]=="ood"]
    assert len(selected)==4*512
    assert {x["case_sha"] for x in selected}=={
        x["case_sha"] for x in fresh}
    assert {x["seed"] for x in selected}=={
        26_300_000+10000*rep+j for j in range(128)}
    assert {x["arm"] for x in selected}==set(ARMS)
    assert all(x["task"]=="MEMORY" and x["flip"]==1 for x in selected)

    # Frozen checkpoints never written, updated or reserialized.
    assert {p.name:sha(p) for p in nroot.iterdir() if p.is_file()}==before_n
    assert {p.name:sha(p) for p in lroot.iterdir() if p.is_file()}==before_l
    result={
        "experiment":"F10-N-C original immutable 26300000+ confirmation",
        "status":"CONFIRM_GENERATED_ONCE_AFTER_SHA_AND_ORIGINAL_OLD_OOD_REGRESSION",
        "replica":rep,
        "original_f10n_run_id":ORIGINAL_F10N_RUN,
        "original_f10n_code_commit":ORIGINAL_F10N_CODE,
        "original_f10l_run_id":ORIGINAL_F10L_RUN,
        "no_training_or_memory_update":True,
        "original_ood_worlds_bit_exact":True,
        "original_dev_ood_predictions_bit_exact":True,
        "uses_unmodified_original_f10n_evaluate":True,
        "only_input_data_changed":True,
        "original_sha_proof":proof,
        "f10l_original_sha":skills["all_5_original_checkpoint_sha"],
        "n_base_worlds":128,"n_cases":512,"n_rows":4*512,
        "namespace":26300000,
        "rows":selected,
    }
    print("F10NC_ORIGINAL_FROZEN_CONFIRM",rep,
          "effect_accuracy",sum(
              x["outcomes"]["normal"]["success"] for x in selected
              if x["arm"]=="learned_effects_uniform")/512,
          "selected_srx",sum(
              x["program"]=="SRX" for x in selected
              if x["arm"]=="learned_effects_uniform"),
          flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--replica",type=int,required=True,choices=range(4))
    p.add_argument("--original-n",type=Path,required=True)
    p.add_argument("--original-l",type=Path,required=True)
    p.add_argument("--original-g0",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=run(a.replica,a.original_n,a.original_l,a.original_g0)
    a.out.write_text(json.dumps(result,separators=(",",":")))
