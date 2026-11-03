"""F10-L-C original frozen evaluator, DATA-only substitution and exact regression.

The function observatory.f10l_stage2.run is used UNMODIFIED. A scoped mock
injects only the independent dataset into its existing select_data boundary.
All original model forward methods, argmax, XOR, shuffles and ablations
therefore execute bit-for-bit identically.
"""
from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path
from unittest.mock import patch
import torch

from observatory import f10l_stage2 as original
from observatory.f10l_confirm_worlds import confirm_cases,regression_existing_ood

ORIGINAL_RUN_ID=41685499939
ORIGINAL_CODE_SHA="b5c32cbfb50a24414607e3a24e30e139ca92fede"
ARMS=("xor_onpolicy","xor_two_actions",
      "sel_onpolicy","sel_uniform_credit","sel_uniform_pair")

def _call_original_on_cases(rep,stage1,cases):
    """Patch ONLY input data source; never actor, parameters or evaluation."""
    assert len(cases)==512
    for c in cases:
        assert c["replica"]==rep
    def data_source(split,replica):
        assert split=="ood" and replica==rep
        return cases
    with patch.object(original,"select_data",side_effect=data_source):
        return original.run(rep,stage1)

def verify_frozen_original_weights(rep,stage1,g0_path):
    gate=json.loads(g0_path.read_text())
    assert gate["technical"]=="F10L_PHASE1_TECHNICAL_COMPLETE"
    assert gate["eligible_for_composition"] and gate["gate_counts"]["G0_both_primitives"]>=3
    assert gate["composition_23400000_not_opened"],"Original G0 archive must predate OOD"
    meta=json.loads((stage1/"train.json").read_text())
    assert meta["status"]=="FROZEN_BEFORE_DEV_AND_ANY_OOD"
    assert meta["replica"]==rep
    assert meta["confirm_23300000_unopened"],"Original training was before confirm"
    assert meta["train_composition_exposure"]==0
    assert set(meta["arms"])==set(ARMS)
    shas={}
    for arm in ARMS:
        info=meta["arms"][arm]
        cp=stage1/(arm+".pt")
        assert original.sha(cp)==info["sha256"]
        assert info["sha256"]==gate["replicas"][str(rep)]["model_sha256"][arm]
        assert info["executed_environment_episodes"]==65536
        assert info["updates"]==512 and not info["loss_supervised_winner_or_answer"]
        shas[arm]=info["sha256"]
    return shas

def run(rep,stage1,g0_path):
    """No OOD confirmation data is instantiated until BOTH checks pass."""
    shas=verify_frozen_original_weights(rep,stage1,g0_path)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # OLD OOD is already open/public in F10-L; regression verifies the new
    # generator/evaluator uses exactly the previous behavior.
    old=original.run(rep,stage1)
    same_ood=regression_existing_ood(rep)
    reproduced=_call_original_on_cases(rep,stage1,same_ood)
    assert old["rows"]==reproduced["rows"],"OLD OOD regression mismatch; stop before confirm"
    assert old["checkpoint_sha"]==shas
    assert reproduced["checkpoint_sha"]==shas
    print("F10LC_ORIGINAL_OOD_BIT_EXACT_REGRESSION",rep,len(old["rows"]),flush=True)

    # FIRST and ONLY confirmation-world generation for this replica.
    confirm=confirm_cases(rep)
    assert len(confirm)==512
    assert len({c["hash"] for c in confirm})==512
    assert {c["seed"] for c in confirm}==set(
        range(23_300_000+10000*rep,23_300_000+10000*rep+128))
    assert not ({c["hash"] for c in confirm}&{c["hash"] for c in same_ood})
    results=_call_original_on_cases(rep,stage1,confirm)
    assert results["checkpoint_sha"]==shas
    assert len(results["rows"])==512
    assert {r["hash"] for r in results["rows"]}=={c["hash"] for c in confirm}
    assert all(r["truth"] in (0,1) for r in results["rows"])
    result={
        "experiment":"INFINITO F10-L-C independent seed confirmation",
        "status":"CONFIRMATION_WORLDS_OPENED_ONCE_AND_EVALUATED",
        "replica":rep,
        "original_model_run_id":ORIGINAL_RUN_ID,
        "original_code_commit":ORIGINAL_CODE_SHA,
        "using_unmodified_original_stage2_run":True,
        "old_ood_regression_passed_bit_exact":True,
        "training_updates_during_confirmation":0,
        "checkpoint_sha":shas,
        "confirmation_namespace":23300000,
        "worlds":128,"n_predictions":512,
        "rows":results["rows"],
    }
    print("F10LC_CONFIRM_FROZEN_ONLY",rep,
          "composed",sum(r["predictions"]["composed"]==r["truth"]
                         for r in result["rows"])/512,flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--replica",type=int,choices=range(4),required=True)
    p.add_argument("--stage1",type=Path,required=True)
    p.add_argument("--gate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=run(a.replica,a.stage1,a.gate)
    a.out.write_text(json.dumps(result,separators=(",",":")))
