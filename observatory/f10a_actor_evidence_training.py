"""F10-A: matched actor training with true, corrupted and empty F4."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from observatory.f9_external_minigrid import ensure_dependencies,file_sha
from observatory.f9o_confirm_long_blackout import validate_training
from observatory.f9d_autonomous_navigation import demonstration
from observatory.f9e_f4_autonomous import clone
from observatory.f9k_temporal_memory_probes import stage_indices
from observatory.f9i_selective_memory import SelectiveF4,DIM,FIELDS
from observatory.f9m_temporal_dropout_training import parameter_signature

BASE=12100000
TRAIN_WORLDS=128
EPOCHS=10
QUALITIES=("clean","adversarial","blank")
REGIMES=("history_f4_normal","history_f4_postcue_off")
ARMS=("evidence_mixed","evidence_clean_control")

def train_seed(index,j):
    if index not in range(4) or j not in range(TRAIN_WORLDS):raise ValueError("F10-A TRAIN only")
    return BASE+10000*index+j

def modify_tail(tail,quality):
    if quality not in QUALITIES:raise ValueError(quality)
    x=tail.copy()
    if quality=="blank":x[:]=0
    if quality=="adversarial":
        for a in range(0,8*FIELDS,FIELDS):
            if x[a+7]>0:
                assert np.isclose(x[a]+x[a+1],1.)
                x[a],x[a+1]=x[a+1],x[a]
    return x

def transform(frames,actions,times,quality,regime,index):
    writer=SelectiveF4("context",8,seed=81000+index)
    outputs=[]
    altered=masked=0
    for t,frame in enumerate(frames):
        x=writer.encode(frame)
        tail=x[148:].copy()
        if t>=times["post_cue"]:
            new=modify_tail(tail,quality)
            altered+=int(not np.array_equal(new,tail))
            tail=new
        if regime==REGIMES[1] and times["post_cue"]<=t<times["fork"]:
            masked+=int(np.any(tail))
            tail[:]=0
        outputs.append(np.concatenate((frame,tail)).astype(np.float32))
    result=np.stack(outputs)
    assert result.shape==(len(actions),DIM) and np.array_equal(result[:,:148],frames)
    return result,altered,masked

def examples(index,j):
    seed=train_seed(index,j)
    raw=[demonstration(seed,size=13,reverse=b) for b in (False,True)]
    times=stage_indices(raw[0][1],raw[1][1],raw[0][2],raw[1][2])
    assert np.array_equal(raw[0][0][times["fork"]],raw[1][0][times["fork"]])
    mixed=[];control=[];corrupt_count=0;mask_count=0
    for frames,actions,seen in raw:
        for regime in REGIMES:
            clean,_,_=transform(frames,actions,times,"clean",regime,index)
            for quality in QUALITIES:
                arr,a,b=transform(frames,actions,times,quality,regime,index)
                assert np.array_equal(arr[:,:148],clean[:,:148])
                mixed.append((arr,actions,seen))
                control.append((clean.copy(),actions,seen))
                corrupt_count+=a;mask_count+=b
    assert len(mixed)==len(control)==12
    return mixed,control,corrupt_count,mask_count

def run(args):
    if args.index not in range(4) or args.worlds!=TRAIN_WORLDS or args.epochs!=EPOCHS:
        raise ValueError("Fixed F10-A train contract")
    torch.set_num_threads(1)
    deps=ensure_dependencies()
    source=Path(args.source)
    old_report=json.loads((source/"report.json").read_text())
    validate_training(old_report,source,args.index)
    checkpoint=source/"continuous6_after_bc.zip"
    original_sha=old_report["arms"]["continuous6"]["bc_sha256"]
    assert file_sha(checkpoint)==original_sha
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    data=[[],[]];changed=masked=0
    for j in range(TRAIN_WORLDS):
        a,b,x,y=examples(args.index,j)
        data[0].extend(a);data[1].extend(b)
        changed+=x;masked+=y
        print("F10A_TRAIN_NEW_WORLD",args.index,j,train_seed(args.index,j),flush=True)
    assert all(len(v)==12*TRAIN_WORLDS for v in data)
    assert changed>0 and masked>0
    assert all(np.array_equal(a[1],b[1]) and np.array_equal(a[0][:,:148],b[0][:,:148])
               for a,b in zip(*data))
    report={"experiment":"F10-A new actor evidence robustness",
            "index":args.index,"phase":"TRAIN_ONLY","train_worlds":TRAIN_WORLDS,
            "train_examples_per_arm":12*TRAIN_WORLDS,
            "source_bc_sha256":original_sha,"source_report_sha256":file_sha(source/"report.json"),
            "source_run":40957255987,"dependencies":deps,
            "counterfactual_teacher_actions_matched":True,"corrupted_frames":changed,
            "history_masked_frames":masked,"actor_architecture_unchanged":True,
            "reader_head_trained":False,"PPO_run":False,"old_actor_changed":False,
            "old_confirm_reused":False,"future_confirm_opened":False,"arms":{}}
    for name,seqs in zip(ARMS,data):
        torch.manual_seed(121000+args.index)
        actor=RecurrentPPO.load(str(checkpoint),device="cpu")
        initial=parameter_signature(actor.policy)
        assert actor.observation_space.shape==(DIM,)
        assert sum(p.numel() for p in actor.policy.parameters())==380552
        losses=clone(actor,seqs,EPOCHS,args.index)
        actor.policy.set_training_mode(False)
        path=out/(name+"_actor.zip")
        actor.save(path)
        report["arms"][name]={"initial_signature":initial,
             "trained_signature":parameter_signature(actor.policy),
             "checkpoint":path.name,"checkpoint_sha256":file_sha(path),
             "epochs":EPOCHS,"examples":len(seqs),"losses":losses}
        (out/"report.json").write_text(json.dumps(report,indent=2))
        print("F10A_TRAINED",args.index,name,losses[-1],flush=True)
        env=actor.get_env()
        if env is not None:env.close()
    assert report["arms"][ARMS[0]]["initial_signature"]==report["arms"][ARMS[1]]["initial_signature"]
    assert report["arms"][ARMS[0]]["trained_signature"]!=report["arms"][ARMS[1]]["trained_signature"]
    print("F10A_TRAIN_DONE",args.index,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--worlds",type=int,default=TRAIN_WORLDS)
    p.add_argument("--epochs",type=int,default=EPOCHS)
    run(p.parse_args())
