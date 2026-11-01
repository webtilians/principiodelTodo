"""INFINITO F9-T: LEARNED F4 READ gate attached to actual RecurrentPPO H/C.

A trainable 276->32->1 head decides F4 READ/SKIP BEFORE a fresh
F4 readout is available. Only the policy head learns from terminal native
MiniGrid reward minus a fixed read-fraction cost. Actor LSTM and F4 writer are
the existing INFINITO F9 checkpoints and modules, unchanged.

No external LLM, semantic RAG, future cue labels or teacher action. Original
F9 historical 900000+ holdout is never imported or evaluated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from sb3_contrib import RecurrentPPO
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9i_selective_memory import SelectiveF4,DIM
from observatory.f9j_recurrent_memory_attribution import frozen_action
from observatory.f9n_stage_autonomy import StageObserver
from observatory.f9q_train_cue_probes import validate_sources
from observatory.f9_external_minigrid import file_sha,ensure_dependencies

ACTOR_DIM=148
STATE_DIM=128
GATE_IN=ACTOR_DIM+STATE_DIM
TRAIN_BASE=6100000
DEV_BASE=6200000
CONFIRM_BASE=6300000
TRAIN_WORLDS=600
DEV_PAIRS=20
CONFIRM_PAIRS=40
INDICES=4
CAPACITY=8
READ_COST=0.15
LR=0.001
ENTROPY_BONUS=0.001
HIDDEN_SIZE=32
MODES=("learned_hc","learned_public_only","always","never",
       "periodic_4","public_event_clock")
LEARNED_MODES=("learned_hc","learned_public_only")

class ReadHead(nn.Module):
    """One Bernoulli decision per frame, causal input: public O + PREVIOUS H/C."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(GATE_IN,HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE,1),
        )
    def forward(self,x):
        if x.ndim!=1 or x.shape[0]!=GATE_IN:
            raise AssertionError("F9-T read gate got invalid input")
        return self.net(x).squeeze(-1)


def seed_for(split,index,j):
    limits={"train":(TRAIN_BASE,TRAIN_WORLDS),
            "dev":(DEV_BASE,DEV_PAIRS),
            "confirm":(CONFIRM_BASE,CONFIRM_PAIRS)}
    if split not in limits or index not in range(INDICES):
        raise ValueError("Unregistered F9-T partition")
    base,n=limits[split]
    if j not in range(n):
        raise ValueError("Index outside preregistered partition")
    seed=base+index*10000+j
    assert seed>=6100000 and seed<6400000
    return seed


def gate_features(raw,prior_state,use_hc=True):
    """F4 content, current goal and future target cannot enter this function."""
    raw=np.asarray(raw,dtype=np.float32)
    if raw.shape!=(ACTOR_DIM,) or not np.isfinite(raw).all():
        raise AssertionError("The gate only sees 148 public values")
    state=np.zeros(STATE_DIM,dtype=np.float32)
    if use_hc and prior_state is not None:
        if len(prior_state)!=2:
            raise AssertionError("Invalid LSTM state")
        h,c=[np.asarray(x,dtype=np.float32) for x in prior_state]
        if h.shape!=(1,1,64) or c.shape!=(1,1,64):
            raise AssertionError("Expected frozen 64+64 actor state")
        state=np.concatenate((h.flatten(),c.flatten()))
    return torch.tensor(np.concatenate((raw,state)),dtype=torch.float32)


def public_object_visible(raw):
    frame=np.asarray(raw,dtype=np.float32)[:147].reshape(7,7,3)
    objmax=max(OBJECT_TO_IDX.values())
    identities=np.rint(frame[:,:,0]*objmax).astype(int)
    return bool(np.any(np.isin(identities,
                       (OBJECT_TO_IDX["key"],OBJECT_TO_IDX["ball"]))))


def scheduled_read(mode,raw,t,first_public_event):
    """Controls depend ONLY on causal public signal and agent step."""
    if mode=="always":return True,first_public_event
    if mode=="never":return False,first_public_event
    if mode=="periodic_4":return t%4==0,first_public_event
    if mode!="public_event_clock":
        raise ValueError(mode)
    if first_public_event is None and public_object_visible(raw):
        first_public_event=t
    read=first_public_event is not None and (t-first_public_event)%4==0
    return read,first_public_event


def gate_decision(mode,head,raw,state,t,first_public,stochastic=False):
    if mode not in MODES:
        raise ValueError("Unknown F9-T mode")
    if mode not in LEARNED_MODES:
        read,first_public=scheduled_read(mode,raw,t,first_public)
        return read,first_public,None,None
    if head is None:
        raise RuntimeError("Trained gate head required")
    features=gate_features(raw,state,use_hc=mode=="learned_hc")
    logits=head(features)
    dist=Bernoulli(logits=logits)
    bit=dist.sample() if stochastic else (torch.sigmoid(logits)>=0.5).to(torch.float32)
    return bool(bit.item()),first_public,dist.log_prob(bit),dist.entropy()


def gated_actor_input(raw,encoded,read):
    """Invariant: the only mutable signals are external F4 readout channels."""
    raw=np.asarray(raw,dtype=np.float32)
    encoded=np.asarray(encoded,dtype=np.float32)
    if encoded.shape!=(DIM,) or raw.shape!=(ACTOR_DIM,) or not np.array_equal(raw,encoded[:ACTOR_DIM]):
        raise AssertionError("Gate changed public frame or F4 encoding width")
    x=encoded.copy()
    if not read:
        x[ACTOR_DIM:]=0.
    if not np.array_equal(x[:ACTOR_DIM],raw):
        raise AssertionError("Gate modified current public sensory observation")
    if read and not np.array_equal(x,encoded):
        raise AssertionError("READ must be unmodified F4")
    return x


def rollout(actor,head,index,seed,reverse,mode,train=False,gate_state="normal"):
    if index not in range(INDICES) or mode not in MODES:
        raise ValueError("Unapproved actor/gate")
    env=build_env(13)
    writer=SelectiveF4("context",CAPACITY,seed=81000+index)
    state=None
    total_reward=0.
    t=0
    read_count=0
    effective_reads=0
    empty_reads=0
    postcue_reads=0
    seen_public=False
    first_public=None
    first_clock=None
    log_probs=[]
    entropies=[]
    observer=None
    decision_trace=[]
    try:
        raw,_=env.reset(seed=seed)
        labels=get_world_labels(env)
        if reverse:
            raw,labels=swap_cue(env,labels)
        observer=StageObserver(labels,13)
        observer.observe(env.unwrapped.agent_pos,0)
        while True:
            # CRUCIAL: choose whether to retrieve BEFORE the writer produces
            # this frame's readout; gate has only current raw and PRIOR actor H/C.
            signal_state=state
            if gate_state=="zero" and state is not None:
                signal_state=(np.zeros_like(state[0]),np.zeros_like(state[1]))
            elif gate_state not in ("normal","zero"):
                raise ValueError(gate_state)
            read,first_clock,logp,entropy=gate_decision(
                mode,head,raw,signal_state,t,first_clock,stochastic=train)
            if logp is not None and train:
                log_probs.append(logp)
                entropies.append(entropy)
            # F4 writes all publicly observed events regardless of read decision.
            encoded=writer.encode(raw)
            x=gated_actor_input(raw,encoded,read)
            if read:
                read_count+=1
                if np.any(encoded[ACTOR_DIM:]):
                    effective_reads+=1
                else:
                    empty_reads+=1
                if seen_public:postcue_reads+=1
            seen_public=seen_public or bool(writer.new_events)
            action,state=frozen_action(actor,x,state,first=t==0)
            if not train:
                decision_trace.append({
                    "t":t,"read":int(read),
                    "writer_slots":len(writer.store.saved),
                    "f4_available":int(np.any(encoded[ACTOR_DIM:])),
                    "first_choice_pending":observer.first_commit_step is None,
                })
            raw,reward,terminated,truncated,_=env.step(action)
            total_reward+=float(reward)
            t+=1
            observer.observe(env.unwrapped.agent_pos,t)
            if terminated or truncated:break
        summary=observer.summary()
        return {
            "seed":seed,"reverse":reverse,"mode":mode,
            "steps":t,"native_return":total_reward,
            "success":int(total_reward>0),
            "first_choice_correct":summary["first_choice_correct"],
            "fork_reached":summary["junction_arrived"],
            "first_committed":summary["first_committed"],
            "first_committed_side":summary["first_committed_side"],
            "read_count":read_count,"read_rate":read_count/t,
            "effective_reads":effective_reads,
            "empty_reads":empty_reads,
            "postcue_reads":postcue_reads,
            "objective_return":total_reward-READ_COST*(read_count/t),
            "trace":decision_trace if not train else [],
            "_log_probs":log_probs,"_entropies":entropies,
        }
    finally:
        env.close()


def train_reader(actor,index,mode,worlds=TRAIN_WORLDS):
    if mode not in LEARNED_MODES or index not in range(INDICES):
        raise ValueError("Only 2 preregistered gate learners and 4 replicas")
    if not 1<=worlds<=TRAIN_WORLDS:
        raise ValueError("Training budget exceeded")
    torch.manual_seed(77000+index)
    head=ReadHead()
    optimizer=torch.optim.Adam(head.parameters(),lr=LR)
    moving_baseline=0.0
    log=[]
    for j in range(worlds):
        for reverse in (False,True):
            ep=rollout(actor,head,index,seed_for("train",index,j),reverse,mode,train=True)
            ret=ep["objective_return"]
            advantage=ret-moving_baseline
            moving_baseline=0.98*moving_baseline+0.02*ret
            if ep["_log_probs"]:
                logprobs=torch.stack(ep["_log_probs"])
                entropy=torch.stack(ep["_entropies"])
                # Actor always remains frozen. The gate alone receives gradient.
                loss=-(logprobs.mean()*advantage)-ENTROPY_BONUS*entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),2.0)
                optimizer.step()
            log.append({"j":j,"reverse":reverse,"success":ep["success"],
                        "native_return":ep["native_return"],
                        "read_rate":ep["read_rate"],"objective_return":ret})
    head.eval()
    return head,{"mode":mode,"index":index,"training_worlds":worlds,
                 "episodes":len(log),"mean_reward":statistics.mean(x["native_return"] for x in log),
                 "mean_read_rate":statistics.mean(x["read_rate"] for x in log),
                 "last_100_episodes":log[-100:]}


def evaluate_pair(actor,index,seed,heads,gate_state="normal"):
    out={}
    for mode in MODES:
        head=heads.get(mode)
        a=rollout(actor,head,index,seed,False,mode,gate_state=gate_state)
        b=rollout(actor,head,index,seed,True,mode,gate_state=gate_state)
        out[mode]={
            "pair_correct":int(a["first_choice_correct"] and b["first_choice_correct"]),
            "both_terminal_success":int(a["success"] and b["success"]),
            "individual_success":a["success"]+b["success"],
            "first_choice_correct":a["first_choice_correct"]+b["first_choice_correct"],
            "read_rate":(a["read_count"]+b["read_count"])/max(1,a["steps"]+b["steps"]),
            "read_counts":[a["read_count"],b["read_count"]],
            "steps":[a["steps"],b["steps"]],
            "traces":[a["trace"],b["trace"]],
            "cue_pair":[a["first_committed_side"],b["first_committed_side"]],
        }
    return out


def load_verified_actor(bc_dir,ppo_dir,index):
    paths,shas,provenance=validate_sources(bc_dir,ppo_dir,index)
    from sb3_contrib import RecurrentPPO
    actor=RecurrentPPO.load(str(paths["ppo_sham"]),device="cpu")
    actor.policy.set_training_mode(False)
    for p in actor.policy.parameters():
        p.requires_grad_(False)
    return actor,{"sha":shas["ppo_sham"],"provenance":provenance}


def execute(args):
    if args.index not in range(INDICES):
        raise ValueError("F9-T requires four independent sources")
    if args.phase=="confirm":
        raise RuntimeError("Confirmation cannot be started from the development entrypoint")
    if args.phase=="train" and args.worlds!=TRAIN_WORLDS:
        raise ValueError("Only preregistered complete training budget")
    if args.phase=="smoke" and args.worlds not in (1,2):
        raise ValueError("Smoke only 1-2 worlds")
    torch.set_num_threads(1)
    ensure_dependencies()
    actor,actor_source=load_verified_actor(args.bc_source,args.ppo_source,args.index)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Fresh F9-T output required")
    heads={}
    train_reports={}
    for mode in LEARNED_MODES:
        head,info=train_reader(actor,args.index,mode,worlds=args.worlds)
        heads[mode]=head
        train_reports[mode]=info
        torch.save(head.state_dict(),out/f"{mode}.pt")
    split="train" if args.phase=="smoke" else "dev"
    count=2 if args.phase=="smoke" else DEV_PAIRS
    pairs=[]
    for j in range(count):
        seed=seed_for(split,args.index,j)
        pair=evaluate_pair(actor,args.index,seed,heads)
        if args.phase=="train":
            # Crucial causal control: switch off H/C ONLY at reader's input.
            # The frozen actor keeps its OWN H/C and receives chosen F4 as usual.
            a=rollout(actor,heads["learned_hc"],args.index,seed,False,
                      "learned_hc",gate_state="zero")
            b=rollout(actor,heads["learned_hc"],args.index,seed,True,
                      "learned_hc",gate_state="zero")
            pair["hc_zero_gate_only"]={
                "pair_correct":int(a["first_choice_correct"] and b["first_choice_correct"]),
                "both_terminal_success":int(a["success"] and b["success"]),
                "read_rate":(a["read_count"]+b["read_count"])/max(1,a["steps"]+b["steps"]),
                "read_counts":[a["read_count"],b["read_count"]],
                "steps":[a["steps"],b["steps"]],
                "traces":[a["trace"],b["trace"]],
            }
        pairs.append(pair)
        print("F9T_PAIR",args.phase,args.index,j,
              {key:{"success":v["both_terminal_success"],"read_rate":round(v["read_rate"],3)}
               for key,v in pair.items()},flush=True)
    summary={mode:{
        "pair_correct":sum(p[mode]["pair_correct"] for p in pairs),
        "both_terminal_success":sum(p[mode]["both_terminal_success"] for p in pairs),
        "mean_read_rate":statistics.mean(p[mode]["read_rate"] for p in pairs),
    } for mode in MODES}
    if args.phase=="train":
        summary["hc_zero_gate_only"]={
            "pair_correct":sum(p["hc_zero_gate_only"]["pair_correct"] for p in pairs),
            "both_terminal_success":sum(p["hc_zero_gate_only"]["both_terminal_success"] for p in pairs),
            "mean_read_rate":statistics.mean(p["hc_zero_gate_only"]["read_rate"] for p in pairs),
        }
    report={"experiment":"F9-T real recurrent agent F4 read-head",
            "phase":args.phase,"index":args.index,
            "source":actor_source,"training":train_reports,"summary":summary,
            "pairs":pairs,"heldout_opened":False,
            "F9_historical_900000_opened":False,
            "no_external_llm":True,"actor_training":False,
            "source_sha256":file_sha(Path(__file__)),
            "protocol_sha256":file_sha(Path("docs/INFINITO_F9T_LEARNED_F4_READ_GATE_PROTOCOL.md"))}
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("F9T_RESULT",args.phase,args.index,json.dumps(summary),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("smoke","train"),required=True)
    parser.add_argument("--index",type=int,required=True)
    parser.add_argument("--worlds",type=int,required=True)
    parser.add_argument("--bc-source",type=Path,required=True)
    parser.add_argument("--ppo-source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    execute(args)

if __name__=="__main__":
    main()
