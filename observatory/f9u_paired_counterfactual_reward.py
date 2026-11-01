"""F9-U: genuinely paired counterfactual reward for our F9 recurrent F4 reader.

Two by two controlled study:
  reward: matched individual per-cue reward vs pair native-return bonus
  input: prior actor H/C+public vs public only.

Every head has same 276->32->1 architecture, same initial weights per
input mode, 600 TRAIN world pairs, 1200 episodes, 600 optimizer updates.
Actor checkpoint and F4 writer are frozen; no OpenAI or teacher label inputs.
Confirmation seeds 7300000+ are not accessible from this entrypoint.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from observatory.f9_external_minigrid import file_sha, ensure_dependencies
from observatory.f9t_learned_f4_read_gate import (
    ReadHead, READ_COST, LR, ENTROPY_BONUS, rollout, load_verified_actor,
)

TRAIN_BASE = 7_100_000
DEV_BASE = 7_200_000
SEALED_BASE = 7_300_000
WORLD_COUNTS = {"train": 600, "dev": 20}
REPLICAS = 4
PAIR_BONUS = 0.5
EMA_RATE = 0.02
ARMS = ("individual_hc", "paired_hc", "individual_public", "paired_public")
CONTROLS = ("always", "never", "periodic_4", "public_event_clock")
METHODS = ("individual", "paired")
INPUT_MODES = ("hc", "public")


def seed_for(split: str, index: int, j: int) -> int:
    if split not in WORLD_COUNTS or index not in range(REPLICAS):
        raise ValueError("Only F9-U TRAIN/DEV are accessible")
    if not 0 <= j < WORLD_COUNTS[split]:
        raise ValueError("Outside preregistered split")
    return (TRAIN_BASE if split == "train" else DEV_BASE) + index*10_000 + j


def arm_mode(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError("Unknown F9-U head")
    return "learned_hc" if arm.endswith("_hc") else "learned_public_only"


def paired_reward(a: dict, b: dict, beta: float = PAIR_BONUS) -> float:
    """Only native environment returns and F4 READ/step count; no oracle labels."""
    if beta != PAIR_BONUS:
        raise ValueError("Frozen pair bonus")
    for ep in (a, b):
        if not (0 <= ep["read_rate"] <= 1):
            raise AssertionError("Invalid read fraction")
    return (
        0.5 * (a["native_return"]+b["native_return"])
        + beta*min(a["native_return"],b["native_return"])
        - READ_COST*0.5*(a["read_rate"]+b["read_rate"])
    )


def head_init(index: int, input_mode: str) -> ReadHead:
    if index not in range(REPLICAS) or input_mode not in INPUT_MODES:
        raise ValueError("Unapproved head initialization")
    # Same initial state for individual and paired treatment; not one shared
    # agent that carries over experience from one treatment to another.
    torch.manual_seed(77_000 + index + (0 if input_mode == "hc" else 100))
    return ReadHead()


def train_head(actor, index: int, arm: str, worlds: int = 600):
    if arm not in ARMS or index not in range(REPLICAS) or not 1 <= worlds <= 600:
        raise ValueError("Invalid F9-U training configuration")
    method, input_mode = arm.split("_", 1)
    head = head_init(index, input_mode)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    torch.manual_seed(117_000+index+(0 if input_mode=="hc" else 100))
    baseline = 0.0
    updates = 0
    episodes = 0
    trace = []
    for j in range(worlds):
        seed = seed_for("train",index,j)
        # The two episodes use identical weights. No optimizer update is
        # allowed between original and inverted cues.
        first = rollout(actor,head,index,seed,False,arm_mode(arm),train=True)
        second = rollout(actor,head,index,seed,True,arm_mode(arm),train=True)
        episodes += 2
        a = torch.stack(first["_log_probs"]).mean()
        b = torch.stack(second["_log_probs"]).mean()
        entropy = 0.5 * (
            torch.stack(first["_entropies"]).mean()
            + torch.stack(second["_entropies"]).mean()
        )
        if method == "individual":
            reward_a = first["objective_return"]
            reward_b = second["objective_return"]
            loss = -0.5 * (
                a * (reward_a-baseline) + b * (reward_b-baseline)
            ) - ENTROPY_BONUS*entropy
            observed = 0.5*(reward_a+reward_b)
        else:
            observed = paired_reward(first,second)
            loss = -0.5 * (a+b)*(observed-baseline) - ENTROPY_BONUS*entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),2.0)
        optimizer.step()
        updates += 1
        baseline = (1-EMA_RATE)*baseline + EMA_RATE*observed
        trace.append({
            "j":j,"reward_a":first["native_return"],"reward_b":second["native_return"],
            "both_succeeded":int(first["success"] and second["success"]),
            "both_first_choices_correct":int(
                first["first_choice_correct"] and second["first_choice_correct"]),
            "read_a":first["read_rate"],"read_b":second["read_rate"],
            "training_objective":float(observed),
            "ema_baseline":float(baseline),
            "loss":float(loss.detach().item()),
        })
    head.eval()
    return head,{
        "arm":arm,
        "seed_index":index,
        "worlds":worlds,
        "episodes":episodes,
        "optimizer_steps":updates,
        "same_initialization_by_input_mode":True,
        "mean_objective":statistics.mean(x["training_objective"] for x in trace),
        "mean_native_success":statistics.mean(
            (int(x["reward_a"]>0)+int(x["reward_b"]>0))/2 for x in trace),
        "mean_train_read_rate":statistics.mean(
            0.5*(x["read_a"]+x["read_b"]) for x in trace),
        "mean_both_first_choices_correct":statistics.mean(
            x["both_first_choices_correct"] for x in trace),
        "last_40":trace[-40:],
    }


def evaluate_one(actor, head, index, seed, reverse, arm, deterministic=True,
                 gate_state="normal"):
    mode = arm_mode(arm) if arm in ARMS else arm
    if not deterministic:
        # Fixed *independent* stochastic evaluation seed, paired across
        # treatment arms and never used in training.
        torch.manual_seed(9_400_000 + seed*2 + int(reverse))
    with torch.no_grad():
        ep = rollout(actor,head,index,seed,reverse,mode,
                     train=not deterministic,gate_state=gate_state)
    return {
        "first_choice_correct":ep["first_choice_correct"],
        "first_committed_side":ep["first_committed_side"],
        "fork_reached":ep["fork_reached"],
        "first_committed":ep["first_committed"],
        "terminal_success":ep["success"],
        "native_return":ep["native_return"],
        "read_count":ep["read_count"],
        "steps":ep["steps"],
        "effective_reads":ep["effective_reads"],
        "empty_reads":ep["empty_reads"],
        "postcue_reads":ep["postcue_reads"],
    }


def evaluate_pair(actor, heads, index, seed, deterministic=True):
    result = {}
    arms = (*ARMS,*CONTROLS)
    for arm in arms:
        head = heads.get(arm)
        a = evaluate_one(actor,head,index,seed,False,arm,deterministic)
        b = evaluate_one(actor,head,index,seed,True,arm,deterministic)
        same_side = int(
            a["first_committed_side"] in ("upper","lower")
            and a["first_committed_side"]==b["first_committed_side"])
        result[arm] = {
            "both_first_correct":int(
                a["first_choice_correct"] and b["first_choice_correct"]),
            "both_terminal_success":int(
                a["terminal_success"] and b["terminal_success"]),
            "individual_first_correct":a["first_choice_correct"]+b["first_choice_correct"],
            "individual_terminal_success":a["terminal_success"]+b["terminal_success"],
            "both_same_side":same_side,
            "total_reads":a["read_count"]+b["read_count"],
            "total_steps":a["steps"]+b["steps"],
            "total_effective_reads":a["effective_reads"]+b["effective_reads"],
            "total_empty_reads":a["empty_reads"]+b["empty_reads"],
            "read_rate":(a["read_count"]+b["read_count"])/max(1,a["steps"]+b["steps"]),
            "first_committed_sides":[a["first_committed_side"],b["first_committed_side"]],
            "episodes":[a,b],
        }
    # Keep original trained head completely frozen and only zero H/C in
    # its gate input; the actor's own H/C state remains unchanged.
    a=evaluate_one(actor,heads["paired_hc"],index,seed,False,
                   "paired_hc",deterministic,gate_state="zero")
    b=evaluate_one(actor,heads["paired_hc"],index,seed,True,
                   "paired_hc",deterministic,gate_state="zero")
    result["paired_hc_zero_gate_hc"]={
        "both_first_correct":int(a["first_choice_correct"] and b["first_choice_correct"]),
        "both_terminal_success":int(a["terminal_success"] and b["terminal_success"]),
        "individual_first_correct":a["first_choice_correct"]+b["first_choice_correct"],
        "individual_terminal_success":a["terminal_success"]+b["terminal_success"],
        "both_same_side":int(a["first_committed_side"] in ("upper","lower")
                        and a["first_committed_side"]==b["first_committed_side"]),
        "total_reads":a["read_count"]+b["read_count"],
        "total_steps":a["steps"]+b["steps"],
        "total_effective_reads":a["effective_reads"]+b["effective_reads"],
        "total_empty_reads":a["empty_reads"]+b["empty_reads"],
        "read_rate":(a["read_count"]+b["read_count"])/max(1,a["steps"]+b["steps"]),
        "first_committed_sides":[a["first_committed_side"],b["first_committed_side"]],
        "episodes":[a,b],
    }
    return result


def summarize(rows):
    if not rows:
        raise AssertionError("No F9-U evaluation rows")
    result={}
    for arm in rows[0]["modes"]:
        p=[row["modes"][arm] for row in rows]
        n=len(p)
        reads=sum(x["total_reads"] for x in p)
        frames=sum(x["total_steps"] for x in p)
        result[arm]={
            "pairs":n,
            "both_first_correct":sum(x["both_first_correct"] for x in p),
            "both_terminal_success":sum(x["both_terminal_success"] for x in p),
            "individual_first_correct":sum(x["individual_first_correct"] for x in p),
            "individual_terminal_success":sum(x["individual_terminal_success"] for x in p),
            "same_side_pairs":sum(x["both_same_side"] for x in p),
            "read_count":reads,
            "frame_count":frames,
            "read_rate":reads/frames,
            "effective_reads":sum(x["total_effective_reads"] for x in p),
            "empty_reads":sum(x["total_empty_reads"] for x in p),
        }
    return result


def execute(args):
    if args.index not in range(REPLICAS):
        raise ValueError("Only four independently trained actor sources")
    if args.phase not in ("smoke","dev"):
        raise ValueError("Confirmation is deliberately NOT implemented")
    if args.phase=="dev" and args.worlds!=600:
        raise ValueError("Full DEV experiment needs preregistered 600 TRAIN worlds")
    if args.phase=="smoke" and args.worlds not in (1,2):
        raise ValueError("Technical smoke must use 1 or 2 TRAIN worlds")
    torch.set_num_threads(1)
    ensure_dependencies()
    actor,source=load_verified_actor(args.bc_source,args.ppo_source,args.index)
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Refuse to overwrite any F9-U result")
    heads={}
    training={}
    for arm in ARMS:
        head,report=train_head(actor,args.index,arm,args.worlds)
        heads[arm]=head
        training[arm]=report
        torch.save(head.state_dict(),root/f"{arm}.pt")
    n=2 if args.phase=="smoke" else 20
    rows=[]
    for j in range(n):
        seed=seed_for("train" if args.phase=="smoke" else "dev",args.index,j)
        modes=evaluate_pair(actor,heads,args.index,seed,deterministic=True)
        # Separate stochastic readout: trained heads only, never selected as
        # a replacement for the preregistered deterministic primary result.
        stochastic={
            arm:{"episodes":[
                evaluate_one(actor,heads[arm],args.index,seed,r,arm,deterministic=False)
                for r in (False,True)
            ]}
            for arm in ARMS
        }
        for arm in ARMS:
            a,b=stochastic[arm]["episodes"]
            stochastic[arm].update({
                "both_first_correct":int(a["first_choice_correct"] and b["first_choice_correct"]),
                "read_rate":(a["read_count"]+b["read_count"])/max(1,a["steps"]+b["steps"]),
            })
        rows.append({"j":j,"seed":seed,"modes":modes,"stochastic":stochastic})
        print("F9U_DEV_PAIR",args.phase,args.index,j,
              {k:(v["both_first_correct"],round(v["read_rate"],3))
               for k,v in modes.items()},flush=True)
    summary=summarize(rows)
    report={
        "experiment":"F9-U neural F4 reader paired native-return bonus versus matched individual REINFORCE",
        "phase":args.phase,
        "index":args.index,
        "source":source,
        "training":training,
        "summary":summary,
        "worlds":rows,
        "frozen_hyperparameters":{
            "pair_bonus":PAIR_BONUS,"read_penalty":READ_COST,
            "learning_rate":LR,"entropy_bonus":ENTROPY_BONUS,
            "optimizer_steps_per_head":args.worlds,
            "training_episodes_per_head":2*args.worlds,
        },
        "actor_trained":False,
        "uses_external_llm":False,
        "historical_f9_holdout_900000_opened":False,
        "f9t_6300000_opened":False,
        "f9u_7300000_opened":False,
        "script_sha256":file_sha(Path(__file__)),
        "protocol_sha256":file_sha(
            Path("docs/INFINITO_F9U_PAIRED_COUNTERFACTUAL_REWARD_PROTOCOL.md")),
    }
    (root/"report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("F9U_RESULT",args.phase,args.index,
          json.dumps(summary,ensure_ascii=False),flush=True)
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("smoke","dev"),required=True)
    parser.add_argument("--index",type=int,required=True)
    parser.add_argument("--worlds",type=int,required=True)
    parser.add_argument("--bc-source",type=Path,required=True)
    parser.add_argument("--ppo-source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    execute(parser.parse_args())
