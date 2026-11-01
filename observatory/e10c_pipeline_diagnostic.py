"""E10-C single, stagewise memory audit on entirely new DEVELOPMENT fixtures.

NO LLM inference. Fake/hash backend ONLY for tests. GitHub Actions must use
REAL text-embedding-3-small for scientific diagnostic, never 1800000+.
Unmodified INFINITO 3.0 via E10-B's existing ExperimentArm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from observatory.e10b_semantic_longitudinal_live import (
    ACTIONS, NAMES, NOISE, VALUES, FAMILIES, World,
    ExperimentArm, CachedSemanticEmbeddings, CONTEXT_BUDGET, TOP_K,
    factual_message, paired_history, question,
)
from src.infinito3.llm_adapter import RecordingLLMAdapter
from src.infinito3.persistent_memory import HashEmbeddingProvider

DEV_SEED_BASE=1900000
WORLD_INDICES=2
FAMILY_COUNT=6
AUDIT_ARMS=("semantic_rag","infinito3","infinito3_gate_open")
STAGES=("not_captured","inactive","not_retrieved","not_selected",
        "not_transmitted","transmitted")

def seed_for(index:int, j:int)->int:
    if index not in range(WORLD_INDICES) or j not in range(FAMILY_COUNT):
        raise ValueError("Only preregistered E10-C development seeds allowed")
    return DEV_SEED_BASE+index*1000+j

def fresh_world(seed:int)->World:
    if seed not in {seed_for(i,j) for i in range(WORLD_INDICES) for j in range(FAMILY_COUNT)}:
        raise ValueError("E10-C does not allow prior E10-B or F9 seeds")
    rng=random.Random(seed)
    family=FAMILIES[seed%1000]
    return World(
        seed=seed,family=family,
        subject=f"{rng.choice(NAMES)}-{rng.randrange(100,999)}",
        options=tuple(rng.sample(VALUES[family],2)),
        actions=tuple(rng.sample(ACTIONS,2)),
        distractors_before=tuple(rng.choice(NOISE) for _ in range(12)),
        distractors_after=tuple(rng.choice(NOISE) for _ in range(12)),
    )

def diagnostic_adapter():
    # No calls to actual LLM even in the REAL-embedding run.
    return RecordingLLMAdapter(
        responder=lambda request:"NO_LLM_CALL",
        provider="E10C-INSTRUMENTATION-ONLY",
        model="no-language-model",
    )

def all_records(arm):
    store=arm.engine.memory_store
    if arm.name in ("infinito3","infinito3_gate_open"):
        return store.all(include_inactive=True)
    return store.all()

def trace_stage(arm, sentence:str, q:str, gate_at_ingestion:dict):
    # One genuine cognitive turn. Use decision.context from the SAME search
    # executed by CognitiveEngine, not a second search that could change rank.
    run=arm.loop.turn(q,top_k=TOP_K,context_budget_tokens=CONTEXT_BUDGET)
    decision=run.cognitive_decision
    assert decision is not None and decision.context_packet is not None
    packet=decision.context_packet
    assert packet.estimated_tokens<=CONTEXT_BUDGET
    assert run.response.provider=="E10C-INSTRUMENTATION-ONLY"
    rows=all_records(arm)
    matching=[r for r in rows if r.content==sentence]
    # The current memory must be active; a superseded older fact should not count.
    current=[r for r in matching if r.status.value=="active"]
    ids={r.id for r in current}
    retrieved={r.id for r in decision.context}
    selected={item.memory_id for item in packet.items if item.memory_id}
    prior_messages=[m.content for m in run.request.messages[:-1]
                    if m.role=="user" and "INFINITO REFERENCE CONTEXT" in m.content]
    reached=any(sentence in msg for msg in prior_messages)
    stored=bool(matching)
    active=bool(current)
    in_search=bool(ids & retrieved)
    in_context=bool(ids & selected)
    if not stored:
        stage="not_captured"
    elif not active:
        stage="inactive"
    elif not in_search:
        stage="not_retrieved"
    elif not in_context:
        stage="not_selected"
    elif not reached:
        stage="not_transmitted"
    else:
        stage="transmitted"
    assert not reached or in_context, "Untracked evidence transmission"
    return {
        "gate":gate_at_ingestion,
        "persisted":stored,
        "active":active,
        "retrieved":in_search,
        "selected":in_context,
        "transmitted":reached,
        "first_missing_stage":stage,
        "active_total":sum(r.status.value=="active" for r in rows),
        "inactive_total":sum(r.status.value!="active" for r in rows),
        "retrieved_total":len(decision.context),
        "selected_total":len(packet.items),
        "context_tokens":packet.estimated_tokens,
        "precision_diagnostics":packet.diagnostics,
        "relevant_record_statuses":[r.status.value for r in matching],
        "retrieved_memory_contents":[r.content for r in decision.context],
        "selected_memory_contents":[item.content for item in packet.items],
        "reference_context":packet.rendered,
    }

def evaluate_world(world:World,embeddings)->dict:
    cases=[]
    q=question(world)
    for reversed_cue in (False,True):
        before,after=paired_history(world,reversed_cue)
        with tempfile.TemporaryDirectory(prefix="e10c-stage-") as temp:
            arms={
                name:ExperimentArm(name,client=None,embeddings=embeddings,
                                   tempdir=Path(temp)/name,
                                   adapter_factory=diagnostic_adapter)
                for name in AUDIT_ARMS
            }
            try:
                for phrase in before:
                    for arm in arms.values():arm.ingest(phrase)
                before_gate={
                    name:gate_data(arm,before[0]) for name,arm in arms.items()
                }
                row_before={
                    name:trace_stage(arm,before[0],q,before_gate[name])
                    for name,arm in arms.items()
                }
                for arm in arms.values():arm.restart()
                for phrase in after:
                    for arm in arms.values():arm.ingest(phrase)
                after_gate={
                    name:gate_data(arm,after[0]) for name,arm in arms.items()
                }
                row_after={
                    name:trace_stage(arm,after[0],q,after_gate[name])
                    for name,arm in arms.items()
                }
                cases.extend([
                    {"cue":"reversed" if reversed_cue else "original",
                     "phase":"before","target_sentence":before[0],
                     "arms":row_before},
                    {"cue":"reversed" if reversed_cue else "original",
                     "phase":"after","target_sentence":after[0],
                     "arms":row_after},
                ])
            finally:
                for arm in arms.values():arm.close()
    return {"seed":world.seed,"family":world.family,
            "subject":world.subject,"query":q,"observations":cases}

def gate_data(arm, text:str)->dict:
    gate=arm.engine.memory_gate.evaluate(text)
    return {
        "should_store":gate.should_store,
        "importance":gate.importance,
        "kind":gate.kind.value,
        "reasons":list(gate.reasons),
    }

def aggregate(rows:list[dict])->dict:
    samples=[(r["family"],sample) for r in rows for sample in r["observations"]]
    if len(samples)!=WORLD_INDICES*FAMILY_COUNT*4:
        raise AssertionError("Must preserve exactly 48 paired decision snapshots")
    summary={"snapshots_per_arm":len(samples),"by_arm":{},"by_family":{},"by_phase":{}}
    for arm in AUDIT_ARMS:
        snapshots=[s["arms"][arm] for _,s in samples]
        summary["by_arm"][arm]={
            "stage_counts":dict(Counter(s["first_missing_stage"] for s in snapshots)),
            "persisted":sum(s["persisted"] for s in snapshots),
            "active":sum(s["active"] for s in snapshots),
            "retrieved":sum(s["retrieved"] for s in snapshots),
            "selected":sum(s["selected"] for s in snapshots),
            "transmitted":sum(s["transmitted"] for s in snapshots),
            "mean_context_tokens":statistics.mean(s["context_tokens"] for s in snapshots),
        }
        summary["by_family"][arm]={
            family:{
                "stage_counts":dict(Counter(s["arms"][arm]["first_missing_stage"]
                    for f,s in samples if f==family)),
                "transmitted":sum(s["arms"][arm]["transmitted"]
                    for f,s in samples if f==family),
                "total":sum(f==family for f,_ in samples),
            }for family in FAMILIES
        }
        summary["by_phase"][arm]={
            phase:{
                "stage_counts":dict(Counter(s["arms"][arm]["first_missing_stage"]
                    for _,s in samples if s["phase"]==phase)),
                "transmitted":sum(s["arms"][arm]["transmitted"]
                    for _,s in samples if s["phase"]==phase),
            } for phase in ("before","after")
        }
    return summary

def run(output:Path,embedding_mode:str):
    if embedding_mode not in ("real","hash-test"):
        raise ValueError("Invalid embedding mode")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("E10-C refuses to overwrite existing output")
    output.mkdir(parents=True,exist_ok=True)
    if embedding_mode=="real":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Real E10-C diagnosis requires OPENAI_API_KEY")
        from openai import OpenAI
        client=OpenAI(api_key=os.environ["OPENAI_API_KEY"],timeout=60,max_retries=2)
        embeddings=CachedSemanticEmbeddings(client)
    else:
        embeddings=HashEmbeddingProvider(dimensions=128)
    rows=[]
    for index in range(WORLD_INDICES):
        for family in range(FAMILY_COUNT):
            world=fresh_world(seed_for(index,family))
            rows.append(evaluate_world(world,embeddings))
            print("E10C_DEV_WORLD",index,family,world.family,flush=True)
    summary=aggregate(rows)
    result={
        "experiment":"INFINITO E10-C diagnostic only (not confirmation)",
        "mode":embedding_mode,
        "semantic_embeddings_real":embedding_mode=="real",
        "llm_calls":0,
        "model_decisions_scored":False,
        "worlds":12,"decision_snapshots_per_arm":48,
        "seed_first":DEV_SEED_BASE,
        "seed_range":"1900000+index*1000+family, index 0..1",
        "core_changed":False,
        "historical_E10B_confirmation_reused":False,
        "source_run_e10b":41109251106,
        "sha":os.environ.get("GITHUB_SHA","local"),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256":hashlib.sha256(Path("docs/INFINITO_E10C_PIPELINE_DIAGNOSTIC_PROTOCOL.md").read_bytes()).hexdigest(),
        "summary":summary,"world_traces":rows,
    }
    (output/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print("E10C_PIPELINE_RESULT",json.dumps({
        "mode":embedding_mode,"summary":summary
    },ensure_ascii=False),flush=True)
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--embedding",choices=("real","hash-test"),default="real")
    args=parser.parse_args()
    run(args.output,args.embedding)
