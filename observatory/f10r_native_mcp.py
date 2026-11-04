"""F10-R: empirical MCP response-action selector versus documented policy.

Study authentic INFINITO MCP stdio ONLY at pinned product commit in an ephemeral
SQLite DB. Never connects to a user's MCP or any persistent personal memory.

This is NOT F10-P's EventEffects model: it is an explicitly goal-conditioned,
tabular outcome learner over tool calls, a fair candidate for incremental
product value against a competent docs policy.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from mcp import Client, StdioServerParameters

GOALS=("store","update","retrieve","erase","skip")
ACTIONS=("noop","remember","recall","forget","list_memories")
WORKFLOWS=(
    ("store","retrieve"),
    ("update","retrieve"),
    ("erase","retrieve"),
    ("store","erase"),
    ("update","erase"),
    ("skip","retrieve"),
)
MODELS=("DOCS","LEARNED","SHUFFLED")
IDS=re.compile(r"^\[([0-9a-f-]{36})\] ")
SOURCE_COMMIT="f6bd6239bfc1734c04c233a5cf8cef050c6d54c2"
ORDER=("noop","recall","remember","forget","list_memories")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,item):
    Path(path).write_text(json.dumps(item,ensure_ascii=False,
                                    sort_keys=True,separators=(",",":")))

def scenario(seed,kind,arm=""):
    # Texts are deliberately anonymous and synthetic; all use unique tags.
    tag=f"F10R{seed}"
    target=f"Para el ensayo sintetico {tag} utilizamos Panel Azul."
    other=f"El control sintetico {tag} mantiene el Indicador Gris."
    old=f"Mi color favorito es azul{tag}."
    new=f"Mi color favorito es verde{tag}."
    question=f"¿Que sabemos sobre el ensayo {tag}?"
    return {
        "seed":seed,"tag":tag,"target":target,"other":other,
        "old":old,"new":new,"question":question,
        "query":tag,"kind":kind,
    }

def objective(intent,sc):
    if intent=="store":
        return {"intent":intent,"statement":sc["target"],
                "target":sc["target"],"query":sc["tag"]}
    if intent=="update":
        return {"intent":intent,"statement":sc["new"],
                "target":sc["old"],"new":sc["new"],
                "old":sc["old"],"query":sc["tag"]}
    if intent=="retrieve":
        fact=sc["other"] if sc["kind"] in (2,5) else (
            sc["new"] if sc["kind"] in (1,4) else sc["target"])
        return {"intent":intent,"target":fact,"statement":fact,
                "query":sc["tag"] if fact!=sc["new"] else "color favorito"}
    if intent=="erase":
        fact=sc["new"] if sc["kind"]==4 else sc["target"]
        return {"intent":intent,"target":fact,"statement":fact,
                "query":sc["tag"],"preserve":sc["other"]}
    if intent=="skip":
        return {"intent":intent,"statement":sc["question"],
                "target":sc["question"],"query":sc["tag"]}
    raise ValueError(intent)

class Session:
    def __init__(self,client):
        self.client=client
        self.tool_calls=0
        self.agent_calls=0
        self.logs=[]

    async def call(self,name,params=None,*,agent=True):
        started=time.perf_counter()
        response=await self.client.call_tool(name,params or {})
        assert not response.is_error
        assert response.content
        result=str(response.content[0].text)
        self.tool_calls+=1
        if agent:self.agent_calls+=1
        self.logs.append({
            "name":name,
            "args":params or {},
            "response":result,
            "agent":agent,
            "ms":round((time.perf_counter()-started)*1000,3),
        })
        return result

    async def list(self,*,agent=False):
        return await self.call("list_memories",{},agent=agent)

    async def remember(self,statement,*,agent=False):
        return await self.call("remember",{"statement":statement},agent=agent)

    async def apply(self,action,goal):
        before=self.agent_calls
        response=""
        if action=="noop":
            pass
        elif action=="remember":
            response=await self.remember(goal["statement"],agent=True)
        elif action=="recall":
            response=await self.call("recall",{"query":goal["query"]})
        elif action=="list_memories":
            response=await self.list(agent=True)
        elif action=="forget":
            # Safety wrapper: never delete on an intent that did not request it,
            # and resolve exact original text to the public MCP-provided UUID.
            if goal["intent"]=="erase":
                listing=await self.list(agent=True)
                ids=[]
                for line in listing.splitlines():
                    match=IDS.match(line)
                    if match and '"'+goal["target"]+'"' in line:
                        ids.append(match.group(1))
                if ids:
                    response=await self.call("forget",{
                        "memory_ids":ids,"keep":[]})
            # If no exact permission and no exact ID, no deletion is allowed.
        else:
            raise ValueError(action)
        return response,self.agent_calls-before

def contains(text,statement):
    return ('"'+statement+'"') in text

async def prepare(session,sc,workflow):
    """Identical initial data for every policy; never counted as agent."""
    assert (await session.remember(sc["other"])).startswith("Remembered")
    if sc["kind"] in (1,4):
        assert (await session.remember(sc["old"])).startswith("Remembered")
    if sc["kind"] in (2,5):
        assert (await session.remember(sc["target"])).startswith("Remembered")
    # store->erase deliberately starts WITHOUT target; the agent must store it.

def train_prep(kind,session,sc):
    # Coroutines are handled in train below, this returns statements to seed.
    return {
        "store":[sc["other"]],
        "update":[sc["old"],sc["other"]],
        "retrieve":[sc["target"],sc["other"]],
        "erase":[sc["target"],sc["other"]],
        "skip":[sc["other"]],
    }[kind]

async def success(session,goal,action,response):
    listing=await session.list(agent=False)
    intent=goal["intent"]
    if intent=="store":
        return contains(listing,goal["statement"])
    if intent=="update":
        return contains(listing,goal["new"]) and not contains(
            listing,goal["old"])
    if intent=="retrieve":
        return action=="recall" and contains(response,goal["target"])
    if intent=="erase":
        return (not contains(listing,goal["target"]) and
                contains(listing,goal["preserve"]))
    if intent=="skip":
        return not contains(listing,goal["statement"])
    raise ValueError(intent)

class OutcomeModel:
    def __init__(self,stats):
        self.stats=stats

    @classmethod
    def fit(cls,observed,permutation=None):
        data=[dict(x) for x in observed]
        if permutation is not None:
            rng=random.Random(permutation)
            labels=[x["success"] for x in data]
            rng.shuffle(labels)
            for x,y in zip(data,labels):
                x["success"]=y
        summary=defaultdict(lambda:{"success":0,"trials":0,"calls":0})
        for r in data:
            key=(r["intent"],r["action"])
            s=summary[key]
            s["success"]+=int(r["success"])
            s["trials"]+=1
            s["calls"]+=r["agent_calls"]
        assert len(summary)==25
        assert all(s["trials"]==4 for s in summary.values())
        return cls({f"{a}|{b}":v for (a,b),v in sorted(summary.items())})

    def choose(self,intent):
        def score(action):
            s=self.stats[f"{intent}|{action}"]
            return ((s["success"]+1)/(s["trials"]+2)
                    - .02*(s["calls"]/s["trials"]))
        return max(ORDER,key=lambda a:(score(a),-ORDER.index(a)))

def docs(intent):
    return {
        "store":"remember","update":"remember","retrieve":"recall",
        "erase":"forget","skip":"noop",
    }[intent]

async def run_training(rep,server_py,tmpdir,output):
    path=Path(tmpdir)/"TRAIN-mcp.db"
    params=StdioServerParameters(
        command=sys.executable,
        args=["-m","infinito3.mcp_server"],
        env={"INFINITO_MEMORY_DB":str(path),"PYTHONPATH":server_py})
    events=[]
    async with Client(params) as client:
        listed=sorted(x.name for x in (await client.list_tools()).tools)
        assert set(["remember","recall","forget","list_memories"])<=set(listed)
        agent=Session(client)
        n=0
        # Each intent/action gets exactly four independent synthetic statement IDs.
        for intent in GOALS:
            for action in ACTIONS:
                for case in range(4):
                    seed=31_100_000+10_000*rep+10_000*GOALS.index(intent)+(
                        100*ACTIONS.index(action))+case
                    # Unique textual statement suffix even across categories.
                    sc=scenario(seed,0)
                    for fact in train_prep(intent,agent,sc):
                        await agent.remember(fact,agent=False)
                    goal=objective(intent,sc)
                    if intent=="erase":
                        goal["preserve"]=sc["other"]
                    response,calls=await agent.apply(action,goal)
                    good=bool(await success(agent,goal,action,response))
                    events.append({
                        "intent":intent,"action":action,"seed":seed,
                        "success":good,"agent_calls":calls,
                        "response":response,
                    })
                    n+=1
        assert n==100
        snap=await agent.list(agent=False)
        assert snap
        output["training_protocol_tools"]=listed
        output["training_call_count"]=agent.tool_calls
        output["training_agent_call_count"]=agent.agent_calls
        output["training_db_name"]=path.name
    return events

async def eval_policy(rep,arm,model,tmpdir):
    path=Path(tmpdir)/f"OOD-{arm}.db"
    params=StdioServerParameters(
        command=sys.executable,args=["-m","infinito3.mcp_server"],
        env={"INFINITO_MEMORY_DB":str(path),
             "PYTHONPATH":os.environ.get("F10R_PRODUCT_SRC","")})
    results=[]
    async with Client(params) as client:
        tools={x.name for x in (await client.list_tools()).tools}
        assert {"remember","recall","forget","list_memories"}<=tools
        session=Session(client)
        for j in range(48):
            seed=31_400_000+10_000*rep+j
            kind=j%6
            sc=scenario(seed,kind,arm)
            wf=WORKFLOWS[kind]
            await prepare(session,sc,wf)
            successes=[]
            actions=[]
            outputs=[]
            calls_start=session.agent_calls
            elapsed_start=time.perf_counter()
            deleted=[]
            collateral=[]
            for intent in wf:
                goal=objective(intent,sc)
                action=(docs(intent) if arm=="DOCS" else model.choose(intent))
                response,n_calls=await session.apply(action,goal)
                okay=bool(await success(session,goal,action,response))
                after=await session.list(agent=False)
                if intent=="erase":
                    deleted.append(goal["target"])
                    collateral.append(contains(after,sc["other"]))
                successes.append(okay)
                actions.append({"intent":intent,"action":action,
                                "agent_calls":n_calls})
                outputs.append({"response":response,"after":after})
            last=await session.list(agent=False)
            if sc["kind"]==4:
                # Original superseded fact must not return to active memory
                no_old=not contains(last,sc["old"])
                successes.append(no_old)
            if deleted:
                successes.append(all(not contains(last,d) for d in deleted))
            pass_all=all(successes)
            results.append({
                "replica":rep,"arm":arm,"j":j,"seed":seed,
                "kind":kind,"workflow":list(wf),
                "scenario_sha":hashlib.sha256(json.dumps(
                    {k:v for k,v in sc.items() if k!="arm"},
                    sort_keys=True).encode()).hexdigest(),
                "success":pass_all,"step_success":successes,
                "actions":actions,"agent_calls":session.agent_calls-calls_start,
                "elapsed_ms":round((time.perf_counter()-elapsed_start)*1000,3),
                "preserve_all":all(collateral),
                "deleted":deleted,
                "deleted_absent_active":all(not contains(last,d) for d in deleted),
                "question_unstored":not contains(last,sc["question"]),
                "final_listing_sha":hashlib.sha256(last.encode()).hexdigest(),
            })
        all_call_count=session.agent_calls
        tool_count=session.tool_calls
        # The evaluator is not an agent and may inspect DB once the MCP run
        # finishes. Its instrumented evidence is NEVER sent to OutcomeModel.
    with sqlite3.connect(path) as conn:
        content=[str(row[0]) for row in conn.execute("SELECT content FROM memories")]
        purged_total=conn.execute(
            "SELECT COALESCE(SUM(purged_count),0) FROM forget_log").fetchone()[0]
    for row in results:
        if row["deleted"]:
            row["deleted_absent_database"]=all(
                text not in content for text in row["deleted"])
        else:
            row["deleted_absent_database"]=True
        # Updating then forgetting must also purge superseded old history.
        row["old_lineage_absent_database"]=(
            row["kind"]!=4 or scenario(row["seed"],4)["old"] not in content)
    return {
        "rows":results,"total_agent_calls":all_call_count,
        "total_tool_calls_including_evaluator":tool_count,
        "total_purges":int(purged_total),
        "db_file":path.name,
    }

async def experiment(rep,product,out):
    out.mkdir(parents=True,exist_ok=False)
    assert product.exists()
    # Master product source is checked out by SHA in CI; never use local
    # INFINITO_MEMORY_DB inherited from user's environment.
    product_src=str(product/"src")
    os.environ["F10R_PRODUCT_SRC"]=product_src
    begin=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="f10r-native-mcp-") as d:
        metadata={"replica":rep,"product_commit":SOURCE_COMMIT,
                  "temp_db_only":True,
                  "purpose":"synthetic only, personal MCP not touched",
                  "mcp_server":"infinito3.mcp_server original",
                  "train_complete_before_ood":True}
        observed=await run_training(rep,product_src,d,metadata)
        save(out/"train_calls.json",observed)
        learned=OutcomeModel.fit(observed)
        shuffled=OutcomeModel.fit(observed,31_1900+rep)
        save(out/"LEARNED.json",learned.stats)
        save(out/"SHUFFLED.json",shuffled.stats)
        metadata["train_sha"]=sha(out/"train_calls.json")
        metadata["learned_sha"]=sha(out/"LEARNED.json")
        metadata["shuffled_sha"]=sha(out/"SHUFFLED.json")
        metadata["learner_choices"]={g:learned.choose(g) for g in GOALS}
        metadata["shuffled_choices"]={g:shuffled.choose(g) for g in GOALS}
        metadata["prereg_sha"]=sha(
            Path(__file__).parents[1]/
            "docs/INFINITO_F10R_PREREG_2026-11-04.md")
        metadata["source_sha"]=sha(__file__)
        metadata["train_cases"]=len(observed)
        metadata["model_locked_pre_ood"]=True
        save(out/"frozen.json",metadata)
        print("F10R_FROZEN_BEFORE_OOD",rep,metadata["learner_choices"],
              metadata["shuffled_choices"],flush=True)
        result={}
        for arm in MODELS:
            x=await eval_policy(
                rep,arm,None if arm=="DOCS" else
                learned if arm=="LEARNED" else shuffled,d)
            save(out/("eval_"+arm+".json"),x)
            result[arm]={
                "successes":sum(row["success"] for row in x["rows"]),
                "total":48,
                "agent_calls":x["total_agent_calls"],
                "purges":x["total_purges"]}
            print("F10R_OOD",rep,arm,result[arm],flush=True)
        assert all(sha(out/k)==v for k,v in (
            ("train_calls.json",metadata["train_sha"]),
            ("LEARNED.json",metadata["learned_sha"]),
            ("SHUFFLED.json",metadata["shuffled_sha"])))
        save(out/"complete.json",{
            "state":"F10R_ORIGINAL_OOD_COMPLETED_FROZEN",
            "frozen_sha":sha(out/"frozen.json"),
            "results":result,
            "elapsed_seconds":round(time.perf_counter()-begin,3),
            "no_production_mcp_used":True})
        print("F10R_COMPLETE",rep,result,flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--replica",type=int,required=True,choices=range(4))
    ap.add_argument("--product",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    asyncio.run(experiment(a.replica,a.product,a.out))

if __name__=="__main__":
    main()
