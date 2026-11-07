"""F10-S: authentic, unmodified INFINITO MCP across TWO independent processes.

Runs exactly 4 replicas x (6 families x 4 cases). Both client processes share
only one ephemeral SQLite DB *within* a replica. No production MCP, cloud
memory, vault, credentials, or real user data are ever used.

The client-side local cached response is an intentional NEGATIVE control:
a remote server cannot retroactively redact text already placed in an
unconnected model's context. New MCP reads MUST obey forget/update.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

from mcp import Client, StdioServerParameters

PRODUCT_SHA="2e563ddb194cb3a0193b77a22cc97575a8baba73"
BASE=32_100_000
N_REPS=4
FAMILIES=("visibility","supersession","selective_forget",
          "restart","concurrency","stale_context_revalidate")
ID_RE=re.compile(r"^\[([0-9a-f-]{36})\] ")
MEMORY_TOOLS=("remember","recall","list_memories","forget")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,
                                    indent=2))

def present(text,statement):
    return '"'+statement+'"' in text

def matching_id(listing,statement):
    matching=[]
    for line in listing.splitlines():
        m=ID_RE.match(line)
        if m and present(line,statement):
            matching.append(m.group(1))
    return matching

def db_rows(path):
    with sqlite3.connect(path,timeout=10) as db:
        memories=[(str(c),str(s)) for c,s in db.execute(
            "SELECT content,status FROM memories").fetchall()]
        purge=int(db.execute(
            "SELECT COALESCE(SUM(purged_count),0) FROM forget_log").fetchone()[0])
    return memories,purge

def active(rows,text):
    return sum(c==text and s=="active" for c,s in rows)

class Link:
    def __init__(self,client,name,trace):
        self.client=client
        self.name=name
        self.trace=trace
        self.calls=0

    async def call(self,tool,args=None):
        payload=args or {}
        res=await self.client.call_tool(tool,payload)
        result=str(res.content[0].text) if res.content else ""
        self.calls+=1
        self.trace.append({
            "agent":self.name,
            "tool":tool,
            "args":payload,
            "output":result,
            "is_error":bool(res.is_error),
        })
        if res.is_error:
            raise RuntimeError(f"{self.name}:{tool}: MCP is_error, {result[:200]}")
        return result

    async def remember(self,text,project=None):
        args={"statement":text}
        if project:
            args.update({"kind":"fact","source":"synthetic F10-S",
                         "project":project})
        return await self.call("remember",args)

    async def recall(self,query,project=None):
        args={"query":query}
        if project: args["project"]=project
        return await self.call("recall",args)

    async def list(self,project=None):
        return await self.call("list_memories",
                               {"project":project} if project else {})

    async def forget(self,ids):
        return await self.call("forget",{"memory_ids":ids,"keep":[]})

def process_config(path):
    assert path.is_absolute()
    return StdioServerParameters(
        command=sys.executable,
        args=["-m","infinito3.mcp_server"],
        env={
            "INFINITO_MEMORY_DB":str(path),
            "INFINITO_LOCAL_MEMORY":"on",
            # isolated stdio process; no remote endpoint or vault.
        },
    )

async def remembered(link,text,project=None):
    result=await link.remember(text,project)
    if not result.startswith("Remembered"):
        raise RuntimeError("MCP refused synthetic fact: "+result[:160])
    return result

async def exact_forget(link,listing,statement):
    ids=matching_id(listing,statement)
    if len(ids)!=1:
        raise ValueError("expected one exact target id; found "+str(len(ids)))
    response=await link.forget(ids)
    if not response.startswith("Deleted"):
        raise RuntimeError("forget not confirmed")
    return response

async def case_run(family,seed,a,b,db,trace,params):
    project=f"F10S{seed}"
    checks={}
    diagnostics={"family":family,"seed":seed,"project":project,
                 "checks":checks}
    def check(name,ok):
        checks[name]=bool(ok)
    if family=="visibility":
        fact=f"Para el proyecto sintetico {project} la referencia es Panel Cobre."
        check("write_ack",(await remembered(a,fact,project)).startswith("Remembered"))
        recalled=await b.recall(project,project)
        listed=await b.list(project)
        check("b_fresh_recall",present(recalled,fact))
        check("b_fresh_list_exactly_once",listed.count('"'+fact+'"')==1)
        rows,_=db_rows(db)
        check("active_in_sqlite",active(rows,fact)==1)
    elif family=="supersession":
        old=f"Mi color favorito es cobalto{seed}."
        new=f"Mi color favorito es esmeralda{seed}."
        await remembered(a,old)
        prior=await b.recall("color favorito")
        check("b_saw_prior",present(prior,old))
        update=await remembered(a,new)
        check("update_ack",update.startswith("Remembered"))
        current=await b.recall("color favorito")
        listed=await b.list()
        rows,_=db_rows(db)
        check("b_sees_new",present(current,new) and present(listed,new))
        check("b_not_old",not present(current,old) and not present(listed,old))
        check("sqlite_one_active_latest",
              active(rows,new)==1 and active(rows,old)==0)
        # Cleanup is OUTSIDE scored checks; avoid pollution of later cases.
        try:
            ids=matching_id(listed,new)
            if len(ids)==1: await a.forget(ids)
        except Exception as exc:
            diagnostics["cleanup_error"]=type(exc).__name__
    elif family=="selective_forget":
        target=f"Para el proyecto sintetico {project} debemos conservar Nota Carmesi."
        neighbor=f"Para el proyecto sintetico {project} debemos conservar Nota Marfil."
        await remembered(a,target,project)
        await remembered(a,neighbor,project)
        before=await b.recall(project,project)
        check("b_initially_saw_target",present(before,target))
        listing=await a.list(project)
        check("target_id_unique",len(matching_id(listing,target))==1)
        await exact_forget(a,listing,target)
        after=await b.recall(project,project)
        listed=await b.list(project)
        rows,purges=db_rows(db)
        check("no_target_fresh_read",not present(after,target) and
              not present(listed,target))
        check("neighbor_preserved",present(after,neighbor) and
              present(listed,neighbor) and active(rows,neighbor)==1)
        check("purged_target_from_all_db_rows",
              all(c!=target for c,status in rows))
        check("sqlite_purge_log_present",purges>=1)
    elif family=="restart":
        target=f"Para el proyecto sintetico {project} elegimos Referencia Menta."
        await remembered(a,target,project)
        pre=await b.recall(project,project)
        check("existing_agent_sees",present(pre,target))
        async with Client(params) as client_c:
            c=Link(client_c,"C-fresh",trace)
            cseen=await c.recall(project,project)
            check("new_process_sees",present(cseen,target))
        listing=await a.list(project)
        await exact_forget(a,listing,target)
        # This is a NEW stdio MCP process after C exits, proving a restart
        # observes the deletion instead of relying on in-process caches.
        async with Client(params) as client_d:
            d=Link(client_d,"D-restarted",trace)
            post=await d.recall(project,project)
            listed=await d.list(project)
            check("restarted_process_no_ghost",
                  not present(post,target) and not present(listed,target))
        rows,_=db_rows(db)
        check("sqlite_no_ghost",all(c!=target for c,status in rows))
    elif family=="concurrency":
        first=f"Para el proyecto sintetico {project} agente A eligio Senda Azafran."
        second=f"Para el proyecto sintetico {project} agente B eligio Senda Plata."
        results=await asyncio.gather(
            a.remember(first,project),b.remember(second,project),
            return_exceptions=True)
        check("both_parallel_writes_ack",all(
            isinstance(x,str) and x.startswith("Remembered")
            for x in results))
        listing=await b.list(project)
        rows,_=db_rows(db)
        check("two_distinct_writes_visible",
              present(listing,first) and present(listing,second) and
              active(rows,first)==1 and active(rows,second)==1)
        v1=f"Mi color favorito es granate{seed}."
        v2=f"Mi color favorito es dorado{seed}."
        simultaneous=await asyncio.gather(
            a.remember(v1),b.remember(v2),return_exceptions=True)
        check("two_concurrent_updates_ack",all(
            isinstance(x,str) and x.startswith("Remembered")
            for x in simultaneous))
        rows,_=db_rows(db)
        active_color=[c for c,s in rows if s=="active" and
                      (c==v1 or c==v2)]
        bview=await b.recall("color favorito")
        check("one_active_exclusive_version",
              len(active_color)==1 and
              sum(present(bview,v) for v in (v1,v2))==1)
        diagnostics["parallel_outputs"]=[
            type(x).__name__ if isinstance(x,BaseException) else str(x)[:180]
            for x in simultaneous]
        # Cleanup only, after scoring. Neither update may be rescued.
        try:
            listing_all=await a.list()
            for v in (v1,v2):
                ids=matching_id(listing_all,v)
                if ids: await a.forget(ids)
        except Exception as exc:
            diagnostics["cleanup_error"]=type(exc).__name__
    elif family=="stale_context_revalidate":
        target=f"Para el proyecto sintetico {project} el codigo era Nube Violeta."
        await remembered(a,target,project)
        cached=await b.recall(project,project)
        check("cache_initially_had_target",present(cached,target))
        listing=await a.list(project)
        await exact_forget(a,listing,target)
        # Old cached text is not a server defect: a prior result remains in
        # the client context unless client explicitly refreshes.
        check("old_context_still_contains_text_expected",present(cached,target))
        new=await b.recall(project,project)
        latest=await b.list(project)
        check("b_refresh_does_not_contain_deleted",
              not present(new,target) and not present(latest,target))
        rows,_=db_rows(db)
        check("database_deleted",all(c!=target for c,status in rows))
    else:
        raise ValueError(family)
    diagnostics["passed"]=all(checks.values())
    return diagnostics

async def run(rep,out):
    assert rep in range(N_REPS)
    out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix=f"infinito-f10s-r{rep}-") as tmp:
        db=Path(tmp).resolve()/"synthetic-shared.db"
        params=process_config(db)
        trace=[]
        cases=[]
        async with Client(params) as client_a:
            a=Link(client_a,"A",trace)
            # A is started first, so initial schema creation is not raced.
            tools_a=sorted(x.name for x in (await client_a.list_tools()).tools)
            async with Client(params) as client_b:
                b=Link(client_b,"B",trace)
                tools_b=sorted(x.name for x in (await client_b.list_tools()).tools)
                assert set(MEMORY_TOOLS)<=set(tools_a)&set(tools_b)
                for fi,family in enumerate(FAMILIES):
                    for j in range(4):
                        seed=BASE+10_000*rep+100*fi+j
                        start=len(trace)
                        try:
                            row=await case_run(family,seed,a,b,db,trace,params)
                        except Exception as error:
                            row={"family":family,"seed":seed,
                                 "checks":{},"passed":False,
                                 "error":repr(error)}
                        row["mcp_calls"]=len(trace)-start
                        cases.append(row)
                        print("F10S_CASE",rep,family,j,
                              "PASS" if row["passed"] else "FAIL",
                              row.get("error",""),flush=True)
        assert len(cases)==24
        rows,purges=db_rows(db)
        sqlite_summary={
            "active":sum(s=="active" for c,s in rows),
            "all_rows":len(rows),
            "purge_log_total":purges,
            "db_exists":db.exists(),
            "db_within_temporary_directory":str(db).startswith(str(Path(tmp))),
        }
        json_save(out/"cases.json",cases)
        with gzip.open(out/"mcp_trace.jsonl.gz","wt",encoding="utf8") as fh:
            for item in trace:
                fh.write(json.dumps(item,sort_keys=True,ensure_ascii=False)+"\n")
        metadata={
            "experiment":"INFINITO_F10S_REAL_MCP_TWO_PROCESS",
            "source_product_sha":"2e563ddb194cb3a0193b77a22cc97575a8baba73",
            "replica":rep,
            "protocol":"actual MCP stdio with 2 independent processes sharing one SQLite file",
            "db_user_memory_touched":False,
            "db_only_ephemeral":True,
            "cases":len(cases),
            "tools_a":tools_a,
            "tools_b":tools_b,
            "case_seed_range":[BASE+10_000*rep,
                               BASE+10_000*rep+503],
            "cases_sha256":digest(out/"cases.json"),
            "trace_sha256":digest(out/"mcp_trace.jsonl.gz"),
            "prereg_sha256":digest(
                Path(__file__).parents[1]/
                "docs/INFINITO_F10S_PREREG_2026-11-04.md"),
            "source_sha256":digest(__file__),
            "sqlite":sqlite_summary,
        }
        json_save(out/"manifest.json",metadata)
        print("F10S_REPLICA_END",rep,
              sum(int(c["passed"]) for c in cases),"/24",
              "calls",len(trace),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--replica",required=True,type=int,choices=range(N_REPS))
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    asyncio.run(run(args.replica,args.out))

if __name__=="__main__":
    main()
