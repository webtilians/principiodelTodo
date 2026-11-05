"""E13-B: isolated A/B/C/D memory-organization ablation, built on the real
E13-A AgentMemory/ranker harness. Research only; never a real user database.

A = current AgentMemory + project filter; B = entity/attribute sets only;
C = fact-source temporal references + explicit two-hop graph (no grouping);
D = B+C with incremental source-level indexes and safe deletion.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib,json,math,platform,random,re,statistics,subprocess,sys,tempfile,time
from pathlib import Path

from observatory import e13a
from src.infinito3.mcp_server import AgentMemory
from src.infinito3.semantic import MeaningRanker

ROOT=Path(__file__).resolve().parents[1]
ARMS=("A_CURRENT","B_SETS","C_RELATIONS","D_HYBRID")
PRIMARY=("current","past","timeline","relation","contradiction")
CONTROL=("forgotten","survivor","hypothesis","unknown")
BUDGETS=(512,256)
MODEL_NAME="minishlab/potion-multilingual-128M"

def seed(partition,rep,world):
    return int.from_bytes(hashlib.sha256(
        f"INFINITO/E13-B/v1/{partition}/{rep}/{world}".encode()
    ).digest(),"big")

def world_data(partition,rep,world):
    # Reuse the E13-A syntactic family, never its consumed seeds/cases.
    # e13a.world_data reads e13a.seed from module globals; restore instantly.
    old=e13a.seed
    e13a.seed=lambda p,r,w:seed(partition,rep,world)
    try:
        sources,probes,deletion,forgotten=e13a.world_data(
            "E13B_"+partition,rep,world)
    finally:
        e13a.seed=old
    primary=sources[0].project
    other="TENANT_B_"+primary
    # Poison another project's records with contradictory values for the SAME
    # subjects; the reader must never see their IDs when querying primary.
    observed=set()
    extras=[]
    for s in sources:
        parsed=e13a.parse(s.text)
        if not parsed or parsed[1]!="estado" or parsed[0] in observed:
            continue
        observed.add(parsed[0])
        extras.append(e13a.Source(
            "tenant_"+str(len(extras)),
            f"{other}: Entidad {parsed[0]} estado: ajeno{len(extras)}.",
            "2024-01-30","fact",other))
        if len(extras)==4:break
    assert len(extras)==4
    sources=sources+extras
    return sources,probes,deletion,forgotten,primary,other

def project_query(query):
    match=re.search(r'entidad (\w+)',query)
    entity=match[1] if match else None
    field=next((s for s in ("estado","responsable","herramienta","ruta")
                if s in query),"estado")
    if "del responsable" in query:field="responsable"
    date=re.search(r"\d{4}-\d{2}-\d{2}",query)
    return entity,field,date[0] if date else "9999-12-31"

class DerivedIndex:
    """Reference-only index, no generated facts, no oracle labels."""
    def __init__(self,arm):
        self.arm=arm
        self.sources={}
        self.group=defaultdict(list)
        self.temporal=defaultdict(list)
        self.maintenance_cpu=0.
        self.maintenance_wall=0.
        self.operations=0
        self.rebuilds=0

    def add(self,s):
        cpu,wall=time.process_time(),time.perf_counter()
        assert s.sid not in self.sources
        self.sources[s.sid]=s
        parsed=e13a.parse(s.text)
        if parsed:
            entity,field,value=parsed
            key=(s.project,entity,field)
            if self.arm in ("B_SETS","D_HYBRID"):
                self.group[key].append(s.sid)
            if self.arm in ("C_RELATIONS","D_HYBRID"):
                self.temporal[key].append(s.sid)
        self.operations+=1
        self.maintenance_cpu+=time.process_time()-cpu
        self.maintenance_wall+=time.perf_counter()-wall

    def forget(self,ids):
        cpu,wall=time.process_time(),time.perf_counter()
        dead=set(ids)
        for sid in dead:self.sources.pop(sid,None)
        for store in (self.group,self.temporal):
            for key in list(store):
                store[key]=[x for x in store[key] if x not in dead]
                if not store[key]:del store[key]
        self.operations+=len(dead)
        self.maintenance_cpu+=time.process_time()-cpu
        self.maintenance_wall+=time.perf_counter()-wall

    def snapshot(self):
        def output(data):
            return {"/".join(key):list(ids) for key,ids in sorted(data.items())}
        # Store source references only. No duplicate factual text.
        return {"mode":self.arm,"groups":output(self.group),
                "temporal":output(self.temporal),
                "source_ids":sorted(self.sources),
                "schema":"E13B_REF_ONLY_V1"}

    def integrity(self):
        for data in (self.group,self.temporal):
            for (project,entity,field),ids in data.items():
                for sid in ids:
                    src=self.sources.get(sid)
                    if not src or src.project!=project:return False
                    parsed=e13a.parse(src.text)
                    if not parsed or parsed[:2]!=(entity,field):return False
        return True

    def candidates(self,query,ranker,project):
        entity,attr,until=project_query(query)
        source_ids=[]
        data=(self.group if self.arm=="B_SETS" else self.temporal)
        main=[self.sources[x] for x in data.get((project,entity,attr),())
              if x in self.sources]
        # Recency is only a retrieval preference. Proof still comes from
        # the original source dates/kinds and shared evidence-only reader.
        main.sort(key=lambda s:(s.date,s.sid),reverse=True)
        main=[s for s in main if s.date<=until and s.kind!="proposal"]
        source_ids.extend(x.sid for x in main)
        if self.arm in ("C_RELATIONS","D_HYBRID") and "del responsable" in query:
            owners={e13a.parse(s.text)[2] for s in main}
            # Explicit two-step typed edges from original owner facts to
            # original tool facts: NOT inferred from test answers.
            for owner in sorted(owners):
                linked=[self.sources[x] for x in self.temporal.get(
                    (project,owner,"herramienta"),()) if x in self.sources]
                linked.sort(key=lambda s:(s.date,s.sid),reverse=True)
                source_ids.extend(s.sid for s in linked if s.kind!="proposal"
                                  and s.date<=until)
        # Rank remaining evidence from the SAME tenant, never query gold.
        remaining=[s for s in self.sources.values() if s.project==project
                   and s.sid not in set(source_ids)]
        order=ranker.rank(query,[s.text for s in remaining])
        source_ids.extend(remaining[i].sid for i in order)
        output=[]
        # D groups duplicate factual records sharing date/kind/tuple; B and
        # C keep all original individual evidence, even literal duplicates.
        seen={}
        for sid in source_ids:
            s=self.sources[sid]
            if self.arm=="D_HYBRID":
                key=(s.project,s.date,s.kind,e13a.parse(s.text) or s.text)
                if key in seen:
                    output[seen[key]][1].append(sid)
                    continue
                seen[key]=len(output)
            output.append((s,[sid]))
        return output

class MemoryArm:
    def __init__(self,arm,folder,ranker):
        self.arm=arm
        self.ranker=ranker
        self.path=Path(folder)
        self.db=self.path/"memory.sqlite"
        self.agent=AgentMemory(str(self.db),ranker=ranker)
        self.records={}
        self.mapping={}
        self.index=None if arm=="A_CURRENT" else DerivedIndex(arm)
        self.cpu=Counter()
        self.wall=Counter()
        self.latencies=[]
        self.add_count=0
        self.forget_count=0
        self.index_path=self.path/"derived-index.json"
        self.raw_path=self.path/"sources.jsonl"

    def persist(self):
        if self.index:
            self.index_path.write_bytes(e13a.encoded(self.index.snapshot()))

    def add(self,s):
        cpu,wall=time.process_time(),time.perf_counter()
        result=self.agent.remember(
            s.text,kind=s.kind,source=s.sid,project=s.project,date=s.date)
        assert result.startswith("Remembered as "),result
        self.mapping[s.sid]=result.split()[2].rstrip(".")
        self.records[s.sid]=s
        if self.index:
            self.index.add(s)
            with self.raw_path.open("ab") as f:
                f.write(e13a.encoded(asdict(s))+b"\n")
            self.persist()
        self.cpu["ingest"]+=time.process_time()-cpu
        self.wall["ingest"]+=time.perf_counter()-wall
        self.add_count+=1

    def query(self,query,budget,project):
        cpu,wall=time.process_time(),time.perf_counter()
        if self.index:
            candidates=self.index.candidates(query,self.ranker,project)
        else:
            result=self.agent.recall(query,limit=20,project=project)
            ids=re.findall(r"^\[([^]]+)\]",result,re.M)
            candidates=[]
            for mid in ids:
                item=self.agent.store.get(mid)
                s=e13a.Source(item.metadata["source"],item.content,
                    item.metadata["said_at"][:10],item.metadata["kind"],
                    item.metadata["project"])
                candidates.append((s,[s.sid]))
            order=self.ranker.rank(query,[s.text for s,p in candidates])
            candidates=[candidates[i] for i in order]
        selected,context,tokens=e13a.pack(candidates,budget)
        answer=e13a.read_answer(query,selected)
        self.cpu["query"]+=time.process_time()-cpu
        elapsed=time.perf_counter()-wall
        self.wall["query"]+=elapsed
        self.latencies.append(elapsed)
        assert all(s.project==project for s,ids in selected)
        return answer,context,tokens,selected

    def forget(self,ids):
        cpu,wall=time.process_time(),time.perf_counter()
        self.agent.forget(sorted(self.mapping[sid] for sid in ids))
        for sid in ids:
            self.records.pop(sid,None)
        if self.index:
            self.index.forget(ids)
            self.rewrite_raw()
            self.persist()
        self.cpu["forget"]+=time.process_time()-cpu
        self.wall["forget"]+=time.perf_counter()-wall
        self.forget_count+=len(ids)

    def rewrite_raw(self):
        self.raw_path.write_bytes(b"".join(
            e13a.encoded(asdict(s))+b"\n" for s in self.records.values()))

    def restart(self):
        cpu,wall=time.process_time(),time.perf_counter()
        self.agent.close()
        self.agent=AgentMemory(str(self.db),ranker=self.ranker)
        if self.index:
            raw=[json.loads(x) for x in self.raw_path.read_bytes().splitlines()]
            self.records={x["sid"]:e13a.Source(**x) for x in raw}
            self.index=DerivedIndex(self.arm)
            for s in self.records.values():self.index.add(s)
            self.index.rebuilds+=1
            self.persist()
        self.cpu["restart"]+=time.process_time()-cpu
        self.wall["restart"]+=time.perf_counter()-wall

    def storage(self):
        sqlite_bytes=len(self.agent.store.to_bytes())
        raw=self.raw_path.stat().st_size if self.raw_path.exists() else 0
        index=self.index_path.stat().st_size if self.index_path.exists() else 0
        return {"sqlite_bytes":sqlite_bytes,"raw_bytes":raw,
                "index_bytes":index,"total_bytes":sqlite_bytes+raw+index}

    def close(self):
        self.agent.close()

def run_world(partition,rep,world,ranker):
    sources,probes,deletion,forgotten,project,other=world_data(
        partition,rep,world)
    ranker._cache.clear()
    cpu,wall=time.process_time(),time.perf_counter()
    for t in dict.fromkeys([s.text for s in sources]+
                           [q.query for q in probes+forgotten]):
        ranker.vector(t)
    embed_cpu=time.process_time()-cpu
    embed_wall=time.perf_counter()-wall
    result={
        "partition":partition,"replica":rep,"world":world,
        "seed":str(seed(partition,rep,world)),
        "project":project,"other_project":other,
        "sources":[asdict(s) for s in sources],
        "delete_ids":sorted(deletion),
        "data_sha256":e13a.digest([asdict(s) for s in sources]),
        "arms":{},
    }
    for arm in ARMS[world%4:]+ARMS[:world%4]:
        with tempfile.TemporaryDirectory(prefix="infinito-e13b-synthetic-") as tmp:
            model=MemoryArm(arm,tmp,ranker)
            for s in sources:model.add(s)
            # Freeze the materialized index before BOTH budget sensitivities:
            # no usage-activated rebuilding between 512 and 256.
            before=model.index.snapshot() if model.index else None
            integrity=not model.index or model.index.integrity()
            st=model.storage()
            rows=[]
            for budget in BUDGETS:
                for q in probes:
                    got,context,tokens,selected=model.query(q.query,budget,project)
                    rows.append({
                        "family":q.family,"query":q.query,
                        "budget":budget,"expected":q.expected,
                        "answer":got,"correct":got==q.expected,
                        "context":context,"tokens":tokens,
                        "provenance_valid":all(
                            sid in model.records and model.records[sid].project==project
                            for _,ids in selected for sid in ids),
                    })
            # A query in another tenant must not surface primary project IDs.
            tenant_query="estado actual entidad "+e13a.parse(sources[0].text)[0]
            tenant_ans,tenant_ctx,tenant_tokens,tenant_selection=model.query(
                tenant_query,512,other)
            tenant_clean=all(s.project==other for s,ids in tenant_selection)
            model.forget(deletion)
            after_forget=[]
            survivors=[e13a.Probe("survivor",p.query,p.expected) for p in probes[5:10]]
            for phase in ("immediate","restart"):
                if phase=="restart":model.restart()
                for q in forgotten+survivors:
                    got,ctx,tok,selected=model.query(q.query,512,project)
                    after_forget.append({
                        "family":q.family,"phase":phase,"query":q.query,
                        "budget":512,"expected":q.expected,"answer":got,
                        "correct":got==q.expected,"context":ctx,"tokens":tok,
                        "provenance_valid":all(
                            sid in model.records for _,ids in selected for sid in ids),
                    })
            rows.extend(after_forget)
            integrity=integrity and tenant_clean and (
                not model.index or model.index.integrity())
            index_after=model.index.snapshot() if model.index else None
            forgotten_ids=set(deletion)
            if index_after:
                assert not forgotten_ids.intersection(index_after["source_ids"])
            result["arms"][arm]={
                "queries":rows,"storage":st,
                "after_forget_storage":model.storage(),
                "index_before":before,"index_after":index_after,
                "database_after":[{"text":r.content,"metadata":r.metadata}
                    for r in model.agent.store.all(True)],
                "archive_after":[asdict(s) for s in model.records.values()],
                "tenant_context":tenant_ctx,
                "tenant_isolated":tenant_clean,
                "integrity":integrity,
                "cpu_s":dict(model.cpu),"wall_s":dict(model.wall),
                "embedding_cpu_s":embed_cpu,"embedding_wall_s":embed_wall,
                "query_latencies_s":model.latencies,
                "query_p95_s":e13a.percentile(model.latencies,.95),
                "index_maintenance_cpu_s":model.index.maintenance_cpu if model.index else 0,
                "index_maintenance_wall_s":model.index.maintenance_wall if model.index else 0,
                "index_operations":model.index.operations if model.index else 0,
                "index_rebuilds":model.index.rebuilds if model.index else 0,
            }
            model.close()
    return result

def percentile(values,fraction):
    return e13a.percentile(values,fraction)

def summarize(worlds):
    systems={}
    for arm in ARMS:
        queries=[q for w in worlds for q in w["arms"][arm]["queries"]]
        primary=[q for q in queries if q["budget"]==512 and q["family"] in PRIMARY]
        by={}
        for f in PRIMARY+CONTROL:
            rows=[q for q in primary if q["family"]==f] if f in PRIMARY else [
                q for q in queries if q["family"]==f]
            if rows:by[f]={"correct":sum(q["correct"] for q in rows),
                           "total":len(rows)}
        systems[arm]={
            "correct":sum(q["correct"] for q in primary),
            "total":len(primary),
            "accuracy":sum(q["correct"] for q in primary)/len(primary),
            "by_family":by,
            "mean_storage_bytes":statistics.mean(
                w["arms"][arm]["storage"]["total_bytes"] for w in worlds),
            "total_cpu_s":sum(
                sum(w["arms"][arm]["cpu_s"].values())+
                w["arms"][arm]["embedding_cpu_s"] for w in worlds),
            "query_p95_s":percentile([t for w in worlds
                   for t in w["arms"][arm]["query_latencies_s"]],.95),
            "index_operations":sum(w["arms"][arm]["index_operations"] for w in worlds),
            "index_maintenance_cpu_s":sum(
                w["arms"][arm]["index_maintenance_cpu_s"] for w in worlds),
        }
    def rate(arm,f):
        x=systems[arm]["by_family"][f]
        return x["correct"]/x["total"]
    contrasts={}
    for against in ("A_CURRENT","B_SETS","D_HYBRID"):
        per=[]
        for w in worlds:
            def metric(a):
                x=[q for q in w["arms"][a]["queries"]
                   if q["budget"]==512 and q["family"]=="relation"]
                return sum(q["correct"] for q in x)/len(x)
            per.append(metric("C_RELATIONS")-metric(against))
        rand=random.Random(1313)
        boot=[statistics.mean(rand.choices(per,k=len(per))) for _ in range(5000)]
        contrasts["C_minus_"+against]={
            "delta":statistics.mean(per),
            "ci95":[percentile(boot,.025),percentile(boot,.975)],
            "world_deltas":per}
    a,c,d=systems["A_CURRENT"],systems["C_RELATIONS"],systems["D_HYBRID"]
    safe=all(arm["integrity"] and arm["tenant_isolated"]
             and all(q["provenance_valid"] and q["tokens"]<=q["budget"]
                     for q in arm["queries"])
             and all(q["correct"] for q in arm["queries"]
                     if q["family"] in CONTROL)
             for w in worlds for arm in w["arms"].values())
    g0=safe and len({w["seed"] for w in worlds})==len(worlds)
    g1=rate("C_RELATIONS","relation")>=.9 and all(
        contrasts["C_minus_"+base]["delta"]>=.10 and
        contrasts["C_minus_"+base]["ci95"][0]>0
        for base in ("A_CURRENT","B_SETS"))
    g2=all(rate(arm,"current")>=rate("A_CURRENT","current")-.02
           for arm in ("C_RELATIONS","D_HYBRID"))
    g3=safe
    g4=(c["mean_storage_bytes"]<=1.35*a["mean_storage_bytes"] and
        c["total_cpu_s"]<=2*a["total_cpu_s"] and
        c["query_p95_s"]<=2*a["query_p95_s"])
    return {
        "systems":systems,"contrasts":contrasts,
        "gates":{"G0":g0,"G1":g1,"G2":g2,"G3":g3,"G4":g4},
        "G5_real_llm":"PENDING_INDEPENDENT_VALIDATION",
        "verdict":"E13B_INCONCLUSIVE_LLM" if all((g0,g1,g2,g3,g4))
                  else "E13B_FAIL" if g0 else "E13B_INCONCLUSIVE",
    }

def fingerprints(paths):
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(
        p.read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in paths}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--partition",choices=("DEV","CONFIRM","LLM"),required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--freeze",type=Path)
    args=ap.parse_args()
    files=[ROOT/p for p in (
        "observatory/e13b.py",
        "tests/test_e13b.py",
        "scripts/audit_e13b.py",
        "scripts/e13b_llm.py",
        "docs/INFINITO_E13B_PREREG_2026-11-05.md")]
    hashes=fingerprints(files)
    model_hashes={x.name:hashlib.sha256(x.read_bytes()).hexdigest()
                  for x in args.model.iterdir() if x.is_file()}
    if args.partition!="DEV":
        assert args.freeze is not None, "CONFIRM and LLM require committed freeze"
        manifest=json.loads(args.freeze.read_text())
        assert hashes==manifest["files"],"code changed after DEV freeze"
        assert model_hashes==manifest["model_files"],"semantic model changed after DEV freeze"
        rel=args.freeze.resolve().relative_to(ROOT).as_posix()
        committed=subprocess.check_output(
            ["git","show","HEAD:"+rel],cwd=ROOT)
        assert json.loads(committed)==manifest,"freeze must be committed before data"
    if args.output.exists():
        ap.error("refuse to overwrite consumed partition")
    args.output.mkdir(parents=True)
    ranker=MeaningRanker(str(args.model))
    ranker.warm()
    worlds=[]
    pairs=[(0,i) for i in range(8)] if args.partition=="DEV" else (
        [(r,w) for r in range(4) for w in range(8)] if args.partition=="CONFIRM"
        else [(r,0) for r in range(8)])
    for rep,world in pairs:
        record=run_world(args.partition,rep,world,ranker)
        worlds.append(record)
        (args.output/f"world-{rep}-{world}.json").write_bytes(e13a.encoded(record))
        print("E13B",args.partition,rep,world,"COMPLETE",flush=True)
    report={"partition":args.partition,"files":hashes,
            "model_files":model_hashes,
            "commit":subprocess.check_output(["git","rev-parse","HEAD"],
                       cwd=ROOT,text=True).strip(),
            "environment":{"python":sys.version,"platform":platform.platform()},
            "api_calls":0,
            "summary":summarize(worlds),
            "artifacts":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in args.output.glob("world-*.json")}}
    (args.output/"summary.json").write_bytes(e13a.encoded(report))
    print("E13B_REPORT",json.dumps(report["summary"],ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
