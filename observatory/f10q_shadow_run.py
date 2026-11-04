"""F10-Q: train unmodified F10-P EventEffects from ACTUAL MCP/SQLite outputs,
then freeze before creating synthetic OOD; shadow-only evaluation.

Requires mcp>=2,<3 and NumPy. Tool is SELECT-only on temporary SQLite.
"""
from __future__ import annotations
import argparse,asyncio,gzip,hashlib,json,os,sqlite3,sys,tempfile,time
from collections import Counter
from pathlib import Path

import numpy as np
from mcp import Client,StdioServerParameters

from observatory.f10q_sqlite_worlds import fixtures,install
from observatory.f10p_events import EventEffects,project

TRAIN_PROGRAMS=(("P",),("P","X"),("S","R"),("S","R","P"),
                ("P","X","P"),("S","P","X"),("S","R","R"),("X",))
ARMS=("E100","E50","E0","E_noX","E_shuffleX")
ALL_ARMS=ARMS+("Frozen_F10P","Zero")
SQL_P="SELECT CAST(? AS INTEGER) AS value"
SQL_S="SELECT id FROM evidence WHERE case_id=? AND accredited=1 ORDER BY revision DESC,id DESC LIMIT 1"
SQL_S_BAD="SELECT id FROM evidence WHERE case_id=? ORDER BY revision DESC,id DESC LIMIT 1"
SQL_R="SELECT payload FROM evidence WHERE id=?"
SQL_X="SELECT CAST(1-? AS INTEGER) AS value"
SQL_X_DISABLED="SELECT CAST(? AS INTEGER) AS value"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,obj):
    Path(path).write_text(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False))

class ShadowSQL:
    def __init__(self,client):
        self.client=client
        self.calls=0
        self.errors=0
        self.writes=0

    async def query(self,sql,params,record):
        assert sql.strip().lower().startswith("select ")
        result=await self.client.call_tool(
            "sql_select",{"sql":sql,"params":params})
        assert not result.is_error
        assert len(result.content)==1
        raw=result.content[0].text
        data=json.loads(raw)
        record.append({"tool":"sql_select","sql":sql,"params":params,
                       "raw":raw})
        self.calls+=1
        if "error" in data:
            self.errors+=1
            return None
        assert set(data)=={"columns","rows"}
        assert isinstance(data["columns"],list)
        assert isinstance(data["rows"],list)
        return data

    async def execute(self,case,seq,mode="normal"):
        """The planner does NOT see case rows, truth, winner, or case_id."""
        work=None
        selected=None
        events=[]
        calls=[]
        abstain=False
        for op in seq:
            if op=="P":
                raw=await self.query(SQL_P,[int(case["public_bit"])],calls)
                if raw is None or not raw["rows"]:
                    events.append({"op":"P","output":{"error":True}})
                    work=None
                else:
                    value=raw["rows"][0][0]
                    if value not in (0,1):
                        events.append({"op":"P","output":{"error":True}})
                        work=None;abstain=True;break
                    work=int(value)
                    events.append({"op":"P","output":{"value":work}})
            elif op=="S":
                raw=await self.query(SQL_S_BAD if mode=="no_accredit" else SQL_S,
                                     [case["case_id"]],calls)
                if raw is None or not raw["rows"]:
                    selected=None
                    events.append({"op":"S","output":{"error":True}})
                    abstain=True;break
                selected=int(raw["rows"][0][0])
                events.append({"op":"S","output":{"ack":True}})
            elif op=="R":
                if selected is None:
                    events.append({"op":"R","output":{"error":True}})
                    work=None
                else:
                    raw=await self.query(SQL_R,[selected],calls)
                    if raw is None or not raw["rows"] or raw["rows"][0][0] not in (0,1):
                        events.append({"op":"R","output":{"error":True}})
                        work=None;abstain=True;break
                    work=int(raw["rows"][0][0])
                    events.append({"op":"R","output":{"value":work}})
            elif op=="X":
                if work is None:
                    events.append({"op":"X","output":{"error":True}})
                else:
                    raw=await self.query(
                        SQL_X_DISABLED if mode=="x_disabled" else SQL_X,
                        [work],calls)
                    if raw is None or not raw["rows"] or raw["rows"][0][0] not in (0,1):
                        events.append({"op":"X","output":{"error":True}})
                        work=None;abstain=True;break
                    work=int(raw["rows"][0][0])
                    events.append({"op":"X","output":{"value":work}})
            else:
                raise ValueError(op)
        # Missing/NUL reads are abstentions, not fabricated binary zeros.
        answer=None if abstain else (0 if work is None else work)
        return {"answer":answer,"abstain":abstain,
                "events":events,"calls":calls,"tool_calls":len(calls)}

async def run(rep,original_dir,out):
    out.mkdir(parents=True,exist_ok=False)
    original_meta=json.loads((original_dir/"frozen.json").read_text())
    snapshot=original_dir/"E100.json"
    assert sha(snapshot)==original_meta["trained_memory_files"]["E100"]["sha256"]
    frozen_old=EventEffects.from_snapshot(json.loads(snapshot.read_text()))
    assert frozen_old.episodes==98_304
    start=time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="f10q-mcp-sqlite-") as work:
        db=Path(work)/"synthetic.db"
        conn=sqlite3.connect(db)
        # TRAIN fixture only here; OOD rows MUST NOT exist until models frozen.
        training=fixtures("train",rep)
        assert len(training)==512
        install(conn,training)
        conn.close()
        server=StdioServerParameters(
            command=sys.executable,
            args=["-m","observatory.f10q_sqlite_mcp"],
            env={"F10Q_SQLITE_DB":str(db)})
        async with Client(server) as client:
            listed=await client.list_tools()
            assert "sql_select" in [x.name for x in listed.tools]
            sql=ShadowSQL(client)
            rejected=await client.call_tool("sql_select",{
                "sql":"INSERT INTO evidence(case_id) VALUES('forbidden')",
                "params":[]})
            assert not rejected.is_error
            assert json.loads(rejected.content[0].text)=={
                "error":"READ_ONLY_SELECT_REQUIRED"}
            assert sqlite3.connect(db).execute(
                "SELECT COUNT(*) FROM evidence WHERE case_id='forbidden'"
            ).fetchone()[0]==0
            models={a:EventEffects() for a in ARMS}
            rng=np.random.default_rng(291700+rep)
            mask=np.random.default_rng(291800+rep)
            shuffle=np.random.default_rng(291900+rep)
            trial_count=0
            with gzip.open(out/"train_mcp.jsonl.gz","wt",encoding="utf8") as f:
                for case in training:
                    for _ in range(2):
                        seq=TRAIN_PROGRAMS[int(rng.integers(len(TRAIN_PROGRAMS)))]
                        assert seq!=("S","R","X")
                        executed=await sql.execute(case,seq)
                        projected={
                            a:project(executed["events"],a,mask,shuffle)
                            for a in ARMS}
                        for a in ARMS:
                            models[a].observe(projected[a],int(case["public_bit"]))
                        trial_count+=1
                        f.write(json.dumps({
                            "trial":trial_count,"seed":case["seed"],
                            "variant":case["variant"],"case_sha":case["case_sha"],
                            "source":case["task"],"flip":case["flip"],
                            "public_bit":case["public_bit"],
                            "program":"".join(seq),
                            "answer":executed["answer"],
                            "tool_calls":executed["calls"],
                            "events":executed["events"],
                            "projected":projected,
                        },separators=(",",":"))+"\n")
            assert trial_count==1024
            assert all(model.episodes==1024 for model in models.values())
            assert sql.calls>=1000
            calls_at_freeze=sql.calls
            for arm,model in models.items():
                save(out/(arm+".json"),model.snapshot())
            meta={
                "experiment":"F10-Q SQLite real MCP shadow",
                "replica":rep,
                "state":"F10Q_ALL_MEMORIES_FROZEN_BEFORE_OOD_GENERATED",
                "training_trials":trial_count,
                "train_unique_cases":512,
                "train_tool_calls":calls_at_freeze,
                "train_programs":["".join(x) for x in TRAIN_PROGRAMS],
                "never_executed_SRX":True,
                "readonly_insert_rejected":True,
                "mcp_tools":[t.name for t in listed.tools],
                "mcp_protocol":"real_stdio_client_and_server",
                "original_f10p_E100_sha":sha(snapshot),
                "original_f10p_frozen_sha":sha(original_dir/"frozen.json"),
                "prereg_sha":sha(
                    Path(__file__).parents[1]/
                    "docs/INFINITO_F10Q_PREREG_2026-11-04.md"),
                "source_sha":sha(__file__),
                "train_log_sha":sha(out/"train_mcp.jsonl.gz"),
                "snapshot_sha":{a:sha(out/(a+".json")) for a in ARMS},
                "frozen_at_seconds":time.perf_counter()-start,
            }
            save(out/"frozen.json",meta)
            print("F10Q_SHA_FROZEN_PRE_OOD",rep,trial_count,calls_at_freeze,
                  flush=True)
            # Deliberately build OOD cases ONLY after frozen.json is on disk.
            ood=fixtures("ood",rep)
            assert len(ood)==512
            conn=sqlite3.connect(db)
            install(conn,ood)
            conn.close()
            loaded={a:EventEffects.from_snapshot(
                json.loads((out/(a+".json")).read_text())) for a in ARMS}
            loaded["Frozen_F10P"]=frozen_old
            allrows=[]
            for arm in ALL_ARMS:
                valid=0;correct=0;abstained=0
                for case in ood:
                    program=() if arm=="Zero" else loaded[arm].plan(
                        case["task"],case["flip"],case["public_bit"])[0]
                    if arm!="Zero":
                        assert program in (
                            (),)+tuple(__import__("observatory.f10p_worlds",fromlist=["PROGRAMS"]).PROGRAMS)
                    executed=await sql.execute(case,program)
                    outcomes={"normal":{
                        "answer":executed["answer"],
                        "abstain":executed["abstain"],
                        "success":(
                            None if case["truth"] is None or
                            executed["answer"] is None
                            else int(executed["answer"]==case["truth"])),
                        "calls":executed["calls"]}}
                    if arm=="E100":
                        for mode in ("x_disabled","no_accredit"):
                            intervention=await sql.execute(case,program,mode)
                            outcomes[mode]={
                                "answer":intervention["answer"],
                                "abstain":intervention["abstain"],
                                "success":(
                                    None if case["truth"] is None or
                                    intervention["answer"] is None else
                                    int(intervention["answer"]==case["truth"])),
                                "calls":intervention["calls"]}
                    allrows.append({
                        **{k:case[k] for k in (
                            "replica","seed","base_world","case_id","variant",
                            "case_sha","n","source_mode","task","flip",
                            "public_bit","invalid","truth")},
                        "arm":arm,"program":"".join(program) or "EMPTY",
                        "outcomes":outcomes,
                    })
                    if case["invalid"] is None:
                        valid+=1
                        correct+=int(outcomes["normal"]["success"]==1)
                    else:
                        abstained+=int(outcomes["normal"]["abstain"])
                print("F10Q_ARM",rep,arm,correct,valid,abstained,flush=True)
            assert len(allrows)==len(ALL_ARMS)*512==3584
            save(out/"ood_rows.json",allrows)
            assert all(sha(out/(a+".json"))==meta["snapshot_sha"][a]
                       for a in ARMS)
            assert sha(out/"train_mcp.jsonl.gz")==meta["train_log_sha"]
            assert sha(snapshot)==meta["original_f10p_E100_sha"]
            save(out/"complete.json",{
                "state":"F10Q_OOD_EVAL_COMPLETE_NO_MODEL_UPDATE",
                "frozen_sha":sha(out/"frozen.json"),
                "ood_sha":sha(out/"ood_rows.json"),
                "ood_cases":512,"ood_prediction_rows":len(allrows),
                "total_mcp_select_calls":sql.calls,
                "model_sha_still_immutable":True,
                "readonly_insert_rejected":True,
                "elapsed_seconds":time.perf_counter()-start,
            })
            print("F10Q_COMPLETE",rep,sql.calls,round(time.perf_counter()-start,2),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--replica",type=int,required=True,choices=range(4))
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    asyncio.run(run(a.replica,a.original,a.out))

if __name__=="__main__":
    main()
