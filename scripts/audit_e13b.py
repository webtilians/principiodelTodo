"""Independent E13-B evidence auditor: no imports from E13-B, E13-A,
their generator, products, indexes or model. Only JSON artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib,json,math,re
from pathlib import Path

ARMS=("A_CURRENT","B_SETS","C_RELATIONS","D_HYBRID")
PRIMARY=("current","past","timeline","relation","contradiction")
def digest(data):
    return hashlib.sha256(json.dumps(
        data,ensure_ascii=False,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()

def parse(text):
    m=re.search(r"Entidad (\w+) (\w+): (\w+)\.",text)
    if not m:m=re.search(r"Para (\w+), (\w+) consta como (\w+)\.",text)
    if not m:raise AssertionError(f"unparseable evidence {text}")
    return m.groups()

def interpret(query,lines):
    m=re.search(r"entidad (\w+)",query)
    assert m,query
    subject=m[1]
    attr=next(a for a in ("estado","responsable","herramienta","ruta") if a in query)
    max_date=re.search(r"\d{4}-\d{2}-\d{2}",query)
    until=max_date[0] if max_date else "9999-12-31"
    facts=[(x["date"],*parse(x["text"])) for x in lines
           if x["kind"] in ("fact","decision","agreement") and x["date"]<=until]
    def recent(entity,attribute):
        a=[(d,v) for d,e,k,v in facts if e==entity and k==attribute]
        if not a:return ()
        last=max(d for d,v in a)
        return tuple(sorted(set(v for d,v in a if d==last)))
    if "cronologia" in query:
        return tuple(sorted(set(f"{d}:{v}" for d,e,k,v in facts
                                if e==subject and k==attr)))
    if "del responsable" in query:
        return tuple(sorted(set(v for owner in recent(subject,"responsable")
                                for v in recent(owner,"herramienta"))))
    return recent(subject,attr)

def audit(folder,disjoint=None):
    folder=Path(folder)
    summary=json.loads((folder/"summary.json").read_bytes())
    world_seeds=set()
    score={arm:Counter() for arm in ARMS}
    files=0;queries=0
    for name,expected in summary["artifacts"].items():
        raw=(folder/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==expected
        w=json.loads(raw)
        assert name==f'world-{w["replica"]}-{w["world"]}.json'
        ns=f'INFINITO/E13-B/v1/{w["partition"]}/{w["replica"]}/{w["world"]}'
        wanted=str(int.from_bytes(hashlib.sha256(ns.encode()).digest(),"big"))
        assert w["seed"]==wanted and wanted not in world_seeds
        world_seeds.add(wanted)
        assert w["partition"]==summary["partition"]
        assert w["project"]!=w["other_project"]
        sources={s["sid"]:s for s in w["sources"]}
        assert len(sources)==len(w["sources"])
        assert digest(w["sources"])==w["data_sha256"]
        deleted=set(w["delete_ids"])
        assert set(w["arms"])==set(ARMS)
        for arm,a in w["arms"].items():
            assert a["integrity"] and a["tenant_isolated"]
            assert len(a["queries"])>=40
            for index_stage in ("index_before","index_after"):
                index=a[index_stage]
                if index is None:
                    assert arm=="A_CURRENT"
                    continue
                assert index["schema"]=="E13B_REF_ONLY_V1"
                assert index["mode"]==arm
                assert set(index["source_ids"])<=set(sources)
                if index_stage=="index_after":
                    assert not deleted.intersection(index["source_ids"])
                for key,ids in list(index["groups"].items())+list(
                        index["temporal"].items()):
                    project,entity,attr=key.split("/")
                    for sid in ids:
                        assert sid in index["source_ids"]
                        s=sources[sid]
                        assert s["project"]==project
                        assert parse(s["text"])[:2]==(entity,attr)
                        if index_stage=="index_after":
                            assert sid not in deleted
            archive={s["sid"]:s for s in a["archive_after"]}
            assert set(archive)==set(sources)-deleted
            assert all(s==sources[sid] for sid,s in archive.items())
            assert not any(sid in deleted for sid in archive)
            for db_record in a["database_after"]:
                assert db_record["metadata"]["source"] not in deleted
            for q in a["queries"]:
                query_context=q["context"]
                lines=[json.loads(item) for item in query_context.splitlines()[1:]]
                assert math.ceil(len(query_context)/4)<=q["budget"]
                assert q["tokens"]<=q["budget"]
                assert q["provenance_valid"] is True
                for line in lines:
                    assert line["project"]==w["project"], "cross-tenant leak"
                    assert line["ids"]
                    for sid in line["ids"]:
                        source=sources[sid]
                        assert line["project"]==source["project"]
                        assert (parse(line["text"]),line["date"],line["kind"])==(
                            parse(source["text"]),source["date"],source["kind"])
                        if "phase" in q:assert sid not in deleted
                got=interpret(q["query"],lines)
                assert list(got)==q["answer"],(arm,q["query"],got,q["answer"])
                assert q["correct"]==(q["answer"]==q["expected"])
                queries+=1
                if q["budget"]==512 and q["family"] in PRIMARY:
                    score[arm]["total"]+=1
                    score[arm]["correct"]+=int(q["correct"])
            tenant=[json.loads(x) for x in a["tenant_context"].splitlines()[1:]]
            assert all(x["project"]==w["other_project"] for x in tenant)
        files+=1
    if disjoint:
        old=audit(disjoint)
        assert not world_seeds.intersection(old["seeds"])
    for arm in ARMS:
        assert score[arm]["correct"]==summary["summary"]["systems"][arm]["correct"]
        assert score[arm]["total"]==summary["summary"]["systems"][arm]["total"]
    return {"partition":summary["partition"],"worlds":files,"queries":queries,
            "scores":{a:dict(score[a]) for a in ARMS},
            "seeds":sorted(world_seeds),
            "integrity":"E13B_INDEPENDENT_ARTIFACT_AUDIT_PASS"}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("folder",type=Path)
    ap.add_argument("--disjoint",type=Path)
    args=ap.parse_args()
    print(json.dumps(audit(args.folder,args.disjoint),indent=2))
