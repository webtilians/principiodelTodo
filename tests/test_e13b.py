"""E13-B study contracts, all using synthetic sources and no LLM/network."""
import json

import pytest

from observatory.e13a import Source,pack,read_answer,parse
from observatory.e13b import (
    ARMS,DerivedIndex,MemoryArm,seed,world_data,project_query)


class LexicalRanker:
    _cache={}
    def rank(self,query,texts):
        terms=set(query.lower().split())
        return sorted(range(len(texts)),
                      key=lambda i:(-len(terms & set(texts[i].lower().split())),i))
    def vector(self,text):
        return (len(text),)
    def warm(self):
        pass


def fact(sid,entity,attr,value,date="2024-01-01",project="TENANT_A",kind="fact"):
    return Source(sid,f"{project}: Entidad {entity} {attr}: {value}.",
                  date,kind,project)


def test_e13b_partitions_are_disjoint_from_e13a_and_each_other():
    dev={seed("DEV",0,w) for w in range(8)}
    confirm={seed("CONFIRM",r,w) for r in range(4) for w in range(8)}
    llm={seed("LLM",r,0) for r in range(8)}
    assert len(dev)==8 and len(confirm)==32 and len(llm)==8
    assert not dev&confirm and not llm&confirm and not llm&dev
    assert seed("DEV",0,0)!=seed("E13A_DEV",0,0)


def test_e13b_worlds_contain_second_tenant_and_stable_reference():
    sources,probes,deleted,forgotten,primary,other=world_data("DEV",0,0)
    assert len(sources)>190 and len(probes)>=20
    assert primary!=other
    assert sum(s.project==other for s in sources)==4
    assert sources==world_data("DEV",0,0)[0]
    assert all(sid.startswith("s") for sid in deleted)


@pytest.mark.parametrize("arm",ARMS[1:])
def test_indexes_reference_original_sources_only_and_purge(arm):
    index=DerivedIndex(arm)
    inputs=[
        fact("p","lima","responsable","toro"),
        fact("q","toro","herramienta","llave"),
        fact("a","lima","estado","antes"),
        fact("b","lima","estado","despues","2024-01-20"),
        fact("x","lima","estado","ajeno","2024-01-21",project="TENANT_B"),
    ]
    before=json.dumps([s.__dict__ for s in inputs],sort_keys=True)
    for s in inputs:index.add(s)
    assert index.integrity()
    candidates=index.candidates(
        "estado actual entidad lima",LexicalRanker(),"TENANT_A")
    assert all(s.project=="TENANT_A" for s,ids in candidates)
    if arm in ("C_RELATIONS","D_HYBRID"):
        selected,context,tokens=pack(
            index.candidates(
                "herramienta del responsable entidad lima",
                LexicalRanker(),"TENANT_A"),512)
        assert read_answer(
            "herramienta del responsable entidad lima",selected)==("llave",)
    index.forget({"q","a"})
    assert index.integrity()
    snap=index.snapshot()
    assert "q" not in snap["source_ids"]
    assert "a" not in snap["source_ids"]
    assert all("q" not in x and "a" not in x
               for x in list(snap["groups"].values())+
               list(snap["temporal"].values()))
    assert json.dumps([s.__dict__ for s in inputs],sort_keys=True)==before


@pytest.mark.parametrize("arm",ARMS)
def test_product_recall_and_generated_index_share_tenant_and_purge(
        tmp_path,arm):
    system=MemoryArm(arm,tmp_path,LexicalRanker())
    main=[
        fact("a","lima","estado","rojo"),
        fact("b","lima","estado","verde","2024-01-20"),
        fact("c","lima","responsable","toro"),
        fact("d","toro","herramienta","martillo"),
        fact("other","lima","estado","morado",project="TENANT_B"),
    ]
    for s in main:system.add(s)
    assert system.query("estado actual entidad lima",512,"TENANT_A")[0]==("verde",)
    assert system.query("estado actual entidad lima",512,"TENANT_B")[0]==("morado",)
    system.forget({"b"})
    system.restart()
    got,context,tokens,selected=system.query(
        "estado actual entidad lima",512,"TENANT_A")
    assert got==("rojo",)
    assert all(s.project=="TENANT_A" for s,ids in selected)
    if system.index:
        assert system.index.integrity()
        assert "b" not in system.index.snapshot()["source_ids"]
    assert all(r.content!=main[1].text for r in system.agent.store.all(True))
    system.close()


def test_temporal_reader_latest_and_proposals_never_promoted():
    sources=[
        fact("old","lima","estado","rojo"),
        fact("new","lima","estado","verde","2024-01-20"),
        fact("claim","lima","estado","azul","2024-01-25",kind="proposal")]
    index=DerivedIndex("C_RELATIONS")
    for s in sources:index.add(s)
    selected,context,tokens=pack(index.candidates(
        "estado actual entidad lima",LexicalRanker(),"TENANT_A"),512)
    assert read_answer("estado actual entidad lima",selected)==("verde",)
    assert tokens<=512
