"""E10-B preconfirm contracts: DEV-only fixtures with fake API ONLY in unit tests.

The live worker explicitly refuses missing API credentials and uses the actual
OpenAI SDK; this test-only mock is never part of prospective confirmation.
"""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from src.infinito3.llm_adapter import RecordingLLMAdapter
from src.infinito3.persistent_memory import HashEmbeddingProvider
from observatory.e10b_semantic_longitudinal_live import (
    ACTIONS,ARMS,DEV_BASE,CONFIRM_BASE,FAMILIES,CONTEXT_BUDGET,
    CachedSemanticEmbeddings, ExperimentArm, ChronologicalSemanticRAG,
    OrderedEvidenceContext, evaluate_world, evaluate_negatives, make_world,
    paired_history,question,answer,seed_for,parse_action,aggregate,criterion,
)


class FakeAPI:
    """Only a dependency-injection stub in local tests: NOT live results."""
    class Embeddings:
        def __init__(self):
            self.provider=HashEmbeddingProvider(dimensions=128)
        def create(self,input,model):
            assert model=="text-embedding-3-small"
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=self.provider.embed(input[0]))])
    def __init__(self):
        self.embeddings=self.Embeddings()


def mocked_adapter():
    return RecordingLLMAdapter(responder=lambda request:"ABSTENER",
                               provider="unit-test-fake",model="not-a-real-llm")


class E10BContracts(unittest.TestCase):
    def test_partition_nonoverlap_and_world_six_families(self):
        dev={seed_for("dev",0,j) for j in range(6)}
        confirm={seed_for("confirm",i,j) for i in range(4) for j in range(6)}
        self.assertEqual(dev,set(range(DEV_BASE,DEV_BASE+6)))
        self.assertEqual(len(confirm),24)
        self.assertEqual(min(confirm),CONFIRM_BASE)
        self.assertEqual(max(confirm),CONFIRM_BASE+3005)
        self.assertFalse(dev&confirm)
        self.assertEqual({make_world(s).family for s in dev},set(FAMILIES))
        for args in (("dev",1,0),("dev",0,6),("confirm",4,0),
                     ("confirm",0,6),("unknown",0,0)):
            with self.assertRaises(ValueError):
                seed_for(*args)

    def test_twins_same_probe_and_noise_opposite_memory_cue(self):
        for j in range(6):
            w=make_world(seed_for("dev",0,j))
            normal=paired_history(w,False)
            inverted=paired_history(w,True)
            self.assertEqual(normal[0][1:],inverted[0][1:])
            self.assertEqual(normal[1][1:],inverted[1][1:])
            self.assertNotEqual(normal[0][0],inverted[0][0])
            self.assertNotEqual(normal[1][0],inverted[1][0])
            self.assertNotEqual(answer(w,w.options[0]),answer(w,w.options[1]))
            self.assertEqual(len(normal[0]),13)
            self.assertEqual(len(normal[1]),13)
            # Explicit interrogative prevents raw-message gate from storing
            # the decision question itself as a new memory.
            self.assertIn("?",question(w))
            self.assertIn("ABSTENER",question(w))
            self.assertNotIn("expected_",question(w))

    def test_common_normalization_fixed_without_hidden_labels(self):
        self.assertEqual(parse_action("  faro. \n"),"FARO")
        self.assertEqual(parse_action("«Nube»"),"NUBE")
        self.assertEqual(parse_action("FARO or NUBE"),"INVALIDO")
        self.assertEqual(parse_action("I think FARO"),"INVALIDO")
        self.assertEqual(parse_action(""),"INVALIDO")

    def test_four_arms_budget_reopen_and_api_requests_dev_only(self):
        client=FakeAPI()
        emb=CachedSemanticEmbeddings(client)
        w=make_world(seed_for("dev",0,0))
        case=evaluate_world(w,client,emb,mocked_adapter)
        self.assertEqual(set(case["arms"]),set(ARMS))
        self.assertEqual(set(case["variants"]),{"original","reversed"})
        self.assertEqual(case["arms"]["no_memory"]["four_correct"],0)
        self.assertGreater(emb.calls,0)
        for variant in case["variants"].values():
            for stage in ("first","second"):
                for arm in ARMS:
                    item=variant[stage][arm]
                    self.assertEqual(item["action"],"ABSTENER")
                    self.assertLessEqual(item["estimated_context_tokens"],CONTEXT_BUDGET)
                    self.assertEqual(item["provider_model"],"not-a-real-llm")
                    self.assertNotIn("expected_",item["context"])

    def test_gate_open_ablation_stores_non_structured_but_core_does_not(self):
        client=FakeAPI()
        emb=CachedSemanticEmbeddings(client)
        w=make_world(seed_for("dev",0,0))
        sentence=paired_history(w,False)[0][0]
        with tempfile.TemporaryDirectory() as d:
            a=ExperimentArm("infinito3",client,emb,Path(d)/"core",mocked_adapter)
            b=ExperimentArm("infinito3_gate_open",client,emb,Path(d)/"ablation",mocked_adapter)
            c=ExperimentArm("semantic_rag",client,emb,Path(d)/"rag",mocked_adapter)
            try:
                a.ingest(sentence);b.ingest(sentence);c.ingest(sentence)
                self.assertEqual(len(a.engine.memory_store.all()),0)
                self.assertEqual(len(b.engine.memory_store.all()),1)
                self.assertEqual(len(c.engine.memory_store.all()),1)
                b.restart()
                self.assertEqual(len(b.engine.memory_store.all()),1)
                self.assertEqual(len(c.engine.memory_store.all()),1)
            finally:
                a.close();b.close();c.close()

    def test_negative_controls_all_abstain_in_fake_unit_test(self):
        client=FakeAPI()
        emb=CachedSemanticEmbeddings(client)
        controls=evaluate_negatives(client,emb,mocked_adapter)
        self.assertEqual(len(controls),6)
        for item in controls:
            self.assertEqual(set(item["actions"]),set(ARMS))
            self.assertTrue(all(x["abstained"]==1 for x in item["actions"].values()))

    def test_descriptive_criterion_does_not_equate_semantic_to_hash(self):
        stats={
            "by_family":{f:{"infinito3":1.0,"semantic_rag":0.25}
                         for f in FAMILIES},
            "arms":{"infinito3":{"missing_evidence_abstentions":6},
                    "semantic_rag":{"missing_evidence_abstentions":6}},
        }
        self.assertTrue(criterion(stats)["e10b_passes"])
        for f in FAMILIES:
            stats["by_family"][f]["infinito3"]=0
        self.assertFalse(criterion(stats)["e10b_passes"])


if __name__=="__main__":
    unittest.main()
