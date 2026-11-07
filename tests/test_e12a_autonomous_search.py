"""E12-A contract tests: fake LLM / hash vectors are instrumentation only."""
import unittest
from dataclasses import replace

from src.infinito3.persistent_memory import HashEmbeddingProvider
from observatory.e12a_autonomous_search import (
    ARMS,CATEGORIES,REFERENCE_BUDGET,TOP_K,
    SemanticArchive,seed_for,make_world,question,fixed_search,
    evaluate_world,preregistered_criterion,summarize,
)

class FakeResult:
    def __init__(self,text):
        self.text=text
        self.usage={"input_tokens":20,"output_tokens":1,"total_tokens":21}
        self.response_id="MOCK-ONLY"
        self.model="mock-not-real"

class FakeLLM:
    def __init__(self):
        self.calls=0
    def ask(self,system,working,prompt,archive=""):
        self.calls+=1
        if "controlador de memoria" in system:
            return FakeResult("SEARCH" if "no_verificado" in working or
                              "valor" not in working else "ABSTENER")
        return FakeResult("ABSTENER")

class TestsE12AutonomousSearch(unittest.TestCase):
    def test_split_disjointness_and_balanced_families(self):
        dev={seed_for("dev",i,j) for i in range(4) for j in range(2)}
        confirm={seed_for("confirm",i,j) for i in range(4) for j in range(8)}
        self.assertEqual(len(dev),8)
        self.assertEqual(len(confirm),32)
        self.assertFalse(dev&confirm)
        self.assertEqual({make_world("dev",i,0).category for i in range(4)},
                         set(CATEGORIES))
        for args in (("confirm",0,8),("dev",4,0),("dev",0,2),
                     ("confirm",-1,1),("confirm",0,-1)):
            with self.assertRaises(ValueError):seed_for(*args)

    def test_no_oracle_leak_in_question_or_working(self):
        for i,cat in enumerate(CATEGORIES):
            for j in range(2):
                w=make_world("dev",i,j)
                self.assertEqual(w.category,cat)
                self.assertNotIn("SEARCH",question(w))
                self.assertNotIn(w.target_archive_value or "NONEXISTENT-VAL",question(w).split("Si no")[0].split("Si es")[0])
                self.assertEqual(len(w.archive),16 if cat!="unknown" else 15)
                matched=[r for r in w.archive if r.subject==w.subject]
                self.assertEqual(len(matched),0 if cat=="unknown" else 1)
                if cat=="stale":
                    self.assertIn("no_verificado",w.working)
                    self.assertNotIn(matched[0].value,w.working)
                if cat=="fresh":
                    self.assertIn("verificado",w.working)
                if cat=="unknown":self.assertEqual(w.expected,"ABSTENER")
                self.assertTrue(fixed_search(w) == (cat!="fresh"))

    def test_one_real_retriever_shared_same_budget(self):
        emb=HashEmbeddingProvider(dimensions=128)
        rag=SemanticArchive(emb)
        for i in range(4):
            w=make_world("dev",i,0)
            context,diag=rag.retrieve(w)
            self.assertLessEqual(diag["reference_tokens_estimate"],REFERENCE_BUDGET)
            self.assertLessEqual(diag["retrieved_records"],TOP_K)
            self.assertNotIn("INSTRUCCIÓN OCULTA",context)
            if i==3:self.assertFalse(diag["target_present"])

    def test_mock_harness_only_and_stats_without_invented_provider(self):
        fake=FakeLLM()
        emb=HashEmbeddingProvider(dimensions=128)
        rows=[]
        for i in range(4):
            w=make_world("dev",i,0)
            row=evaluate_world(w,SemanticArchive(emb),fake)
            self.assertEqual(set(row["outcomes"]),set(ARMS))
            self.assertGreater(fake.calls,0)
            self.assertTrue(all(x["llm_calls"] in (1,2) for x in row["outcomes"].values()))
            self.assertTrue(all(x["input_tokens"]>0 for x in row["outcomes"].values()))
            self.assertEqual(row["outcomes"]["always_search"]["searched"],1)
            self.assertEqual(row["outcomes"]["never_search"]["searched"],0)
            rows.append(row)
        stats=summarize(rows)
        self.assertEqual(stats["worlds"],4)
        self.assertEqual(set(stats["by_category"]),set(CATEGORIES))
        self.assertFalse(preregistered_criterion(stats)["passes"])

    def test_no_archive_read_by_router_before_search(self):
        # The router's first call signature only receives working and question;
        # future truth and full archive are carried in World but not supplied.
        from observatory.e12a_autonomous_search import LLM
        self.assertNotIn("archive_records",LLM.ask.__code__.co_varnames)

if __name__=="__main__":
    unittest.main()
