"""E11-B prospective contracts. Fake reader and hash vectors ONLY in tests."""
import re
import unittest

from src.infinito3.persistent_memory import HashEmbeddingProvider
from observatory.e11b_bounded_semantic_llm import (
    MEMORY_CAPACITY,EVENT_COUNT,ARMS,
    BoundedSemanticRAG,make_world,seed_for,variant_events,expected,question,
    evaluate_world,uniform_retention,criterion,RealSemanticEmbeddingCache,
    CONFIRM_REPLICAS,
)


class TestE11B(unittest.TestCase):
    def test_disjoint_new_partitions_and_independent_query(self):
        dev={seed_for("dev",0,j) for j in range(4)}
        conf={seed_for("confirm",i,j) for i in range(5) for j in range(3)}
        uniform={seed_for("uniform",i,j) for i in range(5) for j in range(400)}
        self.assertEqual(len(dev),4)
        self.assertEqual(len(conf),15)
        self.assertEqual(len(uniform),2000)
        self.assertFalse(dev&conf or dev&uniform or conf&uniform)
        for args in (("dev",1,0),("confirm",5,0),("confirm",0,3),
                     ("uniform",0,400),("train",0,0)):
            with self.assertRaises(ValueError):
                seed_for(*args)

    def test_world_variants_differ_only_target_value_and_update_is_opposite(self):
        for j in range(4):
            w=make_world("dev",0,j,"predictive")
            b1,a1=variant_events(w,False)
            b2,a2=variant_events(w,True)
            self.assertEqual(len(b1),EVENT_COUNT)
            self.assertEqual(len(a1),EVENT_COUNT)
            self.assertEqual(len({e.key for e in b1}),EVENT_COUNT)
            self.assertEqual(len({e.key for e in a1}),EVENT_COUNT)
            self.assertEqual([e for e in b1 if e.key!=w.target_key],
                             [e for e in b2 if e.key!=w.target_key])
            self.assertEqual([e for e in a1 if e.key!=w.target_key],
                             [e for e in a2 if e.key!=w.target_key])
            self.assertNotEqual(expected(w,False,False),expected(w,True,False))
            self.assertNotEqual(expected(w,False,True),expected(w,True,True))
            self.assertNotEqual(expected(w,False,False),expected(w,False,True))
            self.assertEqual(question(w),question(w))
            self.assertIn(w.subject,question(w))
            self.assertTrue(all("¿" not in e.text for e in b1))

    def test_policy_is_bounded_and_reference_does_not_include_evicted_data(self):
        w=make_world("dev",0,0,"predictive")
        fake=HashEmbeddingProvider(dimensions=128)
        model={"learned_weights":[0.7,0.8,1.2,0.2,-0.3,0.35,1.6]}
        for name in ARMS:
            p=BoundedSemanticRAG(name,w.seed,model,fake)
            for e in w.before:
                p.ingest(e)
                self.assertLessEqual(len(p.memory.saved),MEMORY_CAPACITY)
            context,trace=p.reference(question(w))
            self.assertEqual(trace["memory_count"],4)
            self.assertEqual(trace["selected_count"],4)
            self.assertLessEqual(trace["context_tokens"],260)
            self.assertEqual(set(trace["stored_keys"]),set(p.memory.saved))
            for e in w.before:
                if e.key not in p.memory.saved:
                    self.assertNotIn(e.subject,context)
            for e in w.after:
                p.ingest(e)
            self.assertEqual(len(p.memory.saved),4)
            self.assertEqual(len(set(p.memory.saved)),4)

    def test_uniform_stress_uses_no_queries_or_llm(self):
        # No weights fetched or trained with future outcomes in this test.
        model={"learned_weights":[0.7,0.8,1.2,0.2,-0.3,0.35,1.6]}
        models={str(i):model for i in range(CONFIRM_REPLICAS)}
        # Check one uniform world without evaluating all 2000 fixtures in tests.
        w=make_world("uniform",0,0,"uniform")
        self.assertEqual(w.regime,"uniform")
        self.assertNotIn(w.target_key,w.before[0].text)
        self.assertNotIn(w.target_key,question(w))
        self.assertEqual(len(w.before),16)

    def test_realtime_reader_mock_no_science_label(self):
        class FakeLLM:
            def decide(self,q,ctx):
                return {"action":"ABSTENER","raw":"ABSTENER",
                        "response_id":"TEST","provider_model":"mock-not-real",
                        "usage":{"total_tokens":0}}
        fake=HashEmbeddingProvider(dimensions=128)
        models={str(i):{"learned_weights":[0.7,0.8,1.2,0.2,-0.3,0.35,1.6]}
                for i in range(5)}
        w=make_world("dev",0,0,"predictive")
        row=evaluate_world(w,models,fake,FakeLLM())
        self.assertEqual(set(row["arms"]),set(ARMS))
        self.assertEqual(set(row["variants"]),{"original","reversed"})
        self.assertEqual(row["arms"]["rag_learned"]["all_four_correct"],0)
        self.assertEqual(row["arms"]["rag_learned"]["individual_correct"],0)
        self.assertEqual(set(row["variants"]["original"]["first"]),set(ARMS))
        self.assertTrue(all(x["memory_count"]==4 for x in row["variants"]["original"]["first"].values()))
        self.assertTrue(all(x["context_tokens"]<=260 for v in row["variants"].values()
                            for step in ("first","second") for x in v[step].values()))

    def test_gate_must_check_shift_and_predictive(self):
        stable={"arms":{"rag_fifo":{"pair_rate":0.3},
                        "rag_reservoir":{"pair_rate":0.2},
                        "rag_fixed_priority":{"pair_rate":0.5},
                        "rag_learned":{"pair_rate":0.9}},
                "replicas":list(range(5)),
                "by_replica":{str(i):{"rag_fifo":0.3,"rag_reservoir":0.2,
                                      "rag_fixed_priority":0.5,"rag_learned":0.9}
                              for i in range(5)}}
        changed={"arms":{"rag_learned":{"pair_rate":0.05},
                          "rag_reservoir":{"pair_rate":0.3}}}
        controls={a:{"abstained":4} for a in ARMS}
        gate=criterion(stable,changed,controls)
        self.assertTrue(gate["passes_predictive"])
        self.assertFalse(gate["passes_shift_safety"])
        self.assertFalse(gate["overall_progress"])


if __name__=="__main__":
    unittest.main()
