"""New E10-C development-only contracts; no LLM and no E10-B confirmation seeds."""
import tempfile
import unittest
from pathlib import Path

from observatory.e10c_pipeline_diagnostic import (
    AUDIT_ARMS,DEV_SEED_BASE,FAMILIES, STAGES,
    aggregate,evaluate_world,fresh_world,seed_for,
)
from observatory.e10b_semantic_longitudinal_live import paired_history, question
from src.infinito3.persistent_memory import HashEmbeddingProvider


class TestE10CPipelineContracts(unittest.TestCase):
    def test_fresh_diagnostic_seeds_and_all_families(self):
        seeds={seed_for(i,j) for i in range(2) for j in range(6)}
        self.assertEqual(len(seeds),12)
        self.assertEqual(min(seeds),1900000)
        self.assertEqual(max(seeds),1901005)
        self.assertTrue(all(s not in range(1800000,1804006) for s in seeds))
        self.assertEqual({fresh_world(seed_for(0,j)).family for j in range(6)},set(FAMILIES))
        for bad in ((2,0),(0,6),(-1,0)):
            with self.assertRaises(ValueError):seed_for(*bad)
        with self.assertRaises(ValueError):fresh_world(1800000)

    def test_twins_alter_only_fact_and_not_question_or_distractors(self):
        for j in range(6):
            w=fresh_world(seed_for(0,j))
            a,b=paired_history(w,False),paired_history(w,True)
            self.assertEqual(a[0][1:],b[0][1:])
            self.assertEqual(a[1][1:],b[1][1:])
            self.assertNotEqual(a[0][0],b[0][0])
            self.assertNotEqual(a[1][0],b[1][0])
            self.assertEqual(question(w),question(w))

    def test_pipeline_trace_by_arm_uses_only_present_evidence(self):
        # This is an instrumentation test with HASH, not E10-C semantic result.
        w=fresh_world(seed_for(0,0))  # unstructured new alerts
        d=evaluate_world(w,HashEmbeddingProvider(dimensions=128))
        self.assertEqual(len(d["observations"]),4)
        for row in d["observations"]:
            self.assertEqual(set(row["arms"]),set(AUDIT_ARMS))
            for arm,x in row["arms"].items():
                self.assertIn(x["first_missing_stage"],STAGES)
                self.assertLessEqual(x["context_tokens"],320)
                self.assertEqual(x["transmitted"],x["first_missing_stage"]=="transmitted")
                self.assertTrue(x["gate"]["reasons"])
                if x["selected"]:
                    self.assertTrue(x["retrieved"])
                if x["retrieved"]:
                    self.assertTrue(x["active"])
            self.assertEqual(row["arms"]["infinito3"]["first_missing_stage"],"not_captured")
            self.assertFalse(row["arms"]["infinito3"]["gate"]["should_store"])
            self.assertTrue(row["arms"]["infinito3_gate_open"]["persisted"])
            self.assertTrue(row["arms"]["semantic_rag"]["persisted"])

    def test_aggregate_rejects_incomplete_worlds(self):
        with self.assertRaises(AssertionError):
            aggregate([])


if __name__=="__main__":
    unittest.main()
