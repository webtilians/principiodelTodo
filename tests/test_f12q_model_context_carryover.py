"""F12-Q preregistered model-context diagnostic engineering contracts.

Fake runs are never promoted to real scientific evidence.
"""
import json
import os
import unittest
from unittest.mock import patch
from observatory import f12q_model_context_carryover as q

class FakeFrozen:
    real_model=False
    instances=0
    def __init__(self):
        self.calls=0
        self.prompt_tokens=0
        self.completion_tokens=0
        self.inference_seconds=0.0
        self.generator=type("FakeGenerator",(),{
            "n_tokens":0,"cache":None,"n_ctx":lambda self:4096,
        })()
        type(self).instances+=1
    def chat(self,system,prompt,max_tokens=96):
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=4
        self.generator.n_tokens+=len(prompt)//4
        return '{"action":"q0"}'
    def metadata(self):
        return {
            "real_model":False,
            "weights_sha256":q.FIXED_MODEL_SHA256,
            "revision":"8911e8a47f92bac19d6f5c64a2e2095bd2f7d031",
            "backend_version":"0.3.36",
            "temperature":0.0,"n_threads":4,"n_ctx":4096,"max_tokens":96,
            "calls":self.calls,"prompt_tokens":self.prompt_tokens,
            "completion_tokens":self.completion_tokens,
            "inference_seconds":self.inference_seconds,
        }

class F12QContracts(unittest.TestCase):
    def test_source_anchor_sha_is_exact_old_o21(self):
        a=q.anchors_by_id()
        self.assertEqual(len(a),6)
        self.assertEqual(a["O21-start"]["prompt_sha256"],q.SOURCE_SHA)
        self.assertEqual(q.sha(q.SYSTEM),a["O21-start"]["system_sha256"])
        self.assertEqual(q.STUDY,"F12-Q")

    def test_factor_orthogonality_and_fixed_last_predecessor(self):
        self.assertEqual(len(q.CELLS),8)
        self.assertEqual(len(set(q.CELLS)),8)
        self.assertEqual(
            set(q.CELLS),
            {(o,p,i) for o in ("AB","BA")
             for p in (4,6) for i in ("reuse","fresh")})
        for o in ("AB","BA"):
            for p in (4,6):
                prefix=q.schedule(o,p)
                self.assertEqual(len(prefix),p-1)
                self.assertEqual(prefix[-1],"O0-disabled")
                self.assertEqual(
                    set(prefix[-3:-1]),{"O0-start","O7-start"})
                if p==6:
                    self.assertEqual(prefix[:2],("O7-disabled","O21-disabled"))
        self.assertNotEqual(q.schedule("AB",4),q.schedule("BA",4))
        with self.assertRaises(ValueError):
            q.cell_attributes(8)

    def test_reused_vs_fresh_target_has_identical_prompt_but_new_call_count(self):
        with patch.dict(os.environ,{"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"4"}):
            reused=q.run_cell(0,0,FakeFrozen)
            fresh=q.run_cell(1,0,FakeFrozen)
        self.assertEqual(reused["instance_mode"],"reuse")
        self.assertEqual(fresh["instance_mode"],"fresh")
        self.assertEqual(reused["prior_schedule"],fresh["prior_schedule"])
        self.assertEqual(reused["model_instance_init_count"],1)
        self.assertEqual(fresh["model_instance_init_count"],2)
        self.assertEqual(reused["target_call"]["prompt_sha256"],q.SOURCE_SHA)
        self.assertEqual(fresh["target_call"]["prompt_sha256"],q.SOURCE_SHA)
        self.assertEqual(reused["target_call"]["model_calls_before"],3)
        self.assertEqual(reused["target_call"]["model_calls_after"],4)
        self.assertEqual(fresh["target_call"]["model_calls_before"],0)
        self.assertEqual(fresh["target_call"]["model_calls_after"],1)
        self.assertEqual(len(reused["prior_calls"]),3)
        self.assertEqual(len(fresh["prior_calls"]),3)
        self.assertEqual(reused["target_call"]["raw"],fresh["target_call"]["raw"])
        self.assertFalse(reused["target_call"]["cache_before"]["kv_cache_hits_available"])
        self.assertIsNone(reused["target_call"]["cache_after"]["kv_cache_hits"])
        self.assertEqual(fresh["target_call"]["cache_before"]["n_tokens_observed"],0)
        self.assertGreater(reused["target_call"]["cache_before"]["n_tokens_observed"],0)

    def test_model_fake_does_not_meet_frozen_real_gate(self):
        with patch.dict(os.environ,{"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"4"}):
            workers=[]
            for w in (0,1):
                cells=[q.run_cell(i,w,FakeFrozen) for i in range(8)]
                workers.append({
                    "study":"F12-Q","worker":w,
                    "cell_order":(list(range(8)) if w==0 else list(reversed(range(8)))),
                    "cells":cells,
                })
            report=q.aggregate(workers)
        self.assertEqual(report["total_llm_calls"],80)
        self.assertEqual(len(report["factor_cells"]),8)
        self.assertEqual(report["status"],"INVALID_Q_PROTOCOL")
        self.assertFalse(report["gates"]["Q0_complete_16_independent_cells_80_real_calls"])
        self.assertFalse(report["gates"]["Q1_pinned_model_and_fresh_instantiation"])
        self.assertTrue(report["gates"]["Q2_identical_target_raw_between_workers_per_cell"])
        self.assertTrue(report["gates"]["Q3_fresh_target_is_invariant_across_predecessors"])
        self.assertFalse(report["kv_cache_hit_counts_available"])
        self.assertFalse(report["transfer_worlds_evaluated"])
        self.assertFalse(report["product_integrated"])
        with self.assertRaises(ValueError):
            q.aggregate(workers[:1])
        with self.assertRaises(ValueError):
            q.aggregate([workers[0],workers[0]])

if __name__=="__main__":
    unittest.main()
