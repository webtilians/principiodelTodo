"""F12-R preregistered engineering contracts: 12 frozen cells, real-only evidence."""
import os
import unittest
from unittest.mock import patch
from observatory import f12r_o21_position2_reproduction as study

class FakeFrozen:
    real_model=False
    def __init__(self):
        self.calls=0
        self.prompt_tokens=0
        self.completion_tokens=0
        self.inference_seconds=0.0
        self.generator=type("FakeGenerator",(),{
            "n_tokens":0,"cache":None,"n_ctx":lambda self:4096,
        })()
    def chat(self,system,prompt,max_tokens=96):
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=4
        self.generator.n_tokens+=len(prompt)//4
        return '{"action":"q0"}'
    def metadata(self):
        return {
            "real_model":False,"weights_sha256":study.FIXED_MODEL_SHA256,
            "revision":"8911e8a47f92bac19d6f5c64a2e2095bd2f7d031",
            "backend_version":"0.3.36",
            "temperature":0.0,"n_threads":4,"n_ctx":4096,"max_tokens":96,
            "calls":self.calls,"prompt_tokens":self.prompt_tokens,
            "completion_tokens":self.completion_tokens,
            "inference_seconds":self.inference_seconds,
        }

class F12RContracts(unittest.TestCase):
    def test_o21_unchanged_sha_and_original_exact_warmup(self):
        sources=study.anchors_by_id()
        self.assertEqual(sources["O21-start"]["prompt_sha256"],study.SOURCE_SHA)
        self.assertEqual(study.NORMAL,(
            "O0-start","O0-disabled","O7-start","O7-disabled",
            "O21-start","O21-disabled"))
        self.assertEqual(study.INVERSE,tuple(reversed(study.NORMAL)))
        self.assertEqual(study.STUDY,"F12-R")
        warm,block=study.schedule("HP",2)
        self.assertEqual(warm,study.NORMAL*2)
        self.assertEqual(block,("O21-disabled",))
        self.assertEqual(len(warm+block),13)
        self.assertEqual(study.CELLS.index(("HP",2,"reuse")),4)

    def test_twelve_cells_3x2x2_and_position_last_predecessor(self):
        self.assertEqual(study.HISTORIES,("H0","HP","HR"))
        self.assertEqual(study.POSITIONS,(2,4))
        self.assertEqual(study.INSTANCES,("reuse","fresh"))
        self.assertEqual(len(study.CELLS),12)
        self.assertEqual(len(set(study.CELLS)),12)
        self.assertEqual(
            set(study.CELLS),{
                (history,pos,inst)
                for history in study.HISTORIES
                for pos in study.POSITIONS
                for inst in study.INSTANCES})
        for h in study.HISTORIES:
            for p in study.POSITIONS:
                warm,block=study.schedule(h,p)
                self.assertEqual(len(warm),0 if h=="H0" else 12)
                self.assertEqual(len(block),p-1)
                self.assertEqual(block[-1],"O21-disabled")
                if p==4:self.assertEqual(
                    block,("O0-start","O7-start","O21-disabled"))
        with self.assertRaises(ValueError):study.cell_attributes(12)
        with self.assertRaises(ValueError):study.schedule("unknown",2)

    def test_reused_and_reloaded_fake_model_correct_call_boundaries(self):
        with patch.dict(os.environ,{"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"4"}):
            reused=study.run_cell(4,0,FakeFrozen)
            reloaded=study.run_cell(5,0,FakeFrozen)
        self.assertEqual(reused["prior_schedule"],study.NORMAL*2+("O21-disabled",))
        self.assertEqual(reloaded["prior_schedule"],reused["prior_schedule"])
        self.assertEqual(reused["model_instance_init_count"],1)
        self.assertEqual(reloaded["model_instance_init_count"],2)
        self.assertEqual(reused["target_call"]["prompt_sha256"],study.SOURCE_SHA)
        self.assertEqual(reloaded["target_call"]["prompt_sha256"],study.SOURCE_SHA)
        self.assertEqual(reused["target_call"]["model_calls_before"],13)
        self.assertEqual(reused["target_call"]["model_calls_after"],14)
        self.assertEqual(reloaded["target_call"]["model_calls_before"],0)
        self.assertEqual(reloaded["target_call"]["model_calls_after"],1)
        self.assertEqual(len(reused["prior_calls"]),13)
        self.assertEqual(reused["target_call"]["raw"],'{"action":"q0"}')
        self.assertFalse(reused["target_call"]["cache_before"]["kv_cache_hits_available"])
        self.assertIsNone(reused["target_call"]["cache_after"]["kv_cache_hits"])
        self.assertEqual(reloaded["target_call"]["cache_before"]["n_tokens_observed"],0)
        self.assertGreater(reused["target_call"]["cache_before"]["n_tokens_observed"],0)

    def test_fake_all_24_cases_cannot_pass_real_science_gate(self):
        with patch.dict(os.environ,{"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"4"}):
            workers=[]
            for worker in (0,1):
                cells=[study.run_cell(i,worker,FakeFrozen)
                       for i in range(12)]
                workers.append({
                    "study":study.STUDY,"worker":worker,
                    "cell_order":(list(range(12)) if worker==0 else
                                  list(reversed(range(12)))),
                    "cells":cells,
                })
            result=study.aggregate(workers)
        self.assertEqual(result["total_llm_calls"],264)
        self.assertEqual(result["prior_calls"],240)
        self.assertEqual(result["target_calls"],24)
        self.assertEqual(len(result["cell_responses"]),12)
        self.assertEqual(result["status"],"INVALID_R_PROTOCOL")
        self.assertFalse(result["gates"]["R0_24_independent_cells_264_real_inferences"])
        self.assertFalse(result["gates"]["R1_frozen_model_and_target_prompt"])
        self.assertTrue(result["gates"]["R2_replicates_literal_per_cell"])
        self.assertFalse(result["gates"]["R3_original_12call_then_P2_q2"])
        self.assertFalse(result["transfer_worlds_evaluated"])
        self.assertFalse(result["product_integrated"])
        self.assertFalse(result["kv_cache_hit_counts_available"])
        with self.assertRaises(ValueError):
            study.aggregate(workers[:1])
        with self.assertRaises(ValueError):
            study.aggregate([workers[0],workers[0]])

if __name__=="__main__":
    unittest.main()
