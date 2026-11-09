"""F12-S contracts, checked before real 224-call Qwen run.

Fake answers only check engineering and must never produce a science PASS.
"""
import os
import unittest
from unittest.mock import patch
from observatory import f12s_predecessor_instance_factorial as study

class FakeFrozen:
    def __init__(self):
        self.calls=0
        self.prompt_tokens=0
        self.completion_tokens=0
        self.inference_seconds=0.0
        self.generator=type("G",(),{
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
            "real_model":False,"weights_sha256":study.prior.FIXED_MODEL_SHA256,
            "revision":"8911e8a47f92bac19d6f5c64a2e2095bd2f7d031",
            "backend_version":"0.3.36","temperature":0.0,
            "n_threads":4,"n_ctx":4096,"max_tokens":96,
            "calls":self.calls,"prompt_tokens":self.prompt_tokens,
            "completion_tokens":self.completion_tokens,
            "inference_seconds":self.inference_seconds,
        }

class F12SContracts(unittest.TestCase):
    def test_sources_and_constant_exact_12call_warmup(self):
        anchors=study.sources()
        self.assertEqual(len(anchors),6)
        self.assertEqual(anchors["O21-start"]["prompt_sha256"],study.SOURCE_SHA)
        self.assertEqual(study.WARMUP,study.prior.NORMAL*2)
        self.assertEqual(len(study.WARMUP),12)
        self.assertEqual(study.CELLS,(
            ("O21-disabled","reuse"),("O21-disabled","fresh"),
            ("O0-disabled","reuse"),("O0-disabled","fresh"),
        ))

    def test_four_worker_latin_square_each_cell_every_position(self):
        self.assertEqual(len(study.WORKER_ORDERS),4)
        for order in study.WORKER_ORDERS:
            self.assertEqual(set(order),set(range(4)))
        for position in range(4):
            self.assertEqual(
                {study.WORKER_ORDERS[w][position] for w in range(4)},
                set(range(4)))
        with self.assertRaises(ValueError):
            study.cell_attributes(4)

    def test_fake_confirms_only_predecessor_and_instance_change(self):
        with patch.dict(os.environ,{"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"4"}):
            cells=[study.run_cell(i,0,FakeFrozen) for i in range(4)]
        self.assertEqual(len(cells),4)
        target_hashes={c["target_call"]["prompt_sha256"] for c in cells}
        self.assertEqual(target_hashes,{study.SOURCE_SHA})
        for cell in cells:
            self.assertEqual(cell["history"][:12],list(study.WARMUP))
            self.assertEqual(len(cell["prior_calls"]),13)
            self.assertEqual(cell["prior_calls"][12]["anchor_id"],cell["predecessor"])
            self.assertEqual(cell["prior_calls"][12]["kind"],"predecessor")
            self.assertEqual(cell["target_call"]["kind"],"target")
            self.assertEqual(cell["target_call"]["index"],13)
            self.assertEqual(cell["prefix_model"]["calls"],13)
            if cell["instance"]=="fresh":
                self.assertEqual(cell["target_model"]["calls"],1)
                self.assertEqual(cell["target_call"]["model_calls_before"],0)
                self.assertEqual(cell["model_instance_init_count"],2)
                self.assertEqual(cell["target_call"]["cache_before"]["n_tokens_observed"],0)
            else:
                self.assertEqual(cell["target_model"]["calls"],14)
                self.assertEqual(cell["target_call"]["model_calls_before"],13)
                self.assertEqual(cell["model_instance_init_count"],1)
            self.assertFalse(cell["target_call"]["cache_before"]["kv_cache_hits_available"])

    def test_fake_16_cells_never_satisfy_real_gate(self):
        with patch.dict(os.environ,{"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"4"}):
            workers=[]
            for worker in range(4):
                cells=[study.run_cell(index,worker,FakeFrozen)
                       for index in study.WORKER_ORDERS[worker]]
                workers.append({
                    "study":study.STUDY,"worker":worker,
                    "cell_order":list(study.WORKER_ORDERS[worker]),
                    "cells":cells,
                })
            report=study.aggregate(workers)
        self.assertEqual(report["total_llm_calls"],224)
        self.assertEqual(report["prior_calls"],208)
        self.assertEqual(report["target_calls"],16)
        self.assertEqual(report["status"],"INVALID_S_PROTOCOL")
        self.assertFalse(report["gates"]["S0_complete_16_cells_224_real_inferences"])
        self.assertFalse(report["gates"]["S1_pinned_model_instance_boundary_and_source"])
        self.assertTrue(report["gates"]["S2_four_literal_replicates_per_cell"])
        self.assertFalse(report["gates"]["S3_preregistered_interaction_q2_q0_q0_q0"])
        self.assertFalse(report["product_integrated"])
        self.assertFalse(report["transfer_worlds_evaluated"])
        with self.assertRaises(ValueError):
            study.aggregate(workers[:3])
        with self.assertRaises(ValueError):
            study.aggregate([workers[0],workers[0],workers[2],workers[3]])

if __name__=="__main__":
    unittest.main()
