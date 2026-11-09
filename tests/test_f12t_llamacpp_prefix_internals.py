"""F12-T engineering tests, with no LLM inference and no scientific claims."""
import os
import unittest
from unittest.mock import patch

from observatory import f12t_llamacpp_prefix_internals as t

class FakeContext:
    def __init__(self):
        self.requests=[]
        self.clears=0
    def kv_cache_seq_rm(self,seq,p0,p1):
        self.requests.append((seq,p0,p1))
        return True
    def kv_cache_clear(self):
        self.clears+=1

class FakeGenerator:
    def __init__(self):
        self.n_tokens=0
        self._requires_eval=True
        self.cache=None
        self._ctx=FakeContext()
    def n_ctx(self):
        return 4096
    def reset(self):
        self.n_tokens=0
        self._requires_eval=True

class FakeFrozen:
    def __init__(self):
        self.calls=0
        self.prompt_tokens=0
        self.completion_tokens=0
        self.inference_seconds=0.0
        self.generator=FakeGenerator()
    def chat(self,system,prompt,max_tokens=96):
        # Synthetic prefix route only for engineering: not model evidence.
        gen=self.generator
        if gen.n_tokens>0:
            used=gen._ctx.kv_cache_seq_rm(-1,245,-1)
            if not used:
                gen.reset()
                gen._ctx.kv_cache_seq_rm(-1,0,-1)
            else:
                gen._ctx.kv_cache_seq_rm(-1,245,-1)
        else:
            gen._ctx.kv_cache_seq_rm(-1,0,-1)
        gen.n_tokens=612
        gen._requires_eval=False
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=4
        return '{"action":"q0"}'
    def metadata(self):
        return {
            "real_model":False,
            "weights_sha256":t.FIXED_MODEL_SHA256,
            "revision":"8911e8a47f92bac19d6f5c64a2e2095bd2f7d031",
            "backend_version":"0.3.36",
            "temperature":0.0,"n_threads":4,"n_ctx":4096,"max_tokens":96,
            "calls":self.calls,
            "prompt_tokens":self.prompt_tokens,
            "completion_tokens":self.completion_tokens,
            "inference_seconds":self.inference_seconds,
        }

class F12TContracts(unittest.TestCase):
    def test_unchanged_frozen_o21_and_10_cells(self):
        records=t.sources()
        self.assertEqual(len(records),6)
        self.assertEqual(records["O21-start"]["prompt_sha256"],t.SOURCE_SHA)
        self.assertEqual(len(t.CELLS),10)
        self.assertEqual(t.CELLS[0],("O21-disabled","NATIVE"))
        self.assertEqual(t.CELLS[1],("O21-disabled","OBSERVE"))
        self.assertEqual(t.CELLS[2],("O21-disabled","BLOCK_PREFIX"))
        self.assertEqual(t.CELLS[3],("O21-disabled","RESET"))
        self.assertEqual(t.CELLS[4],("O21-disabled","CLEAR_RESET"))
        self.assertEqual(t.CELLS[5],("O0-disabled","NATIVE"))
        self.assertEqual(len(t.WARMUP),12)
        self.assertEqual(
            set(t.WORKER_ORDERS[2]),set(range(10)))
        with self.assertRaises(ValueError):
            t.cell_attributes(10)

    def test_fake_spy_is_noop_and_block_forces_fallback(self):
        a=t.sources()["O21-start"]
        observer=FakeFrozen()
        observer.chat(t.SYSTEM,"prior")
        before=observer.metadata()
        observed=t.intervene_and_chat(observer,a,"OBSERVE")
        self.assertTrue(observed["patch_installed"])
        self.assertTrue(observed["patch_restored"])
        self.assertEqual(observed["blocked_positive_request_count"],0)
        self.assertGreaterEqual(observed["attempted_positive_requests"],1)
        self.assertTrue(all(e["delegated"] for e in observed["kv_seq_rm_calls"]))
        self.assertEqual(observer.metadata()["calls"],before["calls"]+1)

        blocked_model=FakeFrozen()
        blocked_model.chat(t.SYSTEM,"prior")
        blocked=t.intervene_and_chat(blocked_model,a,"BLOCK_PREFIX")
        self.assertTrue(blocked["patch_restored"])
        self.assertEqual(blocked["blocked_positive_request_count"],1)
        self.assertTrue(any(x["blocked"] and x["returned"] is False
                            for x in blocked["kv_seq_rm_calls"]))
        self.assertTrue(any(x["p0"]==0 and x["delegated"]
                            for x in blocked["kv_seq_rm_calls"]))
        self.assertEqual(blocked["raw"],observed["raw"])

    def test_reset_and_clear_reset_do_not_leave_invalid_token_state(self):
        a=t.sources()["O21-start"]
        for intervention in ("RESET","CLEAR_RESET"):
            model=FakeFrozen()
            model.chat(t.SYSTEM,"prior")
            result=t.intervene_and_chat(model,a,intervention)
            self.assertEqual(
                result["state_after_intervention_before_chat"]["n_tokens_observed"],0)
            self.assertTrue(result["state_after_intervention_before_chat"]["requires_eval"])
            self.assertTrue(result["reset_applied"])
            self.assertEqual(result["clear_applied"],intervention=="CLEAR_RESET")
            self.assertFalse(result["patch_installed"])
            self.assertEqual(result["blocked_positive_request_count"],0)
            self.assertEqual(result["raw"],'{"action":"q0"}')

    def test_fake_30_cells_never_promote_scientific_status(self):
        with patch.dict(os.environ,{"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"4"}):
            workers=[]
            for worker in range(3):
                cells=[t.run_cell(i,worker,FakeFrozen)
                       for i in t.WORKER_ORDERS[worker]]
                workers.append({
                    "study":t.STUDY,
                    "worker":worker,
                    "cell_order":list(t.WORKER_ORDERS[worker]),
                    "cells":cells,
                })
            report=t.aggregate(workers)
        self.assertEqual(report["total_llm_calls"],420)
        self.assertEqual(report["prior_calls"],390)
        self.assertEqual(report["target_calls"],30)
        self.assertEqual(report["status"],"INVALID_T_PROTOCOL")
        self.assertFalse(report["gates"]["T0_30_cells_420_real_model_calls"])
        self.assertFalse(report["gates"]["T1_pinned_model_same_target"])
        self.assertFalse(report["gates"]["T4_exact_q2_to_q0_prefix_path_pattern"])
        self.assertFalse(report["transfer_worlds_evaluated"])
        self.assertFalse(report["product_integrated"])
        with self.assertRaises(ValueError):
            t.aggregate(workers[:2])
        with self.assertRaises(ValueError):
            t.aggregate([workers[0],workers[0],workers[2]])

if __name__=="__main__":
    unittest.main()
