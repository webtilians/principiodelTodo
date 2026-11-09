"""F12-U engineering tests. Synthetic traces NEVER constitute real inference."""
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from observatory import f12u_prefix_path_independent as u
from observatory import f12t_recover_original_artifacts as recover

class FakeCtx:
    def __init__(self):self.requests=[]
    def kv_cache_seq_rm(self,seq,p0,p1):
        self.requests.append((seq,p0,p1));return True
class FakeGen:
    def __init__(self):
        self.n_tokens=0
        self.input_ids=[0]*4096
        self._requires_eval=True
        self.cache=None
        self._ctx=FakeCtx()
    def n_ctx(self):return 4096
    @property
    def _input_ids(self):
        return self.input_ids[:self.n_tokens]
    def reset(self):
        self.n_tokens=0
        self._requires_eval=True
    def sample(self,*args,**kwargs):return 8
    def generate(self,tokens,*args,**kwargs):
        tokens=list(tokens)
        n=0
        for a,b in zip(self.input_ids[:self.n_tokens],tokens):
            if a!=b:break
            n+=1
        if n>0:
            self._ctx.kv_cache_seq_rm(-1,n,-1)
        else:
            self.reset()
            self._ctx.kv_cache_seq_rm(-1,0,-1)
        self.input_ids[:len(tokens)]=tokens
        self.n_tokens=len(tokens)
        self._requires_eval=False
        yield self.sample()
class FakeFrozen:
    def __init__(self):
        self.generator=FakeGen()
        self.calls=0
        self.prompt_tokens=0
        self.completion_tokens=0
        self.inference_seconds=0.
    def chat(self,system,prompt,max_tokens=96):
        list(self.generator.generate([1,2,3]))
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=4
        return '{"action":"q0"}'
    def metadata(self):
        return {
            "real_model":False,
            "weights_sha256":u.FIXED_MODEL_SHA256,
            "revision":"8911e8a47f92bac19d6f5c64a2e2095bd2f7d031",
            "backend_version":"0.3.36",
            "temperature":0.0,"n_ctx":4096,"n_threads":4,"max_tokens":96,
            "calls":self.calls,"prompt_tokens":self.prompt_tokens,
            "completion_tokens":self.completion_tokens,
            "inference_seconds":self.inference_seconds,
        }

class Contracts(unittest.TestCase):
    def test_frozen_target_and_cell_treatments(self):
        self.assertEqual(len(u.sources()),6)
        self.assertEqual(u.sources()["O21-start"]["prompt_sha256"],u.SOURCE_SHA)
        self.assertEqual(len(u.CELLS),8)
        self.assertEqual(len(set(u.CELLS)),8)
        self.assertEqual(u.INTERVENTIONS,("TRACE","PREFIX_ZERO","RESET","FRESH"))
        self.assertEqual(len(u.WARMUP),12)
        self.assertEqual(len(u.WORKER_ORDERS),3)
        self.assertTrue(all(set(x)==set(range(8)) for x in u.WORKER_ORDERS))
        with self.assertRaises(ValueError):u.cell_attributes(8)

    def test_sentinel_forces_no_prefix_without_patching_kv_return(self):
        a=u.sources()["O21-start"]
        model=FakeFrozen()
        model.chat(u.old.SYSTEM,"warmup")
        trace=u.trace_target(model,a,"TRACE")
        self.assertGreater(trace["generate_snapshots"][0]["computed_longest_prefix"],0)
        self.assertGreater(trace["kv_seq_rm_events"][0]["p0"],0)
        zero=u.trace_target(model,a,"PREFIX_ZERO")
        self.assertEqual(zero["generate_snapshots"][0]["computed_longest_prefix"],0)
        self.assertEqual(zero["kv_seq_rm_events"][0]["p0"],0)
        self.assertTrue(zero["mutation"]["replaced_by_decode"])
        self.assertTrue(zero["hooks_restored"])
        self.assertEqual(zero["chosen_sampled_token_ids"],[8])
        self.assertEqual(trace["raw"],zero["raw"])
        self.assertTrue(all(x["delegated"] for x in zero["kv_seq_rm_events"]))

    def test_fake_full_cells_never_pass_scientific_gate(self):
        with patch.dict(os.environ,{"PYTHONHASHSEED":"0","OMP_NUM_THREADS":"4"}):
            workers=[]
            for worker in u.WORKERS:
                results=[u.run_cell(i,worker,FakeFrozen)
                         for i in u.WORKER_ORDERS[worker]]
                workers.append({
                    "study":u.STUDY,"worker":worker,
                    "cell_order":list(u.WORKER_ORDERS[worker]),"cells":results,
                })
            manifest={
                "original_run_id":42675902103,"reexecuted_model_inferences":0,
                "strict_saved_and_reevaluated_summary_identical":True,
                "cells_revalidated":30,"original_scientific_gates":{"T0":True},
            }
            result=u.aggregate(workers,manifest)
        self.assertEqual(result["total_llm_calls"],336)
        self.assertEqual(result["prior_llm_calls"],312)
        self.assertEqual(result["target_llm_calls"],24)
        self.assertEqual(result["status"],"INVALID_U_PROTOCOL")
        self.assertFalse(result["gates"]["U1_model_and_identical_target_token_hash"])
        self.assertFalse(result["product_integrated"])
        self.assertFalse(result["transfer_worlds_evaluated"])
        with self.assertRaises(ValueError):u.aggregate(workers[:2],manifest)

    def test_recovery_fails_closed_if_old_artifacts_absent(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)
            with self.assertRaises(ValueError):
                recover.recover(path,path,path/"out")

if __name__=="__main__":unittest.main()
