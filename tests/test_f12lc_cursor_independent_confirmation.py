"""F12-L-C immutable preflight tests: no model fake is scientific evidence."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from collections import Counter

from observatory.f12lc_cursor_independent_confirmation import (
    STUDY, ROLE_MAPS, USED_J_MAPS, USED_K_MAPS, BIT_PAIRS, TOOLS, ARMS,
    CONDITIONS, SHARD_COUNT, HORIZON, collect_probes, new_store, save_probes,
    from_store, grounded_plan, public_context, deterministic_case, shown_action,
    prompt_for, frozen_preflight, run_shard, aggregate,
)

class FakeOnly:
    real_model=False
    def __init__(self):
        self.calls=0
        self.prompt_tokens=0
        self.completion_tokens=0
        self.inference_seconds=0.0
    def metadata(self):
        return {
            "real_model":False,"prompt_tokens":self.prompt_tokens,
            "completion_tokens":self.completion_tokens,
            "inference_seconds":self.inference_seconds,
        }
    def chat(self,system,prompt,max_tokens=96):
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=4
        return '{"action":"verify"}'

class F12LContracts(unittest.TestCase):
    def test_frozen_novel_mapping_and_balanced_inputs(self):
        self.assertEqual(STUDY,"F12-L-C")
        self.assertEqual(len(ROLE_MAPS),24)
        self.assertEqual(len(set(ROLE_MAPS)),24)
        self.assertFalse(set(ROLE_MAPS)&set(USED_J_MAPS))
        self.assertFalse(set(ROLE_MAPS)&set(USED_K_MAPS))
        self.assertEqual(Counter(BIT_PAIRS),{
            (0,0):6,(0,1):6,(1,0):6,(1,1):6})
        self.assertEqual(TOOLS,tuple(f"w{i}" for i in range(6)))
        self.assertEqual(ARMS,("C0","C1","C2","C5"))
        self.assertEqual(CONDITIONS,("stable","changed"))
        self.assertEqual(SHARD_COUNT,4)
        self.assertEqual(HORIZON,7)
    def test_new_worlds_all_preflight_solvable_with_public_evidence(self):
        worlds=frozen_preflight()
        self.assertEqual(len(worlds),24)
        for w in worlds:
            self.assertEqual(len(w["observed"]),10)
            self.assertTrue(w["guard_initial"]["correct_accepted"])
            self.assertTrue(w["guard_initial"]["false_rejected"])
            self.assertNotEqual(w["guard_initial"]["reported_good_name"],
                                w["guard_initial"]["reported_bad_name"])
            self.assertTrue(w["preflight"]["stable"]["success"])
            self.assertTrue(w["preflight"]["changed"]["success"])
            self.assertFalse(any(ev["action"]=="verify"
                for rec in w["observed"] for ev in rec["events"]))
            self.assertLessEqual(len(w["context"]),3500)
    def test_unchanged_treatment_prompt_mechanics_except_alias(self):
        w=frozen_preflight()[0]
        p=w["plan"]
        ctx=w["context"]
        texts={arm:prompt_for("goal",ctx,p,[],arm) for arm in ARMS}
        self.assertNotIn("PLAN POSITION",texts["C0"])
        self.assertIn("CURRENT PLAN POSITION",texts["C1"])
        self.assertNotIn("NEXT EXPECTED PLAN STEP",texts["C1"])
        self.assertIn("NEXT EXPECTED PLAN STEP",texts["C2"])
        self.assertIn("NEXT EXPECTED PLAN STEP",texts["C5"])
        correct=p["plan"][0]
        incorrect=p["plan"][1]
        self.assertIn(f"STEP (1/4): {correct}",texts["C2"])
        self.assertIn(f"STEP (1/4): {incorrect}",texts["C5"])
        self.assertNotIn("ROLE_MAPS",json.dumps(texts))
        self.assertNotIn("_hidden",json.dumps(texts))
    def test_each_shard_with_fake_requires_aggregate_and_cannot_pass(self):
        reports=[]
        with TemporaryDirectory() as d:
            for shard in range(4):
                r=run_shard(FakeOnly(),shard,Path(d))
                self.assertEqual(r["study_status"],"SHARD_ONLY_NOT_SCIENTIFIC_RESULT")
                self.assertFalse(r["confirm_complete"])
                self.assertEqual(len(r["worlds"]),6)
                self.assertEqual(len(r["outcomes"]),48)
                self.assertTrue(Path(d,f"f12lc_shard{shard}.json").is_file())
                reports.append(r)
            full=aggregate(reports,Path(d))
            self.assertEqual(full["status"],"FAKE_TEST_ONLY")
            self.assertFalse(full["gates"]["L0_reproducible_real_model_24_new_worlds_192_complete_episodes"])
            self.assertTrue(full["gates"]["L1_public_only_reference_solves_24_each"])
            self.assertTrue(full["gates"]["L5_complete_shard_trace_cost_and_guard_audit"])
            self.assertEqual(full["security_guard"]["false_initial_cues_rejected"],24)
            self.assertEqual(full["security_guard"]["correct_initial_cues_accepted"],24)
            self.assertFalse(full["scientific_pass_synthetic_only"])
            self.assertFalse(full["product_integrated"])
            self.assertTrue(Path(d,"f12lc_confirm_full.json").is_file())
            with self.assertRaises(ValueError):
                aggregate(reports[:3])
            with self.assertRaises(ValueError):
                aggregate([reports[0],reports[0],reports[2],reports[3]])
    def test_per_world_read_from_authentic_sqlite(self):
        for wid in (0,7,15,23):
            store=new_store()
            try:
                save_probes(store,collect_probes(wid))
                r=from_store(store)
                self.assertEqual([x["source_id"] for x in r],
                                 [f"P{i}" for i in range(10)])
                plan=grounded_plan(r)
                self.assertEqual(len(plan["plan"]),4)
                self.assertNotEqual(
                    shown_action(plan["plan"],0,"C2"),
                    shown_action(plan["plan"],0,"C5"))
                self.assertTrue(
                    deterministic_case(wid,"changed",0,1,plan)["success"])
            finally:
                store.close()

if __name__=="__main__":
    unittest.main()
