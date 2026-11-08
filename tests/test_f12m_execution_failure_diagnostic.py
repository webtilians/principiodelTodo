"""F12-M causal diagnostic contracts; no fake-backend result is science."""
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f12m_execution_failure_diagnostic import (
    STUDY, ROLE_MAPS, USED_J_MAPS, USED_K_MAPS, USED_L_MAPS,
    BIT_PAIRS, TOOLS, ACTIONS, HORIZON, SHARD_COUNT, CONDITIONS,
    ARMS, SELECTION_ARMS, RECOVERY_ARMS, World,
    collect_probes, new_store, save_probes, from_store, grounded_plan,
    expected_index, prompt_for, shown_action,
    frozen_preflight, run_recovery_case, run_selection_termination_case,
    run_shard, aggregate,
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
            "real_model":False,
            "prompt_tokens":self.prompt_tokens,
            "completion_tokens":self.completion_tokens,
            "inference_seconds":self.inference_seconds,
        }
    def chat(self,system,prompt,max_tokens=96):
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=4
        return '{"action":"verify"}'

class FollowPlanFake(FakeOnly):
    def chat(self,system,prompt,max_tokens=96):
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=4
        import re
        match=re.search(r"NEXT EXPECTED PLAN STEP \(\d/4\): (\w+)",prompt)
        if match:
            return json.dumps({"action":match.group(1)})
        return '{"action":"verify"}'

class F12MContracts(unittest.TestCase):
    def test_sealed_worlds_disjoint_balanced_216_identities(self):
        self.assertEqual(STUDY,"F12-M")
        self.assertEqual(len(ROLE_MAPS),24)
        self.assertEqual(len(set(ROLE_MAPS)),24)
        self.assertFalse(set(ROLE_MAPS)&(set(USED_J_MAPS)|set(USED_K_MAPS)|set(USED_L_MAPS)))
        self.assertEqual(Counter(BIT_PAIRS),{
            (0,0):6,(0,1):6,(1,0):6,(1,1):6})
        self.assertEqual(TOOLS,tuple(f"z{i}" for i in range(6)))
        self.assertEqual(ACTIONS,(*TOOLS,"verify"))
        self.assertEqual(HORIZON,7)
        self.assertEqual(SHARD_COUNT,4)
        self.assertEqual(SELECTION_ARMS,("S0","S1","T1"))
        self.assertEqual(RECOVERY_ARMS,("R0","R1","R2"))
        self.assertEqual(len(CONDITIONS),2)
    def test_all_reference_worlds_and_real_public_error(self):
        worlds=frozen_preflight()
        self.assertEqual(len(worlds),24)
        for w in worlds:
            self.assertEqual(len(w["observed"]),10)
            self.assertEqual([r["source_id"] for r in w["observed"]],
                             [f"P{i}" for i in range(10)])
            self.assertEqual(w["controlled_error_prefix_responses"][-1],
                             {"error":"DISABLED_TOOL"})
            self.assertTrue(w["preflight"]["stable"]["success"])
            self.assertTrue(w["preflight"]["changed"]["success"])
            self.assertFalse(any(
                event["action"]=="verify" for r in w["observed"]
                for event in r["events"]))
            self.assertLessEqual(len(w["context"]),3500)
    def test_treatment_prompts_only_difference_cue_and_recovery_hint(self):
        w=frozen_preflight()[0]
        p=w["plan"]
        for mode in SELECTION_ARMS+RECOVERY_ARMS:
            prompt=prompt_for("goal",w["context"],p,[],mode)
            self.assertIn("PUBLIC GOAL",prompt)
            self.assertNotIn("hidden_bit_audit_only",prompt)
            self.assertNotIn("roles_audit_only",prompt)
            if mode in ("S0","R1"):
                self.assertIn("CURRENT PLAN POSITION",prompt)
                self.assertNotIn("NEXT EXPECTED PLAN STEP",prompt)
            else:
                self.assertIn(f"STEP (1/4): {p['plan'][0]}",prompt)
        world=World(0,*BIT_PAIRS[0],"changed")
        seen=[]
        for action in p["plan"][:3]:
            seen.append({"action":action,"response":world.execute(action)})
        self.assertEqual(seen[-1]["response"],{"error":"DISABLED_TOOL"})
        self.assertEqual(expected_index(p["plan"],p["backup"],seen),2)
        prompt0=prompt_for("goal",w["context"],p,seen,"R0")
        prompt2=prompt_for("goal",w["context"],p,seen,"R2")
        self.assertIn(f"STEP (3/4): {p['plan'][2]}",prompt0)
        self.assertIn(f"STEP (3/4): {p['backup']}",prompt2)
        self.assertEqual(len(prompt0),len(prompt2))
        self.assertNotIn("ROLE_MAPS",prompt2)
        self.assertEqual(shown_action(p["plan"],2,"R2",p["backup"],seen),
                         p["backup"])
    def test_termination_guard_only_after_public_cursor_three(self):
        w=frozen_preflight()[0]
        r=run_selection_termination_case(
            0,"stable",*BIT_PAIRS[0],FollowPlanFake(),
            w["context"],w["plan"],"T1")
        self.assertTrue(r["success"])
        self.assertEqual(r["supervisor_calls"],1)
        self.assertEqual(r["calls"],3)
        self.assertEqual(r["actions_total"],4)
        self.assertEqual(r["events"][-1]["origin"],"supervisor")
        self.assertEqual(r["events"][-1]["action"],"verify")
        self.assertEqual(r["events"][-1]["real_cursor_index_before"],3)
        self.assertIsNone(r["events"][-1]["exact_prompt"])
        self.assertEqual(r["events"][-1]["prompt_tokens"],0)
        r0=run_selection_termination_case(
            0,"stable",*BIT_PAIRS[0],FollowPlanFake(),
            w["context"],w["plan"],"S1")
        self.assertTrue(r0["success"])
        self.assertEqual(r0["calls"],4)
        self.assertEqual(r0["supervisor_calls"],0)
    def test_recovery_forced_prefix_never_as_model_tokens(self):
        w=frozen_preflight()[0]
        r=run_recovery_case(
            0,*BIT_PAIRS[0],FollowPlanFake(),
            w["context"],w["plan"],"R2")
        self.assertTrue(r["success"])
        self.assertEqual(r["events"][2]["response"],{"error":"DISABLED_TOOL"})
        self.assertEqual(r["events"][2]["origin"],"controlled_prefix")
        self.assertEqual(r["prefixed_actions"],3)
        self.assertEqual(r["calls"],2)
        self.assertLessEqual(r["actions_total"],7)
        self.assertTrue(r["recovered_after_error"])
        self.assertFalse(r["preemptive_backup"])
        self.assertTrue(all(e["prompt_tokens"]==0 for e in r["events"][:3]))
    def test_fake_shard_aggregation_cannot_claim_success(self):
        reports=[]
        with TemporaryDirectory() as folder:
            p=Path(folder)
            for shard in range(4):
                report=run_shard(FakeOnly(),shard,p)
                self.assertEqual(report["study_status"],"SHARD_ONLY_NO_SCIENTIFIC_CONCLUSION")
                self.assertEqual(len(report["worlds"]),6)
                self.assertEqual(len(report["outcomes"]),54)
                self.assertTrue((p/f"f12m_shard{shard}.json").exists())
                reports.append(report)
            agg=aggregate(reports,p)
            self.assertEqual(agg["status"],"FAKE_TEST_ONLY")
            self.assertTrue(agg["gates"]["M1_reference_and_public_error_prefix"])
            self.assertTrue(agg["gates"]["M5_traces_origins_budgets_and_no_hidden_leaks"])
            self.assertFalse(agg["gates"]["M0_model_worlds_216_complete"])
            self.assertFalse(agg["scientific_pass"])
            self.assertFalse(agg["product_integrated"])
            self.assertTrue((p/"f12m_full.json").exists())
            with self.assertRaises(ValueError):aggregate(reports[:3])
            with self.assertRaises(ValueError):
                aggregate([reports[0],reports[0],reports[2],reports[3]])
            for sh in reports:
                for r in sh["outcomes"]:
                    self.assertLessEqual(r["actions_total"],7)
                    if r["arm"] in RECOVERY_ARMS:
                        self.assertEqual(r["prefixed_actions"],3)
                        self.assertLessEqual(r["calls"],4)
                    for event in r["events"]:
                        if event["origin"]=="llm":
                            self.assertEqual(hashlib.sha256(
                                event["exact_prompt"].encode()).hexdigest(),
                                event["prompt_sha256"])

if __name__=="__main__":
    unittest.main()
