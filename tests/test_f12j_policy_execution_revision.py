"""F12-J contract tests. Synthetic model NEVER counts as scientific evidence."""
import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from observatory.f12j_policy_execution_revision import (
    ACTIONS, ARMS, CONDITIONS, HORIZON, ROLE_MAPS, World, collect_probes,
    new_store, save_probes, from_store, grounded_plan, public_context,
    expected_index, prompt_for, parse_action, deterministic_case, run,
)

class Fake:
    real_model=False
    def __init__(self):
        self.calls=0
        self.prompt_tokens=0
        self.completion_tokens=0
        self.inference_seconds=0.0
    def metadata(self):
        return {"real_model":False,"prompt_tokens":self.prompt_tokens,
                "completion_tokens":self.completion_tokens,
                "inference_seconds":self.inference_seconds}
    def chat(self,system,prompt,max_tokens=96):
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=9
        return '{"action":"verify"}'

class F12JContracts(unittest.TestCase):
    def test_new_eight_worlds_and_no_complete_target_in_probes(self):
        self.assertEqual(len(ROLE_MAPS),8)
        for x in ROLE_MAPS:
            self.assertEqual(set(x),
                {"LOAD","ENABLE","READ","PRIMARY","BACKUP","HOLD"})
        for wid in range(8):
            observed=collect_probes(wid)
            self.assertEqual([x["source_id"] for x in observed],
                             [f"P{i}" for i in range(10)])
            for r in observed:
                self.assertEqual(set(r),{"source_id","public_input","events"})
                self.assertNotIn("_hidden",json.dumps(r))
                self.assertNotIn("PRIMARY",json.dumps(r))
                self.assertNotIn("BACKUP",json.dumps(r))
        self.assertEqual(ARMS,("B0","B1","B2","B3"))
        self.assertEqual(CONDITIONS,("stable","changed"))
        self.assertEqual(HORIZON,7)
    def test_grounded_plan_and_upper_bound_solve_all_without_oracle(self):
        for wid in range(8):
            st=new_store()
            try:
                save_probes(st,collect_probes(wid))
                records=from_store(st)
                self.assertEqual(len(records),10)
                policy=grounded_plan(records)
                self.assertEqual(len(policy["plan"]),4)
                self.assertNotEqual(policy["backup"],policy["plan"][-2])
                self.assertTrue(policy["sources"])
                self.assertLessEqual(len(public_context(records,policy)),3500)
                for condition in CONDITIONS:
                    for pub in (0,1):
                        for hidden in (0,1):
                            result=deterministic_case(wid,condition,pub,hidden,policy)
                            self.assertTrue(result["success"])
                            self.assertLessEqual(result["calls"],HORIZON)
            finally:st.close()
    def test_public_error_is_required_to_advance_to_backup(self):
        wid=0
        policy=grounded_plan(collect_probes(wid))
        plan=policy["plan"]
        history=[
            {"action":plan[0],"response":{"ack":True}},
            {"action":plan[1],"response":{"value":1}},
            {"action":plan[2],"response":{"error":"DISABLED_TOOL"}},
        ]
        self.assertEqual(expected_index(plan,policy["backup"],history),2)
        updated=history+[{"action":policy["backup"],"response":{"value":0}}]
        self.assertEqual(expected_index(plan,policy["backup"],updated),3)
        ctx=public_context(collect_probes(wid),policy)
        b1=prompt_for("goal",ctx,policy,history,"B1")
        b2=prompt_for("goal",ctx,policy,history,"B2")
        self.assertIn("NEXT EXPECTED PLAN STEP",b1)
        self.assertIn("DISABLED_TOOL",b2)
        self.assertNotIn("DISABLED_TOOL",b1)
        self.assertNotIn("_hidden",b2)
        self.assertNotIn("PRIMARY",b2)
    def test_parse_action_fails_closed(self):
        for wrong in ("verify",'{"action":"z0"}','{"x":"u0"}',
                      '{"action":"u1","other":true}',""):
            self.assertEqual(parse_action(wrong),"INVALID_LLM_OUTPUT")
        for a in ACTIONS:
            self.assertEqual(parse_action(json.dumps({"action":a})),a)
    def test_fake_runs_64_episodes_but_not_scientific_pass(self):
        with TemporaryDirectory() as d:
            report=run(Fake(),Path(d))
            self.assertEqual(report["status"],"FAKE_TEST_ONLY")
            self.assertFalse(report["gates"]["J0_real_model_and_64_clean_paired_episodes"])
            self.assertTrue(report["gates"]["J1_strong_deterministic_upper_bound"])
            self.assertEqual(len(report["outcomes"]),64)
            self.assertFalse(report["scientific_pass"])
            self.assertTrue(Path(d,"f12j_full_report.json").is_file())
            for c in CONDITIONS:
                for a in ARMS:
                    self.assertEqual(report["scores"][c][a]["episodes"],8)
            for x in report["outcomes"]:
                self.assertLessEqual(x["calls"],HORIZON)

if __name__=="__main__":unittest.main()
