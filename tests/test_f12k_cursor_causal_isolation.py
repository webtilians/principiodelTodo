"""F12-K pre-inference contracts; fake backend cannot count as Qwen science."""
import hashlib
import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from observatory.f12k_cursor_causal_isolation import (
    STUDY, ROLE_MAPS, BIT_PAIRS, TOOLS, ACTIONS, ARMS,
    CONDITIONS, HORIZON, NEUTRAL, REPAIR, collect_probes,
    new_store, save_probes, from_store, grounded_plan, public_context,
    expected_index, shown_action, prompt_for, parse_action,
    deterministic_case, run,
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

class F12KPreflight(unittest.TestCase):
    def test_preregistered_new_worlds_are_unique_and_balanced(self):
        self.assertEqual(STUDY,"F12-K")
        self.assertEqual(len(ROLE_MAPS),10)
        self.assertEqual(len(set(ROLE_MAPS)),10)
        self.assertEqual(ARMS,("C0","C1","C2","C3","C4","C5"))
        self.assertEqual(CONDITIONS,("stable","changed"))
        self.assertEqual(HORIZON,7)
        self.assertEqual(len(BIT_PAIRS),10)
        self.assertEqual(sum(p for p,_ in BIT_PAIRS),5)
        self.assertEqual(sum(h for _,h in BIT_PAIRS),5)
        for roles in ROLE_MAPS:
            self.assertEqual(set(roles),{"LOAD","ENABLE","READ","PRIMARY","BACKUP","HOLD"})
        self.assertEqual(TOOLS,tuple(f"v{i}" for i in range(6)))
    def test_public_effects_all_worlds_and_deterministic_upper_bound(self):
        for wid in range(10):
            st=new_store()
            try:
                save_probes(st,collect_probes(wid))
                observed=from_store(st)
                self.assertEqual([r["source_id"] for r in observed],
                                 [f"P{i}" for i in range(10)])
                self.assertTrue(all(set(r)=={"source_id","public_input","events"}
                                    for r in observed))
                self.assertNotIn('"PRIMARY"',json.dumps(observed))
                self.assertNotIn('"BACKUP"',json.dumps(observed))
                plan=grounded_plan(observed)
                self.assertIsNotNone(plan)
                self.assertEqual(len(plan["plan"]),4)
                self.assertEqual(plan["plan"][-1],"verify")
                self.assertNotEqual(plan["backup"],plan["plan"][2])
                self.assertLessEqual(len(public_context(observed,plan)),3500)
                for condition in CONDITIONS:
                    for pub in (0,1):
                        for hidden in (0,1):
                            result=deterministic_case(wid,condition,pub,hidden,plan)
                            self.assertTrue(result["success"])
                            self.assertLessEqual(result["calls"],HORIZON)
            finally:st.close()
    def test_cursor_modes_isolate_action_progress_and_length(self):
        plan_info=grounded_plan(collect_probes(0))
        plan=plan_info["plan"]
        ctx=public_context(collect_probes(0),plan_info)
        self.assertEqual(len(NEUTRAL),len(REPAIR))
        assert plan[0]!=plan[1]
        self.assertIsNone(shown_action(plan,0,"C0"))
        self.assertIsNone(shown_action(plan,0,"C1"))
        self.assertEqual(shown_action(plan,0,"C2"),plan[0])
        self.assertEqual(shown_action(plan,0,"C3"),plan[0])
        self.assertEqual(shown_action(plan,0,"C4"),plan[0])
        self.assertEqual(shown_action(plan,0,"C5"),plan[1])
        prompts={mode:prompt_for("goal",ctx,plan_info,[],mode) for mode in ARMS}
        self.assertNotIn("CURRENT PLAN POSITION",prompts["C0"])
        self.assertIn("CURRENT PLAN POSITION",prompts["C1"])
        self.assertNotIn("NEXT EXPECTED PLAN STEP",prompts["C1"])
        for mode in ("C2","C3","C4","C5"):
            self.assertIn("NEXT EXPECTED PLAN STEP",prompts[mode])
        self.assertIn(REPAIR,prompts["C4"])
        self.assertNotIn(REPAIR,prompts["C3"])
        self.assertIn(NEUTRAL,prompts["C3"])
        self.assertIn(f"STEP (1/4): {plan[1]}",prompts["C5"])
        self.assertIn(f"STEP (1/4): {plan[0]}",prompts["C2"])
        for text in prompts.values():
            self.assertNotIn("_hidden",text)
            self.assertNotIn("ROLE_MAPS",text)
    def test_cursor_feedback_does_not_advance_on_error(self):
        p=grounded_plan(collect_probes(0))
        plan=p["plan"]
        evidence=[
            {"action":plan[0],"response":{"ack":True}},
            {"action":plan[1],"response":{"value":1}},
            {"action":plan[2],"response":{"error":"DISABLED_TOOL"}},
        ]
        self.assertEqual(expected_index(plan,p["backup"],evidence),2)
        evidence.append({"action":p["backup"],"response":{"value":0}})
        self.assertEqual(expected_index(plan,p["backup"],evidence),3)
    def test_strict_action_schema(self):
        for wrong in ("verify", '{"tool":"v0"}','{"action":"u0"}',
                      '{"action":"v0","oracle":true}',""):
            self.assertEqual(parse_action(wrong),"INVALID_LLM_OUTPUT")
        for action in ACTIONS:
            self.assertEqual(parse_action(json.dumps({"action":action})),action)
    def test_fake_end_to_end_never_scientific(self):
        with TemporaryDirectory() as d:
            result=run(FakeOnly(),Path(d))
            self.assertEqual(result["status"],"FAKE_TEST_ONLY")
            self.assertEqual(len(result["outcomes"]),120)
            self.assertEqual(len(result["worlds"]),10)
            self.assertTrue(result["gates"]["K1_deterministic_solution_10_of_10_each"])
            self.assertTrue(result["gates"]["K7_complete_prompts_sha_costs_and_public_cursor"])
            self.assertFalse(result["gates"]["K0_real_pinned_10_worlds_120_episodes_no_private_leak"])
            self.assertFalse(result["scientific_pass"])
            self.assertFalse(result["confirm_opened"])
            for c in CONDITIONS:
                for arm in ARMS:
                    self.assertEqual(result["scores"][c][arm]["episodes"],10)
            for row in result["outcomes"]:
                self.assertLessEqual(row["calls"],7)
                for event in row["events"]:
                    self.assertEqual(hashlib.sha256(event["exact_prompt"].encode()).hexdigest(),
                                     event["prompt_sha256"])
            self.assertTrue(Path(d,"f12k_full_report.json").exists())

if __name__=="__main__":
    unittest.main()
