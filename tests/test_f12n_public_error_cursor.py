"""F12-N preregistered public-error cursor isolation and audit contracts.

Fake models exercise only engineering gates; never count as scientific evidence.
"""
import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f12n_public_error_cursor import (
    STUDY, ROLE_MAPS, USED_J_MAPS, USED_K_MAPS, USED_L_MAPS, USED_M_MAPS,
    BIT_PAIRS, TOOLS, ACTIONS, ARMS, CONDITIONS, HORIZON, SHARD_COUNT,
    World, collect_probes, grounded_plan, expected_index,
    prompt_for, shown_action, has_public_disabled_primary,
    frozen_preflight, run_episode, run_shard, aggregate,
)

class FakeOnly:
    real_model=False
    def __init__(self):
        self.prompt_tokens=0
        self.completion_tokens=0
        self.inference_seconds=0.0
        self.calls=0
    def metadata(self):
        return {
            "real_model":False, "prompt_tokens":self.prompt_tokens,
            "completion_tokens":self.completion_tokens,
            "inference_seconds":self.inference_seconds,
        }
    def chat(self,system,prompt,max_tokens=96):
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=4
        return '{"action":"verify"}'

class FollowCueFake(FakeOnly):
    def chat(self,system,prompt,max_tokens=96):
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=4
        match=re.search(r"NEXT EXPECTED PLAN STEP \(\d/4\): (\w+)",prompt)
        if match is None:
            match=re.search(r"NEXT ADMISSIBLE ACTION: (\w+)",prompt)
        tool=match.group(1) if match is not None else "verify"
        return json.dumps({"action":tool})

class F12NContracts(unittest.TestCase):
    def test_frozen_24_worlds_exclude_previous_four_studies(self):
        self.assertEqual(STUDY,"F12-N")
        self.assertEqual(len(ROLE_MAPS),24)
        self.assertEqual(len(set(ROLE_MAPS)),24)
        used=set(USED_J_MAPS)|set(USED_K_MAPS)|set(USED_L_MAPS)|set(USED_M_MAPS)
        self.assertFalse(set(ROLE_MAPS)&used)
        self.assertEqual(Counter(BIT_PAIRS),
                         {(0,0):6,(0,1):6,(1,0):6,(1,1):6})
        self.assertEqual(TOOLS,tuple(f"q{i}" for i in range(6)))
        self.assertEqual(ACTIONS,(*TOOLS,"verify"))
        self.assertEqual(ARMS,("F0","F1","F2","F3"))
        self.assertEqual(CONDITIONS,("stable","changed","forced"))
        self.assertEqual(HORIZON,7)
        self.assertEqual(SHARD_COUNT,4)

    def test_24_worlds_public_sqlite_and_reference_solutions(self):
        worlds=frozen_preflight()
        self.assertEqual(len(worlds),24)
        for w in worlds:
            self.assertEqual([r["source_id"] for r in w["observed"]],
                             [f"P{i}" for i in range(10)])
            self.assertFalse(any(e["action"]=="verify"
                                 for probe in w["observed"] for e in probe["events"]))
            self.assertLessEqual(len(w["context"]),3500)
            self.assertTrue(w["preflight"]["stable"]["success"])
            self.assertTrue(w["preflight"]["changed"]["success"])
            self.assertEqual(w["forced_prefix_responses"][-1],
                             {"error":"DISABLED_TOOL"})

    def test_cue_replacement_only_after_authentic_primary_failure(self):
        plan=grounded_plan(collect_probes(0))
        primary=plan["plan"][2]
        backup=plan["backup"]
        self.assertEqual(shown_action(plan,[],"F0"),plan["plan"][0])
        self.assertEqual(shown_action(plan,[],"F1"),plan["plan"][0])
        self.assertIsNone(shown_action(plan,[],"F2"))
        prefix=[
            {"origin":"llm","action":plan["plan"][0],"response":{"ack":True}},
            {"origin":"llm","action":plan["plan"][1],"response":{"value":0}},
        ]
        self.assertEqual(expected_index(plan["plan"],backup,prefix),2)
        self.assertEqual(shown_action(plan,prefix,"F1"),primary)
        self.assertFalse(has_public_disabled_primary(plan,prefix))
        normal=prefix+[{"origin":"llm","action":primary,"response":{"value":1}}]
        self.assertFalse(has_public_disabled_primary(plan,normal))
        precondition=prefix+[{"origin":"llm","action":primary,"response":{"error":"PRECONDITION"}}]
        self.assertFalse(has_public_disabled_primary(plan,precondition))
        unrelated=prefix+[{"origin":"llm","action":backup,"response":{"error":"DISABLED_TOOL"}}]
        self.assertFalse(has_public_disabled_primary(plan,unrelated))
        unauth=prefix+[{"action":primary,"response":{"error":"DISABLED_TOOL"}}]
        self.assertFalse(has_public_disabled_primary(plan,unauth))
        disabled=prefix+[{"origin":"llm","action":primary,"response":{"error":"DISABLED_TOOL"}}]
        self.assertEqual(expected_index(plan["plan"],backup,disabled),2)
        self.assertTrue(has_public_disabled_primary(plan,disabled))
        self.assertEqual(shown_action(plan,disabled,"F1"),backup)
        self.assertEqual(shown_action(plan,disabled,"F3"),backup)
        self.assertEqual(shown_action(plan,disabled,"F0"),primary)
        ungrounded=dict(plan,sources=["P0","P1"])
        self.assertFalse(has_public_disabled_primary(ungrounded,disabled))
        self.assertEqual(shown_action(ungrounded,disabled,"F1"),primary)
        succeeded=disabled+[{"origin":"llm","action":backup,"response":{"value":1}}]
        self.assertEqual(expected_index(plan["plan"],backup,succeeded),3)
        self.assertEqual(shown_action(plan,succeeded,"F1"),"verify")

    def test_stable_prompts_identical_and_error_changes_only_one_action_name(self):
        w=frozen_preflight()[0]
        plan=w["plan"]
        self.assertEqual(
            prompt_for("goal",w["context"],plan,[],"F0"),
            prompt_for("goal",w["context"],plan,[],"F1"))
        world=World(0,*BIT_PAIRS[0],"changed")
        history=[]
        for action in plan["plan"][:3]:
            history.append({"origin":"llm","action":action,
                            "response":world.execute(action)})
        self.assertEqual(history[-1]["response"],{"error":"DISABLED_TOOL"})
        fixed=prompt_for("goal",w["context"],plan,history,"F0")
        adaptive=prompt_for("goal",w["context"],plan,history,"F1")
        self.assertNotEqual(fixed,adaptive)
        self.assertEqual(len(fixed),len(adaptive))
        self.assertEqual(fixed.replace(
            f"STEP (3/4): {plan['plan'][2]}",
            f"STEP (3/4): {plan['backup']}"),adaptive)
        self.assertIn(f"PLAN POSITION (3/4). NEXT ADMISSIBLE ACTION: {plan['backup']}",
                      prompt_for("goal",w["context"],plan,history,"F3"))
        self.assertNotIn("NEXT EXPECTED PLAN STEP",
                         prompt_for("goal",w["context"],plan,history,"F2"))
        self.assertNotIn("hidden_bit_audit_only",adaptive)
        self.assertNotIn("roles_audit_only",adaptive)

    def test_fake_follow_cue_never_claims_llm_learning_and_forced_error_rescued(self):
        w=frozen_preflight()[0]
        pub,hid=BIT_PAIRS[0]
        stable_fixed=run_episode(0,"stable",pub,hid,FollowCueFake(),
                                 w["context"],w["plan"],"F0")
        stable_adaptive=run_episode(0,"stable",pub,hid,FollowCueFake(),
                                    w["context"],w["plan"],"F1")
        sig=lambda r:[(e["action"],e["response"],e["exact_prompt"])
                      for e in r["events"]]
        self.assertEqual(sig(stable_fixed),sig(stable_adaptive))
        self.assertTrue(stable_fixed["success"])
        fixed=run_episode(0,"forced",pub,hid,FollowCueFake(),
                          w["context"],w["plan"],"F0")
        adaptive=run_episode(0,"forced",pub,hid,FollowCueFake(),
                             w["context"],w["plan"],"F1")
        self.assertFalse(fixed["success"])
        self.assertTrue(adaptive["success"])
        self.assertEqual(fixed["prefixed_actions"],3)
        self.assertEqual(adaptive["prefixed_actions"],3)
        self.assertEqual(fixed["actions_total"],7)
        self.assertEqual(adaptive["actions_total"],5)
        self.assertEqual(adaptive["calls"],2)
        self.assertTrue(adaptive["recovered_after_error"])
        self.assertTrue(adaptive["switched_any"])
        self.assertFalse(adaptive["repeated_primary_posterror"])
        self.assertEqual(adaptive["events"][2]["origin"],"controlled_prefix")
        self.assertTrue(all(e["prompt_tokens"]==0
                            for e in adaptive["events"][:3]))

    def test_four_shard_fake_aggregate_blocks_missing_runs_and_science(self):
        reports=[]
        with TemporaryDirectory() as directory:
            path=Path(directory)
            for shard in range(SHARD_COUNT):
                data=run_shard(FakeOnly(),shard,path)
                self.assertEqual(len(data["worlds"]),6)
                self.assertEqual(len(data["outcomes"]),72)
                self.assertEqual(data["study_status"],
                                 "SHARD_ONLY_NO_SCIENTIFIC_CONCLUSION")
                self.assertTrue((path/f"f12n_shard{shard}.json").exists())
                reports.append(data)
            result=aggregate(reports,path)
            self.assertEqual(result["status"],"FAKE_TEST_ONLY")
            self.assertFalse(result["gates"]["N0_real_model_24_disjoint_worlds_288_complete"])
            self.assertTrue(result["gates"]["N1_public_reference_and_authentic_forced_errors"])
            self.assertTrue(result["gates"]["N2_stable_noop_and_no_premature_switch"])
            self.assertTrue(result["gates"]["N5_complete_provenance_and_budget"])
            self.assertFalse(result["scientific_pass"])
            self.assertFalse(result["product_integrated"])
            self.assertTrue((path/"f12n_full.json").exists())
            with self.assertRaises(ValueError):
                aggregate(reports[:3])
            with self.assertRaises(ValueError):
                aggregate([reports[0],reports[0],reports[2],reports[3]])
            for report in reports:
                for row in report["outcomes"]:
                    self.assertLessEqual(row["actions_total"],7)
                    if row["condition"]=="forced":
                        self.assertEqual(row["prefixed_actions"],3)
                        self.assertLessEqual(row["calls"],4)
                    for event in row["events"]:
                        if event["origin"]=="llm":
                            self.assertEqual(
                                hashlib.sha256(event["exact_prompt"].encode()).hexdigest(),
                                event["prompt_sha256"])

if __name__=="__main__":
    unittest.main()
