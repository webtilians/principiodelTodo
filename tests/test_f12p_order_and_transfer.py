"""F12-P engineering contracts. Fake model never authorizes a science gate."""
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory import f12p_order_history_gate as h
from observatory import f12p_cross_family_wording_transfer as t

class Fake:
    real_model=False
    def __init__(self,follow=False):
        self.prompt_tokens=0
        self.completion_tokens=0
        self.inference_seconds=0.0
        self.follow=follow
    def metadata(self):
        return {"real_model":False,"prompt_tokens":self.prompt_tokens,
                "completion_tokens":self.completion_tokens,
                "inference_seconds":self.inference_seconds}
    def chat(self,system,prompt,max_tokens=96):
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=4
        if self.follow:
            match=re.search(
                r"NEXT (?:EXPECTED PLAN STEP|ADMISSIBLE ACTION|ACTION) "
                r"\(\d/\d\): (r[0-6]|verify)",prompt)
            if match:return json.dumps({"action":match.group(1)})
        return '{"action":"q0"}' if '"action":"q0"' in system else '{"action":"verify"}'

class F12PContracts(unittest.TestCase):
    def test_history_72_unique_interleaved_positions(self):
        anchors=h.anchors()
        self.assertEqual(len(anchors),6)
        self.assertEqual(h.ORDERS,((0,1,2,3,4,5),(5,4,3,2,1,0),(2,5,0,3,1,4)))
        a=h.run_worker(Fake(),0)
        b=h.run_worker(Fake(),1)
        self.assertEqual(len(a["rows"]),36)
        self.assertEqual(len(b["rows"]),36)
        out=h.aggregate([a,b])
        self.assertEqual(out["status"],"FAKE_TEST_ONLY")
        self.assertFalse(out["gates"]["H0_real_72_interleaved_frozen_prompts"])
        self.assertTrue(out["gates"]["H1_literal_output_invariant_to_order_and_worker"])
        self.assertEqual(out["unique_raw_outputs"],{x["id"]:1 for x in anchors})
        self.assertFalse(out["transfer_worlds_evaluated"])
        with self.assertRaises(ValueError):h.aggregate([a])
        with self.assertRaises(ValueError):h.aggregate([a,a])

    def test_distinct_family_schemas_and_public_probes(self):
        self.assertEqual(t.STUDY,"F12-P-T")
        self.assertEqual(t.FAMILIES,("auth","deps"))
        self.assertEqual(t.ARMS,("E","A","N"))
        self.assertEqual(t.CONDITIONS,("stable","changed","forced"))
        self.assertEqual(t.TOOLS,tuple(f"r{i}" for i in range(7)))
        self.assertEqual(len(t.MAPS["auth"]),12)
        self.assertEqual(len(t.MAPS["deps"]),12)
        self.assertEqual(len(set(t.MAPS["auth"])),12)
        self.assertEqual(len(set(t.MAPS["deps"])),12)
        self.assertEqual(len(t.AUTH_ROLES),7)
        self.assertEqual(len(t.DEPS_ROLES),7)
        worlds=t.preflight()
        self.assertEqual(len(worlds),24)
        for w in worlds:
            self.assertEqual([r["source_id"] for r in w["observed"]],
                             [f"P{i}" for i in range(6)])
            self.assertTrue(w["preflight"]["stable"]["success"])
            self.assertTrue(w["preflight"]["changed"]["success"])
            self.assertLessEqual(w["preflight"]["changed"]["actions"],7)
            self.assertEqual(w["forced_prefix_feedback"][-1],
                             {"error":t.FAILURE[w["family"]]})
            self.assertNotIn("roles_audit_only",w["context"])

    def test_3_text_treatments_and_adaptive_public_error(self):
        for wid in (0,12):
            world=t.preflight()[wid]
            fam=world["family"]
            plan=world["plan"]
            prompts={arm:t.prompt_for(fam,world["context"],plan,[],arm)
                     for arm in t.ARMS}
            self.assertEqual(len(set(prompts.values())),3)
            self.assertIn("NEXT EXPECTED PLAN STEP (1/",prompts["E"])
            self.assertIn("NEXT ADMISSIBLE ACTION (1/",prompts["A"])
            self.assertIn("NEXT ACTION (1/",prompts["N"])
            self.assertTrue(all("roles_audit_only" not in p for p in prompts.values()))
            self.assertFalse(t.has_public_error(fam,plan,[]))
            w=t.World(wid,"changed")
            history=[]
            for action in plan["plan"][:-1]:
                history.append({
                    "origin":"controlled_prefix","action":action,
                    "response":w.execute(action),
                })
            self.assertEqual(history[-1]["response"],{"error":t.FAILURE[fam]})
            self.assertTrue(t.has_public_error(fam,plan,history))
            self.assertEqual(t.action_cue(fam,plan,history),plan["backup"])
            for arm in t.ARMS:
                self.assertIn(plan["backup"],
                              t.prompt_for(fam,world["context"],plan,history,arm))
            altered=history[:-1]+[{
                "origin":"controlled_prefix","action":plan["plan"][-2],
                "response":{"error":"NOT_LOCKED" if fam=="deps" else "UNAUTHORIZED"}
            }]
            self.assertFalse(t.has_public_error(fam,plan,altered))

    def test_fake_follower_solves_both_new_families_without_learning(self):
        worlds=t.preflight()
        for wid in (0,3,12,20):
            w=worlds[wid]
            for cond in t.CONDITIONS:
                for arm in t.ARMS:
                    result=t.run_episode(
                        wid,cond,w["context"],w["plan"],arm,Fake(True))
                    self.assertTrue(result["success"])
                    self.assertLessEqual(result["actions_total"],7)
                    if cond=="forced":
                        self.assertEqual(result["prefix_actions"],
                                         len(w["plan"]["plan"])-1)
                        self.assertEqual(result["llm_calls"],2)
                        self.assertTrue(result["recovered_after_error"])

    def test_fake_transfer_216_valid_ids_no_scientific_go(self):
        shards=[]
        with TemporaryDirectory() as directory:
            root=Path(directory)
            for shard in range(4):
                report=t.run_shard(Fake(),shard,root)
                self.assertEqual(len(report["outcomes"]),54)
                self.assertEqual(len(report["worlds"]),6)
                self.assertTrue((root/f"f12p_transfer_shard{shard}.json").exists())
                shards.append(report)
            total=t.aggregate(shards,root)
            self.assertEqual(total["status"],"FAKE_TEST_ONLY")
            self.assertFalse(total["gates"]["T0_real_24_new_family_worlds_216_complete"])
            self.assertTrue(total["gates"]["T1_public_reference_and_forced_error"])
            self.assertTrue(total["gates"]["T2_only_public_errors_switch_no_stable_switch"])
            self.assertTrue(total["gates"]["T5_trace_costs_and_no_private_leaks"])
            self.assertFalse(total["scientific_pass"])
            self.assertFalse(total["product_integrated"])
            self.assertTrue((root/"f12p_transfer_full.json").exists())
            with self.assertRaises(ValueError):t.aggregate(shards[:3])
            with self.assertRaises(ValueError):
                t.aggregate([shards[0],shards[0],shards[2],shards[3]])

if __name__=="__main__":unittest.main()
