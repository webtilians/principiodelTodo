"""F12-O test contracts: prior-world replay gate, novel-world factor only if PASS."""
import json
import re
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory import f12o_repeatability_gate as replay
from observatory import f12o_cursor_factorial as factor

class Fake:
    real_model=False
    def __init__(self,follow_cue=False):
        self.prompt_tokens=0
        self.completion_tokens=0
        self.inference_seconds=0.0
        self.calls=0
        self.follow_cue=follow_cue
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
        if self.follow_cue:
            match=re.search(
                r"(?:NEXT EXPECTED PLAN STEP|NEXT ADMISSIBLE ACTION)"
                r"(?: \(\d/4\))?: (s[0-5]|verify)",prompt)
            if match:return json.dumps({"action":match.group(1)})
        return '{"action":"verify"}'

class F12OContracts(unittest.TestCase):
    def test_legacy_anchor_o21_exact_prompt_checksum_is_frozen(self):
        anchors=replay.build_anchors()
        self.assertEqual(len(anchors),6)
        self.assertEqual({a["id"] for a in anchors},{
            "O0-start","O0-disabled","O7-start","O7-disabled",
            "O21-start","O21-disabled"})
        self.assertEqual(next(a["prompt_sha256"] for a in anchors
                              if a["id"]=="O21-start"),
                         replay.O21_START_SHA)
        self.assertEqual(replay.REPEATS,5)
        self.assertEqual(replay.WORKERS,2)

    def test_fake_repeatability_gate_never_unlocks_new_worlds(self):
        a=replay.run_worker(Fake(),0)
        b=replay.run_worker(Fake(),1)
        self.assertEqual(len(a["replays"]),30)
        self.assertEqual(len(b["replays"]),30)
        result=replay.aggregate([a,b])
        self.assertEqual(result["status"],"FAKE_TEST_ONLY")
        self.assertFalse(result["gates"]["R0_exact_60_real_inferences_locked_sources"])
        self.assertTrue(result["gates"]["R1_literal_raw_equality_across_processes"])
        self.assertFalse(result["factor_worlds_evaluated"])
        with self.assertRaises(ValueError):replay.aggregate([a])
        with self.assertRaises(ValueError):replay.aggregate([a,a])

    def test_24_novel_maps_input_balance_and_preflight(self):
        self.assertEqual(factor.STUDY,"F12-O-F")
        self.assertEqual(len(factor.ROLE_MAPS),24)
        self.assertFalse(set(factor.ROLE_MAPS)&factor.USED)
        self.assertEqual(Counter(factor.BIT_PAIRS),
                         {(0,0):6,(0,1):6,(1,0):6,(1,1):6})
        self.assertEqual(factor.ARMS,("A00","A01","A10","A11"))
        self.assertEqual(factor.CONDITIONS,("stable","changed","forced"))
        self.assertEqual(factor.TOOLS,tuple(f"s{i}" for i in range(6)))
        worlds=factor.frozen_preflight()
        self.assertEqual(len(worlds),24)
        for w in worlds:
            self.assertTrue(w["preflight"]["stable"]["success"])
            self.assertTrue(w["preflight"]["changed"]["success"])
            self.assertLessEqual(w["preflight"]["changed"]["actions"],7)
            self.assertEqual(w["forced_prefix_responses"][-1],
                             {"error":"DISABLED_TOOL"})
            self.assertEqual([r["source_id"] for r in w["observed"]],
                             [f"P{i}" for i in range(10)])

    def test_factorial_prompts_are_orthogonal_and_same_public_rule(self):
        w=factor.frozen_preflight()[0]
        ps={a:factor.prompt_for(w["context"],w["plan"],[],a)
            for a in factor.ARMS}
        self.assertEqual(len(set(ps.values())),4)
        self.assertIn("NEXT EXPECTED PLAN STEP (1/4):",ps["A00"])
        self.assertIn("PLAN POSITION (1/4). NEXT EXPECTED PLAN STEP:",ps["A01"])
        self.assertIn("NEXT ADMISSIBLE ACTION (1/4):",ps["A10"])
        self.assertIn("PLAN POSITION (1/4). NEXT ADMISSIBLE ACTION:",ps["A11"])
        self.assertTrue(all("s0..s5" in p for p in ps.values()))
        self.assertTrue(all("roles_audit_only" not in p for p in ps.values()))
        self.assertTrue(all("hidden_bit_audit_only" not in p for p in ps.values()))
        world=factor.World(0,*factor.BIT_PAIRS[0],"changed")
        events=[]
        for action in w["plan"]["plan"][:3]:
            events.append({"origin":"controlled_prefix","action":action,
                           "response":world.execute(action)})
        self.assertTrue(factor.has_valid_public_error(w["plan"],events))
        for arm in factor.ARMS:
            self.assertEqual(factor.cue(w["plan"],events),w["plan"]["backup"])
            self.assertIn(w["plan"]["backup"],
                          factor.prompt_for(w["context"],w["plan"],events,arm))
        unrelated=events[:-1]+[
            {"origin":"controlled_prefix","action":w["plan"]["backup"],
             "response":{"error":"DISABLED_TOOL"}}]
        self.assertFalse(factor.has_valid_public_error(w["plan"],unrelated))

    def test_adaptive_prompt_success_with_fake_cue_follower_but_not_science(self):
        w=factor.frozen_preflight()[0]
        for c in ("stable","forced"):
            for arm in factor.ARMS:
                result=factor.run_episode(0,c,*factor.BIT_PAIRS[0],Fake(True),
                                          w["context"],w["plan"],arm)
                self.assertTrue(result["success"])
                self.assertLessEqual(result["actions_total"],7)
                if c=="forced":
                    self.assertEqual(result["prefixed_actions"],3)
                    self.assertEqual(result["calls"],2)
                    self.assertTrue(result["recovered_after_error"])
                else:
                    self.assertEqual(result["prefixed_actions"],0)
                    self.assertEqual(result["switches"],0)

    def test_fake_factor_shards_all_288_and_cannot_count_as_llm(self):
        shards=[]
        with TemporaryDirectory() as d:
            for shard in range(4):
                r=factor.run_shard(Fake(),shard,Path(d))
                self.assertEqual(len(r["worlds"]),6)
                self.assertEqual(len(r["outcomes"]),72)
                self.assertTrue(Path(d,f"f12o_factor_shard{shard}.json").exists())
                shards.append(r)
            summary=factor.aggregate(shards,Path(d))
            self.assertEqual(summary["status"],"FAKE_TEST_ONLY")
            self.assertFalse(summary["gates"]["F0_real_24_disjoint_288_complete"])
            self.assertTrue(summary["gates"]["F1_public_reference_and_forced_prefix"])
            self.assertTrue(summary["gates"]["F2_same_public_switch_stable_noop"])
            self.assertFalse(summary["scientific_pass"])
            self.assertFalse(summary["product_integrated"])
            self.assertTrue(Path(d,"f12o_factor_full.json").exists())
            with self.assertRaises(ValueError):factor.aggregate(shards[:3])
            with self.assertRaises(ValueError):
                factor.aggregate([shards[0],shards[0],shards[2],shards[3]])

if __name__=="__main__":unittest.main()
