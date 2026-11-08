"""F12-I contracts, no synthetic model allowed to count as real inference."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from observatory.f12i_observed_plan_induction import (
    ALLOWED, ARMS, CONTEXT_LIMIT, ROLE_MAPS, PROBES, World,
    collect_probes, new_store, save_probes, from_memory, render,
    masked_flip_public, public_effect_planner, eval_offline,
    parse_plan, run,
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
    def chat(self,sys,prompt,max_tokens=192):
        self.calls+=1
        self.prompt_tokens+=len(prompt)//4
        self.completion_tokens+=30
        if "[none]" in prompt:
            return '{"plan":["t0","t1","t2","verify"],"sources":[]}'
        return '{"plan":["t0","t1","t2","verify"],"sources":["OBS0","OBS4"]}'

class F12IContracts(unittest.TestCase):
    def test_fixed_worlds_are_permutations_and_probes_never_compose_target(self):
        self.assertEqual(len(ROLE_MAPS),8)
        self.assertEqual(len(PROBES),10)
        for roles in ROLE_MAPS:
            self.assertEqual(set(roles),{"LOAD","ENABLE","READ","FLIP","HOLD"})
        self.assertFalse(any(p==("ENABLE","READ","FLIP","verify") for p in PROBES))
        self.assertEqual(ARMS,("A0","A1","A2","A3"))
    def test_no_hidden_map_in_public_memory(self):
        for wid in range(8):
            records=collect_probes(wid)
            self.assertEqual([r["source_id"] for r in records],
                             [f"OBS{i}" for i in range(10)])
            for r in records:
                self.assertEqual(set(r),{"source_id","public_input","events"})
                self.assertNotIn("hidden",json.dumps(r))
                self.assertNotIn("roles",json.dumps(r))
            self.assertLessEqual(len(render(records)),CONTEXT_LIMIT)
    def test_sqlite_retrieval_real_and_positive_control(self):
        for wid in range(8):
            store=new_store()
            try:
                save_probes(store,collect_probes(wid))
                records=from_memory(store)
                self.assertEqual(len(records),10)
                plan=public_effect_planner(records)
                self.assertEqual(plan["status"],"ENGINEERED_EFFECT_INFERENCE")
                self.assertEqual(len(plan["plan"]),4)
                self.assertEqual(sum(x["success"] for x in eval_offline(wid,plan["plan"])),4)
                masked=masked_flip_public(records)
                self.assertEqual(masked[2]["events"],records[2]["events"][:1])
                self.assertEqual(masked[3]["events"],records[3]["events"][:1])
                self.assertEqual(len(masked),10)
            finally:
                store.close()
    def test_strict_invalid_plan_and_no_fabricated_source(self):
        for s in ('{"plan":["t0","verify"],"sources":["FAKE"]}',
                  '{"plan":["t0"],"sources":["OBS0"]}',
                  '{"plan":["unknown","verify"],"sources":["OBS0"]}',
                  '{"plan":[],"sources":[]}',
                  '{"plan":["t0","verify"],"sources":[]}',
                  "not json"):
            self.assertNotEqual(parse_plan(s,{"OBS0"},True)["status"],"OK")
        good=parse_plan('{"plan":["t0","verify"],"sources":["OBS0"]}',{"OBS0"},True)
        self.assertEqual(good["status"],"OK")
        self.assertEqual(eval_offline(0,[])[0]["success"],False)
    def test_fake_runs_complete_but_no_scientific_pass(self):
        with TemporaryDirectory() as d:
            result=run(Fake(),Path(d))
            self.assertEqual(result["status"],"FAKE_TEST_ONLY")
            self.assertFalse(result["gates"]["G0_real_pinned_8_worlds_32_plans_128_eval"])
            self.assertEqual(len(result["worlds"]),8)
            self.assertEqual(sum(len(w["scores"][a]["trials"])
                                 for w in result["worlds"] for a in ARMS),128)
            self.assertTrue(result["gates"]["G3_engineered_planner_positive_control"])
            self.assertFalse(result["scientific_pass"])
            self.assertTrue(Path(d,"f12i_full_report.json").is_file())

if __name__=="__main__":
    unittest.main()
