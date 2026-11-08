"""F12-H isolated contracts; fake inference can NEVER count as scientific evidence."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f12h_effect_policy_transfer import (
    ACTIONS, ARMS, CATALOG, EVAL_BUDGETS, HELDOUT, MEMORY_LIMIT,
    TRAIN, ToolWorld, discover_effect_evidence, engineered_effect_plan,
    episodic_context, frozen_snapshot, has_train_composition, new_store,
    parse_action, parse_plan, play, plan_context, prompt_for, public_record,
    run, sources_from_memory, store_episode,
)


class FakeOnly:
    """Uses goal text and current PUBLIC feedback, not the private Case bits."""
    real_model = False

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.inference_seconds = 0.0

    def metadata(self):
        return {
            "real_model": False, "test_only": True,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "inference_seconds": self.inference_seconds,
        }

    def chat(self, system, prompt, max_tokens=96):
        self.calls += 1
        self.prompt_tokens += len(prompt) // 4
        self.completion_tokens += 14
        self.inference_seconds += .001
        if max_tokens == 192:
            if "[no episodes available]" in prompt:
                return '{"plan":[],"sources":[]}'
            return '{"plan":["op_s","op_r","op_x","verify"],"sources":["Z3","Z6"]}'
        is_hidden = "Use the hidden source" in prompt
        invert = "invert the bit" in prompt
        text = prompt.split("OBSERVED TOOL RESPONSES IN THIS EPISODE:\n", 1)[-1]
        rows = json.loads(text.split("\nChoose one action as strict JSON.", 1)[0])
        chosen = [x["action"] for x in rows]
        if is_hidden:
            if "op_s" not in chosen:
                next_action = "op_s"
            elif "op_r" not in chosen:
                next_action = "op_r"
            elif invert and "op_x" not in chosen:
                next_action = "op_x"
            else:
                next_action = "verify"
        else:
            if "op_p" not in chosen:
                next_action = "op_p"
            elif invert and "op_x" not in chosen:
                next_action = "op_x"
            else:
                next_action = "verify"
        return json.dumps({"action": next_action})


class F12HContracts(unittest.TestCase):
    def test_new_split_preserves_never_executed_hidden_flip(self):
        self.assertEqual(len(TRAIN), 9)
        self.assertEqual(len(HELDOUT), 6)
        self.assertEqual([x.source_id for x in TRAIN], [f"Z{i}" for i in range(9)])
        self.assertEqual([x.source_id for x in HELDOUT], [f"Q{i}" for i in range(6)])
        self.assertEqual(len({x.source_id for x in TRAIN + HELDOUT}), 15)
        self.assertEqual({(x.source, x.flip) for x in TRAIN},
                         {("public", 0), ("public", 1), ("hidden", 0)})
        self.assertEqual({(x.source, x.flip) for x in HELDOUT}, {("hidden", 1)})
        self.assertEqual([x._hidden_bit for x in HELDOUT], [0, 1, 0, 1, 0, 1])
        self.assertEqual(ARMS, ("A0", "A1", "A2", "A3", "A4"))
        self.assertEqual(EVAL_BUDGETS, (4, 6))
        self.assertEqual(MEMORY_LIMIT, 1600)

    def test_world_requires_observed_public_effects_and_verification(self):
        c = HELDOUT[0]  # hidden=0, target=1: known only to simulator
        w = ToolWorld(c)
        self.assertEqual(w.execute("op_r")[0], {"error": "unmet precondition"})
        self.assertEqual(w.execute("verify")[0], {"verified": False})
        self.assertEqual(w.execute("op_s")[0], {"ack": True})
        self.assertEqual(w.execute("op_r")[0], {"value": 0})
        self.assertEqual(w.execute("op_j")[0], {"value": 0})
        self.assertEqual(w.execute("verify")[0], {"verified": False})
        self.assertEqual(w.execute("op_x")[0], {"value": 1})
        self.assertEqual(w.execute("verify"), ({"verified": True}, True))

    def test_tool_parser_strict(self):
        for bad in ("op_r", '{"tool":"op_r"}', "abc", "",
                    '{"action":"op_r","private":"value"}',
                    '{"action":"secret"}'):
            self.assertEqual(parse_action(bad), "INVALID_LLM_OUTPUT")
        for action in ACTIONS:
            self.assertEqual(parse_action(json.dumps({"action": action})), action)

    def test_private_source_never_enters_memory_or_model_prompt(self):
        fake = FakeOnly()
        for case in TRAIN + HELDOUT:
            prompt = prompt_for(case, [], "")
            self.assertNotIn("_hidden_bit", prompt)
            self.assertNotIn("_goal", prompt)
            self.assertIn(CATALOG, prompt)
            self.assertIn(case.service, prompt)
            result = play(case, fake, budget=7)
            record = public_record(case, result)
            self.assertEqual(set(record), {
                "source_id", "source", "flip", "public_bit", "service",
                "goal", "events", "verified_success",
            })
            self.assertNotIn("_hidden_bit", json.dumps(record))
            self.assertTrue(result["success"])

    def test_genuine_sqlite_retrieval_covers_separate_skills(self):
        fake, store = FakeOnly(), new_store()
        try:
            for c in TRAIN:
                store_episode(store, public_record(c, play(c, fake, budget=7)))
            sources = sources_from_memory(store)
            self.assertEqual(len(sources), 9)
            shown, refs = episodic_context(store)
            self.assertLessEqual(len(shown), MEMORY_LIMIT)
            self.assertTrue(any(sources[x]["source"] == "hidden" for x in refs))
            self.assertTrue(any(sources[x]["flip"] == 1 for x in refs))
            self.assertFalse(any(has_train_composition(r) for r in sources.values()))
            evidence = discover_effect_evidence(sources)
            self.assertEqual(evidence["x_input_values_seen"], [0, 1])
            self.assertGreaterEqual(len(evidence["sr_source_ids"]), 2)
            planner = engineered_effect_plan(evidence)
            self.assertEqual(planner["plan"],
                             ["op_s", "op_r", "op_x", "verify"])
            self.assertTrue(planner["sources"])
        finally:
            store.close()

    def test_plan_contract_does_not_rescue_hallucination(self):
        allowed = {"Z0", "Z3", "Z6"}
        good = json.dumps({"plan": ["op_s", "op_r", "op_x", "verify"],
                           "sources": ["Z3", "Z6"]})
        self.assertEqual(parse_plan(good, allowed)["parse_status"], "OK")
        for raw in (
            '{"plan":["op_s","op_r","op_x"],"sources":["Z3"]}',
            '{"plan":["op_s","op_r","op_x","verify"],"sources":["Y0"]}',
            '{"plan":["op_s","op_r","op_x","verify"],"sources":[]}',
            '{"plan":["op_s","delete","verify"],"sources":["Z0"]}',
            '{"plan":[],"sources":[],"oracle":true}',
            "do it",
        ):
            self.assertNotEqual(parse_plan(raw, allowed)["parse_status"], "OK")
        self.assertEqual(parse_plan('{"plan":[],"sources":[]}', allowed)["plan"], [])

    def test_fake_end_to_end_complete_but_no_scientific_go(self):
        with TemporaryDirectory() as tmp:
            report = run(FakeOnly(), Path(tmp))
            self.assertEqual(report["status"], "FAKE_TEST_ONLY")
            self.assertFalse(report["gates"]["P0_integrity_9_train_60_eval_frozen_real_model"])
            self.assertEqual(len(report["outcomes"]), 60)
            self.assertTrue(report["memory_frozen"])
            self.assertFalse(report["scientific_pass"])
            self.assertFalse(report["confirm_opened"])
            self.assertTrue(Path(tmp, "f12h_full_report.json").exists())
            for b in EVAL_BUDGETS:
                for arm in ARMS:
                    self.assertEqual(report["scores"][str(b)][arm]["episodes"], 6)
            for episode in report["outcomes"]:
                self.assertLessEqual(len(episode["context_text"]), MEMORY_LIMIT)
                self.assertLessEqual(episode["tool_calls"], episode["budget"])
                self.assertNotIn("_hidden_bit", episode["context_text"])


if __name__ == "__main__":
    unittest.main()
