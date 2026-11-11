"""F12-B technical contracts, no model download and no fake scientific results."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f12b_llm_competence import (
    ACTIONS, CASES, MODEL, MAX_ACTIONS, TOOL_CATALOG, SYSTEM,
    BlockedModel, FrozenQwen, SimulatedTools, decode_action,
    prompt_for, run_case, run_preflight,
)


class TestBackend:
    real_model = False

    def __init__(self, invalid=False):
        self.invalid = invalid
        self.calls = 0

    def chat(self, system, message):
        self.calls += 1
        return '{"action":"inspect"}' if not self.invalid else "inspect"


class F12BContracts(unittest.TestCase):
    def test_model_is_larger_and_strictly_frozen_real_inference(self):
        self.assertEqual(MODEL, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(MAX_ACTIONS, 6)
        with self.assertRaises(BlockedModel):
            FrozenQwen("Qwen/Qwen2.5-0.5B-Instruct")

    def test_four_fresh_competence_cases_balanced_and_disjoint_from_f12a(self):
        self.assertEqual(len(CASES), 4)
        self.assertEqual([c.case_id for c in CASES], ["K0", "K1", "K2", "K3"])
        self.assertEqual(sum(c.hidden_fault == "access" for c in CASES), 2)
        self.assertEqual(sum(c.hidden_fault == "dependency" for c in CASES), 2)
        self.assertFalse({c.case_id for c in CASES} & {"T0", "T1", "P0", "P1"})

    def test_hidden_cause_never_appears_as_explicit_label_in_prompt(self):
        for case in CASES:
            prompt = prompt_for(case, [])
            self.assertNotIn("hidden_fault", prompt)
            self.assertNotIn("Case(", prompt)
            self.assertNotIn('fault":', prompt)
            self.assertIn("ALLOWED TOOLS:", prompt)
            self.assertIn("TOOL CONTRACT:", prompt)
            self.assertIn("OBSERVED TOOLS SO FAR:", prompt)
            self.assertIn("untrusted", SYSTEM.lower())
            self.assertEqual(len(ACTIONS), 6)

    def test_tools_require_audit_and_verification_without_real_operations(self):
        for case in CASES:
            sim = SimulatedTools(case)
            target = case.hidden_fault
            self.assertFalse(sim.execute("verify")[2])
            self.assertFalse(sim.execute("repair_" + target)[2])
            self.assertFalse(sim.execute("unknown")[2])
            self.assertFalse(sim.execute("inspect")[2])
            self.assertFalse(sim.execute("audit_" + target)[2])
            self.assertFalse(sim.execute("repair_" + target)[2])
            self.assertTrue(sim.execute("verify")[2])
            self.assertTrue(sim.verified)

    def test_invalid_actions_not_corrected_into_valid_actions(self):
        for raw in ("inspect", "code fence json action inspect",
                    '{"tool":"inspect"}', '{"action":23}', '{"action":null}', ""):
            self.assertEqual(decode_action(raw), "INVALID_OUTPUT")
        self.assertEqual(decode_action('{"action":"inspect"}'), "inspect")
        sim = SimulatedTools(CASES[0])
        self.assertEqual(sim.execute(decode_action("inspect"))[1], -.20)

    def test_fake_backend_cannot_yield_real_competence_go(self):
        fake = TestBackend()
        with TemporaryDirectory() as d:
            report = run_preflight(fake, Path(d))
            self.assertEqual(report["status"], "FAKE_BACKEND_TEST_ONLY")
            self.assertFalse(report["gates"]["G0_live_model"])
            self.assertEqual(len(report["cases"]), 4)
            self.assertFalse(report["train_memory_used"])
            self.assertFalse(report["policy_extraction"])
            self.assertFalse(report["fine_tuning"])
            self.assertFalse(report["dev_opened"])
            self.assertFalse(report["confirm_opened"])
            self.assertTrue(Path(d, "summary.json").exists())
            self.assertTrue(Path(d, "preflight_report.json").exists())

    def test_no_action_after_success_and_max_budget(self):
        bad = TestBackend(invalid=True)
        res = run_case(CASES[0], bad)
        self.assertFalse(res["success"])
        self.assertEqual(res["call_count"], MAX_ACTIONS)
        self.assertEqual(res["valid_actions"], 0)
        self.assertEqual(res["tool_actions"], ["INVALID_OUTPUT"] * MAX_ACTIONS)
        self.assertTrue(all("llm_response" in e and "observation" in e
                            and "prompt_sha256" in e for e in res["observations"]))
        with self.assertRaises(ValueError):
            run_case(CASES[0], bad, 10)

    def test_preflight_prompt_includes_actual_feedback_but_no_hidden_cause(self):
        context = [{"step": 0, "llm_action": "inspect", "observation": "test finding",
                    "reward": -.01, "done": False}]
        prompt = prompt_for(CASES[0], context)
        self.assertIn("test finding", prompt)
        self.assertIn("audit_access", TOOL_CATALOG)
        self.assertIn("audit_dependency", TOOL_CATALOG)
        self.assertNotIn("hidden_fault", prompt)


if __name__ == "__main__":
    unittest.main()
