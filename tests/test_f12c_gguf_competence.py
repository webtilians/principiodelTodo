"""F12-C contracts: no private oracle, no fake inference counted as progress."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f12c_gguf_competence import (
    ACTIONS, CASES, MODEL_FILE, MODEL_REPO, QUANTIZATION, HORIZON,
    FrozenGGUF, ProviderBlocked, InMemoryTools, action_from_text,
    prompt_for, execute_case, run,
)

class FakeLLM:
    real_model = False

    def __init__(self):
        self.calls = 0

    def chat(self, system, prompt):
        self.calls += 1
        return '{"action":"inspect"}'

    def metadata(self):
        return {"real_model": False}


class F12CContracts(unittest.TestCase):
    def test_pinned_model_and_disjoint_new_cases(self):
        self.assertEqual(MODEL_REPO, "bartowski/Qwen2.5-3B-Instruct-GGUF")
        self.assertEqual(MODEL_FILE, "Qwen2.5-3B-Instruct-Q5_K_M.gguf")
        self.assertEqual(QUANTIZATION, "Q5_K_M")
        self.assertEqual([c.id for c in CASES], [f"L{i}" for i in range(6)])
        self.assertEqual(sum(c.hidden_fault == "access" for c in CASES), 3)
        self.assertEqual(sum(c.hidden_fault == "dependency" for c in CASES), 3)
        self.assertFalse({c.id for c in CASES} & {"T0","T1","P0","P1","K0","K1","K2","K3"})
        self.assertEqual(HORIZON, 6)
        with self.assertRaises(ProviderBlocked):
            FrozenGGUF(repo="different-repo")

    def test_prompts_contain_only_public_goal_tools_feedback(self):
        for case in CASES:
            text = prompt_for(case, [])
            self.assertNotIn("hidden_fault", text)
            self.assertNotIn("Case(", text)
            self.assertNotIn("evaluator", text)
            self.assertIn(case.report, text)
            self.assertIn("TOOL DESCRIPTIONS", text)
            self.assertIn("OBSERVATIONS", text)
            self.assertNotIn("optimal action", text)

    def test_synthetic_tools_require_correct_audit_then_verify(self):
        for case in CASES:
            sim = InMemoryTools(case)
            typ = case.hidden_fault
            wrong = "dependency" if typ == "access" else "access"
            self.assertFalse(sim.execute("verify")[2])
            self.assertFalse(sim.execute("repair_" + typ)[2])
            self.assertFalse(sim.execute("audit_" + wrong)[2])
            self.assertFalse(sim.execute("repair_" + wrong)[2])
            self.assertFalse(sim.execute("inspect")[2])
            self.assertFalse(sim.execute("audit_" + typ)[2])
            self.assertFalse(sim.execute("repair_" + typ)[2])
            self.assertTrue(sim.execute("verify")[2])
            self.assertTrue(sim.verified)
            self.assertEqual(len(ACTIONS), 6)

    def test_exact_json_no_heuristic_action_recovery(self):
        self.assertEqual(action_from_text('{"action":"audit_access"}'), "audit_access")
        for text in ("audit_access", '{"tool":"audit_access"}',
                     '{"action":null}', '{"action":12}', "fenced-json syntax", ""):
            self.assertEqual(action_from_text(text), "INVALID_LLM_OUTPUT")
        self.assertFalse(InMemoryTools(CASES[0]).execute("INVALID_LLM_OUTPUT")[2])

    def test_fake_agent_never_satisfies_real_gate(self):
        fake = FakeLLM()
        with TemporaryDirectory() as folder:
            result = run(fake, Path(folder))
            self.assertEqual(result["status"], "FAKE_BACKEND_TEST_ONLY")
            self.assertFalse(result["gates"]["G0_genuine_pinned_model"])
            self.assertEqual(len(result["cases"]), 6)
            self.assertFalse(result["memory_used"])
            self.assertFalse(result["learning_used"])
            self.assertFalse(result["dev_opened"])
            self.assertFalse(result["confirm_opened"])
            self.assertTrue(Path(folder, "preflight_report.json").exists())
            self.assertTrue(Path(folder, "summary.json").exists())
            self.assertEqual(result["total_calls"], 36)

    def test_trace_provenance_fully_visible_no_privileged_label(self):
        fake = FakeLLM()
        result = execute_case(CASES[1], fake)
        self.assertEqual(result["calls"], HORIZON)
        self.assertEqual(result["valid_calls"], HORIZON)
        for step in result["observations"]:
            self.assertEqual(step["llm_action"], "inspect")
            self.assertIn("prompt_sha256", step)
            self.assertIn("llm_response", step)
            self.assertIn("observed_reward", step)
            self.assertNotIn("hidden_fault", step)
            self.assertNotIn("optimal_action", step)
        self.assertEqual(fake.calls, HORIZON)


if __name__ == "__main__":
    unittest.main()
