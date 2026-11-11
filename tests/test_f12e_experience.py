"""F12-E scientific contracts (fake test model NEVER counts as evidence)."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f12e_experience import (
    ARMS, EVAL_BUDGETS, EVAL_CASES, FIXED_MODEL_REVISION,
    FIXED_MODEL_SHA256, MODEL_REPO, MODEL_FILE, MEMORY_LIMIT,
    SYSTEM, TOOLS, TRAIN_CASES, TRAIN_HORIZON, CompatibilityTools,
    parse_action, play, prompt_for, new_store, record_episode,
    render_episodic, render_strategy, promotion_evidence,
    frozen_content_snapshot, run,
)
from src.infinito3.types import MemoryKind


class FakeOnly:
    """Synthetic test model has no access to hidden compatibility data."""
    real_model = False

    def __init__(self):
        self.calls = 0
        self.syntheses = 0

    def chat(self, system, prompt, max_tokens=96):
        self.calls += 1
        if max_tokens == 512:
            self.syntheses += 1
            return "Hypothesis: check versioned successful experiences S0 and S1. May not transfer."
        # Only parse feedback in the current episode, not in reference memories.
        history = prompt.rsplit("OBSERVED TOOLS AND RESULTS IN THIS EPISODE:\n", 1)[-1]
        history = history.split("\nChoose exactly one", 1)[0]
        try:
            rows = json.loads(history)
        except (TypeError, ValueError):
            rows = []
        accepted = any("Patch accepted:" in x["observed_feedback"] for x in rows)
        rejected = any("Patch rejected:" in x["observed_feedback"] for x in rows)
        if accepted:
            return '{"action":"verify"}'
        if rejected:
            return '{"action":"apply_sigma"}'
        return '{"action":"apply_delta"}'

    def metadata(self):
        return {"real_model": False, "test_only": True}


class F12EScientificContracts(unittest.TestCase):
    def test_exact_proven_model_same_as_f12d_and_predeclared_splits(self):
        self.assertEqual(MODEL_REPO, "bartowski/Qwen2.5-7B-Instruct-GGUF")
        self.assertEqual(MODEL_FILE, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        self.assertEqual(FIXED_MODEL_REVISION, "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031")
        self.assertEqual(FIXED_MODEL_SHA256,
                         "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423")
        self.assertEqual([x.id for x in TRAIN_CASES], [f"S{i}" for i in range(6)])
        self.assertEqual([x.id for x in EVAL_CASES], [f"H{i}" for i in range(6)])
        self.assertFalse({x.id for x in TRAIN_CASES} & {x.id for x in EVAL_CASES})
        self.assertEqual([x.revision for x in TRAIN_CASES], ["R1"] * 3 + ["R2"] * 3)
        self.assertEqual(sum(x.revision == "R1" for x in EVAL_CASES), 3)
        self.assertEqual(sum(x.revision == "R2" for x in EVAL_CASES), 3)
        self.assertEqual({x.marker for x in TRAIN_CASES},
                         {x.marker for x in EVAL_CASES})
        self.assertEqual(set(ARMS), {"A0", "A1", "A2"})
        self.assertEqual(EVAL_BUDGETS, (2, 3))
        self.assertEqual(TRAIN_HORIZON, 4)
        self.assertEqual(MEMORY_LIMIT, 1200)

    def test_private_map_is_not_part_of_actor_prompts(self):
        for case in TRAIN_CASES + EVAL_CASES:
            prompt = prompt_for(case, [], "")
            self.assertIn(case.goal, prompt)
            self.assertNotIn("_compatible", prompt)
            self.assertNotIn("Case(", prompt)
            self.assertNotIn("evaluator truth", prompt)
            self.assertIn("untrusted", SYSTEM.lower())
            self.assertNotIn("S0", prompt)
            self.assertFalse(any(k in prompt for k in ("correct_operation", "hidden_mapping")))
            self.assertEqual(set(TOOLS), {"apply_delta", "apply_sigma", "inspect", "verify"})

    def test_tool_rejects_wrong_patch_and_requires_verify(self):
        for case in TRAIN_CASES + EVAL_CASES:
            tool = CompatibilityTools(case)
            wrong = "sigma" if case._compatible == "delta" else "delta"
            result, _, done = tool.execute("apply_" + wrong)
            self.assertIn("rejected", result)
            self.assertFalse(done)
            self.assertFalse(tool.execute("verify")[2])
            result, _, done = tool.execute("apply_" + case._compatible)
            self.assertIn("accepted", result)
            self.assertFalse(done)
            self.assertTrue(tool.execute("verify")[2])
            self.assertTrue(tool.verified)

    def test_invalid_json_consumes_action_no_heuristic_corrector(self):
        for raw in ("inspect", '{"tool":"apply_delta"}', '{"action":false}', ""):
            self.assertEqual(parse_action(raw), "INVALID_LLM_OUTPUT")
        self.assertEqual(parse_action('{"action":"apply_delta"}'), "apply_delta")
        with self.assertRaises(ValueError):
            play(TRAIN_CASES[0], FakeOnly(), "", budget=5)
        with self.assertRaises(ValueError):
            play(TRAIN_CASES[0], FakeOnly(), "a" * 1201, budget=2)

    def test_real_infinito_sqlite_contains_observations_not_private_truth(self):
        store = new_store()
        try:
            model = FakeOnly()
            for case in TRAIN_CASES:
                outcome = play(case, model, "", budget=4)
                record_episode(store, case, outcome)
            records = store.all()
            self.assertEqual(len(records), 6)
            self.assertTrue(all(r.kind == MemoryKind.EPISODIC for r in records))
            self.assertTrue(all("_compatible" not in r.content for r in records))
            self.assertEqual({r.metadata["source_id"] for r in records},
                             {f"S{i}" for i in range(6)})
            for case in EVAL_CASES:
                text, sources = render_episodic(store, case)
                self.assertLessEqual(len(text), MEMORY_LIMIT)
                self.assertTrue(sources)
                self.assertTrue(all(x["public_revision"] in ("R1", "R2") for x in sources))
                self.assertEqual({x["public_revision"] for x in sources}, {"R1", "R2"})
        finally:
            store.close()

    def test_policy_evidence_requires_success_and_real_source_citations(self):
        f = FakeOnly()
        a = [play(c, f, "", budget=4) for c in TRAIN_CASES]
        good = promotion_evidence("Hypothesis observed S0 and S1", a)
        self.assertTrue(good["minimum_evidence_met"])
        self.assertFalse(good["substantive_content_verified"])
        self.assertFalse(promotion_evidence("I think DELTA always works", a)
                         ["minimum_evidence_met"])
        failed = [{**r, "success": False} for r in a]
        self.assertFalse(promotion_evidence("S0 S1 S2", failed)["minimum_evidence_met"])

    def test_end_to_end_fake_never_passes_as_real_and_frozen_memory(self):
        with TemporaryDirectory() as dest:
            f = FakeOnly()
            result = run(f, Path(dest))
            self.assertEqual(result["status"], "FAKE_BACKEND_TEST_ONLY")
            self.assertEqual(result["episode_count"], 6)
            self.assertEqual(result["strategy_count"], 1)
            self.assertEqual(len(result["evaluation"]), 36)
            self.assertTrue(result["content_frozen_after_train"])
            self.assertFalse(result["gates"]["P0_real_frozen_model_and_no_leak"])
            self.assertFalse(result["scientific_pass"])
            self.assertFalse(result["dev_opened"])
            self.assertFalse(result["confirm_opened"])
            self.assertGreater(f.calls, 35)
            self.assertEqual(f.syntheses, 1)
            self.assertEqual(set(result["aggregate_scores"]), {"2", "3"})
            for budget in ("2", "3"):
                for arm in ARMS:
                    self.assertEqual(result["aggregate_scores"][budget][arm]["episodes"], 6)
            self.assertTrue(Path(dest, "pilot_report.json").is_file())
            self.assertTrue(Path(dest, "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
