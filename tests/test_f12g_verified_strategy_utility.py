"""F12-G contracts: fake agents never count as actual LLM results."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f12g_verified_strategy_utility import (
    TRAIN, HELDOUT, ARMS, EVAL_BUDGETS, CONTEXT_LIMIT,
    compact_episode, store_strategy, gather_references, new_store,
    make_context, eval_case, eval_prompt, run,
)
from observatory.f12f_evidence_validator import (
    FIXED_MODEL_SHA256, FIXED_MODEL_REVISION,
    public_record, store_experience, validate_evidence,
)
from src.infinito3.types import MemoryKind


class FakeOnly:
    real_model = False

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.inference_seconds = 0.

    def chat(self, system, prompt, max_tokens=96):
        self.calls += 1
        self.prompt_tokens += len(prompt) // 4
        self.completion_tokens += 9
        self.inference_seconds += 0.01
        if max_tokens == 192:
            data = json.loads(prompt)
            events = data["events"]
            accepted = [
                e for e in events
                if e["action"] in ("apply_delta", "apply_sigma")
                and "Patch accepted:" in e["feedback"]]
            if not accepted:
                return "{}"
            item = accepted[-1]
            idx = events.index(item)
            verified = any(
                e["action"] == "verify"
                and "Verification passed:" in e["feedback"]
                for e in events[idx + 1:])
            return json.dumps({
                "source_id": data["source_id"], "profile": data["profile"],
                "revision": data["revision"],
                "operation": item["action"].removeprefix("apply_"),
                "observed_outcome": "accepted",
                "verified_after_operation": verified,
            })
        if "OBSERVED TOOL RESPONSES:" in prompt:
            history = prompt.split("OBSERVED TOOL RESPONSES:\n")[-1].split(
                "\nNext tool, JSON only.")[0]
            rows = json.loads(history)
            if not rows:
                return '{"action":"inspect"}'
            if any("Patch accepted:" in x["tool_feedback"] for x in rows):
                return '{"action":"verify"}'
            if any("Patch rejected:" in x["tool_feedback"] for x in rows):
                return '{"action":"apply_sigma"}'
            return '{"action":"apply_delta"}'
        if "OBSERVED EVENTS IN THE CURRENT EPISODE:" in prompt:
            history = prompt.split("OBSERVED EVENTS IN THE CURRENT EPISODE:\n")[-1].split(
                "\nChoose the next single", 1)[0]
            rows = json.loads(history)
            if not rows:
                return '{"action":"apply_delta"}'
            if any("Patch accepted:" in x["feedback"] for x in rows):
                return '{"action":"verify"}'
            if any("Patch rejected:" in x["feedback"] for x in rows):
                return '{"action":"apply_sigma"}'
            return '{"action":"verify"}'
        raise AssertionError("Unexpected fake prompt")

    def metadata(self):
        return {"real_model": False, "model": "fake-only"}


class F12GContracts(unittest.TestCase):
    def test_new_cases_public_versions_and_disjoint_splits(self):
        self.assertEqual(len(TRAIN), 8)
        self.assertEqual(len(HELDOUT), 8)
        self.assertEqual([c.source_id for c in TRAIN], [f"W{i}" for i in range(8)])
        self.assertEqual([c.source_id for c in HELDOUT], [f"Y{i}" for i in range(8)])
        self.assertEqual(len({x.source_id for x in TRAIN + HELDOUT}), 16)
        self.assertEqual({c.profile for c in TRAIN},
                         {"quartz", "reed", "cobalt", "dusk"})
        self.assertEqual({c.profile for c in TRAIN}, {c.profile for c in HELDOUT})
        self.assertEqual([c.revision for c in TRAIN], ["R1"] * 4 + ["R2"] * 4)
        self.assertEqual(sum(c.revision == "R1" for c in HELDOUT), 4)
        self.assertEqual(sum(c.revision == "R2" for c in HELDOUT), 4)
        self.assertEqual(ARMS, ("A0", "A1", "A2", "A3", "A4"))
        self.assertEqual(EVAL_BUDGETS, (2, 3))
        self.assertEqual(CONTEXT_LIMIT, 1200)
        self.assertEqual(FIXED_MODEL_REVISION,
                         "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031")
        self.assertEqual(FIXED_MODEL_SHA256,
                         "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423")

    def test_episode_rendering_is_source_faithful_not_truth_label(self):
        source = {
            "source_id": "W0", "profile": "quartz", "revision": "R1",
            "verified_success": True, "events": [
                {"action": "apply_delta", "feedback": "Patch rejected: incompatible with this revision."},
                {"action": "apply_sigma", "feedback": "Patch accepted: compatible with this revision."},
                {"action": "verify", "feedback": "Verification passed: service compatible."}]}
        text = compact_episode(source)
        self.assertIn("apply_delta=rejected", text)
        self.assertIn("apply_sigma=accepted", text)
        self.assertIn("verify=verified_success", text)
        self.assertNotIn("_compatible", text)

    def test_strong_episodic_contains_both_versions(self):
        eps, raw, valid = new_store(), new_store(), new_store()
        try:
            for source_id, version in (("W0", "R1"), ("W4", "R2")):
                public = {
                    "source_id": source_id, "profile": "quartz",
                    "revision": version, "service": "public-only",
                    "goal": "public revision", "verified_success": True,
                    "events": [
                        {"step": 0, "action": "apply_delta",
                         "feedback": "Patch accepted: compatible with this revision.",
                         "reward": -.01, "done": False},
                        {"step": 1, "action": "verify",
                         "feedback": "Verification passed: service compatible.",
                         "reward": 1., "done": True}]}
                store_experience(eps, public)
            ref, sources, _ = make_context(eps, raw, valid, "quartz", "A1")
            self.assertIn("revision=R1", ref)
            self.assertIn("revision=R2", ref)
            self.assertLessEqual(len(ref), CONTEXT_LIMIT)
            self.assertEqual(len(sources), 2)
            recent, _, _ = make_context(eps, raw, valid, "quartz", "A4")
            self.assertIn("revision=R2", recent)
            self.assertNotIn("revision=R1", recent)
        finally:
            eps.close()
            raw.close()
            valid.close()

    def test_wrong_claim_is_filtered_and_not_removed_from_raw_arm(self):
        eps, raw, valid = new_store(), new_store(), new_store()
        try:
            observed = {
                "source_id": "W0", "profile": "quartz", "revision": "R1",
                "service": "service", "goal": "goal", "verified_success": True,
                "events": [
                    {"step": 0, "action": "apply_delta",
                     "feedback": "Patch accepted: compatible with this revision.",
                     "reward": -.01, "done": False},
                    {"step": 1, "action": "verify",
                     "feedback": "Verification passed: service compatible.",
                     "reward": 1., "done": True},
                ]}
            store_experience(eps, observed)
            wrong = {
                "source_id": "W0", "profile": "quartz",
                "revision": "R1", "operation": "sigma",
                "observed_outcome": "accepted", "verified_after_operation": True,
            }
            accepted = {
                **wrong, "operation": "delta",
            }
            source_db = {"W0": observed}
            self.assertFalse(validate_evidence(wrong, source_db)["promotable"])
            self.assertTrue(validate_evidence(accepted, source_db)["promotable"])
            store_strategy(raw, wrong, "llm_unverified")
            store_strategy(valid, accepted, "source_verified_only")
            unverified, _, _ = make_context(eps, raw, valid, "quartz", "A2")
            verified, _, _ = make_context(eps, raw, valid, "quartz", "A3")
            self.assertIn("sigma", unverified)
            self.assertIn("delta", verified)
            self.assertTrue(len(unverified) <= CONTEXT_LIMIT)
            self.assertTrue(len(verified) <= CONTEXT_LIMIT)
        finally:
            eps.close()
            raw.close()
            valid.close()

    def test_identical_tools_and_budget_cannot_overrun(self):
        fake = FakeOnly()
        case = HELDOUT[0]
        a = eval_case(case, fake, "", budget=2)
        b = eval_case(case, fake, "public memory", budget=3)
        self.assertLessEqual(a["calls"], 2)
        self.assertLessEqual(b["calls"], 3)
        self.assertTrue(all("raw_response" in e and "prompt_sha256" in e
                            and "input_tokens" in e for e in a["events"]))
        with self.assertRaises(ValueError):
            eval_case(case, fake, "", budget=4)
        with self.assertRaises(ValueError):
            eval_case(case, fake, "x" * 1201, budget=2)
        self.assertNotIn("_compatible", eval_prompt(case, [], ""))

    def test_80_evaluations_no_fake_scientific_pass_and_snapshot(self):
        with TemporaryDirectory() as dirname:
            model = FakeOnly()
            result = run(model, Path(dirname))
            self.assertEqual(result["status"], "FAKE_TEST_BACKEND_ONLY")
            self.assertEqual(len(result["train_sources"]), 8)
            self.assertEqual(len(result["outcomes"]), 80)
            self.assertFalse(result["gates"]["P0_real_model_complete_and_memory_frozen"])
            self.assertTrue(result["memory_frozen_after_train"])
            self.assertFalse(result["scientific_pass"])
            self.assertFalse(result["dev_opened"])
            self.assertFalse(result["confirm_opened"])
            self.assertEqual(len(result["exact_rendered_contexts"]), 8)
            self.assertTrue((Path(dirname) / "f12g_full_report.json").is_file())
            for b in ("2", "3"):
                for a in ARMS:
                    self.assertEqual(result["scores"][b][a]["episodes"], 8)


if __name__ == "__main__":
    unittest.main()
