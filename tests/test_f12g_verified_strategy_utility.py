"""F12-G contracts; fake backend may test mechanics but never LLM evidence."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f12g_verified_strategy_utility import (
    ARMS, EVAL_BUDGETS, FIXED_MODEL_REVISION,
    FIXED_MODEL_SHA256, HELDOUT_CASES, MAX_MEMORY, TRAIN_CASES,
    TRAIN_BUDGET, observed_record, parse_action,
    parse_claim, play, prompt_for, store_episode, observed_sources,
    strong_episodic_context, proposed_context, chronological_control,
    validate_evidence, run,
)
from src.infinito3.persistent_memory import SQLiteCognitiveMemoryStore, HashEmbeddingProvider


class FakeOnly:
    real_model = False
    def __init__(self):
        self.calls = 0

    def metadata(self):
        return {"real_model": False, "model": "unit-test-only",
                "prompt_tokens": self.calls * 20,
                "completion_tokens": self.calls * 5,
                "inference_seconds": self.calls * .01}

    def chat(self, system, prompt, max_tokens=96):
        self.calls += 1
        if max_tokens == 192:
            s = json.loads(prompt)
            patches = [e for e in s["events"]
                       if e["action"] in ("apply_delta", "apply_sigma")
                       and "Patch accepted:" in e["feedback"]]
            accepted = patches[-1]
            c = dict(
                source_id=s["source_id"], profile=s["profile"], revision=s["revision"],
                operation=accepted["action"].removeprefix("apply_"),
                observed_outcome="accepted", verified_after_operation=True)
            # Deliberately hallucinate W0; must reject (test only).
            if c["source_id"] == "W0":
                c["operation"] = "sigma" if c["operation"] == "delta" else "delta"
            return json.dumps(c)
        seen = prompt.split("OBSERVED TOOLS IN THIS EPISODE:\n", 1)[-1].split(
            "\nChoose the next tool, JSON only.", 1)[0]
        try:
            rows = json.loads(seen)
        except (ValueError, TypeError):
            rows = []
        if any("Patch accepted:" in e["tool_feedback"] for e in rows):
            return '{"action":"verify"}'
        if any("Patch rejected:" in e["tool_feedback"] for e in rows):
            return '{"action":"apply_sigma"}'
        return '{"action":"apply_delta"}'


class F12GContracts(unittest.TestCase):
    def test_new_sealed_split_and_model(self):
        self.assertEqual(FIXED_MODEL_REVISION,
                         "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031")
        self.assertEqual(FIXED_MODEL_SHA256,
                         "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423")
        self.assertEqual([c.id for c in TRAIN_CASES], [f"W{i}" for i in range(6)])
        self.assertEqual([c.id for c in HELDOUT_CASES], [f"U{i}" for i in range(6)])
        self.assertEqual({c.profile for c in TRAIN_CASES},
                         {c.profile for c in HELDOUT_CASES})
        self.assertEqual(sum(c.revision == "R1" for c in HELDOUT_CASES), 3)
        self.assertEqual(sum(c.revision == "R2" for c in HELDOUT_CASES), 3)
        self.assertEqual(EVAL_BUDGETS, (2, 3))
        self.assertEqual(ARMS, ("A0", "A1", "A2", "A3", "A4"))
        self.assertEqual(TRAIN_BUDGET, 5)
        self.assertEqual(MAX_MEMORY, 1000)

    def test_hidden_truth_never_in_prompt_or_memory(self):
        f = FakeOnly()
        for case in TRAIN_CASES + HELDOUT_CASES:
            prompt = prompt_for(case, [], "")
            self.assertNotIn("_compatible", prompt)
            self.assertNotIn("Case(", prompt)
            self.assertIn("public revision", prompt.lower())
            self.assertIn("REFERENCE (untrusted", prompt)
            r = play(case, f, "", 5)
            safe = observed_record(case, r)
            self.assertEqual(set(safe),
                             {"source_id", "profile", "revision", "service",
                              "goal", "events", "verified_success"})
            self.assertNotIn("_compatible", json.dumps(safe))
            self.assertEqual(r["success"], True)

    def test_same_budget_and_strict_json(self):
        f = FakeOnly()
        for raw in ("apply_delta", '{"tool":"apply_delta"}', "not json", ""):
            self.assertEqual(parse_action(raw), "INVALID_LLM_OUTPUT")
        self.assertEqual(parse_action('{"action":"verify"}'), "verify")
        for invalid in ('{}', '{"operation":"delta"}', 'invalid-json-object'):
            self.assertNotEqual(parse_claim(invalid)[1], "OK")
        with self.assertRaises(ValueError):
            play(TRAIN_CASES[0], f, "", 8)
        with self.assertRaises(ValueError):
            play(TRAIN_CASES[0], f, "x" * (MAX_MEMORY + 1), 2)

    def test_grounding_rejects_hallucinated_operation_without_oracle(self):
        store = SQLiteCognitiveMemoryStore(path=":memory:",
                                             embedding_provider=HashEmbeddingProvider())
        try:
            f = FakeOnly()
            for case in TRAIN_CASES:
                store_episode(store, observed_record(case, play(case, f, "", 5)))
            sources = observed_sources(store)
            correct = {"source_id":"W0","profile":"coral","revision":"R1",
                       "operation":"delta","observed_outcome":"accepted",
                       "verified_after_operation":True}
            false = {**correct, "operation":"sigma"}
            self.assertTrue(validate_evidence(correct, sources)["promotable"])
            self.assertFalse(validate_evidence(false, sources)["promotable"])
            self.assertFalse(validate_evidence({**correct,"source_id":"NEVER"}, sources)["promotable"])
            self.assertEqual(validate_evidence(correct, sources, "R2")["reason"],
                             "STALE_FOR_TARGET")
            for case in HELDOUT_CASES:
                context, ids = strong_episodic_context(store, case)
                self.assertTrue(context)
                self.assertEqual(len(ids), 2)
                self.assertIn("R1", context)
                self.assertIn("R2", context)
                self.assertLessEqual(len(context), MAX_MEMORY)
                c4, ids4 = chronological_control(sources, case)
                self.assertEqual(len(ids4), 1)
                self.assertIn("CHRONOLOGY_OBSERVED", c4)
                self.assertIn(case.revision, c4)
        finally:
            store.close()

    def test_unverified_and_verified_same_proposal_set_differ_only_filter(self):
        c = dict(source_id="W0", profile="coral", revision="R1",
                 operation="sigma", observed_outcome="accepted",
                 verified_after_operation=True)
        entries = [
            {"claim":c,"validation":{"promotable":False}},
            {"claim":{**c,"source_id":"W3","revision":"R2","operation":"sigma"},
             "validation":{"promotable":True}},
        ]
        t = HELDOUT_CASES[0]
        a2, ids2 = proposed_context(entries, t, verified_only=False)
        a3, ids3 = proposed_context(entries, t, verified_only=True)
        self.assertEqual(len(ids2), 2)
        self.assertEqual(len(ids3), 1)
        self.assertIn("UNVERIFIED_LLM", a2)
        self.assertIn("EVIDENCE_VERIFIED", a3)
        self.assertNotIn("W0", a3)

    def test_fake_backend_test_only_complete_pairs_and_model_frozen(self):
        with TemporaryDirectory() as d:
            fake = FakeOnly()
            r = run(fake, Path(d))
            self.assertEqual(r["status"], "FAKE_BACKEND_TEST_ONLY")
            self.assertEqual(len(r["train"]), 6)
            self.assertEqual(len(r["evaluation"]), 60)
            self.assertEqual(r["episode_records"], 6)
            self.assertEqual(r["strategy_records"], 5)
            self.assertEqual(r["promotable_count"], 5)
            self.assertEqual(r["rejected_count"], 1)
            self.assertTrue(r["memory_unchanged"])
            self.assertFalse(r["gates"]["G0_real_pinned_60_paired_no_leak"])
            self.assertFalse(r["scientific_pass"])
            self.assertFalse(r["dev_opened"])
            self.assertFalse(r["confirm_opened"])
            for budget in ("2","3"):
                for arm in ARMS:
                    self.assertEqual(r["scores"][budget][arm]["episodes"], 6)
            self.assertTrue(all(len(x["context_text"]) <= MAX_MEMORY
                                for x in r["evaluation"]))
            self.assertTrue(Path(d, "full_report.json").exists())
            self.assertTrue(Path(d, "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
