"""F12-F contracts: public-source grounding; fake inference is tests only."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f12f_evidence_validator import (
    CASES, CLAIM_FIELDS, FIXED_MODEL_REVISION, FIXED_MODEL_SHA256,
    ToolSandbox, store_experience, get_observed_sources, parse_candidate,
    parse_action, validate_evidence, cited_only, recency_only,
    evidence_claims_from_public_trace, make_adversarial, _confusion, run_actor, run,
)
from src.infinito3.persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind


def claim(source_id="V0", profile="ember", revision="R1", operation="delta",
          observed_outcome="accepted", verified_after_operation=True):
    return dict(source_id=source_id, profile=profile, revision=revision,
                operation=operation, observed_outcome=observed_outcome,
                verified_after_operation=verified_after_operation)


def observed_source(source_id="V0", profile="ember", revision="R1", accepted="delta",
                    verified=True, prior_bad=True):
    events = []
    if prior_bad:
        wrong = "sigma" if accepted == "delta" else "delta"
        events.append({"step": len(events), "action": "apply_" + wrong,
                       "feedback": "Patch rejected: incompatible with this revision.",
                       "reward": -.12, "done": False})
    events.append({"step": len(events), "action": "apply_" + accepted,
                   "feedback": "Patch accepted: compatible with this revision.",
                   "reward": -.01, "done": False})
    if verified:
        events.append({"step": len(events), "action": "verify",
                       "feedback": "Verification passed: service compatible.",
                       "reward": 1., "done": True})
    return {
        "source_id": source_id, "profile": profile, "revision": revision,
        "goal": "A fully public goal", "service": "Service", "events": events,
        "verified_success": verified,
    }


class FakeModel:
    real_model = False
    def __init__(self):
        self.calls = 0

    def chat(self, system, prompt, max_tokens=96):
        self.calls += 1
        if max_tokens == 192:
            source = json.loads(prompt)
            candidates = list(evidence_claims_from_public_trace(source))
            target = next((c for c in candidates if c["observed_outcome"] == "accepted"), None)
            return json.dumps(target) if target else "{}"
        if max_tokens == 600:
            rows = json.loads(prompt)
            answers = []
            for source in rows:
                candidates = list(evidence_claims_from_public_trace(source))
                target = next((c for c in candidates if c["observed_outcome"] == "accepted"), None)
                if target:
                    answers.append(target)
            return json.dumps(answers)
        history = prompt.split("OBSERVED TOOL RESPONSES:\n", 1)[1].split(
            "\nNext tool, JSON only.")[0]
        try:
            events = json.loads(history)
        except (TypeError, ValueError):
            events = []
        if not events:
            return '{"action":"inspect"}'
        if any("Patch accepted:" in x["tool_feedback"] for x in events):
            return '{"action":"verify"}'
        if any("Patch rejected:" in x["tool_feedback"] for x in events):
            return '{"action":"apply_sigma"}'
        return '{"action":"apply_delta"}'

    def metadata(self):
        return {"real_model": False, "model": "fake-only-unit-tests"}


class F12FSourceGroundingContracts(unittest.TestCase):
    def test_new_prospective_split_and_pinned_model(self):
        self.assertEqual(FIXED_MODEL_REVISION,
                         "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031")
        self.assertEqual(FIXED_MODEL_SHA256,
                         "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423")
        self.assertEqual([c.source_id for c in CASES], [f"V{i}" for i in range(8)])
        self.assertEqual([c.revision for c in CASES], ["R1"] * 4 + ["R2"] * 4)
        self.assertEqual({c.profile for c in CASES[:4]},
                         {c.profile for c in CASES[4:]})
        self.assertFalse({c.source_id for c in CASES} &
                         {"S0", "S4", "H0", "H5", "M0", "K0", "L0", "T0"})
        self.assertEqual(set(CLAIM_FIELDS), set(claim()))

    def test_invalid_and_missing_sources_fail_closed(self):
        sources = {"V0": observed_source()}
        for invalid in (
            {**claim(), "source_id": "DOES_NOT_EXIST"},
            {**claim(), "profile": "harbor"},
            {**claim(), "revision": "R2"},
            {**claim(), "operation": "sigma"},
            {**claim(), "observed_outcome": "rejected"},
            {**claim(), "verified_after_operation": False},
            {**claim(), "action": "apply_delta"},
        ):
            self.assertFalse(validate_evidence(invalid, sources)["promotable"])
        self.assertTrue(validate_evidence(claim(), sources)["promotable"])
        self.assertTrue(cited_only({**claim(), "operation": "sigma"}, sources))
        self.assertTrue(recency_only({**claim(), "operation": "sigma"}, sources))

    def test_verify_has_to_follow_accepted_operation_not_precede(self):
        s = observed_source(verified=False, prior_bad=False)
        s["events"].insert(0, {
            "step": -1, "action": "verify",
            "feedback": "Verification passed: service compatible.",
            "reward": 1., "done": True})
        result = validate_evidence(claim(), {"V0": s})
        self.assertFalse(result["promotable"])
        self.assertTrue(validate_evidence(
            {**claim(), "verified_after_operation": False}, {"V0": s})["supported"])

    def test_accepted_only_and_negative_observations_are_not_promoted(self):
        s = observed_source(verified=False)
        sources = {"V0": s}
        accepted_unverified = {**claim(), "verified_after_operation": False}
        self.assertTrue(validate_evidence(accepted_unverified, sources)["supported"])
        self.assertFalse(validate_evidence(accepted_unverified, sources)["promotable"])
        rejected_action = {**claim(), "operation": "sigma",
                           "observed_outcome": "rejected", "verified_after_operation": False}
        self.assertTrue(validate_evidence(rejected_action, sources)["supported"])
        self.assertFalse(validate_evidence(rejected_action, sources)["promotable"])

    def test_f12e_s4_s5_regression_false_sigma_claim_rejected(self):
        # Known regression; NOT new independent confirmation.
        s4 = observed_source("S4", "cinder", "R2", "delta", True)
        s5 = observed_source("S5", "spruce", "R2", "delta", True)
        sources = {"S4": s4, "S5": s5}
        for sid, profile in (("S4", "cinder"), ("S5", "spruce")):
            false_quote = claim(sid, profile, "R2", "sigma", "accepted", True)
            correct_quote = claim(sid, profile, "R2", "delta", "accepted", True)
            self.assertTrue(cited_only(false_quote, sources))
            self.assertFalse(validate_evidence(false_quote, sources)["promotable"])
            self.assertTrue(validate_evidence(correct_quote, sources)["promotable"])

    def test_revisions_are_historically_true_but_not_current(self):
        old = observed_source("V0", "ember", "R1", "delta", True)
        new = observed_source("V4", "ember", "R2", "sigma", True)
        sources = {"V0": old, "V4": new}
        old_claim = claim()
        self.assertTrue(validate_evidence(old_claim, sources)["promotable"])
        self.assertEqual(validate_evidence(old_claim, sources, "R2")["reason"],
                         "STALE_FOR_TARGET")
        self.assertFalse(recency_only(old_claim, sources))
        new_claim = claim("V4", "ember", "R2", "sigma", "accepted", True)
        self.assertTrue(recency_only(new_claim, sources))
        self.assertTrue(validate_evidence(new_claim, sources, "R2")["promotable"])

    def test_generated_controls_are_exclusively_public_record_derived(self):
        s = observed_source()
        sources = {"V0": s}
        controls = make_adversarial(s, sources)
        self.assertEqual(len(controls), 7)
        report = _confusion(controls, "evidence")
        self.assertEqual(report["TP"], 1)
        self.assertEqual(report["FP"], 0)
        self.assertEqual(report["FN"], 0)
        self.assertGreater(_confusion(controls, "baseline_citation_promotes")["FP"], 0)

    def test_structured_extraction_strictly_requires_json(self):
        valid = json.dumps(claim())
        self.assertEqual(parse_candidate(valid)[1], "OK")
        for invalid in ('{"wrong":true}', 'not-json ' + valid, '[{}]', 'random text'):
            self.assertNotEqual(parse_candidate(invalid)[1], "OK")
        self.assertEqual(parse_candidate("{}", multiple=False)[1], "ABSTAINED")
        self.assertEqual(parse_candidate("[]", multiple=True)[1], "OK")
        self.assertEqual(parse_action("apply_delta"), "INVALID_LLM_OUTPUT")

    def test_memory_receives_only_public_observations(self):
        store = SQLiteCognitiveMemoryStore(
            path=":memory:", embedding_provider=HashEmbeddingProvider())
        try:
            fake = FakeModel()
            for case in CASES:
                actor = run_actor(case, fake)
                entry = store_experience(store, actor)
                self.assertEqual(entry.kind, MemoryKind.EPISODIC)
                self.assertNotIn("_compatible", entry.content)
            records = get_observed_sources(store)
            self.assertEqual(len(records), 8)
            for record in records.values():
                self.assertEqual(set(record),
                                 {"source_id", "profile", "revision", "service",
                                  "goal", "events", "verified_success"})
        finally:
            store.close()

    def test_end_to_end_fake_cannot_count_as_llm_evidence(self):
        with TemporaryDirectory() as folder:
            report = run(FakeModel(), Path(folder))
            self.assertEqual(report["status"], "FAKE_BACKEND_TECHNICAL_TEST_ONLY")
            self.assertFalse(report["gates"]["G0_real_model_same_weights_public_sources_only"])
            self.assertEqual(report["source_count"], 8)
            self.assertEqual(report["episode_count"], 8)
            self.assertGreaterEqual(report["generated_claims_total"], 8)
            self.assertGreaterEqual(report["control_positives"], 2)
            self.assertEqual(report["confusion"]["evidence"]["FP"], 0)
            self.assertTrue(report["gates"]["G3_comparators_all_scored"])
            self.assertFalse(report["truth_map_shared_with_verifier"])
            self.assertFalse(report["scientific_pass"])
            self.assertFalse(report["dev_opened"])
            self.assertFalse(report["confirm_opened"])
            self.assertTrue((Path(folder) / "full_audit.json").exists())
            self.assertTrue((Path(folder) / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
