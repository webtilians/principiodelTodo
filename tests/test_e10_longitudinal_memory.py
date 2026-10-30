"""E10 preconfirmation contracts: only synthetic DEV worlds, no confirmation reads."""
import tempfile
import unittest
from pathlib import Path

from observatory.e10_longitudinal_memory import (
    ARMS,BUDGET,CONFIRM_BASE,DEV_BASE,DecisionReader,
    Arm,create_world,seed_for,history,question,expected_action,
    evaluate_world,evaluate_negative_control,paired_bootstrap_by_index,
)
from src.infinito3.types import LLMMessage,LLMRequest


class E10Contracts(unittest.TestCase):
    def test_dev_and_confirmation_seeds_are_disjoint(self):
        d={seed_for("dev",i,j) for i in range(3) for j in range(12)}
        c={seed_for("confirm",i,j) for i in range(8) for j in range(16)}
        self.assertEqual(len(d),36)
        self.assertEqual(len(c),128)
        self.assertEqual(min(d),DEV_BASE)
        self.assertEqual(max(d),DEV_BASE+2011)
        self.assertEqual(min(c),CONFIRM_BASE)
        self.assertEqual(max(c),CONFIRM_BASE+7015)
        self.assertFalse(d&c)
        self.assertTrue(all(s>=1000000 for s in d|c))
        for args in (("dev",3,0),("dev",0,12),
                     ("confirm",8,0),("confirm",0,16)):
            with self.assertRaises(ValueError):
                seed_for(*args)

    def test_pair_only_historical_cue_changed_not_prompt_or_noise(self):
        for j in range(12):
            world=create_world(seed_for("dev",0,j))
            a1,a2=history(world,False)
            b1,b2=history(world,True)
            self.assertEqual(a1[1:],b1[1:])
            self.assertEqual(a2[1:],b2[1:])
            self.assertNotEqual(a1[0],b1[0])
            self.assertNotEqual(a2[0],b2[0])
            self.assertNotEqual(expected_action(world,world.options[0]),
                                expected_action(world,world.options[1]))
            self.assertEqual(question(world),question(world))

    def test_decision_reader_cannot_access_hidden_or_guess(self):
        w=create_world(seed_for("dev",0,0))
        question_text=question(w)
        adapter=DecisionReader()
        request=LLMRequest(messages=[LLMMessage("developer","You are an agent."),
                                      LLMMessage("user",question_text)])
        self.assertEqual(adapter(request),"ABSTENER")
        for j in (0,1):
            sentence=history(w,bool(j))[0][0]
            ask=LLMRequest(messages=[LLMMessage("developer","System"),
                       LLMMessage("user","INFINITO REFERENCE CONTEXT\n"+sentence),
                       LLMMessage("user",question_text)])
            self.assertEqual(adapter(ask),
                             expected_action(w,w.options[1 if j else 0]))
        bad=LLMRequest(messages=[LLMMessage("developer","System"),
                        LLMMessage("user","INFINITO REFERENCE CONTEXT\nMe gusta el jazz."),
                        LLMMessage("user",question_text)])
        self.assertEqual(adapter(bad),"ABSTENER")

    def test_four_real_pipelines_and_longitudinal_update_dev_only(self):
        w=create_world(seed_for("dev",0,0))
        result=evaluate_world(w)
        self.assertEqual(set(result["arms"]),set(ARMS))
        self.assertEqual(set(result["variants"]),{"original","reversed"})
        for arm in ARMS:
            self.assertIn(result["arms"][arm]["all_four_correct"],(0,1))
            self.assertLessEqual(result["arms"][arm]["individual_correct"],4)
            for variant in result["variants"].values():
                for stage in ("before","after"):
                    self.assertLessEqual(variant[stage][arm]["context_tokens"],BUDGET)
                    self.assertTrue(variant[stage][arm]["inferred_with_common_reader"])
            self.assertEqual(result["arms"]["no_memory"]["all_four_correct"],0)
        # Reopened SQL store is actual file-backed persistence; no hidden
        # episodic teacher labels can reach DecisionReader.
        with tempfile.TemporaryDirectory() as d:
            arm=Arm("infinito3",Path(d))
            try:
                sentence=history(w,False)[0][0]
                arm.observe(sentence)
                arm.restart()
                self.assertTrue(arm.engine.memory_store.all())
                self.assertEqual(arm.decide(question(w))["action"],
                                 expected_action(w,w.options[0]))
            finally:
                arm.close()

    def test_negative_controls_never_hallucinate_missing_fact(self):
        negative=evaluate_negative_control()
        self.assertEqual(set(negative),set(ARMS))
        self.assertTrue(all(row["action"]=="ABSTENER" and row["abstained"]
                            for row in negative.values()))

    def test_criterion_bootstrap_fixed_from_indices(self):
        summary={"indices":list(range(8)),
                 "by_index":{str(i):{"infinito3":1.0,"fifo_8":0.5,"rag_hash":0.75}
                             for i in range(8)}}
        result=paired_bootstrap_by_index(summary)
        self.assertTrue(result["E10_primary_passes_both_baselines"])
        summary["by_index"]["0"]["infinito3"]=0.
        negative=paired_bootstrap_by_index(summary)
        self.assertFalse(negative["E10_primary_passes_both_baselines"])


if __name__=="__main__":
    unittest.main()
