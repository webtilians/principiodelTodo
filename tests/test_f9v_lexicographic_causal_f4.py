"""F9-V objective, paired causal labels, and information boundary contracts."""
import unittest
from unittest.mock import patch
import numpy as np
import torch

from observatory.f9v_lexicographic_causal_f4 import (
    seed_for,lex_choice,scalar_choice,head_inputs,fit_heads,
    summarize,eval_case,EVALUATE_MODES,
)

class F9VCausalLexTests(unittest.TestCase):
    def test_lexicography_pays_for_read_only_if_it_changes_correctness(self):
        self.assertEqual(lex_choice(1,0),1)
        self.assertEqual(lex_choice(1,1),0)
        self.assertEqual(lex_choice(0,0),0)
        self.assertEqual(lex_choice(0,1),0)
        self.assertEqual(scalar_choice(1,0),0)
        for a,b in ((-1,0),(0,2)):
            with self.assertRaises(ValueError): lex_choice(a,b)

    def test_new_partitions_reserve_sealed_holdouts(self):
        a={seed_for("train",i,j) for i in range(4) for j in range(32)}
        b={seed_for("dev",i,j) for i in range(4) for j in range(16)}
        self.assertEqual(len(a),128)
        self.assertEqual(len(b),64)
        self.assertFalse(a&b)
        self.assertTrue(all(8100000<=x<8200000 for x in a))
        self.assertTrue(all(8200000<=x<8300000 for x in b))
        for args in (("confirm",0,0),("dev",4,0),("train",0,32),
                     ("dev",0,16),("train",0,-1)):
            with self.assertRaises(ValueError): seed_for(*args)

    @staticmethod
    def case(raw_value, h_value, effect,regime="history_f4_normal"):
        r=1 if effect==1 else 0
        s=1 if effect==-1 else 0
        if effect==0:
            r=s=1
        return {
            "public":[raw_value]*148,
            "hc":[h_value]*128,
            "label":lex_choice(r,s),
            "scalar_label":scalar_choice(r,s),
            "regime":regime,
            "read_branch":{"first_arm_correct":r,"first_arm":"upper",
                           "committed":1,"steps_until_first_arm_or_exit":2},
            "skip_branch":{"first_arm_correct":s,"first_arm":"lower",
                           "committed":1,"steps_until_first_arm_or_exit":2},
        }

    def test_only_public_and_prior_hc_go_to_head(self):
        case=self.case(0.5,0.2,1)
        h=head_inputs([case],True)
        p=head_inputs([case],False)
        self.assertEqual(tuple(h.shape),(1,276))
        self.assertTrue(torch.all(h[:,148:]==.2).item())
        self.assertTrue(torch.all(p[:,148:]==0).item())
        self.assertTrue(torch.equal(h[:,:148],p[:,:148]))
        bad=dict(case,public=[0.]*149)
        with self.assertRaises(AssertionError):head_inputs([bad],True)

    def test_predictor_and_oracle_are_separate_controls(self):
        # An oracle may access labels in the evaluator. A deployed head
        # is only supplied 276 observational features via head_inputs.
        c=self.case(0.1,0.5,1)
        models,rep=fit_heads([c,self.case(0.2,-0.5,-1)],0)
        self.assertEqual(set(models),{"lex_hc","lex_public","scalar_hc"})
        self.assertEqual(rep["lex_hc"]["samples"],2)
        result=eval_case(c,models)
        self.assertEqual(set(result),set(EVALUATE_MODES))
        self.assertEqual(result["oracle"]["read"],1)
        self.assertEqual(result["always"]["read"],1)
        self.assertEqual(result["never"]["read"],0)
        self.assertIn(result["lex_hc"]["read"],(0,1))

    def test_summary_uses_full_paired_denominator_not_only_committed(self):
        c=[self.case(.1,.2,1),self.case(.1,.3,-1)]
        chosen=[]
        for case in c:
            x={}
            for mode in EVALUATE_MODES:
                x[mode]={"first_arm_correct":1 if mode=="always" else 0,
                         "first_arm":"upper","read":1 if mode=="always" else 0,
                         "label_match":1,"committed":1}
            chosen.append(x)
        # second regime also requires complete pairs; here test one only
        c[1]["regime"]="history_f4_normal"
        # summary iterates two regimes; the empty one is fine.
        summary=summarize(c,chosen)
        self.assertEqual(summary["always"]["first_arm_correct"],2)
        self.assertEqual(summary["never"]["first_arm_correct"],0)
        self.assertEqual(summary["always"]["regimes"]["history_f4_normal"]["pairs_both_first_correct"],1)
        self.assertEqual(summary["always"]["regimes"]["history_f4_postcue_off"]["cases"],0)

if __name__=="__main__":
    unittest.main()
