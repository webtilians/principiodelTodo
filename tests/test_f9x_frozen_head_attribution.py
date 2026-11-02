"""F9-X validation contracts. No old confirmation or new evaluation is opened here."""
import unittest
import numpy as np
import torch

from observatory.f9x_frozen_head_attribution import (
    seed_for, patched_vector, vector, choice_metrics, score, sensitivity, BASE, WORLDS
)
from observatory.f9t_learned_f4_read_gate import ReadHead


def case(h=1., c=2., raw=3., effect=1):
    a = {
        "public": [raw]*148, "hc": [h]*64 + [c]*64,
        "effect": effect,
        "read_branch": {"first_arm_correct": int(effect == 1)},
        "skip_branch": {"first_arm_correct": int(effect != 1)},
    }
    return a


class F9XTests(unittest.TestCase):
    def test_reserved_worlds_only(self):
        seeds = {seed_for(i,j) for i in range(4) for j in range(WORLDS)}
        self.assertEqual(len(seeds), 128)
        self.assertTrue(all(9500000 <= v < 9540000 for v in seeds))
        for idx,j in ((-1,0),(4,0),(0,-1),(0,32)):
            with self.assertRaises(ValueError):seed_for(idx,j)
        self.assertFalse(any(8300000 <= v < 8400000 or
                             9300000 <= v < 9400000 or
                             9600000 <= v < 9700000 for v in seeds))

    def test_reciprocal_group_patches_do_not_change_public(self):
        a,b = case(1.,2.),case(7.,9.)
        ph = patched_vector(a,b,"cue_H")
        pc = patched_vector(a,b,"cue_C")
        both = patched_vector(a,b,"cue_HC")
        for x in (ph,pc,both):
            self.assertTrue(np.array_equal(x[:148],vector(a)[:148]))
        self.assertTrue(np.all(ph[148:212]==7))
        self.assertTrue(np.all(ph[212:]==2))
        self.assertTrue(np.all(pc[148:212]==1))
        self.assertTrue(np.all(pc[212:]==9))
        self.assertTrue(np.array_equal(both,vector(b)))
        with self.assertRaises(ValueError):
            patched_vector(a,b,"use_privileged_cue_label")

    def test_frozen_head_scoring_and_ig_no_weight_updates(self):
        torch.manual_seed(17)
        h = ReadHead()
        before = [p.detach().clone() for p in h.parameters()]
        a = vector(case())
        base = np.zeros(276,dtype=np.float32)
        prediction = score(h,a)
        self.assertIn(prediction["read"],(0,1))
        attribution = sensitivity(h,a,base,ig_steps=4)
        self.assertEqual(set(attribution["ig_abs"]),{"public","H","C"})
        self.assertTrue(all(np.isfinite(v) for v in attribution["ig_abs"].values()))
        for p, old in zip(h.parameters(),before):
            self.assertTrue(torch.equal(p,old))

    def test_cases_not_selected_by_oracle(self):
        samples = [case(effect=1),case(effect=-1),case(effect=0)]
        value = choice_metrics(samples,[1,0,0])
        self.assertEqual(value["cases"],3)
        self.assertEqual(value["reads"],1)
        self.assertEqual(value["useful_read"],1)
        self.assertEqual(value["first_correct"],3)


if __name__=="__main__":
    unittest.main()
