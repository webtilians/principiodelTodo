"""F9-W preregistered new-world scarcity experiment contracts."""
import unittest
import torch
from observatory.f9w_positive_evidence_scarcity import (
    seed_for,ARMS,fit_arm,arm_data,evaluate,SHORT_WORLDS,
)

class F9WContracts(unittest.TestCase):
    def test_new_splits_and_reserved_confirmation(self):
        tr={seed_for("train",i,j) for i in range(4) for j in range(128)}
        de={seed_for("dev",i,j) for i in range(4) for j in range(16)}
        self.assertEqual(len(tr),512)
        self.assertEqual(len(de),64)
        self.assertFalse(tr&de)
        self.assertTrue(all(9100000<=x<9200000 for x in tr))
        self.assertTrue(all(9200000<=x<9300000 for x in de))
        for args in (("confirm",0,0),("train",0,128),
                     ("dev",0,16),("train",0,-1),("train",4,0)):
            with self.assertRaises(ValueError):seed_for(*args)

    @staticmethod
    def synthetic(label=0,h=.1):
        return {"public":[0.0]*148,"hc":[h]*128,"label":label,
                "effect":1 if label else 0,
                "read_branch":{"first_arm_correct":1,"first_arm":"upper"},
                "skip_branch":{"first_arm_correct":0 if label else 1,
                               "first_arm":"lower"},
                "regime":"history_f4_normal"}

    def test_short_subset_is_exact_prefix_of_long(self):
        all_cases=[self.synthetic(j%5==0) for j in range(128*4)]
        self.assertEqual(len(arm_data(all_cases,"short32_hc")),128)
        self.assertEqual(len(arm_data(all_cases,"short32_weighted_hc")),128)
        self.assertEqual(len(arm_data(all_cases,"long128_hc")),512)
        self.assertIs(arm_data(all_cases,"short32_hc")[0],all_cases[0])

    def test_equal_fixed_optimizer_steps_and_weight_ratio(self):
        # Unit test uses 4 cases: short arm can still train on available rows.
        data=[self.synthetic(x) for x in (1,0,0,0)]
        a,report=fit_arm(data,0,"short32_weighted_hc")
        b,plain=fit_arm(data,0,"short32_hc")
        self.assertEqual(report["pos_weight"],3)
        self.assertEqual(plain["pos_weight"],1)
        self.assertEqual(report["optimizer_steps"],200)
        self.assertEqual(plain["optimizer_steps"],200)
        self.assertTrue(all(not x.requires_grad for x in a.parameters()))

    def test_no_positive_does_not_invent_label(self):
        data=[self.synthetic(0)]*4
        _,r=fit_arm(data,1,"short32_weighted_hc")
        self.assertEqual(r["n_positive"],0)
        self.assertEqual(r["pos_weight"],1)
        self.assertTrue(r["no_positive_train"])

if __name__=="__main__":
    unittest.main()
