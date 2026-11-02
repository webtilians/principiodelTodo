"""F9-Y split, information boundary, LOCO and loss contracts."""
import unittest
import numpy as np
import torch

from observatory.f9y_conditional_value_transfer import (
    CELLS, seed_for, cell_index, feature, pair_features,
    train_head, prob, summarize, shuffled_feature, bce, auc,
    TRAIN_BASE, DEV_BASE, TRAIN_N, DEV_N,
)


def synthetic_case(seed, cue, regime, effect):
    return {
        "seed":seed,
        "cue":cue,"regime":regime,"effect":effect,
        "public":[0.01]*148,
        "hc":[float(seed%100)]*64+[float((seed+1)%100)]*64,
        "read_branch":{"read":1,"first_arm_correct":int(effect==1)},
        "skip_branch":{"read":0,"first_arm_correct":int(effect!=1)},
    }


class F9YTests(unittest.TestCase):
    def test_new_splits_disjoint_and_never_open_confirm(self):
        train={seed_for("train",i,j) for i in range(4) for j in range(TRAIN_N)}
        dev={seed_for("dev",i,j) for i in range(4) for j in range(DEV_N)}
        self.assertEqual(len(train),512)
        self.assertEqual(len(dev),128)
        self.assertFalse(train & dev)
        self.assertEqual(min(train),10_100_000)
        self.assertEqual(min(dev),10_200_000)
        self.assertTrue(all(10_100_000<=v<10_140_000 for v in train))
        self.assertTrue(all(10_200_000<=v<10_240_000 for v in dev))
        for args in (("confirm",0,0),("train",4,0),
                     ("dev",0,32),("train",0,128),("dev",0,-1)):
            with self.assertRaises(ValueError):seed_for(*args)

    def test_only_hc_separates_controlled_inputs(self):
        c=synthetic_case(10100000,"key","history_f4_normal",1)
        a=feature(c,"controlled_hc")
        b=feature(c,"controlled_nuisance")
        self.assertEqual(a.shape,(280,))
        self.assertTrue(np.array_equal(a[:148],b[:148]))
        self.assertTrue(np.array_equal(a[276:],b[276:]))
        self.assertTrue(np.all(b[148:276]==0))
        self.assertEqual(list(a[276:]),[1.,0.,1.,0.])
        d=feature(c,"deploy_hc")
        p=feature(c,"deploy_public")
        self.assertEqual(d.shape,(276,))
        self.assertTrue(np.array_equal(d[:148],p[:148]))
        self.assertTrue(np.all(p[148:]==0.))

    def test_in_stratum_shuffle_keeps_public_and_tags(self):
        cases=[]
        for j in range(DEV_N):
            for k,(cue,history) in enumerate(CELLS):
                cases.append(synthetic_case(10200000+j,cue,history,(j+k)%3-1))
        i=0
        for group in ("H","C","HC"):
            x=shuffled_feature(cases,i,"loco_hc",group)
            original=feature(cases[i],"loco_hc")
            self.assertTrue(np.array_equal(x[:148],original[:148]))
            self.assertTrue(np.array_equal(x[276:],original[276:]))
            self.assertFalse(np.array_equal(x[148:276],original[148:276]))
        with self.assertRaises(ValueError):
            shuffled_feature(cases,0,"loco_hc","D")

    def test_balanced_training_is_not_implicit(self):
        data=[
            synthetic_case(10100000+i, CELLS[i%4][0],
                           CELLS[i%4][1],1 if i in (3,9) else 0)
            for i in range(20)]
        torch.set_num_threads(1)
        a,meta=train_head(data,"controlled_hc",0,0)
        b,_=train_head(data,"controlled_nuisance",0,0)
        self.assertEqual(meta["pos_weight"],1.)
        self.assertEqual(meta["positive"],2)
        self.assertEqual(meta["optimizer_steps"],200)
        self.assertEqual(meta["train_states"],20)
        for model in (a,b):
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertGreaterEqual(prob(a,feature(data[0],"controlled_hc")),0.)
        self.assertLessEqual(prob(b,feature(data[0],"controlled_nuisance")),1.)

    def test_metrics_and_no_cherry_picked_positive_cases(self):
        cases=[
            synthetic_case(1,"key","history_f4_normal",1),
            synthetic_case(2,"key","history_f4_normal",-1),
            synthetic_case(3,"key","history_f4_normal",0)]
        x=summarize(cases,[.9,.1,.1])
        self.assertEqual(x["states"],3)
        self.assertEqual(x["positive"],1)
        self.assertEqual(x["reads"],1)
        self.assertEqual(x["first_correct"],3)
        self.assertEqual(x["read_positive"],1)
        self.assertEqual(x["read_harmful"],0)
        self.assertAlmostEqual(x["auc"],1.)
        self.assertIsNone(auc([0,0],[.2,.4]))
        self.assertGreater(bce(1,.1),bce(1,.9))


if __name__=="__main__":
    unittest.main()
