"""F10-H prospective preflight on TRAIN worlds only (never open DEV/OOD)."""
import unittest

import numpy as np
import torch

from observatory.f10h_worlds import (
    seed_for,world,KEYS,FEATURES,EVENTS,VARIANTS,SOURCE_MODES,
)
from observatory.f10g_models import Actor,count,gru_width
from observatory.f10h_experiment import tensors,model_for,ARMS

class F10HFactorialIntegrity(unittest.TestCase):
    def test_ranges_and_reserved_confirmation(self):
        self.assertEqual(seed_for("train",0,0),19100000)
        self.assertEqual(seed_for("dev",0,0),19200000)
        self.assertEqual(seed_for("ood",0,0),19400000)
        for args in (("confirm",0,0),("train",4,0),("dev",0,128),
                     ("ood",-1,0),("train",0,512)):
            with self.assertRaises(ValueError):seed_for(*args)

    def test_four_public_counterfactuals_only_sign_and_time_differ(self):
        for j in range(24):
            cases=world("train",0,j)
            self.assertEqual({r["variant"] for r in cases},set(VARIANTS))
            self.assertEqual(len({r["hash"] for r in cases}),4)
            self.assertEqual(len({r["k"] for r in cases}),1)
            self.assertEqual(len({r["query"] for r in cases}),1)
            self.assertEqual(len({r["truth"] for r in cases}),2)
            self.assertEqual({r["source_mode"] for r in cases},{"iid"})
            self.assertTrue(all(r["signed_count"]>=4 and
                                r["signed_value_changes"]>=3 for r in cases))
            by={r["variant"]:r for r in cases}
            for x in cases:
                self.assertEqual(x["memory"].shape,(EVENTS,FEATURES))
                self.assertEqual(x["history"].shape,(EVENTS+1,FEATURES+1))
                np.testing.assert_array_equal(x["memory"],x["history"][:-1,:FEATURES])
                q=x["query"]
                signed=[v for v in x["memory"] if v[q] and v[KEYS+1]]
                self.assertGreaterEqual(len(signed),4)
                self.assertEqual(x["truth"],
                     int(max(signed,key=lambda v:v[KEYS+2])[KEYS]>0)^x["inversion"])
            for a,b in (("U_before","S_before"),("U_after","S_after")):
                diff=by[a]["memory"]!=by[b]["memory"]
                self.assertEqual(int(np.count_nonzero(diff)),1)
                self.assertEqual(int(np.count_nonzero(diff[:,KEYS+1])),1)
            for a,b in (("U_before","U_after"),("S_before","S_after")):
                diff=by[a]["memory"]!=by[b]["memory"]
                self.assertEqual(int(np.count_nonzero(diff)),2)
                self.assertEqual(int(np.count_nonzero(diff[:,KEYS+2])),2)
                np.testing.assert_array_equal(by[a]["history"][-1],by[b]["history"][-1])
            for c in cases[1:]:
                np.testing.assert_array_equal(c["memory"][:,:KEYS+1],
                                              cases[0]["memory"][:,:KEYS+1])
                np.testing.assert_array_equal(c["memory"][:,KEYS+3:],
                                              cases[0]["memory"][:,KEYS+3:])

    def test_train_factorial_model_capacity_and_no_value_leak(self):
        torch.set_num_threads(1)
        cases=world("train",0,0)
        x,m,y,w=tensors(cases)
        self.assertEqual(x.shape,(4,49,30))
        self.assertEqual(m.shape,(4,48,29))
        self.assertEqual(set(ARMS),
                         {"joint","selection_only","interpretation_only","both","gru_only"})
        params=count(model_for("joint"))
        self.assertTrue(all(count(model_for(a))==params for a in ARMS[:-1]))
        self.assertLessEqual(abs(count(model_for("gru_only"))-params)/params,.05)
        torch.manual_seed(195000)
        model=Actor("unary").eval()
        with torch.no_grad():
            read,scores,weights,h=model(x,m)
            altered=m.clone();altered[:,:,KEYS]*=-1
            read2,scores2,weights2,h2=model(x,altered)
            self.assertTrue(torch.equal(scores,scores2))
            self.assertTrue(torch.equal(weights,weights2))
            self.assertTrue(torch.equal(h,h2))
            self.assertTrue(torch.equal(model(x,m,"skip")[0],
                                        model(x,altered,"skip")[0]))
            val=m[torch.arange(4),w,KEYS]
            isolated=model.interpret(h,val,torch.ones_like(val),x[:,-1,-1])
            self.assertTrue(torch.equal(isolated,model(x,m,patch_value=val)[0]))

if __name__=="__main__":unittest.main()
