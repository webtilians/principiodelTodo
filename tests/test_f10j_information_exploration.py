"""F10-J pre-run tests: TRAIN worlds only, no DEV/OOD/CONFIRM opened."""
import unittest
import numpy as np
import torch

from observatory.f10j_worlds import world,seed_for,KEYS,EVENTS,FEATURES,VARIANTS
from observatory.f10j_experiment import (
    data_tensors,rollout,reward,expert_loss,ARMS,SLOT_EPS,REWARD_NOISE
)
from observatory.f10i_actor import ExperienceActor,count_parameters

class F10JPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_new_registered_split_namespaces(self):
        self.assertEqual(seed_for("train",0,0),21_100_000)
        self.assertEqual(seed_for("dev",0,0),21_200_000)
        self.assertEqual(seed_for("ood",0,0),21_400_000)
        for t in (("confirm",0,0),("train",4,0),("train",0,512),
                  ("dev",0,128),("ood",-1,0)):
            with self.assertRaises(ValueError):seed_for(*t)

    def test_train_worlds_counterfactual_and_no_value_leak(self):
        # Deliberately TRAIN ONLY. OOD reserved until models frozen.
        seen=set()
        for j in range(64):
            w=world("train",0,j)
            by={r["variant"]:r for r in w}
            self.assertEqual(set(by),set(VARIANTS))
            self.assertEqual(len({r["hash"] for r in w}),4)
            mode=bool(j%2==0)
            self.assertEqual({r["informative"] for r in w},{mode})
            self.assertEqual({r["k"] for r in w},{(6,9)[(j//2)%2]})
            seen.add((mode,w[0]["k"]))
            self.assertEqual({r["source_mode"] for r in w},{"iid"})
            if mode:
                self.assertNotEqual(by["S_late"]["truth"],by["U_late"]["truth"])
                self.assertEqual(by["U_late"]["truth"],by["U_early"]["truth"])
                self.assertEqual(by["U_late"]["truth"],by["S_early"]["truth"])
            else:
                self.assertEqual(len({r["truth"] for r in w}),1)
            for r in w:
                self.assertEqual(r["history"].shape,(49,30))
                self.assertEqual(r["memory"].shape,(48,29))
                np.testing.assert_array_equal(
                    r["history"][:EVENTS,:KEYS],r["memory"][:EVENTS,:KEYS])
                self.assertEqual(int(np.count_nonzero(r["history"][:EVENTS,KEYS])),0)
                self.assertEqual(float(r["history"][-1,KEYS+1]),float(mode))
                self.assertEqual(int(r["history"][-1,KEYS+2]>0),r["public_bit"])
                valid=(r["memory"][:,r["q"]]>.5)&(r["memory"][:,KEYS+1]>.5)
                sel=np.flatnonzero(valid)
                chosen=sel[np.argmax(r["memory"][sel,KEYS+2])]
                self.assertEqual(chosen,r["winner"])
                self.assertEqual(r["signed_changes"]>=2,True)
                self.assertTrue(any(x[r["q"]]>.5 and x[KEYS+1]==0 and
                    x[KEYS+2]>r["memory"][chosen,KEYS+2] for x in r["memory"]))
                expected=(int(r["memory"][chosen,KEYS]>0) if mode else r["public_bit"])^r["xor"]
                self.assertEqual(expected,r["truth"])
            for v in VARIANTS[1:]:
                np.testing.assert_array_equal(
                    by["U_late"]["memory"][:,:KEYS+1],
                    by[v]["memory"][:,:KEYS+1])
                np.testing.assert_array_equal(
                    by["U_late"]["memory"][:,KEYS+3:],
                    by[v]["memory"][:,KEYS+3:])
        self.assertEqual(len(seen),4)

    def test_no_supervised_labels_enter_rl_policy_or_credit(self):
        cases=world("train",0,0)+world("train",0,1)
        x,m,y,w,info,delay=data_tensors(cases)
        torch.manual_seed(215000)
        actor=ExperienceActor()
        self.assertGreater(count_parameters(actor),0)
        self.assertEqual(len(ARMS),5)
        self.assertEqual(SLOT_EPS,.5)
        self.assertEqual(REWARD_NOISE,.10)
        with torch.no_grad():
            skip=actor.predict(x,m,"force_skip")["action"]
            altered=m.clone()
            altered[:,:,KEYS]*=-1
            # Recurrent context never receives F4 values.
            self.assertTrue(torch.equal(skip,actor.predict(x,altered,"force_skip")["action"]))
            self.assertTrue(torch.equal(actor.encode(x,m)[3],actor.encode(x,altered)[3]))
            self.assertTrue(torch.equal(actor.encode(x,m)[4],actor.encode(x,altered)[4]))
        a=rollout(actor,x,m,gate=1,slot_mix=True)
        b=rollout(actor,x,m,gate=0,slot_mix=True)
        self.assertTrue(torch.equal(a["gate"],torch.ones(len(x))))
        self.assertTrue(torch.equal(b["gate"],torch.zeros(len(x))))
        self.assertEqual(a["action"].shape,(len(x),))
        self.assertEqual(a["gate_probability"].shape,(len(x),))
        noise=torch.zeros(len(x),dtype=torch.bool)
        ra=reward(a,y,delay,noise)
        rb=reward(b,y,delay,noise)
        self.assertEqual(ra.shape,rb.shape)
        self.assertTrue(torch.isfinite(ra).all())
        self.assertTrue(torch.isfinite(rb).all())
        self.assertTrue(torch.isfinite(expert_loss(actor,x,m,w,y,info)))

    def test_rl_update_has_finite_gradient_without_labels(self):
        cases=world("train",0,4)
        x,m,y,w,info,delay=data_tensors(cases)
        torch.manual_seed(215000)
        actor=ExperienceActor()
        a=rollout(actor,x,m,gate=1,slot_mix=True)
        b=rollout(actor,x,m,gate=0,slot_mix=True)
        noise=torch.zeros(len(x),dtype=torch.bool)
        ra=reward(a,y,delay,noise)
        rb=reward(b,y,delay,noise)
        delta=(ra-rb).detach()
        loss=-(delta*a["gate_probability"]).mean()-.5*(
            delta*(a["slot_log_prob"]+a["action_log_prob"]-
                   b["action_log_prob"])).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in actor.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in actor.parameters()))

if __name__=="__main__":unittest.main()
