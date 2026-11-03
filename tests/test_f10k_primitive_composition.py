"""F10-K prospective preflight: TRAIN worlds ONLY, never DEV/OOD/CONFIRM."""
import unittest
import numpy as np
import torch

from observatory.f10k_worlds import (
    world,seed_for,MAX_SLOTS,FEAT,VARIANTS,TYPES,
)
from observatory.f10k_experiment import (
    SkillActor,tensorize,env_reward,order_for,ARMS,TOTAL_EPISODES,
)

class F10KPrimitives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_splits_are_explicit_and_confirm_sealed(self):
        self.assertEqual(seed_for("train",0,"P",0),22100000)
        self.assertEqual(seed_for("train",0,"I",0),22102000)
        self.assertEqual(seed_for("train",0,"S",0),22104000)
        self.assertEqual(seed_for("dev",0,"P",0),22200000)
        self.assertEqual(seed_for("ood",0,"M",0),22400000)
        for case in (("confirm",0,"M",0),("train",4,"P",0),
                     ("train",0,"M",0),("dev",0,"M",0),
                     ("ood",0,"S",0),("dev",0,"P",64),
                     ("ood",0,"M",128)):
            with self.assertRaises(ValueError):seed_for(*case)

    def test_train_only_public_and_interpret_bits(self):
        for task in ("P","I"):
            for j in range(20):
                cases=world("train",0,task,j)
                self.assertEqual({r["variant"] for r in cases},set(VARIANTS))
                self.assertEqual(len({r["case_sha"] for r in cases}),4)
                self.assertEqual({r["task"] for r in cases},{task})
                self.assertEqual({r["truth"] for r in cases},{0,1})
                for row in cases:
                    self.assertEqual(row["features"].shape,(MAX_SLOTS,FEAT))
                    self.assertEqual(row["values"].shape,(MAX_SLOTS,))
                    self.assertEqual(row["valid"].shape,(MAX_SLOTS,))
                    self.assertIn(row["truth"],(0,1))
                    if task=="P":
                        self.assertEqual(row["n"],0)
                        self.assertEqual(row["truth"],
                                         row["public_bit"]^row["flip"])
                    else:
                        self.assertEqual(row["n"],1)
                        self.assertEqual(row["flip"],0)
                        self.assertEqual(row["values"][row["winner"]],
                                         float(row["truth"]*2-1))

    def test_training_signed_version_worlds_and_private_content(self):
        for j in range(64):
            rows=world("train",0,"S",j)
            self.assertEqual({r["variant"] for r in rows},set(VARIANTS))
            self.assertEqual({r["flip"] for r in rows},{0})
            self.assertEqual({r["n"] for r in rows},{(5,7)[j%2]})
            by={r["variant"]:r for r in rows}
            self.assertEqual(by["U_before"]["truth"],by["U_after"]["truth"])
            self.assertEqual(by["U_after"]["truth"],by["S_before"]["truth"])
            self.assertNotEqual(by["U_after"]["truth"],by["S_after"]["truth"])
            for row in rows:
                f=row["features"]
                valid=np.flatnonzero(row["valid"]>.5)
                signed=np.array([i for i in valid if f[i,0]>.5])
                latest=signed[np.argmax(f[signed,1])]
                self.assertEqual(row["winner"],latest)
                self.assertEqual(int(row["values"][latest]>0),row["truth"])
                self.assertTrue(any(f[i,0]==0 and f[i,1]>f[latest,1]
                                    for i in valid))
                self.assertEqual(len(set(f[valid,1])),len(valid))
            # Changing the report VALUE cannot alter selection logits.
            f,v,valid,p,flip,task,y,w=tensorize(rows)
            model=SkillActor()
            with torch.no_grad():
                logits=model.selector(f)
                v2=v.clone()
                v2*=-1
                self.assertTrue(torch.equal(logits,model.selector(f)))
                self.assertTrue(torch.equal(f,f.clone()))
                self.assertEqual(v2.shape,v.shape)

    def test_matched_experience_and_phase_order_are_preregistered(self):
        a=order_for("joint_shared",0)
        b=order_for("staged_shared",0)
        c=order_for("staged_unshared",0)
        d=order_for("supervised_ceiling",0)
        self.assertEqual(len(a),TOTAL_EPISODES)
        self.assertEqual(len(b),TOTAL_EPISODES)
        self.assertEqual(sorted(a.tolist()),sorted(b.tolist()))
        self.assertTrue(np.array_equal(b,c))
        self.assertTrue(np.array_equal(b,d))
        self.assertFalse(np.array_equal(a,b))
        self.assertTrue(set(b[:4096]).issubset(set(range(4096))))
        self.assertTrue(set(b[4096:8192]).issubset(set(range(4096,6144))))
        for seq in (a,b,c,d):
            counts=np.bincount((seq//2048),minlength=3)
            self.assertEqual(counts.tolist(),[8192]*3)

    def test_rl_reward_gradient_does_not_require_supervised_label(self):
        samples=world("train",0,"P",0)+world("train",0,"I",1)+world("train",0,"S",2)
        f,v,valid,p,flip,task,y,w=tensorize(samples)
        torch.manual_seed(225000)
        actor=SkillActor()
        out=actor.compute(f,v,valid,p,flip,task)
        reward=env_reward(out["action"],y)
        loss=-(reward.detach()*(out["action_logprob"]+
                                  out["slot_logprob"])).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(x.grad is not None and float(x.grad.abs().sum())>0
                            for x in actor.parameters()))
        # Flip ablation affects processor but not selector logits.
        altered_flip=(1-flip)
        with torch.no_grad():
            logits=actor.selector(f)
            actor.compute(f,v,valid,p,altered_flip,task)
            self.assertTrue(torch.equal(logits,actor.selector(f)))

if __name__=="__main__":
    unittest.main()
