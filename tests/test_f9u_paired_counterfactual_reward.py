"""F9-U contracts; fake trajectories below are NEVER scientific results."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from observatory.f9u_paired_counterfactual_reward import (
    ARMS, CONTROLS, PAIR_BONUS, READ_COST,
    head_init,seed_for,paired_reward,loss_for_pair,train_head,
)


def fake_episode(return_value,read_rate,logp):
    return {
        "native_return":return_value,
        "read_rate":read_rate,
        "objective_return":return_value-READ_COST*read_rate,
        "success":int(return_value>0),
        "first_choice_correct":int(return_value>0),
        "_log_probs":[logp],
        "_entropies":[torch.tensor(0.0)],
    }


class F9UPairedCounterfactualContracts(unittest.TestCase):
    def test_new_dev_seeds_strictly_disjoint_from_previous(self):
        a={seed_for("train",i,j) for i in range(4) for j in range(600)}
        b={seed_for("dev",i,j) for i in range(4) for j in range(20)}
        self.assertEqual(len(a),2400)
        self.assertEqual(len(b),80)
        self.assertFalse(a&b)
        self.assertTrue(all(7100000<=x<7200000 for x in a))
        self.assertTrue(all(7200000<=x<7300000 for x in b))
        for split,idx,j in [
            ("confirm",0,0),("heldout",0,0),("train",4,0),
            ("train",0,600),("dev",0,20),("train",0,-1)]:
            with self.assertRaises(ValueError):
                seed_for(split,idx,j)

    def test_paired_reward_is_symmetric_and_encourages_both_successes(self):
        a={"native_return":1.0,"read_rate":0.2}
        b={"native_return":1.0,"read_rate":0.2}
        c={"native_return":0.0,"read_rate":0.2}
        # Two successes: 1 + 0.5 - 0.03 = 1.47;
        # only one: 0.5 + 0 - 0.03 = 0.47.
        self.assertAlmostEqual(paired_reward(a,b),1.47)
        self.assertAlmostEqual(paired_reward(a,c),0.47)
        self.assertAlmostEqual(paired_reward(c,a),paired_reward(a,c))
        self.assertGreater(paired_reward(a,b),paired_reward(a,c))
        with self.assertRaises(ValueError):
            paired_reward(a,b,beta=PAIR_BONUS+0.1)

    def test_same_init_by_information_input_and_distinct_methods(self):
        for index in range(4):
            a=head_init(index,"hc")
            b=head_init(index,"hc")
            self.assertTrue(all(torch.equal(x,y)
                for x,y in zip(a.parameters(),b.parameters())))
            x=head_init(index,"public")
            y=head_init(index,"public")
            self.assertTrue(all(torch.equal(v,w)
                for v,w in zip(x.parameters(),y.parameters())))

    def test_loss_uses_pair_while_control_uses_individual_returns(self):
        a_log=torch.tensor(-0.3,requires_grad=True)
        b_log=torch.tensor(-0.8,requires_grad=True)
        a=fake_episode(1.0,0.2,a_log)
        b=fake_episode(0.0,0.2,b_log)
        individual,ri=loss_for_pair(a,b,0.0,"individual")
        paired,rp=loss_for_pair(a,b,0.0,"paired")
        self.assertAlmostEqual(ri,0.47)
        self.assertAlmostEqual(rp,0.47)
        # Even when the numeric value matches here, the estimator differs
        # structurally: the pair treatment shares one reward across BOTH logs.
        ga=torch.autograd.grad(individual,[a_log,b_log],retain_graph=True)
        gb=torch.autograd.grad(paired,[a_log,b_log])
        self.assertNotAlmostEqual(float(ga[0]),float(ga[1]))
        self.assertAlmostEqual(float(gb[0]),float(gb[1]))
        with self.assertRaises(ValueError):
            loss_for_pair(a,b,0.0,"oracle")

    def test_both_cues_processed_before_each_optimizer_step(self):
        seen=[]
        def fake_rollout(actor,head,index,seed,reverse,mode,train):
            self.assertTrue(train)
            weight=next(head.parameters()).flatten()[0]
            seen.append((seed,reverse,float(weight.detach())))
            x=torch.ones(276)
            logp=-torch.nn.functional.softplus(-head(x))
            return fake_episode(1.0 if reverse else 0.0,0.25,logp)
        with patch("observatory.f9u_paired_counterfactual_reward.rollout",
                   side_effect=fake_rollout):
            model,stats=train_head(object(),0,"paired_hc",worlds=2)
        self.assertEqual(stats["episodes"],4)
        self.assertEqual(stats["optimizer_steps"],2)
        self.assertEqual([v[1] for v in seen],[False,True,False,True])
        self.assertEqual(seen[0][0],seen[1][0])
        self.assertEqual(seen[2][0],seen[3][0])
        self.assertEqual(seen[0][2],seen[1][2])
        self.assertEqual(seen[2][2],seen[3][2])
        self.assertIsNotNone(model)

    def test_no_external_model_or_future_oracle_inputs(self):
        self.assertEqual(len(ARMS),4)
        self.assertEqual(set(ARMS),{
            "individual_hc","paired_hc","individual_public","paired_public"})
        self.assertEqual(set(CONTROLS),{
            "always","never","periodic_4","public_event_clock"})

if __name__=="__main__":
    unittest.main()
