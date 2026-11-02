"""F10-A contracts: protected partitions, F4 readout-only corruption, paired label gate."""
import unittest
import numpy as np
from observatory.f10a_actor_evidence_training import (
  train_seed,modify_tail,transform,QUALITIES,REGIMES,TRAIN_WORLDS
)
from observatory.f10a_actor_positivity_dev import dev_seed
from observatory.f10a_actor_dev_gate import CELLS,aggregate
from observatory.f9i_selective_memory import DIM,FIELDS
from minigrid.core.constants import OBJECT_TO_IDX,COLOR_TO_IDX

class NewActorTests(unittest.TestCase):
    def test_worlds_never_overlap_and_confirmation_sealed(self):
        a={train_seed(i,j) for i in range(4) for j in range(TRAIN_WORLDS)}
        b={dev_seed(i,j) for i in range(4) for j in range(32)}
        self.assertEqual(len(a),512)
        self.assertEqual(len(b),128)
        self.assertFalse(a&b)
        self.assertEqual(min(a),12100000)
        self.assertEqual(min(b),12200000)
        self.assertTrue(max(b)<12300000)
        for call in ((train_seed,(0,128)),(train_seed,(4,0)),(dev_seed,(0,32)),(dev_seed,(4,0))):
            with self.assertRaises(ValueError):call[0](*call[1])

    def test_true_fake_and_missing_affect_only_F4(self):
        tail=np.zeros(DIM-148,np.float32)
        tail[:8]=[1,0,.4,.3,.7,.2,.0,1]
        tail[8:16]=[0,1,.5,.2,.6,.3,.125,1]
        tail[16:24]=[.8,.8,.5,.5,.5,.5,.25,0]
        original=tail.copy()
        real=modify_tail(tail,"clean")
        corrupt=modify_tail(tail,"adversarial")
        blank=modify_tail(tail,"blank")
        self.assertTrue(np.array_equal(real,original))
        self.assertTrue(np.array_equal(corrupt[2:8],tail[2:8]))
        self.assertEqual(list(corrupt[:2]),[0.,1.])
        self.assertEqual(list(corrupt[8:10]),[1.,0.])
        self.assertTrue(np.array_equal(corrupt[16:24],original[16:24]))
        self.assertTrue(np.array_equal(tail,original))
        self.assertTrue(np.all(blank==0))
        self.assertEqual(len(QUALITIES),3)

    def test_teacher_treatments_preserve_raw_actions_and_history(self):
        raw=np.zeros((7,148),np.float32)
        obj=float(OBJECT_TO_IDX["key"])/max(OBJECT_TO_IDX.values())
        green=float(COLOR_TO_IDX["green"])/max(COLOR_TO_IDX.values())
        raw[:2,0]=obj
        raw[:2,1]=green
        act=np.array([0,1,0,2,2,0,1])
        times={"post_cue":2,"fork":5}
        normal=[]
        for mode in QUALITIES:
            x,_,_=transform(raw,act,times,mode,REGIMES[0],0)
            normal.append(x)
            self.assertEqual(x.shape,(7,DIM))
            self.assertTrue(np.array_equal(x[:,:148],raw))
        self.assertTrue(np.any(normal[0][3,148:]))
        self.assertTrue(np.any(normal[0][3,148:]!=normal[1][3,148:]))
        self.assertTrue(np.all(normal[2][3,148:]==0))
        for mode in QUALITIES:
            x,_,_=transform(raw,act,times,mode,REGIMES[1],0)
            self.assertTrue(np.all(x[2:5,148:]==0))
            self.assertTrue(np.array_equal(x[:2,:],normal[0][:2,:]))
        self.assertEqual(len(CELLS),4)

if __name__=="__main__":unittest.main()
