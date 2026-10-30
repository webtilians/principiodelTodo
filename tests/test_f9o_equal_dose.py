"""F9-O exact-dose control, frozen model parity and partition isolation."""
import unittest
import tempfile
import hashlib
import numpy as np
from pathlib import Path

from observatory.f9d_autonomous_navigation import demonstration,checked_seed
from observatory.f9o_long_blackout_training import (
    ARMS,OFFSETS,train_seed,mask_timeline,prepare_dataset,make_model,
)
from observatory.f9o_confirm_long_blackout import (
    confirmation_seed,validate_training,
)
from observatory.f9m_temporal_dropout_training import parameter_signature


class F9OContracts(unittest.TestCase):
    def test_exact_temporal_difference_equal_mask_dose(self):
        self.assertEqual(ARMS,("fragmented6","continuous6"))
        self.assertEqual(len(OFFSETS["fragmented6"]),6)
        self.assertEqual(len(OFFSETS["continuous6"]),6)
        self.assertEqual(OFFSETS["continuous6"],frozenset(range(6)))
        self.assertNotEqual(OFFSETS["continuous6"],OFFSETS["fragmented6"])
        # We inspect TRAIN-only S13 teacher episodes, never 600000 confirmation.
        worlds=[
            demonstration(train_seed(0,j),size=13,reverse=reverse)
            for j in range(600) for reverse in (False,True)
        ]
        for frames,actions,vis in worlds:
            a,aa=mask_timeline(frames,"fragmented6",0)
            b,bb=mask_timeline(frames,"continuous6",0)
            np.testing.assert_array_equal(a[:,:148],frames)
            np.testing.assert_array_equal(b[:,:148],frames)
            self.assertEqual(aa["scheduled"],bb["scheduled"])
            self.assertEqual(aa["effective"],bb["effective"])
            self.assertEqual(aa["effective"],6)
            self.assertEqual(len(set(aa["changed_indices"])),6)
            self.assertEqual(len(set(bb["changed_indices"])),6)
            self.assertNotEqual(aa["changed_indices"],bb["changed_indices"])
        a,sa=prepare_dataset(worlds,"fragmented6",0)
        b,sb=prepare_dataset(worlds,"continuous6",0)
        self.assertEqual(sa["effective"],sb["effective"])
        self.assertEqual(sa["teacher_sha256"],sb["teacher_sha256"])
        for x,y in zip(a,b):
            np.testing.assert_array_equal(x[1],y[1])
            np.testing.assert_array_equal(x[0][:,:148],y[0][:,:148])

    def test_identical_new_context8_model_initialization(self):
        a=make_model("fragmented6",0)
        b=make_model("continuous6",0)
        try:
            self.assertEqual(a.observation_space.shape,(660,))
            self.assertEqual(parameter_signature(a.policy),parameter_signature(b.policy))
            self.assertEqual(sum(x.numel() for x in a.policy.parameters()),380552)
        finally:
            a.get_env().close()
            b.get_env().close()

    def test_validation_gate_accepts_six_exact_frames_per_demonstration(self):
        # Synthetic files only; verify gate BEFORE any expensive PPO / confirmation.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            arms={}
            for arm in ARMS:
                details={
                    "initial_policy_sha256":"matched",
                    "parameters":380552,
                    "input_width":660,
                    "actual_rl_steps":20096,
                    "bc_mask_stats":{
                        "episodes":1200,"incomplete":0,
                        "scheduled":7200,"effective":7200,
                        "teacher_sha256":"same-demo",
                    },
                }
                for phase,field in (("after_bc","bc_sha256"),
                                    ("after_rl","rl_sha256")):
                    path=root/f"{arm}_{phase}.zip"
                    payload=(arm+phase).encode()
                    path.write_bytes(payload)
                    details[field]=hashlib.sha256(payload).hexdigest()
                arms[arm]=details
            report={
                "index":0,"training_seed_base":500000,
                "worlds":600,"epochs":40,
                "requested_rl_steps":20000,
                "training_dose_per_teacher_episode":6,
                "ppo_blackout":False,
                "heldout_opened":False,
                "confirmation_opened":False,
                "teacher_training_only":True,
                "arms":arms,
            }
            validate_training(report,root,0)
            report["arms"]["fragmented6"]["bc_mask_stats"]["effective"]=7199
            with self.assertRaises(AssertionError):
                validate_training(report,root,0)

    def test_sealed_new_partitions(self):
        self.assertEqual(train_seed(0,0),500000)
        self.assertEqual(train_seed(3,599),530599)
        self.assertEqual(confirmation_seed(0,13,0),600000)
        self.assertEqual(confirmation_seed(3,13,39),630039)
        self.assertEqual(confirmation_seed(3,7,19),630019)
        self.assertLess(confirmation_seed(3,13,39),900000)
        self.assertNotEqual(train_seed(0,0),checked_seed("development",0,0))
        for fn,items in (
            (train_seed,[(4,0),(0,600),(-1,0)]),
            (lambda i,j:confirmation_seed(i,13,j),[(4,0),(0,40),(-1,0)])
        ):
            for args in items:
                with self.assertRaises(ValueError):
                    fn(*args)
        with self.assertRaises(ValueError):
            checked_seed("heldout",0,0)
        with self.assertRaises(AssertionError):
            validate_training({"index":0},Path("."),0)


if __name__=="__main__":
    unittest.main()
