import copy
import unittest
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from observatory.selective_memory import Event, LearnedRelevance
from observatory.f9g_context_memory import ContextF4
from observatory.f9i_selective_memory import SelectiveF4, corrupt_one, DIM
from observatory.f9d_autonomous_navigation import checked_seed
from observatory import f9i_selective_navigation as nav


def frame(kind="key", x=1, y=1, color=0):
    raw = np.zeros(148,np.float32)
    raw[:147].reshape(7,7,3)[x,y,:2] = (
        OBJECT_TO_IDX[kind]/max(OBJECT_TO_IDX.values()),
        color/max(COLOR_TO_IDX.values()))
    return raw


class SelectiveTests(unittest.TestCase):
    def test_fifo_exact_f9h_parity(self):
        old,new = ContextF4(),SelectiveF4("fifo")
        for i in range(90):
            raw = frame("ball" if i%2 else "key",i%7,(i//7)%7)
            np.testing.assert_array_equal(old.encode(raw),new.encode(raw))

    def test_bounded_capacity_and_context_retains_earliest(self):
        for capacity in (8,64):
            writers = [SelectiveF4(m,capacity,17) for m in ("fifo","context","random","zero")]
            for i in range(90):
                for writer in writers:
                    out=writer.encode(frame())
                    self.assertEqual(out.shape,(DIM,))
                    self.assertLessEqual(len(writer.store.order),capacity)
                    self.assertTrue(np.isfinite(out).all())
            self.assertNotIn(0,writers[0].store.saved)
            self.assertIn(0,writers[1].store.saved)
            self.assertFalse(writers[3].store.saved)

    def test_context_priority_blind_to_identity(self):
        a,b=SelectiveF4("context",8),SelectiveF4("context",8)
        for i in range(40):
            a.encode(frame("key",i%7,(i//7)%7))
            b.encode(frame("ball",i%7,(i//7)%7))
            self.assertEqual(a.store.order,b.store.order)

    def test_single_corruption_exactly_two_channels_and_no_mutation(self):
        writer=SelectiveF4("context",8)
        writer.encode(frame())
        obs=writer.encode(frame("ball",2,2))
        before=copy.deepcopy(writer.store.__dict__)
        out=corrupt_one(obs,writer,0)
        self.assertEqual(np.flatnonzero(obs!=out).tolist(),[148,149])
        self.assertEqual(before,writer.store.__dict__)
        np.testing.assert_array_equal(corrupt_one(obs,writer,1000),obs)
        np.testing.assert_array_equal(obs[:148],out[:148])

    def test_random_preassigned_and_reset_has_no_payload_cache(self):
        a,b=SelectiveF4("random",8,41),SelectiveF4("random",8,41)
        for i in range(100):
            np.testing.assert_array_equal(a.encode(frame()),b.encode(frame()))
        a.reset()
        self.assertEqual(a.store.saved,{})
        self.assertEqual(a.serial,0)

    def test_split_bounds(self):
        train={checked_seed("train",i,j) for i in range(4) for j in range(600)}
        dev={checked_seed("development",i,j) for i in range(4) for j in range(100)}
        self.assertFalse(train & dev)
        with self.assertRaises(ValueError):
            checked_seed("heldout",0,0)

    def test_real_learned_relevance_is_not_serial_compatible(self):
        learner=LearnedRelevance(8,0)
        store=learner.begin()
        for i in range(8):
            learner.consider(store,Event(i,0))
        with self.assertRaises(IndexError):
            learner.consider(store,Event(8,0))
        learner.feedback(0)  # Valid only AFTER an externally observed query.
        self.assertEqual(learner.feedback_count,1)
        learner.freeze()
        before=list(learner.weights)
        learner.feedback(1)
        self.assertEqual(before,learner.weights)

    def test_model_capacity_and_native_controls(self):
        original=nav.EVENT_CAPACITY
        models=[]
        try:
            for capacity in (8,64):
                nav.EVENT_CAPACITY=capacity
                for mode in nav.MODES:
                    model=nav.make_model(mode,0,2)
                    models.append(model)
                    self.assertEqual(model.action_space.n,7)
                    self.assertEqual(model.observation_space.shape,(660,))
            self.assertEqual(len({sum(p.numel() for p in m.policy.parameters()) for m in models}),1)
        finally:
            for model in models:
                model.get_env().close()
            nav.EVENT_CAPACITY=original


if __name__=="__main__":
    unittest.main()
