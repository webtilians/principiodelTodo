"""F9-G: strictly observable contextual F4 memory, no oracle leakage."""
import unittest
import numpy as np

from observatory.f9g_contextual_memory import (
    ContextualF4, visible_spatial_bits, unpack_spatial_bits, demonstration_pair_probe,
    READOUT_WIDTH, TOTAL_WIDTH,
)
from observatory.f9e_f4_autonomous import VisibleF4
from observatory.f9d_autonomous_navigation import checked_seed, demonstration
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX


def simulated_frame(key_on_left=True):
    """Both key and ball visible; only their RELATIVE PIXEL POSITIONS change."""
    frame=np.zeros(148,dtype=np.float32)
    image=frame[:147].reshape(49,3)
    objectmax=max(OBJECT_TO_IDX.values())
    green=COLOR_TO_IDX['green']/max(COLOR_TO_IDX.values())
    for pos, typ in [(9,'key' if key_on_left else 'ball'),
                     (11,'ball' if key_on_left else 'key')]:
        image[pos]=(OBJECT_TO_IDX[typ]/objectmax,green,0)
    frame[147]=1/3
    return frame


class ContextualMemoryTests(unittest.TestCase):
    def test_spatial_object_information_not_just_set_of_seen_types(self):
        a,b=simulated_frame(True),simulated_frame(False)
        fa,fb=VisibleF4(),VisibleF4()
        # F9-E cannot distinguish these paired views: both types present.
        np.testing.assert_array_equal(fa.encode(a)[148:],fb.encode(b)[148:])
        self.assertNotEqual(visible_spatial_bits(a),visible_spatial_bits(b))
        sa=ContextualF4().encode(a)
        sb=ContextualF4().encode(b)
        self.assertEqual(sa.shape,(TOTAL_WIDTH,))
        self.assertFalse(np.array_equal(sa[148:],sb[148:]))
        self.assertEqual(len(unpack_spatial_bits(visible_spatial_bits(a))),98)

    def test_preserve_anchor_across_empty_frames_and_changing_views(self):
        writer=ContextualF4()
        source=simulated_frame(True)
        original=writer.encode(source)[148:].copy()
        # Move through 25 frames; first snapshot remains pinned.
        for step in range(25):
            if step%3==0:
                frame=simulated_frame(not bool(step%2))
            else:
                frame=np.zeros(148,dtype=np.float32)
            writer.encode(frame)
        final=writer.readout()
        np.testing.assert_array_equal(
            original[:98],final[:98]
        )
        self.assertEqual(final[99],1.)
        self.assertLessEqual(len(writer.store.order),3)
        self.assertEqual(type(writer.store).__name__,'Store')
        writer.reset()
        self.assertEqual(writer.store.saved,{})
        self.assertFalse(np.any(writer.readout()))

    def test_controls_keep_same_current_observation_and_readout_dimension(self):
        obs=simulated_frame()
        results={}
        for mode in ('context','zero','erase','swap_identity'):
            writer=ContextualF4(mode)
            composed=writer.encode(obs)
            np.testing.assert_array_equal(composed[:148],obs)
            self.assertEqual(composed.shape,(TOTAL_WIDTH,))
            self.assertTrue((composed>=0).all() and (composed<=1).all())
            results[mode]=composed[148:]
        self.assertTrue(np.any(results['context']))
        self.assertFalse(np.any(results['zero']))
        self.assertFalse(np.any(results['erase']))
        self.assertFalse(np.array_equal(results['swap_identity'],results['context']))
        self.assertEqual(results['zero'].shape,(READOUT_WIDTH,))

    def test_native_teacher_decision_pair_uses_identical_current_frame(self):
        for size in (7,13):
            for i in range(10):
                with self.subTest(size=size,world=i):
                    seed=checked_seed('development',0,i)
                    probe=demonstration_pair_probe(seed,size)
                    self.assertTrue(probe['identical_public_frame_at_decision'])
                    self.assertTrue(probe['decision_readout_distinct'])
                    self.assertEqual(probe['memory_readout_width'],READOUT_WIDTH)

    def test_refuses_heldout_and_invalid_frames(self):
        with self.assertRaises(ValueError):
            checked_seed('heldout',0,0)
        with self.assertRaises(ValueError):
            ContextualF4('oracle')
        with self.assertRaises(ValueError):
            visible_spatial_bits(np.zeros(149,dtype=np.float32))


if __name__=="__main__":
    unittest.main()
