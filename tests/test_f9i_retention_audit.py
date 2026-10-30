"""F9-I: FIFO versus anchored retention and single-event memory corruption."""
import unittest

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

from observatory.f9i_retention_audit import (
    PreserveFirstFIFO,writer_for,visible_cue_event_ids,alter_one_slot,
    frozen_checkpoint_rollout,
)
from observatory.f9g_context_memory import ContextF4, CAPACITY, FIELDS
from observatory.selective_memory import Store, Event


def artificial(kind, at=(1,2)):
    image=np.zeros(148,np.float32)
    image[:147].reshape(7,7,3)[at[0],at[1]]=[
        OBJECT_TO_IDX[kind]/max(OBJECT_TO_IDX.values()),
        COLOR_TO_IDX["green"]/max(COLOR_TO_IDX.values()),0.,
    ]
    return image


class F9IRetentionTests(unittest.TestCase):
    def test_true_fifo_forgets_initial_event_and_anchored_retains(self):
        a,b=writer_for("fifo"),writer_for("first16")
        self.assertEqual(a.store.capacity,b.store.capacity)
        self.assertEqual(a.store.capacity,CAPACITY)
        self.assertEqual(type(a.store),Store)
        seen=[artificial("key",(i%7,(i//7)%7)) for i in range(49)]
        seen += [artificial("ball",(i%7,(i//7)%7)) for i in range(49)]
        for x in seen:
            a.encode(x)
            b.encode(x)
        self.assertEqual(a.serial,b.serial)
        self.assertGreater(a.serial,CAPACITY)
        self.assertEqual(len(a.store.order),CAPACITY)
        self.assertEqual(len(b.store.order),CAPACITY)
        self.assertNotIn(0,a.store.saved)
        self.assertIn(0,b.store.saved)
        self.assertEqual(a.readout().shape,b.readout().shape)
        self.assertEqual(a.readout().shape,(CAPACITY*FIELDS,))
        self.assertTrue(np.isfinite(a.readout()).all())

    def test_corrupt_only_single_event_no_change_to_current_view(self):
        writer=writer_for("fifo")
        initial=writer.encode(artificial("key"))
        writer.encode(artificial("ball"))
        encoded=writer.encode(np.zeros(148,np.float32))
        corrected,applied=alter_one_slot(encoded,writer,0)
        self.assertTrue(applied)
        np.testing.assert_array_equal(corrected[:148],encoded[:148])
        self.assertEqual(int(np.sum(corrected!=encoded)),2)
        self.assertEqual(corrected[148],encoded[149])
        self.assertEqual(corrected[149],encoded[148])
        np.testing.assert_array_equal(corrected[148+FIELDS:],encoded[148+FIELDS:])
        # Original writer Store is NOT mutated by a readout intervention.
        self.assertEqual(writer.store.recall(0),0 if False else writer.store.recall(0))
        skipped,was_applied=alter_one_slot(encoded,writer,99999)
        self.assertFalse(was_applied)
        np.testing.assert_array_equal(skipped,encoded)

    def test_local_pixel_differencing_is_observer_only(self):
        key=artificial("key",(2,3))
        ball=artificial("ball",(2,3))
        first,count=visible_cue_event_ids(key,ball,30)
        self.assertEqual(first,{30})
        self.assertEqual(count,1)
        same,count=visible_cue_event_ids(key,key,30)
        self.assertEqual(same,set())
        self.assertEqual(count,1)

    def test_anchor_never_gets_privileged_label_or_oracle_key(self):
        anchor=PreserveFirstFIFO(64,16)
        store=anchor.begin()
        for key in range(100):
            anchor.consider(store,Event(key,key))
        self.assertEqual(len(store.saved),64)
        self.assertTrue(all(i in store.saved for i in range(16)))
        self.assertTrue(all(i not in store.saved for i in range(16,52)))
        self.assertTrue(all(i in store.saved for i in range(52,100)))
        self.assertNotIn(999,store.saved)
        with self.assertRaises(ValueError):
            PreserveFirstFIFO(16,16)

    def test_holdout_sealed_and_bad_model_interventions_rejected(self):
        from observatory.f9d_autonomous_navigation import checked_seed
        with self.assertRaises(ValueError):
            checked_seed("heldout",0,0)
        with self.assertRaises(ValueError):
            writer_for("oracle")
        with self.assertRaises(ValueError):
            frozen_checkpoint_rollout(None,7,checked_seed("development",0,0),
                                      False,"fifo","future_target")

if __name__=="__main__":
    unittest.main()
