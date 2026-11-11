"""E11-A deterministic contracts and no-future-feedback tests, no confirm access."""
import inspect
import unittest

from observatory.e11_learned_relevance import (
    MEMORY_CAPACITY,EVENT_COUNT,REPLICAS,FEATURE_NAMES,
    BoundedMemory,Event,FIFO,Reservoir,RankSelector,FeedbackLearner,
    seed_for,episode_for,run_trial,fixed_priority,recency_priority,
    eval_cohort,training_models,bootstrap,
)


class E11LearnedSelectionTests(unittest.TestCase):
    def test_disjoint_ids_and_splits_and_frozen_negative_regimes(self):
        train={seed_for("train",r,j) for r in range(5) for j in range(20)}
        dev={seed_for("dev",r,j) for r in range(5) for j in range(20)}
        confirm={seed_for("confirm",r,j) for r in range(5) for j in range(20)}
        self.assertEqual(len(train),100)
        self.assertFalse(train&dev or dev&confirm or train&confirm)
        self.assertEqual(MEMORY_CAPACITY,4)
        self.assertEqual(EVENT_COUNT,16)
        self.assertEqual(REPLICAS,5)
        self.assertEqual(len(FEATURE_NAMES),7)
        for args in (("confirm",5,0),("confirm",0,640),
                     ("train",0,2500),("dev",0,160)):
            with self.assertRaises(ValueError):
                seed_for(*args)
        t=episode_for("train",0,0,"predictive")
        d=episode_for("dev",0,0,"predictive")
        c=episode_for("confirm",0,0,"predictive")
        self.assertEqual(len({e.key for e in t.events}),16)
        self.assertFalse({e.key for e in t.events}&{e.key for e in d.events})
        self.assertFalse({e.key for e in d.events}&{e.key for e in c.events})

    def test_memory_is_really_bounded_not_hidden_index(self):
        ep=episode_for("train",0,0,"predictive")
        policies=(FIFO(9),Reservoir(9),RankSelector(9,fixed_priority()))
        for policy in policies:
            memory=policy.begin()
            for e in ep.events:
                policy.consider(memory,e)
                self.assertLessEqual(len(memory.saved),MEMORY_CAPACITY)
            self.assertEqual(len(memory.saved),MEMORY_CAPACITY)
            self.assertEqual(set(memory.saved),set(memory.order))
            self.assertFalse(hasattr(memory,"all_events"))
        m=BoundedMemory()
        for e in ep.events[:4]:m.put(e)
        with self.assertRaises(ValueError):m.put(ep.events[4])
        old=m.order[0]
        m.put(ep.events[4],old)
        self.assertIsNone(m.recall(old))

    def test_selector_does_not_receive_future_query_key(self):
        params=list(inspect.signature(RankSelector.consider).parameters)
        self.assertEqual(params,["self","memory","event"])
        ep=episode_for("train",0,1,"predictive")
        learned=FeedbackLearner()
        m=RankSelector(0,learned.weights).begin()
        scores=list(learned.weights)
        p=RankSelector(0,learned.weights)
        for e in ep.events:
            p.consider(m,e)
        self.assertEqual(scores,learned.weights)
        self.assertEqual(learned.examples,0)
        learned.feedback(ep.events,ep.future_query_key)
        self.assertEqual(learned.examples,1)
        self.assertNotEqual(scores,learned.weights)

    def test_same_stream_across_negative_and_predictive_regimes(self):
        worlds=[episode_for("dev",0,0,x) for x in ("predictive","uniform","reversed")]
        self.assertEqual(worlds[0].events,worlds[1].events)
        self.assertEqual(worlds[1].events,worlds[2].events)
        self.assertEqual(worlds[0].seed,worlds[1].seed)

    def test_original_F4_key_specific_strategy_not_reused(self):
        # E11 features are transferable; identity must never enter the vector.
        ep=episode_for("train",0,0,"predictive")
        x=ep.events[0]
        renamed=Event("never-seen-key",x.payload,x.urgent,x.pending,
                      x.previously_queried,x.source,x.position)
        self.assertEqual(x.features(),renamed.features())

    def test_training_reproducible_and_negative_control_frozen(self):
        # Training is deterministic across replicas and script runs.
        a=training_models()
        b=training_models()
        self.assertEqual(a,b)
        self.assertEqual(len(a),5)
        for m in a.values():
            self.assertEqual(m["train_examples"],2500)
            self.assertEqual(len(m["learned_weights"]),7)
            self.assertEqual(len(m["shuffled_weights"]),7)
        rows,stats=eval_cohort("dev",a,"predictive")
        self.assertEqual(len(rows),800)
        self.assertEqual(stats["episodes"],800)
        for name,row in stats["policies"].items():
            self.assertGreaterEqual(row["hit"],0)
            self.assertLessEqual(row["hit"],800)
        self.assertEqual(stats["policies"]["fifo"]["hit"],
                         stats["policies"]["fifo"]["rate"]*800)

    def test_shared_capacity_every_policy(self):
        model={"learned_weights":[0.]*7,"shuffled_weights":[0.]*7}
        ep=episode_for("train",0,3,"predictive")
        row=run_trial(ep,model)
        self.assertEqual(row["seed"],ep.seed)
        for v in row["policies"].values():
            self.assertEqual(v["saved_count"],4)
            self.assertEqual(v["memory_capacity_violations"],0)
            self.assertEqual(v["hit"],v["correct"])

if __name__=="__main__":
    unittest.main()
