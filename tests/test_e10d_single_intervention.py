"""E10-D fixtures and policy contracts. Mock API ONLY within unit tests."""
import unittest
from pathlib import Path
from src.infinito3.memory import RuleBasedMemoryGate
from src.infinito3.types import MemoryKind
from src.infinito3.persistent_memory import HashEmbeddingProvider
from tests.test_e10b_semantic_longitudinal_live import FakeAPI,mocked_adapter
from observatory.e10b_semantic_longitudinal_live import CachedSemanticEmbeddings
from observatory.e10d_single_intervention import (
    ARMS,FAMILIES,BroadDeclarativeGate,FixedPlanIntentBuilder,
    seed_for,world_for,fact,histories,question,evaluate_world,
    primary_criterion,
)


class TestSingleIntervention(unittest.TestCase):
    def test_generic_no_domain_specific_gate(self):
        gate=BroadDeclarativeGate()
        for sentence in [
            "El acceso al proyecto Cauce pasa a otro equipo.",
            "Los archivos se conservarán durante cuarenta y cinco días.",
            "La revisión del plan Cobalto empieza a las siete.",
            "La ventana de mantenimiento cambió tras la reunión.",
        ]:
            g=gate.evaluate(sentence)
            self.assertTrue(g.should_store,sentence)
            self.assertIn("general_declarative_five_tokens",g.reasons)
            self.assertEqual(g.kind,MemoryKind.EPISODIC)
            self.assertEqual(g.importance,.60)
        self.assertFalse(gate.evaluate("¿Dónde están mis copias de seguridad?").should_store)
        self.assertFalse(gate.evaluate("Gracias").should_store)
        old=RuleBasedMemoryGate().evaluate("Mi color favorito es verde.")
        new=gate.evaluate("Mi color favorito es verde.")
        self.assertEqual(old,new)

    def test_plan_intent_correction_is_only_bilingual_false_positive(self):
        self.assertFalse(FixedPlanIntentBuilder._asks_for_goals(
            "¿Cuál es el horario vigente de revisión del plan Cauce-275?"))
        self.assertFalse(FixedPlanIntentBuilder._asks_for_goals(
            "La frecuencia del plan de mantenimiento es mensual."))
        self.assertTrue(FixedPlanIntentBuilder._asks_for_goals(
            "What is my plan for tomorrow?"))
        self.assertTrue(FixedPlanIntentBuilder._asks_for_goals(
            "Which plans are currently pending?"))
        self.assertTrue(FixedPlanIntentBuilder._asks_for_goals(
            "¿Cuáles son mis tareas pendientes?"))

    def test_new_seeds_and_contrafactual_distractors(self):
        d={seed_for("dev",0,j) for j in range(6)}
        c={seed_for("confirm",i,j) for i in range(4) for j in range(6)}
        self.assertEqual(len(d),6)
        self.assertEqual(len(c),24)
        self.assertFalse(d&c)
        self.assertEqual(min(d),2100000)
        self.assertEqual(min(c),2200000)
        self.assertEqual({world_for(x).family for x in d},set(FAMILIES))
        for j in range(6):
            w=world_for(seed_for("dev",0,j))
            n=histories(w,False);inv=histories(w,True)
            self.assertEqual(n[0][1:],inv[0][1:])
            self.assertEqual(n[1][1:],inv[1][1:])
            self.assertNotEqual(n[0][0],inv[0][0])
            self.assertNotEqual(n[1][0],inv[1][0])
            self.assertEqual(len(n[0]),13)
            self.assertEqual(len(n[1]),13)
            self.assertIn("¿",question(w))
        with self.assertRaises(ValueError):world_for(1800000)
        with self.assertRaises(ValueError):seed_for("confirm",4,0)

    def test_end_to_end_mock_only_new_dev_world_provenance(self):
        api=FakeAPI()
        emb=CachedSemanticEmbeddings(api)
        w=world_for(seed_for("dev",0,5))
        r=evaluate_world(w,api,emb,mocked_adapter)
        self.assertEqual(set(r["arms"]),set(ARMS))
        self.assertEqual(len(r["variants"]),2)
        self.assertEqual(r["arms"]["e10d_intervention"]["four_correct"],0)
        for pair in r["variants"].values():
            for t in ("first","second"):
                for arm in ARMS:
                    x=pair[t][arm]
                    self.assertEqual(x["provider_model"],"not-a-real-llm")
                    self.assertEqual(x["action"],"ABSTENER")
                    self.assertLessEqual(x["estimated_context_tokens"],320)
        self.assertGreater(emb.calls,0)

    def test_fixed_primary_gate_positive_and_negative(self):
        from observatory.e10d_single_intervention import summary
        # Pure metric example; no real decisions or confirmation seeds.
        stats={"by_family":{f:{"e10d_intervention":1,"semantic_rag":0}
                             for f in FAMILIES},
               "by_arm":{"e10d_intervention":{"negatives_abstained":6},
                         "semantic_rag":{"negatives_abstained":6}}}
        self.assertTrue(primary_criterion(stats)["passes"])
        for f in FAMILIES:
            stats["by_family"][f]["e10d_intervention"]=0
            stats["by_family"][f]["semantic_rag"]=1
        self.assertFalse(primary_criterion(stats)["passes"])


if __name__=="__main__":
    unittest.main()
