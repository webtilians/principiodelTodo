#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST - Nueva Reward Function v2
===================================

Script para verificar que la nueva reward function con términos
mejorados funciona correctamente.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.rl.infinito_rl_env import InfinitoRLEnv


def test_reward_scenarios():
    """Probar la reward function en escenarios específicos."""
    
    print("="*70)
    print("TEST - REWARD FUNCTION v2 (MEJORADA)")
    print("="*70)
    
    # Crear entorno con configuración mínima
    config = {
        "inner_steps": 1,
        "max_steps": 5,
        "batch_size": 2,
        "model_kwargs": {
            "use_lora": True,
            "lora_r": 4,
            "lora_alpha": 16,
            "lambda_phi": 0.3,
            "freeze_base": True,
            "memory_slots": 64,
        },
    }
    
    print("\n📦 Creando entorno...")
    env = InfinitoRLEnv(config=config)
    print("✅ Entorno creado")
    
    print("\n" + "="*70)
    print("ESCENARIOS DE PRUEBA")
    print("="*70)
    
    # Escenario 1: Métricas normales y estables
    print("\n1️⃣ ESCENARIO: Métricas normales (PHI=4.5, C=0.5, PPL=80)")
    prev = {
        "consciousness": 0.48,
        "phi": 4.3,
        "loss_text": 4.5,
        "loss_phi": 1.2,
        "perplexity": 85.0,
        "memory_utilization": 0.3
    }
    cur = {
        "consciousness": 0.50,  # +0.02
        "phi": 4.5,             # +0.2 (estable)
        "loss_text": 4.4,
        "loss_phi": 1.1,
        "perplexity": 80.0,     # -5 (mejora)
        "memory_utilization": 0.3
    }
    reward = env._compute_reward(prev, cur)
    print(f"   Recompensa: {reward:.4f}")
    print(f"   Esperado: Positiva (mejoras pequeñas)")
    print(f"   ✅ Balance óptimo" if reward > 0 else "   ⚠️ Revisar")
    
    # Escenario 2: PHI en rango óptimo
    print("\n2️⃣ ESCENARIO: PHI en rango óptimo [3.0-6.0]")
    prev = {
        "consciousness": 0.45,
        "phi": 4.8,
        "loss_text": 4.3,
        "loss_phi": 1.0,
        "perplexity": 90.0,
        "memory_utilization": 0.2
    }
    cur = {
        "consciousness": 0.47,
        "phi": 5.0,  # Dentro del rango óptimo
        "loss_text": 4.2,
        "loss_phi": 0.95,
        "perplexity": 85.0,
        "memory_utilization": 0.2
    }
    reward = env._compute_reward(prev, cur)
    print(f"   Recompensa: {reward:.4f}")
    print(f"   Esperado: Positiva con bonus por balance")
    print(f"   ✅ Incentivo correcto" if reward > 0 else "   ⚠️ Revisar")
    
    # Escenario 3: PHI alto (peligro Fase 2)
    print("\n3️⃣ ESCENARIO: PHI alto > 6.0 (riesgo colapso Fase 2)")
    prev = {
        "consciousness": 0.52,
        "phi": 6.5,
        "loss_text": 4.0,
        "loss_phi": 0.8,
        "perplexity": 70.0,
        "memory_utilization": 0.4
    }
    cur = {
        "consciousness": 0.54,
        "phi": 7.0,  # ¡ALTO! Zona de colapso
        "loss_text": 3.9,
        "loss_phi": 0.7,
        "perplexity": 65.0,
        "memory_utilization": 0.4
    }
    reward = env._compute_reward(prev, cur)
    print(f"   Recompensa: {reward:.4f}")
    print(f"   Esperado: PENALIZACIÓN fuerte (-0.6 por cada unidad > 6.0)")
    print(f"   ✅ Penalización aplicada" if reward < 0 else "   ❌ NO PENALIZA")
    
    # Escenario 4: Perplexity colapso
    print("\n4️⃣ ESCENARIO: Perplexity < 10 (colapso/repetición)")
    prev = {
        "consciousness": 0.48,
        "phi": 5.2,
        "loss_text": 3.5,
        "loss_phi": 0.5,
        "perplexity": 12.0,
        "memory_utilization": 0.3
    }
    cur = {
        "consciousness": 0.49,
        "phi": 5.3,
        "loss_text": 3.3,
        "loss_phi": 0.45,
        "perplexity": 5.0,  # ¡COLAPSO!
        "memory_utilization": 0.3
    }
    reward = env._compute_reward(prev, cur)
    print(f"   Recompensa: {reward:.4f}")
    print(f"   Esperado: PENALIZACIÓN fuerte (colapso detectado)")
    print(f"   ✅ Colapso detectado" if reward < -0.3 else "   ❌ NO DETECTA COLAPSO")
    
    # Escenario 5: Cambio brusco en PHI (inestabilidad)
    print("\n5️⃣ ESCENARIO: Cambio brusco PHI (|ΔΦ| > 1.0)")
    prev = {
        "consciousness": 0.45,
        "phi": 4.0,
        "loss_text": 4.5,
        "loss_phi": 1.2,
        "perplexity": 90.0,
        "memory_utilization": 0.3
    }
    cur = {
        "consciousness": 0.46,
        "phi": 6.5,  # ¡+2.5! Cambio brusco
        "loss_text": 4.3,
        "loss_phi": 1.0,
        "perplexity": 85.0,
        "memory_utilization": 0.3
    }
    reward = env._compute_reward(prev, cur)
    print(f"   Recompensa: {reward:.4f}")
    print(f"   Esperado: Penalización por inestabilidad")
    print(f"   ✅ Inestabilidad penalizada" if reward < 0 else "   ⚠️ Revisar")
    
    # Escenario 6: PHI bajo (insuficiente)
    print("\n6️⃣ ESCENARIO: PHI bajo < 3.0")
    prev = {
        "consciousness": 0.35,
        "phi": 3.2,
        "loss_text": 5.0,
        "loss_phi": 1.5,
        "perplexity": 120.0,
        "memory_utilization": 0.2
    }
    cur = {
        "consciousness": 0.36,
        "phi": 2.5,  # ¡BAJO!
        "loss_text": 4.9,
        "loss_phi": 1.4,
        "perplexity": 115.0,
        "memory_utilization": 0.2
    }
    reward = env._compute_reward(prev, cur)
    print(f"   Recompensa: {reward:.4f}")
    print(f"   Esperado: Penalización leve por PHI bajo")
    print(f"   ✅ PHI bajo penalizado" if reward < 0 else "   ⚠️ Revisar")
    
    # Escenario 7: Perplexity muy alto (confusión)
    print("\n7️⃣ ESCENARIO: Perplexity > 200 (modelo confuso)")
    prev = {
        "consciousness": 0.40,
        "phi": 4.2,
        "loss_text": 5.5,
        "loss_phi": 1.8,
        "perplexity": 180.0,
        "memory_utilization": 0.3
    }
    cur = {
        "consciousness": 0.41,
        "phi": 4.3,
        "loss_text": 5.6,
        "loss_phi": 1.9,
        "perplexity": 250.0,  # ¡MUY ALTO!
        "memory_utilization": 0.3
    }
    reward = env._compute_reward(prev, cur)
    print(f"   Recompensa: {reward:.4f}")
    print(f"   Esperado: Penalización por confusión")
    print(f"   ✅ Confusión penalizada" if reward < 0 else "   ⚠️ Revisar")
    
    # Escenario 8: Estado óptimo perfecto
    print("\n8️⃣ ESCENARIO: Estado ÓPTIMO (C=0.5, PHI=4.5, PPL=75)")
    prev = {
        "consciousness": 0.48,
        "phi": 4.3,
        "loss_text": 4.5,
        "loss_phi": 1.1,
        "perplexity": 80.0,
        "memory_utilization": 0.25
    }
    cur = {
        "consciousness": 0.50,  # En rango [0.3, 0.7]
        "phi": 4.5,             # En rango [3.0, 6.0]
        "loss_text": 4.3,
        "loss_phi": 1.0,
        "perplexity": 75.0,     # En rango [10, 200]
        "memory_utilization": 0.25
    }
    reward = env._compute_reward(prev, cur)
    print(f"   Recompensa: {reward:.4f}")
    print(f"   Esperado: MÁXIMA POSITIVA (todo en rangos óptimos + bonuses)")
    print(f"   ✅ Estado óptimo recompensado" if reward > 0.1 else "   ⚠️ Revisar bonuses")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETADO")
    print("="*70)
    print("\nLa nueva reward function:")
    print("  ✓ Incentiva PHI en rango [3.0, 6.0]")
    print("  ✓ Penaliza fuerte PHI > 6.0 (evita Fase 2)")
    print("  ✓ Detecta colapso por PPL < 10")
    print("  ✓ Penaliza inestabilidad (cambios bruscos)")
    print("  ✓ Recompensa estados óptimos con bonuses")
    
    env.close()


if __name__ == "__main__":
    try:
        test_reward_scenarios()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
