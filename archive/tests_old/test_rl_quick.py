#!/usr/bin/env python3
"""Test rápido del modelo RL - versión simplificada."""

import torch
import numpy as np
from stable_baselines3 import PPO
import json

print("="*70)
print("🧪 TEST RÁPIDO - MODELO RL ENTRENADO")
print("="*70)

# Cargar config
with open("outputs/rl_phi_text_scheduler/env_config.json", 'r') as f:
    env_config = json.load(f)
print("✅ Config cargada")

# Importar entorno
from src.rl.infinito_rl_env import InfinitoRLEnv

# Crear entorno
print("📦 Creando entorno...")
env = InfinitoRLEnv(config=env_config)
print("✅ Entorno creado")

# Probar checkpoints
checkpoints = [
    ("30K", "outputs/rl_phi_text_scheduler/checkpoints/ppo_infinito_scheduler_30000_steps.zip"),
    ("50K", "outputs/rl_phi_text_scheduler/ppo_infinito_scheduler_final.zip"),
]

for name, path in checkpoints:
    print(f"\n{'='*70}")
    print(f"🎯 Modelo {name}: {path}")
    print(f"{'='*70}")
    
    try:
        # Cargar modelo
        print(f"📥 Cargando modelo {name}...")
        model = PPO.load(path, env=env)
        print(f"✅ Modelo {name} cargado exitosamente")
        
        # Hacer rollout de prueba
        print(f"\n🎮 Ejecutando episodio de prueba...")
        obs, info = env.reset()
        
        actions_log = []
        rewards_log = []
        phi_log = []
        
        for step in range(20):  # Solo 20 pasos de prueba
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            
            actions_log.append(int(action))
            rewards_log.append(float(reward))
            phi_log.append(info.get('phi', 0))
            
            if terminated or truncated:
                break
        
        # Estadísticas
        actions_count = {
            'TEXT': actions_log.count(0),
            'PHI': actions_log.count(1),
            'MIXED': actions_log.count(2)
        }
        
        print(f"\n📊 Resultados ({len(actions_log)} pasos):")
        print(f"  Acciones:")
        for action_name, count in actions_count.items():
            pct = (count / len(actions_log) * 100) if actions_log else 0
            print(f"    {action_name:5s}: {count:2d} ({pct:5.1f}%)")
        
        print(f"\n  Métricas:")
        print(f"    PHI promedio:   {np.mean(phi_log):.3f}")
        print(f"    Reward promedio: {np.mean(rewards_log):+.3f}")
        print(f"    Reward total:    {np.sum(rewards_log):+.3f}")
        
        # Verificar objetivos
        phi_in_range = np.mean([(3 <= p <= 6) for p in phi_log]) * 100
        mixed_used = (actions_count['MIXED'] / len(actions_log) * 100) if actions_log else 0
        
        print(f"\n  ✅ Verificación:")
        print(f"    Φ en [3-6]: {phi_in_range:5.1f}% {'✅' if phi_in_range > 50 else '⚠️'}")
        print(f"    Uso MIXED:  {mixed_used:5.1f}% {'✅' if mixed_used > 10 else '⚠️'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

env.close()
print("\n" + "="*70)
print("✅ TEST COMPLETADO")
print("="*70)
