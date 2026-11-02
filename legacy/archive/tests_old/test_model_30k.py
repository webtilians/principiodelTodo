#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de generación con modelo RL 30K (óptimo).
Versión simplificada para evitar errores de importación.
"""

import torch
import numpy as np
from stable_baselines3 import PPO
import json
import sys

print("="*70)
print("TEST GENERACION - MODELO RL 30K (OPTIMO)")
print("="*70)

# 1. Cargar configuración
print("\n[*] Cargando configuracion...")
try:
    with open("outputs/rl_phi_text_scheduler/env_config.json", 'r') as f:
        env_config = json.load(f)
    print("✅ Configuración cargada")
except Exception as e:
    print(f"❌ Error cargando config: {e}")
    sys.exit(1)

# 2. Importar entorno
print("\n📦 Importando entorno RL...")
try:
    from src.rl.infinito_rl_env import InfinitoRLEnv
    print("✅ Entorno importado")
except Exception as e:
    print(f"❌ Error importando entorno: {e}")
    sys.exit(1)

# 3. Crear entorno
print("\n🏗️ Creando entorno...")
try:
    env = InfinitoRLEnv(config=env_config)
    print("✅ Entorno creado")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
except Exception as e:
    print(f"❌ Error creando entorno: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Cargar modelo 30K
checkpoint_path = "outputs/rl_phi_text_scheduler/checkpoints/ppo_infinito_scheduler_30000_steps.zip"
print(f"\n🎯 Cargando modelo óptimo (30K)...")
print(f"   Ruta: {checkpoint_path}")

try:
    model = PPO.load(checkpoint_path, env=env)
    print("✅ Modelo 30K cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Ejecutar episodios de prueba
print("\n" + "="*70)
print("🎮 EJECUTANDO PRUEBAS DE GENERACIÓN")
print("="*70)

num_episodes = 3
max_steps_per_episode = 30

all_results = []

for episode in range(num_episodes):
    print(f"\n{'─'*70}")
    print(f"📝 Episodio {episode + 1}/{num_episodes}")
    print(f"{'─'*70}")
    
    # Reset
    obs, info = env.reset()
    
    # Tracking
    actions_log = []
    rewards_log = []
    phi_log = []
    consciousness_log = []
    ppl_log = []
    
    print(f"\nIniciando episodio (max {max_steps_per_episode} pasos)...\n")
    
    for step in range(max_steps_per_episode):
        # Predecir acción
        action, _states = model.predict(obs, deterministic=False)
        action_name = ['TEXT', 'PHI', 'MIXED'][action]
        
        # Ejecutar
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Guardar
        actions_log.append(int(action))
        rewards_log.append(float(reward))
        phi_log.append(info.get('phi', 0))
        consciousness_log.append(info.get('consciousness', 0))
        ppl_log.append(info.get('perplexity', 0))
        
        # Mostrar cada 5 pasos
        if (step + 1) % 5 == 0 or step == 0:
            phi = info.get('phi', 0)
            c = info.get('consciousness', 0)
            ppl = info.get('perplexity', 0)
            
            print(f"  Step {step+1:2d}: {action_name:5s} | "
                  f"Φ={phi:5.2f} | C={c:4.2f} | PPL={ppl:6.1f} | "
                  f"R={reward:+6.3f}")
        
        if terminated or truncated:
            print(f"\n  ✓ Episodio terminado en {step+1} pasos")
            break
    
    # Estadísticas del episodio
    actions_count = {
        'TEXT': actions_log.count(0),
        'PHI': actions_log.count(1),
        'MIXED': actions_log.count(2)
    }
    
    total_actions = len(actions_log)
    
    print(f"\n📊 Estadísticas del episodio:")
    print(f"  Total pasos: {total_actions}")
    
    print(f"\n  Acciones:")
    for action_name, count in actions_count.items():
        pct = (count / total_actions * 100) if total_actions > 0 else 0
        bar = '█' * int(pct / 3)  # Barra visual
        print(f"    {action_name:5s}: {count:2d} ({pct:5.1f}%) {bar}")
    
    phi_mean = np.mean(phi_log)
    phi_std = np.std(phi_log)
    phi_min = np.min(phi_log)
    phi_max = np.max(phi_log)
    
    print(f"\n  Métricas INFINITO:")
    print(f"    Φ: {phi_mean:5.2f} ± {phi_std:4.2f} [{phi_min:5.2f}, {phi_max:5.2f}]")
    print(f"    C: {np.mean(consciousness_log):5.2f} ± {np.std(consciousness_log):4.2f}")
    print(f"    PPL: {np.mean(ppl_log):6.1f} ± {np.std(ppl_log):5.1f}")
    
    reward_total = np.sum(rewards_log)
    reward_mean = np.mean(rewards_log)
    
    print(f"\n  Rewards:")
    print(f"    Total: {reward_total:+7.3f}")
    print(f"    Media: {reward_mean:+7.3f}")
    print(f"    Rango: [{np.min(rewards_log):+7.3f}, {np.max(rewards_log):+7.3f}]")
    
    # Verificaciones
    phi_in_range = np.mean([(3 <= p <= 6) for p in phi_log]) * 100
    ppl_safe = np.mean([p >= 10 for p in ppl_log]) * 100
    mixed_pct = (actions_count['MIXED'] / total_actions * 100) if total_actions > 0 else 0
    
    print(f"\n  ✅ Verificaciones:")
    print(f"    Φ en [3-6]:  {phi_in_range:5.1f}% {'✅' if phi_in_range > 70 else '⚠️'}")
    print(f"    PPL >= 10:   {ppl_safe:5.1f}% {'✅' if ppl_safe > 90 else '⚠️'}")
    print(f"    Uso MIXED:   {mixed_pct:5.1f}% {'✅' if mixed_pct > 10 else '⚠️'}")
    
    # Guardar resultados
    all_results.append({
        'episode': episode + 1,
        'steps': total_actions,
        'actions': actions_count,
        'phi_mean': phi_mean,
        'phi_in_range': phi_in_range,
        'ppl_mean': np.mean(ppl_log),
        'ppl_safe': ppl_safe,
        'mixed_pct': mixed_pct,
        'reward_total': reward_total,
        'reward_mean': reward_mean,
    })

# Resumen final
print("\n" + "="*70)
print("📊 RESUMEN FINAL - MODELO 30K")
print("="*70)

# Promedios
avg_phi = np.mean([r['phi_mean'] for r in all_results])
avg_phi_in_range = np.mean([r['phi_in_range'] for r in all_results])
avg_mixed = np.mean([r['mixed_pct'] for r in all_results])
avg_reward = np.mean([r['reward_mean'] for r in all_results])
avg_ppl_safe = np.mean([r['ppl_safe'] for r in all_results])

print(f"\n🎯 Promedios ({num_episodes} episodios):")
print(f"  Φ promedio:       {avg_phi:6.3f}")
print(f"  Φ en [3-6]:       {avg_phi_in_range:6.2f}%")
print(f"  PPL >= 10:        {avg_ppl_safe:6.2f}%")
print(f"  Uso MIXED:        {avg_mixed:6.2f}%")
print(f"  Reward promedio:  {avg_reward:+7.3f}")

# Distribución total de acciones
total_text = sum(r['actions']['TEXT'] for r in all_results)
total_phi = sum(r['actions']['PHI'] for r in all_results)
total_mixed = sum(r['actions']['MIXED'] for r in all_results)
total_all = total_text + total_phi + total_mixed

print(f"\n🎮 Distribución total de acciones:")
print(f"  TEXT:  {total_text:3d} ({total_text/total_all*100:5.1f}%)")
print(f"  PHI:   {total_phi:3d} ({total_phi/total_all*100:5.1f}%)")
print(f"  MIXED: {total_mixed:3d} ({total_mixed/total_all*100:5.1f}%)")

# Evaluación final
print(f"\n{'='*70}")
print("💡 EVALUACIÓN FINAL")
print(f"{'='*70}")

checks = []
checks.append(("Φ en rango óptimo [3-6]", avg_phi_in_range > 70, avg_phi_in_range))
checks.append(("PPL seguro (>= 10)", avg_ppl_safe > 90, avg_ppl_safe))
checks.append(("Usa modo MIXED", avg_mixed > 10, avg_mixed))
checks.append(("Reward positivo", avg_reward > 0, avg_reward))

passed = sum(1 for _, status, _ in checks if status)

print(f"\nChecks pasados: {passed}/{len(checks)}\n")

for name, status, value in checks:
    icon = "✅" if status else "⚠️"
    print(f"  {icon} {name}: {value:.2f}%")

if passed == len(checks):
    print(f"\n🎉 EXCELENTE: Modelo 30K funciona perfectamente!")
    print(f"   ✅ Listo para producción")
elif passed >= len(checks) - 1:
    print(f"\n✅ BUENO: Modelo 30K funciona bien")
    print(f"   ⚠️ Revisar el check que falló")
else:
    print(f"\n⚠️ ACEPTABLE: Modelo 30K necesita ajustes")
    print(f"   Revisar checks fallidos")

print(f"\n{'='*70}")
print("✅ TEST COMPLETADO")
print(f"{'='*70}")

# Cerrar entorno
env.close()

print("\nModelo 30K evaluado exitosamente.")
print(f"Resultados guardados en memoria: {len(all_results)} episodios")
