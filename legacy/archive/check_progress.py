#!/usr/bin/env python3
"""Script rápido para revisar progreso del entrenamiento RL."""

import numpy as np
import os
from datetime import datetime

eval_file = 'outputs/rl_phi_text_scheduler/eval_logs/evaluations.npz'

if os.path.exists(eval_file):
    print("="*70)
    print("📊 PROGRESO ENTRENAMIENTO RL v2")
    print("="*70)
    
    data = np.load(eval_file)
    timesteps = data['timesteps']
    results = data['results']
    ep_lengths = data['ep_lengths']
    
    print(f"\n🎯 Evaluaciones realizadas: {len(timesteps)}")
    print(f"📈 Timesteps evaluados: {timesteps.tolist()}")
    
    print(f"\n💰 REWARDS PROMEDIO:")
    for i, (ts, reward) in enumerate(zip(timesteps, results.mean(axis=1))):
        improvement = ((reward - results.mean(axis=1)[0]) / abs(results.mean(axis=1)[0]) * 100) if i > 0 else 0
        status = "✅" if reward > results.mean(axis=1)[max(0, i-1)] else "⚠️"
        print(f"  {status} {ts:>6,} steps: {reward:>8.4f}  (mejora: {improvement:>+6.1f}%)")
    
    print(f"\n📏 LONGITUD PROMEDIO EPISODIOS:")
    for ts, length in zip(timesteps, ep_lengths.mean(axis=1)):
        print(f"    {ts:>6,} steps: {length:.1f} pasos")
    
    # Calcular tendencia
    if len(results) >= 2:
        first_reward = results.mean(axis=1)[0]
        last_reward = results.mean(axis=1)[-1]
        total_improvement = ((last_reward - first_reward) / abs(first_reward) * 100)
        
        print(f"\n📊 RESUMEN:")
        print(f"  Reward inicial: {first_reward:.4f}")
        print(f"  Reward actual:  {last_reward:.4f}")
        print(f"  Mejora total:   {total_improvement:+.1f}%")
        
        if last_reward > -0.05:
            print(f"  Estado: ✅ EXCELENTE (cerca de reward positivo)")
        elif last_reward > first_reward:
            print(f"  Estado: ✅ MEJORANDO")
        else:
            print(f"  Estado: ⚠️ Necesita más tiempo")
    
    print("\n" + "="*70)
    
else:
    print("❌ No se encontró archivo de evaluaciones")
    print(f"   Ruta esperada: {eval_file}")
