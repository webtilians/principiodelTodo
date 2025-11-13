#!/usr/bin/env python3
"""Análisis detallado de los resultados del entrenamiento RL v2."""

import numpy as np
import json
from pathlib import Path

print("="*70)
print("📊 ANÁLISIS DETALLADO - ENTRENAMIENTO RL V2")
print("="*70)

# Cargar evaluaciones
eval_file = 'outputs/rl_phi_text_scheduler/eval_logs/evaluations.npz'
data = np.load(eval_file)

timesteps = data['timesteps']
results = data['results']  # Shape: (n_evals, n_episodes)
ep_lengths = data['ep_lengths']

print(f"\n📈 Datos cargados:")
print(f"  Evaluaciones: {len(timesteps)}")
print(f"  Episodios por eval: {results.shape[1]}")
print(f"  Timesteps: {timesteps.tolist()}")

# Análisis detallado por checkpoint
print(f"\n{'='*70}")
print("📊 ANÁLISIS POR CHECKPOINT")
print(f"{'='*70}")

best_checkpoint = None
best_reward = -float('inf')

for i, ts in enumerate(timesteps):
    rewards = results[i]  # Rewards de todos los episodios en esta eval
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    min_reward = rewards.min()
    max_reward = rewards.max()
    
    # Tracking del mejor
    if mean_reward > best_reward:
        best_reward = mean_reward
        best_checkpoint = ts
    
    status = "🏆" if ts == best_checkpoint else ("✅" if mean_reward > 0 else "⚠️")
    
    print(f"\n{status} Checkpoint {ts:>6,} steps:")
    print(f"  Reward: {mean_reward:+7.3f} ± {std_reward:5.3f}")
    print(f"  Rango:  [{min_reward:+7.3f}, {max_reward:+7.3f}]")
    print(f"  Episodios individuales: {[f'{r:+.2f}' for r in rewards]}")
    
    # Comparar con anterior
    if i > 0:
        prev_mean = results[i-1].mean()
        change = mean_reward - prev_mean
        change_pct = (change / abs(prev_mean) * 100) if prev_mean != 0 else 0
        trend = "↗️" if change > 0 else "↘️"
        print(f"  vs anterior: {change:+.3f} ({change_pct:+.1f}%) {trend}")

print(f"\n{'='*70}")
print(f"🏆 MEJOR CHECKPOINT: {best_checkpoint:,} steps")
print(f"   Reward: {best_reward:+.3f}")
print(f"{'='*70}")

# Análisis de estabilidad
print(f"\n📉 ANÁLISIS DE ESTABILIDAD:")

# Varianza entre evaluaciones
eval_means = results.mean(axis=1)
eval_stds = results.std(axis=1)

print(f"  Varianza entre checkpoints:")
print(f"    Mean de means:  {eval_means.mean():+.3f}")
print(f"    Std de means:   {eval_means.std():.3f}")
print(f"    Range:          [{eval_means.min():+.3f}, {eval_means.max():+.3f}]")

print(f"\n  Varianza intra-checkpoint (promedio):")
print(f"    Mean de stds:   {eval_stds.mean():.3f}")
print(f"    Episodios más estables: {timesteps[eval_stds.argmin()]:,} steps (std={eval_stds.min():.3f})")
print(f"    Episodios menos estables: {timesteps[eval_stds.argmax()]:,} steps (std={eval_stds.max():.3f})")

# Tendencias
print(f"\n📈 TENDENCIAS:")

# Primera mitad vs segunda mitad
first_half = eval_means[:len(eval_means)//2]
second_half = eval_means[len(eval_means)//2:]

print(f"  Primera mitad (0-{timesteps[len(timesteps)//2-1]:,}):")
print(f"    Mean: {first_half.mean():+.3f}")
print(f"    Best: {first_half.max():+.3f}")

print(f"  Segunda mitad ({timesteps[len(timesteps)//2]:,}-{timesteps[-1]:,}):")
print(f"    Mean: {second_half.mean():+.3f}")
print(f"    Best: {second_half.max():+.3f}")

improvement = ((second_half.mean() - first_half.mean()) / abs(first_half.mean()) * 100)
print(f"  Mejora: {improvement:+.1f}%")

# Checkpoints recomendados
print(f"\n{'='*70}")
print("🎯 RECOMENDACIONES")
print(f"{'='*70}")

# Top 3 checkpoints
top_3_indices = np.argsort(eval_means)[-3:][::-1]

print(f"\n📌 Top 3 checkpoints por reward promedio:")
for rank, idx in enumerate(top_3_indices, 1):
    ts = timesteps[idx]
    mean_rew = eval_means[idx]
    std_rew = eval_stds[idx]
    print(f"  {rank}. {ts:>6,} steps: {mean_rew:+.3f} ± {std_rew:.3f}")

# Checkpoint más estable entre los buenos
good_checkpoints = eval_means > 5.0  # Reward > 5
if good_checkpoints.any():
    good_indices = np.where(good_checkpoints)[0]
    stds_good = eval_stds[good_indices]
    most_stable_idx = good_indices[stds_good.argmin()]
    
    print(f"\n📍 Checkpoint más estable (reward > 5.0):")
    print(f"   {timesteps[most_stable_idx]:,} steps:")
    print(f"   Reward: {eval_means[most_stable_idx]:+.3f} ± {eval_stds[most_stable_idx]:.3f}")

# Conclusión
print(f"\n{'='*70}")
print("💡 CONCLUSIONES")
print(f"{'='*70}")

if best_reward > 7.0:
    status = "✅ EXCELENTE"
elif best_reward > 5.0:
    status = "✅ BUENO"
elif best_reward > 0:
    status = "⚠️ ACEPTABLE"
else:
    status = "❌ NECESITA MEJORA"

print(f"\nEstado general: {status}")
print(f"Mejor reward alcanzado: {best_reward:+.3f} en {best_checkpoint:,} steps")

if best_checkpoint < timesteps[-1]:
    print(f"\n⚠️ ATENCIÓN: El mejor checkpoint no es el final")
    print(f"   Posible overfitting después de {best_checkpoint:,} steps")
    print(f"   Recomendación: Usar checkpoint {best_checkpoint:,} para producción")
else:
    print(f"\n✅ El entrenamiento mejoró hasta el final")
    print(f"   Usar checkpoint final ({timesteps[-1]:,} steps)")

# Análisis de variabilidad
if eval_means.std() > 3.0:
    print(f"\n⚠️ Alta variabilidad entre checkpoints (std={eval_means.std():.2f})")
    print(f"   Considerar: Mayor batch size, más inner steps, o ajustar reward weights")
elif eval_means.std() < 1.0:
    print(f"\n✅ Baja variabilidad - entrenamiento estable (std={eval_means.std():.2f})")

print(f"\n{'='*70}")
