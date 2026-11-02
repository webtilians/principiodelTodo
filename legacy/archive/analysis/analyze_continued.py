#!/usr/bin/env python3
"""Analiza los resultados del entrenamiento continuo."""

import numpy as np
from pathlib import Path

eval_file = Path("outputs/rl_continued/eval_logs/evaluations.npz")

if not eval_file.exists():
    print("❌ No hay evaluaciones disponibles")
    exit(1)

data = np.load(eval_file)
timesteps = data['timesteps']
results = data['results']

print("=" * 70)
print("📊 ANÁLISIS - ENTRENAMIENTO CONTINUO DESDE 30K")
print("=" * 70)

# Baseline 30K
baseline_reward = 7.251
baseline_std = 0.040

print(f"\n📍 Baseline (30K original):")
print(f"   Reward: +{baseline_reward:.3f} ± {baseline_std:.3f}")

print(f"\n📈 Resultados del entrenamiento continuo:")
print(f"   Total evaluaciones: {len(timesteps)}")

for i, ts in enumerate(timesteps):
    rewards = results[i]
    mean_r = rewards.mean()
    std_r = rewards.std()
    
    # Comparar con baseline
    if mean_r > baseline_reward:
        improvement = ((mean_r - baseline_reward) / baseline_reward) * 100
        status = f"✅ +{improvement:.2f}%"
    else:
        decline = ((baseline_reward - mean_r) / baseline_reward) * 100
        status = f"⚠️  -{decline:.2f}%"
    
    # Verificar overfitting
    variance_ratio = std_r / baseline_std
    if variance_ratio > 10:
        overfitting = "🚨 OVERFITTING"
    elif variance_ratio > 3:
        overfitting = "⚠️  Varianza alta"
    else:
        overfitting = "✅ Estable"
    
    print(f"\n   {ts:,} steps:")
    print(f"      Reward: {mean_r:+.3f} ± {std_r:.3f}")
    print(f"      vs 30K: {status}")
    print(f"      Varianza: x{variance_ratio:.1f} {overfitting}")

# Mejor checkpoint
best_idx = np.argmax([r.mean() for r in results])
best_ts = timesteps[best_idx]
best_reward = results[best_idx].mean()
best_std = results[best_idx].std()

print("\n" + "=" * 70)
print("🏆 MEJOR CHECKPOINT")
print("=" * 70)
print(f"   Timesteps: {best_ts:,}")
print(f"   Reward: {best_reward:+.3f} ± {best_std:.3f}")

if best_reward > baseline_reward:
    improvement = ((best_reward - baseline_reward) / baseline_reward) * 100
    print(f"   Mejora: +{improvement:.2f}% vs 30K ✅")
    print(f"\n💡 RECOMENDACIÓN: Usar checkpoint {best_ts:,} steps")
else:
    print(f"   ⚠️  No superó el baseline 30K")
    print(f"\n💡 RECOMENDACIÓN: Mantener checkpoint 30K original")

print("=" * 70)
