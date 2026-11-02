#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 ANÁLISIS DE RESULTADOS - Entrenamiento RL INFINITO
"""

import json
import numpy as np
from pathlib import Path

print("="*70)
print("📊 ANÁLISIS RESULTADOS - AGENTE RL INFINITO")
print("="*70)

# Cargar training stats
stats_path = Path("outputs/rl_phi_text_scheduler/training_stats.json")
with open(stats_path) as f:
    stats = json.load(f)

print("\n🕒 DURACIÓN:")
duration_hours = stats["duration_seconds"] / 3600
print(f"  Total timesteps: {stats['total_timesteps']:,}")
print(f"  Duración: {duration_hours:.2f} horas ({stats['duration_seconds']/60:.1f} min)")
print(f"  Inicio: {stats['start_time']}")
print(f"  Fin: {stats['end_time']}")

print("\n⚙️ CONFIGURACIÓN:")
env_cfg = stats["env_config"]
print(f"  Inner steps: {env_cfg['inner_steps']}")
print(f"  Max steps per episode: {env_cfg['max_steps']}")
print(f"  Batch size: {env_cfg['batch_size']}")

print("\n🧠 MODELO:")
model_cfg = env_cfg["model_kwargs"]
print(f"  LoRA r: {model_cfg['lora_r']}")
print(f"  LoRA alpha: {model_cfg['lora_alpha']}")
print(f"  Lambda phi: {model_cfg['lambda_phi']}")
print(f"  Memory slots: {model_cfg['memory_slots']}")

print("\n🎯 RECOMPENSAS:")
rewards = env_cfg["reward_weights"]
print(f"  α (Consciousness): {rewards['alpha']}")
print(f"  β (PHI): {rewards['beta']}")
print(f"  γ (Perplexity): {rewards['gamma']}")
print(f"  δ (Cost): {rewards['delta']}")

print("\n🤖 PPO CONFIG:")
ppo_cfg = stats["ppo_config"]
print(f"  Learning rate: {ppo_cfg['learning_rate']}")
print(f"  Steps per rollout: {ppo_cfg['n_steps']}")
print(f"  Batch size: {ppo_cfg['batch_size']}")
print(f"  Epochs per update: {ppo_cfg['n_epochs']}")
print(f"  Gamma (discount): {ppo_cfg['gamma']}")

# Cargar evaluaciones
eval_path = Path("outputs/rl_phi_text_scheduler/eval_logs/evaluations.npz")
if eval_path.exists():
    print("\n📈 RESULTADOS DE EVALUACIÓN:")
    with np.load(eval_path) as data:
        timesteps = data['timesteps']
        results = data['results']
        ep_lengths = data['ep_lengths']
        
        print(f"\n  Evaluaciones realizadas: {len(timesteps)}")
        print(f"  Timesteps evaluados: {timesteps.tolist()}")
        
        print("\n  📊 RECOMPENSAS POR EVALUACIÓN:")
        for i, (ts, rewards_eval) in enumerate(zip(timesteps, results)):
            mean_reward = rewards_eval.mean()
            std_reward = rewards_eval.std()
            min_reward = rewards_eval.min()
            max_reward = rewards_eval.max()
            
            print(f"\n  Eval {i+1} (timestep {ts}):")
            print(f"    Mean reward: {mean_reward:.4f} ± {std_reward:.4f}")
            print(f"    Range: [{min_reward:.4f}, {max_reward:.4f}]")
            print(f"    Episodios: {len(rewards_eval)}")
        
        # Comparación entre evaluaciones
        if len(timesteps) > 1:
            print("\n  📈 MEJORA:")
            initial_mean = results[0].mean()
            final_mean = results[-1].mean()
            improvement = final_mean - initial_mean
            improvement_pct = (improvement / abs(initial_mean)) * 100 if initial_mean != 0 else 0
            
            print(f"    Recompensa inicial: {initial_mean:.4f}")
            print(f"    Recompensa final: {final_mean:.4f}")
            print(f"    Mejora absoluta: {improvement:.4f}")
            print(f"    Mejora relativa: {improvement_pct:+.1f}%")
            
            if improvement > 0:
                print(f"    ✅ El agente MEJORÓ durante el entrenamiento")
            elif improvement < 0:
                print(f"    ⚠️ El agente EMPEORÓ (posible overfitting o inestabilidad)")
            else:
                print(f"    ➡️ Sin cambio significativo")

# Verificar mejor modelo
best_model_path = Path("outputs/rl_phi_text_scheduler/best_model/best_model.zip")
if best_model_path.exists():
    size_mb = best_model_path.stat().st_size / (1024**2)
    print(f"\n💾 MEJOR MODELO:")
    print(f"    Guardado en: {best_model_path}")
    print(f"    Tamaño: {size_mb:.2f} MB")

# Checkpoints
checkpoint_dir = Path("outputs/rl_phi_text_scheduler/checkpoints")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.zip"))
    print(f"\n📦 CHECKPOINTS:")
    print(f"    Total: {len(checkpoints)}")
    if checkpoints:
        for ckpt in sorted(checkpoints)[:5]:  # Primeros 5
            size_mb = ckpt.stat().st_size / (1024**2)
            print(f"    - {ckpt.name} ({size_mb:.2f} MB)")

print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETADO")
print("="*70)
print("\n💡 PRÓXIMOS PASOS:")
print("  1. Ver logs de TensorBoard:")
print("     tensorboard --logdir outputs/rl_phi_text_scheduler/tensorboard")
print("\n  2. Ejecutar demo con agente entrenado:")
print("     python experiments/run_infinito_with_scheduler.py --episodes 3")
print("\n  3. Entrenar por más tiempo si hay mejora:")
print("     python experiments/train_phi_text_scheduler.py --timesteps 50000")
