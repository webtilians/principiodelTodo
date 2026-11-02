#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎓 ENTRENAMIENTO AGENTE RL - Scheduler Φ vs Texto
================================================

Entrena un agente PPO que aprende a decidir dinámicamente entre
modo TEXTO, PHI o MIXTO para INFINITO.

Objetivo:
---------
El agente maximiza una recompensa compuesta:
  r = α·ΔC + β·ΔΦ + γ·Δperplexity - δ·cost

Aprendiendo cuándo optimizar cada componente según el estado del sistema.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from datetime import datetime
import json

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️  stable-baselines3 no disponible")
    print("   Instalar con: pip install stable-baselines3")
    SB3_AVAILABLE = False
    sys.exit(1)

from src.rl.infinito_rl_env import InfinitoRLEnv
from src.rl.rich_metrics_callback import RichMetricsCallback


def train_ppo_scheduler(
    total_timesteps: int = 100_000,
    inner_steps: int = 5,
    max_steps: int = 100,
    batch_size: int = 4,
    learning_rate: float = 3e-4,
    output_dir: str = "outputs/rl_phi_text_scheduler",
    tensorboard_log: str = None,
    save_freq: int = 10_000,
    eval_freq: int = 5_000,
    n_eval_episodes: int = 5,
):
    """
    Entrena un agente PPO para controlar INFINITO.
    
    Args:
        total_timesteps: Timesteps totales de entrenamiento
        inner_steps: Pasos de INFINITO por step RL
        max_steps: Pasos máximos por episodio RL
        batch_size: Batch size para INFINITO
        learning_rate: Learning rate de PPO
        output_dir: Directorio de salida
        tensorboard_log: Directorio logs TensorBoard
        save_freq: Frecuencia de checkpoints
        eval_freq: Frecuencia de evaluación
        n_eval_episodes: Episodios de evaluación
    """
    
    print("="*70)
    print("ENTRENAMIENTO AGENTE PPO - SCHEDULER Φ vs TEXTO")
    print("="*70)
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Inner steps: {inner_steps}")
    print(f"  Max steps per episode: {max_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Output dir: {output_dir}")
    print("="*70)
    
    # Crear directorios
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    
    if tensorboard_log is None:
        tensorboard_log = f"{output_dir}/tensorboard"
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # Configuración del entorno
    env_config = {
        "inner_steps": inner_steps,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "model_kwargs": {
            "use_lora": True,
            "lora_r": 4,  # Reducido para RL (más rápido)
            "lora_alpha": 16,
            "lambda_phi": 0.3,
            "freeze_base": True,
            "memory_slots": 128,  # Reducido para RL
        },
        "reward_weights": {
            "alpha": 1.0,   # ΔC
            "beta": 0.5,    # ΔΦ
            "gamma": 0.1,   # Δperplexity
            "delta": 0.2,   # coste
        },
    }
    
    # Guardar config
    config_path = f"{output_dir}/env_config.json"
    with open(config_path, 'w') as f:
        json.dump(env_config, f, indent=2)
    print(f"\n✅ Configuración guardada en: {config_path}")
    
    # Crear entorno de entrenamiento
    print("\n📦 Creando entorno de entrenamiento...")
    train_env = InfinitoRLEnv(config=env_config)
    
    # Verificar entorno
    print("\n🔍 Verificando entorno con Gym checker...")
    try:
        check_env(train_env, warn=True)
        print("✅ Entorno verificado correctamente")
    except Exception as e:
        print(f"⚠️  Warning en verificación: {e}")
    
    # Crear entorno de evaluación
    print("\n📦 Creando entorno de evaluación...")
    eval_env = InfinitoRLEnv(config=env_config)
    
    # Configurar callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"{output_dir}/checkpoints",
        name_prefix="ppo_infinito_scheduler",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/best_model",
        log_path=f"{output_dir}/eval_logs",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    
    # Callback de métricas en tiempo real
    rich_metrics_callback = RichMetricsCallback(
        total_timesteps=total_timesteps,
        log_freq=500,  # Mostrar cada 500 steps
        verbose=1
    )
    
    # Combinar callbacks
    callback_list = CallbackList([checkpoint_callback, eval_callback, rich_metrics_callback])
    
    # Crear modelo PPO
    print("\n🤖 Creando modelo PPO...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=2048,           # Pasos antes de update
        batch_size=64,          # Batch size para PPO
        n_epochs=10,            # Epochs por update
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,        # GAE parameter
        clip_range=0.2,         # PPO clip range
        clip_range_vf=None,     # Value function clip
        ent_coef=0.01,          # Entropy coefficient
        vf_coef=0.5,            # Value function coefficient
        max_grad_norm=0.5,      # Gradient clipping
        use_sde=False,          # State Dependent Exploration
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # Actor-Critic architecture
        ),
        verbose=0,  # Reducido para no interferir con RichMetricsCallback
        seed=42,
        device='auto',
    )
    
    print("\n📊 Arquitectura PPO:")
    print(f"  Policy: MlpPolicy")
    print(f"  Hidden layers: [128, 128] (actor), [128, 128] (critic)")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Discount (γ): 0.99")
    print(f"  GAE (λ): 0.95")
    print(f"  PPO clip: 0.2")
    
    # Entrenar
    print("\n"+"="*70)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("="*70)
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=10,
            progress_bar=False,  # Desactivado para usar nuestro callback
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n"+"="*70)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*70)
        print(f"  Duración: {duration}")
        print(f"  Timesteps: {total_timesteps:,}")
        
        # Guardar modelo final
        final_path = f"{output_dir}/ppo_infinito_scheduler_final"
        model.save(final_path)
        print(f"  Modelo final guardado: {final_path}")
        
        # Guardar estadísticas
        stats = {
            "total_timesteps": total_timesteps,
            "duration_seconds": duration.total_seconds(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "env_config": env_config,
            "ppo_config": {
                "learning_rate": learning_rate,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
            }
        }
        
        stats_path = f"{output_dir}/training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"  Estadísticas guardadas: {stats_path}")
        
        print("\n📊 Ver logs con TensorBoard:")
        print(f"  tensorboard --logdir {tensorboard_log}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por usuario")
        model.save(f"{output_dir}/ppo_infinito_scheduler_interrupted")
        print(f"  Modelo guardado: {output_dir}/ppo_infinito_scheduler_interrupted")
    
    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Limpiar
        train_env.close()
        eval_env.close()
        print("\n✅ Recursos liberados")


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar agente PPO para scheduler Φ vs Texto en INFINITO"
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=100_000,
        help='Total timesteps de entrenamiento (default: 100,000)'
    )
    
    parser.add_argument(
        '--inner-steps',
        type=int,
        default=5,
        help='Pasos de INFINITO por step RL (default: 5)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Pasos máximos por episodio RL (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size para INFINITO (default: 4)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate de PPO (default: 3e-4)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/rl_phi_text_scheduler',
        help='Directorio de salida'
    )
    
    parser.add_argument(
        '--save-freq',
        type=int,
        default=10_000,
        help='Frecuencia de checkpoints (default: 10,000)'
    )
    
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=5_000,
        help='Frecuencia de evaluación (default: 5,000)'
    )
    
    args = parser.parse_args()
    
    # Entrenar
    train_ppo_scheduler(
        total_timesteps=args.timesteps,
        inner_steps=args.inner_steps,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
    )


if __name__ == "__main__":
    if not SB3_AVAILABLE:
        print("❌ Error: stable-baselines3 no disponible")
        print("   Instalar con: pip install stable-baselines3")
        sys.exit(1)
    
    main()
