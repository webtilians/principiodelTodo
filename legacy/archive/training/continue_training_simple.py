#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔄 CONTINUAR ENTRENAMIENTO RL - Versión Simplificada
====================================================

Versión simplificada que evita problemas de carga del modelo.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from src.rl.infinito_rl_env import InfinitoRLEnv


def continue_training_simple(
    checkpoint_path: str = "outputs/rl_phi_text_scheduler/checkpoints/ppo_infinito_scheduler_30000_steps.zip",
    additional_timesteps: int = 30_000,
    output_dir: str = "outputs/rl_continued",
    save_freq: int = 3_000,
    eval_freq: int = 3_000,
):
    """Versión simplificada del entrenamiento continuo."""
    
    print("=" * 70)
    print("  🔄 ENTRENAMIENTO RL CONTINUO - VERSIÓN SIMPLIFICADA")
    print("=" * 70)
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Pasos adicionales: {additional_timesteps:,}")
    print("=" * 70)
    
    # Crear directorios
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    
    # Cargar configuración
    config_path = "outputs/rl_phi_text_scheduler/env_config.json"
    with open(config_path, 'r') as f:
        env_config = json.load(f)
    
    print("\n📦 Cargando modelo PPO...")
    model = PPO.load(checkpoint_path, device='cuda')
    print("✅ Modelo cargado")
    
    # Crear entornos
    print("\n� Creando entornos...")
    train_env = InfinitoRLEnv(config=env_config)
    eval_env = InfinitoRLEnv(config=env_config)
    print("✅ Entornos creados")
    
    # Establecer entorno
    model.set_env(train_env)
    
    print("\n🛡️  Nota: Continuando con parámetros del checkpoint 30K")
    print(f"   Configuración original preservada")
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=f"{output_dir}/checkpoints",
        name_prefix="ppo_continued",
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{output_dir}/best_model",
        log_path=f"{output_dir}/eval_logs",
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    callbacks = CallbackList([checkpoint_cb, eval_cb])
    
    # Entrenar
    print("\n" + "=" * 70)
    print("  🚀 INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    
    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False,  # Continuar desde donde quedó
        )
        
        # Guardar final
        final_path = f"{output_dir}/ppo_final.zip"
        model.save(final_path)
        print(f"\n✅ Modelo final: {final_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        train_env.close()
        eval_env.close()
        print("\n✅ Recursos liberados")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                       default="outputs/rl_phi_text_scheduler/checkpoints/ppo_infinito_scheduler_30000_steps.zip")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--output", type=str, default="outputs/rl_continued")
    parser.add_argument("--save-freq", type=int, default=3_000)
    parser.add_argument("--eval-freq", type=int, default=3_000)
    
    args = parser.parse_args()
    
    continue_training_simple(
        checkpoint_path=args.checkpoint,
        additional_timesteps=args.timesteps,
        output_dir=args.output,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
    )
