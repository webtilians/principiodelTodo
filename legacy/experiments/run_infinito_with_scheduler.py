#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 DEMOSTRACIÓN - INFINITO con Scheduler RL
==========================================

Ejecuta INFINITO controlado por un agente RL entrenado que decide
dinámicamente entre modo TEXTO, PHI o MIXTO.

Muestra:
- Acciones del agente en cada paso
- Evolución de métricas (C, Φ, perplexity)
- Balance adaptativo según el estado del sistema
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para guardar plots
import matplotlib.pyplot as plt
from datetime import datetime

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️  stable-baselines3 no disponible")
    SB3_AVAILABLE = False
    PPO = None

from src.rl.infinito_rl_env import InfinitoRLEnv


def run_infinito_with_scheduler(
    model_path: str,
    n_episodes: int = 3,
    max_steps: int = 100,
    render: bool = False,
    save_plots: bool = True,
    output_dir: str = "outputs/rl_demo"
):
    """
    Ejecuta INFINITO con un agente RL entrenado.
    
    Args:
        model_path: Path al modelo PPO entrenado
        n_episodes: Número de episodios a ejecutar
        max_steps: Pasos máximos por episodio
        render: Si renderizar (no implementado)
        save_plots: Si guardar gráficos
        output_dir: Directorio de salida
    """
    
    if not SB3_AVAILABLE:
        print("❌ Error: stable-baselines3 no disponible")
        return
    
    print("="*70)
    print("DEMOSTRACIÓN - INFINITO CON SCHEDULER RL")
    print("="*70)
    print(f"  Modelo: {model_path}")
    print(f"  Episodios: {n_episodes}")
    print(f"  Max steps: {max_steps}")
    print("="*70)
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar modelo RL
    print("\n📦 Cargando modelo PPO...")
    try:
        rl_model = PPO.load(model_path)
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Configurar entorno
    env_config = {
        "inner_steps": 5,
        "max_steps": max_steps,
        "batch_size": 4,
        "model_kwargs": {
            "use_lora": True,
            "lora_r": 4,
            "lora_alpha": 16,
            "lambda_phi": 0.3,
            "freeze_base": True,
            "memory_slots": 128,
        },
        "reward_weights": {
            "alpha": 1.0,
            "beta": 0.5,
            "gamma": 0.1,
            "delta": 0.2,
        },
    }
    
    print("\n📦 Creando entorno...")
    env = InfinitoRLEnv(config=env_config)
    print("✅ Entorno creado")
    
    # Modo names
    mode_names = {0: "TEXT", 1: "PHI", 2: "MIXED"}
    
    # Ejecutar episodios
    for episode in range(n_episodes):
        print("\n" + "="*70)
        print(f"EPISODIO {episode + 1}/{n_episodes}")
        print("="*70)
        
        # Historiales
        obs_history = []
        action_history = []
        reward_history = []
        metrics_history = []
        
        # Reset
        result = env.reset()
        # Parsear resultado (gymnasium devuelve (obs, info))
        if isinstance(result, tuple):
            obs, reset_info = result
        else:
            obs = result
        
        done = False
        step = 0
        total_reward = 0.0
        
        print(f"\n🔄 Observación inicial: {obs}")
        
        while not done and step < max_steps:
            # Predecir acción con agente RL
            action, _states = rl_model.predict(obs, deterministic=True)
            
            # Ejecutar acción
            result = env.step(int(action))
            
            # Parsear resultado (compatible con gym y gymnasium)
            if len(result) == 5:  # gymnasium
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:  # gym
                obs, reward, done, info = result
            
            # Guardar datos
            obs_history.append(obs.copy())
            action_history.append(int(action))
            reward_history.append(float(reward))
            metrics_history.append(info['latest_metrics'].copy())
            
            total_reward += reward
            step += 1
            
            # Log cada 10 pasos
            if step % 10 == 0 or done:
                mode = mode_names[int(action)]
                metrics = info['latest_metrics']
                print(f"\n  Paso {step}/{max_steps}:")
                print(f"    Acción: {action} ({mode})")
                print(f"    Recompensa: {reward:+.4f} (acum: {total_reward:+.4f})")
                print(f"    C: {metrics['consciousness']:.3f} | Φ: {metrics['phi']:.3f} | PPL: {metrics['perplexity']:.2f}")
                print(f"    Loss_text: {metrics['loss_text']:.4f} | Loss_phi: {metrics['loss_phi']:.4f}")
        
        print(f"\n✅ Episodio completado")
        print(f"  Pasos totales: {step}")
        print(f"  Recompensa total: {total_reward:+.4f}")
        print(f"  Recompensa promedio: {total_reward/step:+.4f}")
        
        # Estadísticas de acciones
        action_counts = {0: 0, 1: 0, 2: 0}
        for a in action_history:
            action_counts[a] += 1
        
        print(f"\n📊 Distribución de acciones:")
        for act, count in action_counts.items():
            pct = 100.0 * count / len(action_history) if action_history else 0
            print(f"    {mode_names[act]:<6}: {count:3d} ({pct:5.1f}%)")
        
        # Guardar gráficos
        if save_plots:
            plot_path = f"{output_dir}/episode_{episode+1}_metrics.png"
            plot_episode_metrics(
                obs_history,
                action_history,
                reward_history,
                metrics_history,
                save_path=plot_path
            )
            print(f"\n💾 Gráficos guardados: {plot_path}")
    
    # Cerrar entorno
    env.close()
    print("\n✅ Demostración completada")


def plot_episode_metrics(obs_history, action_history, reward_history, 
                         metrics_history, save_path: str):
    """
    Genera gráficos de métricas de un episodio.
    
    Args:
        obs_history: Lista de observaciones
        action_history: Lista de acciones
        reward_history: Lista de recompensas
        metrics_history: Lista de métricas
        save_path: Path para guardar el plot
    """
    steps = np.arange(len(obs_history))
    
    # Extraer datos
    consciousness = [m['consciousness'] for m in metrics_history]
    phi = [m['phi'] for m in metrics_history]
    loss_text = [m['loss_text'] for m in metrics_history]
    loss_phi = [m['loss_phi'] for m in metrics_history]
    perplexity = [m['perplexity'] for m in metrics_history]
    
    # Crear figura
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('INFINITO RL Scheduler - Métricas del Episodio', fontsize=14, fontweight='bold')
    
    # 1. Consciousness & PHI
    ax = axes[0, 0]
    ax.plot(steps, consciousness, 'b-', linewidth=2, label='Consciousness (C)')
    ax.plot(steps, [p/10.0 for p in phi], 'r--', linewidth=2, label='PHI (Φ/10)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Value')
    ax.set_title('Consciousness & PHI')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Losses
    ax = axes[0, 1]
    ax.plot(steps, loss_text, 'g-', linewidth=2, label='Loss Text')
    ax.plot(steps, loss_phi, 'm--', linewidth=2, label='Loss PHI')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Perplexity
    ax = axes[1, 0]
    ax.plot(steps, perplexity, 'c-', linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Perplexity')
    ax.set_title('Perplexity')
    ax.grid(True, alpha=0.3)
    
    # 4. Rewards
    ax = axes[1, 1]
    ax.plot(steps, reward_history, 'orange', linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.set_title('Recompensa por Paso')
    ax.grid(True, alpha=0.3)
    
    # 5. Acciones (histogram)
    ax = axes[2, 0]
    action_counts = {0: 0, 1: 0, 2: 0}
    for a in action_history:
        action_counts[a] += 1
    
    modes = ['TEXT', 'PHI', 'MIXED']
    counts = [action_counts[0], action_counts[1], action_counts[2]]
    colors = ['green', 'red', 'blue']
    
    ax.bar(modes, counts, color=colors, alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Distribución de Acciones')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Acciones timeline
    ax = axes[2, 1]
    action_colors = ['green' if a == 0 else 'red' if a == 1 else 'blue' for a in action_history]
    ax.scatter(steps, action_history, c=action_colors, alpha=0.6, s=50)
    ax.set_xlabel('Step')
    ax.set_ylabel('Action')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['TEXT', 'PHI', 'MIXED'])
    ax.set_title('Secuencia de Acciones')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Demostración de INFINITO con scheduler RL"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='outputs/rl_phi_text_scheduler/ppo_infinito_scheduler_final.zip',
        help='Path al modelo PPO entrenado'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=3,
        help='Número de episodios a ejecutar (default: 3)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100,
        help='Pasos máximos por episodio (default: 100)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='No guardar gráficos'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/rl_demo',
        help='Directorio de salida para gráficos'
    )
    
    args = parser.parse_args()
    
    run_infinito_with_scheduler(
        model_path=args.model,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        save_plots=not args.no_plots,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
