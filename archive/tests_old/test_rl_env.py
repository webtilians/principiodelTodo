#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST RÁPIDO - Entorno RL de INFINITO
======================================

Script para verificar que el entorno RL funciona correctamente
sin necesidad de entrenar un agente completo.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl.infinito_rl_env import InfinitoRLEnv, GYMNASIUM_AVAILABLE


def test_environment():
    """Test básico del entorno RL."""
    
    print("="*70)
    print("TEST RÁPIDO - Entorno RL de INFINITO")
    print("="*70)
    print(f"Gymnasium available: {GYMNASIUM_AVAILABLE}")
    print("="*70)
    
    # Configuración mínima para test rápido
    config = {
        "inner_steps": 2,      # Muy pocos pasos para test
        "max_steps": 5,        # Episodio corto
        "batch_size": 2,       # Batch pequeño
        "model_kwargs": {
            "use_lora": True,
            "lora_r": 4,
            "lora_alpha": 16,
            "lambda_phi": 0.3,
            "freeze_base": True,
            "memory_slots": 64,  # Memoria reducida
        },
    }
    
    print("\n📦 Creando entorno...")
    env = InfinitoRLEnv(config=config)
    print("✅ Entorno creado correctamente")
    
    print("\n🔍 Espacios:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Test reset
    print("\n🔄 Testeando reset()...")
    result = env.reset()
    # Parsear resultado (gymnasium devuelve (obs, info))
    if isinstance(result, tuple):
        obs, info = result
    else:
        obs = result
    print(f"  Observación shape: {obs.shape}")
    print(f"  Observación: {obs}")
    print("✅ Reset exitoso")
    
    # Test steps
    print("\n🎮 Testeando step() con acciones aleatorias...")
    mode_names = {0: "TEXT", 1: "PHI", 2: "MIXED"}
    
    for i in range(3):
        action = env.action_space.sample()
        print(f"\n  Step {i+1}:")
        print(f"    Acción: {action} ({mode_names[action]})")
        
        result = env.step(action)
        
        # Parsear resultado (compatible con gym y gymnasium)
        if len(result) == 5:  # gymnasium
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # gym
            obs, reward, done, info = result
        
        print(f"    Observación shape: {obs.shape}")
        print(f"    Recompensa: {reward:.4f}")
        print(f"    Done: {done}")
        print(f"    Modo aplicado: {info['mode']}")
        
        metrics = info['latest_metrics']
        print(f"    Métricas:")
        print(f"      C: {metrics['consciousness']:.3f}")
        print(f"      Φ: {metrics['phi']:.3f}")
        print(f"      Loss text: {metrics['loss_text']:.4f}")
        print(f"      Loss phi: {metrics['loss_phi']:.4f}")
        
        if done:
            print("  Episodio terminado")
            break
    
    print("\n✅ Step exitoso")
    
    # Test close
    print("\n🧹 Testeando close()...")
    env.close()
    print("✅ Close exitoso")
    
    print("\n"+"="*70)
    print("✅ TODOS LOS TESTS PASARON")
    print("="*70)
    print("\nEl entorno RL está listo para entrenar un agente PPO.")
    print("Ejecutar: python experiments/train_phi_text_scheduler.py")


if __name__ == "__main__":
    try:
        test_environment()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
