#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monitor de progreso del entrenamiento RL continuo."""

import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def monitor_training():
    """Monitorea el progreso del entrenamiento."""
    
    eval_file = Path("outputs/rl_continued/eval_logs/evaluations.npz")
    checkpoints_dir = Path("outputs/rl_continued/checkpoints")
    
    print("=" * 70)
    print("  📊 MONITOR DE ENTRENAMIENTO RL CONTINUO")
    print("=" * 70)
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    last_checkpoint_count = 0
    last_eval_count = 0
    start_time = datetime.now()
    
    while True:
        try:
            # Verificar checkpoints
            if checkpoints_dir.exists():
                checkpoints = list(checkpoints_dir.glob("ppo_continued_*.zip"))
                if len(checkpoints) > last_checkpoint_count:
                    last_checkpoint_count = len(checkpoints)
                    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    steps = latest.stem.split('_')[-2]
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                          f"💾 Nuevo checkpoint guardado: {steps} steps")
            
            # Verificar evaluaciones
            if eval_file.exists():
                try:
                    data = np.load(eval_file)
                    timesteps = data['timesteps']
                    results = data['results']
                    
                    if len(timesteps) > last_eval_count:
                        # Nueva evaluación
                        idx = -1
                        ts = timesteps[idx]
                        rewards = results[idx]
                        mean_r = rewards.mean()
                        std_r = rewards.std()
                        
                        # Progreso
                        total_steps = 60_000  # 30K original + 30K nuevos
                        current_steps = ts
                        progress_pct = (current_steps / total_steps) * 100
                        
                        # Tiempo estimado
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if current_steps > 30_000:
                            steps_done = current_steps - 30_000
                            steps_remaining = 30_000 - steps_done
                            if steps_done > 0:
                                time_per_step = elapsed / steps_done
                                eta_seconds = time_per_step * steps_remaining
                                eta = timedelta(seconds=int(eta_seconds))
                            else:
                                eta = "Calculando..."
                        else:
                            eta = "Calculando..."
                        
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                              f"📈 Evaluación en {ts:,} steps")
                        print(f"   Reward: {mean_r:+.3f} ± {std_r:.3f}")
                        print(f"   Progreso: {progress_pct:.1f}%")
                        print(f"   ETA: {eta}")
                        
                        # Comparar con 30K
                        baseline_reward = 7.251
                        baseline_std = 0.040
                        
                        if mean_r > baseline_reward:
                            improvement = ((mean_r - baseline_reward) / baseline_reward) * 100
                            print(f"   ✅ Mejor que 30K (+{improvement:.2f}%)")
                        else:
                            decline = ((baseline_reward - mean_r) / baseline_reward) * 100
                            print(f"   ⚠️  Peor que 30K (-{decline:.2f}%)")
                        
                        # Verificar overfitting
                        variance_ratio = std_r / baseline_std
                        if variance_ratio > 10:
                            print(f"   🚨 OVERFITTING DETECTADO (varianza x{variance_ratio:.1f})")
                        elif variance_ratio > 3:
                            print(f"   ⚠️  Varianza aumentando (x{variance_ratio:.1f})")
                        else:
                            print(f"   ✅ Varianza controlada (x{variance_ratio:.1f})")
                        
                        last_eval_count = len(timesteps)
                
                except Exception as e:
                    pass
            
            # Esperar antes de siguiente check
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Monitor detenido por usuario")
            break
        except Exception as e:
            print(f"\n❌ Error en monitor: {e}")
            time.sleep(30)
    
    print("\n" + "=" * 70)
    print("  Monitor finalizado")
    print("=" * 70)


if __name__ == "__main__":
    monitor_training()
