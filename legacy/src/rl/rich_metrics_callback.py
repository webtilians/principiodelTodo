#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 CALLBACK PERSONALIZADO - Métricas en Tiempo Real
==================================================

Callback para mostrar métricas detalladas durante entrenamiento RL.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime, timedelta


class RichMetricsCallback(BaseCallback):
    """
    Callback que muestra métricas ricas durante el entrenamiento.
    
    Muestra:
    - Rewards (media, min, max, std)
    - Métricas de INFINITO (C, Φ, PPL)
    - Distribución de acciones
    - Tiempo estimado restante
    - Progreso visual
    """
    
    def __init__(self, total_timesteps: int, log_freq: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_freq = log_freq
        
        # Historiales
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_phi = []
        self.episode_consciousness = []
        self.episode_ppl = []
        self.action_counts = {0: 0, 1: 0, 2: 0}  # TEXT, PHI, MIXED
        
        # Tiempo
        self.start_time = None
        self.last_log_time = None
        
        # Display
        self.action_names = {0: "TEXT", 1: "PHI", 2: "MIXED"}
    
    def _on_training_start(self) -> None:
        """Llamado al inicio del entrenamiento."""
        self.start_time = datetime.now()
        self.last_log_time = self.start_time
        
        print("\n" + "="*80)
        print("🚀 ENTRENAMIENTO INICIADO")
        print("="*80)
        print(f"Inicio: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total timesteps: {self.total_timesteps:,}")
        print("="*80 + "\n")
    
    def _on_step(self) -> bool:
        """Llamado en cada step del entrenamiento."""
        
        # Registrar acción
        if len(self.locals.get('actions', [])) > 0:
            action = int(self.locals['actions'][0])
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
        
        # Log cada log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            self._log_metrics()
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Llamado al final de cada rollout."""
        # Recopilar métricas de episodios completos
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                
                # Métricas de INFINITO
                if 'latest_metrics' in info:
                    metrics = info['latest_metrics']
                    self.episode_phi.append(metrics.get('phi', 0))
                    self.episode_consciousness.append(metrics.get('consciousness', 0))
                    self.episode_ppl.append(metrics.get('perplexity', 0))
    
    def _log_metrics(self) -> None:
        """Loggear métricas en consola."""
        current_time = datetime.now()
        
        # Progreso
        progress = (self.num_timesteps / self.total_timesteps) * 100
        
        # Tiempo
        elapsed = current_time - self.start_time
        if self.num_timesteps > 0:
            time_per_step = elapsed.total_seconds() / self.num_timesteps
            remaining_steps = self.total_timesteps - self.num_timesteps
            eta = timedelta(seconds=int(time_per_step * remaining_steps))
        else:
            eta = timedelta(0)
        
        # Barra de progreso
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Header
        print(f"\n{'='*80}")
        print(f"📊 TIMESTEP {self.num_timesteps:,} / {self.total_timesteps:,} ({progress:.1f}%)")
        print(f"{'='*80}")
        print(f"[{bar}] {progress:.1f}%")
        print(f"⏱️  Transcurrido: {elapsed}  |  ETA: {eta}")
        
        # Rewards
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-10:]  # Últimos 10
            print(f"\n💰 REWARDS (últimos {len(recent_rewards)} episodios):")
            print(f"   Media: {np.mean(recent_rewards):+.4f}  |  "
                  f"Std: {np.std(recent_rewards):.4f}")
            print(f"   Min: {np.min(recent_rewards):+.4f}  |  "
                  f"Max: {np.max(recent_rewards):+.4f}")
        
        # Métricas INFINITO
        if self.episode_phi:
            recent_phi = self.episode_phi[-10:]
            recent_c = self.episode_consciousness[-10:]
            recent_ppl = self.episode_ppl[-10:]
            
            print(f"\n🧠 MÉTRICAS INFINITO (últimos {len(recent_phi)} episodios):")
            print(f"   Φ (PHI):         {np.mean(recent_phi):.3f} ± {np.std(recent_phi):.3f}  "
                  f"[{np.min(recent_phi):.2f}, {np.max(recent_phi):.2f}]")
            print(f"   C (Conscious):   {np.mean(recent_c):.3f} ± {np.std(recent_c):.3f}  "
                  f"[{np.min(recent_c):.2f}, {np.max(recent_c):.2f}]")
            print(f"   PPL (Perplex):   {np.mean(recent_ppl):.1f} ± {np.std(recent_ppl):.1f}  "
                  f"[{np.min(recent_ppl):.1f}, {np.max(recent_ppl):.1f}]")
            
            # Alertas
            avg_phi = np.mean(recent_phi)
            avg_ppl = np.mean(recent_ppl)
            
            if avg_phi > 6.0:
                print(f"   ⚠️  PHI alto ({avg_phi:.2f} > 6.0) - Riesgo de colapso Fase 2")
            elif 3.0 <= avg_phi <= 6.0:
                print(f"   ✅ PHI en rango óptimo [3.0, 6.0]")
            else:
                print(f"   ⚠️  PHI bajo ({avg_phi:.2f} < 3.0)")
            
            if avg_ppl < 10.0:
                print(f"   🚨 PPL MUY BAJO ({avg_ppl:.1f}) - Posible colapso/repetición")
            elif avg_ppl > 200.0:
                print(f"   ⚠️  PPL alto ({avg_ppl:.1f}) - Modelo confuso")
            else:
                print(f"   ✅ PPL en rango seguro [10, 200]")
        
        # Distribución de acciones
        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            print(f"\n🎮 DISTRIBUCIÓN DE ACCIONES (total: {total_actions:,}):")
            for action, count in sorted(self.action_counts.items()):
                pct = (count / total_actions) * 100
                bar_len = int(pct / 2)  # Escala a 50 chars max
                bar_viz = '█' * bar_len + '░' * (50 - bar_len)
                print(f"   {self.action_names[action]:6s}: {bar_viz} {pct:5.1f}% ({count:,})")
            
            # Análisis de estrategia
            text_pct = (self.action_counts[0] / total_actions) * 100
            phi_pct = (self.action_counts[1] / total_actions) * 100
            mixed_pct = (self.action_counts[2] / total_actions) * 100
            
            if mixed_pct == 0:
                print(f"   ⚠️  MIXED nunca usado - Agente no explora modo intermedio")
            elif mixed_pct > 20:
                print(f"   ✅ Buena exploración de MIXED ({mixed_pct:.1f}%)")
            
            if abs(text_pct - phi_pct) < 5:
                print(f"   📊 Estrategia balanceada TEXT/PHI")
            elif text_pct > phi_pct + 20:
                print(f"   📊 Estrategia dominada por TEXT ({text_pct:.0f}% vs {phi_pct:.0f}%)")
            elif phi_pct > text_pct + 20:
                print(f"   📊 Estrategia dominada por PHI ({phi_pct:.0f}% vs {text_pct:.0f}%)")
        
        # Longitud de episodios
        if self.episode_lengths:
            recent_lengths = self.episode_lengths[-10:]
            print(f"\n📏 LONGITUD EPISODIOS (últimos {len(recent_lengths)}):")
            print(f"   Media: {np.mean(recent_lengths):.1f} steps  |  "
                  f"Min: {np.min(recent_lengths)}  |  "
                  f"Max: {np.max(recent_lengths)}")
        
        print(f"{'='*80}\n")
        
        self.last_log_time = current_time
    
    def _on_training_end(self) -> None:
        """Llamado al final del entrenamiento."""
        end_time = datetime.now()
        total_duration = end_time - self.start_time
        
        print("\n" + "="*80)
        print("🏁 ENTRENAMIENTO FINALIZADO")
        print("="*80)
        print(f"Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duración total: {total_duration}")
        print(f"Timesteps completados: {self.num_timesteps:,}")
        
        # Resumen final
        if self.episode_rewards:
            print(f"\n📈 RESUMEN FINAL:")
            print(f"   Episodios completados: {len(self.episode_rewards)}")
            print(f"   Reward promedio: {np.mean(self.episode_rewards):+.4f}")
            print(f"   Reward final (últimos 10): {np.mean(self.episode_rewards[-10:]):+.4f}")
            
            if self.episode_phi:
                print(f"   PHI promedio: {np.mean(self.episode_phi):.3f}")
                print(f"   PPL promedio: {np.mean(self.episode_ppl):.1f}")
        
        if sum(self.action_counts.values()) > 0:
            total = sum(self.action_counts.values())
            print(f"\n🎮 DISTRIBUCIÓN FINAL DE ACCIONES:")
            for action, count in sorted(self.action_counts.items()):
                pct = (count / total) * 100
                print(f"   {self.action_names[action]:6s}: {pct:5.1f}% ({count:,})")
        
        print("="*80 + "\n")
