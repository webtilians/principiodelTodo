#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 SCRIPT DE PRODUCCIÓN - Generación con Modelo RL 30K Óptimo
==============================================================

Script listo para producción que usa el modelo RL entrenado (30K steps)
para generar texto con balance automático entre calidad textual y PHI.

Características:
- Carga eficiente del modelo 30K (óptimo)
- Generación de texto con control adaptativo
- Métricas en tiempo real (PHI, C, PPL)
- Configuración flexible
- Manejo robusto de errores

Uso:
    python generate_with_rl_30k.py --prompt "Your prompt here" --max-length 200
    
Autor: INFINITO Team
Fecha: 2025-11-11
Versión: 1.0.0
"""

import torch
import numpy as np
import json
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# Importar componentes necesarios
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️ stable-baselines3 no disponible. Instalar: pip install stable-baselines3")
    SB3_AVAILABLE = False

try:
    from src.rl.infinito_rl_env import InfinitoRLEnv
    ENV_AVAILABLE = True
except ImportError:
    print("⚠️ InfinitoRLEnv no disponible. Verificar estructura del proyecto.")
    ENV_AVAILABLE = False


class RLTextGenerator:
    """
    Generador de texto usando el modelo RL 30K óptimo.
    
    Maneja la carga del modelo, configuración del entorno y generación de texto
    con control adaptativo PHI vs Texto.
    """
    
    def __init__(
        self,
        checkpoint_path: str = "outputs/rl_phi_text_scheduler/checkpoints/ppo_infinito_scheduler_30000_steps.zip",
        config_path: str = "outputs/rl_phi_text_scheduler/env_config.json",
        device: Optional[str] = None
    ):
        """
        Inicializar generador.
        
        Args:
            checkpoint_path: Ruta al checkpoint del modelo RL 30K
            config_path: Ruta a la configuración del entorno
            device: Device ('cuda', 'cpu' o None para auto)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.config_path = Path(config_path)
        
        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Verificar disponibilidad
        if not SB3_AVAILABLE or not ENV_AVAILABLE:
            raise RuntimeError("Dependencias faltantes. Verificar instalación.")
        
        # Verificar archivos
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {self.checkpoint_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config no encontrado: {self.config_path}")
        
        # Estado
        self.model = None
        self.env = None
        self.config = None
        self.loaded = False
        
        print(f"✅ RLTextGenerator inicializado")
        print(f"   Device: {self.device}")
        print(f"   Checkpoint: {self.checkpoint_path}")
    
    def load(self) -> None:
        """Cargar modelo y entorno."""
        if self.loaded:
            print("⚠️ Modelo ya cargado")
            return
        
        print("\n📦 Cargando modelo RL 30K...")
        
        # 1. Cargar configuración
        print("   [1/3] Cargando configuración...")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        print("   ✅ Configuración cargada")
        
        # 2. Crear entorno
        print("   [2/3] Creando entorno INFINITO...")
        self.env = InfinitoRLEnv(config=self.config)
        print("   ✅ Entorno creado")
        
        # 3. Cargar modelo PPO
        print("   [3/3] Cargando modelo PPO 30K...")
        self.model = PPO.load(str(self.checkpoint_path), env=self.env)
        print("   ✅ Modelo PPO cargado")
        
        self.loaded = True
        print("\n✅ Modelo RL 30K listo para generar")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        max_steps: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        verbose: bool = True,
        return_metrics: bool = True
    ) -> Dict:
        """
        Generar texto usando el agente RL.
        
        Args:
            prompt: Texto inicial
            max_length: Longitud máxima del texto generado (tokens)
            max_steps: Pasos máximos de RL (decisiones del agente)
            temperature: Temperatura de muestreo
            top_k: Top-k para muestreo
            top_p: Top-p (nucleus) para muestreo
            verbose: Mostrar progreso
            return_metrics: Incluir métricas detalladas en el resultado
        
        Returns:
            Dict con:
                - text: Texto generado
                - metrics: Métricas del proceso (si return_metrics=True)
                - actions: Acciones tomadas por el agente
                - stats: Estadísticas de generación
        """
        if not self.loaded:
            raise RuntimeError("Modelo no cargado. Llamar load() primero.")
        
        if verbose:
            print("\n" + "="*70)
            print("🚀 GENERACIÓN CON MODELO RL 30K")
            print("="*70)
            print(f"Prompt: '{prompt}'")
            print(f"Max length: {max_length} tokens")
            print(f"Max RL steps: {max_steps}")
            print(f"Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
            print("="*70)
        
        # Inicializar tracking
        start_time = datetime.now()
        actions_log = []
        rewards_log = []
        phi_log = []
        consciousness_log = []
        ppl_log = []
        
        # Resetear entorno
        obs, info = self.env.reset()
        
        # Tokenizar prompt - usar el tokenizer del dataset
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        if verbose:
            print(f"\n📝 Generando ({max_steps} pasos RL)...")
            print()
        
        # Loop de generación con RL
        step = 0
        done = False
        
        while step < max_steps and not done:
            # Agente decide acción (TEXT, PHI o MIXED)
            action, _states = self.model.predict(obs, deterministic=False)
            action_name = ['TEXT', 'PHI', 'MIXED'][action]
            
            # Ejecutar acción en el entorno
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Guardar métricas
            actions_log.append(int(action))
            rewards_log.append(float(reward))
            
            metrics = info.get('latest_metrics', {})
            phi_log.append(float(metrics.get('phi', 0)))
            consciousness_log.append(float(metrics.get('consciousness', 0)))
            ppl_log.append(float(metrics.get('perplexity', 0)))
            
            # Mostrar progreso cada 10 pasos
            if verbose and (step + 1) % 10 == 0:
                phi = phi_log[-1]
                c = consciousness_log[-1]
                ppl = ppl_log[-1]
                
                print(f"  Step {step+1:2d}: {action_name:5s} | "
                      f"Φ={phi:5.2f} | C={c:4.2f} | PPL={ppl:6.1f} | "
                      f"R={reward:+6.3f}")
            
            step += 1
        
        # Generar texto final usando el modelo entrenado
        if verbose:
            print(f"\n📖 Generando texto final...")
        
        generated_ids = None
        try:
            # Cargar tokenizer
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            
            with torch.no_grad():
                generated_ids = self.env.model.gpt2.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"⚠️ Error en generación final: {e}")
            generated_text = prompt + " [Error en generación]"
        
        # Calcular estadísticas
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        actions_count = {
            'TEXT': actions_log.count(0),
            'PHI': actions_log.count(1),
            'MIXED': actions_log.count(2)
        }
        
        total_actions = len(actions_log)
        
        stats = {
            'duration_seconds': duration,
            'rl_steps': total_actions,
            'tokens_generated': len(generated_ids[0]) - len(input_ids[0]) if generated_ids is not None else 0,
            'actions_distribution': {
                'TEXT': {'count': actions_count['TEXT'], 'percentage': (actions_count['TEXT'] / total_actions * 100) if total_actions > 0 else 0},
                'PHI': {'count': actions_count['PHI'], 'percentage': (actions_count['PHI'] / total_actions * 100) if total_actions > 0 else 0},
                'MIXED': {'count': actions_count['MIXED'], 'percentage': (actions_count['MIXED'] / total_actions * 100) if total_actions > 0 else 0}
            },
            'phi_mean': float(np.mean(phi_log)) if phi_log else 0.0,
            'phi_std': float(np.std(phi_log)) if phi_log else 0.0,
            'phi_range': [float(np.min(phi_log)), float(np.max(phi_log))] if phi_log else [0.0, 0.0],
            'phi_in_optimal_range_pct': float(np.mean([(3 <= p <= 6) for p in phi_log]) * 100) if phi_log else 0.0,
            'consciousness_mean': float(np.mean(consciousness_log)) if consciousness_log else 0.0,
            'perplexity_mean': float(np.mean(ppl_log)) if ppl_log else 0.0,
            'perplexity_safe_pct': float(np.mean([p >= 10 for p in ppl_log]) * 100) if ppl_log else 0.0,
            'total_reward': float(np.sum(rewards_log)) if rewards_log else 0.0,
            'mean_reward': float(np.mean(rewards_log)) if rewards_log else 0.0,
        }
        
        # Mostrar estadísticas
        if verbose:
            print("\n" + "="*70)
            print("📊 ESTADÍSTICAS DE GENERACIÓN")
            print("="*70)
            
            print(f"\n⏱️ Tiempo: {duration:.2f}s")
            print(f"📏 Tokens generados: {stats['tokens_generated']}")
            print(f"🎮 Pasos RL: {total_actions}")
            
            print(f"\n🎯 Distribución de acciones:")
            for action_name, data in stats['actions_distribution'].items():
                count = data['count']
                pct = data['percentage']
                bar = '█' * int(pct / 3)
                print(f"  {action_name:5s}: {count:2d} ({pct:5.1f}%) {bar}")
            
            print(f"\n🧠 Métricas INFINITO:")
            print(f"  PHI (Φ):")
            print(f"    Promedio: {stats['phi_mean']:6.3f}")
            print(f"    Std:      {stats['phi_std']:6.3f}")
            print(f"    Rango:    [{stats['phi_range'][0]:6.3f}, {stats['phi_range'][1]:6.3f}]")
            print(f"    En [3-6]: {stats['phi_in_optimal_range_pct']:6.2f}% {'✅' if stats['phi_in_optimal_range_pct'] > 70 else '⚠️'}")
            
            print(f"\n  Consciousness (C): {stats['consciousness_mean']:6.3f}")
            print(f"  Perplexity (PPL): {stats['perplexity_mean']:7.2f}")
            print(f"    >= 10: {stats['perplexity_safe_pct']:5.1f}% {'✅' if stats['perplexity_safe_pct'] > 90 else '⚠️'}")
            
            print(f"\n💰 Rewards:")
            print(f"    Total: {stats['total_reward']:+8.3f}")
            print(f"    Media: {stats['mean_reward']:+8.3f}")
            
            print("\n" + "="*70)
            print("📄 TEXTO GENERADO")
            print("="*70)
            print(generated_text)
            print("="*70)
        
        # Construir resultado
        result = {
            'text': generated_text,
            'stats': stats,
            'prompt': prompt
        }
        
        if return_metrics:
            result['metrics'] = {
                'actions': actions_log,
                'rewards': rewards_log,
                'phi': phi_log,
                'consciousness': consciousness_log,
                'perplexity': ppl_log
            }
        
        return result
    
    def close(self) -> None:
        """Liberar recursos."""
        if self.env is not None:
            self.env.close()
            self.env = None
        
        self.model = None
        self.loaded = False
        torch.cuda.empty_cache()
        
        print("✅ Recursos liberados")


def main():
    """Función principal con CLI."""
    parser = argparse.ArgumentParser(
        description="Generación de texto con modelo RL 30K óptimo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python generate_with_rl_30k.py --prompt "The nature of consciousness"
  python generate_with_rl_30k.py --prompt "AI will" --max-length 150 --temperature 0.9
  python generate_with_rl_30k.py --prompt "In the future" --quiet
        """
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default="The nature of consciousness",
        help='Texto inicial para generar (default: "The nature of consciousness")'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=200,
        help='Longitud máxima en tokens (default: 200)'
    )
    
    parser.add_argument(
        '--max-steps',
        type=int,
        default=50,
        help='Pasos máximos de RL (default: 50)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Temperatura de muestreo (default: 0.8)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='Top-k para muestreo (default: 40)'
    )
    
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p (nucleus) para muestreo (default: 0.9)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Modo silencioso (solo texto generado)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default="outputs/rl_phi_text_scheduler/checkpoints/ppo_infinito_scheduler_30000_steps.zip",
        help='Ruta al checkpoint del modelo (default: modelo 30K)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Guardar resultado en archivo JSON'
    )
    
    args = parser.parse_args()
    
    try:
        # Crear generador
        generator = RLTextGenerator(
            checkpoint_path=args.checkpoint,
            device=None  # Auto-detect
        )
        
        # Cargar modelo
        generator.load()
        
        # Generar
        result = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            max_steps=args.max_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            verbose=not args.quiet,
            return_metrics=True
        )
        
        # Guardar si se especificó
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Resultado guardado en: {output_path}")
        
        # Si modo silencioso, solo imprimir texto
        if args.quiet:
            print(result['text'])
        
        # Cerrar
        generator.close()
        
        return 0
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
