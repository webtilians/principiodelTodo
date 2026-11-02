#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 ANALIZADOR DE CALIDAD DE GENERACIÓN OPTIMIZADA
===============================================

Herramienta para diagnosticar por qué la generación sigue siendo incoherente
a pesar de la optimización de hiperparámetros IIT.
"""

import sys
import os

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import numpy as np
import json
import math
from datetime import datetime
from transformers import GPT2Tokenizer
from infinito_v5_2_refactored import InfinitoV52Refactored


class GenerationQualityAnalyzer:
    """Analizador de calidad de generación."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer)
    
    def load_optimized_model(self, config):
        """Carga un modelo con configuración optimizada."""
        model = InfinitoV52Refactored(
            vocab_size=self.vocab_size,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            memory_slots=256,
            dropout=config['dropout'],
            use_improved_memory=True,
            use_improved_iit=True,
            use_learnable_phi=True,
            use_stochastic_exploration=True,
            lambda_phi=config['lambda_phi'],
            seed=42
        ).to(self.device)
        
        return model
    
    def analyze_model_internals(self, model, prompt="The future of artificial intelligence"):
        """Analiza los estados internos del modelo."""
        print(f"🧠 ANÁLISIS DE ESTADOS INTERNOS")
        print(f"Prompt: '{prompt}'")
        print("="*50)
        
        model.eval()
        
        # Tokenizar prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            # Forward pass con métricas
            logits, metrics = model(input_tensor, return_metrics=True)
            
            # Analizar distribución de probabilidades
            probs = torch.softmax(logits[0, -1], dim=-1)  # Última posición
            top_k = 20
            top_probs, top_indices = torch.topk(probs, top_k)
            
            print(f"📊 Distribución de probabilidades (top {top_k}):")
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                token = self.tokenizer.decode([idx])
                print(f"   {i+1:2d}. '{token}' -> {prob.item():.4f}")
            
            # Analizar entropía
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            print(f"\n📈 Entropía de la distribución: {entropy.item():.4f}")
            print(f"   (Rango normal: 3-8, mayor = más diversidad)")
            
            # Analizar métricas IIT
            if metrics:
                print(f"\n🧮 Métricas IIT:")
                for key, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   {key}: {value.item():.4f}")
                    elif isinstance(value, (int, float)):
                        print(f"   {key}: {value:.4f}")
                    else:
                        print(f"   {key}: {str(value)}")
            
            # Analizar pesos de atención si están disponibles
            if hasattr(model, 'blocks'):
                attention_weights = []
                for block in model.blocks:
                    if hasattr(block, 'attention') and hasattr(block.attention, 'attention_weights'):
                        attention_weights.append(block.attention.attention_weights)
                
                if attention_weights:
                    print(f"\n🎯 Análisis de atención:")
                    for i, weights in enumerate(attention_weights):
                        if weights is not None:
                            mean_attn = weights.mean().item()
                            max_attn = weights.max().item()
                            print(f"   Capa {i+1}: Media={mean_attn:.4f}, Máx={max_attn:.4f}")
        
        return {
            'entropy': entropy.item(),
            'top_probs': [(self.tokenizer.decode([idx]), prob.item()) 
                         for prob, idx in zip(top_probs, top_indices)],
            'iit_metrics': metrics if metrics else {}
        }
    
    def generate_with_analysis(self, model, prompt, max_length=50, temperature=1.0):
        """Genera texto analizando cada paso."""
        print(f"\n🎨 GENERACIÓN CON ANÁLISIS PASO A PASO")
        print(f"Prompt: '{prompt}'")
        print(f"Max length: {max_length}, Temperature: {temperature}")
        print("="*60)
        
        model.eval()
        
        # Tokenizar prompt inicial
        input_ids = self.tokenizer.encode(prompt)
        current_ids = input_ids.copy()
        
        generated_tokens = []
        step_analysis = []
        
        for step in range(max_length):
            input_tensor = torch.tensor([current_ids], device=self.device)
            
            with torch.no_grad():
                logits, metrics = model(input_tensor, return_metrics=True)
                
                # Aplicar temperatura
                next_token_logits = logits[0, -1] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Analizar distribución
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                top_prob, top_idx = torch.max(probs, dim=-1)
                
                # Muestreo
                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()
                next_token_text = self.tokenizer.decode([next_token_id])
                
                # Guardar análisis del paso
                step_info = {
                    'step': step,
                    'token': next_token_text,
                    'token_id': next_token_id,
                    'probability': probs[next_token_id].item(),
                    'entropy': entropy.item(),
                    'top_probability': top_prob.item(),
                    'iit_phi': metrics.get('integration_phi', 0) if metrics else 0
                }
                step_analysis.append(step_info)
                
                # Mostrar progreso
                if step < 10 or step % 10 == 0:  # Mostrar los primeros 10 y luego cada 10
                    print(f"Step {step:2d}: '{next_token_text}' "
                          f"(P={step_info['probability']:.3f}, "
                          f"H={step_info['entropy']:.3f}, "
                          f"PHI={step_info['iit_phi']:.3f})")
                
                # Añadir token a la secuencia
                current_ids.append(next_token_id)
                generated_tokens.append(next_token_text)
                
                # Parar si es EOS
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                
                # Truncar secuencia si es muy larga (memory management)
                if len(current_ids) > 200:
                    current_ids = current_ids[-150:]  # Mantener últimos 150 tokens
        
        generated_text = ''.join(generated_tokens)
        full_text = prompt + generated_text
        
        print(f"\n📝 Texto completo generado:")
        print(f"'{full_text}'")
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'full_text': full_text,
            'step_analysis': step_analysis,
            'generation_stats': {
                'avg_entropy': np.mean([s['entropy'] for s in step_analysis]),
                'avg_probability': np.mean([s['probability'] for s in step_analysis]),
                'avg_phi': np.mean([s['iit_phi'] for s in step_analysis]),
                'total_steps': len(step_analysis)
            }
        }
    
    def diagnose_quality_issues(self, model, prompts=None):
        """Diagnóstica problemas de calidad en la generación."""
        if prompts is None:
            prompts = [
                "The future of artificial intelligence",
                "Machine learning algorithms can",
                "In the field of science",
                "Technology has revolutionized"
            ]
        
        print(f"\n🔍 DIAGNÓSTICO DE PROBLEMAS DE CALIDAD")
        print("="*60)
        
        issues = []
        all_analyses = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n🔸 Analizando prompt {i+1}: '{prompt}'")
            
            # Análisis de estados internos
            internal_analysis = self.analyze_model_internals(model, prompt)
            
            # Generación con análisis
            generation_analysis = self.generate_with_analysis(model, prompt, max_length=20)
            
            all_analyses.append({
                'prompt': prompt,
                'internal_analysis': internal_analysis,
                'generation_analysis': generation_analysis
            })
            
            # Detectar problemas específicos
            stats = generation_analysis['generation_stats']
            
            # Problema 1: Entropía muy baja (repetición)
            if stats['avg_entropy'] < 2.0:
                issues.append({
                    'type': 'low_entropy',
                    'prompt': prompt,
                    'value': stats['avg_entropy'],
                    'description': f"Entropía promedio muy baja ({stats['avg_entropy']:.3f}), indica repetición excesiva"
                })
            
            # Problema 2: Entropía muy alta (incoherencia)
            if stats['avg_entropy'] > 7.0:
                issues.append({
                    'type': 'high_entropy',
                    'prompt': prompt,
                    'value': stats['avg_entropy'],
                    'description': f"Entropía promedio muy alta ({stats['avg_entropy']:.3f}), indica generación aleatoria"
                })
            
            # Problema 3: PHI muy bajo (poca integración IIT)
            if stats['avg_phi'] < 0.1:
                issues.append({
                    'type': 'low_phi',
                    'prompt': prompt,
                    'value': stats['avg_phi'],
                    'description': f"PHI promedio muy bajo ({stats['avg_phi']:.3f}), poca integración de información"
                })
            
            # Problema 4: Probabilidades muy uniformes
            if stats['avg_probability'] < 0.1:
                issues.append({
                    'type': 'uniform_probs',
                    'prompt': prompt,
                    'value': stats['avg_probability'],
                    'description': f"Probabilidades muy uniformes ({stats['avg_probability']:.3f}), modelo indeciso"
                })
        
        # Resumen de problemas
        print(f"\n⚠️  PROBLEMAS DETECTADOS:")
        if not issues:
            print("   ✅ No se detectaron problemas graves de calidad")
        else:
            for issue in issues:
                print(f"   🔴 {issue['type']}: {issue['description']}")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        
        entropy_values = [a['generation_analysis']['generation_stats']['avg_entropy'] for a in all_analyses]
        phi_values = [a['generation_analysis']['generation_stats']['avg_phi'] for a in all_analyses]
        prob_values = [a['generation_analysis']['generation_stats']['avg_probability'] for a in all_analyses]
        
        avg_entropy = np.mean(entropy_values)
        avg_phi = np.mean(phi_values)
        avg_prob = np.mean(prob_values)
        
        if avg_entropy < 3.0:
            print("   📈 Aumentar temperatura o usar nucleus sampling para más diversidad")
        elif avg_entropy > 6.0:
            print("   📉 Reducir temperatura o aumentar repetition penalty")
        
        if avg_phi < 0.5:
            print("   🧠 Aumentar lambda_phi para mayor integración IIT")
            print("   🏋️ Considerar más entrenamiento para desarrollar representaciones IIT")
        
        if avg_prob < 0.2:
            print("   🎯 El modelo necesita más entrenamiento para desarrollar confianza")
            print("   📚 Considerar dataset más grande o más épocas")
        
        return {
            'analyses': all_analyses,
            'issues': issues,
            'summary_stats': {
                'avg_entropy': avg_entropy,
                'avg_phi': avg_phi,
                'avg_probability': avg_prob
            }
        }


def main():
    """Función principal."""
    # Configuración optimizada encontrada
    optimized_config = {
        'hidden_dim': 256,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.25,
        'lr': 2e-4,
        'lambda_phi': 0.010
    }
    
    print("🔍 ANALIZADOR DE CALIDAD DE GENERACIÓN OPTIMIZADA")
    print("="*60)
    print(f"Configuración: {optimized_config}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyzer = GenerationQualityAnalyzer(device)
    
    # Crear modelo con configuración optimizada
    model = analyzer.load_optimized_model(optimized_config)
    
    # Realizar diagnóstico completo
    diagnosis = analyzer.diagnose_quality_issues(model)
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'generation_quality_analysis_{timestamp}.json'
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': optimized_config,
        'diagnosis': diagnosis
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Análisis completo guardado en: {output_file}")
    print(f"✅ Diagnóstico completado!")


if __name__ == '__main__':
    main()