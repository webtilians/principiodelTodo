#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 EVALUADOR COMPARATIVO DE GENERACIÓN DE TEXTO
==============================================

Herramienta para comparar la generación original vs mejorada
y cuantificar las mejoras en diversidad, coherencia y calidad.
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
import numpy as np
import argparse
import json
from datetime import datetime
from collections import Counter
import re
from transformers import GPT2Tokenizer
from infinito_v5_2_refactored import InfinitoV52Refactored
from improved_text_generation import ImprovedTextGenerator, load_model_and_tokenizer


class GenerationEvaluator:
    """Evaluador comparativo de técnicas de generación."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.improved_generator = ImprovedTextGenerator(model, tokenizer, device)
    
    def generate_baseline(self, prompt, max_length=100):
        """Generación baseline (simple greedy/sampling)."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.model(generated_ids)
                
                # Obtener logits del último token
                if isinstance(outputs, tuple):
                    logits = outputs[0][:, -1, :]
                else:
                    logits = outputs[:, -1, :]
                
                # Sampling simple con temperatura
                temperature = 0.7
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Agregar token generado
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Verificar fin de secuencia
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decodificar resultado
        generated_text = self.tokenizer.decode(
            generated_ids[0][len(input_ids[0]):], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def calculate_text_metrics(self, text):
        """Calcula métricas detalladas de un texto."""
        if not text or not text.strip():
            return {
                'ttr': 0.0,
                'unique_words': 0,
                'total_words': 0,
                'avg_word_length': 0.0,
                'repetition_2gram': 0.0,
                'repetition_3gram': 0.0,
                'sentence_count': 0,
                'avg_sentence_length': 0.0
            }
        
        # Preprocessing
        words = re.findall(r'\b\w+\b', text.lower())
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Métricas básicas
        total_words = len(words)
        unique_words = len(set(words))
        ttr = unique_words / total_words if total_words > 0 else 0
        
        # Longitud promedio de palabras
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
        
        # Repetición de n-gramas
        def calculate_ngram_repetition(words, n):
            if len(words) < n:
                return 0.0
            
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts = Counter(ngrams)
            repeated = sum(count - 1 for count in ngram_counts.values() if count > 1)
            return repeated / len(ngrams) if ngrams else 0.0
        
        rep_2gram = calculate_ngram_repetition(words, 2)
        rep_3gram = calculate_ngram_repetition(words, 3)
        
        # Métricas de oraciones
        sentence_count = len(sentences)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / sentence_count if sentence_count > 0 else 0
        
        return {
            'ttr': ttr,
            'unique_words': unique_words,
            'total_words': total_words,
            'avg_word_length': avg_word_length,
            'repetition_2gram': rep_2gram,
            'repetition_3gram': rep_3gram,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length
        }
    
    def evaluate_comparison(self, prompts, num_samples=3):
        """Compara generación baseline vs mejorada para múltiples prompts."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'prompts': [],
            'summary': {}
        }
        
        all_baseline_metrics = []
        all_improved_metrics = []
        
        print(f"📊 EVALUACIÓN COMPARATIVA DE GENERACIÓN")
        print("="*60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n🎯 Prompt {i}/{len(prompts)}: '{prompt}'")
            print("-" * 50)
            
            prompt_results = {
                'prompt': prompt,
                'baseline': {'texts': [], 'metrics': []},
                'improved': {'texts': [], 'metrics': []}
            }
            
            # Generar muestras baseline
            print("🔄 Generando baseline...")
            for j in range(num_samples):
                baseline_text = self.generate_baseline(prompt)
                metrics = self.calculate_text_metrics(baseline_text)
                
                prompt_results['baseline']['texts'].append(baseline_text)
                prompt_results['baseline']['metrics'].append(metrics)
                all_baseline_metrics.append(metrics)
                
                print(f"   Baseline {j+1}: TTR={metrics['ttr']:.3f}, Palabras={metrics['total_words']}")
            
            # Generar muestras mejoradas
            print("🔄 Generando mejoradas...")
            improved_texts = self.improved_generator.generate_multiple_samples(
                prompt, 
                num_samples=num_samples,
                temperature=0.85,
                top_p=0.9,
                repetition_penalty=1.15,
                frequency_penalty=0.05
            )
            
            for j, improved_text in enumerate(improved_texts):
                metrics = self.calculate_text_metrics(improved_text)
                
                prompt_results['improved']['texts'].append(improved_text)
                prompt_results['improved']['metrics'].append(metrics)
                all_improved_metrics.append(metrics)
                
                print(f"   Mejorada {j+1}: TTR={metrics['ttr']:.3f}, Palabras={metrics['total_words']}")
            
            results['prompts'].append(prompt_results)
        
        # Calcular métricas promedio
        def average_metrics(metrics_list):
            if not metrics_list:
                return {}
            
            keys = metrics_list[0].keys()
            return {key: sum(m[key] for m in metrics_list) / len(metrics_list) for key in keys}
        
        baseline_avg = average_metrics(all_baseline_metrics)
        improved_avg = average_metrics(all_improved_metrics)
        
        # Calcular mejoras
        improvements = {}
        for key in baseline_avg:
            if baseline_avg[key] > 0:
                if key in ['repetition_2gram', 'repetition_3gram']:
                    # Para repetición, menor es mejor
                    improvement = (baseline_avg[key] - improved_avg[key]) / baseline_avg[key] * 100
                else:
                    # Para otras métricas, mayor es mejor
                    improvement = (improved_avg[key] - baseline_avg[key]) / baseline_avg[key] * 100
                improvements[key] = improvement
            else:
                improvements[key] = 0.0
        
        results['summary'] = {
            'baseline_avg': baseline_avg,
            'improved_avg': improved_avg,
            'improvements': improvements,
            'total_samples': len(all_baseline_metrics)
        }
        
        # Mostrar resumen
        self._print_summary(baseline_avg, improved_avg, improvements)
        
        return results
    
    def _print_summary(self, baseline_avg, improved_avg, improvements):
        """Imprime resumen de la comparación."""
        print(f"\n📈 RESUMEN DE MEJORAS")
        print("="*60)
        
        metrics_info = {
            'ttr': ('Diversidad (TTR)', '↑ Mayor es mejor'),
            'unique_words': ('Palabras únicas', '↑ Mayor es mejor'),
            'total_words': ('Total palabras', '→ Neutro'),
            'avg_word_length': ('Long. promedio palabra', '→ Neutro'),
            'repetition_2gram': ('Repetición 2-gram', '↓ Menor es mejor'),
            'repetition_3gram': ('Repetición 3-gram', '↓ Menor es mejor'),
            'sentence_count': ('Número oraciones', '→ Neutro'),
            'avg_sentence_length': ('Long. promedio oración', '→ Neutro')
        }
        
        for key, (name, direction) in metrics_info.items():
            baseline_val = baseline_avg[key]
            improved_val = improved_avg[key]
            improvement = improvements[key]
            
            status = "🟢" if improvement > 5 else "🟡" if improvement > -5 else "🔴"
            
            print(f"{status} {name:25} | "
                  f"Base: {baseline_val:6.3f} → "
                  f"Mejorado: {improved_val:6.3f} | "
                  f"Cambio: {improvement:+6.1f}% {direction}")
        
        # Métricas clave
        print(f"\n🎯 MÉTRICAS CLAVE:")
        print(f"   TTR (Diversidad):     {baseline_avg['ttr']:.3f} → {improved_avg['ttr']:.3f} ({improvements['ttr']:+.1f}%)")
        print(f"   Repetición 2-gram:    {baseline_avg['repetition_2gram']:.3f} → {improved_avg['repetition_2gram']:.3f} ({improvements['repetition_2gram']:+.1f}%)")
        print(f"   Repetición 3-gram:    {baseline_avg['repetition_3gram']:.3f} → {improved_avg['repetition_3gram']:.3f} ({improvements['repetition_3gram']:+.1f}%)")
        
        # Evaluación general
        key_improvements = [
            improvements['ttr'],
            improvements['repetition_2gram'],  # Mayor es mejor (reducción de repetición)
            improvements['repetition_3gram']   # Mayor es mejor (reducción de repetición)
        ]
        
        avg_improvement = sum(key_improvements) / len(key_improvements)
        
        if avg_improvement > 20:
            evaluation = "🌟 EXCELENTE"
        elif avg_improvement > 10:
            evaluation = "✅ MUY BUENO"
        elif avg_improvement > 5:
            evaluation = "👍 BUENO"
        elif avg_improvement > 0:
            evaluation = "🟡 MODERADO"
        else:
            evaluation = "❌ NECESITA MEJORAS"
        
        print(f"\n🏆 EVALUACIÓN GENERAL: {evaluation}")
        print(f"   Mejora promedio clave: {avg_improvement:.1f}%")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Evaluación comparativa de generación de texto')
    
    parser.add_argument('model_path', help='Path al modelo (.pt)')
    parser.add_argument('--prompts', nargs='+', 
                       default=[
                           'The future of artificial intelligence',
                           'In a world where technology',
                           'Science has always been',
                           'Once upon a time in'
                       ],
                       help='Lista de prompts para evaluar')
    parser.add_argument('--samples', type=int, default=3,
                       help='Número de muestras por método')
    parser.add_argument('--output', type=str, default=None,
                       help='Archivo para guardar resultados JSON')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device para inferencia')
    
    args = parser.parse_args()
    
    # Cargar modelo
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.device)
    
    # Crear evaluador
    evaluator = GenerationEvaluator(model, tokenizer, device)
    
    # Ejecutar evaluación
    results = evaluator.evaluate_comparison(args.prompts, args.samples)
    
    # Guardar resultados si se especifica
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Resultados guardados en: {args.output}")
    
    print(f"\n✅ Evaluación completada!")


if __name__ == '__main__':
    main()