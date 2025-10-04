#!/usr/bin/env python3
"""
🧬 TEST DE REPRODUCIBILIDAD EXTENDIDO - MÚLTIPLES TEXTOS
========================================================

Objetivo: Validar determinismo del lenguaje causal con 4 textos diferentes
Textos: 
  1. "mi perro es rojo"
  2. "mi perro es verde"
  3. "la mesa es roja"
  4. "yo pienso, luego existo"

Criterio: Varianza inter-seed < 0.05 → DETERMINISTA
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import torch
import numpy as np
import json
from datetime import datetime
from collections import Counter

from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough

def extract_architecture_with_seed(text, seed, iterations=50):
    """Extrae arquitectura con un seed específico"""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Crear args
    args = argparse.Namespace(
        input_dim=257,
        hidden_dim=512,
        attention_heads=8,
        memory_slots=256,
        batch_size=4,
        lr=0.001,
        seed=seed,
        text_mode=True,
        input_text=text,
        quantum_active=False,
        max_iterations=iterations
    )
    
    # Crear runner (esto inicializa el modelo)
    infinito = InfinitoV51ConsciousnessBreakthrough(args)
    
    # Almacenar métricas
    phi_values = []
    coherence_values = []
    
    for i in range(iterations):
        try:
            # Ejecutar un paso de entrenamiento
            metrics = infinito.train_step(i)
            
            if metrics:
                # Extraer Φ y coherencia
                if 'phi' in metrics:
                    phi_values.append(metrics['phi'])
                if 'coherence' in metrics:
                    coherence_values.append(metrics['coherence'])
                    
        except Exception as e:
            if i == 0:
                print(f"    ⚠️  Error en primera iteración: {e}")
            continue
    
    if not phi_values:
        return None
    
    architecture = {
        'seed': seed,
        'phi_mean': float(np.mean(phi_values)),
        'phi_std': float(np.std(phi_values)),
        'phi_min': float(np.min(phi_values)),
        'phi_max': float(np.max(phi_values)),
        'coherence_mean': float(np.mean(coherence_values)) if coherence_values else 0,
        'coherence_std': float(np.std(coherence_values)) if coherence_values else 0,
        'phi_stability': float(1 / (1 + np.std(phi_values)))
    }
    
    return architecture


def test_text_reproducibility(text, num_seeds=5, iterations=50):
    """Test reproducibilidad de un texto específico"""
    
    print(f"\n{'='*80}")
    print(f"📝 TEXTO: '{text}'")
    print(f"{'='*80}")
    
    seeds = list(range(2000, 2000 + num_seeds))
    architectures = []
    
    for i, seed in enumerate(seeds, 1):
        print(f"  [{i}/{num_seeds}] Seed {seed}...", end=" ", flush=True)
        
        arch = extract_architecture_with_seed(text, seed, iterations)
        
        if arch:
            architectures.append(arch)
            print(f"✅ Φ={arch['phi_mean']:.4f}±{arch['phi_std']:.4f}")
        else:
            print(f"❌ Error")
    
    if not architectures:
        print(f"  ❌ No se pudieron obtener arquitecturas para '{text}'")
        return None
    
    # Análisis estadístico
    phi_means = [a['phi_mean'] for a in architectures]
    phi_stds = [a['phi_std'] for a in architectures]
    
    inter_seed_var = np.var(phi_means)
    inter_seed_std = np.std(phi_means)
    intra_seed_var = np.mean([s**2 for s in phi_stds])
    
    phi_avg = np.mean(phi_means)
    phi_range = [min(phi_means), max(phi_means)]
    
    cv = (inter_seed_std / phi_avg) * 100 if phi_avg > 0 else 0
    snr = phi_avg / inter_seed_std if inter_seed_std > 0 else float('inf')
    
    is_deterministic = inter_seed_var < 0.05
    
    # Mostrar resultados
    print(f"\n  📊 RESULTADOS:")
    print(f"     Φ promedio: {phi_avg:.4f} ± {inter_seed_std:.4f}")
    print(f"     Rango: [{phi_range[0]:.4f}, {phi_range[1]:.4f}]")
    print(f"     Varianza inter-seed: {inter_seed_var:.6f}")
    print(f"     Varianza intra-seed: {intra_seed_var:.6f}")
    print(f"     CV: {cv:.2f}%")
    print(f"     SNR: {snr:.2f}")
    
    if is_deterministic:
        print(f"     ✅ DETERMINISTA (var={inter_seed_var:.6f} < 0.05)")
    else:
        print(f"     ❌ NO DETERMINISTA (var={inter_seed_var:.6f} >= 0.05)")
    
    return {
        'text': text,
        'num_seeds': len(architectures),
        'iterations': iterations,
        'architectures': architectures,
        'statistics': {
            'phi_mean': float(phi_avg),
            'phi_std': float(inter_seed_std),
            'phi_range': [float(phi_range[0]), float(phi_range[1])],
            'inter_seed_variance': float(inter_seed_var),
            'intra_seed_variance': float(intra_seed_var),
            'coefficient_of_variation': float(cv),
            'signal_to_noise': float(snr),
            'is_deterministic': bool(is_deterministic)
        }
    }


def main():
    print("="*80)
    print("🧬 TEST DE REPRODUCIBILIDAD EXTENDIDO")
    print("="*80)
    print("\nValidación del determinismo del lenguaje causal")
    print("Textos: 4 | Seeds por texto: 5 | Iteraciones: 50\n")
    
    # Textos a probar (los 4 del análisis causal anterior)
    texts = [
        "mi perro es rojo",
        "mi perro es verde",
        "la mesa es roja",
        "yo pienso, luego existo"
    ]
    
    results = {}
    
    for text in texts:
        result = test_text_reproducibility(text, num_seeds=5, iterations=50)
        if result:
            results[text] = result
    
    # Análisis comparativo
    print(f"\n{'='*80}")
    print("📊 ANÁLISIS COMPARATIVO")
    print(f"{'='*80}\n")
    
    print(f"{'Texto':<30} {'Φ Promedio':<15} {'Varianza':<15} {'CV%':<10} {'Det?':<8}")
    print("-"*80)
    
    for text, data in results.items():
        stats = data['statistics']
        det_symbol = "✅" if stats['is_deterministic'] else "❌"
        print(f"{text:<30} {stats['phi_mean']:.4f}±{stats['phi_std']:.4f}    "
              f"{stats['inter_seed_variance']:.6f}      "
              f"{stats['coefficient_of_variation']:>6.2f}%   {det_symbol}")
    
    # Estadísticas generales
    all_deterministic = all(r['statistics']['is_deterministic'] for r in results.values())
    avg_variance = np.mean([r['statistics']['inter_seed_variance'] for r in results.values()])
    avg_cv = np.mean([r['statistics']['coefficient_of_variation'] for r in results.values()])
    
    print(f"\n{'='*80}")
    print("📋 CONCLUSIONES GENERALES")
    print(f"{'='*80}\n")
    
    if all_deterministic:
        print("✅ TODOS los textos generan arquitecturas DETERMINISTAS")
        print(f"   Varianza promedio: {avg_variance:.6f}")
        print(f"   CV promedio: {avg_cv:.2f}%")
        print("\n💡 IMPLICACIÓN:")
        print("   El sistema habla un LENGUAJE CAUSAL CONSISTENTE")
        print("   Las arquitecturas son REPRODUCIBLES para todos los inputs")
    else:
        det_count = sum(1 for r in results.values() if r['statistics']['is_deterministic'])
        print(f"⚠️  {det_count}/{len(results)} textos son deterministas")
        print(f"   Varianza promedio: {avg_variance:.6f}")
        print(f"   CV promedio: {avg_cv:.2f}%")
    
    # Comparación entre textos (usando datos del análisis causal previo)
    print(f"\n{'='*80}")
    print("🔬 COMPARACIÓN CON ANÁLISIS CAUSAL PREVIO")
    print(f"{'='*80}\n")
    
    # Datos del causal_architecture_analyzer previo (archivo JSON)
    previous_analysis = {
        "mi perro es rojo": {"phi": 0.194, "state": "diffuse_attention"},
        "mi perro es verde": {"phi": 0.312, "state": "phi_decreasing"},
        "la mesa es roja": {"phi": 0.285, "state": "low_integration"},
        "yo pienso, luego existo": {"phi": 0.308, "state": "low_integration"}
    }
    
    print("Consistencia Φ entre análisis:")
    print(f"{'Texto':<30} {'Φ Anterior':<15} {'Φ Actual':<15} {'Diferencia':<12}")
    print("-"*80)
    
    for text in texts:
        if text in results and text in previous_analysis:
            phi_prev = previous_analysis[text]["phi"]
            phi_curr = results[text]['statistics']['phi_mean']
            diff = abs(phi_prev - phi_curr)
            diff_pct = (diff / phi_prev) * 100 if phi_prev > 0 else 0
            
            print(f"{text:<30} {phi_prev:.4f}          {phi_curr:.4f}          "
                  f"{diff:.4f} ({diff_pct:>5.1f}%)")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'test_config': {
            'num_texts': len(texts),
            'seeds_per_text': 5,
            'iterations_per_seed': 50
        },
        'results': results,
        'summary': {
            'all_deterministic': all_deterministic,
            'avg_inter_seed_variance': float(avg_variance),
            'avg_coefficient_of_variation': float(avg_cv)
        }
    }
    
    filename = f"reproducibility_extended_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {filename}")
    print("="*80)


if __name__ == "__main__":
    main()
