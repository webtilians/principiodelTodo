#!/usr/bin/env python3
"""
🧬 TEST DE REPRODUCIBILIDAD DEL LENGUAJE CAUSAL
===============================================

Objetivo: ¿El sistema genera arquitecturas reproducibles?
Método: Ejecutar "mi perro es rojo" 10 veces con seeds diferentes
Criterio: Varianza de Φ entre seeds < 0.05 → DETERMINISTA
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

print("=" * 80)
print("🧬 TEST DE REPRODUCIBILIDAD DEL LENGUAJE CAUSAL")
print("=" * 80)
print(f"\nTexto de prueba: 'mi perro es rojo'")
print(f"Número de seeds: 10")
print(f"Iteraciones por seed: 50\n")

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
    
    print(f"  [Seed {seed}] Ejecutando {iterations} iteraciones...")
    
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
            
            if (i + 1) % 10 == 0:
                avg_phi = np.mean(phi_values[-10:]) if phi_values else 0
                print(f"    Iter {i+1}/{iterations}: Φ={avg_phi:.3f}")
                
        except Exception as e:
            if i == 0:  # Solo mostrar error en primera iteración
                print(f"    ⚠️  Error: {e}")
            continue
    
    if not phi_values:
        print(f"    ❌ No se pudieron obtener valores")
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
    
    print(f"    ✅ Φ = {architecture['phi_mean']:.4f} ± {architecture['phi_std']:.4f}")
    
    return architecture


def main():
    text = "mi perro es rojo"
    num_seeds = 10
    iterations = 50
    
    # Seeds: 1000-1009
    seeds = list(range(1000, 1000 + num_seeds))
    architectures = []
    
    print("Ejecutando análisis...\n")
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n[Ejecución {i}/{num_seeds}]")
        arch = extract_architecture_with_seed(text, seed, iterations)
        
        if arch:
            architectures.append(arch)
        else:
            print(f"  ⚠️  Saltando seed {seed}")
    
    if not architectures:
        print("\n❌ ERROR: No se pudieron obtener arquitecturas")
        return
    
    # Análisis estadístico
    print("\n" + "=" * 80)
    print("ANÁLISIS DE REPRODUCIBILIDAD")
    print("=" * 80)
    
    phi_means = [a['phi_mean'] for a in architectures]
    phi_stds = [a['phi_std'] for a in architectures]
    coherence_means = [a['coherence_mean'] for a in architectures]
    
    # Varianzas
    inter_seed_var = np.var(phi_means)
    inter_seed_std = np.std(phi_means)
    intra_seed_var = np.mean([s**2 for s in phi_stds])
    
    print(f"\n📊 VARIANZA DE Φ:")
    print(f"  Entre seeds (inter-seed):  {inter_seed_var:.6f}")
    print(f"  Desviación estándar:       {inter_seed_std:.6f}")
    print(f"  Dentro de seeds (intra-seed): {intra_seed_var:.6f}")
    print(f"  Rango Φ: [{min(phi_means):.4f}, {max(phi_means):.4f}]")
    print(f"  Φ promedio: {np.mean(phi_means):.4f} ± {inter_seed_std:.4f}")
    
    print(f"\n🔗 COHERENCIA:")
    print(f"  Media: {np.mean(coherence_means):.4f} ± {np.std(coherence_means):.4f}")
    
    # Determinismo
    print("\n" + "=" * 80)
    print("EVALUACIÓN DEL DETERMINISMO")
    print("=" * 80)
    
    threshold = 0.05
    is_deterministic = inter_seed_var < threshold
    
    print(f"\n🎯 Umbral de varianza: {threshold}")
    print(f"📏 Varianza observada: {inter_seed_var:.6f}")
    
    if is_deterministic:
        print(f"\n✅ RESULTADO: El lenguaje causal ES DETERMINISTA")
        print(f"   Varianza ({inter_seed_var:.6f}) < {threshold}")
        print(f"   Las arquitecturas son REPRODUCIBLES")
    else:
        print(f"\n❌ RESULTADO: El lenguaje causal NO es determinista")
        print(f"   Varianza ({inter_seed_var:.6f}) >= {threshold}")
        print(f"   Hay variabilidad significativa")
    
    # Métricas adicionales
    snr = np.mean(phi_means) / inter_seed_std if inter_seed_std > 0 else float('inf')
    cv = (inter_seed_std / np.mean(phi_means)) * 100 if np.mean(phi_means) > 0 else 0
    
    print(f"\n📡 Ratio señal/ruido: {snr:.2f}")
    print(f"📊 Coeficiente de variación: {cv:.2f}%")
    
    if cv < 5:
        print(f"   ✅ Variabilidad MUY BAJA")
    elif cv < 10:
        print(f"   ✅ Variabilidad BAJA")
    elif cv < 20:
        print(f"   ⚠️  Variabilidad MODERADA")
    else:
        print(f"   ❌ Variabilidad ALTA")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'text': text,
        'num_seeds': len(architectures),
        'iterations_per_seed': iterations,
        'architectures': architectures,
        'statistics': {
            'phi_inter_seed_variance': float(inter_seed_var),
            'phi_inter_seed_std': float(inter_seed_std),
            'phi_intra_seed_variance': float(intra_seed_var),
            'phi_mean': float(np.mean(phi_means)),
            'phi_range': [float(min(phi_means)), float(max(phi_means))],
            'coherence_mean': float(np.mean(coherence_means)),
            'signal_to_noise': float(snr),
            'coefficient_of_variation': float(cv),
            'is_deterministic': bool(is_deterministic)  # Convertir numpy.bool_ a Python bool
        },
        'timestamp': timestamp
    }
    
    # Guardar en carpeta de resultados organizados
    os.makedirs('results/reproducibility', exist_ok=True)
    filename = f"results/reproducibility/reproducibility_test_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {filename}")
    
    # Conclusión final
    print("\n" + "=" * 80)
    print("📋 CONCLUSIÓN")
    print("=" * 80)
    
    if is_deterministic:
        print("\n🎯 El sistema habla un LENGUAJE CAUSAL DETERMINISTA")
        print("   ✅ Genera la misma 'palabra causal' para el mismo input")
        print("   ✅ Podemos confiar en la consistencia de las arquitecturas")
    else:
        print("\n⚠️  El lenguaje causal tiene VARIABILIDAD INTRÍNSECA")
        print("   ⚠️  Se requiere promediado para estabilidad")
    
    print(f"\n   Variabilidad: {cv:.1f}% (coeficiente de variación)")
    print("=" * 80)


if __name__ == "__main__":
    main()
