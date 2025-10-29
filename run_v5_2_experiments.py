#!/usr/bin/env python3
"""
🧪 EXPERIMENTO COMPLETO: INFINITO V5.2 vs V5.1
============================================

Experimento comparativo que demuestra las mejoras de V5.2:
1. Memoria con priorización vs FIFO
2. Métricas estándar (perplexity)
3. Validación estadística rigurosa
4. Comparación contra baselines

Objetivo: Demostrar que V5.2 mantiene/mejora performance con mejor arquitectura.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from datetime import datetime
import json

# Import V5.2
from infinito_v5_2_refactored import InfinitoV52Refactored

# Import módulos de validación
from core import (
    StandardNLPMetrics,
    StatisticalTests,
    BenchmarkComparison,
    BaselineMetrics
)


def generate_text_data(vocab_size=1000, num_samples=20, seq_len=15):
    """Genera datos de texto sintéticos para el experimento."""
    print("📊 Generando datos de experimento...")
    
    # Crear secuencias con cierta estructura (no completamente random)
    data = []
    for i in range(num_samples):
        # Secuencia con patrón: inicio, medio, fin
        seq = torch.randint(0, vocab_size // 3, (seq_len,))
        seq[seq_len//3:2*seq_len//3] += vocab_size // 3
        seq[2*seq_len//3:] += 2 * vocab_size // 3
        data.append(seq)
    
    return torch.stack(data)


def experiment_memory_prioritization():
    """Experimento 1: Comparar memoria con priorización vs sin priorización."""
    
    print("\n" + "="*70)
    print("🧪 EXPERIMENTO 1: Memoria con Priorización")
    print("="*70)
    
    vocab_size = 1000
    batch_size = 4
    seq_len = 15
    
    # Generar datos
    data = generate_text_data(vocab_size, num_samples=10, seq_len=seq_len)
    
    print("\n📋 Configuración:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Secuencias: {len(data)}")
    print(f"  Longitud: {seq_len}")
    
    # Modelo CON priorización
    print("\n🔹 Modelo V5.2 CON priorización...")
    model_improved = InfinitoV52Refactored(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=3,
        use_improved_memory=True,
        use_stochastic_exploration=True
    )
    model_improved.eval()
    
    # Modelo SIN priorización (legacy)
    print("\n🔸 Modelo V5.2 SIN priorización (legacy)...")
    model_legacy = InfinitoV52Refactored(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=3,
        use_improved_memory=False,
        use_stochastic_exploration=False
    )
    model_legacy.eval()
    
    # Procesar varias secuencias para llenar memoria
    print("\n🔄 Procesando secuencias...")
    
    results_improved = []
    results_legacy = []
    
    with torch.no_grad():
        for i, sequence in enumerate(data):
            input_ids = sequence.unsqueeze(0)  # [1, seq_len]
            
            # Modelo mejorado
            _, metrics_improved = model_improved(input_ids, return_metrics=True)
            results_improved.append(metrics_improved)
            
            # Modelo legacy
            _, metrics_legacy = model_legacy(input_ids, return_metrics=True)
            results_legacy.append(metrics_legacy)
    
    # Comparar estadísticas de memoria
    print("\n📊 Resultados:")
    
    stats_improved = model_improved.get_memory_statistics()
    stats_legacy = model_legacy.get_memory_statistics()
    
    print("\n🔹 Memoria CON priorización:")
    print(f"  Utilización: {stats_improved['utilization']:.2%}")
    print(f"  Slots ocupados: {stats_improved['occupied_slots']}")
    print(f"  Importancia promedio: {stats_improved['avg_importance']:.4f}")
    print(f"  Edad máxima: {stats_improved['max_age']:.0f}")
    
    print("\n🔸 Memoria SIN priorización (legacy):")
    if 'note' in stats_legacy:
        print(f"  {stats_legacy['note']}")
    else:
        print(f"  Utilización: {stats_legacy.get('utilization', 0):.2%}")
    
    # Comparar métricas de integración
    avg_phi_improved = np.mean([m['integration_phi'] for m in results_improved])
    avg_phi_legacy = np.mean([m['integration_phi'] for m in results_legacy])
    
    print("\n📈 Métricas de integración promedio:")
    print(f"  CON priorización: Phi = {avg_phi_improved:.4f}")
    print(f"  SIN priorización: Phi = {avg_phi_legacy:.4f}")
    print(f"  Diferencia: {avg_phi_improved - avg_phi_legacy:+.4f}")
    
    return {
        'improved': results_improved,
        'legacy': results_legacy,
        'stats_improved': stats_improved,
        'stats_legacy': stats_legacy
    }


def experiment_standard_metrics():
    """Experimento 2: Calcular métricas estándar (perplexity)."""
    
    print("\n" + "="*70)
    print("🧪 EXPERIMENTO 2: Métricas Estándar NLP")
    print("="*70)
    
    vocab_size = 1000
    seq_len = 20
    num_samples = 50
    
    print("\n📋 Configuración:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Muestras: {num_samples}")
    print(f"  Longitud: {seq_len}")
    
    # Generar datos
    data = generate_text_data(vocab_size, num_samples, seq_len)
    
    # Crear targets (shifted)
    inputs = data[:, :-1]
    targets = data[:, 1:]
    
    # Modelo V5.2
    print("\n🚀 Inicializando modelo V5.2...")
    model = InfinitoV52Refactored(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=4,
        use_improved_memory=True
    )
    model.eval()
    
    # Calcular perplexity por lote
    print("\n🔄 Calculando perplexity...")
    
    perplexities = []
    batch_size = 5
    
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            perplexity = model.calculate_perplexity(batch_inputs, batch_targets)
            perplexities.append(perplexity)
    
    avg_perplexity = np.mean(perplexities)
    std_perplexity = np.std(perplexities)
    
    print("\n📊 Resultados:")
    print(f"  Perplexity promedio: {avg_perplexity:.2f}")
    print(f"  Desviación estándar: {std_perplexity:.2f}")
    print(f"  Rango: [{min(perplexities):.2f}, {max(perplexities):.2f}]")
    
    # Comparar contra baseline random
    baseline_perplexity = vocab_size  # Random model
    
    print(f"\n🎯 Comparación con baseline:")
    print(f"  Random baseline: ~{baseline_perplexity:.0f}")
    print(f"  V5.2 modelo: {avg_perplexity:.2f}")
    
    if avg_perplexity < baseline_perplexity * 0.5:
        print(f"  ✅ Modelo aprende (50% mejor que random)")
    else:
        print(f"  ⚠️  Modelo necesita más entrenamiento")
    
    return {
        'avg_perplexity': avg_perplexity,
        'std_perplexity': std_perplexity,
        'perplexities': perplexities,
        'baseline': baseline_perplexity
    }


def experiment_reproducibility():
    """Experimento 3: Test de reproducibilidad con estadística rigurosa."""
    
    print("\n" + "="*70)
    print("🧪 EXPERIMENTO 3: Reproducibilidad (Estadística Rigurosa)")
    print("="*70)
    
    vocab_size = 500
    seq_len = 10
    num_runs = 10
    
    print("\n📋 Configuración:")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Runs por grupo: {num_runs}")
    
    # Mismo input para todos los runs
    test_input = torch.randint(0, vocab_size, (1, seq_len))
    
    def run_model_with_seed(seed):
        """Ejecuta modelo con seed específico."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = InfinitoV52Refactored(
            vocab_size=vocab_size,
            hidden_dim=128,
            num_layers=2,
            use_improved_memory=True
        )
        model.eval()
        
        with torch.no_grad():
            _, metrics = model(test_input, return_metrics=True)
        
        return metrics['integration_phi']
    
    print("\n🔄 Ejecutando Grupo A (seeds 1-10)...")
    group_a = [run_model_with_seed(seed) for seed in range(1, num_runs + 1)]
    
    print("🔄 Ejecutando Grupo B (seeds 11-20)...")
    group_b = [run_model_with_seed(seed) for seed in range(11, 11 + num_runs)]
    
    # Convertir a numpy
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    
    print("\n📊 Resultados:")
    print(f"  Grupo A: mean={group_a.mean():.4f}, std={group_a.std():.4f}")
    print(f"  Grupo B: mean={group_b.mean():.4f}, std={group_b.std():.4f}")
    
    # Test estadístico RIGUROSO
    print("\n📈 Test Estadístico (T-Test):")
    
    test_results = StatisticalTests.test_reproducibility(
        group_a,
        group_b,
        alpha=0.05
    )
    
    print(f"  T-statistic: {test_results['t_statistic']:.4f}")
    print(f"  P-value: {test_results['p_value']:.4f}")
    print(f"  Cohen's d: {test_results['cohens_d']:.4f} ({test_results['effect_size_interpretation']})")
    
    if test_results['is_reproducible']:
        print(f"\n  ✅ REPRODUCIBLE (p > 0.05)")
        print(f"     No hay diferencia significativa entre seeds")
    else:
        print(f"\n  ⚠️  NO REPRODUCIBLE (p < 0.05)")
        print(f"     Hay variabilidad significativa entre seeds")
    
    return test_results


def experiment_benchmark_comparison():
    """Experimento 4: Comparación contra baselines."""
    
    print("\n" + "="*70)
    print("🧪 EXPERIMENTO 4: Comparación con Baselines")
    print("="*70)
    
    vocab_size = 1000
    seq_len = 15
    
    # Generar datos de prueba
    test_data = generate_text_data(vocab_size, 20, seq_len)
    inputs = test_data[:, :-1]
    targets = test_data[:, 1:]
    
    # Modelo V5.2
    print("\n🚀 Evaluando V5.2...")
    model = InfinitoV52Refactored(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=4,
        use_improved_memory=True
    )
    model.eval()
    
    # Calcular métricas
    perplexities = []
    phi_values = []
    
    with torch.no_grad():
        for i in range(len(inputs)):
            input_seq = inputs[i:i+1]
            target_seq = targets[i:i+1]
            
            # Perplexity
            perplexity = model.calculate_perplexity(input_seq, target_seq)
            perplexities.append(perplexity)
            
            # Phi
            _, metrics = model(input_seq, return_metrics=True)
            phi_values.append(metrics['integration_phi'])
    
    avg_perplexity = np.mean(perplexities)
    avg_phi = np.mean(phi_values)
    
    # Crear benchmark
    benchmark = BenchmarkComparison(['perplexity', 'phi_estimate'])
    
    # Baseline Random
    benchmark.add_random_baseline({
        'perplexity': vocab_size,
        'phi_estimate': 1.5
    })
    
    # Baseline "Pre-trained" (hipotético)
    benchmark.add_baseline({
        'perplexity': 150.0,  # Típico de modelo entrenado
        'phi_estimate': 3.0
    })
    
    # V5.2
    benchmark.add_current_model({
        'perplexity': avg_perplexity,
        'phi_estimate': avg_phi
    })
    
    print("\n📊 Reporte de Comparación:")
    print(benchmark.generate_comparison_report())
    
    return {
        'perplexity': avg_perplexity,
        'phi': avg_phi,
        'benchmark': benchmark
    }


def save_results(results, filename=None):
    """Guarda resultados del experimento."""
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/experiments/v5.2_experiment_{timestamp}.json"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convertir a formato serializable
    def make_serializable(obj):
        """Convierte objetos numpy/torch a tipos Python nativos."""
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items() 
                    if not k.startswith('_') and not callable(v)}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        elif isinstance(obj, (bool, int, float, str)):
            return obj
        else:
            return str(obj)
    
    serializable_results = make_serializable(results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {filename}")


def main():
    """Ejecuta suite completa de experimentos."""
    
    print("\n" + "="*70)
    print("🚀 SUITE DE EXPERIMENTOS: INFINITO V5.2")
    print("="*70)
    print()
    print("Experimentos a ejecutar:")
    print("  1. Memoria con priorización vs FIFO")
    print("  2. Métricas estándar (perplexity)")
    print("  3. Reproducibilidad con test estadístico")
    print("  4. Comparación con baselines")
    print()
    input("Presiona Enter para comenzar...")
    
    all_results = {}
    
    # Experimento 1
    exp1_results = experiment_memory_prioritization()
    all_results['memory_prioritization'] = {
        'avg_phi_improved': np.mean([m['integration_phi'] for m in exp1_results['improved']]),
        'avg_phi_legacy': np.mean([m['integration_phi'] for m in exp1_results['legacy']]),
        'memory_stats_improved': exp1_results['stats_improved']
    }
    
    # Experimento 2
    exp2_results = experiment_standard_metrics()
    all_results['standard_metrics'] = exp2_results
    
    # Experimento 3
    exp3_results = experiment_reproducibility()
    all_results['reproducibility'] = exp3_results
    
    # Experimento 4
    exp4_results = experiment_benchmark_comparison()
    all_results['benchmark'] = {
        'perplexity': exp4_results['perplexity'],
        'phi': exp4_results['phi']
    }
    
    # Guardar resultados
    save_results(all_results)
    
    # Resumen final
    print("\n" + "="*70)
    print("✅ SUITE DE EXPERIMENTOS COMPLETADA")
    print("="*70)
    
    print("\n📊 RESUMEN DE RESULTADOS:")
    
    print("\n🔹 Memoria con Priorización:")
    print(f"  Mejora en Phi: {all_results['memory_prioritization']['avg_phi_improved'] - all_results['memory_prioritization']['avg_phi_legacy']:+.4f}")
    
    print("\n📏 Métricas Estándar:")
    print(f"  Perplexity: {all_results['standard_metrics']['avg_perplexity']:.2f}")
    print(f"  vs Random: {all_results['standard_metrics']['baseline']:.0f}")
    
    print("\n📈 Reproducibilidad:")
    if all_results['reproducibility']['is_reproducible']:
        print(f"  ✅ REPRODUCIBLE (p={all_results['reproducibility']['p_value']:.4f})")
    else:
        print(f"  ⚠️  Variabilidad detectada (p={all_results['reproducibility']['p_value']:.4f})")
    
    print("\n🎯 Benchmark:")
    print(f"  Perplexity V5.2: {all_results['benchmark']['perplexity']:.2f}")
    print(f"  Phi V5.2: {all_results['benchmark']['phi']:.4f}")
    
    print("\n" + "="*70)
    print("🎉 ¡Experimentos completados exitosamente!")
    print("="*70)


if __name__ == '__main__':
    main()
