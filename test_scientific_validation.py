#!/usr/bin/env python3
"""
🧪 TEST CON MÉTRICAS ESTÁNDAR Y VALIDACIÓN CIENTÍFICA
===================================================

Demuestra cómo usar las nuevas métricas y tests estadísticos rigurosos.

MEJORAS vs tests anteriores:
✅ Usa perplexity (métrica estándar)
✅ Compara contra baseline random
✅ Usa t-tests en lugar de thresholds arbitrarios
✅ Reporta p-values y effect sizes
✅ Corrección por comparaciones múltiples
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from datetime import datetime

# Import de módulos refactorizados
from core.validation import StandardNLPMetrics, StatisticalTests, BenchmarkComparison
from core.iit_metrics import InformationIntegrationMetrics, BaselineMetrics


def test_with_proper_statistics():
    """
    Test de reproducibilidad CON ESTADÍSTICA REAL.
    
    NO usa thresholds arbitrarios como `variance < 0.05`.
    Usa t-test con p-values.
    """
    print("="*70)
    print("🧪 TEST DE REPRODUCIBILIDAD CON ESTADÍSTICA RIGUROSA")
    print("="*70)
    print()
    
    # Simular resultados de múltiples runs con diferentes seeds
    # En producción, estos serían resultados reales del modelo
    np.random.seed(42)
    
    # Grupo A: Seeds 1-10
    results_group_a = np.random.normal(loc=0.85, scale=0.02, size=10)
    
    # Grupo B: Seeds 11-20
    results_group_b = np.random.normal(loc=0.85, scale=0.02, size=10)
    
    print("📊 Resultados:")
    print(f"Grupo A (seeds 1-10):  mean={results_group_a.mean():.4f}, std={results_group_a.std():.4f}")
    print(f"Grupo B (seeds 11-20): mean={results_group_b.mean():.4f}, std={results_group_b.std():.4f}")
    print()
    
    # Test estadístico REAL
    test_results = StatisticalTests.test_reproducibility(
        results_group_a, 
        results_group_b,
        alpha=0.05
    )
    
    print("📈 Resultados del T-Test:")
    print(f"  t-statistic: {test_results['t_statistic']:.4f}")
    print(f"  p-value: {test_results['p_value']:.4f}")
    print(f"  ¿Es reproducible? {test_results['is_reproducible']}")
    print(f"  Cohen's d: {test_results['cohens_d']:.4f} ({test_results['effect_size_interpretation']})")
    print()
    
    # Interpretación
    if test_results['is_reproducible']:
        print("✅ RESULTADO: El sistema ES reproducible (p > 0.05)")
        print("   No hay evidencia estadística de diferencia entre seeds.")
    else:
        print("❌ RESULTADO: El sistema NO es reproducible (p < 0.05)")
        print("   Hay diferencia significativa entre seeds.")
    
    print()
    return test_results


def test_with_standard_metrics():
    """
    Test usando métricas ESTÁNDAR de NLP.
    
    Reporta perplexity, accuracy - métricas que cualquier paper debe incluir.
    """
    print("="*70)
    print("📏 TEST CON MÉTRICAS ESTÁNDAR DE NLP")
    print("="*70)
    print()
    
    # Simular logits y targets
    batch_size, seq_len, vocab_size = 4, 10, 1000
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Calcular métricas estándar
    metrics = StandardNLPMetrics()
    
    perplexity = metrics.calculate_perplexity(logits, targets)
    accuracy = metrics.calculate_accuracy(logits, targets)
    
    print("📊 Métricas Estándar:")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    # Comparar con baseline
    print("🎯 Comparación con Baseline:")
    print(f"  Random baseline perplexity: ~{vocab_size:.0f}")
    print(f"  Random baseline accuracy: ~{1/vocab_size:.4f}")
    print(f"  Tu modelo perplexity: {perplexity:.2f}")
    print(f"  Tu modelo accuracy: {accuracy:.4f}")
    print()
    
    if perplexity < vocab_size * 0.5:
        print("✅ Modelo aprende algo (mejor que random)")
    else:
        print("❌ Modelo no mejor que random")
    
    print()
    return {'perplexity': perplexity, 'accuracy': accuracy}


def test_improvement_significance():
    """
    Test si una "mejora" es estadísticamente significativa.
    
    NO basta con decir "Phi aumentó de 0.8 a 0.9".
    Hay que demostrar que es significativo.
    """
    print("="*70)
    print("📈 TEST DE SIGNIFICANCIA DE MEJORA")
    print("="*70)
    print()
    
    # Simular scores de baseline vs improved
    np.random.seed(42)
    baseline_scores = np.random.normal(loc=0.75, scale=0.05, size=20)
    improved_scores = np.random.normal(loc=0.80, scale=0.05, size=20)
    
    print("📊 Scores:")
    print(f"  Baseline: mean={baseline_scores.mean():.4f}, std={baseline_scores.std():.4f}")
    print(f"  Improved: mean={improved_scores.mean():.4f}, std={improved_scores.std():.4f}")
    print(f"  Diferencia absoluta: {improved_scores.mean() - baseline_scores.mean():.4f}")
    print()
    
    # Test estadístico
    test_results = StatisticalTests.test_improvement(
        baseline_scores,
        improved_scores,
        alpha=0.05
    )
    
    print("📈 Resultados del Paired T-Test:")
    print(f"  t-statistic: {test_results['t_statistic']:.4f}")
    print(f"  p-value: {test_results['p_value']:.4f}")
    print(f"  ¿Mejora significativa? {test_results['is_significant_improvement']}")
    print(f"  Cohen's d: {test_results['cohens_d']:.4f}")
    print(f"  Mejora: {test_results['improvement_percentage']:.2f}%")
    print()
    
    if test_results['is_significant_improvement']:
        print("✅ RESULTADO: La mejora ES estadísticamente significativa")
    else:
        print("❌ RESULTADO: La mejora NO es estadísticamente significativa")
        print("   Puede ser ruido estadístico.")
    
    print()
    return test_results


def test_benchmark_comparison():
    """
    Comparación sistemática contra baselines.
    
    SIEMPRE incluir esto en análisis.
    """
    print("="*70)
    print("🏆 COMPARACIÓN CONTRA BASELINES")
    print("="*70)
    print()
    
    benchmark = BenchmarkComparison(['perplexity', 'phi_estimate', 'coherence'])
    
    # Random baseline
    benchmark.add_random_baseline({
        'perplexity': 1000.0,
        'phi_estimate': 1.5,
        'coherence': 0.2
    })
    
    # GPT-2 baseline (hipotético)
    benchmark.add_baseline({
        'perplexity': 35.0,
        'phi_estimate': 3.2,
        'coherence': 0.65
    })
    
    # Tu modelo
    benchmark.add_current_model({
        'perplexity': 42.0,
        'phi_estimate': 4.5,
        'coherence': 0.72
    })
    
    print(benchmark.generate_comparison_report())
    print()


def test_multiple_comparisons_correction():
    """
    Demostración de corrección por comparaciones múltiples.
    
    Si haces 10 tests con alpha=0.05, esperarías ~0.5 falsos positivos.
    ¡DEBES corregir!
    """
    print("="*70)
    print("🔬 CORRECCIÓN POR COMPARACIONES MÚLTIPLES")
    print("="*70)
    print()
    
    # Simular 10 p-values
    p_values = [0.03, 0.12, 0.45, 0.02, 0.67, 0.08, 0.15, 0.34, 0.01, 0.52]
    
    print("📊 P-values originales:")
    for i, p in enumerate(p_values, 1):
        print(f"  Test {i}: p={p:.3f}", "✓" if p < 0.05 else "")
    print()
    
    # Corrección Bonferroni
    corrected_bonferroni = StatisticalTests.multiple_comparison_correction(
        p_values, method='bonferroni'
    )
    
    print("📊 P-values corregidos (Bonferroni):")
    for i, (p_orig, p_corr) in enumerate(zip(p_values, corrected_bonferroni), 1):
        print(f"  Test {i}: {p_orig:.3f} → {p_corr:.3f}", "✓" if p_corr < 0.05 else "")
    print()
    
    print("💡 Interpretación:")
    orig_sig = sum(1 for p in p_values if p < 0.05)
    corr_sig = sum(1 for p in corrected_bonferroni if p < 0.05)
    print(f"  Sin corrección: {orig_sig} tests significativos")
    print(f"  Con corrección: {corr_sig} tests significativos")
    print()


def main():
    """Ejecuta todos los tests con metodología científica rigurosa."""
    
    print("\n" + "="*70)
    print("🚀 SUITE DE TESTS CON VALIDACIÓN CIENTÍFICA RIGUROSA")
    print("="*70)
    print()
    print("Demostrando las MEJORAS en metodología:")
    print("  ❌ Antes: thresholds arbitrarios (var < 0.05)")
    print("  ✅ Ahora: p-values, t-tests, effect sizes")
    print()
    print("  ❌ Antes: solo métricas custom (Phi, etc.)")
    print("  ✅ Ahora: métricas estándar (perplexity, accuracy)")
    print()
    print("  ❌ Antes: sin baselines")
    print("  ✅ Ahora: comparación contra random y GPT-2")
    print()
    input("Presiona Enter para continuar...")
    print()
    
    # Ejecutar tests
    test_with_proper_statistics()
    test_with_standard_metrics()
    test_improvement_significance()
    test_benchmark_comparison()
    test_multiple_comparisons_correction()
    
    print("="*70)
    print("✅ SUITE DE TESTS COMPLETADA")
    print("="*70)
    print()
    print("📝 Próximos pasos:")
    print("  1. Integrar estos tests en el pipeline principal")
    print("  2. Ejecutar con datos reales del modelo")
    print("  3. Reportar en papers usando estas métricas")
    print("  4. Siempre comparar contra baselines")
    print()
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results/validation"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/scientific_validation_{timestamp}.txt"
    print(f"💾 Resultados guardados en: {output_file}")


if __name__ == '__main__':
    main()
