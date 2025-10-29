#!/usr/bin/env python3
"""
🔬 TEST DE REPRODUCIBILIDAD MEJORADA
====================================

Valida que el parámetro 'seed' en InfinitoV52Refactored 
mejore la reproducibilidad (p-value > 0.05).

Resultado esperado:
- Antes (sin seed): p-value = 0.039 ❌
- Después (con seed): p-value > 0.05 ✅
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from infinito_v5_2_refactored import InfinitoV52Refactored
from core import StatisticalTests


def test_reproducibility_with_seed():
    """Test con seed fijado - debe ser reproducible."""
    
    print("\n" + "="*70)
    print("🔬 TEST: Reproducibilidad CON seed fijado")
    print("="*70)
    
    vocab_size = 500
    seq_len = 10
    num_runs = 10
    seed = 42  # 🔒 Seed fijado
    
    # Mismo input para todos los runs
    test_input = torch.randint(0, vocab_size, (1, seq_len))
    
    def run_model_with_fixed_seed(run_number):
        """Ejecuta modelo CON seed fijado."""
        model = InfinitoV52Refactored(
            vocab_size=vocab_size,
            hidden_dim=128,
            num_layers=2,
            use_improved_memory=True,
            seed=seed  # 🔒 Mismo seed para todos
        )
        model.eval()
        
        with torch.no_grad():
            _, metrics = model(test_input, return_metrics=True)
        
        return metrics['integration_phi']
    
    print(f"\n🔄 Ejecutando Grupo A (todos con seed={seed})...")
    group_a = [run_model_with_fixed_seed(i) for i in range(1, num_runs + 1)]
    
    print(f"🔄 Ejecutando Grupo B (todos con seed={seed})...")
    group_b = [run_model_with_fixed_seed(i) for i in range(11, 11 + num_runs)]
    
    # Convertir a numpy
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    
    print("\n📊 Resultados:")
    print(f"  Grupo A: mean={group_a.mean():.4f}, std={group_a.std():.6f}")
    print(f"  Grupo B: mean={group_b.mean():.4f}, std={group_b.std():.6f}")
    
    # Test estadístico
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
        print(f"     Seed fijado funciona correctamente!")
    else:
        print(f"\n  ⚠️  NO REPRODUCIBLE (p < 0.05)")
        print(f"     Puede haber variabilidad residual")
    
    return test_results


def test_reproducibility_without_seed():
    """Test SIN seed fijado - debe tener variabilidad."""
    
    print("\n" + "="*70)
    print("🔬 TEST: Reproducibilidad SIN seed fijado (control)")
    print("="*70)
    
    vocab_size = 500
    seq_len = 10
    num_runs = 10
    
    # Mismo input para todos los runs
    test_input = torch.randint(0, vocab_size, (1, seq_len))
    
    def run_model_without_seed(run_number):
        """Ejecuta modelo SIN seed fijado."""
        # Variar el seed implícitamente
        torch.manual_seed(run_number)
        np.random.seed(run_number)
        
        model = InfinitoV52Refactored(
            vocab_size=vocab_size,
            hidden_dim=128,
            num_layers=2,
            use_improved_memory=True,
            seed=None  # ⚠️ Sin seed fijado
        )
        model.eval()
        
        with torch.no_grad():
            _, metrics = model(test_input, return_metrics=True)
        
        return metrics['integration_phi']
    
    print(f"\n🔄 Ejecutando Grupo A (seeds variables 1-10)...")
    group_a = [run_model_without_seed(i) for i in range(1, num_runs + 1)]
    
    print(f"🔄 Ejecutando Grupo B (seeds variables 11-20)...")
    group_b = [run_model_without_seed(i) for i in range(11, 11 + num_runs)]
    
    # Convertir a numpy
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    
    print("\n📊 Resultados:")
    print(f"  Grupo A: mean={group_a.mean():.4f}, std={group_a.std():.6f}")
    print(f"  Grupo B: mean={group_b.mean():.4f}, std={group_b.std():.6f}")
    
    # Test estadístico
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
        print(f"\n  ⚠️  REPRODUCIBLE (p > 0.05)")
        print(f"     Sorpresivamente estable sin seed!")
    else:
        print(f"\n  ✅ NO REPRODUCIBLE (p < 0.05)")
        print(f"     Comportamiento esperado sin seed fijado")
    
    return test_results


def main():
    """Ejecuta ambos tests y compara."""
    
    print("\n" + "="*70)
    print("🧪 VALIDACIÓN DE REPRODUCIBILIDAD MEJORADA")
    print("="*70)
    print()
    print("Se ejecutarán 2 tests:")
    print("  1. CON seed fijado (debería ser reproducible)")
    print("  2. SIN seed fijado (control - puede variar)")
    print()
    
    # Test CON seed
    results_with_seed = test_reproducibility_with_seed()
    
    # Test SIN seed (control)
    results_without_seed = test_reproducibility_without_seed()
    
    # Comparación
    print("\n" + "="*70)
    print("📊 COMPARACIÓN DE RESULTADOS")
    print("="*70)
    
    print("\n🔒 CON seed fijado:")
    print(f"  P-value: {results_with_seed['p_value']:.4f}")
    print(f"  Reproducible: {'✅ SÍ' if results_with_seed['is_reproducible'] else '❌ NO'}")
    
    print("\n⚠️  SIN seed fijado (control):")
    print(f"  P-value: {results_without_seed['p_value']:.4f}")
    print(f"  Reproducible: {'✅ SÍ' if results_without_seed['is_reproducible'] else '❌ NO'}")
    
    # Conclusión
    print("\n" + "="*70)
    print("✅ CONCLUSIÓN")
    print("="*70)
    
    if results_with_seed['is_reproducible']:
        print("\n✅ ÉXITO: Seed fijado mejora la reproducibilidad")
        print(f"   P-value mejoró de ~0.039 a {results_with_seed['p_value']:.4f}")
        print("   Grupos A y B son estadísticamente indistinguibles")
    else:
        print("\n⚠️  PARCIAL: Aún hay variabilidad")
        print(f"   P-value actual: {results_with_seed['p_value']:.4f}")
        print("   Puede deberse a:")
        print("   - Dropout en capas")
        print("   - Ruido estocástico")
        print("   - Operaciones no determinísticas en GPU")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
