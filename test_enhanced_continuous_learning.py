#!/usr/bin/env python3
"""
🧪 TEST: Validación de Captura Temprana
========================================

Valida que la captura temprana soluciona el problema de saturación
y genera patrones diferenciados para textos diferentes.

Tests:
1. Patrones diferentes para textos diferentes
2. Patrones similares para textos similares
3. Similitud dentro de rangos esperados
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough
from continuous_learning_enhanced import (
    EnhancedContinuousLearningServer,
    EnhancedPhiPatternExtractor
)


def test_early_capture_differentiation():
    """
    TEST PRINCIPAL: Captura temprana genera patrones diferentes
    """
    print("\n" + "="*70)
    print("🧪 TEST: Diferenciación con Captura Temprana")
    print("="*70)
    
    # Setup
    class Args:
        def __init__(self):
            self.input_dim = 257
            self.batch_size = 4
            self.hidden_dim = 512
            self.attention_heads = 8
            self.memory_slots = 256
            self.lr = 1e-3
            self.seed = 42
            self.input_text = None
            self.text_mode = True
            self.max_iter = 1000
            self.comparative = False
            self.comparative_iterations = 100
            self.bootstrap_samples = 1000
    
    print("\n🚀 Inicializando INFINITO V5.1...")
    args = Args()
    infinito = InfinitoV51ConsciousnessBreakthrough(args)
    
    # Crear servidor MEJORADO
    print("🔄 Creando servidor con captura temprana (iter 8)...")
    server = EnhancedContinuousLearningServer(
        infinito,
        similarity_threshold=0.85,
        capture_iteration=8
    )
    
    # Casos de prueba
    test_cases = [
        # (texto1, texto2, similitud_esperada_min, similitud_esperada_max, descripción)
        ("mi perro es rojo", "mi perro es rojo", 0.98, 1.00, "Idénticos"),
        ("mi perro es rojo", "mi perro es verde", 0.85, 0.95, "Muy similares"),
        ("mi perro es rojo", "mi gato es azul", 0.65, 0.85, "Estructura similar, diferente tema"),
        ("mi perro es rojo", "yo pienso luego existo", 0.00, 0.70, "Totalmente diferentes"),
    ]
    
    print(f"\n📋 Ejecutando {len(test_cases)} casos de prueba...")
    print(f"{'='*70}\n")
    
    results = []
    
    for i, (text1, text2, sim_min, sim_max, desc) in enumerate(test_cases, 1):
        print(f"{'─'*70}")
        print(f"CASO {i}: {desc}")
        print(f"{'─'*70}")
        print(f"   Texto 1: '{text1}'")
        print(f"   Texto 2: '{text2}'")
        print(f"   Similitud esperada: {sim_min:.0%} - {sim_max:.0%}")
        
        # Procesar ambos textos
        result1 = server.process_input(text1, verbose=False)
        result2 = server.process_input(text2, verbose=False)
        
        # Extraer patrones
        pattern1 = np.array(result1['causal_pattern'])
        pattern2 = np.array(result2['causal_pattern'])
        
        # Calcular similitud
        similarity = np.dot(pattern1, pattern2) / (
            np.linalg.norm(pattern1) * np.linalg.norm(pattern2) + 1e-8
        )
        
        # Validar
        passed = sim_min <= similarity <= sim_max
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"\n   📊 Resultados:")
        print(f"      Similitud obtenida: {similarity:.1%}")
        print(f"      Φ₁ = {result1['phi']:.4f}, Φ₂ = {result2['phi']:.4f}")
        print(f"      C₁ = {result1['consciousness']:.4f}, C₂ = {result2['consciousness']:.4f}")
        print(f"      Status: {status}")
        
        # Analizar diferencias en features
        diff = np.abs(pattern1 - pattern2)
        max_diff_idx = np.argmax(diff)
        
        if len(pattern1) >= 14:
            feature_names = server.pattern_extractor.feature_names
            print(f"\n   🔍 Feature más diferente:")
            print(f"      {feature_names[max_diff_idx]}: {diff[max_diff_idx]:.3f}")
        
        results.append({
            'case': desc,
            'similarity': similarity,
            'expected_min': sim_min,
            'expected_max': sim_max,
            'passed': passed
        })
        
        print()
    
    # Resumen
    print(f"{'='*70}")
    print(f"📊 RESUMEN DE TESTS")
    print(f"{'='*70}")
    
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['passed'] else "❌"
        print(f"   {status} Caso {i}: {result['case']}")
        print(f"      Similitud: {result['similarity']:.1%} (esperado: {result['expected_min']:.0%}-{result['expected_max']:.0%})")
    
    print(f"\n   Total: {passed}/{total} tests pasados")
    
    if passed == total:
        print(f"\n🎉 ¡TODOS LOS TESTS PASARON!")
        print(f"✅ La captura temprana SOLUCIONA el problema de saturación")
    else:
        print(f"\n⚠️  {total - passed} test(s) fallaron")
        print(f"💡 Considera ajustar:")
        print(f"   - capture_iteration (actual: {server.capture_iteration})")
        print(f"   - similarity_threshold (actual: {server.memory_bank.similarity_threshold})")
    
    print(f"{'='*70}")
    
    return passed == total


def test_pattern_features():
    """
    TEST: Validar que features enriquecidas se calculan correctamente
    """
    print("\n" + "="*70)
    print("🧪 TEST: Features Enriquecidas (14 dimensiones)")
    print("="*70)
    
    # Setup
    class Args:
        def __init__(self):
            self.input_dim = 257
            self.batch_size = 4
            self.hidden_dim = 512
            self.attention_heads = 8
            self.memory_slots = 256
            self.lr = 1e-3
            self.seed = 42
            self.input_text = None
            self.text_mode = True
            self.max_iter = 1000
            self.comparative = False
            self.comparative_iterations = 100
            self.bootstrap_samples = 1000
    
    infinito = InfinitoV51ConsciousnessBreakthrough(Args())
    server = EnhancedContinuousLearningServer(infinito)
    
    # Procesar un texto
    text = "mi perro es rojo"
    result = server.process_input(text, verbose=False)
    
    pattern = np.array(result['causal_pattern'])
    
    print(f"\n📊 Patrón extraído para '{text}':")
    print(f"   Dimensiones: {len(pattern)}")
    print(f"   Valores: {pattern[:5]}... (primeros 5)")
    
    # Validar
    assert len(pattern) == 14, f"Patrón debería tener 14 dimensiones, tiene {len(pattern)}"
    
    # Verificar que no todos son iguales
    unique_values = len(np.unique(pattern.round(decimals=2)))
    print(f"   Valores únicos (redondeados): {unique_values}/14")
    
    assert unique_values > 5, "Patrón tiene muy poca variabilidad"
    
    # Mostrar features con nombres
    print(f"\n📋 Features detalladas:")
    print(server.pattern_extractor.pattern_to_string(pattern))
    
    print(f"\n✅ Test de features enriquecidas PASADO")
    return True


def test_memory_persistence():
    """
    TEST: Validar que memoria persiste correctamente
    """
    print("\n" + "="*70)
    print("🧪 TEST: Persistencia de Memoria")
    print("="*70)
    
    # Setup
    class Args:
        def __init__(self):
            self.input_dim = 257
            self.batch_size = 4
            self.hidden_dim = 512
            self.attention_heads = 8
            self.memory_slots = 256
            self.lr = 1e-3
            self.seed = 42
            self.input_text = None
            self.text_mode = True
            self.max_iter = 1000
            self.comparative = False
            self.comparative_iterations = 100
            self.bootstrap_samples = 1000
    
    infinito = InfinitoV51ConsciousnessBreakthrough(Args())
    server = EnhancedContinuousLearningServer(infinito)
    
    # Procesar algunos textos
    texts = ["mi perro es rojo", "yo pienso luego existo"]
    
    for text in texts:
        server.process_input(text, verbose=False)
    
    # Guardar
    test_file = 'test_enhanced_memory.json'
    server.memory_bank.save_to_disk(test_file)
    
    # Crear nuevo servidor y cargar
    infinito2 = InfinitoV51ConsciousnessBreakthrough(Args())
    server2 = EnhancedContinuousLearningServer(infinito2)
    server2.memory_bank.load_from_disk(test_file)
    
    # Validar
    assert len(server2.memory_bank.patterns) == len(server.memory_bank.patterns)
    
    print(f"\n✅ Memoria guardada y cargada correctamente")
    print(f"   Patrones guardados: {len(server.memory_bank.patterns)}")
    print(f"   Patrones cargados: {len(server2.memory_bank.patterns)}")
    
    # Limpiar
    if os.path.exists(test_file):
        os.remove(test_file)
    
    return True


def run_all_enhanced_tests():
    """Ejecutar todos los tests de la versión mejorada"""
    print("\n" + "="*70)
    print("🧪 SUITE DE TESTS - Continuous Learning Enhanced")
    print("="*70)
    
    tests = [
        ("Diferenciación con Captura Temprana", test_early_capture_differentiation),
        ("Features Enriquecidas", test_pattern_features),
        ("Persistencia de Memoria", test_memory_persistence),
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n{'#'*70}")
        print(f"# TEST: {name}")
        print(f"{'#'*70}")
        
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Error en {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"📋 RESUMEN FINAL")
    print(f"{'='*70}")
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print(f"\n   Total: {passed}/{total} tests pasados ({passed/total*100:.0f}%)")
    
    if passed == total:
        print(f"\n🎉 ¡ÉXITO TOTAL!")
        print(f"✅ La versión mejorada soluciona el problema de saturación")
        print(f"✅ Los patrones son diferenciados y estables")
    else:
        print(f"\n⚠️  Algunos tests fallaron - revisar configuración")
    
    print(f"{'='*70}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_enhanced_tests()
    sys.exit(0 if success else 1)
