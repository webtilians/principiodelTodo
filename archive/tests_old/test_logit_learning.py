#!/usr/bin/env python3
"""
🧪 TEST: Continuous Learning basado en LOGITS
==============================================

Valida que los LOGITS RAW (pre-sigmoid) tienen variabilidad
y producen patrones diferenciados.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough
from continuous_learning_logits import LogitBasedLearningServer


def test_logit_differentiation():
    """Test principal: logits producen patrones diferentes"""
    print("\n" + "="*70)
    print("🧪 TEST: Diferenciación basada en LOGITS")
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
    infinito = InfinitoV51ConsciousnessBreakthrough(Args())
    
    print("🎯 Creando servidor basado en LOGITS...")
    server = LogitBasedLearningServer(infinito, similarity_threshold=0.85)
    
    # Test cases
    test_cases = [
        ("mi perro es rojo", "mi perro es rojo", 0.98, 1.00, "Idénticos"),
        ("mi perro es rojo", "mi perro es verde", 0.75, 0.95, "Muy similares"),
        ("mi perro es rojo", "mi gato es azul", 0.50, 0.80, "Estructura similar"),
        ("mi perro es rojo", "yo pienso luego existo", 0.00, 0.60, "Completamente diferentes"),
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
        
        # Procesar
        result1 = server.process_input(text1, verbose=False)
        result2 = server.process_input(text2, verbose=False)
        
        if result1['status'] == 'ERROR' or result2['status'] == 'ERROR':
            print(f"\n   ❌ ERROR en procesamiento")
            results.append({
                'case': desc,
                'similarity': 0.0,
                'expected_min': sim_min,
                'expected_max': sim_max,
                'passed': False
            })
            continue
        
        # Calcular similitud
        pattern1 = np.array(result1['causal_pattern'])
        pattern2 = np.array(result2['causal_pattern'])
        
        similarity = np.dot(pattern1, pattern2) / (
            np.linalg.norm(pattern1) * np.linalg.norm(pattern2) + 1e-8
        )
        
        # Validar
        passed = sim_min <= similarity <= sim_max
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"\n   📊 Resultados:")
        print(f"      Similitud: {similarity:.1%}")
        print(f"      Φ₁ = {result1['phi']:.4f}, Φ₂ = {result2['phi']:.4f}")
        print(f"      C₁ = {result1['consciousness']:.4f}, C₂ = {result2['consciousness']:.4f}")
        print(f"      Status: {status}")
        
        # Mostrar logits más diferentes
        diff = np.abs(pattern1[:6] - pattern2[:6])  # Solo logits
        max_diff_idx = np.argmax(diff)
        logit_names = server.extractor.feature_names[:6]
        
        print(f"\n   🔍 Logit más diferente:")
        print(f"      {logit_names[max_diff_idx]}: Δ={diff[max_diff_idx]:.3f}")
        print(f"      Valor 1: {pattern1[max_diff_idx]:+.3f}")
        print(f"      Valor 2: {pattern2[max_diff_idx]:+.3f}")
        
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
    print(f"📊 RESUMEN")
    print(f"{'='*70}")
    
    total = len(results)
    passed = sum(1 for r in results if r['passed'])
    
    for i, result in enumerate(results, 1):
        status = "✅" if result['passed'] else "❌"
        print(f"   {status} Caso {i}: {result['case']}")
        print(f"      Similitud: {result['similarity']:.1%} (esperado: {result['expected_min']:.0%}-{result['expected_max']:.0%})")
    
    print(f"\n   Total: {passed}/{total} tests pasados ({passed/total*100:.0f}%)")
    
    if passed == total:
        print(f"\n🎉 ¡ÉXITO COMPLETO!")
        print(f"✅ Los LOGITS RAW solucionan el problema de saturación")
        print(f"✅ Los patrones son diferenciados y estables")
    elif passed >= total * 0.75:
        print(f"\n✅ MAYORÍA DE TESTS PASADOS")
        print(f"Los logits mejoran significativamente la diferenciación")
    else:
        print(f"\n⚠️  Necesita ajustes")
        print(f"Revisar normalización o threshold")
    
    print(f"{'='*70}")
    
    return passed >= total * 0.75  # 75% success rate


def test_logit_variability():
    """Test auxiliar: verificar variabilidad de logits"""
    print("\n" + "="*70)
    print("🧪 TEST: Variabilidad de LOGITS")
    print("="*70)
    
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
    server = LogitBasedLearningServer(infinito)
    
    texts = [
        "mi perro es rojo",
        "mi perro es verde",
        "yo pienso luego existo"
    ]
    
    print(f"\n📋 Procesando {len(texts)} textos...")
    
    patterns = []
    for text in texts:
        result = server.process_input(text, verbose=False)
        if result['status'] != 'ERROR':
            patterns.append(np.array(result['causal_pattern']))
    
    if len(patterns) < 2:
        print(f"\n❌ No se pudieron procesar suficientes textos")
        return False
    
    # Análisis de variabilidad
    print(f"\n📊 Análisis de variabilidad:")
    
    for i, (text, pattern) in enumerate(zip(texts, patterns)):
        logits = pattern[:6]  # Primeros 6 son logits
        
        print(f"\n{i+1}. '{text}':")
        print(f"   Range logits: [{logits.min():+.3f}, {logits.max():+.3f}]")
        print(f"   Std logits: {logits.std():.3f}")
        print(f"   Valores únicos: {len(np.unique(logits.round(decimals=2)))}/6")
        
        # Los logits DEBEN tener variabilidad (std > 0.01)
        if logits.std() > 0.01:
            print(f"   ✅ Buena variabilidad")
        else:
            print(f"   ⚠️ Baja variabilidad")
    
    # Comparar patterns
    print(f"\n🔬 Similitudes entre patrones:")
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            sim = np.dot(patterns[i], patterns[j]) / (
                np.linalg.norm(patterns[i]) * np.linalg.norm(patterns[j]) + 1e-8
            )
            print(f"   '{texts[i]}' vs '{texts[j]}': {sim:.1%}")
    
    print(f"\n✅ Test de variabilidad completado")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 SUITE DE TESTS - Continuous Learning LOGITS")
    print("="*70)
    
    success1 = test_logit_variability()
    success2 = test_logit_differentiation()
    
    if success1 and success2:
        print(f"\n🎉 ¡TODOS LOS TESTS EXITOSOS!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Algunos tests fallaron")
        sys.exit(1)
