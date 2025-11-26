#!/usr/bin/env python3
"""
🧪 TEST: Continuous Learning basado en EMBEDDINGS
==================================================

Valida que los embeddings producen patrones diferenciados.
Esta es la SOLUCIÓN DEFINITIVA al problema de saturación.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough
from continuous_learning_embeddings import EmbeddingBasedLearningServer


def test_embedding_differentiation():
    """Test principal: embeddings producen patrones diferentes"""
    print("\n" + "="*70)
    print("🧪 TEST: Diferenciación basada en EMBEDDINGS")
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
    
    print("🎯 Creando servidor basado en EMBEDDINGS...")
    server = EmbeddingBasedLearningServer(infinito, similarity_threshold=0.70)
    
    # Test cases con rangos más realistas para embeddings
    test_cases = [
        ("mi perro es rojo", "mi perro es rojo", 0.98, 1.00, "Idénticos"),
        ("mi perro es rojo", "mi perro es verde", 0.70, 0.95, "Muy similares (cambia 1 palabra)"),
        ("mi perro es rojo", "mi gato es azul", 0.40, 0.75, "Estructura similar (cambian 2 palabras)"),
        ("mi perro es rojo", "yo pienso luego existo", 0.00, 0.40, "Completamente diferentes"),
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
        
        # Mostrar diferencias en componentes
        diff = np.abs(pattern1 - pattern2)
        max_diff_idx = np.argmax(diff)
        
        print(f"\n   🔍 Componente más diferente:")
        print(f"      Componente {max_diff_idx+1}: Δ={diff[max_diff_idx]:.3f}")
        print(f"      Valor 1: {pattern1[max_diff_idx]:.3f}")
        print(f"      Valor 2: {pattern2[max_diff_idx]:.3f}")
        
        # Estadísticas de variabilidad
        print(f"\n   📈 Variabilidad de patrones:")
        print(f"      Pattern 1 - std: {pattern1.std():.3f}, range: [{pattern1.min():.2f}, {pattern1.max():.2f}]")
        print(f"      Pattern 2 - std: {pattern2.std():.3f}, range: [{pattern2.min():.2f}, {pattern2.max():.2f}]")
        
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
    print(f"📊 RESUMEN FINAL")
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
        print(f"✅ Los EMBEDDINGS solucionan el problema de saturación")
        print(f"✅ Los patrones son únicos y diferenciados")
        print(f"✅ El sistema de aprendizaje continuo funciona correctamente")
    elif passed >= total * 0.75:
        print(f"\n✅ MAYORÍA DE TESTS PASADOS")
        print(f"Los embeddings mejoran significativamente la diferenciación")
        print(f"Ajustar threshold si es necesario")
    else:
        print(f"\n⚠️  Necesita ajustes")
        print(f"Revisar extracción de embeddings o threshold")
    
    print(f"{'='*70}")
    
    return passed == total


def test_pattern_uniqueness():
    """Test: Verificar que cada texto genera patrón único"""
    print("\n" + "="*70)
    print("🧪 TEST: Unicidad de Patrones")
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
    server = EmbeddingBasedLearningServer(infinito)
    
    texts = [
        "hola mundo",
        "adios mundo",
        "python es genial",
        "javascript es rapido",
        "inteligencia artificial",
    ]
    
    print(f"\n📋 Procesando {len(texts)} textos diferentes...")
    
    patterns = []
    for text in texts:
        result = server.process_input(text, verbose=False)
        if result['status'] != 'ERROR':
            patterns.append(np.array(result['causal_pattern']))
            print(f"   ✅ '{text}': pattern generado")
    
    # Verificar que todos son diferentes
    print(f"\n🔬 Verificando unicidad...")
    all_unique = True
    
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            sim = np.dot(patterns[i], patterns[j]) / (
                np.linalg.norm(patterns[i]) * np.linalg.norm(patterns[j]) + 1e-8
            )
            
            if sim > 0.95:  # Si son >95% similares, no son únicos
                print(f"   ⚠️ '{texts[i]}' y '{texts[j]}' son muy similares ({sim:.1%})")
                all_unique = False
            else:
                print(f"   ✅ '{texts[i]}' vs '{texts[j]}': {sim:.1%} (diferentes)")
    
    if all_unique:
        print(f"\n✅ Todos los patrones son únicos")
        return True
    else:
        print(f"\n⚠️ Algunos patrones son muy similares")
        return False


def test_same_text_recognition():
    """Test: El mismo texto debe ser reconocido"""
    print("\n" + "="*70)
    print("🧪 TEST: Reconocimiento de Texto Idéntico")
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
    server = EmbeddingBasedLearningServer(infinito, similarity_threshold=0.95)
    
    text = "el cielo es azul"
    
    print(f"\n📋 Procesando texto: '{text}'")
    
    # Primera vez - debe ser NEW
    result1 = server.process_input(text, verbose=False)
    print(f"\n   Primera vez:")
    print(f"      Status: {result1['status']}")
    print(f"      {result1['message']}")
    
    if result1['status'] != 'NEW':
        print(f"   ❌ FAIL: Primera vez debería ser NEW")
        return False
    
    # Segunda vez - debe ser RECOGNIZED
    result2 = server.process_input(text, verbose=False)
    print(f"\n   Segunda vez:")
    print(f"      Status: {result2['status']}")
    print(f"      {result2['message']}")
    
    if result2['status'] == 'RECOGNIZED':
        print(f"      Similitud: {result2['similarity']:.1%}")
        print(f"      Visto: {result2['seen_count']} veces")
        
        if result2['similarity'] > 0.95:
            print(f"\n   ✅ PASS: Texto reconocido correctamente")
            return True
        else:
            print(f"\n   ❌ FAIL: Similitud muy baja ({result2['similarity']:.1%})")
            return False
    else:
        print(f"\n   ❌ FAIL: Debería ser RECOGNIZED")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 SUITE DE TESTS - Continuous Learning EMBEDDINGS")
    print("="*70)
    print("Esta es la SOLUCIÓN DEFINITIVA al problema de saturación")
    print("="*70)
    
    tests = [
        ("Reconocimiento de texto idéntico", test_same_text_recognition),
        ("Unicidad de patrones", test_pattern_uniqueness),
        ("Diferenciación por similitud", test_embedding_differentiation),
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n{'#'*70}")
        print(f"# {name}")
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
    print(f"📋 RESUMEN FINAL DE TODOS LOS TESTS")
    print(f"{'='*70}")
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print(f"\n   Total: {passed}/{total} tests pasados ({passed/total*100:.0f}%)")
    
    if passed == total:
        print(f"\n🎉 ¡ÉXITO TOTAL!")
        print(f"✅ La solución basada en embeddings funciona perfectamente")
        print(f"✅ El sistema de aprendizaje continuo está listo para usar")
        print(f"✅ No requiere modelo entrenado")
        print(f"✅ Reconocimiento de patrones funcional")
    else:
        print(f"\n⚠️  Algunos tests fallaron - revisar implementación")
    
    print(f"{'='*70}")
    
    sys.exit(0 if passed == total else 1)
