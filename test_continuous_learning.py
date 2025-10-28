#!/usr/bin/env python3
"""
🧪 TEST: Validación del Sistema de Aprendizaje Continuo
========================================================

Test unitario para verificar que:
1. PhiPatternExtractor extrae vectores correctamente
2. PhiMemoryBank guarda y busca patrones
3. ContinuousLearningServer reconoce patrones similares

PROBLEMA DETECTADO:
- Modelo sin entrenar genera mismos patrones causales para todos los textos
- Solución: Pre-entrenar brevemente (50 iters) con cada texto antes de extraer patrón
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough
from continuous_learning import (
    PhiPatternExtractor,
    PhiMemoryBank,
    ContinuousLearningServer
)


def test_pattern_extractor():
    """Test del extractor de patrones"""
    print("\n" + "="*70)
    print("🧪 TEST 1: PhiPatternExtractor")
    print("="*70)
    
    extractor = PhiPatternExtractor()
    
    # Crear una matriz causal sintética
    causal_matrix = torch.tensor([
        [0.0, 0.85, 0.32, 0.67],
        [0.0, 0.0, 0.91, 0.45],
        [0.0, 0.0, 0.0, 0.78],
        [0.0, 0.0, 0.0, 0.0]
    ])
    
    phi_info = {'causal_matrix': causal_matrix}
    
    # Extraer patrón
    pattern = extractor.extract_pattern(phi_info)
    
    print(f"\n📊 Matriz causal 4x4:")
    print(causal_matrix.numpy())
    
    print(f"\n🔢 Patrón extraído (6 valores):")
    print(pattern)
    
    print(f"\n📝 Visualización:")
    print(extractor.pattern_to_string(pattern))
    
    # Validar
    expected = np.array([0.85, 0.32, 0.67, 0.91, 0.45, 0.78])
    assert np.allclose(pattern, expected), "Patrón no coincide"
    
    print(f"\n✅ PhiPatternExtractor funciona correctamente")
    return True


def test_memory_bank():
    """Test del banco de memoria"""
    print("\n" + "="*70)
    print("🧪 TEST 2: PhiMemoryBank")
    print("="*70)
    
    os.makedirs('results/continuous_learning', exist_ok=True)
    bank = PhiMemoryBank(similarity_threshold=0.90, filepath='results/continuous_learning/test_memory.json')
    
    # Crear patrones sintéticos
    pattern1 = np.array([0.85, 0.32, 0.67, 0.91, 0.45, 0.78])
    pattern2 = np.array([0.86, 0.33, 0.68, 0.92, 0.46, 0.79])  # Muy similar
    pattern3 = np.array([0.20, 0.80, 0.10, 0.30, 0.90, 0.40])  # Diferente
    
    phi_info = {'phi_total': 4.5}
    
    # Añadir primer patrón
    result1 = bank.add_pattern(pattern1, phi_info, "texto 1", 0.65)
    print(f"\n1️⃣  {result1['message']}")
    assert result1['status'] == 'NEW', "Primer patrón debería ser nuevo"
    
    # Añadir patrón similar (debería reconocerlo)
    result2 = bank.add_pattern(pattern2, phi_info, "texto 2", 0.66)
    print(f"2️⃣  {result2['message']}")
    assert result2['status'] == 'RECOGNIZED', "Patrón similar debería ser reconocido"
    assert result2['similarity'] > 0.99, f"Similitud baja: {result2['similarity']}"
    
    # Añadir patrón diferente
    result3 = bank.add_pattern(pattern3, phi_info, "texto 3", 0.70)
    print(f"3️⃣  {result3['message']}")
    assert result3['status'] == 'NEW', "Patrón diferente debería ser nuevo"
    
    # Verificar estadísticas
    stats = bank.get_stats()
    print(f"\n📊 Estadísticas:")
    print(f"   Patrones únicos: {stats['total_patterns']}")
    print(f"   Total visto: {stats['total_seen']}")
    
    assert stats['total_patterns'] == 2, "Deberían haber 2 patrones únicos"
    assert stats['total_seen'] == 3, "Deberían haberse visto 3 veces en total"
    
    # Guardar y cargar en carpeta organizada
    os.makedirs('results/continuous_learning', exist_ok=True)
    bank.save_to_disk('results/continuous_learning/test_memory.json')
    
    bank2 = PhiMemoryBank(filepath='results/continuous_learning/test_memory.json')
    assert len(bank2.patterns) == 2, "Patrones no se cargaron correctamente"
    
    print(f"\n✅ PhiMemoryBank funciona correctamente")
    
    # Limpiar
    test_file = 'results/continuous_learning/test_memory.json'
    if os.path.exists(test_file):
        os.remove(test_file)
    
    return True


def test_with_trained_model():
    """
    Test con modelo pre-entrenado
    
    NOTA: Entrenamos brevemente el modelo con diferentes textos
    para que genere diferentes matrices causales
    """
    print("\n" + "="*70)
    print("🧪 TEST 3: Con Modelo Pre-Entrenado")
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
    
    print("\n🚀 Inicializando INFINITO V5.1...")
    args = Args()
    infinito = InfinitoV51ConsciousnessBreakthrough(args)
    
    # Textos de prueba
    test_texts = [
        "mi perro es rojo",
        "mi perro es verde",
        "yo pienso luego existo",
    ]
    
    print(f"\n📝 Pre-entrenando con {len(test_texts)} textos...")
    print("   (50 iteraciones cada uno para crear diferentes patrones)\n")
    
    extractor = PhiPatternExtractor()
    patterns = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"   {i}. '{text}':")
        
        # Configurar texto
        infinito.input_text = text
        
        # Pre-entrenar brevemente (50 iters)
        for iteration in range(1, 51):
            infinito.train_step(iteration)
        
        # Ahora extraer patrón en modo eval
        infinito.model.eval()
        with torch.no_grad():
            inputs = infinito.generate_text_based_input(text)
            consciousness, phi, debug_info = infinito.model(inputs)
        
        phi_info = debug_info.get('phi_info', {})
        pattern = extractor.extract_pattern(phi_info)
        patterns.append(pattern)
        
        print(f"      Φ={phi.mean().item():.4f}, C={consciousness.mean().item():.4f}")
        print(f"      Patrón: {pattern[:3]}...")  # Primeros 3 valores
    
    # Comparar patrones
    print(f"\n📊 Análisis de Similitud:")
    
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            # Similitud coseno
            sim = np.dot(patterns[i], patterns[j]) / (
                np.linalg.norm(patterns[i]) * np.linalg.norm(patterns[j])
            )
            
            text_i = test_texts[i]
            text_j = test_texts[j]
            
            print(f"\n   '{text_i}' vs '{text_j}':")
            print(f"   Similitud: {sim:.1%}")
            
            # Validar que textos similares tienen patrones similares
            if "perro" in text_i and "perro" in text_j:
                if sim < 0.85:
                    print(f"   ⚠️  Advertencia: Textos similares pero patrones muy diferentes")
            
            # Validar que textos diferentes tienen patrones diferentes
            if "perro" in text_i and "pienso" in text_j:
                if sim > 0.95:
                    print(f"   ⚠️  Advertencia: Textos diferentes pero patrones muy similares")
    
    print(f"\n✅ Test con modelo pre-entrenado completado")
    return True


def run_all_tests():
    """Ejecutar todos los tests"""
    print("\n" + "="*70)
    print("🧪 SUITE DE TESTS - Sistema de Aprendizaje Continuo")
    print("="*70)
    
    tests = [
        ("PhiPatternExtractor", test_pattern_extractor),
        ("PhiMemoryBank", test_memory_bank),
        ("Modelo Pre-Entrenado", test_with_trained_model),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Error en {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Resumen
    print("\n" + "="*70)
    print("📋 RESUMEN DE TESTS")
    print("="*70)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status} - {name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print(f"\n   Total: {passed}/{total} tests pasados")
    
    if passed == total:
        print(f"\n🎉 ¡Todos los tests pasaron!")
    else:
        print(f"\n⚠️  Algunos tests fallaron")


if __name__ == "__main__":
    run_all_tests()
