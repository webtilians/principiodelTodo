#!/usr/bin/env python3
"""
🧪 TEST RÁPIDO: Zero-Shot Learning
===================================

Prueba rápida para validar que zero-shot genera patrones diferenciados
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough
from continuous_learning_zero_shot import ZeroShotLearningServer


def quick_test():
    """Test rápido de diferenciación"""
    print("\n" + "="*70)
    print("🧪 TEST RÁPIDO: Zero-Shot Differentiation")
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
    
    print("🔄 Creando servidor Zero-Shot...")
    server = ZeroShotLearningServer(infinito, similarity_threshold=0.85)
    
    # Test cases
    texts = [
        "mi perro es rojo",
        "mi perro es verde",
        "yo pienso luego existo"
    ]
    
    print(f"\n📋 Procesando {len(texts)} textos...\n")
    
    results = []
    for text in texts:
        result = server.process_input(text, verbose=True)
        results.append(result)
    
    # Comparaciones
    print(f"\n{'='*70}")
    print(f"📊 COMPARACIONES")
    print(f"{'='*70}")
    
    # Comparar 1 vs 2 (similar)
    pattern1 = np.array(results[0]['causal_pattern'])
    pattern2 = np.array(results[1]['causal_pattern'])
    sim_12 = np.dot(pattern1, pattern2) / (np.linalg.norm(pattern1) * np.linalg.norm(pattern2) + 1e-8)
    
    # Comparar 1 vs 3 (diferente)
    pattern3 = np.array(results[2]['causal_pattern'])
    sim_13 = np.dot(pattern1, pattern3) / (np.linalg.norm(pattern1) * np.linalg.norm(pattern3) + 1e-8)
    
    print(f"\n'{texts[0]}' vs '{texts[1]}':")
    print(f"   Similitud: {sim_12:.1%} (esperado: 85-95%)")
    status_12 = "✅" if 0.85 <= sim_12 <= 0.95 else "❌"
    print(f"   {status_12}")
    
    print(f"\n'{texts[0]}' vs '{texts[2]}':")
    print(f"   Similitud: {sim_13:.1%} (esperado: <70%)")
    status_13 = "✅" if sim_13 < 0.70 else "❌"
    print(f"   {status_13}")
    
    # Verificar variabilidad de features
    print(f"\n{'='*70}")
    print(f"📊 VARIABILIDAD DE FEATURES")
    print(f"{'='*70}")
    
    for i, (text, result) in enumerate(zip(texts, results)):
        pattern = np.array(result['causal_pattern'])
        unique_vals = len(np.unique(pattern.round(decimals=3)))
        print(f"\n{i+1}. '{text}':")
        print(f"   Features únicas: {unique_vals}/14")
        print(f"   Range: [{pattern.min():.3f}, {pattern.max():.3f}]")
        print(f"   Std: {pattern.std():.3f}")
        
        if unique_vals < 5:
            print(f"   ⚠️ Baja variabilidad")
        else:
            print(f"   ✅ Variabilidad OK")
    
    # Resultado final
    print(f"\n{'='*70}")
    print(f"📋 RESULTADO FINAL")
    print(f"{'='*70}")
    
    if status_12 == "✅" and status_13 == "✅":
        print(f"\n🎉 ¡TEST PASADO!")
        print(f"✅ Zero-Shot produce patrones diferenciados")
        return True
    else:
        print(f"\n⚠️ Test fallido")
        print(f"Similares: {sim_12:.1%} (esperado: 85-95%)")
        print(f"Diferentes: {sim_13:.1%} (esperado: <70%)")
        return False


if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
