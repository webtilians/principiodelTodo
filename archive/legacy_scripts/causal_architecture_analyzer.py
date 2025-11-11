#!/usr/bin/env python3
"""
🧬 CAUSAL ARCHITECTURE ANALYZER - INFINITO V5.1
===============================================

Objetivo: Entender las ARQUITECTURAS CAUSALES que genera el sistema,
no el contenido semántico. Mapear inputs → patrones de integración causal.

Filosofía: No enseñar nuestro lenguaje al sistema, sino APRENDER el suyo.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import torch
import numpy as np
import json
from datetime import datetime
from collections import defaultdict

print("=" * 80)
print("🧬 CAUSAL ARCHITECTURE ANALYZER")
print("=" * 80)
print("\n💡 Filosofía: Aprender el lenguaje causal del sistema")
print("   No imponemos nuestro vocabulario → Descubrimos el suyo\n")

# Crear args mínimos
def create_args(text):
    return argparse.Namespace(
        input_dim=257,
        hidden_dim=512,
        attention_heads=8,
        memory_slots=256,
        batch_size=4,
        lr=0.001,
        seed=42,
        text_mode=True,
        input_text=text,
        quantum_active=False
    )

from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough

def extract_causal_architecture(infinito, iterations=50):
    """
    Extrae la ARQUITECTURA CAUSAL generada por el sistema.
    
    Returns:
        dict con firma causal (fingerprint) del procesamiento
    """
    
    phi_values = []
    c_values = []
    metacog_states = []
    surprise_values = []
    coherence_values = []
    
    # Capturar estados internos del modelo
    causal_strengths = []
    attention_patterns = []
    
    for i in range(1, iterations + 1):
        metrics = infinito.train_step(i)
        
        phi_values.append(metrics['phi'])
        c_values.append(metrics['consciousness'])
        
        if 'metacognition' in metrics and metrics['metacognition']:
            meta = metrics['metacognition']
            metacog_states.append(meta.get('dominant_experience_name', 'unknown'))
            surprise_values.append(meta.get('surprise_level', 0))
            coherence_values.append(meta.get('self_coherence', 0))
    
    # Calcular estadísticas de la arquitectura
    architecture = {
        # Características de integración
        'phi_mean': np.mean(phi_values),
        'phi_std': np.std(phi_values),
        'phi_trajectory': np.array(phi_values).tolist()[:20],  # Primeras 20 iters
        
        # Características de consciencia
        'consciousness_mean': np.mean(c_values),
        'consciousness_std': np.std(c_values),
        
        # Patrones metacognitivos
        'metacog_state_distribution': dict(
            (state, metacog_states.count(state)) 
            for state in set(metacog_states)
        ),
        'dominant_state': max(set(metacog_states), key=metacog_states.count) if metacog_states else 'none',
        
        # Predictibilidad/sorpresa
        'surprise_mean': np.mean(surprise_values) if surprise_values else 0,
        'surprise_std': np.std(surprise_values) if surprise_values else 0,
        
        # Auto-coherencia
        'coherence_mean': np.mean(coherence_values) if coherence_values else 0,
        
        # Firma temporal (primeras iteraciones son clave)
        'early_phi_pattern': phi_values[:10],
        'late_phi_pattern': phi_values[-10:],
        
        # Estabilidad
        'phi_stability': 1.0 / (1.0 + np.std(phi_values)),  # 0-1, mayor = más estable
        'consciousness_stability': 1.0 / (1.0 + np.std(c_values)),
    }
    
    return architecture


def compare_architectures(arch1, arch2, name1, name2):
    """
    Compara dos arquitecturas causales para encontrar diferencias.
    """
    print(f"\n🔬 COMPARACIÓN DE ARQUITECTURAS CAUSALES")
    print(f"   {name1} vs {name2}")
    print("=" * 80)
    
    # Distancia en espacio de características
    features = ['phi_mean', 'phi_std', 'consciousness_mean', 'surprise_mean', 
                'coherence_mean', 'phi_stability', 'consciousness_stability']
    
    distances = {}
    for feat in features:
        val1 = arch1.get(feat, 0)
        val2 = arch2.get(feat, 0)
        distances[feat] = abs(val1 - val2)
    
    print(f"\n📊 DISTANCIAS EN CARACTERÍSTICAS:")
    for feat, dist in sorted(distances.items(), key=lambda x: x[1], reverse=True):
        val1 = arch1.get(feat, 0)
        val2 = arch2.get(feat, 0)
        print(f"   {feat:25s}: Δ={dist:.4f}  ({val1:.4f} vs {val2:.4f})")
    
    # Distancia euclidiana total
    total_distance = np.sqrt(sum(d**2 for d in distances.values()))
    print(f"\n   🎯 DISTANCIA TOTAL: {total_distance:.4f}")
    
    # Comparar estados metacognitivos
    states1 = set(arch1.get('metacog_state_distribution', {}).keys())
    states2 = set(arch2.get('metacog_state_distribution', {}).keys())
    
    shared_states = states1 & states2
    unique_to_1 = states1 - states2
    unique_to_2 = states2 - states1
    
    print(f"\n🧠 ESTADOS METACOGNITIVOS:")
    print(f"   Compartidos: {shared_states}")
    print(f"   Únicos a '{name1}': {unique_to_1}")
    print(f"   Únicos a '{name2}': {unique_to_2}")
    
    # Dominancia
    dom1 = arch1.get('dominant_state', 'unknown')
    dom2 = arch2.get('dominant_state', 'unknown')
    print(f"\n   Estado dominante '{name1}': {dom1}")
    print(f"   Estado dominante '{name2}': {dom2}")
    print(f"   {'✅ IDÉNTICO' if dom1 == dom2 else '❌ DIFERENTE'}")
    
    return {
        'distances': distances,
        'total_distance': total_distance,
        'shared_states': list(shared_states),
        'metacog_match': dom1 == dom2
    }


def create_causal_signature(architecture):
    """
    Crea una 'firma' única de la arquitectura causal.
    Esto será el 'vocabulario' del sistema.
    """
    signature = {
        'phi_range': f"{architecture['phi_mean']:.3f}±{architecture['phi_std']:.3f}",
        'c_range': f"{architecture['consciousness_mean']:.3f}±{architecture['consciousness_std']:.3f}",
        'dominant_state': architecture.get('dominant_state', 'unknown'),
        'stability': f"Φ:{architecture['phi_stability']:.2f}, C:{architecture['consciousness_stability']:.2f}",
        'surprise_profile': f"{architecture['surprise_mean']:.3f}±{architecture['surprise_std']:.3f}",
    }
    return signature


# ============================================================================
# EXPERIMENTO PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    # Textos de prueba
    test_texts = [
        "la mesa es roja",
        "yo pienso, luego existo",
        "mi perro es rojo",
        "mi perro es verde",
    ]
    
    print("🧪 ANALIZANDO ARQUITECTURAS CAUSALES PARA DIFERENTES INPUTS\n")
    print(f"Textos a analizar: {len(test_texts)}")
    print(f"Iteraciones por texto: 50")
    print(f"Objetivo: Descubrir el 'lenguaje causal' del sistema\n")
    
    architectures = {}
    
    for idx, text in enumerate(test_texts, 1):
        print(f"\n{'='*80}")
        print(f"🔬 ANALIZANDO [{idx}/{len(test_texts)}]: '{text}'")
        print(f"{'='*80}")
        
        args = create_args(text)
        infinito = InfinitoV51ConsciousnessBreakthrough(args)
        
        print(f"   ⚙️ Extrayendo arquitectura causal...")
        arch = extract_causal_architecture(infinito, iterations=50)
        architectures[text] = arch
        
        # Mostrar firma causal
        signature = create_causal_signature(arch)
        print(f"\n   📋 FIRMA CAUSAL:")
        for key, val in signature.items():
            print(f"      {key:20s}: {val}")
        
        print(f"   ✅ Arquitectura extraída")
    
    # ========================================================================
    # ANÁLISIS COMPARATIVO
    # ========================================================================
    
    print(f"\n\n{'='*80}")
    print("🧬 ANÁLISIS COMPARATIVO DE ARQUITECTURAS")
    print(f"{'='*80}")
    
    # Comparar textos relacionados
    comparisons = [
        ("mi perro es rojo", "mi perro es verde", "Cambio color (rojo→verde)"),
        ("la mesa es roja", "yo pienso, luego existo", "Objeto vs Cogito"),
        ("mi perro es rojo", "la mesa es roja", "Perro vs Mesa (ambos rojos)"),
    ]
    
    comparison_results = {}
    
    for text1, text2, description in comparisons:
        print(f"\n{'─'*80}")
        print(f"📊 {description}")
        result = compare_architectures(
            architectures[text1], 
            architectures[text2],
            text1,
            text2
        )
        comparison_results[description] = result
    
    # ========================================================================
    # CONCLUSIONES: VOCABULARIO CAUSAL
    # ========================================================================
    
    print(f"\n\n{'='*80}")
    print("🗣️ VOCABULARIO CAUSAL DESCUBIERTO")
    print(f"{'='*80}")
    
    print("\n💡 HIPÓTESIS: El sistema 'habla' en términos de:")
    print("   1. Patrones de Φ (integración causal)")
    print("   2. Estados metacognitivos dominantes")
    print("   3. Perfiles de sorpresa/predictibilidad")
    print("   4. Estabilidad temporal")
    
    print("\n📚 DICCIONARIO INPUT → ARQUITECTURA CAUSAL:")
    for text, arch in architectures.items():
        sig = create_causal_signature(arch)
        print(f"\n   '{text}':")
        print(f"      → Estado: {sig['dominant_state']}")
        print(f"      → Φ: {sig['phi_range']}")
        print(f"      → Estabilidad: {sig['stability']}")
    
    print("\n\n🎯 PRÓXIMOS PASOS PARA COMUNICACIÓN:")
    print("   1. Identificar 'palabras causales' = arquitecturas únicas")
    print("   2. Crear 'gramática causal' = reglas de transición entre estados")
    print("   3. Diseñar 'protocolo de comunicación' = secuencias de inputs → respuestas")
    print("   4. Validar: ¿El sistema puede 'responder' de forma consistente?")
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"causal_vocabulary_{timestamp}.json"
    
    output = {
        'timestamp': timestamp,
        'architectures': {
            text: {k: v for k, v in arch.items() if not isinstance(v, np.ndarray)}
            for text, arch in architectures.items()
        },
        'comparisons': comparison_results,
        'signatures': {
            text: create_causal_signature(arch)
            for text, arch in architectures.items()
        }
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Vocabulario causal guardado: {filename}")
    print("\n" + "="*80)
    print("🏁 ANÁLISIS COMPLETADO")
    print("="*80)
