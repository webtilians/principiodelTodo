"""
🔬 ANÁLISIS SIMPLIFICADO: ¿Dónde se pierde la señal semántica?
===============================================================

Enfoque directo:
1. Comparar embeddings crudos (SemanticTextEmbedder)
2. Comparar después de generate_text_based_input()
3. Comparar después de múltiples train_steps()
"""

import torch
import sys
sys.path.append('src')

from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough, SemanticTextEmbedder
from argparse import Namespace
import numpy as np

def analyze_where_signal_is_lost():
    print("="*80)
    print("🔬 RASTREO DE PÉRDIDA DE SEÑAL SEMÁNTICA")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    # Textos con máxima diferencia
    textos = [
        "mi perro es rojo",
        "yo pienso, luego existo"
    ]
    
    # ETAPA 1: Embeddings crudos
    print("\n" + "="*80)
    print("ETAPA 1: EMBEDDINGS CRUDOS (SemanticTextEmbedder)")
    print("="*80)
    
    embedder = SemanticTextEmbedder(embed_dim=256, use_glove=False)
    
    emb1 = embedder.text_to_tensor(textos[0], device)
    emb2 = embedder.text_to_tensor(textos[1], device)
    
    diff_stage1 = (emb1 - emb2).norm().item()
    
    print(f"\n📝 '{textos[0]}':")
    print(f"   Norm: {emb1.norm().item():.6f}")
    print(f"   Mean: {emb1.mean().item():.6f}")
    print(f"   Primeros 10 valores: {emb1[0, :10].tolist()}")
    
    print(f"\n📝 '{textos[1]}':")
    print(f"   Norm: {emb2.norm().item():.6f}")
    print(f"   Mean: {emb2.mean().item():.6f}")
    print(f"   Primeros 10 valores: {emb2[0, :10].tolist()}")
    
    print(f"\n📊 L2 Distance entre embeddings: {diff_stage1:.6f}")
    print(f"   → Baseline de diferencia semántica")
    
    # ETAPA 2: Después de generate_text_based_input
    print("\n" + "="*80)
    print("ETAPA 2: POST-TRANSFORMACIÓN (generate_text_based_input)")
    print("="*80)
    
    args = Namespace(
        batch_size=4,
        input_dim=256,
        hidden_dim=512,
        attention_heads=8,
        memory_slots=256,
        lr=1e-3,
        text_mode=True,
        input_text=None,
        quantum_active=False,
        target_consciousness=0.6,
        max_consciousness=0.9
    )
    
    model = InfinitoV51ConsciousnessBreakthrough(args)
    
    with torch.no_grad():
        input1 = model.generate_text_based_input(textos[0])
        input2 = model.generate_text_based_input(textos[1])
    
    # Calcular diferencia (considerar batch y sequence)
    # Flatten para comparar
    flat1 = input1.flatten()
    flat2 = input2.flatten()
    
    diff_stage2 = (flat1 - flat2).norm().item()
    
    print(f"\n📝 '{textos[0]}':")
    print(f"   Shape: {input1.shape}")
    print(f"   Norm (total): {input1.norm().item():.6f}")
    print(f"   Mean: {input1.mean().item():.6f}")
    
    print(f"\n📝 '{textos[1]}':")
    print(f"   Shape: {input2.shape}")
    print(f"   Norm (total): {input2.norm().item():.6f}")
    print(f"   Mean: {input2.mean().item():.6f}")
    
    print(f"\n📊 L2 Distance (flattened): {diff_stage2:.6f}")
    print(f"   Ratio vs embedding crudo: {(diff_stage2/diff_stage1):.2%}")
    
    if diff_stage2 / diff_stage1 < 0.1:
        print(f"   ❌ PÉRDIDA CRÍTICA: {(1 - diff_stage2/diff_stage1)*100:.1f}% perdido")
        print(f"   → Problema en generate_text_based_input()")
    else:
        print(f"   ✅ Señal preservada")
    
    # ETAPA 3: Después de train_step (aquí está el problema sospechoso)
    print("\n" + "="*80)
    print("ETAPA 3: DESPUÉS DE TRAIN_STEP (arquitectura entrenada)")
    print("="*80)
    
    # Crear modelos separados para cada texto
    results = {}
    
    for texto in textos:
        print(f"\n📝 Procesando '{texto}'...")
        
        # Crear args con el texto específico
        args_texto = Namespace(
            batch_size=4,
            input_dim=256,
            hidden_dim=512,
            attention_heads=8,
            memory_slots=256,
            lr=1e-3,
            text_mode=True,
            input_text=texto,  # ← AQUÍ va el texto
            quantum_active=False,
            target_consciousness=0.6,
            max_consciousness=0.9
        )
        
        # Crear modelo fresh con este texto
        model_fresh = InfinitoV51ConsciousnessBreakthrough(args_texto)
        
        # Ejecutar varias iteraciones
        phi_values = []
        
        for iter_num in range(10):
            result = model_fresh.train_step(iteration=iter_num)  # No pasa text_input
            phi_values.append(result['phi'])
        
        results[texto] = {
            'phi_mean': np.mean(phi_values),
            'phi_std': np.std(phi_values),
            'phi_trajectory': phi_values
        }
        
        print(f"   Φ mean: {results[texto]['phi_mean']:.6f}")
        print(f"   Φ std:  {results[texto]['phi_std']:.6f}")
        print(f"   Trajectory: {[f'{p:.3f}' for p in phi_values[:5]]}")
    
    phi_diff = abs(results[textos[0]]['phi_mean'] - results[textos[1]]['phi_mean'])
    
    print(f"\n📊 Diferencia en Φ medio: {phi_diff:.6f}")
    print(f"   Ratio vs embedding crudo: {(phi_diff/diff_stage1):.2%}")
    
    if phi_diff / diff_stage1 < 0.01:
        print(f"   ❌ PÉRDIDA CATASTRÓFICA: {(1 - phi_diff/diff_stage1)*100:.1f}% perdido")
        print(f"   → Problema en el training loop")
    else:
        print(f"   ✅ Algo de señal se preserva")
    
    # DIAGNÓSTICO FINAL
    print("\n" + "="*80)
    print("🎯 DIAGNÓSTICO FINAL")
    print("="*80)
    
    print(f"\n📊 Resumen de preservación de señal:")
    print(f"   Embedding crudo:            {diff_stage1:.6f} (100%)")
    print(f"   Post-transformación:        {diff_stage2:.6f} ({(diff_stage2/diff_stage1)*100:.1f}%)")
    print(f"   Post-entrenamiento (Φ):     {phi_diff:.6f} ({(phi_diff/diff_stage1)*100:.2f}%)")
    
    print(f"\n🔍 Análisis por etapa:")
    
    loss_stage2 = (1 - diff_stage2/diff_stage1) * 100 if diff_stage1 > 0 else 0
    loss_stage3 = (1 - phi_diff/diff_stage2) * 100 if diff_stage2 > 0 else 0
    
    print(f"\n   1️⃣ Embedding → Transformación:")
    if loss_stage2 > 50:
        print(f"      ❌ CRÍTICO: {loss_stage2:.1f}% perdido")
        print(f"      → Revisar generate_text_based_input()")
        print(f"      → El semantic embedding se diluye al crear input multimodal")
    elif loss_stage2 > 10:
        print(f"      ⚠️ MODERADO: {loss_stage2:.1f}% perdido")
    else:
        print(f"      ✅ OK: {loss_stage2:.1f}% perdido (aceptable)")
    
    print(f"\n   2️⃣ Transformación → Entrenamiento:")
    if loss_stage3 > 50:
        print(f"      ❌ CRÍTICO: {loss_stage3:.1f}% perdido")
        print(f"      → Revisar train_step() y loss function")
        print(f"      → Los gradientes están borrando la señal")
    elif loss_stage3 > 10:
        print(f"      ⚠️ MODERADO: {loss_stage3:.1f}% perdido")
    else:
        print(f"      ✅ OK: {loss_stage3:.1f}% perdido (aceptable)")
    
    print(f"\n🎯 CONCLUSIÓN:")
    
    if loss_stage2 > 50:
        print(f"\n   El problema PRINCIPAL está en generate_text_based_input()")
        print(f"   El embedding se pierde al transformarlo a input multimodal")
        print(f"\n   💡 SOLUCIÓN:")
        print(f"      - No diluir el semantic embedding con ondas sinusoidales")
        print(f"      - Usar el embedding directamente sin modulaciones excesivas")
        print(f"      - Verificar que semantic_tiled conserva la información")
    
    elif loss_stage3 > 50:
        print(f"\n   El problema PRINCIPAL está en el training loop")
        print(f"   Los gradientes borran la información del input")
        print(f"\n   💡 SOLUCIÓN:")
        print(f"      - Añadir regularización que preserve el input semántico")
        print(f"      - Reducir learning rate")
        print(f"      - Condicionar el loss al semantic embedding")
    
    else:
        print(f"\n   El problema es SISTÉMICO (toda la pipeline)")
        print(f"   Cada etapa pierde un poco de señal")
        print(f"\n   💡 SOLUCIÓN:")
        print(f"      - Optimización integral de toda la arquitectura")
        print(f"      - Re-inyectar semantic embedding en cada iteración")

if __name__ == "__main__":
    analyze_where_signal_is_lost()
