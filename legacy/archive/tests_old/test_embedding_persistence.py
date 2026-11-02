"""
🔬 TEST DE PERSISTENCIA DEL EMBEDDING SEMÁNTICO
==============================================

Objetivo: Rastrear si el embedding semántico inicial persiste durante el entrenamiento
o si la dinámica de optimización lo "borra" inmediatamente.

Método:
1. Crear modelo con texto específico
2. Capturar estado interno en múltiples puntos durante train_step()
3. Medir cuánto cambian las activaciones desde el input inicial
4. Comparar con otro texto para ver si la diferencia persiste
"""

import torch
import sys
sys.path.append('src')

from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough, SemanticTextEmbedder
import numpy as np
from argparse import Namespace

def test_embedding_persistence():
    """
    Test que rastrea la persistencia del embedding durante el entrenamiento
    """
    
    print("="*80)
    print("🔬 TEST DE PERSISTENCIA DEL EMBEDDING SEMÁNTICO")
    print("="*80)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)  # Seed fijo para comparar textos
    
    textos = [
        "mi perro es rojo",
        "mi perro es verde",
    ]
    
    # Primero: verificar embeddings puros del SemanticTextEmbedder
    print("\n🔍 PASO 1: Verificando SemanticTextEmbedder directamente")
    print("="*80)
    
    embedder = SemanticTextEmbedder(embed_dim=256, use_glove=False)  # Solo TF-IDF
    
    for texto in textos:
        emb = embedder.text_to_tensor(texto, device)
        print(f"\n📝 '{texto}':")
        print(f"   Embedding norm:  {emb.norm().item():.6f}")
        print(f"   Embedding mean:  {emb.mean().item():.6f}")
        print(f"   Embedding std:   {emb.std().item():.6f}")
        print(f"   Embedding min:   {emb.min().item():.6f}")
        print(f"   Embedding max:   {emb.max().item():.6f}")
    
    # Comparar embeddings
    emb1 = embedder.text_to_tensor(textos[0], device)
    emb2 = embedder.text_to_tensor(textos[1], device)
    
    diff = (emb1 - emb2).abs()
    cosine_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
    
    print(f"\n📊 Comparación entre embeddings:")
    print(f"   L1 distance:      {diff.sum().item():.6f}")
    print(f"   L2 distance:      {diff.norm().item():.6f}")
    print(f"   Cosine similarity: {cosine_sim.item():.6f}")
    print(f"   Mean abs diff:    {diff.mean().item():.6f}")
    print(f"   Max abs diff:     {diff.max().item():.6f}")
    
    if diff.norm().item() < 0.001:
        print(f"\n   ❌ PROBLEMA: Los embeddings son IDÉNTICOS")
        print(f"      El TF-IDF no está diferenciando 'rojo' vs 'verde'")
        return
    else:
        print(f"\n   ✅ Los embeddings son DIFERENTES")
        print(f"      El SemanticTextEmbedder funciona correctamente")
    
    # Segundo: verificar cómo se transforman en el modelo
    print("\n" + "="*80)
    print("🔍 PASO 2: Verificando transformación en generate_text_based_input()")
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
    model = InfinitoV51ConsciousnessBreakthrough(args)
    
    results = {}
    
    for texto in textos:
        print(f"\n📝 Analizando: '{texto}'")
        print("-" * 60)
        
        # Obtener embedding inicial ANTES de cualquier entrenamiento
        with torch.no_grad():
            initial_input = model.generate_text_based_input(texto)
            initial_visual = initial_input['visual'][0].clone()  # Take first batch element
            initial_executive = initial_input['executive'][0].clone()
            
            print(f"   🧠 Embedding inicial (post-transform):")
            print(f"      Visual norm: {initial_visual.norm().item():.6f}")
            print(f"      Executive norm: {initial_executive.norm().item():.6f}")
            print(f"      Visual mean: {initial_visual.mean().item():.6f}")
            print(f"      Executive mean: {initial_executive.mean().item():.6f}")
            
            # Verificar si el semantic embedder genera algo
            if hasattr(model, 'semantic_embedder'):
                emb = model.semantic_embedder.text_to_tensor(texto, device)
                print(f"      Semantic embedding (raw) norm: {emb.norm().item():.6f}")
                print(f"      Semantic embedding (raw) mean: {emb.mean().item():.6f}")
        
        # Ejecutar algunas iteraciones y monitorear cambios
        phi_trajectory = []
        visual_drift = []
        executive_drift = []
        
        print(f"\n   📊 Monitoreando drift durante entrenamiento:")
        
        for iter_num in range(10):
            # Train step
            result = model.train_step(
                iteration=iter_num,
                text_input=texto
            )
            
            phi = result['phi']
            phi_trajectory.append(phi)
            
            # Medir cuánto se ha alejado del embedding inicial
            with torch.no_grad():
                current_input = model.generate_text_based_input(texto)
                current_visual = current_input['visual'][0]  # First batch element
                current_executive = current_input['executive'][0]
                
                # Distancia L2 desde el inicial
                visual_distance = (current_visual - initial_visual).norm().item()
                executive_distance = (current_executive - initial_executive).norm().item()
                
                visual_drift.append(visual_distance)
                executive_drift.append(executive_distance)
                
                if iter_num in [0, 2, 5, 9]:
                    print(f"      Iter {iter_num:2d}: Φ={phi:.4f}, "
                          f"Visual drift={visual_distance:.6f}, "
                          f"Executive drift={executive_distance:.6f}")
        
        results[texto] = {
            'phi_trajectory': phi_trajectory,
            'visual_drift': visual_drift,
            'executive_drift': executive_drift,
            'initial_visual_norm': initial_visual.norm().item(),
            'initial_executive_norm': initial_executive.norm().item(),
        }
    
    # Análisis comparativo
    print("\n" + "="*80)
    print("📊 ANÁLISIS COMPARATIVO DE DRIFT")
    print("="*80)
    
    texto1, texto2 = textos[0], textos[1]
    
    print(f"\n🔴 {texto1}:")
    print(f"   Visual drift final:     {results[texto1]['visual_drift'][-1]:.6f}")
    print(f"   Executive drift final:  {results[texto1]['executive_drift'][-1]:.6f}")
    print(f"   Φ mean:                 {np.mean(results[texto1]['phi_trajectory']):.6f}")
    
    print(f"\n🟢 {texto2}:")
    print(f"   Visual drift final:     {results[texto2]['visual_drift'][-1]:.6f}")
    print(f"   Executive drift final:  {results[texto2]['executive_drift'][-1]:.6f}")
    print(f"   Φ mean:                 {np.mean(results[texto2]['phi_trajectory']):.6f}")
    
    # Comparar trayectorias de drift
    print(f"\n📈 DIFERENCIA EN TRAYECTORIAS DE DRIFT:")
    drift_diff_visual = np.abs(np.array(results[texto1]['visual_drift']) - 
                                np.array(results[texto2]['visual_drift']))
    drift_diff_executive = np.abs(np.array(results[texto1]['executive_drift']) - 
                                   np.array(results[texto2]['executive_drift']))
    
    print(f"   Visual drift diff (mean):     {drift_diff_visual.mean():.6f}")
    print(f"   Executive drift diff (mean):  {drift_diff_executive.mean():.6f}")
    print(f"   Visual drift diff (max):      {drift_diff_visual.max():.6f}")
    print(f"   Executive drift diff (max):   {drift_diff_executive.max():.6f}")
    
    # Diagnóstico
    print("\n" + "="*80)
    print("🎯 DIAGNÓSTICO")
    print("="*80)
    
    # ¿Los embeddings iniciales son diferentes?
    initial_diff_visual = abs(results[texto1]['initial_visual_norm'] - 
                              results[texto2]['initial_visual_norm'])
    initial_diff_executive = abs(results[texto1]['initial_executive_norm'] - 
                                 results[texto2]['initial_executive_norm'])
    
    print(f"\n1️⃣ Diferencia en embeddings INICIALES:")
    print(f"   Visual norm diff:     {initial_diff_visual:.6f}")
    print(f"   Executive norm diff:  {initial_diff_executive:.6f}")
    
    if initial_diff_visual < 0.001 and initial_diff_executive < 0.001:
        print(f"   ❌ Los embeddings iniciales son IDÉNTICOS")
        print(f"   → Problema: SemanticTextEmbedder no diferencia los textos")
    else:
        print(f"   ✅ Los embeddings iniciales son DIFERENTES")
    
    print(f"\n2️⃣ Persistencia del embedding durante entrenamiento:")
    final_drift = max(results[texto1]['visual_drift'][-1], 
                     results[texto2]['visual_drift'][-1])
    initial_norm = max(results[texto1]['initial_visual_norm'],
                      results[texto2]['initial_visual_norm'])
    
    drift_ratio = final_drift / initial_norm if initial_norm > 0 else 0
    
    print(f"   Drift final / Norm inicial = {drift_ratio:.2f}")
    
    if drift_ratio > 1.0:
        print(f"   ❌ El embedding se DESTRUYE durante entrenamiento")
        print(f"   → El input inicial se pierde completamente")
    elif drift_ratio > 0.5:
        print(f"   ⚠️ El embedding se DEGRADA significativamente")
        print(f"   → El input inicial se debilita mucho")
    else:
        print(f"   ✅ El embedding se MANTIENE relativamente estable")
    
    print(f"\n3️⃣ Diferenciación entre textos durante entrenamiento:")
    
    if drift_diff_visual.mean() < 0.001:
        print(f"   ❌ Los textos evolucionan IDÉNTICAMENTE")
        print(f"   → La red ignora diferencias en el input")
    else:
        print(f"   ✅ Los textos evolucionan DIFERENTEMENTE")
        print(f"   → La red responde al input diferente")
    
    # Conclusión final
    print("\n" + "="*80)
    print("🔍 CONCLUSIÓN FINAL")
    print("="*80)
    
    if initial_diff_visual < 0.001:
        print("\n❌ PROBLEMA EN LA ETAPA 1: GENERACIÓN DE EMBEDDINGS")
        print("   El SemanticTextEmbedder no diferencia 'rojo' vs 'verde'")
        print("   Acción: Verificar TF-IDF y vocabulario")
    elif drift_ratio > 0.5:
        print("\n❌ PROBLEMA EN LA ETAPA 2: PROPAGACIÓN EN LA RED")
        print("   El embedding inicial se pierde durante el forward pass")
        print("   Acción: Verificar cómo el input afecta las capas internas")
    elif drift_diff_visual.mean() < 0.001:
        print("\n❌ PROBLEMA EN LA ETAPA 3: OPTIMIZACIÓN")
        print("   Los gradientes fuerzan convergencia al mismo punto")
        print("   Acción: Verificar loss function y target conditioning")
    else:
        print("\n✅ TODO FUNCIONA CORRECTAMENTE")
        print("   Los embeddings son diferentes y persisten")

if __name__ == "__main__":
    test_embedding_persistence()
