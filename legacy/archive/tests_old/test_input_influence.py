#!/usr/bin/env python3
"""
🔬 TEST CRÍTICO: ¿El input textual afecta la arquitectura?
==========================================================

Hipótesis a probar:
  Si usamos el MISMO seed para diferentes textos:
  - ¿Obtenemos la misma arquitectura? → Input NO importa
  - ¿Obtenemos arquitecturas diferentes? → Input SÍ importa

Método:
  Seed fijo = 42 (como causal_architecture_analyzer.py)
  4 textos diferentes
  50 iteraciones cada uno
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import torch
import numpy as np
from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough

def test_with_fixed_seed(text, seed=42, iterations=50):
    """Test un texto con seed fijo"""
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Crear args
    args = argparse.Namespace(
        input_dim=257,
        hidden_dim=512,
        attention_heads=8,
        memory_slots=256,
        batch_size=4,
        lr=0.001,
        seed=seed,
        text_mode=True,
        input_text=text,
        quantum_active=False,
        max_iterations=iterations
    )
    
    # Crear runner
    infinito = InfinitoV51ConsciousnessBreakthrough(args)
    
    # Ejecutar iteraciones
    phi_values = []
    
    for i in range(iterations):
        try:
            metrics = infinito.train_step(i)
            if metrics and 'phi' in metrics:
                phi_values.append(metrics['phi'])
        except Exception as e:
            if i == 0:
                print(f"    ⚠️  Error: {e}")
            continue
    
    if not phi_values:
        return None
    
    return {
        'text': text,
        'seed': seed,
        'phi_mean': float(np.mean(phi_values)),
        'phi_std': float(np.std(phi_values)),
        'phi_trajectory': phi_values[:10]  # Primeras 10 iters
    }


def main():
    print("="*80)
    print("🔬 TEST CRÍTICO: ¿El INPUT TEXTUAL afecta la ARQUITECTURA CAUSAL?")
    print("="*80)
    print("\nMétodo: SEED FIJO (42) + TEXTOS DIFERENTES")
    print("Si todos dan el mismo Φ → Input NO importa")
    print("Si dan Φ diferentes → Input SÍ importa\n")
    
    texts = [
        "mi perro es rojo",
        "mi perro es verde",
        "la mesa es roja",
        "yo pienso, luego existo"
    ]
    
    results = []
    
    for text in texts:
        print(f"\n📝 Testeando: '{text}'")
        result = test_with_fixed_seed(text, seed=42, iterations=50)
        
        if result:
            results.append(result)
            print(f"   ✅ Φ = {result['phi_mean']:.6f} ± {result['phi_std']:.6f}")
            print(f"   📈 Trayectoria inicial: {[f'{p:.3f}' for p in result['phi_trajectory'][:5]]}")
        else:
            print(f"   ❌ Error")
    
    # Análisis
    print("\n" + "="*80)
    print("📊 RESULTADOS")
    print("="*80)
    
    print(f"\n{'Texto':<30} {'Φ Mean':<15} {'Φ Std':<15}")
    print("-"*60)
    
    phi_means = []
    for r in results:
        print(f"{r['text']:<30} {r['phi_mean']:<15.6f} {r['phi_std']:<15.6f}")
        phi_means.append(r['phi_mean'])
    
    # Calcular varianza entre textos
    variance_between_texts = np.var(phi_means)
    std_between_texts = np.std(phi_means)
    max_diff = max(phi_means) - min(phi_means)
    
    print("\n" + "="*80)
    print("🎯 CONCLUSIÓN")
    print("="*80)
    
    print(f"\nVarianza entre textos: {variance_between_texts:.8f}")
    print(f"Desviación estándar: {std_between_texts:.8f}")
    print(f"Diferencia máxima: {max_diff:.8f}")
    print(f"Rango: [{min(phi_means):.6f}, {max(phi_means):.6f}]")
    
    # Umbral de diferencia significativa
    threshold = 0.001  # Si la varianza es menor que esto, son prácticamente iguales
    
    if variance_between_texts < threshold:
        print(f"\n❌ RESULTADO: Los textos NO afectan la arquitectura")
        print(f"   Varianza ({variance_between_texts:.8f}) < {threshold}")
        print(f"   El INPUT TEXTUAL es IGNORADO por el sistema")
        print("\n💡 DIAGNÓSTICO:")
        print("   - El seed determina completamente la arquitectura")
        print("   - El semantic embedder NO está influyendo")
        print("   - La dinámica de entrenamiento domina sobre el input")
    else:
        print(f"\n✅ RESULTADO: Los textos SÍ afectan la arquitectura")
        print(f"   Varianza ({variance_between_texts:.8f}) >= {threshold}")
        print(f"   El INPUT TEXTUAL influye en la arquitectura causal")
        print("\n💡 INTERPRETACIÓN:")
        print("   - Diferentes textos generan diferentes patrones de Φ")
        print("   - El semantic embedder está funcionando")
        print("   - El sistema puede discriminar entre inputs")
    
    # Comparar trayectorias
    print("\n" + "="*80)
    print("📈 TRAYECTORIAS DE Φ (primeras 10 iteraciones)")
    print("="*80)
    
    for r in results:
        print(f"\n{r['text']}:")
        print(f"  {[f'{p:.3f}' for p in r['phi_trajectory']]}")
    
    # Verificar si las trayectorias son idénticas
    if len(results) >= 2:
        traj1 = results[0]['phi_trajectory']
        all_identical = True
        for r in results[1:]:
            if not np.allclose(traj1, r['phi_trajectory'], atol=1e-6):
                all_identical = False
                break
        
        if all_identical:
            print("\n⚠️  ALERTA: Las trayectorias son IDÉNTICAS")
            print("   → El sistema genera la misma secuencia independientemente del texto")
        else:
            print("\n✅ Las trayectorias son DIFERENTES")
            print("   → El texto influye en la evolución del sistema")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
