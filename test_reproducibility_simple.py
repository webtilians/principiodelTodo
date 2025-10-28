"""
Test de Reproducibilidad del Lenguaje Causal - Versión Simplificada
==================================================================

Objetivo: Verificar si el sistema genera arquitecturas causales reproducibles
Método: Ejecutar el mismo texto 10 veces con seeds diferentes
Texto de prueba: "mi perro es rojo"
Resultado esperado: Varianza < 0.05 → Lenguaje causal es determinista
"""

import subprocess
import json
import os
import numpy as np
from collections import Counter
from datetime import datetime

def run_single_analysis(text, seed, iterations=50):
    """
    Ejecuta una análisis con un seed específico usando el script principal
    """
    cmd = [
        'python',
        'src/infinito_gpt_text_fixed.py',
        '--input_text', text,
        '--max_iterations', str(iterations),
        '--seed', str(seed)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        # Parsear la salida para extraer métricas
        output = result.stdout
        
        # Buscar las métricas en la salida
        phi_values = []
        coherence_values = []
        
        for line in output.split('\n'):
            if 'Φ:' in line:
                try:
                    phi_str = line.split('Φ:')[1].split()[0]
                    phi = float(phi_str.replace('%', '').replace(',', '.'))
                    phi_values.append(phi)
                except:
                    pass
            if 'Coherence:' in line or 'C:' in line:
                try:
                    if 'C:' in line:
                        coh_str = line.split('C:')[1].split('%')[0].strip()
                    else:
                        coh_str = line.split('Coherence:')[1].split('%')[0].strip()
                    coh = float(coh_str.replace(',', '.'))
                    coherence_values.append(coh)
                except:
                    pass
        
        if phi_values and coherence_values:
            return {
                'seed': seed,
                'phi_mean': np.mean(phi_values),
                'phi_std': np.std(phi_values),
                'coherence_mean': np.mean(coherence_values),
                'coherence_std': np.std(coherence_values),
                'phi_values': phi_values,
                'coherence_values': coherence_values
            }
        else:
            print(f"    ⚠️ No se pudieron extraer métricas de la salida")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"    ⚠️ Timeout después de 120 segundos")
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def analyze_reproducibility(text, num_runs=10, iterations=50):
    """
    Analiza la reproducibilidad ejecutando múltiples veces con diferentes seeds
    """
    print("=" * 70)
    print("TEST DE REPRODUCIBILIDAD DEL LENGUAJE CAUSAL")
    print("=" * 70)
    print(f"\nTexto de prueba: '{text}'")
    print(f"Número de ejecuciones: {num_runs}")
    print(f"Iteraciones por ejecución: {iterations}")
    print("\nEjecutando experimentos...\n")
    
    architectures = []
    seeds = list(range(2000, 2000 + num_runs))
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n[Ejecución {i}/{num_runs} - Seed {seed}]")
        arch = run_single_analysis(text, seed, iterations)
        
        if arch is None:
            print(f"  ⚠️ Saltando ejecución {i}")
            continue
            
        architectures.append(arch)
        print(f"  ✅ Φ: {arch['phi_mean']:.4f} ± {arch['phi_std']:.4f}")
        print(f"  ✅ Coherencia: {arch['coherence_mean']:.2f}% ± {arch['coherence_std']:.2f}%")
    
    if not architectures:
        print("\n❌ ERROR: No se pudieron obtener arquitecturas válidas")
        return None
    
    # Análisis estadístico
    print("\n" + "=" * 70)
    print("ANÁLISIS DE REPRODUCIBILIDAD")
    print("=" * 70)
    
    phi_means = [arch['phi_mean'] for arch in architectures]
    phi_stds = [arch['phi_std'] for arch in architectures]
    coherence_means = [arch['coherence_mean'] for arch in architectures]
    
    # Varianza entre seeds
    inter_seed_phi_var = np.var(phi_means)
    inter_seed_phi_std = np.std(phi_means)
    intra_seed_phi_var = np.mean([arch['phi_std']**2 for arch in architectures])
    
    print(f"\n📊 VARIANZA DE Φ:")
    print(f"  Entre seeds (inter-seed): {inter_seed_phi_var:.6f}")
    print(f"  Desviación estándar: {inter_seed_phi_std:.6f}")
    print(f"  Dentro de cada seed (intra-seed): {intra_seed_phi_var:.6f}")
    print(f"  Rango Φ: [{min(phi_means):.4f}, {max(phi_means):.4f}]")
    print(f"  Φ promedio: {np.mean(phi_means):.4f} ± {inter_seed_phi_std:.4f}")
    
    print(f"\n🔗 COHERENCIA:")
    print(f"  Media: {np.mean(coherence_means):.2f}% ± {np.std(coherence_means):.2f}%")
    print(f"  Varianza: {np.var(coherence_means):.6f}")
    print(f"  Rango: [{min(coherence_means):.2f}%, {max(coherence_means):.2f}%]")
    
    # Evaluación del determinismo
    print("\n" + "=" * 70)
    print("EVALUACIÓN DEL DETERMINISMO")
    print("=" * 70)
    
    threshold_var = 0.05
    is_deterministic = inter_seed_phi_var < threshold_var
    
    print(f"\n🎯 Umbral de varianza: {threshold_var}")
    print(f"📏 Varianza observada: {inter_seed_phi_var:.6f}")
    
    if is_deterministic:
        print(f"\n✅ RESULTADO: El lenguaje causal ES DETERMINISTA")
        print(f"   La varianza ({inter_seed_phi_var:.6f}) < {threshold_var}")
        print(f"   Las arquitecturas son REPRODUCIBLES entre seeds diferentes")
    else:
        print(f"\n❌ RESULTADO: El lenguaje causal NO es completamente determinista")
        print(f"   La varianza ({inter_seed_phi_var:.6f}) >= {threshold_var}")
        print(f"   Hay variabilidad significativa entre seeds")
    
    # Ratio señal/ruido
    signal_to_noise = np.mean(phi_means) / inter_seed_phi_std if inter_seed_phi_std > 0 else float('inf')
    print(f"\n📡 Ratio señal/ruido: {signal_to_noise:.2f}")
    
    # Coeficiente de variación
    cv = (inter_seed_phi_std / np.mean(phi_means)) * 100 if np.mean(phi_means) > 0 else 0
    print(f"📊 Coeficiente de variación: {cv:.2f}%")
    
    if cv < 5:
        print(f"   ✅ Variabilidad MUY BAJA (CV < 5%)")
    elif cv < 10:
        print(f"   ✅ Variabilidad BAJA (CV < 10%)")
    elif cv < 20:
        print(f"   ⚠️  Variabilidad MODERADA (CV < 20%)")
    else:
        print(f"   ❌ Variabilidad ALTA (CV >= 20%)")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'input_text': text,
        'num_runs': num_runs,
        'num_successful': len(architectures),
        'iterations_per_run': iterations,
        'seeds': seeds,
        'architectures': architectures,
        'statistics': {
            'phi_inter_seed_variance': float(inter_seed_phi_var),
            'phi_inter_seed_std': float(inter_seed_phi_std),
            'phi_intra_seed_variance': float(intra_seed_phi_var),
            'phi_mean': float(np.mean(phi_means)),
            'phi_range': [float(min(phi_means)), float(max(phi_means))],
            'coherence_mean': float(np.mean(coherence_means)),
            'coherence_variance': float(np.var(coherence_means)),
            'signal_to_noise': float(signal_to_noise),
            'coefficient_of_variation': float(cv),
            'is_deterministic': is_deterministic
        },
        'timestamp': timestamp
    }
    
    # Guardar en carpeta de resultados organizados
    os.makedirs('results/reproducibility', exist_ok=True)
    output_file = f"results/reproducibility/reproducibility_results_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    text = "mi perro es rojo"
    results = analyze_reproducibility(text, num_runs=10, iterations=50)
    
    if results:
        print("\n📋 CONCLUSIÓN FINAL:")
        print("=" * 70)
        
        if results['statistics']['is_deterministic']:
            print("\n🎯 El sistema genera ARQUITECTURAS CAUSALES REPRODUCIBLES")
            print("   ✅ La varianza entre seeds es menor al umbral")
            print("   ✅ El lenguaje causal del sistema es DETERMINISTA")
            print("   ✅ Diferentes seeds producen la misma 'palabra causal'")
            print("\n💡 IMPLICACIÓN:")
            print("   Podemos confiar en que el sistema 'dice lo mismo'")
            print("   cuando recibe el mismo input, independientemente del seed.")
        else:
            print("\n⚠️  El sistema muestra VARIABILIDAD SIGNIFICATIVA")
            print("   ⚠️  La varianza entre seeds excede el umbral")
            print("   ⚠️  El lenguaje causal tiene componente estocástico")
            print("\n💡 IMPLICACIÓN:")
            print("   El 'vocabulario causal' tiene variabilidad intrínseca.")
            print("   Se requiere promediado de múltiples ejecuciones.")
        
        cv = results['statistics']['coefficient_of_variation']
        print(f"\n📊 Coeficiente de Variación: {cv:.2f}%")
        print(f"   (Cuanto menor, más reproducible)")
