#!/usr/bin/env python3
"""
🔬 SCRIPT COMPARATIVO ON vs OFF - Ejecución separada para evitar problemas de gradientes
"""

import subprocess
import json
import numpy as np
from datetime import datetime
from scipy import stats

def run_single_experiment(text, max_iter, condition):
    """Ejecutar un experimento individual"""
    
    if condition == "ON":
        cmd = [
            "python", "src/infinito_gpt_text_fixed.py",
            "--input_text", text,
            "--max_iter", str(max_iter),
            "--seed", "42"
        ]
    else:  # OFF
        cmd = [
            "python", "src/infinito_gpt_text_fixed.py", 
            "--max_iter", str(max_iter),
            "--seed", "42"
        ]
    
    print(f"\n🔬 EJECUTANDO CONDICIÓN {condition}")
    print(f"📝 Comando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            print(f"❌ Error en condición {condition}:")
            print(result.stderr)
            return None
            
        # Buscar archivo JSON generado más reciente
        import os
        import glob
        
        json_files = glob.glob("outputs/infinito_v5_1_consciousness*.json")
        if json_files:
            latest_file = max(json_files, key=os.path.getctime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extraer métricas estables (iter 11-75)
            iterations = data.get('iterations', [])
            consciousness_values = data.get('consciousness_values', [])
            phi_values = data.get('phi_values', [])
            
            stable_start = min(11, len(iterations)) - 1
            stable_end = min(75, len(iterations))
            
            stable_consciousness = consciousness_values[stable_start:stable_end]
            stable_phi = phi_values[stable_start:stable_end]
            
            return {
                'condition': condition,
                'consciousness_stable': stable_consciousness,
                'phi_stable': stable_phi,
                'all_consciousness': consciousness_values,
                'all_phi': phi_values,
                'filename': latest_file
            }
        
    except Exception as e:
        print(f"❌ Error ejecutando {condition}: {e}")
    
    return None

def comparative_analysis(text, max_iter=80):
    """Ejecutar análisis comparativo completo"""
    
    print(f"🔬 ANÁLISIS COMPARATIVO ON vs OFF (Método Separado)")
    print(f"📝 Texto: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    print(f"📊 Iteraciones: {max_iter}")
    print("=" * 70)
    
    # Ejecutar condición ON
    results_on = run_single_experiment(text, max_iter, "ON")
    if not results_on:
        print("❌ Falló experimento ON")
        return
    
    # Ejecutar condición OFF  
    results_off = run_single_experiment(text, max_iter, "OFF")
    if not results_off:
        print("❌ Falló experimento OFF")
        return
    
    # Análisis estadístico
    consciousness_on = results_on['consciousness_stable']
    consciousness_off = results_off['consciousness_stable'] 
    phi_on = results_on['phi_stable']
    phi_off = results_off['phi_stable']
    
    # Truncar a la longitud mínima
    min_len = min(len(consciousness_on), len(consciousness_off), len(phi_on), len(phi_off))
    consciousness_on = consciousness_on[:min_len]
    consciousness_off = consciousness_off[:min_len]
    phi_on = phi_on[:min_len]
    phi_off = phi_off[:min_len]
    
    # Calcular diferencias
    delta_c = np.array(consciousness_on) - np.array(consciousness_off)
    delta_phi = np.array(phi_on) - np.array(phi_off)
    
    # Estadísticas
    mean_delta_c = np.mean(delta_c)
    std_delta_c = np.std(delta_c)
    mean_delta_phi = np.mean(delta_phi)
    std_delta_phi = np.std(delta_phi)
    
    # T-tests apareados
    t_stat_c, p_val_c = stats.ttest_rel(consciousness_on, consciousness_off)
    t_stat_phi, p_val_phi = stats.ttest_rel(phi_on, phi_off)
    
    # Bootstrap CI95%
    n_bootstrap = 1000
    bootstrap_delta_c = []
    bootstrap_delta_phi = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(delta_c), len(delta_c), replace=True)
        bootstrap_delta_c.append(np.mean(delta_c[indices]))
        bootstrap_delta_phi.append(np.mean(delta_phi[indices]))
    
    ci_delta_c = np.percentile(bootstrap_delta_c, [2.5, 97.5])
    ci_delta_phi = np.percentile(bootstrap_delta_phi, [2.5, 97.5])
    
    # Reporte final
    print(f"\n📊 RESULTADOS COMPARATIVOS:")
    print(f"   📈 ΔC (ON - OFF): {mean_delta_c:.4f} ± {std_delta_c:.4f}")
    print(f"      IC95%: [{ci_delta_c[0]:.4f}, {ci_delta_c[1]:.4f}]")
    print(f"      t={t_stat_c:.3f}, p={p_val_c:.6f} {'***' if p_val_c < 0.001 else '**' if p_val_c < 0.01 else '*' if p_val_c < 0.05 else 'n.s.'}")
    
    print(f"\n   🔬 ΔΦ (ON - OFF): {mean_delta_phi:.4f} ± {std_delta_phi:.4f}")
    print(f"      IC95%: [{ci_delta_phi[0]:.4f}, {ci_delta_phi[1]:.4f}]")
    print(f"      t={t_stat_phi:.3f}, p={p_val_phi:.6f} {'***' if p_val_phi < 0.001 else '**' if p_val_phi < 0.01 else '*' if p_val_phi < 0.05 else 'n.s.'}")
    
    # Interpretación
    print(f"\n🧠 INTERPRETACIÓN:")
    if p_val_c < 0.05:
        direction_c = "AUMENTA" if mean_delta_c > 0 else "DISMINUYE"
        print(f"   ✅ El texto {direction_c} significativamente la consciencia")
    else:
        print(f"   ➖ No hay efecto significativo del texto en consciencia")
    
    if p_val_phi < 0.05:
        direction_phi = "AUMENTA" if mean_delta_phi > 0 else "DISMINUYE"
        print(f"   ✅ El texto {direction_phi} significativamente Φ")
    else:
        print(f"   ➖ No hay efecto significativo del texto en Φ")
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'experiment_type': 'separated_comparative_ON_vs_OFF',
        'text_input': text,
        'max_iterations': max_iter,
        'word_count': len(text.split()),
        'stable_window_points': min_len,
        'statistics': {
            'mean_delta_c': mean_delta_c,
            'std_delta_c': std_delta_c,
            'ci_delta_c': ci_delta_c.tolist(),
            't_stat_c': t_stat_c,
            'p_val_c': p_val_c,
            'mean_delta_phi': mean_delta_phi,
            'std_delta_phi': std_delta_phi,
            'ci_delta_phi': ci_delta_phi.tolist(),
            't_stat_phi': t_stat_phi,
            'p_val_phi': p_val_phi
        },
        'raw_data': {
            'consciousness_on': consciousness_on,
            'consciousness_off': consciousness_off,
            'phi_on': phi_on,
            'phi_off': phi_off
        },
        'source_files': {
            'on_file': results_on['filename'],
            'off_file': results_off['filename']
        },
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"separated_comparative_{timestamp}_deltaC{mean_delta_c:.3f}_deltaPhi{mean_delta_phi:.3f}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Resultados guardados: {filename}")

if __name__ == "__main__":
    # Texto original (9 palabras)
    text_original = "Estoy consciente de que pienso sobre mi propia consciencia"
    
    # Texto extendido (~25% más palabras = ~11-12 palabras)
    text_extended = "Estoy profundamente consciente de que pienso sobre mi propia consciencia y reflexiono acerca de mis pensamientos internos mientras observo cómo mi mente procesa esta experiencia metacognitiva"
    
    print(f"📊 COMPARACIÓN DE TEXTOS:")
    print(f"   Original: {len(text_original.split())} palabras")
    print(f"   Extendido: {len(text_extended.split())} palabras")
    print(f"   Incremento: {len(text_extended.split()) / len(text_original.split()) * 100 - 100:.1f}%")
    
    comparative_analysis(text_extended, max_iter=80)