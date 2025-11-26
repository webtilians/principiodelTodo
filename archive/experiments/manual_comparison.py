#!/usr/bin/env python3
"""
📊 ANÁLISIS MANUAL COMPARATIVO - Texto Extendido vs Sin Texto
"""

import json
import numpy as np
from scipy import stats
from datetime import datetime

def load_experiment_data(filename):
    """Cargar datos de experimento desde JSON"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error cargando {filename}: {e}")
        return None

def extract_stable_window(data, start_iter=11, end_iter=75):
    """Extraer ventana estable de los datos"""
    iterations = data.get('iterations', [])
    consciousness = data.get('consciousness_values', [])
    phi = data.get('phi_values', [])
    
    # Encontrar índices correspondientes a la ventana
    start_idx = next((i for i, iter_num in enumerate(iterations) if iter_num >= start_iter), 0)
    end_idx = next((i for i, iter_num in enumerate(iterations) if iter_num > end_iter), len(iterations))
    
    stable_consciousness = consciousness[start_idx:end_idx]
    stable_phi = phi[start_idx:end_idx]
    
    return stable_consciousness, stable_phi, end_idx - start_idx

def analyze_text_content(data):
    """Analizar contenido del texto si está presente"""
    runtime_state = data.get('runtime_state', {})
    
    if runtime_state.get('has_input_text', False):
        # Buscar texto en config
        config = data.get('config', {})
        input_text = config.get('input_text', 'No disponible')
        
        word_count = len(input_text.split()) if input_text != 'No disponible' else 0
        
        return {
            'has_text': True,
            'text': input_text,
            'word_count': word_count,
            'text_mode': runtime_state.get('text_mode', False)
        }
    else:
        return {
            'has_text': False,
            'text': None,
            'word_count': 0,
            'text_mode': False
        }

def main():
    print("📊 ANÁLISIS COMPARATIVO MANUAL - TEXTO EXTENDIDO vs SIN TEXTO")
    print("=" * 70)
    
    # Archivos a analizar
    files_to_analyze = [
        "outputs/infinito_v5_1_consciousness_TEXT_20251001_130733_C0.931_PHI0.291.json",  # Texto extendido
        "outputs/infinito_v5_1_consciousness_20251001_132648_C0.929_PHI0.290.json"        # Sin texto
    ]
    
    results = {}
    
    for filename in files_to_analyze:
        print(f"\n📁 Analizando: {filename}")
        data = load_experiment_data(filename)
        
        if data is None:
            continue
            
        # Análisis del contenido
        text_info = analyze_text_content(data)
        
        # Extraer ventana estable
        stable_c, stable_phi, n_points = extract_stable_window(data)
        
        condition = "ON" if text_info['has_text'] else "OFF"
        
        results[condition] = {
            'filename': filename,
            'text_info': text_info,
            'stable_consciousness': stable_c,
            'stable_phi': stable_phi,
            'n_points': n_points,
            'final_c': data.get('final_consciousness', 0),
            'final_phi': data.get('final_phi', 0),
            'max_c': data.get('max_consciousness', 0)
        }
        
        print(f"   Condición: {condition}")
        print(f"   Texto: {text_info['text'][:50] if text_info['text'] else 'N/A'}{'...' if text_info['text'] and len(text_info['text']) > 50 else ''}")
        print(f"   Palabras: {text_info['word_count']}")
        print(f"   Puntos estables: {n_points}")
        print(f"   C final: {data.get('final_consciousness', 0):.4f}")
        print(f"   Φ final: {data.get('final_phi', 0):.4f}")
    
    # Análisis comparativo
    if 'ON' in results and 'OFF' in results:
        print(f"\n📊 ANÁLISIS ESTADÍSTICO COMPARATIVO")
        print("=" * 50)
        
        # Extraer datos
        consciousness_on = results['ON']['stable_consciousness']
        consciousness_off = results['OFF']['stable_consciousness']
        phi_on = results['ON']['stable_phi']
        phi_off = results['OFF']['stable_phi']
        
        # Asegurar mismo tamaño
        min_len = min(len(consciousness_on), len(consciousness_off), len(phi_on), len(phi_off))
        consciousness_on = consciousness_on[:min_len]
        consciousness_off = consciousness_off[:min_len]
        phi_on = phi_on[:min_len]
        phi_off = phi_off[:min_len]
        
        # Calcular diferencias
        delta_c = np.array(consciousness_on) - np.array(consciousness_off)
        delta_phi = np.array(phi_on) - np.array(phi_off)
        
        # Estadísticas básicas
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
        
        # Información del texto
        text_on = results['ON']['text_info']
        text_off = results['OFF']['text_info']
        
        print(f"\n🔤 INFORMACIÓN DEL TEXTO:")
        print(f"   ON:  '{text_on['text'][:60] if text_on['text'] else 'N/A'}{'...' if text_on['text'] and len(text_on['text']) > 60 else ''}'")
        print(f"   OFF: Sin texto")
        print(f"   Palabras ON: {text_on['word_count']}")
        print(f"   Incremento vs texto original (9 palabras): {(text_on['word_count'] / 9 - 1) * 100:.1f}%")
        
        # Resultados
        print(f"\n🎯 RESULTADOS COMPARATIVOS (Ventana estable iter 11-75):")
        print(f"   📈 ΔC (ON - OFF): {mean_delta_c:.4f} ± {std_delta_c:.4f}")
        print(f"      IC95%: [{ci_delta_c[0]:.4f}, {ci_delta_c[1]:.4f}]")
        print(f"      t={t_stat_c:.3f}, p={p_val_c:.6f} {'***' if p_val_c < 0.001 else '**' if p_val_c < 0.01 else '*' if p_val_c < 0.05 else 'n.s.'}")
        
        print(f"\n   🔬 ΔΦ (ON - OFF): {mean_delta_phi:.4f} ± {std_delta_phi:.4f}")
        print(f"      IC95%: [{ci_delta_phi[0]:.4f}, {ci_delta_phi[1]:.4f}]")
        print(f"      t={t_stat_phi:.3f}, p={p_val_phi:.6f} {'***' if p_val_phi < 0.001 else '**' if p_val_phi < 0.01 else '*' if p_val_phi < 0.05 else 'n.s.'}")
        
        print(f"\n   📊 Puntos analizados: {min_len}")
        
        # Interpretación
        print(f"\n🧠 INTERPRETACIÓN DEL TEXTO EXTENDIDO:")
        if p_val_c < 0.05:
            direction_c = "AUMENTA" if mean_delta_c > 0 else "DISMINUYE"
            effect_size_c = abs(mean_delta_c) / std_delta_c
            print(f"   ✅ El texto extendido {direction_c} significativamente la consciencia")
            print(f"      Tamaño del efecto: {effect_size_c:.3f}")
        else:
            print(f"   ➖ No hay efecto significativo del texto extendido en consciencia")
        
        if p_val_phi < 0.05:
            direction_phi = "AUMENTA" if mean_delta_phi > 0 else "DISMINUYE"
            effect_size_phi = abs(mean_delta_phi) / std_delta_phi
            print(f"   ✅ El texto extendido {direction_phi} significativamente Φ")
            print(f"      Tamaño del efecto: {effect_size_phi:.3f}")
        else:
            print(f"   ➖ No hay efecto significativo del texto extendido en Φ")
        
        # Comparación con experimento anterior (texto corto)
        print(f"\n🔍 COMPARACIÓN CON HALLAZGOS PREVIOS:")
        print(f"   Texto original (9 palabras): ΔC≈-0.014***, ΔΦ≈-0.009***") 
        print(f"   Texto extendido ({text_on['word_count']} palabras): ΔC={mean_delta_c:.4f}{'***' if p_val_c < 0.001 else '**' if p_val_c < 0.01 else '*' if p_val_c < 0.05 else 'n.s.'}, ΔΦ={mean_delta_phi:.4f}{'***' if p_val_phi < 0.001 else '**' if p_val_phi < 0.01 else '*' if p_val_phi < 0.05 else 'n.s.'}")
        
        # Guardar resultados
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_results = {
            'experiment_type': 'manual_comparative_extended_text',
            'analysis_timestamp': datetime.now().isoformat(),
            'text_comparison': {
                'original_words': 9,
                'extended_words': text_on['word_count'],
                'word_increase_percent': (text_on['word_count'] / 9 - 1) * 100
            },
            'source_files': {
                'on_file': results['ON']['filename'],
                'off_file': results['OFF']['filename']
            },
            'stable_window': {'start': 11, 'end': 75, 'points': min_len},
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
            'effect_sizes': {
                'consciousness': abs(mean_delta_c) / std_delta_c if std_delta_c > 0 else 0,
                'phi': abs(mean_delta_phi) / std_delta_phi if std_delta_phi > 0 else 0
            }
        }
        
        filename = f"manual_comparative_extended_text_{timestamp}_deltaC{mean_delta_c:.3f}_deltaPhi{mean_delta_phi:.3f}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Análisis guardado: {filename}")
        
    else:
        print(f"\n❌ No se pueden realizar comparaciones - faltan condiciones")
        print(f"   Disponibles: {list(results.keys())}")

if __name__ == "__main__":
    main()