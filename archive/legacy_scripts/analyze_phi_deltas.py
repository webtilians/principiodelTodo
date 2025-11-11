#!/usr/bin/env python3
"""
🔬 INFINITO V5.1 - Análisis de Deltas Φ (Paso 2)
Análisis de variabilidad cuántica en φ con reproducibilidad (seed=42)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def analyze_phi_deltas(json_file):
    """Analiza deltas de Φ en los datos del experimento"""
    
    print(f"🔬 ANÁLISIS DE DELTAS Φ - {json_file}")
    print("=" * 60)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extraer valores de Φ
    phi_values = data.get('phi_values', [])
    iterations = data.get('iterations', [])
    consciousness_values = data.get('consciousness_values', [])
    
    if len(phi_values) < 2:
        print("⚠️  No hay suficientes datos para calcular deltas")
        return
    
    # Calcular deltas Φ
    phi_array = np.array(phi_values)
    phi_deltas = np.abs(np.diff(phi_array))
    
    print(f"📊 ESTADÍSTICAS BÁSICAS:")
    print(f"   Total iteraciones: {len(phi_values)}")
    print(f"   Φ promedio: {np.mean(phi_values):.4f} bits")
    print(f"   Φ std: {np.std(phi_values):.4f} bits")
    print(f"   Φ min: {np.min(phi_values):.4f} bits")
    print(f"   Φ max: {np.max(phi_values):.4f} bits")
    print()
    
    print(f"🔬 ANÁLISIS DE DELTAS Φ:")
    print(f"   Delta promedio: {np.mean(phi_deltas):.4f} bits")
    print(f"   Delta std: {np.std(phi_deltas):.4f} bits")
    print(f"   Delta max: {np.max(phi_deltas):.4f} bits")
    print(f"   Delta min: {np.min(phi_deltas):.4f} bits")
    print()
    
    # Análisis de significancia cuántica (>0.05)
    significant_deltas = phi_deltas[phi_deltas > 0.05]
    print(f"⚡ EVENTOS CUÁNTICOS DETECTADOS (Δφ > 0.05):")
    print(f"   Total eventos: {len(significant_deltas)}")
    print(f"   Porcentaje: {len(significant_deltas)/len(phi_deltas)*100:.1f}%")
    
    if len(significant_deltas) > 0:
        print(f"   Delta cuántico promedio: {np.mean(significant_deltas):.4f} bits")
        print(f"   Delta cuántico max: {np.max(significant_deltas):.4f} bits")
        
        # Listar los eventos cuánticos más significativos
        print(f"\n🚀 TOP 5 EVENTOS CUÁNTICOS:")
        top_indices = np.argsort(phi_deltas)[-5:][::-1]
        for i, idx in enumerate(top_indices):
            if idx < len(iterations)-1:
                delta_val = phi_deltas[idx]
                if delta_val > 0.01:  # Solo mostrar deltas relevantes
                    print(f"   {i+1}. Iter {iterations[idx]}->{iterations[idx+1]}: Δφ={delta_val:.4f}")
    
    print()
    
    # Análisis de correlación con consciencia
    if len(consciousness_values) == len(phi_values):
        phi_cons_corr = np.corrcoef(phi_values, consciousness_values)[0,1]
        print(f"🧠 CORRELACIÓN Φ-CONSCIENCIA: {phi_cons_corr:.4f}")
        
        # Análisis dinámico: consciencia cuando hay eventos cuánticos
        consciousness_array = np.array(consciousness_values[1:])  # Alinear con deltas
        quantum_consciousness = consciousness_array[phi_deltas > 0.05]
        if len(quantum_consciousness) > 0:
            print(f"   Consciencia durante eventos cuánticos: {np.mean(quantum_consciousness):.4f}")
            print(f"   Consciencia general: {np.mean(consciousness_values):.4f}")
    
    print()
    
    # Determinar si el ruido cuántico está funcionando
    variability_score = np.std(phi_deltas)
    print(f"📈 EVALUACIÓN DEL RUIDO CUÁNTICO:")
    print(f"   Variabilidad Φ: {variability_score:.4f}")
    
    if variability_score > 0.02:
        print("   ✅ Sistema cuántico funcionando correctamente")
        print("   ✅ Variabilidad suficiente para FactDecoder")
    elif variability_score > 0.01:
        print("   ⚠️  Variabilidad moderada - considerar aumentar ruido")
    else:
        print("   ❌ Variabilidad baja - necesita más ruido cuántico")
    
    print()
    print("🔍 RECOMENDACIÓN PARA FACTDECODER:")
    target_events = len(significant_deltas)
    if target_events > 20:
        print(f"   ✅ {target_events} eventos cuánticos detectados")
        print("   ✅ FactDecoder tendrá suficientes datos para análisis")
    else:
        print(f"   ⚠️  Solo {target_events} eventos cuánticos")
        print("   📊 Considerar reducir threshold o aumentar ruido")
    
    return {
        'mean_phi': np.mean(phi_values),
        'std_phi': np.std(phi_values),
        'mean_delta': np.mean(phi_deltas),
        'std_delta': np.std(phi_deltas),
        'quantum_events': len(significant_deltas),
        'quantum_percentage': len(significant_deltas)/len(phi_deltas)*100,
        'variability_score': variability_score
    }

def main():
    """Función principal de análisis"""
    parser = argparse.ArgumentParser(description="Análisis de deltas Φ INFINITO V5.1")
    parser.add_argument('--file', type=str, help='Archivo JSON específico a analizar')
    
    args = parser.parse_args()
    
    if args.file:
        json_files = [Path(args.file)]
    else:
        # Buscar archivos más recientes
        src_outputs = Path("src/outputs")
        if src_outputs.exists():
            json_files = list(src_outputs.glob("infinito_v5_1_consciousness_*.json"))
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            json_files = json_files[:1]  # Solo el más reciente
        else:
            print("❌ No se encontraron archivos de experimentos")
            return
    
    for json_file in json_files:
        if json_file.exists():
            stats = analyze_phi_deltas(json_file)
            print(f"\n📋 RESUMEN FINAL:")
            print(f"   Archivo: {json_file.name}")
            print(f"   Eventos cuánticos: {stats['quantum_events']}")
            print(f"   Variabilidad: {stats['variability_score']:.4f}")
            print(f"   Recomendación: {'FactDecoder READY' if stats['quantum_events'] > 15 else 'AJUSTAR RUIDO'}")

if __name__ == "__main__":
    main()