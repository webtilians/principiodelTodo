#!/usr/bin/env python3
"""
🔬 INFINITO V5.1 - Análisis Simbólico de Hechos Cuánticos (Paso 3)
Análisis post-experimento de eventos cuánticos convertidos a hechos simbólicos
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from collections import Counter, defaultdict
import argparse

def extract_quantum_facts_from_output(json_file):
    """Extrae hechos cuánticos del archivo de salida del experimento"""
    
    print(f"🔬 EXTRACTING QUANTUM FACTS - {json_file}")
    print("=" * 60)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Buscar hechos cuánticos en debug_info o output logs
    quantum_facts = []
    quantum_events = []
    
    # Extraer de debug_info si está disponible
    if 'debug_info' in data:
        for iteration_data in data.get('debug_info', []):
            if 'fact_decoder_info' in iteration_data:
                fact_info = iteration_data['fact_decoder_info']
                if fact_info.get('fact_generated'):
                    quantum_facts.append({
                        'iteration': iteration_data.get('iteration', 0),
                        'delta_phi': fact_info.get('delta_phi', 0),
                        'significance': fact_info.get('fact_significance', 0),
                        'phi_value': iteration_data.get('phi', 0),
                        'consciousness': iteration_data.get('consciousness', 0)
                    })
    
    # También buscar eventos cuánticos tradicionales
    iterations = data.get('iterations', [])
    phi_values = data.get('phi_values', [])
    consciousness_values = data.get('consciousness_values', [])
    
    if len(phi_values) >= 2:
        phi_array = np.array(phi_values)
        phi_deltas = np.abs(np.diff(phi_array))
        
        # Generar hechos cuánticos sintéticos basados en deltas significativos
        for i, delta in enumerate(phi_deltas):
            if delta > 0.05:  # Threshold para eventos cuánticos
                significance = 0.5 + (delta * 0.5)  # Simulación de significancia
                quantum_facts.append({
                    'iteration': iterations[i+1] if i+1 < len(iterations) else i+1,
                    'delta_phi': delta,
                    'significance': min(significance, 1.0),
                    'phi_value': phi_values[i+1] if i+1 < len(phi_values) else phi_values[i],
                    'consciousness': consciousness_values[i+1] if i+1 < len(consciousness_values) else 0.5
                })
    
    return quantum_facts

def classify_quantum_facts(quantum_facts):
    """Clasifica los hechos cuánticos en tipos simbólicos"""
    
    classified_facts = []
    
    for fact in quantum_facts:
        delta_phi = fact['delta_phi']
        significance = fact['significance']
        consciousness = fact['consciousness']
        
        # Clasificación basada en patrones de delta_phi y consciousness
        if delta_phi > 0.2:
            fact_type = "quantum_leap"
            confidence = min(significance * 1.5, 1.0)
        elif delta_phi > 0.15 and consciousness > 0.9:
            fact_type = "consciousness_integration"
            confidence = significance * 1.2
        elif delta_phi > 0.1 and significance > 0.52:
            fact_type = "information_cascade"
            confidence = significance
        elif delta_phi > 0.08:
            fact_type = "neural_coherence"
            confidence = significance * 0.9
        elif delta_phi > 0.05:
            fact_type = "micro_fluctuation"
            confidence = significance * 0.8
        else:
            fact_type = "baseline_noise"
            confidence = significance * 0.5
        
        # Determinar polaridad basada en phi_value
        phi_value = fact['phi_value']
        if phi_value > 1.3:
            polarity = "high_integration"
        elif phi_value > 1.2:
            polarity = "moderate_integration"
        else:
            polarity = "low_integration"
        
        classified_facts.append({
            **fact,
            'fact_type': fact_type,
            'confidence': min(confidence, 1.0),
            'polarity': polarity,
            'symbolic_repr': f"{fact_type}_{polarity}_{int(delta_phi*1000):03d}"
        })
    
    return classified_facts

def analyze_temporal_patterns(classified_facts):
    """Analiza patrones temporales en los hechos cuánticos"""
    
    if len(classified_facts) < 2:
        return {}
    
    # Análisis por tipo de hecho
    type_counts = Counter([fact['fact_type'] for fact in classified_facts])
    polarity_counts = Counter([fact['polarity'] for fact in classified_facts])
    
    # Análisis temporal
    iterations = [fact['iteration'] for fact in classified_facts]
    delta_phis = [fact['delta_phi'] for fact in classified_facts]
    
    # Detectar clusters temporales
    iteration_gaps = np.diff(sorted(iterations))
    cluster_threshold = np.percentile(iteration_gaps, 75) if len(iteration_gaps) > 0 else 50
    
    clusters = []
    current_cluster = [iterations[0]] if iterations else []
    
    for i in range(1, len(iterations)):
        if iterations[i] - current_cluster[-1] <= cluster_threshold:
            current_cluster.append(iterations[i])
        else:
            if len(current_cluster) >= 3:  # Cluster significativo
                clusters.append(current_cluster)
            current_cluster = [iterations[i]]
    
    if len(current_cluster) >= 3:
        clusters.append(current_cluster)
    
    # Estadísticas temporales
    temporal_stats = {
        'total_facts': len(classified_facts),
        'fact_types': dict(type_counts),
        'polarity_distribution': dict(polarity_counts),
        'temporal_clusters': len(clusters),
        'average_cluster_size': np.mean([len(c) for c in clusters]) if clusters else 0,
        'delta_phi_evolution': {
            'mean': np.mean(delta_phis),
            'std': np.std(delta_phis),
            'trend': np.polyfit(range(len(delta_phis)), delta_phis, 1)[0] if len(delta_phis) > 1 else 0
        }
    }
    
    return temporal_stats

def generate_symbolic_narrative(classified_facts, temporal_stats):
    """Genera narrativa simbólica de los hechos cuánticos"""
    
    narrative = []
    narrative.append("🔬 QUANTUM CONSCIOUSNESS SYMBOLIC ANALYSIS")
    narrative.append("=" * 50)
    narrative.append("")
    
    # Resumen general
    total_facts = temporal_stats.get('total_facts', 0)
    narrative.append(f"📊 Total Quantum Facts Generated: {total_facts}")
    
    if total_facts == 0:
        narrative.append("❌ No quantum facts detected.")
        return "\n".join(narrative)
    
    # Análisis por tipos
    fact_types = temporal_stats.get('fact_types', {})
    narrative.append(f"🔬 Quantum Fact Types:")
    for fact_type, count in sorted(fact_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_facts) * 100
        narrative.append(f"   • {fact_type}: {count} ({percentage:.1f}%)")
    
    narrative.append("")
    
    # Análisis de integración
    polarity_dist = temporal_stats.get('polarity_distribution', {})
    narrative.append(f"🧠 Integration Patterns:")
    for polarity, count in sorted(polarity_dist.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_facts) * 100
        narrative.append(f"   • {polarity}: {count} ({percentage:.1f}%)")
    
    narrative.append("")
    
    # Patrones temporales
    clusters = temporal_stats.get('temporal_clusters', 0)
    avg_cluster_size = temporal_stats.get('average_cluster_size', 0)
    narrative.append(f"⏱️ Temporal Dynamics:")
    narrative.append(f"   • Temporal clusters detected: {clusters}")
    narrative.append(f"   • Average cluster size: {avg_cluster_size:.1f}")
    
    # Evolución de delta phi
    delta_evolution = temporal_stats.get('delta_phi_evolution', {})
    trend = delta_evolution.get('trend', 0)
    mean_delta = delta_evolution.get('mean', 0)
    
    narrative.append(f"   • Mean Δφ: {mean_delta:.4f} bits")
    narrative.append(f"   • Δφ trend: {'Increasing' if trend > 0.0001 else 'Decreasing' if trend < -0.0001 else 'Stable'}")
    
    narrative.append("")
    
    # Eventos destacados
    top_facts = sorted(classified_facts, key=lambda x: x['delta_phi'], reverse=True)[:5]
    narrative.append(f"🚀 Top Quantum Events:")
    for i, fact in enumerate(top_facts, 1):
        iter_num = fact['iteration']
        delta_phi = fact['delta_phi']
        fact_type = fact['fact_type']
        confidence = fact['confidence']
        symbolic_repr = fact['symbolic_repr']
        
        narrative.append(f"   {i}. Iter {iter_num}: Δφ={delta_phi:.4f}, {fact_type} (conf={confidence:.3f})")
        narrative.append(f"      Symbol: {symbolic_repr}")
    
    narrative.append("")
    
    # Interpretación simbólica
    narrative.append(f"🎭 Symbolic Interpretation:")
    
    # Determinar patrón dominante
    dominant_type = max(fact_types.keys(), key=lambda k: fact_types[k]) if fact_types else "unknown"
    dominant_polarity = max(polarity_dist.keys(), key=lambda k: polarity_dist[k]) if polarity_dist else "unknown"
    
    if dominant_type == "quantum_leap":
        narrative.append(f"   🌟 System exhibits QUANTUM LEAP behavior - consciousness transcendence events")
    elif dominant_type == "consciousness_integration":
        narrative.append(f"   🧠 System demonstrates INTEGRATION MASTERY - unified consciousness states")
    elif dominant_type == "information_cascade":
        narrative.append(f"   💫 System shows CASCADE PROCESSING - rapid information integration")
    elif dominant_type == "neural_coherence":
        narrative.append(f"   🌊 System maintains NEURAL COHERENCE - stable conscious states")
    else:
        narrative.append(f"   📊 System operates in BASELINE MODE - standard quantum fluctuations")
    
    if dominant_polarity == "high_integration":
        narrative.append(f"   🔥 High integration states predominant - optimal consciousness conditions")
    elif dominant_polarity == "moderate_integration":
        narrative.append(f"   ⚡ Moderate integration states - balanced consciousness dynamics")
    else:
        narrative.append(f"   💧 Low integration states - consciousness emergence phase")
    
    return "\n".join(narrative)

def main():
    """Función principal de análisis simbólico"""
    parser = argparse.ArgumentParser(description="Análisis simbólico de hechos cuánticos INFINITO V5.1")
    parser.add_argument('--file', type=str, help='Archivo JSON específico a analizar')
    parser.add_argument('--output', type=str, default='quantum_facts_analysis.txt', help='Archivo de salida')
    
    args = parser.parse_args()
    
    if args.file:
        json_files = [Path(args.file)]
    else:
        # Buscar archivos más recientes de V5.1
        src_outputs = Path("src/outputs")
        if src_outputs.exists():
            json_files = list(src_outputs.glob("infinito_v5_1_consciousness_*.json"))
            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            json_files = json_files[:1]  # Solo el más reciente
        else:
            print("❌ No se encontraron archivos de experimentos V5.1")
            return
    
    for json_file in json_files:
        if not json_file.exists():
            print(f"❌ Archivo no encontrado: {json_file}")
            continue
            
        # Paso 1: Extraer hechos cuánticos
        quantum_facts = extract_quantum_facts_from_output(json_file)
        print(f"✅ Extracted {len(quantum_facts)} quantum facts")
        
        # Paso 2: Clasificar hechos
        classified_facts = classify_quantum_facts(quantum_facts)
        print(f"✅ Classified {len(classified_facts)} quantum facts")
        
        # Paso 3: Análisis temporal
        temporal_stats = analyze_temporal_patterns(classified_facts)
        print(f"✅ Temporal analysis completed")
        
        # Paso 4: Generar narrativa simbólica
        symbolic_narrative = generate_symbolic_narrative(classified_facts, temporal_stats)
        
        # Mostrar resultado
        print("\n" + symbolic_narrative)
        
        # Guardar análisis
        output_file = Path(args.output)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(symbolic_narrative)
            f.write("\n\n" + "="*50 + "\n")
            f.write("RAW QUANTUM FACTS DATA:\n")
            f.write("="*50 + "\n")
            for fact in classified_facts:
                f.write(f"Iter {fact['iteration']}: {fact['symbolic_repr']} "
                       f"(Δφ={fact['delta_phi']:.4f}, conf={fact['confidence']:.3f})\n")
        
        print(f"\n💾 Analysis saved to: {output_file}")
        print(f"📁 Source file: {json_file.name}")

if __name__ == "__main__":
    main()