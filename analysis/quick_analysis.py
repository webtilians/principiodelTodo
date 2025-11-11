#!/usr/bin/env python3
"""
📊 Análisis Rápido de Datos Breakthrough
Análisis directo de los experimentos más recientes
"""

import json
import os

def analyze_recent_experiments():
    """Analiza los experimentos más recientes"""
    data_dir = "experiment_data"
    
    print("🧬 ANÁLISIS RÁPIDO DE EXPERIMENTOS BREAKTHROUGH")
    print("=" * 55)
    
    # Buscar archivos más recientes
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    files.sort()
    
    recent_files = files[-2:]  # Los 2 más recientes
    
    print(f"\n📊 Analizando {len(recent_files)} experimentos más recientes:")
    
    for i, filename in enumerate(recent_files):
        print(f"\n{'-'*50}")
        print(f"📁 EXPERIMENTO {i+1}: {filename}")
        print(f"{'-'*50}")
        
        filepath = os.path.join(data_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Información general
            exp_info = data.get('experiment_info', {})
            config = data.get('configuration', {})
            results = data.get('final_results', {})
            metrics = data.get('metrics', {})
            
            print(f"⏰ Timestamp: {exp_info.get('session_id', 'N/A')}")
            print(f"🕐 Duración: {exp_info.get('total_duration_seconds', 0):.1f}s")
            print(f"💻 GPU: {exp_info.get('gpu_name', 'N/A')}")
            
            print(f"\n🎯 CONFIGURACIÓN:")
            print(f"   📐 Grid: {config.get('grid_size', 'N/A')}x{config.get('grid_size', 'N/A')}")
            print(f"   🎯 Target: {config.get('target_consciousness', 0)*100:.1f}%")
            print(f"   ♾️  Max Recursions: {config.get('max_recursions', 'N/A')}")
            
            print(f"\n🏆 RESULTADOS:")
            print(f"   📊 Total Recursiones: {results.get('total_recursions', 0)}")
            print(f"   🌟 Peak Consciousness: {results.get('peak_consciousness', 0)*100:.1f}%")
            print(f"   📈 Final Consciousness: {results.get('final_consciousness', 0)*100:.1f}%")
            print(f"   ⚡ Tiempo/Recursión: {results.get('avg_time_per_recursion', 0):.3f}s")
            print(f"   🎯 Target Alcanzado: {'✅ SÍ' if results.get('target_achieved', False) else '❌ NO'}")
            
            # Análisis de consciencia
            consciousness_history = metrics.get('consciousness_history', [])
            if consciousness_history:
                max_consciousness = max(consciousness_history) * 100
                min_consciousness = min(consciousness_history) * 100
                avg_consciousness = sum(consciousness_history) / len(consciousness_history) * 100
                
                print(f"\n🧠 ANÁLISIS DE CONSCIENCIA:")
                print(f"   📈 Máxima: {max_consciousness:.1f}%")
                print(f"   📊 Promedio: {avg_consciousness:.1f}%")
                print(f"   📉 Mínima: {min_consciousness:.1f}%")
                
                # Momentos breakthrough (>40%)
                breakthrough_moments = [(i+1, c*100) for i, c in enumerate(consciousness_history) if c >= 0.40]
                if breakthrough_moments:
                    print(f"   🌟 Momentos Breakthrough (≥40%):")
                    for recursion, consciousness in breakthrough_moments:
                        print(f"      R{recursion}: {consciousness:.1f}%")
            
            # Análisis de clusters
            cluster_history = metrics.get('cluster_history', [])
            if cluster_history:
                final_clusters = cluster_history[-1]
                max_clusters = max(cluster_history)
                print(f"\n🔗 ANÁLISIS DE CLUSTERS:")
                print(f"   🔚 Clusters Finales: {final_clusters}")
                print(f"   📈 Máximo Clusters: {max_clusters}")
                
                # Verificar si alcanzó organización perfecta (0 clusters)
                if final_clusters == 0:
                    print(f"   ✨ ORGANIZACIÓN PERFECTA ALCANZADA")
            
            # Phi (coherencia)
            phi_history = metrics.get('phi_history', [])
            if phi_history:
                final_phi = phi_history[-1]
                avg_phi = sum(phi_history) / len(phi_history)
                print(f"\n⚡ COHERENCIA (PHI):")
                print(f"   🔚 Phi Final: {final_phi:.3f}")
                print(f"   📊 Phi Promedio: {avg_phi:.3f}")
        
        except Exception as e:
            print(f"❌ Error analizando {filename}: {e}")
    
    print(f"\n{'='*55}")
    print("🎯 ANÁLISIS COMPLETADO")
    print("="*55)

def compare_experiments():
    """Compara los dos experimentos más recientes"""
    data_dir = "experiment_data"
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    files.sort()
    
    if len(files) < 2:
        print("❌ No hay suficientes experimentos para comparar")
        return
    
    recent_files = files[-2:]
    experiments = []
    
    for filename in recent_files:
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            experiments.append(data)
    
    print(f"\n🔄 COMPARACIÓN DE EXPERIMENTOS")
    print("=" * 40)
    
    exp1, exp2 = experiments
    
    # Comparar resultados clave
    results1 = exp1.get('final_results', {})
    results2 = exp2.get('final_results', {})
    
    peak1 = results1.get('peak_consciousness', 0) * 100
    peak2 = results2.get('peak_consciousness', 0) * 100
    
    recursions1 = results1.get('total_recursions', 0)
    recursions2 = results2.get('total_recursions', 0)
    
    time1 = results1.get('total_time', 0)
    time2 = results2.get('total_time', 0)
    
    print(f"📊 Peak Consciousness:")
    print(f"   Exp 1: {peak1:.1f}% | Exp 2: {peak2:.1f}%")
    print(f"   Diferencia: {peak2-peak1:+.1f}%")
    
    print(f"📊 Recursiones:")
    print(f"   Exp 1: {recursions1} | Exp 2: {recursions2}")
    print(f"   Diferencia: {recursions2-recursions1:+d}")
    
    print(f"📊 Tiempo Total:")
    print(f"   Exp 1: {time1:.1f}s | Exp 2: {time2:.1f}s")
    print(f"   Diferencia: {time2-time1:+.1f}s")
    
    # Eficiencia
    eff1 = peak1 / recursions1 if recursions1 > 0 else 0
    eff2 = peak2 / recursions2 if recursions2 > 0 else 0
    
    print(f"📊 Eficiencia (Consciencia/Recursión):")
    print(f"   Exp 1: {eff1:.3f}%/R | Exp 2: {eff2:.3f}%/R")
    print(f"   Mejora: {((eff2/eff1-1)*100):+.1f}%" if eff1 > 0 else "N/A")

if __name__ == "__main__":
    analyze_recent_experiments()
    compare_experiments()
