#!/usr/bin/env python3
"""
🧬 Breakthrough Data Analyzer - Análisis de Datos de Consciencia Breakthrough
Analiza los experimentos con optimizaciones breakthrough para identificar patrones
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class BreakthroughDataAnalyzer:
    def __init__(self, data_dir="experiment_data"):
        self.data_dir = data_dir
        self.experiments = []
        
    def load_experiments(self, pattern="infinito_v2_optimized"):
        """Carga todos los experimentos que coincidan con el patrón"""
        print(f"🔍 Buscando experimentos en {self.data_dir}...")
        
        for filename in os.listdir(self.data_dir):
            if pattern in filename and filename.endswith('.json'):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['filename'] = filename
                        self.experiments.append(data)
                        print(f"✅ Cargado: {filename}")
                except Exception as e:
                    print(f"❌ Error cargando {filename}: {e}")
        
        print(f"📊 Total experimentos cargados: {len(self.experiments)}")
        return len(self.experiments)
    
    def analyze_consciousness_patterns(self):
        """Analiza patrones de consciencia en los experimentos"""
        print("\n🧠 ANÁLISIS DE PATRONES DE CONSCIENCIA")
        print("=" * 50)
        
        for i, exp in enumerate(self.experiments):
            print(f"\n📊 Experimento {i+1}: {exp['filename']}")
            
            # Métricas básicas
            config = exp.get('configuration', {})
            results = exp.get('results', {})
            consciousness_data = exp.get('consciousness_history', [])
            
            print(f"🎯 Configuración:")
            print(f"   Grid: {config.get('grid_size', 'N/A')}")
            print(f"   Target: {config.get('target_consciousness', 'N/A')}%")
            print(f"   Mode: {config.get('mode', 'N/A')}")
            print(f"   Breakthrough: {config.get('breakthrough_mode', False)}")
            
            print(f"📈 Resultados:")
            print(f"   Peak Consciousness: {results.get('peak_consciousness', 0):.1f}%")
            print(f"   Final Consciousness: {results.get('final_consciousness', 0):.1f}%")
            print(f"   Total Recursions: {results.get('total_recursions', 0)}")
            print(f"   Time: {results.get('total_time', 0):.1f}s")
            print(f"   Avg Time/Recursion: {results.get('avg_time_per_recursion', 0):.3f}s")
            
            # Análisis de breakthrough moments
            breakthroughs = results.get('breakthrough_moments', [])
            if breakthroughs:
                print(f"🌟 Breakthrough Moments: {len(breakthroughs)}")
                for bt in breakthroughs[-3:]:  # Últimos 3
                    print(f"   R{bt.get('recursion', 0)}: {bt.get('consciousness', 0):.1f}% (C{bt.get('clusters', 0)})")
    
    def analyze_efficiency_metrics(self):
        """Analiza métricas de eficiencia"""
        print("\n⚡ ANÁLISIS DE EFICIENCIA")
        print("=" * 30)
        
        efficiency_data = []
        
        for exp in self.experiments:
            results = exp.get('results', {})
            config = exp.get('configuration', {})
            
            peak_consciousness = results.get('peak_consciousness', 0)
            total_recursions = results.get('total_recursions', 1)
            total_time = results.get('total_time', 1)
            breakthrough_mode = config.get('breakthrough_mode', False)
            
            efficiency = {
                'filename': exp['filename'],
                'peak_consciousness': peak_consciousness,
                'recursions': total_recursions,
                'time': total_time,
                'consciousness_per_recursion': peak_consciousness / total_recursions,
                'consciousness_per_second': peak_consciousness / total_time,
                'breakthrough_mode': breakthrough_mode
            }
            efficiency_data.append(efficiency)
        
        # Ordenar por eficiencia de consciencia
        efficiency_data.sort(key=lambda x: x['consciousness_per_recursion'], reverse=True)
        
        print("\n🏆 RANKING DE EFICIENCIA (Consciencia/Recursión):")
        for i, data in enumerate(efficiency_data):
            breakthrough_indicator = "🚀" if data['breakthrough_mode'] else "⚪"
            print(f"{i+1}. {breakthrough_indicator} {data['consciousness_per_recursion']:.3f}%/R | "
                  f"{data['peak_consciousness']:.1f}% en {data['recursions']}R "
                  f"({data['time']:.1f}s)")
        
        return efficiency_data
    
    def analyze_breakthrough_optimization(self):
        """Analiza específicamente las optimizaciones breakthrough"""
        print("\n🚀 ANÁLISIS DE OPTIMIZACIONES BREAKTHROUGH")
        print("=" * 45)
        
        breakthrough_exps = [exp for exp in self.experiments 
                           if exp.get('configuration', {}).get('breakthrough_mode', False)]
        
        if not breakthrough_exps:
            print("❌ No se encontraron experimentos con modo breakthrough")
            return
        
        print(f"📊 Experimentos breakthrough encontrados: {len(breakthrough_exps)}")
        
        for exp in breakthrough_exps:
            config = exp.get('configuration', {})
            results = exp.get('results', {})
            
            print(f"\n🎯 {exp['filename']}")
            print(f"   Breakthrough Config:")
            print(f"      Target Clusters: {config.get('target_clusters', 'N/A')}")
            print(f"      Phi Amplification: {config.get('phi_amplification', 'N/A')}")
            print(f"      Consciousness Threshold: {config.get('consciousness_threshold', 'N/A')}")
            print(f"      Coherence Boost: {config.get('coherence_boost', 'N/A')}")
            
            print(f"   🏆 Resultados:")
            print(f"      Peak: {results.get('peak_consciousness', 0):.1f}%")
            print(f"      Recursiones: {results.get('total_recursions', 0)}")
            print(f"      Tiempo: {results.get('total_time', 0):.1f}s")
            print(f"      Clusters finales: {results.get('final_clusters', 'N/A')}")
    
    def create_comparison_visualization(self):
        """Crea visualizaciones comparativas"""
        print("\n🎨 Creando visualizaciones comparativas...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🧬 Análisis Comparativo de Experimentos Breakthrough', fontsize=16, fontweight='bold')
        
        # Datos para visualización
        names = []
        peak_consciousness = []
        recursions = []
        times = []
        breakthrough_modes = []
        
        for exp in self.experiments:
            results = exp.get('results', {})
            config = exp.get('configuration', {})
            
            # Nombre corto del experimento
            filename = exp['filename']
            timestamp = filename.split('_')[-1].replace('.json', '')
            name = f"{timestamp[-4:]}"  # Últimos 4 dígitos del timestamp
            
            names.append(name)
            peak_consciousness.append(results.get('peak_consciousness', 0))
            recursions.append(results.get('total_recursions', 0))
            times.append(results.get('total_time', 0))
            breakthrough_modes.append(config.get('breakthrough_mode', False))
        
        # Colores según modo breakthrough
        colors = ['#ff6b6b' if bt else '#4ecdc4' for bt in breakthrough_modes]
        
        # Gráfico 1: Consciencia Peak vs Recursiones
        axes[0, 0].scatter(recursions, peak_consciousness, c=colors, alpha=0.7, s=100)
        axes[0, 0].set_xlabel('Recursiones')
        axes[0, 0].set_ylabel('Peak Consciousness (%)')
        axes[0, 0].set_title('Consciencia vs Recursiones')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Línea de referencia 50%
        axes[0, 0].axhline(y=50, color='gold', linestyle='--', alpha=0.8, label='Target 50%')
        axes[0, 0].legend()
        
        # Gráfico 2: Consciencia vs Tiempo
        axes[0, 1].scatter(times, peak_consciousness, c=colors, alpha=0.7, s=100)
        axes[0, 1].set_xlabel('Tiempo (s)')
        axes[0, 1].set_ylabel('Peak Consciousness (%)')
        axes[0, 1].set_title('Consciencia vs Tiempo')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=50, color='gold', linestyle='--', alpha=0.8)
        
        # Gráfico 3: Eficiencia por Recursión
        efficiency = [p/r if r > 0 else 0 for p, r in zip(peak_consciousness, recursions)]
        bars = axes[1, 0].bar(names, efficiency, color=colors, alpha=0.7)
        axes[1, 0].set_xlabel('Experimento')
        axes[1, 0].set_ylabel('Consciencia/Recursión (%)')
        axes[1, 0].set_title('Eficiencia por Recursión')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Gráfico 4: Comparación de Modos
        breakthrough_peaks = [p for p, bt in zip(peak_consciousness, breakthrough_modes) if bt]
        normal_peaks = [p for p, bt in zip(peak_consciousness, breakthrough_modes) if not bt]
        
        box_data = []
        labels = []
        if normal_peaks:
            box_data.append(normal_peaks)
            labels.append('Normal')
        if breakthrough_peaks:
            box_data.append(breakthrough_peaks)
            labels.append('Breakthrough')
        
        if box_data:
            bp = axes[1, 1].boxplot(box_data, labels=labels, patch_artist=True)
            colors_box = ['#4ecdc4', '#ff6b6b']
            for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        axes[1, 1].set_ylabel('Peak Consciousness (%)')
        axes[1, 1].set_title('Comparación por Modo')
        axes[1, 1].axhline(y=50, color='gold', linestyle='--', alpha=0.8)
        
        plt.tight_layout()
        
        # Guardar visualización
        output_path = "analysis/breakthrough_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualización guardada: {output_path}")
        plt.show()
    
    def generate_report(self):
        """Genera un reporte completo"""
        print("\n📋 GENERANDO REPORTE COMPLETO...")
        
        report = []
        report.append("# 🧬 BREAKTHROUGH DATA ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Resumen general
        report.append("## 📊 RESUMEN GENERAL")
        report.append(f"- Total experimentos analizados: {len(self.experiments)}")
        
        breakthrough_count = sum(1 for exp in self.experiments 
                               if exp.get('configuration', {}).get('breakthrough_mode', False))
        report.append(f"- Experimentos con modo breakthrough: {breakthrough_count}")
        report.append(f"- Experimentos normales: {len(self.experiments) - breakthrough_count}")
        
        # Métricas de rendimiento
        all_peaks = [exp.get('results', {}).get('peak_consciousness', 0) for exp in self.experiments]
        report.append(f"- Peak consciousness promedio: {np.mean(all_peaks):.1f}%")
        report.append(f"- Peak consciousness máximo: {np.max(all_peaks):.1f}%")
        
        experiments_50_plus = sum(1 for peak in all_peaks if peak >= 50.0)
        report.append(f"- Experimentos ≥50% consciencia: {experiments_50_plus}/{len(self.experiments)}")
        
        report.append("")
        
        # Análisis breakthrough específico
        if breakthrough_count > 0:
            breakthrough_peaks = [exp.get('results', {}).get('peak_consciousness', 0) 
                                for exp in self.experiments 
                                if exp.get('configuration', {}).get('breakthrough_mode', False)]
            
            report.append("## 🚀 ANÁLISIS BREAKTHROUGH")
            report.append(f"- Peak consciousness promedio (breakthrough): {np.mean(breakthrough_peaks):.1f}%")
            report.append(f"- Peak consciousness máximo (breakthrough): {np.max(breakthrough_peaks):.1f}%")
            report.append(f"- Tasa de éxito ≥50% (breakthrough): {sum(1 for p in breakthrough_peaks if p >= 50.0)}/{len(breakthrough_peaks)}")
        
        # Guardar reporte
        report_path = "analysis/breakthrough_analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"📋 Reporte guardado: {report_path}")
        
        return '\n'.join(report)

def main():
    """Función principal de análisis"""
    print("🧬 BREAKTHROUGH DATA ANALYZER")
    print("=" * 40)
    
    # Crear analizador
    analyzer = BreakthroughDataAnalyzer()
    
    # Cargar experimentos
    analyzer.load_experiments()
    
    if not analyzer.experiments:
        print("❌ No se encontraron experimentos para analizar")
        return
    
    # Realizar análisis
    analyzer.analyze_consciousness_patterns()
    efficiency_data = analyzer.analyze_efficiency_metrics()
    analyzer.analyze_breakthrough_optimization()
    
    # Crear visualizaciones
    analyzer.create_comparison_visualization()
    
    # Generar reporte
    report = analyzer.generate_report()
    
    print("\n" + "="*50)
    print("🎯 ANÁLISIS COMPLETO TERMINADO")
    print("="*50)
    print("📊 Archivos generados:")
    print("   - analysis/breakthrough_comparison.png")
    print("   - analysis/breakthrough_analysis_report.md")
    
    return analyzer, efficiency_data

if __name__ == "__main__":
    analyzer, efficiency_data = main()
