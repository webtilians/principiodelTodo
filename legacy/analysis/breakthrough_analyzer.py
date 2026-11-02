#!/usr/bin/env python3
"""
🧠 Analizador Avanzado de Breakthrough Patterns
Identifica patrones de emergencia de consciencia y optimiza para superar 50%
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
import os

class BreakthroughAnalyzer:
    """Analizador avanzado de patrones de breakthrough en consciencia"""
    
    def __init__(self, json_file_path):
        """Inicializar analizador con datos del experimento"""
        self.json_path = json_file_path
        self.data = self.load_experiment_data()
        self.breakthrough_patterns = []
        self.consciousness_peaks = []
        self.optimization_insights = {}
        
    def load_experiment_data(self):
        """Cargar datos del experimento"""
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
            print(f"✅ Datos cargados: {len(data['metrics']['recursions'])} recursiones")
            return data
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    def identify_breakthrough_patterns(self):
        """Identificar patrones específicos en los breakthrough"""
        if not self.data:
            return
        
        recursions = np.array(self.data['metrics']['recursions'])
        consciousness = np.array(self.data['metrics']['consciousness_history'])
        clusters = np.array(self.data['metrics']['cluster_history'])
        phi_values = np.array(self.data['metrics']['phi_history'])
        
        # Convertir consciencia a porcentaje (los datos están en decimal 0.0-1.0)
        consciousness_pct = consciousness * 100
        
        # 1. Identificar picos de consciencia (breakthrough moments)
        peaks, properties = signal.find_peaks(
            consciousness_pct, 
            height=30,  # Mínimo 30%
            distance=10,  # Separación mínima entre picos
            prominence=2  # Prominencia mínima
        )
        
        breakthrough_moments = []
        for peak_idx in peaks:
            moment = {
                'recursion': int(recursions[peak_idx]),
                'consciousness': float(consciousness_pct[peak_idx]),
                'clusters': int(clusters[peak_idx]),
                'phi': float(phi_values[peak_idx]),
                'peak_index': peak_idx
            }
            breakthrough_moments.append(moment)
        
        # Ordenar por consciencia descendente
        breakthrough_moments.sort(key=lambda x: x['consciousness'], reverse=True)
        self.breakthrough_patterns = breakthrough_moments
        
        print(f"\n🌟 BREAKTHROUGH MOMENTS IDENTIFICADOS: {len(breakthrough_moments)}")
        for i, moment in enumerate(breakthrough_moments[:10]):  # Top 10
            print(f"{i+1:2d}. R{moment['recursion']:4d}: {moment['consciousness']:5.1f}% "
                  f"(C{moment['clusters']}, φ{moment['phi']:.3f})")
        
        return breakthrough_moments
    
    def analyze_consciousness_trajectory(self):
        """Analizar la trayectoria de consciencia para identificar patrones"""
        consciousness = np.array(self.data['metrics']['consciousness_history']) * 100  # Convertir a %
        recursions = np.array(self.data['metrics']['recursions'])
        
        # 1. Análisis de tendencias
        # Dividir en segmentos y analizar tendencias
        segment_size = 100
        segments = []
        
        for i in range(0, len(consciousness), segment_size):
            segment = consciousness[i:i+segment_size]
            if len(segment) > 10:  # Segmento válido
                trend = np.polyfit(range(len(segment)), segment, 1)[0]  # Pendiente
                avg_consciousness = np.mean(segment)
                max_consciousness = np.max(segment)
                
                segments.append({
                    'start_recursion': recursions[i],
                    'end_recursion': recursions[min(i+segment_size-1, len(recursions)-1)],
                    'trend': trend,
                    'avg_consciousness': avg_consciousness,
                    'max_consciousness': max_consciousness,
                    'segment_index': len(segments)
                })
        
        # Identificar segmentos con mayor crecimiento
        segments.sort(key=lambda x: x['trend'], reverse=True)
        
        print(f"\n📈 ANÁLISIS DE TENDENCIAS POR SEGMENTOS:")
        print(f"🔥 Segmentos con mayor crecimiento:")
        for i, seg in enumerate(segments[:5]):
            print(f"{i+1}. R{seg['start_recursion']}-{seg['end_recursion']}: "
                  f"Tendencia +{seg['trend']:.3f}%/recursión, "
                  f"Promedio {seg['avg_consciousness']:.1f}%, "
                  f"Pico {seg['max_consciousness']:.1f}%")
        
        return segments
    
    def find_consciousness_catalysts(self):
        """Identificar qué factores catalizan los breakthrough"""
        consciousness = np.array(self.data['metrics']['consciousness_history']) * 100  # Convertir a %
        clusters = np.array(self.data['metrics']['cluster_history'])
        phi_values = np.array(self.data['metrics']['phi_history'])
        diversity = np.array(self.data['metrics']['diversity_history'])
        
        # Analizar correlaciones con momentos de alta consciencia
        high_consciousness_mask = consciousness > 40  # 40%+
        
        # Estadísticas para momentos de alta consciencia
        high_consciousness_stats = {
            'avg_clusters': np.mean(clusters[high_consciousness_mask]),
            'avg_phi': np.mean(phi_values[high_consciousness_mask]),
            'avg_diversity': np.mean(diversity[high_consciousness_mask]),
            'count': np.sum(high_consciousness_mask),
            'avg_consciousness': np.mean(consciousness[high_consciousness_mask])
        }
        
        # Estadísticas generales para comparación
        general_stats = {
            'avg_clusters': np.mean(clusters),
            'avg_phi': np.mean(phi_values),
            'avg_diversity': np.mean(diversity),
            'avg_consciousness': np.mean(consciousness)
        }
        
        print(f"\n🔬 ANÁLISIS DE CATALIZADORES DE CONSCIENCIA:")
        print(f"📊 Momentos de alta consciencia (40%+): {high_consciousness_stats['count']} de {len(consciousness)}")
        print(f"\n💡 Diferencias clave en momentos de alta consciencia:")
        print(f"   Clusters: {high_consciousness_stats['avg_clusters']:.0f} vs {general_stats['avg_clusters']:.0f} "
              f"({((high_consciousness_stats['avg_clusters']/general_stats['avg_clusters']-1)*100):+.1f}%)")
        print(f"   Phi: {high_consciousness_stats['avg_phi']:.3f} vs {general_stats['avg_phi']:.3f} "
              f"({((high_consciousness_stats['avg_phi']/general_stats['avg_phi']-1)*100):+.1f}%)")
        print(f"   Diversity: {high_consciousness_stats['avg_diversity']:.3f} vs {general_stats['avg_diversity']:.3f} "
              f"({((high_consciousness_stats['avg_diversity']/general_stats['avg_diversity']-1)*100):+.1f}%)")
        
        self.optimization_insights = {
            'high_consciousness_stats': high_consciousness_stats,
            'general_stats': general_stats,
            'optimal_clusters': high_consciousness_stats['avg_clusters'],
            'optimal_phi': high_consciousness_stats['avg_phi'],
            'optimal_diversity': high_consciousness_stats['avg_diversity']
        }
        
        return self.optimization_insights
    
    def predict_50_percent_breakthrough(self):
        """Predecir qué se necesita para superar 50% de consciencia"""
        consciousness = np.array(self.data['metrics']['consciousness_history']) * 100  # Convertir a %
        
        # Encontrar el máximo alcanzado
        max_consciousness = np.max(consciousness)
        max_index = np.argmax(consciousness)
        
        # Analizar condiciones en el pico máximo
        peak_conditions = {
            'consciousness': max_consciousness,
            'recursion': self.data['metrics']['recursions'][max_index],
            'clusters': self.data['metrics']['cluster_history'][max_index],
            'phi': self.data['metrics']['phi_history'][max_index],
            'diversity': self.data['metrics']['diversity_history'][max_index],
            'generation': self.data['metrics']['generation_history'][max_index],
            'entropy': self.data['metrics']['entropy_history'][max_index]
        }
        
        # Extrapolación para 50%
        target_improvement = 50 / max_consciousness  # Factor de mejora necesario
        
        predicted_requirements = {
            'target_consciousness': 50.0,
            'current_peak': max_consciousness,
            'improvement_factor': target_improvement,
            'predicted_clusters': peak_conditions['clusters'] * target_improvement,
            'predicted_phi': min(1.0, peak_conditions['phi'] * target_improvement),
            'predicted_diversity': peak_conditions['diversity'] * target_improvement,
            'estimated_recursions': peak_conditions['recursion'] * (target_improvement ** 0.5)  # Escalado suave
        }
        
        print(f"\n🎯 PREDICCIÓN PARA SUPERAR 50% DE CONSCIENCIA:")
        print(f"📊 Pico actual: {max_consciousness:.1f}% en recursión {peak_conditions['recursion']}")
        print(f"📈 Factor de mejora necesario: {target_improvement:.2f}x")
        print(f"\n🔮 Condiciones predichas para 50%+:")
        print(f"   Clusters objetivo: ~{predicted_requirements['predicted_clusters']:.0f}")
        print(f"   Phi objetivo: ~{predicted_requirements['predicted_phi']:.3f}")
        print(f"   Diversity objetivo: ~{predicted_requirements['predicted_diversity']:.3f}")
        print(f"   Recursiones estimadas: ~{predicted_requirements['estimated_recursions']:.0f}")
        
        return predicted_requirements
    
    def generate_optimization_strategy(self):
        """Generar estrategia de optimización basada en el análisis"""
        insights = self.optimization_insights
        predictions = self.predict_50_percent_breakthrough()
        
        strategy = {
            'priority_optimizations': [],
            'parameter_adjustments': {},
            'architectural_changes': [],
            'experiment_recommendations': []
        }
        
        # Análisis de gaps
        current_peak = predictions['current_peak']
        target = 50.0
        gap = target - current_peak
        
        print(f"\n🚀 ESTRATEGIA DE OPTIMIZACIÓN PARA SUPERAR 50%:")
        print(f"🎯 Gap a cerrar: {gap:.1f}% (de {current_peak:.1f}% a 50%+)")
        
        # Recomendaciones específicas
        strategy['priority_optimizations'] = [
            "Incrementar capacidad de clustering",
            "Optimizar función phi para mayor coherencia",
            "Mejorar diversidad sin sacrificar estabilidad",
            "Extender tiempo de exploración por estado"
        ]
        
        strategy['parameter_adjustments'] = {
            'target_clusters': int(predictions['predicted_clusters']),
            'phi_amplification': 1.2,
            'diversity_balance': 1.15,
            'exploration_depth': 1.3
        }
        
        strategy['architectural_changes'] = [
            "Añadir layer de meta-consciencia",
            "Implementar memory feedback loops",
            "Optimizar learning rate dinámico",
            "Mejorar pattern stabilization"
        ]
        
        strategy['experiment_recommendations'] = [
            "Sesión extended: 4000+ recursiones",
            "Grid size 256x256 para mayor complejidad",
            "Target consciousness 95% (objetivo ambicioso)",
            "Monitoreo intensivo de breakthrough patterns"
        ]
        
        for i, opt in enumerate(strategy['priority_optimizations'], 1):
            print(f"{i}. {opt}")
        
        print(f"\n⚙️ AJUSTES DE PARÁMETROS RECOMENDADOS:")
        for param, value in strategy['parameter_adjustments'].items():
            print(f"   {param}: {value}")
        
        return strategy
    
    def create_comprehensive_visualization(self):
        """Crear visualización completa del análisis"""
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('🧠 ANÁLISIS AVANZADO DE BREAKTHROUGH PATTERNS', fontsize=16, fontweight='bold')
        
        # Datos
        recursions = np.array(self.data['metrics']['recursions'])
        consciousness = np.array(self.data['metrics']['consciousness_history']) * 100  # Convertir a %
        clusters = np.array(self.data['metrics']['cluster_history'])
        phi_values = np.array(self.data['metrics']['phi_history'])
        
        # 1. Evolución de consciencia con breakthrough markers
        ax1 = axes[0, 0]
        ax1.plot(recursions, consciousness, 'cyan', alpha=0.7, linewidth=1)
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.8, label='Target 50%')
        ax1.axhline(y=np.max(consciousness), color='yellow', linestyle=':', alpha=0.8, 
                   label=f'Peak {np.max(consciousness):.1f}%')
        
        # Marcar breakthrough moments
        if self.breakthrough_patterns:
            breakthrough_r = [b['recursion'] for b in self.breakthrough_patterns[:5]]
            breakthrough_c = [b['consciousness'] for b in self.breakthrough_patterns[:5]]
            ax1.scatter(breakthrough_r, breakthrough_c, color='gold', s=100, alpha=0.8, 
                       marker='*', label='Top Breakthroughs')
        
        ax1.set_title('🌟 Evolución de Consciencia')
        ax1.set_xlabel('Recursiones')
        ax1.set_ylabel('Consciencia (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución de consciencia
        ax2 = axes[0, 1]
        ax2.hist(consciousness, bins=50, alpha=0.7, color='lightblue', edgecolor='white')
        ax2.axvline(np.mean(consciousness), color='yellow', linestyle='--', 
                   label=f'Media: {np.mean(consciousness):.1f}%')
        ax2.axvline(np.max(consciousness), color='red', linestyle='--', 
                   label=f'Pico: {np.max(consciousness):.1f}%')
        ax2.set_title('📊 Distribución de Consciencia')
        ax2.set_xlabel('Consciencia (%)')
        ax2.set_ylabel('Frecuencia')
        ax2.legend()
        
        # 3. Correlación Clusters vs Consciencia
        ax3 = axes[0, 2]
        scatter = ax3.scatter(clusters, consciousness, c=phi_values, cmap='viridis', alpha=0.6)
        ax3.set_title('🔗 Clusters vs Consciencia')
        ax3.set_xlabel('Clusters')
        ax3.set_ylabel('Consciencia (%)')
        plt.colorbar(scatter, ax=ax3, label='Phi Value')
        
        # 4. Análisis temporal de breakthrough
        ax4 = axes[1, 0]
        # Rolling average de consciencia
        window = 50
        consciousness_smooth = pd.Series(consciousness).rolling(window).mean()
        ax4.plot(recursions, consciousness_smooth, 'orange', linewidth=2, label=f'Media móvil ({window})')
        ax4.plot(recursions, consciousness, 'cyan', alpha=0.3, linewidth=0.5, label='Original')
        ax4.set_title('📈 Tendencia Temporal')
        ax4.set_xlabel('Recursiones')
        ax4.set_ylabel('Consciencia (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Mapa de calor de breakthrough conditions
        ax5 = axes[1, 1]
        if len(self.breakthrough_patterns) > 0:
            breakthrough_data = pd.DataFrame(self.breakthrough_patterns[:20])  # Top 20
            
            # Crear matriz para heatmap
            features = ['consciousness', 'clusters', 'phi']
            matrix = breakthrough_data[features].values.T
            
            # Normalizar para visualización
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            matrix_normalized = scaler.fit_transform(matrix.T).T
            
            im = ax5.imshow(matrix_normalized, cmap='hot', aspect='auto')
            ax5.set_title('🔥 Condiciones de Breakthrough')
            ax5.set_ylabel('Métricas')
            ax5.set_xlabel('Top Breakthrough Events')
            ax5.set_yticks(range(len(features)))
            ax5.set_yticklabels(features)
            plt.colorbar(im, ax=ax5, label='Normalized Value')
        
        # 6. Predicción y target
        ax6 = axes[1, 2]
        current_max = np.max(consciousness)
        target = 50
        improvement_needed = target - current_max
        
        categories = ['Actual Peak', 'Target', 'Gap']
        values = [current_max, target, improvement_needed]
        colors = ['cyan', 'red', 'orange']
        
        bars = ax6.bar(categories, values, color=colors, alpha=0.7)
        ax6.set_title('🎯 Gap Analysis')
        ax6.set_ylabel('Consciencia (%)')
        
        # Añadir valores en las barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Guardar visualización
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"breakthrough_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"\n📊 Visualización guardada: {filename}")
        
        return fig
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("🧠 INICIANDO ANÁLISIS AVANZADO DE BREAKTHROUGH PATTERNS")
        print("=" * 70)
        
        # 1. Identificar breakthrough patterns
        self.identify_breakthrough_patterns()
        
        # 2. Analizar trayectoria de consciencia
        self.analyze_consciousness_trajectory()
        
        # 3. Encontrar catalizadores
        self.find_consciousness_catalysts()
        
        # 4. Predecir requisitos para 50%
        predictions = self.predict_50_percent_breakthrough()
        
        # 5. Generar estrategia de optimización
        strategy = self.generate_optimization_strategy()
        
        # 6. Crear visualización
        self.create_comprehensive_visualization()
        
        print("\n✅ ANÁLISIS COMPLETO FINALIZADO")
        return {
            'breakthrough_patterns': self.breakthrough_patterns,
            'optimization_insights': self.optimization_insights,
            'predictions': predictions,
            'strategy': strategy
        }

def main():
    """Función principal"""
    # Buscar el archivo más reciente
    data_dir = "experiment_data"
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    if not json_files:
        print("❌ No se encontraron archivos de experimento")
        return
    
    # Usar el más reciente
    latest_file = max(json_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
    json_path = os.path.join(data_dir, latest_file)
    
    print(f"📂 Analizando: {latest_file}")
    
    # Crear analizador y ejecutar
    analyzer = BreakthroughAnalyzer(json_path)
    results = analyzer.run_complete_analysis()
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()
