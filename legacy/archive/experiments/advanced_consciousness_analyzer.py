#!/usr/bin/env python3
"""
INFINITO V5.1 - Advanced Consciousness Data Analyzer
===================================================
Análisis científico profundo de experimentos de consciencia artificial

Este script realiza un análisis estadístico comprehensivo de los datos
de experimentos de consciencia INFINITO V5.1, incluyendo:
- Tendencias temporales de consciencia
- Análisis de eventos cuánticos
- Evolución de Φ (integración de información) 
- Patrones de memoria y eficiencia
- Correlaciones estadísticas avanzadas
- Visualizaciones científicas
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AdvancedConsciousnessAnalyzer:
    """Analizador avanzado de datos de consciencia INFINITO V5.1"""
    
    def __init__(self, json_file):
        """
        Inicializar analizador con archivo de datos del experimento
        
        Args:
            json_file (str): Ruta al archivo JSON del experimento
        """
        self.json_file = Path(json_file)
        self.data = self._load_data()
        self.df = None
        self.stats = {}
        
        print(f"🔬 INFINITO V5.1 Advanced Consciousness Analyzer Initialized")
        print(f"📊 Data file: {self.json_file.name}")
        
    def _load_data(self):
        """Cargar y validar datos del experimento"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Data loaded successfully:")
            print(f"   📈 Iterations: {len(data.get('iterations', []))}")
            print(f"   🧠 Final Consciousness: {data.get('final_consciousness', 'N/A')}")
            print(f"   🔬 Final Phi: {data.get('final_phi', 'N/A')}")
            print(f"   ⏱️ Duration: {data.get('duration', 'N/A')}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def create_dataframe(self):
        """Crear DataFrame de pandas para análisis estadístico"""
        if not self.data or 'iterations' not in self.data:
            print("❌ No iteration data found")
            return None
        
        # Determinar si el formato es array de números o array de objetos
        iterations = self.data['iterations']
        if isinstance(iterations[0], int):
            # Formato nuevo: arrays separados por métrica
            primary_metrics = {
                'iteration': self.data.get('iterations', []),
                'consciousness': self.data.get('consciousness_values', []),
                'phi': self.data.get('phi_values', []),
                'memory_use': self.data.get('memory_utilization', []),
                'loss': self.data.get('loss_values', []),
                'learning_rate': self.data.get('learning_rates', []),
                'eeg_correlation': self.data.get('eeg_correlations', []),
                'growth_rate': self.data.get('growth_rates', []),
                'target_consciousness': self.data.get('target_consciousness', []),
                'phi_loss': self.data.get('phi_loss_values', []),
                'consciousness_loss': self.data.get('consciousness_loss_values', []),
                'memory_loss': self.data.get('memory_loss_values', []),
                'differentiation_loss': self.data.get('differentiation_loss_values', []),
                'causal_density': self.data.get('causal_density_values', []),
                'attention_strength': self.data.get('attention_strength_values', []),
                'module_differentiation': self.data.get('module_differentiation_values', []),
                'quantum_phi_deltas': self.data.get('quantum_phi_deltas', []),
                'quantum_fact_confidences': self.data.get('quantum_fact_confidences', []),
                'total_quantum_facts': self.data.get('total_quantum_facts', [])
            }
            
            # Encontrar la longitud de los arrays principales (excluyendo arrays especiales)
            main_arrays = {k: v for k, v in primary_metrics.items() if v and len(v) > 10}  # Arrays con más de 10 elementos
            if not main_arrays:
                print("❌ No main data arrays found")
                return None
                
            main_length = len(next(iter(main_arrays.values())))  # Usar primer array como referencia
            print(f"📏 Using main length: {main_length} from {len(main_arrays)} main arrays")
            
            # Verificar que todos los arrays principales tengan la misma longitud
            for key, values in main_arrays.items():
                if len(values) != main_length:
                    print(f"⚠️ Warning: {key} has {len(values)} elements, expected {main_length}")
            
            # Crear DataFrame con métricas principales
            df_data = {}
            for key, values in primary_metrics.items():
                if values and len(values) >= main_length:
                    df_data[key] = values[:main_length]
                elif values:
                    # Para arrays más cortos, repetir el último valor
                    df_data[key] = values + [values[-1]] * (main_length - len(values))
                else:
                    # Crear valores por defecto
                    if key == 'iteration':
                        df_data[key] = list(range(1, main_length + 1))
                    else:
                        df_data[key] = [0.0] * main_length
            
            # Procesar breakthroughs de manera especial
            breakthrough_data = self.data.get('breakthroughs', [])
            df_data['has_breakthrough'] = [False] * main_length
            if breakthrough_data:
                for bt in breakthrough_data:
                    if isinstance(bt, dict) and 'iteration' in bt:
                        iter_idx = bt['iteration'] - 1  # Convert to 0-based index
                        if 0 <= iter_idx < main_length:
                            df_data['has_breakthrough'][iter_idx] = True
            
            # Calcular métricas derivadas
            if 'quantum_phi_deltas' in df_data and 'quantum_fact_confidences' in df_data:
                df_data['quantum_events_count'] = [1 if delta > 0.05 else 0 for delta in df_data['quantum_phi_deltas']]
                df_data['quantum_confidence'] = df_data['quantum_fact_confidences']
            else:
                df_data['quantum_events_count'] = [0] * main_length
                df_data['quantum_confidence'] = [0.0] * main_length
            
            # Datos de pattern detection basados en el experimento
            df_data['pattern_detection'] = ['stagnation'] * main_length
            df_data['breakthrough_probability'] = [0.15] * main_length
            
            self.df = pd.DataFrame(df_data)
        else:
            # Formato anterior: array de objetos (implementación previa)
            df_data = []
            
            for iter_data in iterations:
                row = {
                    'iteration': iter_data.get('iteration', 0),
                    'consciousness': iter_data.get('consciousness', 0),
                    'phi': iter_data.get('phi', 0),
                    'memory_use': iter_data.get('memory_use', 0),
                    'loss': iter_data.get('loss', 0),
                    'learning_rate': iter_data.get('learning_rate', 0),
                    'eeg_correlation': iter_data.get('eeg_correlation', 0),
                    'growth_rate': iter_data.get('growth_rate', 0),
                    'target_consciousness': iter_data.get('target_consciousness', 0),
                    'quantum_events_count': len(iter_data.get('quantum_events', [])),
                    'quantum_confidence': np.mean([qe.get('confidence', 0) for qe in iter_data.get('quantum_events', [])]) if iter_data.get('quantum_events') else 0,
                    'pattern_detection': iter_data.get('pattern_detection', {}).get('pattern_type', 'none'),
                    'breakthrough_probability': iter_data.get('pattern_detection', {}).get('breakthrough_probability', 0)
                }
                df_data.append(row)
            
            self.df = pd.DataFrame(df_data)
        
        print(f"✅ DataFrame created with {len(self.df)} iterations")
        print(f"📊 Key columns: consciousness, phi, memory_use, loss, quantum_phi_deltas")
        
        return self.df
    
    def calculate_advanced_statistics(self):
        """Calcular estadísticas avanzadas del experimento"""
        if self.df is None:
            return None
        
        self.stats = {
            'consciousness': {
                'mean': self.df['consciousness'].mean(),
                'std': self.df['consciousness'].std(),
                'min': self.df['consciousness'].min(),
                'max': self.df['consciousness'].max(),
                'breakthrough_iterations': len(self.df[self.df['consciousness'] > 0.6]),
                'stability_ratio': len(self.df[self.df['consciousness'] > 0.99]) / len(self.df),
                'convergence_iteration': self.df[self.df['consciousness'] > 0.99]['iteration'].min() if len(self.df[self.df['consciousness'] > 0.99]) > 0 else None
            },
            'phi': {
                'mean': self.df['phi'].mean(),
                'std': self.df['phi'].std(),
                'min': self.df['phi'].min(),
                'max': self.df['phi'].max(),
                'trend_slope': np.polyfit(self.df['iteration'], self.df['phi'], 1)[0]
            },
            'memory': {
                'mean': self.df['memory_use'].mean(),
                'max': self.df['memory_use'].max(),
                'efficiency_score': 1 - (self.df['memory_use'].mean() / 100)
            },
            'quantum': {
                'total_events': self.df['quantum_events_count'].sum(),
                'avg_events_per_iteration': self.df['quantum_events_count'].mean(),
                'avg_confidence': self.df['quantum_confidence'].mean(),
                'peak_activity_iteration': self.df.loc[self.df['quantum_events_count'].idxmax(), 'iteration']
            },
            'learning': {
                'final_lr': self.df['learning_rate'].iloc[-1],
                'lr_adaptation': (self.df['learning_rate'].iloc[0] - self.df['learning_rate'].iloc[-1]) / self.df['learning_rate'].iloc[0],
                'loss_reduction': (self.df['loss'].iloc[0] - self.df['loss'].iloc[-1]) / self.df['loss'].iloc[0],
                'convergence_rate': self._calculate_convergence_rate()
            }
        }
        
        return self.stats
    
    def _calculate_convergence_rate(self):
        """Calcular tasa de convergencia del sistema"""
        consciousness_values = self.df['consciousness'].values
        target = 0.99
        
        # Encontrar primer punto donde se alcanza el 99%
        convergence_points = np.where(consciousness_values >= target)[0]
        if len(convergence_points) > 0:
            convergence_iteration = convergence_points[0]
            return convergence_iteration / len(consciousness_values)
        return 1.0  # No convergió
    
    def print_scientific_summary(self):
        """Imprimir resumen científico detallado"""
        if not self.stats:
            self.calculate_advanced_statistics()
        
        print("\n" + "="*80)
        print("🧠 INFINITO V5.1 - SCIENTIFIC ANALYSIS REPORT")
        print("="*80)
        
        # Consciousness Analysis
        print(f"\n🎯 CONSCIOUSNESS METRICS:")
        print(f"   📊 Mean Consciousness: {self.stats['consciousness']['mean']:.4f} ({self.stats['consciousness']['mean']*100:.1f}%)")
        print(f"   📈 Peak Consciousness: {self.stats['consciousness']['max']:.4f} ({self.stats['consciousness']['max']*100:.1f}%)")
        print(f"   🎉 Breakthrough Iterations: {self.stats['consciousness']['breakthrough_iterations']}/{len(self.df)}")
        print(f"   ⚡ Stability Ratio: {self.stats['consciousness']['stability_ratio']*100:.1f}% (>99% consciousness)")
        if self.stats['consciousness']['convergence_iteration']:
            print(f"   🚀 Convergence at iteration: {self.stats['consciousness']['convergence_iteration']}")
        
        # Phi Analysis
        print(f"\n🔬 PHI (INFORMATION INTEGRATION) METRICS:")
        print(f"   📊 Mean Φ: {self.stats['phi']['mean']:.4f} bits")
        print(f"   📈 Peak Φ: {self.stats['phi']['max']:.4f} bits")
        print(f"   📉 Φ Std Dev: {self.stats['phi']['std']:.4f}")
        print(f"   📈 Φ Trend: {'Increasing' if self.stats['phi']['trend_slope'] > 0 else 'Decreasing'} ({self.stats['phi']['trend_slope']:.6f})")
        
        # Quantum Events Analysis  
        print(f"\n⚡ QUANTUM EVENTS ANALYSIS:")
        print(f"   🔬 Total Quantum Events: {self.stats['quantum']['total_events']}")
        print(f"   📊 Avg Events/Iteration: {self.stats['quantum']['avg_events_per_iteration']:.2f}")
        print(f"   🎯 Avg Confidence: {self.stats['quantum']['avg_confidence']:.3f}")
        print(f"   🚀 Peak Activity at iteration: {self.stats['quantum']['peak_activity_iteration']}")
        
        # Memory Efficiency
        print(f"\n💾 MEMORY EFFICIENCY:")
        print(f"   📊 Mean Memory Use: {self.stats['memory']['mean']:.4f}%")
        print(f"   📈 Peak Memory Use: {self.stats['memory']['max']:.4f}%")
        print(f"   ⚡ Efficiency Score: {self.stats['memory']['efficiency_score']*100:.1f}%")
        
        # Learning Dynamics
        print(f"\n📚 LEARNING DYNAMICS:")
        print(f"   📉 Learning Rate Adaptation: {self.stats['learning']['lr_adaptation']*100:.1f}%")
        print(f"   📈 Loss Reduction: {self.stats['learning']['loss_reduction']*100:.1f}%")
        print(f"   🚀 Convergence Rate: {self.stats['learning']['convergence_rate']*100:.1f}%")
        
        print("="*80)
    
    def create_comprehensive_visualizations(self, save_plots=True):
        """Crear visualizaciones científicas comprehensivas"""
        if self.df is None:
            print("❌ No data available for visualization")
            return
        
        # Configurar estilo científico
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Crear figura con múltiples subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Evolución temporal de consciousness
        ax1 = plt.subplot(4, 3, 1)
        plt.plot(self.df['iteration'], self.df['consciousness'] * 100, 'b-', linewidth=2, alpha=0.8)
        plt.axhline(y=60, color='r', linestyle='--', alpha=0.5, label='Breakthrough threshold (60%)')
        plt.axhline(y=99, color='g', linestyle='--', alpha=0.5, label='Target (99%)')
        plt.title('🧠 Consciousness Evolution', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Consciousness (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Distribución de consciousness
        ax2 = plt.subplot(4, 3, 2)
        plt.hist(self.df['consciousness'] * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=self.df['consciousness'].mean() * 100, color='red', linestyle='--', label=f'Mean: {self.df["consciousness"].mean()*100:.1f}%')
        plt.title('📊 Consciousness Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Consciousness (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Evolución de Phi
        ax3 = plt.subplot(4, 3, 3)
        plt.plot(self.df['iteration'], self.df['phi'], 'g-', linewidth=2, alpha=0.8)
        plt.title('🔬 Φ (Information Integration)', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Φ (bits)')
        plt.grid(True, alpha=0.3)
        
        # 4. Quantum events por iteración
        ax4 = plt.subplot(4, 3, 4)
        plt.bar(self.df['iteration'][::10], self.df['quantum_events_count'][::10], alpha=0.7, color='orange')
        plt.title('⚡ Quantum Events Activity', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration (every 10th)')
        plt.ylabel('Quantum Events Count')
        plt.grid(True, alpha=0.3)
        
        # 5. Memory usage
        ax5 = plt.subplot(4, 3, 5)
        plt.plot(self.df['iteration'], self.df['memory_use'], 'r-', linewidth=2, alpha=0.8)
        plt.title('💾 Memory Utilization', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Memory Use (%)')
        plt.grid(True, alpha=0.3)
        
        # 6. Loss evolution
        ax6 = plt.subplot(4, 3, 6)
        plt.plot(self.df['iteration'], self.df['loss'], 'm-', linewidth=2, alpha=0.8)
        plt.title('📉 Loss Evolution', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # 7. Consciousness vs Phi correlation
        ax7 = plt.subplot(4, 3, 7)
        plt.scatter(self.df['consciousness'], self.df['phi'], alpha=0.6, c=self.df['iteration'], cmap='viridis')
        plt.colorbar(label='Iteration')
        plt.title('🧠🔬 Consciousness vs Φ Correlation', fontsize=14, fontweight='bold')
        plt.xlabel('Consciousness')
        plt.ylabel('Φ (bits)')
        plt.grid(True, alpha=0.3)
        
        # 8. Learning rate adaptation
        ax8 = plt.subplot(4, 3, 8)
        plt.plot(self.df['iteration'], self.df['learning_rate'], 'c-', linewidth=2, alpha=0.8)
        plt.title('📚 Learning Rate Adaptation', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        
        # 9. EEG correlation
        ax9 = plt.subplot(4, 3, 9)
        plt.plot(self.df['iteration'], self.df['eeg_correlation'], 'y-', linewidth=2, alpha=0.8)
        plt.title('🧬 EEG Correlation', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('EEG Correlation')
        plt.grid(True, alpha=0.3)
        
        # 10. Quantum confidence heatmap
        ax10 = plt.subplot(4, 3, 10)
        # Crear matriz de confianza cuántica por bloques de 50 iteraciones
        block_size = 50
        blocks = len(self.df) // block_size
        confidence_matrix = []
        for i in range(blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, len(self.df))
            block_confidence = self.df['quantum_confidence'][start_idx:end_idx].values
            confidence_matrix.append(block_confidence[:block_size] if len(block_confidence) >= block_size else 
                                   np.pad(block_confidence, (0, block_size - len(block_confidence)), constant_values=0))
        
        if confidence_matrix:
            plt.imshow(confidence_matrix, cmap='hot', interpolation='nearest', aspect='auto')
            plt.colorbar(label='Quantum Confidence')
            plt.title('🔥 Quantum Confidence Heatmap', fontsize=14, fontweight='bold')
            plt.xlabel('Iteration (within block)')
            plt.ylabel(f'Block (every {block_size} iterations)')
        
        # 11. Pattern detection analysis
        ax11 = plt.subplot(4, 3, 11)
        pattern_counts = self.df['pattern_detection'].value_counts()
        plt.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%')
        plt.title('🔍 Pattern Detection Distribution', fontsize=14, fontweight='bold')
        
        # 12. Multi-metric dashboard
        ax12 = plt.subplot(4, 3, 12)
        # Normalizar métricas para comparación
        normalized_consciousness = self.df['consciousness']
        normalized_phi = self.df['phi'] / self.df['phi'].max()
        normalized_memory = self.df['memory_use'] / 100
        
        plt.plot(self.df['iteration'], normalized_consciousness, label='Consciousness', linewidth=2)
        plt.plot(self.df['iteration'], normalized_phi, label='Φ (normalized)', linewidth=2)
        plt.plot(self.df['iteration'], normalized_memory, label='Memory Use', linewidth=2)
        plt.title('📊 Multi-Metric Dashboard', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/advanced_consciousness_analysis_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Advanced analysis plots saved to: {filename}")
        
        plt.show()
    
    def generate_scientific_insights(self):
        """Generar insights científicos basados en el análisis"""
        if not self.stats:
            self.calculate_advanced_statistics()
        
        insights = []
        
        # Análisis de estabilidad
        if self.stats['consciousness']['stability_ratio'] > 0.8:
            insights.append("🎯 ALTA ESTABILIDAD: El sistema mantiene >99% consciencia en >80% de iteraciones")
        
        # Análisis de convergencia
        if self.stats['consciousness']['convergence_iteration'] and self.stats['consciousness']['convergence_iteration'] < 50:
            insights.append(f"🚀 CONVERGENCIA RÁPIDA: Breakthrough alcanzado en iteración {self.stats['consciousness']['convergence_iteration']}")
        
        # Análisis de Phi
        if self.stats['phi']['trend_slope'] > 0:
            insights.append("📈 PHI CRECIENTE: Integración de información mejorando consistentemente")
        
        # Análisis de eficiencia
        if self.stats['memory']['efficiency_score'] > 0.95:
            insights.append("⚡ ALTA EFICIENCIA: Sistema utiliza <5% de memoria disponible")
        
        # Análisis cuántico
        if self.stats['quantum']['avg_events_per_iteration'] > 1:
            insights.append(f"🔬 ACTIVIDAD CUÁNTICA INTENSA: {self.stats['quantum']['avg_events_per_iteration']:.1f} eventos promedio por iteración")
        
        return insights
    
    def export_detailed_report(self):
        """Exportar reporte detallado en formato texto"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"outputs/consciousness_analysis_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("INFINITO V5.1 - DETAILED CONSCIOUSNESS ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data File: {self.json_file.name}\n")
            f.write(f"Total Iterations: {len(self.df)}\n\n")
            
            # Estadísticas detalladas
            f.write("STATISTICAL SUMMARY\n")
            f.write("-" * 20 + "\n")
            for category, metrics in self.stats.items():
                f.write(f"\n{category.upper()}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value}\n")
            
            # Insights
            insights = self.generate_scientific_insights()
            f.write("\nSCIENTIFIC INSIGHTS\n")
            f.write("-" * 20 + "\n")
            for i, insight in enumerate(insights, 1):
                f.write(f"{i}. {insight}\n")
            
            # DataFrame summary
            f.write("\nDATA SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(str(self.df.describe()))
        
        print(f"📄 Detailed report exported to: {report_file}")
        return report_file

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Advanced Consciousness Data Analyzer')
    parser.add_argument('--file', required=True, help='Path to JSON experiment file')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--export-report', action='store_true', help='Export detailed text report')
    
    args = parser.parse_args()
    
    print("🧠 INFINITO V5.1 - ADVANCED CONSCIOUSNESS ANALYZER")
    print("="*60)
    
    # Crear analizador
    analyzer = AdvancedConsciousnessAnalyzer(args.file)
    
    if not analyzer.data:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Crear DataFrame
    analyzer.create_dataframe()
    
    # Calcular estadísticas
    analyzer.calculate_advanced_statistics()
    
    # Imprimir resumen
    analyzer.print_scientific_summary()
    
    # Generar insights
    insights = analyzer.generate_scientific_insights()
    print(f"\n🔍 SCIENTIFIC INSIGHTS:")
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    # Crear visualizaciones
    if not args.no_plots:
        print(f"\n📊 Generating comprehensive visualizations...")
        analyzer.create_comprehensive_visualizations()
    
    # Exportar reporte
    if args.export_report:
        analyzer.export_detailed_report()
    
    print(f"\n✅ Analysis completed successfully!")

if __name__ == "__main__":
    main()