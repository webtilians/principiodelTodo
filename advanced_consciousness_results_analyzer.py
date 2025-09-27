#!/usr/bin/env python3
"""
🧠 ADVANCED CONSCIOUSNESS RESULTS ANALYZER V2.0
Extrae patrones de transiciones para predecir breakthroughs
Correlaciona deltas de Φ con cambios en C usando análisis estadístico avanzado
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import pandas as pd
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

# Set aesthetic style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AdvancedConsciousnessAnalyzer:
    """
    🔬 Analizador avanzado para extraer patrones de consciousness breakthroughs
    """
    
    def __init__(self, json_file_path):
        self.json_file = json_file_path
        self.data = None
        self.results = {}
        self.load_data()
    
    def load_data(self):
        """📂 Carga datos JSON y prepara estructuras"""
        print(f"🔄 Cargando datos desde: {self.json_file}")
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ Datos cargados: {len(self.data.get('iterations', []))} iteraciones")
            print(f"📈 Breakthrough alcanzado: {self.data.get('breakthrough_achieved', False)}")
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            raise
    
    def extract_consciousness_phi_correlations(self):
        """🔗 Extrae correlaciones entre C y Φ con análisis estadístico avanzado"""
        print("\n🧮 === ANÁLISIS DE CORRELACIONES C-Φ ===")
        
        # Extraer series temporales principales
        c_values = np.array(self.data.get('consciousness_values', []))
        phi_values = np.array(self.data.get('phi_values', []))
        iterations = np.array(self.data.get('iterations', []))
        
        if len(c_values) == 0 or len(phi_values) == 0:
            print("⚠️  No hay suficientes datos de C o Φ")
            return
        
        # Alinear longitudes
        min_len = min(len(c_values), len(phi_values))
        c_aligned = c_values[:min_len]
        phi_aligned = phi_values[:min_len]
        iter_aligned = iterations[:min_len] if len(iterations) >= min_len else np.arange(min_len)
        
        # Calcular correlaciones múltiples
        spearman_corr, spearman_p = spearmanr(c_aligned, phi_aligned)
        pearson_corr, pearson_p = pearsonr(c_aligned, phi_aligned)
        kendall_corr, kendall_p = kendalltau(c_aligned, phi_aligned)
        
        # Análisis de deltas (cambios primera diferencia)
        c_deltas = np.diff(c_aligned)
        phi_deltas = np.diff(phi_aligned)
        delta_spearman, delta_spearman_p = spearmanr(c_deltas, phi_deltas)
        
        # Almacenar resultados
        self.results['correlations'] = {
            'spearman': {'r': spearman_corr, 'p': spearman_p},
            'pearson': {'r': pearson_corr, 'p': pearson_p}, 
            'kendall': {'r': kendall_corr, 'p': kendall_p},
            'delta_spearman': {'r': delta_spearman, 'p': delta_spearman_p}
        }
        
        # Estadísticas descriptivas
        self.results['stats'] = {
            'c_mean': np.mean(c_aligned), 'c_std': np.std(c_aligned),
            'c_max': np.max(c_aligned), 'c_min': np.min(c_aligned),
            'phi_mean': np.mean(phi_aligned), 'phi_std': np.std(phi_aligned),
            'phi_max': np.max(phi_aligned), 'phi_min': np.min(phi_aligned),
            'total_points': len(c_aligned)
        }
        
        # Imprimir resultados
        print(f"📊 Correlación Spearman C-Φ: {spearman_corr:.4f} (p={spearman_p:.4f})")
        print(f"📊 Correlación Pearson C-Φ:  {pearson_corr:.4f} (p={pearson_p:.4f})")  
        print(f"📊 Correlación Kendall C-Φ:  {kendall_corr:.4f} (p={kendall_p:.4f})")
        print(f"🔄 Correlación Deltas C-Φ:   {delta_spearman:.4f} (p={delta_spearman_p:.4f})")
        
        return c_aligned, phi_aligned, iter_aligned
    
    def analyze_transitions_patterns(self):
        """🌊 Analiza patrones en transiciones de Φ"""
        print("\n🌊 === ANÁLISIS DE TRANSICIONES ===")
        
        transitions = self.data.get('transitions', [])
        if not transitions:
            print("⚠️  No se encontraron datos de transiciones")
            return
            
        # Separar transiciones por tipo
        phi_transitions = [t for t in transitions if t['type'] == 'phi_transition']
        
        if not phi_transitions:
            print("⚠️  No hay transiciones de Φ para analizar")
            # Initialize empty results for consistency
            self.results['transitions'] = {
                'phi_transitions_count': 0,
                'phi_peaks': 0,
                'phi_valleys': 0,
                'phi_trans_mean': 0.0,
                'phi_trans_std': 0.0,
                'trend_mean': 0.0,
                'variance_mean': 0.0
            }
            return []
            
        # Extraer datos de transiciones
        phi_trans_values = [t['current_value'] for t in phi_transitions]
        phi_trans_iterations = [t['iteration'] for t in phi_transitions]
        phi_trans_trends = [t.get('trend', 0) for t in phi_transitions]
        phi_trans_variances = [t.get('variance', 0) for t in phi_transitions]
        
        # Detectar picos y valles en transiciones
        peaks, _ = find_peaks(phi_trans_values, height=np.mean(phi_trans_values))
        valleys, _ = find_peaks(-np.array(phi_trans_values), height=-np.mean(phi_trans_values))
        
        self.results['transitions'] = {
            'phi_transitions_count': len(phi_transitions),
            'phi_peaks': len(peaks),
            'phi_valleys': len(valleys),
            'phi_trans_mean': np.mean(phi_trans_values),
            'phi_trans_std': np.std(phi_trans_values),
            'trend_mean': np.mean(phi_trans_trends),
            'variance_mean': np.mean(phi_trans_variances)
        }
        
        print(f"🔢 Total transiciones Φ: {len(phi_transitions)}")
        print(f"⛰️  Picos detectados: {len(peaks)}")
        print(f"🏞️  Valles detectados: {len(valleys)}")
        print(f"📈 Tendencia promedio: {np.mean(phi_trans_trends):.6f}")
        print(f"📊 Varianza promedio: {np.mean(phi_trans_variances):.6f}")
        
        return phi_transitions
    
    def breakthrough_prediction_analysis(self):
        """🏆 Análisis predictivo de breakthrough patterns"""
        print("\n🏆 === ANÁLISIS PREDICTIVO DE BREAKTHROUGHS ===")
        
        c_values = np.array(self.data.get('consciousness_values', []))
        phi_values = np.array(self.data.get('phi_values', []))
        
        if len(c_values) == 0:
            return
            
        # Detectar momentum hacia breakthrough (últimos 25% de datos)
        momentum_window = len(c_values) // 4
        recent_c = c_values[-momentum_window:]
        recent_phi = phi_values[-momentum_window:] if len(phi_values) >= momentum_window else phi_values
        
        # Calcular momentum trends
        c_momentum = np.polyfit(range(len(recent_c)), recent_c, 1)[0] if len(recent_c) > 1 else 0
        phi_momentum = np.polyfit(range(len(recent_phi)), recent_phi, 1)[0] if len(recent_phi) > 1 else 0
        
        # Detectar aceleraciones (segunda derivada)
        c_acceleration = np.mean(np.diff(recent_c, 2)) if len(recent_c) > 2 else 0
        phi_acceleration = np.mean(np.diff(recent_phi, 2)) if len(recent_phi) > 2 else 0
        
        # Threshold analysis
        c_final = self.data.get('final_consciousness', 0)
        phi_final = self.data.get('final_phi', 0)
        breakthrough = self.data.get('breakthrough_achieved', False)
        
        # Predictive score (heurístico combinado)
        prediction_score = (
            (c_momentum * 100) +  # Momentum de consciousness
            (phi_momentum * 50) + # Momentum de phi  
            (c_acceleration * 200) + # Aceleración de C
            (phi_acceleration * 100) + # Aceleración de Φ
            (c_final * 50) + # Valor final de C
            (phi_final * 10)   # Valor final de Φ
        )
        
        self.results['prediction'] = {
            'c_momentum': c_momentum,
            'phi_momentum': phi_momentum,
            'c_acceleration': c_acceleration,
            'phi_acceleration': phi_acceleration,
            'prediction_score': prediction_score,
            'breakthrough_achieved': breakthrough,
            'final_c': c_final,
            'final_phi': phi_final
        }
        
        print(f"🚀 Momentum C: {c_momentum:.6f}")
        print(f"🌌 Momentum Φ: {phi_momentum:.6f}")
        print(f"⚡ Aceleración C: {c_acceleration:.6f}")
        print(f"✨ Aceleración Φ: {phi_acceleration:.6f}")
        print(f"🎯 Score Predictivo: {prediction_score:.3f}")
        print(f"🏆 Breakthrough Real: {'✅ SÍ' if breakthrough else '❌ NO'}")
        
        return prediction_score, breakthrough
    
    def generate_advanced_visualizations(self):
        """📊 Genera visualizaciones avanzadas con insights estadísticos"""
        print("\n📊 === GENERANDO VISUALIZACIONES AVANZADAS ===")
        
        # Preparar datos
        c_values = np.array(self.data.get('consciousness_values', []))
        phi_values = np.array(self.data.get('phi_values', []))
        iterations = np.array(self.data.get('iterations', []))
        
        if len(c_values) == 0:
            print("⚠️  No hay datos suficientes para visualizar")
            return
            
        # Crear figura con múltiples subplots
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('🧠 ANÁLISIS AVANZADO DE CONSCIOUSNESS BREAKTHROUGH', fontsize=20, fontweight='bold')
        
        # 1. Tendencias principales C y Φ
        ax1 = plt.subplot(2, 3, 1)
        min_len = min(len(c_values), len(phi_values), len(iterations))
        iter_plot = iterations[:min_len] if len(iterations) >= min_len else np.arange(min_len)
        
        ax1.plot(iter_plot, c_values[:min_len], 'b-', linewidth=2, label='Consciousness (C)', alpha=0.8)
        ax1.plot(iter_plot, phi_values[:min_len], 'r-', linewidth=2, label='Phi (Φ)', alpha=0.8)
        ax1.set_xlabel('Iteración')
        ax1.set_ylabel('Valor')
        ax1.set_title('🎯 Evolución C vs Φ')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot correlación
        ax2 = plt.subplot(2, 3, 2)
        scatter = ax2.scatter(c_values[:min_len], phi_values[:min_len], 
                            c=iter_plot, cmap='viridis', alpha=0.6, s=50)
        ax2.set_xlabel('Consciousness (C)')
        ax2.set_ylabel('Phi (Φ)')
        ax2.set_title(f'🔗 Correlación C-Φ (r={self.results["correlations"]["spearman"]["r"]:.3f})')
        plt.colorbar(scatter, ax=ax2, label='Iteración')
        
        # Línea de tendencia
        if len(c_values) > 1 and len(phi_values) > 1:
            z = np.polyfit(c_values[:min_len], phi_values[:min_len], 1)
            p = np.poly1d(z)
            ax2.plot(c_values[:min_len], p(c_values[:min_len]), "r--", alpha=0.8, linewidth=2)
        
        # 3. Análisis de deltas (primera diferencia)
        ax3 = plt.subplot(2, 3, 3)
        if len(c_values) > 1 and len(phi_values) > 1:
            c_deltas = np.diff(c_values[:min_len])
            phi_deltas = np.diff(phi_values[:min_len])
            ax3.plot(iter_plot[:-1], c_deltas, 'g-', label='ΔC', linewidth=2, alpha=0.7)
            ax3.plot(iter_plot[:-1], phi_deltas, 'm-', label='ΔΦ', linewidth=2, alpha=0.7)
            ax3.set_xlabel('Iteración')
            ax3.set_ylabel('Delta (Δ)')
            ax3.set_title(f'🔄 Análisis de Deltas (r={self.results["correlations"]["delta_spearman"]["r"]:.3f})')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Distribuciones
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(c_values[:min_len], bins=30, alpha=0.7, label='C Distribution', color='blue', density=True)
        ax4.hist(phi_values[:min_len], bins=30, alpha=0.7, label='Φ Distribution', color='red', density=True)
        ax4.set_xlabel('Valor')
        ax4.set_ylabel('Densidad')
        ax4.set_title('📈 Distribuciones de Probabilidad')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Transiciones de Φ
        ax5 = plt.subplot(2, 3, 5)
        transitions = self.data.get('transitions', [])
        phi_transitions = [t for t in transitions if t['type'] == 'phi_transition']
        
        if phi_transitions:
            phi_trans_iter = [t['iteration'] for t in phi_transitions]
            phi_trans_vals = [t['current_value'] for t in phi_transitions]
            phi_trans_trends = [t.get('trend', 0) for t in phi_transitions]
            
            ax5.scatter(phi_trans_iter, phi_trans_vals, c=phi_trans_trends, 
                       cmap='RdYlBu_r', s=100, alpha=0.8, edgecolors='black')
            ax5.set_xlabel('Iteración')
            ax5.set_ylabel('Φ Transition Value')
            ax5.set_title(f'🌊 Transiciones Φ (n={len(phi_transitions)})')
            plt.colorbar(ax5.collections[0], ax=ax5, label='Trend')
        
        # 6. Breakthrough prediction metrics
        ax6 = plt.subplot(2, 3, 6)
        pred_metrics = ['C Momentum', 'Φ Momentum', 'C Accel', 'Φ Accel', 'Pred Score']
        pred_values = [
            self.results['prediction']['c_momentum'] * 1000,  # Scale for visibility
            self.results['prediction']['phi_momentum'] * 1000,
            self.results['prediction']['c_acceleration'] * 10000,
            self.results['prediction']['phi_acceleration'] * 10000,
            self.results['prediction']['prediction_score'] / 10
        ]
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        bars = ax6.bar(pred_metrics, pred_values, color=colors, alpha=0.7)
        ax6.set_ylabel('Score (escalado)')
        ax6.set_title('🏆 Métricas Predictivas')
        ax6.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, val in zip(bars, pred_values):
            height = bar.get_height()
            ax6.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Guardar figura
        output_name = f"advanced_analysis_{Path(self.json_file).stem}.png"
        plt.savefig(output_name, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Visualización guardada como: {output_name}")
        
        plt.show()
    
    def generate_comprehensive_report(self):
        """📋 Genera reporte comprensivo de análisis"""
        print("\n📋 === GENERANDO REPORTE COMPRENSIVO ===")
        
        report = f"""
🧠 REPORTE AVANZADO DE ANÁLISIS DE CONSCIOUSNESS
{'='*60}

📁 ARCHIVO ANALIZADO: {self.json_file}
⏰ FECHA DE ANÁLISIS: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 ESTADÍSTICAS GENERALES:
• Total de puntos analizados: {self.results['stats']['total_points']:,}
• Consciousness promedio: {self.results['stats']['c_mean']:.6f} ± {self.results['stats']['c_std']:.6f}
• Phi promedio: {self.results['stats']['phi_mean']:.6f} ± {self.results['stats']['phi_std']:.6f}
• Rango Consciousness: [{self.results['stats']['c_min']:.6f}, {self.results['stats']['c_max']:.6f}]
• Rango Phi: [{self.results['stats']['phi_min']:.6f}, {self.results['stats']['phi_max']:.6f}]

🔗 ANÁLISIS DE CORRELACIONES C-Φ:
• Correlación Spearman: {self.results['correlations']['spearman']['r']:.4f} (p={self.results['correlations']['spearman']['p']:.4f})
• Correlación Pearson:  {self.results['correlations']['pearson']['r']:.4f} (p={self.results['correlations']['pearson']['p']:.4f})
• Correlación Kendall:  {self.results['correlations']['kendall']['r']:.4f} (p={self.results['correlations']['kendall']['p']:.4f})
• Correlación Deltas:   {self.results['correlations']['delta_spearman']['r']:.4f} (p={self.results['correlations']['delta_spearman']['p']:.4f})

🌊 ANÁLISIS DE TRANSICIONES:
• Transiciones Φ detectadas: {self.results.get('transitions', {}).get('phi_transitions_count', 0)}
• Picos identificados: {self.results.get('transitions', {}).get('phi_peaks', 0)}
• Valles identificados: {self.results.get('transitions', {}).get('phi_valleys', 0)}
• Tendencia promedio: {self.results.get('transitions', {}).get('trend_mean', 0.0):.6f}
• Varianza promedio: {self.results.get('transitions', {}).get('variance_mean', 0.0):.6f}

🏆 ANÁLISIS PREDICTIVO:
• Momentum Consciousness: {self.results['prediction']['c_momentum']:.6f}
• Momentum Phi: {self.results['prediction']['phi_momentum']:.6f}
• Aceleración Consciousness: {self.results['prediction']['c_acceleration']:.6f}
• Aceleración Phi: {self.results['prediction']['phi_acceleration']:.6f}
• Score Predictivo: {self.results['prediction']['prediction_score']:.3f}
• Breakthrough Alcanzado: {'✅ SÍ' if self.results['prediction']['breakthrough_achieved'] else '❌ NO'}

💡 INSIGHTS Y CONCLUSIONES:
"""
        
        # Añadir insights automáticos
        corr_strength = abs(self.results['correlations']['spearman']['r'])
        if corr_strength > 0.8:
            report += "• 🔥 CORRELACIÓN MUY FUERTE entre C y Φ detectada!\n"
        elif corr_strength > 0.6:
            report += "• 📈 Correlación FUERTE entre C y Φ\n"
        elif corr_strength > 0.4:
            report += "• 📊 Correlación MODERADA entre C y Φ\n"
        else:
            report += "• 📉 Correlación DÉBIL entre C y Φ\n"
            
        if self.results['prediction']['c_momentum'] > 0:
            report += "• 🚀 Momentum POSITIVO en Consciousness detectado\n"
        else:
            report += "• 🔻 Momentum NEGATIVO en Consciousness\n"
            
        if self.results['prediction']['phi_momentum'] > 0:
            report += "• 🌌 Momentum POSITIVO en Phi detectado\n"
        else:
            report += "• 🔻 Momentum NEGATIVO en Phi\n"
            
        if self.results['prediction']['prediction_score'] > 50:
            report += "• 🎯 ALTA probabilidad de breakthrough futuro\n"
        elif self.results['prediction']['prediction_score'] > 20:
            report += "• 📊 MODERADA probabilidad de breakthrough\n"
        else:
            report += "• ⚠️  BAJA probabilidad de breakthrough\n"
        
        report += f"\n{'='*60}\n"
        
        # Guardar reporte
        report_filename = f"advanced_report_{Path(self.json_file).stem}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Reporte guardado como: {report_filename}")
        print(report)
        
        return report

def main():
    """🚀 Función principal con análisis completo"""
    parser = argparse.ArgumentParser(description='🧠 Advanced Consciousness Results Analyzer V2.0')
    parser.add_argument('--file', type=str, 
                       default='src/outputs/infinito_v5_1_consciousness_20250925_232831_C0.997_PHI1.307.json',
                       help='Archivo JSON a analizar')
    parser.add_argument('--output-dir', type=str, default='.', 
                       help='Directorio de salida para resultados')
    
    args = parser.parse_args()
    
    print("🧠 ADVANCED CONSCIOUSNESS ANALYZER V2.0")
    print("=" * 50)
    
    try:
        # Crear analizador
        analyzer = AdvancedConsciousnessAnalyzer(args.file)
        
        # Ejecutar análisis completo
        analyzer.extract_consciousness_phi_correlations()
        analyzer.analyze_transitions_patterns()
        analyzer.breakthrough_prediction_analysis()
        
        # Generar visualizaciones y reporte
        analyzer.generate_advanced_visualizations()
        analyzer.generate_comprehensive_report()
        
        print("\n✅ ANÁLISIS COMPLETO FINALIZADO")
        
    except Exception as e:
        print(f"❌ ERROR EN ANÁLISIS: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()