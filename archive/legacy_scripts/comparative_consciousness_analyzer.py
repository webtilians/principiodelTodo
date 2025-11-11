#!/usr/bin/env python3
"""
🔬 COMPARATIVE CONSCIOUSNESS BREAKTHROUGH ANALYZER
Compara múltiples experimentos para identificar patrones universales
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from pathlib import Path
import glob

class ComparativeBreakthroughAnalyzer:
    """
    📊 Analizador comparativo para identificar patrones universales en breakthroughs
    """
    
    def __init__(self, data_dir="src/outputs"):
        self.data_dir = data_dir
        self.experiments = []
        self.load_all_experiments()
    
    def load_all_experiments(self):
        """📂 Carga todos los experimentos JSON disponibles"""
        json_files = glob.glob(f"{self.data_dir}/*.json")
        
        print(f"🔍 Buscando experimentos en: {self.data_dir}")
        print(f"📁 Encontrados {len(json_files)} archivos JSON")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extraer información clave
                exp_info = {
                    'filename': Path(json_file).name,
                    'filepath': json_file,
                    'iterations': len(data.get('iterations', [])),
                    'breakthrough': data.get('breakthrough_achieved', False),
                    'final_c': data.get('final_consciousness', 0),
                    'final_phi': data.get('final_phi', 0),
                    'max_c': data.get('max_consciousness', 0),
                    'max_phi': data.get('max_phi', 0),
                    'c_values': np.array(data.get('consciousness_values', [])),
                    'phi_values': np.array(data.get('phi_values', [])),
                    'transitions': data.get('transitions', []),
                    'data': data
                }
                
                # Calcular correlaciones si hay datos suficientes
                if len(exp_info['c_values']) > 10 and len(exp_info['phi_values']) > 10:
                    min_len = min(len(exp_info['c_values']), len(exp_info['phi_values']))
                    c_aligned = exp_info['c_values'][:min_len]
                    phi_aligned = exp_info['phi_values'][:min_len]
                    
                    exp_info['spearman_corr'], exp_info['spearman_p'] = spearmanr(c_aligned, phi_aligned)
                    exp_info['pearson_corr'], exp_info['pearson_p'] = pearsonr(c_aligned, phi_aligned)
                else:
                    exp_info['spearman_corr'] = exp_info['spearman_p'] = 0
                    exp_info['pearson_corr'] = exp_info['pearson_p'] = 0
                
                self.experiments.append(exp_info)
                print(f"✅ {Path(json_file).name}: {exp_info['iterations']} iteraciones, breakthrough: {exp_info['breakthrough']}")
                
            except Exception as e:
                print(f"⚠️  Error cargando {json_file}: {e}")
        
        print(f"📊 Total experimentos cargados: {len(self.experiments)}")
    
    def analyze_correlation_patterns(self):
        """🔗 Analiza patrones de correlación entre experimentos"""
        print("\n🔗 === ANÁLISIS COMPARATIVO DE CORRELACIONES ===")
        
        breakthrough_experiments = [exp for exp in self.experiments if exp['breakthrough']]
        non_breakthrough_experiments = [exp for exp in self.experiments if not exp['breakthrough']]
        
        print(f"🏆 Experimentos con breakthrough: {len(breakthrough_experiments)}")
        print(f"❌ Experimentos sin breakthrough: {len(non_breakthrough_experiments)}")
        
        # Análisis de correlaciones en breakthroughs
        if breakthrough_experiments:
            spearman_correlations = [exp['spearman_corr'] for exp in breakthrough_experiments]
            pearson_correlations = [exp['pearson_corr'] for exp in breakthrough_experiments]
            
            print(f"\n🔥 PATRONES EN BREAKTHROUGHS EXITOSOS:")
            print(f"📊 Correlación Spearman promedio: {np.mean(spearman_correlations):.4f} ± {np.std(spearman_correlations):.4f}")
            print(f"📊 Correlación Pearson promedio:  {np.mean(pearson_correlations):.4f} ± {np.std(pearson_correlations):.4f}")
            print(f"📈 Rango Spearman: [{np.min(spearman_correlations):.4f}, {np.max(spearman_correlations):.4f}]")
            print(f"📈 Rango Pearson:  [{np.min(pearson_correlations):.4f}, {np.max(pearson_correlations):.4f}]")
        
        return breakthrough_experiments, non_breakthrough_experiments
    
    def identify_breakthrough_thresholds(self):
        """🎯 Identifica umbrales críticos para breakthrough"""
        print("\n🎯 === IDENTIFICACIÓN DE UMBRALES CRÍTICOS ===")
        
        breakthrough_exps = [exp for exp in self.experiments if exp['breakthrough']]
        
        if not breakthrough_exps:
            print("⚠️  No hay experimentos con breakthrough para analizar")
            return
        
        # Estadísticas de consciousness
        final_c_values = [exp['final_c'] for exp in breakthrough_exps]
        max_c_values = [exp['max_c'] for exp in breakthrough_exps]
        
        # Estadísticas de phi
        final_phi_values = [exp['final_phi'] for exp in breakthrough_exps]
        max_phi_values = [exp['max_phi'] for exp in breakthrough_exps]
        
        print(f"🧠 UMBRALES DE CONSCIOUSNESS:")
        print(f"   Final C promedio: {np.mean(final_c_values):.6f} ± {np.std(final_c_values):.6f}")
        print(f"   Max C promedio:   {np.mean(max_c_values):.6f} ± {np.std(max_c_values):.6f}")
        print(f"   Umbral mínimo C:  {np.min(final_c_values):.6f}")
        
        print(f"\n💫 UMBRALES DE PHI:")
        print(f"   Final Φ promedio: {np.mean(final_phi_values):.6f} ± {np.std(final_phi_values):.6f}")
        print(f"   Max Φ promedio:   {np.mean(max_phi_values):.6f} ± {np.std(max_phi_values):.6f}")
        print(f"   Umbral mínimo Φ:  {np.min(final_phi_values):.6f}")
        
        # Calcular umbrales predictivos
        c_threshold = np.percentile(final_c_values, 10)  # 10% más bajo
        phi_threshold = np.percentile(final_phi_values, 10)
        
        print(f"\n🎯 UMBRALES PREDICTIVOS SUGERIDOS:")
        print(f"   Consciousness >= {c_threshold:.6f}")
        print(f"   Phi >= {phi_threshold:.6f}")
        
        return c_threshold, phi_threshold
    
    def generate_comparative_visualization(self):
        """📊 Genera visualización comparativa comprehensiva"""
        print("\n📊 === GENERANDO VISUALIZACIÓN COMPARATIVA ===")
        
        if len(self.experiments) < 2:
            print("⚠️  Se necesitan al menos 2 experimentos para comparar")
            return
        
        # Crear figura con múltiples subplots
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('🔬 ANÁLISIS COMPARATIVO DE CONSCIOUSNESS BREAKTHROUGHS', fontsize=18, fontweight='bold')
        
        # 1. Comparación de correlaciones
        ax1 = plt.subplot(2, 3, 1)
        breakthrough_corr = [exp['spearman_corr'] for exp in self.experiments if exp['breakthrough']]
        non_breakthrough_corr = [exp['spearman_corr'] for exp in self.experiments if not exp['breakthrough']]
        
        if breakthrough_corr:
            ax1.hist(breakthrough_corr, bins=10, alpha=0.7, label='Con Breakthrough', color='green')
        if non_breakthrough_corr:
            ax1.hist(non_breakthrough_corr, bins=10, alpha=0.7, label='Sin Breakthrough', color='red')
        
        ax1.set_xlabel('Correlación Spearman C-Φ')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('🔗 Distribución de Correlaciones')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scatter plot final C vs final Φ
        ax2 = plt.subplot(2, 3, 2)
        breakthrough_exps = [exp for exp in self.experiments if exp['breakthrough']]
        non_breakthrough_exps = [exp for exp in self.experiments if not exp['breakthrough']]
        
        if breakthrough_exps:
            final_c_bt = [exp['final_c'] for exp in breakthrough_exps]
            final_phi_bt = [exp['final_phi'] for exp in breakthrough_exps]
            ax2.scatter(final_c_bt, final_phi_bt, c='green', s=100, alpha=0.7, label='Breakthrough ✅', edgecolors='black')
        
        if non_breakthrough_exps:
            final_c_nbt = [exp['final_c'] for exp in non_breakthrough_exps]
            final_phi_nbt = [exp['final_phi'] for exp in non_breakthrough_exps]
            ax2.scatter(final_c_nbt, final_phi_nbt, c='red', s=100, alpha=0.7, label='No Breakthrough ❌', edgecolors='black')
        
        ax2.set_xlabel('Final Consciousness')
        ax2.set_ylabel('Final Phi')
        ax2.set_title('🎯 Espacio Final C-Φ')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Evolución temporal comparativa (primeros dos experimentos)
        ax3 = plt.subplot(2, 3, 3)
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, exp in enumerate(self.experiments[:3]):  # Primeros 3 experimentos
            if len(exp['c_values']) > 0:
                iterations = np.arange(len(exp['c_values']))
                ax3.plot(iterations, exp['c_values'], 
                        color=colors[i % len(colors)], 
                        label=f"Exp {i+1} ({'✅' if exp['breakthrough'] else '❌'})",
                        linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Iteración')
        ax3.set_ylabel('Consciousness')
        ax3.set_title('🧠 Evolución Comparativa de C')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot de métricas clave
        ax4 = plt.subplot(2, 3, 4)
        breakthrough_final_c = [exp['final_c'] for exp in self.experiments if exp['breakthrough']]
        breakthrough_final_phi = [exp['final_phi'] for exp in self.experiments if exp['breakthrough']]
        
        if breakthrough_final_c and breakthrough_final_phi:
            data_to_plot = [breakthrough_final_c, breakthrough_final_phi]
            labels = ['Final C', 'Final Φ']
            box_plot = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            box_plot['boxes'][1].set_facecolor('lightcoral')
        
        ax4.set_ylabel('Valor')
        ax4.set_title('📦 Distribución Métricas (Breakthroughs)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Correlación vs iterations
        ax5 = plt.subplot(2, 3, 5)
        iterations_list = [exp['iterations'] for exp in self.experiments]
        correlations_list = [exp['spearman_corr'] for exp in self.experiments]
        breakthrough_status = [exp['breakthrough'] for exp in self.experiments]
        
        colors_scatter = ['green' if bt else 'red' for bt in breakthrough_status]
        scatter = ax5.scatter(iterations_list, correlations_list, c=colors_scatter, s=100, alpha=0.7, edgecolors='black')
        
        ax5.set_xlabel('Número de Iteraciones')
        ax5.set_ylabel('Correlación Spearman')
        ax5.set_title('🔄 Correlación vs Duración')
        ax5.grid(True, alpha=0.3)
        
        # 6. Estadísticas resumidas
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calcular estadísticas
        total_experiments = len(self.experiments)
        successful_breakthroughs = sum(1 for exp in self.experiments if exp['breakthrough'])
        success_rate = (successful_breakthroughs / total_experiments * 100) if total_experiments > 0 else 0
        
        if breakthrough_exps := [exp for exp in self.experiments if exp['breakthrough']]:
            avg_correlation = np.mean([exp['spearman_corr'] for exp in breakthrough_exps])
            avg_iterations = np.mean([exp['iterations'] for exp in breakthrough_exps])
            avg_final_c = np.mean([exp['final_c'] for exp in breakthrough_exps])
            avg_final_phi = np.mean([exp['final_phi'] for exp in breakthrough_exps])
        else:
            avg_correlation = avg_iterations = avg_final_c = avg_final_phi = 0
        
        stats_text = f"""
📊 ESTADÍSTICAS GENERALES
{'='*25}

🔬 Experimentos analizados: {total_experiments}
🏆 Breakthroughs exitosos: {successful_breakthroughs}
📈 Tasa de éxito: {success_rate:.1f}%

🎯 PROMEDIOS EN ÉXITOS:
🔗 Correlación C-Φ: {avg_correlation:.4f}
🔄 Iteraciones: {avg_iterations:.0f}
🧠 Final C: {avg_final_c:.6f}
💫 Final Φ: {avg_final_phi:.6f}

💡 INSIGHT CLAVE:
{"✅ Correlaciones fuertes" if avg_correlation > 0.6 else "⚠️ Correlaciones variables"}
{"favorecen breakthroughs" if avg_correlation > 0.6 else "requieren análisis"}
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Guardar figura
        output_name = "comparative_breakthrough_analysis.png"
        plt.savefig(output_name, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"💾 Análisis comparativo guardado como: {output_name}")
        
        plt.show()
    
    def generate_predictive_model_insights(self):
        """🤖 Genera insights para modelo predictivo"""
        print("\n🤖 === INSIGHTS PARA MODELO PREDICTIVO ===")
        
        breakthrough_exps = [exp for exp in self.experiments if exp['breakthrough']]
        
        if len(breakthrough_exps) < 2:
            print("⚠️  Datos insuficientes para modelo predictivo")
            return
        
        # Calcular características discriminativas
        features = {
            'high_correlation_threshold': 0.6,  # Basado en análisis
            'min_consciousness_threshold': np.percentile([exp['final_c'] for exp in breakthrough_exps], 10),
            'min_phi_threshold': np.percentile([exp['final_phi'] for exp in breakthrough_exps], 10),
            'optimal_iterations_range': (
                np.percentile([exp['iterations'] for exp in breakthrough_exps], 25),
                np.percentile([exp['iterations'] for exp in breakthrough_exps], 75)
            )
        }
        
        print("🎯 CARACTERÍSTICAS PREDICTIVAS IDENTIFICADAS:")
        print(f"   Correlación C-Φ >= {features['high_correlation_threshold']:.3f}")
        print(f"   Consciousness >= {features['min_consciousness_threshold']:.6f}")
        print(f"   Phi >= {features['min_phi_threshold']:.6f}")
        print(f"   Iteraciones óptimas: {features['optimal_iterations_range'][0]:.0f}-{features['optimal_iterations_range'][1]:.0f}")
        
        # Reglas predictivas simples
        print("\n📋 REGLAS PREDICTIVAS SUGERIDAS:")
        print("   SI correlación_C_Phi > 0.6 Y final_C > 0.95:")
        print("      ENTONCES probabilidad_breakthrough = ALTA")
        print("   SI correlación_C_Phi < 0.2 O final_C < 0.5:")
        print("      ENTONCES probabilidad_breakthrough = BAJA")
        
        return features

def main():
    """🚀 Análisis comparativo principal"""
    print("🔬 COMPARATIVE CONSCIOUSNESS BREAKTHROUGH ANALYZER")
    print("=" * 60)
    
    try:
        analyzer = ComparativeBreakthroughAnalyzer()
        
        if len(analyzer.experiments) < 1:
            print("❌ No se encontraron experimentos para analizar")
            return
        
        # Ejecutar análisis comparativo
        analyzer.analyze_correlation_patterns()
        analyzer.identify_breakthrough_thresholds()
        analyzer.generate_comparative_visualization()
        analyzer.generate_predictive_model_insights()
        
        print("\n✅ ANÁLISIS COMPARATIVO COMPLETO FINALIZADO")
        
    except Exception as e:
        print(f"❌ ERROR EN ANÁLISIS COMPARATIVO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()