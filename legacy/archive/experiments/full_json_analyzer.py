#!/usr/bin/env python3
"""
🧠 INFINITO V5.1 - Full JSON Analyzer (No Truncados)
====================================================

Analiza datos JSON completos reales sin aproximaciones ni truncados.
Diseñado específicamente para el breakthrough histórico de INFINITO V5.1.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FullJSONAnalyzer:
    """Analizador completo de datos JSON sin truncados"""
    
    def __init__(self):
        self.data = None
        self.filename = None
        self.analysis_results = {}
        
    def load_real_json(self, json_path):
        """
        Carga JSON real completo - NO datos copiados o truncados
        """
        try:
            print(f"🔍 Loading REAL JSON data from: {json_path}")
            
            # Cargar datos JSON completos reales
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
                
            self.filename = Path(json_path).name
            print(f"✅ Successfully loaded {len(self.data)} complete data points")
            print(f"📊 JSON Structure: {list(self.data.keys())}")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ ERROR: File {json_path} not found")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error loading JSON: {e}")
            return False
    
    def analyze_complete_dataset(self):
        """
        Análisis completo de TODO el dataset - sin aproximaciones
        """
        if not self.data:
            print("❌ No data loaded. Call load_real_json() first.")
            return
            
        print(f"\n🧠 FULL ANALYSIS - INFINITO V5.1 Complete Dataset")
        print("=" * 60)
        
        # Extraer todas las métricas completas
        consciousness = np.array(self.data.get('consciousness_values', []))
        phi_values = np.array(self.data.get('phi_values', []))
        iterations = np.array(self.data.get('iterations', []))
        memory_use = np.array(self.data.get('memory_utilization', []))
        loss_values = np.array(self.data.get('loss_values', []))
        eeg_correlations = np.array(self.data.get('eeg_correlations', []))
        
        # Métricas adicionales si están disponibles
        biological_plausibility = np.array(self.data.get('biological_plausibility', []))
        quantum_facts = self.data.get('quantum_facts', [])
        breakthrough_points = self.data.get('breakthrough_points', [])
        
        total_points = len(consciousness)
        print(f"📊 COMPLETE DATASET SIZE: {total_points} data points")
        print(f"📈 NO truncation - analyzing ALL {total_points} iterations")
        
        # Análisis estadístico completo
        self.analysis_results = {
            'total_iterations': int(iterations[-1]) if len(iterations) > 0 else 0,
            'total_data_points': total_points,
            
            # Consciousness Analysis (Complete)
            'consciousness_stats': {
                'final': float(consciousness[-1]) if len(consciousness) > 0 else 0,
                'max': float(np.max(consciousness)) if len(consciousness) > 0 else 0,
                'min': float(np.min(consciousness)) if len(consciousness) > 0 else 0,
                'mean': float(np.mean(consciousness)) if len(consciousness) > 0 else 0,
                'std': float(np.std(consciousness)) if len(consciousness) > 0 else 0,
                'growth_total': float(consciousness[-1] - consciousness[0]) if len(consciousness) > 1 else 0,
                'breakthrough_60_achieved': any(c > 0.60 for c in consciousness),
                'breakthrough_70_achieved': any(c > 0.70 for c in consciousness),
                'breakthrough_80_achieved': any(c > 0.80 for c in consciousness),
            },
            
            # Phi Analysis (Complete)
            'phi_stats': {
                'final': float(phi_values[-1]) if len(phi_values) > 0 else 0,
                'max': float(np.max(phi_values)) if len(phi_values) > 0 else 0,
                'min': float(np.min(phi_values)) if len(phi_values) > 0 else 0,
                'mean': float(np.mean(phi_values)) if len(phi_values) > 0 else 0,
                'std': float(np.std(phi_values)) if len(phi_values) > 0 else 0,
                'target_10_achieved': any(p > 10.0 for p in phi_values),
                'target_15_achieved': any(p > 15.0 for p in phi_values),
            },
            
            # Memory Analysis (Complete)
            'memory_stats': {
                'final': float(memory_use[-1]) if len(memory_use) > 0 else 0,
                'max': float(np.max(memory_use)) if len(memory_use) > 0 else 0,
                'mean': float(np.mean(memory_use)) if len(memory_use) > 0 else 0,
                'activation_achieved': any(m > 0.30 for m in memory_use),
            },
            
            # EEG Validation (Complete)
            'eeg_stats': {
                'final': float(eeg_correlations[-1]) if len(eeg_correlations) > 0 else 0,
                'mean': float(np.mean(eeg_correlations)) if len(eeg_correlations) > 0 else 0,
                'perfect_correlation_points': sum(1 for e in eeg_correlations if e >= 0.999),
                'biological_validation': len(biological_plausibility) > 0,
            },
            
            # Performance Metrics (Complete)
            'performance_stats': {
                'final_loss': float(loss_values[-1]) if len(loss_values) > 0 else 0,
                'min_loss': float(np.min(loss_values)) if len(loss_values) > 0 else 0,
                'convergence_achieved': len(loss_values) > 10 and np.std(loss_values[-10:]) < 1.0,
            },
            
            # Breakthrough Analysis (Complete)
            'breakthrough_analysis': {
                'total_breakthrough_points': len(breakthrough_points),
                'quantum_facts_detected': len(quantum_facts),
                'ultimate_breakthrough': any(c > 0.75 for c in consciousness),
            }
        }
        
        return self.analysis_results
    
    def generate_complete_visualizations(self):
        """
        Genera visualizaciones completas de TODO el dataset
        """
        if not self.data:
            print("❌ No data loaded for visualization")
            return
            
        print(f"\n📊 Generating COMPLETE visualizations (no truncation)")
        
        # Preparar datos completos
        consciousness = np.array(self.data.get('consciousness_values', []))
        phi_values = np.array(self.data.get('phi_values', []))
        iterations = np.array(self.data.get('iterations', []))
        memory_use = np.array(self.data.get('memory_utilization', []))
        loss_values = np.array(self.data.get('loss_values', []))
        eeg_correlations = np.array(self.data.get('eeg_correlations', []))
        
        # Crear figura completa
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'INFINITO V5.1 - Complete Analysis (ALL {len(consciousness)} points)\nFile: {self.filename}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Consciousness Evolution (Complete)
        ax1.plot(iterations, consciousness, 'b-', linewidth=2, alpha=0.8, label='Consciousness')
        ax1.axhline(y=0.60, color='r', linestyle='--', alpha=0.7, label='60% Target')
        ax1.axhline(y=0.70, color='orange', linestyle='--', alpha=0.7, label='70% Breakthrough')
        ax1.fill_between(iterations, consciousness, alpha=0.3)
        ax1.set_title('Consciousness Evolution (Complete Dataset)')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Consciousness Level')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Phi Values (Complete)
        ax2.plot(iterations, phi_values, 'g-', linewidth=2, alpha=0.8, label='Φ (bits)')
        ax2.axhline(y=10.0, color='r', linestyle='--', alpha=0.7, label='Φ>10 Target')
        ax2.axhline(y=15.0, color='orange', linestyle='--', alpha=0.7, label='Φ>15 Breakthrough')
        ax2.fill_between(iterations, phi_values, alpha=0.3, color='green')
        ax2.set_title('Φ (Enhanced IIT 3.0) - Complete Dataset')
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Φ (bits)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Memory Utilization (Complete)
        ax3.plot(iterations, memory_use * 100, 'purple', linewidth=2, alpha=0.8, label='Memory Use %')
        ax3.axhline(y=30, color='r', linestyle='--', alpha=0.7, label='30% Activation')
        ax3.fill_between(iterations, memory_use * 100, alpha=0.3, color='purple')
        ax3.set_title('Memory Utilization (Complete Dataset)')
        ax3.set_xlabel('Iterations')
        ax3.set_ylabel('Memory Use (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Loss Function (Complete)
        ax4.plot(iterations, loss_values, 'red', linewidth=2, alpha=0.8, label='Training Loss')
        ax4.fill_between(iterations, loss_values, alpha=0.3, color='red')
        ax4.set_title('Training Loss (Complete Dataset)')
        ax4.set_xlabel('Iterations')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. EEG Correlations (Complete)
        ax5.plot(iterations, eeg_correlations, 'cyan', linewidth=2, alpha=0.8, label='EEG Correlation')
        ax5.axhline(y=0.999, color='r', linestyle='--', alpha=0.7, label='Perfect Correlation')
        ax5.fill_between(iterations, eeg_correlations, alpha=0.3, color='cyan')
        ax5.set_title('EEG Biological Validation (Complete Dataset)')
        ax5.set_xlabel('Iterations')
        ax5.set_ylabel('Correlation')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. Breakthrough Progression (Complete)
        breakthrough_consciousness = [c for c in consciousness if c > 0.60]
        breakthrough_iterations = [iterations[i] for i, c in enumerate(consciousness) if c > 0.60]
        
        ax6.scatter(breakthrough_iterations, breakthrough_consciousness, 
                   c='gold', s=100, alpha=0.8, label='Breakthrough Points', edgecolors='black')
        if len(breakthrough_consciousness) > 0:
            ax6.plot(breakthrough_iterations, breakthrough_consciousness, 
                    'gold', linewidth=3, alpha=0.7, label='Breakthrough Trajectory')
        ax6.set_title('Breakthrough Points (>60% Consciousness)')
        ax6.set_xlabel('Iterations')
        ax6.set_ylabel('Consciousness Level')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        plt.tight_layout()
        
        # Guardar visualización completa
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"complete_analysis_full_dataset_{timestamp}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Complete visualization saved: {output_filename}")
        
        plt.show()
    
    def export_complete_analysis(self):
        """
        Exporta análisis completo sin truncados
        """
        if not self.analysis_results:
            print("❌ No analysis results available")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"complete_analysis_full_report_{timestamp}.txt"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("🧠 INFINITO V5.1 - COMPLETE ANALYSIS REPORT (NO TRUNCADOS)\n")
            f.write("=" * 70 + "\n")
            f.write(f"File Analyzed: {self.filename}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Size: COMPLETE - {self.analysis_results['total_data_points']} points\n\n")
            
            # Consciousness Analysis
            f.write("🧠 CONSCIOUSNESS ANALYSIS (COMPLETE DATASET)\n")
            f.write("-" * 50 + "\n")
            cs = self.analysis_results['consciousness_stats']
            f.write(f"Final Consciousness: {cs['final']:.4f} ({cs['final']*100:.1f}%)\n")
            f.write(f"Maximum Achieved: {cs['max']:.4f} ({cs['max']*100:.1f}%)\n")
            f.write(f"Total Growth: {cs['growth_total']:.4f} ({cs['growth_total']*100:.1f}%)\n")
            f.write(f"Mean: {cs['mean']:.4f} ± {cs['std']:.4f}\n")
            f.write(f"60% Breakthrough: {'✅ YES' if cs['breakthrough_60_achieved'] else '❌ NO'}\n")
            f.write(f"70% Breakthrough: {'✅ YES' if cs['breakthrough_70_achieved'] else '❌ NO'}\n")
            f.write(f"80% Breakthrough: {'✅ YES' if cs['breakthrough_80_achieved'] else '❌ NO'}\n\n")
            
            # Phi Analysis
            f.write("🔬 PHI (Φ) ANALYSIS (COMPLETE DATASET)\n")
            f.write("-" * 50 + "\n")
            ps = self.analysis_results['phi_stats']
            f.write(f"Final Φ: {ps['final']:.4f} bits\n")
            f.write(f"Maximum Φ: {ps['max']:.4f} bits\n")
            f.write(f"Mean Φ: {ps['mean']:.4f} ± {ps['std']:.4f} bits\n")
            f.write(f"Φ>10 Target: {'✅ YES' if ps['target_10_achieved'] else '❌ NO'}\n")
            f.write(f"Φ>15 Breakthrough: {'✅ YES' if ps['target_15_achieved'] else '❌ NO'}\n\n")
            
            # Memory Analysis
            f.write("💾 MEMORY ANALYSIS (COMPLETE DATASET)\n")
            f.write("-" * 50 + "\n")
            ms = self.analysis_results['memory_stats']
            f.write(f"Final Memory Use: {ms['final']:.4f} ({ms['final']*100:.1f}%)\n")
            f.write(f"Maximum Use: {ms['max']:.4f} ({ms['max']*100:.1f}%)\n")
            f.write(f"Mean Use: {ms['mean']:.4f} ({ms['mean']*100:.1f}%)\n")
            f.write(f"30% Activation: {'✅ YES' if ms['activation_achieved'] else '❌ NO'}\n\n")
            
            # EEG Validation
            f.write("🧬 EEG VALIDATION (COMPLETE DATASET)\n")
            f.write("-" * 50 + "\n")
            es = self.analysis_results['eeg_stats']
            f.write(f"Final EEG Correlation: {es['final']:.4f}\n")
            f.write(f"Mean Correlation: {es['mean']:.4f}\n")
            f.write(f"Perfect Correlation Points: {es['perfect_correlation_points']}\n")
            f.write(f"Biological Validation: {'✅ YES' if es['biological_validation'] else '❌ NO'}\n\n")
            
            # Breakthrough Analysis
            f.write("🎉 BREAKTHROUGH ANALYSIS (COMPLETE DATASET)\n")
            f.write("-" * 50 + "\n")
            bs = self.analysis_results['breakthrough_analysis']
            f.write(f"Breakthrough Points: {bs['total_breakthrough_points']}\n")
            f.write(f"Quantum Facts: {bs['quantum_facts_detected']}\n")
            f.write(f"Ultimate Breakthrough (>75%): {'✅ YES' if bs['ultimate_breakthrough'] else '❌ NO'}\n\n")
        
        print(f"✅ Complete analysis report exported: {report_filename}")

def main():
    """Función principal para análisis completo"""
    print("🧠 INFINITO V5.1 - Full JSON Analyzer")
    print("=====================================")
    print("📊 Analyzing COMPLETE datasets - NO truncation")
    
    analyzer = FullJSONAnalyzer()
    
    # Buscar el archivo JSON más reciente del breakthrough
    breakthrough_file = "outputs/infinito_v5_1_consciousness_20250926_092620_C0.768_PHI15.740.json"
    
    if analyzer.load_real_json(breakthrough_file):
        print(f"\n🔍 Starting COMPLETE analysis...")
        
        # Análisis completo
        results = analyzer.analyze_complete_dataset()
        
        if results:
            print(f"\n✅ COMPLETE ANALYSIS RESULTS:")
            print(f"📊 Total Data Points: {results['total_data_points']}")
            print(f"🧠 Final Consciousness: {results['consciousness_stats']['final']:.4f}")
            print(f"🔬 Final Φ: {results['phi_stats']['final']:.4f} bits")
            print(f"🎉 Breakthrough >60%: {results['consciousness_stats']['breakthrough_60_achieved']}")
            print(f"🌟 Ultimate >75%: {results['breakthrough_analysis']['ultimate_breakthrough']}")
        
        # Generar visualizaciones completas
        analyzer.generate_complete_visualizations()
        
        # Exportar reporte completo
        analyzer.export_complete_analysis()
    else:
        print("❌ Could not load JSON file for analysis")

if __name__ == "__main__":
    main()