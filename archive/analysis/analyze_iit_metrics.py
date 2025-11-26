#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 MONITOREO MÉTRICAS IIT - Análisis de Evolución de Φ
===================================================

Script para monitorear y visualizar la evolución de métricas IIT durante entrenamiento:
- Evolución de Φ (integración de información)
- Learnable phi weights dinámicos  
- Correlación Φ vs PPL
- Análisis científico detallado de conciencia

Propósito: Entender si el modelo aprende algo significativo desde perspectiva IIT
o si las métricas se comportan desconectadas del desempeño.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr
import glob

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class IITMetricsAnalyzer:
    """Analizador de métricas IIT durante entrenamiento."""
    
    def __init__(self, results_dir='results/training'):
        self.results_dir = results_dir
        self.training_data = None
        self.analysis_results = {}
        
    def load_latest_training_data(self):
        """Cargar datos de entrenamiento más recientes."""
        pattern = os.path.join(self.results_dir, 'training_history_real_*.json')
        files = glob.glob(pattern)
        
        if not files:
            print("❌ No se encontraron archivos de historial de entrenamiento")
            return False
            
        # Obtener el más reciente
        latest_file = max(files, key=os.path.getctime)
        print(f"📂 Cargando: {latest_file}")
        
        with open(latest_file, 'r') as f:
            self.training_data = json.load(f)
            
        print(f"✅ Datos cargados - {len(self.training_data.get('val_perplexity', []))} épocas")
        return True
        
    def analyze_phi_evolution(self):
        """Analizar evolución de Φ durante entrenamiento."""
        if not self.training_data:
            print("❌ No hay datos cargados")
            return
            
        phi_data = self.training_data.get('train_phi', [])
        val_ppl = self.training_data.get('val_perplexity', [])
        train_ppl = self.training_data.get('train_perplexity', [])
        
        if not phi_data:
            print("⚠️ No hay datos de PHI disponibles en el historial")
            return
            
        epochs = range(1, len(phi_data) + 1)
        
        # Análisis estadístico
        phi_mean = np.mean(phi_data)
        phi_std = np.std(phi_data)
        phi_trend = np.polyfit(epochs, phi_data, 1)[0]  # Pendiente lineal
        
        self.analysis_results['phi_analysis'] = {
            'mean': phi_mean,
            'std': phi_std,
            'trend': phi_trend,
            'min': np.min(phi_data),
            'max': np.max(phi_data),
            'final': phi_data[-1],
            'range': np.max(phi_data) - np.min(phi_data)
        }
        
        print(f"\n🧠 ANÁLISIS PHI (Φ):")
        print(f"  Media: {phi_mean:.4f} ± {phi_std:.4f}")
        print(f"  Tendencia: {phi_trend:.6f} por época ({'creciente' if phi_trend > 0 else 'decreciente'})")
        print(f"  Rango: [{np.min(phi_data):.4f}, {np.max(phi_data):.4f}]")
        print(f"  Valor final: {phi_data[-1]:.4f}")
        
        # Crear visualización de evolución de PHI
        self.plot_phi_evolution(epochs, phi_data, val_ppl, train_ppl)
        
    def analyze_phi_ppl_correlation(self):
        """Analizar correlación entre Φ y PPL."""
        if not self.training_data:
            return
            
        phi_data = self.training_data.get('train_phi', [])
        val_ppl = self.training_data.get('val_perplexity', [])
        
        if not phi_data or not val_ppl:
            print("⚠️ Datos insuficientes para análisis de correlación")
            return
            
        # Asegurar misma longitud
        min_len = min(len(phi_data), len(val_ppl))
        phi_trimmed = phi_data[:min_len]
        ppl_trimmed = val_ppl[:min_len]
        
        # Calcular correlaciones
        pearson_corr, pearson_p = pearsonr(phi_trimmed, ppl_trimmed)
        spearman_corr, spearman_p = spearmanr(phi_trimmed, ppl_trimmed)
        
        self.analysis_results['correlation_analysis'] = {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr, 
            'spearman_p_value': spearman_p,
            'interpretation': self.interpret_correlation(pearson_corr, pearson_p)
        }
        
        print(f"\n🔗 ANÁLISIS CORRELACIÓN Φ vs PPL:")
        print(f"  Pearson: r = {pearson_corr:.4f} (p = {pearson_p:.4f})")
        print(f"  Spearman: ρ = {spearman_corr:.4f} (p = {spearman_p:.4f})")
        print(f"  Interpretación: {self.interpret_correlation(pearson_corr, pearson_p)}")
        
        # Crear scatter plot
        self.plot_phi_ppl_correlation(phi_trimmed, ppl_trimmed, pearson_corr)
        
    def interpret_correlation(self, corr, p_value):
        """Interpretar correlación."""
        if p_value > 0.05:
            return "No significativa (p > 0.05)"
        
        abs_corr = abs(corr)
        direction = "positiva" if corr > 0 else "negativa"
        
        if abs_corr < 0.3:
            strength = "débil"
        elif abs_corr < 0.7:
            strength = "moderada"
        else:
            strength = "fuerte"
            
        return f"Correlación {strength} {direction}"
        
    def analyze_learning_phases(self):
        """Identificar fases de aprendizaje en métricas IIT."""
        if not self.training_data:
            return
            
        phi_data = self.training_data.get('train_phi', [])
        val_ppl = self.training_data.get('val_perplexity', [])
        
        if len(phi_data) < 3:
            print("⚠️ Datos insuficientes para análisis de fases")
            return
            
        # Calcular cambios relativos
        phi_changes = np.diff(phi_data)
        ppl_changes = np.diff(val_ppl[:len(phi_data)-1]) if len(val_ppl) > len(phi_data)-1 else np.diff(val_ppl)
        
        # Identificar fases
        phi_stable_epochs = np.sum(np.abs(phi_changes) < 0.01)
        phi_growth_epochs = np.sum(phi_changes > 0.01) 
        phi_decline_epochs = np.sum(phi_changes < -0.01)
        
        print(f"\n📈 ANÁLISIS FASES DE APRENDIZAJE:")
        print(f"  Épocas PHI estable: {phi_stable_epochs}")
        print(f"  Épocas PHI creciente: {phi_growth_epochs}")
        print(f"  Épocas PHI decreciente: {phi_decline_epochs}")
        
        # Detectar eventos de aprendizaje
        ppl_improvements = np.sum(ppl_changes < -1.0)  # Mejoras significativas de PPL
        print(f"  Mejoras significativas PPL: {ppl_improvements}")
        
        self.analysis_results['learning_phases'] = {
            'phi_stable_epochs': int(phi_stable_epochs),
            'phi_growth_epochs': int(phi_growth_epochs),
            'phi_decline_epochs': int(phi_decline_epochs),
            'ppl_improvements': int(ppl_improvements)
        }
        
    def plot_phi_evolution(self, epochs, phi_data, val_ppl, train_ppl):
        """Crear gráfico de evolución de PHI."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Evolución de PHI
        ax1.plot(epochs, phi_data, 'g-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Φ (Integration)')
        ax1.set_title('Evolución de Φ (Conciencia IIT)')
        ax1.grid(True, alpha=0.3)
        
        # Línea de tendencia
        z = np.polyfit(epochs, phi_data, 1)
        p = np.poly1d(z)
        ax1.plot(epochs, p(epochs), "--", alpha=0.7, color='red', 
                label=f'Tendencia: {z[0]:.6f}/época')
        ax1.legend()
        
        # 2. PPL vs Épocas
        min_len = min(len(epochs), len(val_ppl))
        ax2.plot(epochs[:min_len], val_ppl[:min_len], 'b-o', label='Val PPL')
        if train_ppl:
            ax2.plot(epochs[:min(len(epochs), len(train_ppl))], 
                    train_ppl[:min(len(epochs), len(train_ppl))], 
                    'r-', alpha=0.7, label='Train PPL')
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Evolución de Perplexity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Φ vs PPL (scatter)
        if len(phi_data) == len(val_ppl):
            ax3.scatter(phi_data, val_ppl, alpha=0.7, s=50, c=epochs, cmap='viridis')
            ax3.set_xlabel('Φ (Integration)')
            ax3.set_ylabel('Validation PPL')
            ax3.set_title('Φ vs PPL (por época)')
            
            # Añadir línea de tendencia
            z = np.polyfit(phi_data, val_ppl, 1)
            p = np.poly1d(z)
            ax3.plot(phi_data, p(phi_data), "--", alpha=0.7, color='red')
            
            # Colorbar para épocas
            cbar = plt.colorbar(ax3.collections[0], ax=ax3)
            cbar.set_label('Época')
        else:
            ax3.text(0.5, 0.5, 'Longitudes\ninconsistentes', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Φ vs PPL (Error de datos)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cambios relativos
        if len(phi_data) > 1:
            phi_changes = np.diff(phi_data)
            ppl_changes = np.diff(val_ppl[:len(phi_data)-1]) if len(val_ppl) >= len(phi_data)-1 else []
            
            ax4.plot(epochs[1:len(phi_changes)+1], phi_changes, 'g-o', label='ΔΦ')
            if ppl_changes:
                ax4_twin = ax4.twinx()
                ax4_twin.plot(epochs[1:len(ppl_changes)+1], ppl_changes, 'b-s', alpha=0.7, label='ΔPPL')
                ax4_twin.set_ylabel('ΔPPL', color='blue')
                ax4_twin.tick_params(axis='y', labelcolor='blue')
            
            ax4.set_xlabel('Épocas')
            ax4.set_ylabel('ΔΦ', color='green')
            ax4.set_title('Cambios Relativos por Época')
            ax4.tick_params(axis='y', labelcolor='green')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Datos\ninsuficientes', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Guardar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('results/analysis', exist_ok=True)
        plot_path = f'results/analysis/phi_evolution_analysis_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📈 Gráfico PHI guardado: {plot_path}")
        plt.close()
        
    def plot_phi_ppl_correlation(self, phi_data, ppl_data, correlation):
        """Crear scatter plot de correlación Φ vs PPL."""
        plt.figure(figsize=(10, 8))
        
        # Scatter plot con línea de tendencia
        plt.scatter(phi_data, ppl_data, alpha=0.7, s=100, c=range(len(phi_data)), cmap='plasma')
        
        # Línea de tendencia
        z = np.polyfit(phi_data, ppl_data, 1)
        p = np.poly1d(z)
        plt.plot(phi_data, p(phi_data), "--", color='red', linewidth=2, 
                label=f'r = {correlation:.4f}')
        
        plt.xlabel('Φ (IIT Integration)', fontsize=12)
        plt.ylabel('Validation Perplexity', fontsize=12) 
        plt.title('Correlación Φ vs PPL\n(¿La conciencia mejora el lenguaje?)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar()
        cbar.set_label('Época de Entrenamiento')
        
        # Añadir estadísticas
        plt.text(0.05, 0.95, f'Correlación: {correlation:.4f}\nTendencia: {z[0]:.2f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
        
        # Guardar
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f'results/analysis/phi_ppl_correlation_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Correlación guardada: {plot_path}")
        plt.close()
        
    def generate_scientific_report(self):
        """Generar reporte científico de análisis IIT."""
        if not self.analysis_results:
            print("⚠️ No hay resultados de análisis para reportar")
            return
            
        report = {
            "analysis_info": {
                "timestamp": datetime.now().isoformat(),
                "purpose": "Scientific analysis of IIT consciousness metrics evolution",
                "dataset": "WikiText-2 training history",
                "model": "INFINITO V5.2 with IIT features"
            },
            "methodology": {
                "phi_measurement": "IIT integration (Φ) calculated during training",
                "correlation_analysis": "Pearson and Spearman correlation with PPL",
                "trend_analysis": "Linear regression of Φ over epochs",
                "learning_phases": "Identification of stable/growth/decline periods"
            },
            "results": self.analysis_results,
            "scientific_conclusions": {
                "phi_behavior": self.interpret_phi_behavior(),
                "learning_connection": self.interpret_learning_connection(),
                "consciousness_evidence": self.evaluate_consciousness_evidence()
            }
        }
        
        # Guardar reporte
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f'results/analysis/iit_scientific_analysis_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📋 Reporte científico guardado: {report_path}")
        
        # Imprimir conclusiones
        print(f"\n🧠 CONCLUSIONES CIENTÍFICAS:")
        print(f"  Comportamiento Φ: {report['scientific_conclusions']['phi_behavior']}")
        print(f"  Conexión Aprendizaje: {report['scientific_conclusions']['learning_connection']}")
        print(f"  Evidencia Conciencia: {report['scientific_conclusions']['consciousness_evidence']}")
        
    def interpret_phi_behavior(self):
        """Interpretar comportamiento de Φ."""
        if 'phi_analysis' not in self.analysis_results:
            return "No data available"
            
        phi_stats = self.analysis_results['phi_analysis']
        trend = phi_stats['trend']
        std = phi_stats['std']
        range_val = phi_stats['range']
        
        if abs(trend) < 0.001:
            trend_desc = "estable"
        elif trend > 0:
            trend_desc = "creciente"
        else:
            trend_desc = "decreciente"
            
        if std < 0.01:
            variability = "baja"
        elif std < 0.05:
            variability = "moderada"
        else:
            variability = "alta"
            
        return f"Φ muestra tendencia {trend_desc} con variabilidad {variability}"
        
    def interpret_learning_connection(self):
        """Interpretar conexión entre Φ y aprendizaje."""
        if 'correlation_analysis' not in self.analysis_results:
            return "No correlation data available"
            
        corr_stats = self.analysis_results['correlation_analysis']
        interpretation = corr_stats['interpretation']
        
        return f"Correlación Φ-PPL: {interpretation}"
        
    def evaluate_consciousness_evidence(self):
        """Evaluar evidencia de conciencia."""
        phi_behavior = self.interpret_phi_behavior()
        learning_connection = self.interpret_learning_connection()
        
        if "significativa" in learning_connection and "creciente" in phi_behavior:
            return "Evidencia positiva de características de conciencia funcionales"
        elif "No significativa" in learning_connection:
            return "Φ no correlaciona con performance - posible métrica desconectada"
        else:
            return "Evidencia mixta - requiere mayor investigación"

def main():
    """Ejecutar análisis completo de métricas IIT."""
    print("📊 ANÁLISIS CIENTÍFICO DE MÉTRICAS IIT")
    print("="*60)
    
    analyzer = IITMetricsAnalyzer()
    
    # Cargar datos
    if not analyzer.load_latest_training_data():
        return
        
    print("\n🔍 Iniciando análisis...")
    
    # Análisis de evolución de PHI
    analyzer.analyze_phi_evolution()
    
    # Análisis de correlación
    analyzer.analyze_phi_ppl_correlation()
    
    # Análisis de fases de aprendizaje
    analyzer.analyze_learning_phases()
    
    # Generar reporte científico
    analyzer.generate_scientific_report()
    
    print("\n✅ ANÁLISIS COMPLETO")
    print("📂 Resultados en: results/analysis/")

if __name__ == '__main__':
    main()