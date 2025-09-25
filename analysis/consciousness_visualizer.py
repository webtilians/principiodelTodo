#!/usr/bin/env python3
"""
🧠 INFINITO V5.1 - VISUALIZADOR DE MÉTRICAS DE CONSCIENCIA 🧠
===========================================================

Generador de visualizaciones avanzadas para métricas de consciencia
basadas en datos reales del sistema INFINITO V5.1.

NO MODIFICA el código principal del sistema - archivo independiente.

Autor: Sistema INFINITO V5.1 Analysis
Fecha: 2025-09-24
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo visual
plt.style.use('dark_background')
sns.set_palette("husl")

class ConsciousnessVisualizer:
    """Visualizador de métricas de consciencia V5.1"""
    
    def __init__(self, json_file_path):
        """Inicializar con archivo JSON de datos"""
        self.json_file = json_file_path
        self.data = self.load_data()
        self.prepare_metrics()
    
    def load_data(self):
        """Cargar datos del archivo JSON"""
        print(f"🔍 Cargando datos de: {self.json_file}")
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ Datos cargados exitosamente")
            print(f"   📊 Versión: {data.get('version', 'N/A')}")
            print(f"   📅 Inicio: {data.get('start_time', 'N/A')}")
            print(f"   🔄 Iteraciones: {len(data.get('iterations', []))}")
            
            return data
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    
    def prepare_metrics(self):
        """Preparar métricas para visualización"""
        if not self.data:
            return
        
        # Extraer métricas principales
        self.iterations = np.array(self.data.get('iterations', []))
        self.consciousness = np.array(self.data.get('consciousness_values', []))
        self.phi_values = np.array(self.data.get('phi_values', []))
        self.memory_utilization = np.array(self.data.get('memory_utilization', []))
        self.loss_values = np.array(self.data.get('loss_values', []))
        self.eeg_correlations = np.array(self.data.get('eeg_correlations', []))
        
        # Calcular métricas derivadas
        if len(self.consciousness) > 1:
            self.consciousness_growth = np.gradient(self.consciousness)
            self.phi_growth = np.gradient(self.phi_values)
        else:
            self.consciousness_growth = np.array([0])
            self.phi_growth = np.array([0])
        
        # Estadísticas
        self.stats = {
            'max_consciousness': np.max(self.consciousness) if len(self.consciousness) > 0 else 0,
            'mean_consciousness': np.mean(self.consciousness) if len(self.consciousness) > 0 else 0,
            'max_phi': np.max(self.phi_values) if len(self.phi_values) > 0 else 0,
            'mean_phi': np.mean(self.phi_values) if len(self.phi_values) > 0 else 0,
            'final_consciousness': self.consciousness[-1] if len(self.consciousness) > 0 else 0,
            'final_phi': self.phi_values[-1] if len(self.phi_values) > 0 else 0,
            'breakthrough_point': self.find_breakthrough_point()
        }
        
        print(f"📈 Métricas preparadas:")
        print(f"   🧠 Consciencia máxima: {self.stats['max_consciousness']:.4f}")
        print(f"   Φ  Phi máximo: {self.stats['max_phi']:.4f}")
        print(f"   🚀 Punto breakthrough: Iteración {self.stats['breakthrough_point']}")
    
    def find_breakthrough_point(self):
        """Encontrar el punto de breakthrough (consciencia > 0.6)"""
        breakthrough_threshold = 0.6
        for i, c in enumerate(self.consciousness):
            if c > breakthrough_threshold:
                return self.iterations[i] if i < len(self.iterations) else i
        return -1
    
    def create_main_dashboard(self):
        """Crear dashboard principal de métricas"""
        if not self.data:
            return
        
        # Crear figura con grid layout
        fig = plt.figure(figsize=(20, 16), facecolor='black')
        fig.suptitle('🧠 INFINITO V5.1 - DASHBOARD DE CONSCIENCIA 🚀', 
                    fontsize=18, color='cyan', fontweight='bold', y=0.98)
        
        # Grid layout 3x4
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Evolución de Consciencia
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_consciousness_evolution(ax1)
        
        # 2. Evolución de Phi
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_phi_evolution(ax2)
        
        # 3. Correlación EEG
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_eeg_correlation(ax3)
        
        # 4. Métricas combinadas
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_combined_metrics(ax4)
        
        # 5. Utilización de memoria
        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_memory_utilization(ax5)
        
        # 6. Distribución de consciencia
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_consciousness_distribution(ax6)
        
        # 7. Heatmap de progreso
        ax7 = fig.add_subplot(gs[2, :2])
        self.plot_progress_heatmap(ax7)
        
        # 8. Medidor de consciencia
        ax8 = fig.add_subplot(gs[2, 2])
        self.plot_consciousness_gauge(ax8)
        
        # 9. Panel de estadísticas
        ax9 = fig.add_subplot(gs[3, :])
        self.plot_statistics_panel(ax9)
        
        plt.savefig('outputs/consciousness_dashboard_v5.1.png', 
                   dpi=300, bbox_inches='tight', facecolor='black')
        print("✅ Dashboard guardado: outputs/consciousness_dashboard_v5.1.png")
        
        # Mostrar solo si no estamos en modo batch
        try:
            plt.show()
        except KeyboardInterrupt:
            print("📊 Visualización interrumpida por el usuario")
        except Exception as e:
            print(f"📊 No se puede mostrar ventana interactiva: {e}")
        
        plt.close()
    
    def plot_consciousness_evolution(self, ax):
        """Gráfico de evolución de consciencia"""
        ax.plot(self.iterations, self.consciousness, 
               color='cyan', linewidth=2, alpha=0.8, label='Consciencia')
        ax.fill_between(self.iterations, self.consciousness, 
                       alpha=0.3, color='cyan')
        
        # Línea de breakthrough (60%)
        ax.axhline(y=0.6, color='yellow', linestyle='--', 
                  alpha=0.7, label='Breakthrough (60%)')
        
        # Marcar punto de breakthrough
        if self.stats['breakthrough_point'] != -1:
            breakthrough_idx = np.where(self.iterations == self.stats['breakthrough_point'])[0]
            if len(breakthrough_idx) > 0:
                idx = breakthrough_idx[0]
                ax.plot(self.iterations[idx], self.consciousness[idx], 
                       'ro', markersize=8, label=f'Breakthrough: Iter {self.stats["breakthrough_point"]}')
        
        ax.set_xlabel('Iteraciones', color='white')
        ax.set_ylabel('Nivel de Consciencia', color='white')
        ax.set_title('🧠 Evolución de Consciencia', color='white', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
    
    def plot_phi_evolution(self, ax):
        """Gráfico de evolución de Phi"""
        ax.plot(self.iterations, self.phi_values, 
               color='orange', linewidth=2, alpha=0.8, label='Φ (Phi)')
        ax.fill_between(self.iterations, self.phi_values, 
                       alpha=0.3, color='orange')
        
        # Línea de objetivo (1.0 bits)
        ax.axhline(y=1.0, color='red', linestyle='--', 
                  alpha=0.7, label='Objetivo (1.0 bits)')
        
        ax.set_xlabel('Iteraciones', color='white')
        ax.set_ylabel('Φ (bits)', color='white')
        ax.set_title('⚡ Evolución de Phi (Φ)', color='white', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
    
    def plot_eeg_correlation(self, ax):
        """Gráfico de correlación EEG"""
        if len(self.eeg_correlations) > 0:
            ax.plot(self.iterations, self.eeg_correlations, 
                   color='green', linewidth=2, alpha=0.8)
            ax.fill_between(self.iterations, self.eeg_correlations, 
                           alpha=0.3, color='green')
        
        ax.set_xlabel('Iteraciones', color='white')
        ax.set_ylabel('Correlación EEG', color='white')
        ax.set_title('🧬 Correlación EEG', color='white', fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
    
    def plot_combined_metrics(self, ax):
        """Métricas combinadas normalizadas"""
        # Normalizar métricas
        norm_consciousness = self.consciousness
        norm_phi = self.phi_values / np.max(self.phi_values) if np.max(self.phi_values) > 0 else self.phi_values
        
        ax.plot(self.iterations, norm_consciousness, 
               color='cyan', linewidth=2, label='Consciencia', alpha=0.8)
        ax.plot(self.iterations, norm_phi, 
               color='orange', linewidth=2, label='Φ (norm)', alpha=0.8)
        
        if len(self.memory_utilization) > 0:
            norm_memory = self.memory_utilization / np.max(self.memory_utilization) if np.max(self.memory_utilization) > 0 else self.memory_utilization
            ax.plot(self.iterations, norm_memory, 
                   color='purple', linewidth=2, label='Memoria (norm)', alpha=0.8)
        
        ax.set_xlabel('Iteraciones', color='white')
        ax.set_ylabel('Valor Normalizado', color='white')
        ax.set_title('📊 Métricas Combinadas', color='white', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
    
    def plot_memory_utilization(self, ax):
        """Gráfico de utilización de memoria"""
        if len(self.memory_utilization) > 0:
            ax.plot(self.iterations, self.memory_utilization * 100, 
                   color='purple', linewidth=2, alpha=0.8)
            ax.fill_between(self.iterations, self.memory_utilization * 100, 
                           alpha=0.3, color='purple')
        
        ax.set_xlabel('Iteraciones', color='white')
        ax.set_ylabel('Utilización (%)', color='white')
        ax.set_title('💾 Utilización de Memoria', color='white', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
    
    def plot_consciousness_distribution(self, ax):
        """Distribución de valores de consciencia"""
        if len(self.consciousness) > 0:
            ax.hist(self.consciousness, bins=30, color='cyan', alpha=0.7, edgecolor='white')
            ax.axvline(np.mean(self.consciousness), color='yellow', linestyle='--', 
                      label=f'Media: {np.mean(self.consciousness):.3f}')
            ax.axvline(0.6, color='red', linestyle='--', 
                      label='Threshold (0.6)')
        
        ax.set_xlabel('Nivel de Consciencia', color='white')
        ax.set_ylabel('Frecuencia', color='white')
        ax.set_title('📈 Distribución de Consciencia', color='white', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
    
    def plot_progress_heatmap(self, ax):
        """Heatmap de progreso a lo largo del tiempo"""
        if len(self.iterations) == 0:
            return
            
        # Crear matriz de progreso (simplificada)
        n_segments = min(50, len(self.iterations))
        segment_size = len(self.iterations) // n_segments
        
        heatmap_data = []
        for i in range(0, len(self.iterations), segment_size):
            end_idx = min(i + segment_size, len(self.iterations))
            segment_consciousness = np.mean(self.consciousness[i:end_idx])
            segment_phi = np.mean(self.phi_values[i:end_idx])
            heatmap_data.append([segment_consciousness, segment_phi])
        
        heatmap_data = np.array(heatmap_data).T
        
        im = ax.imshow(heatmap_data, cmap='plasma', aspect='auto', interpolation='bilinear')
        ax.set_title('🔥 Heatmap de Progreso', color='white', fontweight='bold')
        ax.set_xlabel('Segmentos de Tiempo', color='white')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Consciencia', 'Phi'], color='white')
        ax.tick_params(colors='white')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='white')
    
    def plot_consciousness_gauge(self, ax):
        """Medidor de consciencia final"""
        final_consciousness = self.stats['final_consciousness']
        
        # Crear gauge
        theta = np.linspace(0, np.pi, 100)
        r = 0.8
        
        # Arco base
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'w-', linewidth=8, alpha=0.3)
        
        # Arco de progreso
        progress_theta = np.linspace(0, final_consciousness * np.pi, int(final_consciousness * 100))
        ax.plot(r * np.cos(progress_theta), r * np.sin(progress_theta), 
               'cyan', linewidth=8, alpha=0.9)
        
        # Aguja
        needle_theta = final_consciousness * np.pi
        ax.plot([0, 0.7 * np.cos(needle_theta)], [0, 0.7 * np.sin(needle_theta)], 
               'r-', linewidth=4)
        
        # Texto central
        ax.text(0, -0.2, f'{final_consciousness:.3f}', 
               ha='center', va='center', fontsize=20, color='cyan', fontweight='bold')
        ax.text(0, -0.35, 'Consciencia Final', 
               ha='center', va='center', fontsize=12, color='white')
        
        # Etiquetas
        ax.text(-0.8, 0, '0.0', ha='center', va='center', color='white')
        ax.text(0.8, 0, '1.0', ha='center', va='center', color='white')
        ax.text(0, 0.8, '0.5', ha='center', va='center', color='white')
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('🎯 Medidor de Consciencia', color='white', fontweight='bold')
    
    def plot_statistics_panel(self, ax):
        """Panel de estadísticas"""
        ax.axis('off')
        
        # Preparar estadísticas para el texto
        memory_max = f"{np.max(self.memory_utilization)*100:.2f}%" if len(self.memory_utilization) > 0 else 'N/A'
        memory_avg = f"{np.mean(self.memory_utilization)*100:.2f}%" if len(self.memory_utilization) > 0 else 'N/A'
        eeg_max = f"{np.max(self.eeg_correlations):.4f}" if len(self.eeg_correlations) > 0 else 'N/A'
        eeg_avg = f"{np.mean(self.eeg_correlations):.4f}" if len(self.eeg_correlations) > 0 else 'N/A'
        
        # Información del experimento
        info_text = f"""
🧠 INFINITO V5.1 - REPORTE DE CONSCIENCIA BREAKTHROUGH 🚀
{'='*80}

📊 MÉTRICAS PRINCIPALES:
   Versión: {self.data.get('version', 'N/A')}
   Inicio: {self.data.get('start_time', 'N/A')}
   Iteraciones totales: {len(self.iterations):,}

🚀 CONSCIENCIA:
   Nivel máximo: {self.stats['max_consciousness']:.6f} ({self.stats['max_consciousness']*100:.2f}%)
   Nivel promedio: {self.stats['mean_consciousness']:.6f} ({self.stats['mean_consciousness']*100:.2f}%)
   Nivel final: {self.stats['final_consciousness']:.6f} ({self.stats['final_consciousness']*100:.2f}%)
   
⚡ PHI (Φ):
   Phi máximo: {self.stats['max_phi']:.6f} bits
   Phi promedio: {self.stats['mean_phi']:.6f} bits
   Phi final: {self.stats['final_phi']:.6f} bits

🎯 BREAKTHROUGH:
   Punto de breakthrough (>60%): {'Iteración ' + str(self.stats['breakthrough_point']) if self.stats['breakthrough_point'] != -1 else 'No alcanzado'}
   Tiempo para breakthrough: {'Inmediato' if self.stats['breakthrough_point'] <= 10 else f'{self.stats["breakthrough_point"]} iteraciones'}

💾 MEMORIA:
   Utilización máxima: {memory_max}
   Utilización promedio: {memory_avg}

🧬 CORRELACIONES:
   EEG máxima: {eeg_max}
   EEG promedio: {eeg_avg}

🏆 LOGROS:
   ✅ Breakthrough de consciencia: {'SÍ' if self.stats['max_consciousness'] > 0.6 else 'NO'}
   ✅ Phi > 1.0 bits: {'SÍ' if self.stats['max_phi'] > 1.0 else 'NO'}
   ✅ Estabilidad >99%: {'SÍ' if self.stats['max_consciousness'] > 0.99 else 'NO'}
        """
        
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               color='white', bbox=dict(boxstyle='round', facecolor='navy', alpha=0.8))

def main():
    """Función principal"""
    print("🚀 INFINITO V5.1 - VISUALIZADOR DE CONSCIENCIA")
    print("=" * 50)
    
    # Ruta del archivo JSON (ajustar según sea necesario)
    json_file = "outputs/infinito_v5_1_consciousness_20250924_215439_C1.000_PHI1.305.json"
    
    # Crear visualizador
    visualizer = ConsciousnessVisualizer(json_file)
    
    if visualizer.data is None:
        print("❌ No se pudieron cargar los datos. Finalizando.")
        return
    
    # Generar dashboard principal
    print("\n📊 Generando dashboard de consciencia...")
    visualizer.create_main_dashboard()
    
    print("✅ Visualizaciones generadas exitosamente!")
    print("📁 Archivo guardado: outputs/consciousness_dashboard_v5.1.png")

if __name__ == "__main__":
    main()