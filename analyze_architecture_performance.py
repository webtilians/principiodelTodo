#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analizador de Rendimiento de Arquitecturas
=========================================

Compara el rendimiento de diferentes arquitecturas INFINITO V5.2
analizando perplexity, convergencia, y eficiencia de parámetros.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

class ArchitecturePerformanceAnalyzer:
    def __init__(self):
        """Inicializa el analizador de rendimiento"""
        self.results = {}
        self.training_histories = {}
        
    def load_training_history(self, history_file):
        """Carga historial de entrenamiento desde archivo JSON"""
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extraer información del modelo
            model_info = {
                'architecture': data.get('config', {}).get('model_size', 'unknown'),
                'total_params': data.get('model_info', {}).get('total_parameters', 0),
                'd_model': data.get('config', {}).get('d_model', 0),
                'num_layers': data.get('config', {}).get('num_layers', 0),
                'num_heads': data.get('config', {}).get('num_heads', 0),
                'epochs': data.get('config', {}).get('epochs', 0),
                'batch_size': data.get('config', {}).get('batch_size', 0),
                'lr': data.get('config', {}).get('lr', 0)
            }
            
            # Extraer métricas de entrenamiento
            training_metrics = {
                'train_loss': data.get('training_history', {}).get('train_loss', []),
                'val_loss': data.get('training_history', {}).get('val_loss', []),
                'train_ppl': data.get('training_history', {}).get('train_ppl', []),
                'val_ppl': data.get('training_history', {}).get('val_ppl', []),
                'phi_values': data.get('training_history', {}).get('phi_values', [])
            }
            
            return model_info, training_metrics
            
        except Exception as e:
            print(f"[ERROR] No se pudo cargar {history_file}: {str(e)}")
            return None, None
    
    def analyze_convergence(self, train_loss, val_loss):
        """Analiza la convergencia del entrenamiento"""
        if not train_loss or not val_loss:
            return {
                'converged': False,
                'convergence_epoch': -1,
                'final_improvement': 0,
                'stability': 0
            }
        
        # Detectar convergencia
        convergence_threshold = 0.01  # 1% de mejora mínima
        window_size = 3
        
        converged = False
        convergence_epoch = -1
        
        for i in range(window_size, len(val_loss)):
            recent_window = val_loss[i-window_size:i]
            improvement = (max(recent_window) - min(recent_window)) / max(recent_window)
            
            if improvement < convergence_threshold:
                converged = True
                convergence_epoch = i
                break
        
        # Mejora final
        if len(val_loss) >= 2:
            final_improvement = (val_loss[0] - val_loss[-1]) / val_loss[0]
        else:
            final_improvement = 0
        
        # Estabilidad (varianza en últimas épocas)
        if len(val_loss) >= 3:
            last_epochs = val_loss[-3:]
            stability = 1 / (1 + np.var(last_epochs))
        else:
            stability = 0
        
        return {
            'converged': converged,
            'convergence_epoch': convergence_epoch,
            'final_improvement': final_improvement,
            'stability': stability
        }
    
    def calculate_efficiency_metrics(self, model_info, training_metrics):
        """Calcula métricas de eficiencia del modelo"""
        
        # Parámetros por millón
        params_millions = model_info['total_params'] / 1_000_000
        
        # Perplexity final
        final_ppl = training_metrics['val_ppl'][-1] if training_metrics['val_ppl'] else float('inf')
        
        # Eficiencia: 1/PPL per million parameters
        if final_ppl > 0 and params_millions > 0:
            efficiency = (1 / final_ppl) / params_millions
        else:
            efficiency = 0
        
        # Ratio de mejora por época
        if training_metrics['val_ppl'] and len(training_metrics['val_ppl']) >= 2:
            initial_ppl = training_metrics['val_ppl'][0]
            epochs = len(training_metrics['val_ppl'])
            improvement_per_epoch = (initial_ppl - final_ppl) / epochs
        else:
            improvement_per_epoch = 0
        
        # Velocidad de convergencia
        convergence_analysis = self.analyze_convergence(
            training_metrics['train_loss'],
            training_metrics['val_loss']
        )
        
        convergence_speed = 0
        if convergence_analysis['converged'] and convergence_analysis['convergence_epoch'] > 0:
            convergence_speed = 1 / convergence_analysis['convergence_epoch']
        
        return {
            'params_millions': params_millions,
            'final_ppl': final_ppl,
            'efficiency': efficiency,
            'improvement_per_epoch': improvement_per_epoch,
            'convergence_speed': convergence_speed,
            'convergence_analysis': convergence_analysis
        }
    
    def load_all_histories(self, directory='.'):
        """Carga todos los historiales de entrenamiento disponibles"""
        history_patterns = ['training_history*.json', '*history*.json', '*_results.json']
        
        found_files = []
        for pattern in history_patterns:
            found_files.extend(Path(directory).glob(pattern))
        
        print(f"[INFO] Encontrados {len(found_files)} archivos de historial")
        
        for file_path in found_files:
            print(f"[LOAD] Cargando {file_path}")
            model_info, training_metrics = self.load_training_history(file_path)
            
            if model_info and training_metrics:
                architecture = model_info['architecture']
                
                # Calcular métricas de eficiencia
                efficiency_metrics = self.calculate_efficiency_metrics(model_info, training_metrics)
                
                # Almacenar resultados
                self.results[f"{architecture}_{file_path.stem}"] = {
                    'model_info': model_info,
                    'training_metrics': training_metrics,
                    'efficiency_metrics': efficiency_metrics,
                    'file_path': str(file_path)
                }
                
                print(f"[OK] {architecture}: PPL {efficiency_metrics['final_ppl']:.2f}, {efficiency_metrics['params_millions']:.1f}M params")
    
    def create_comparison_dataframe(self):
        """Crea DataFrame para comparación de arquitecturas"""
        if not self.results:
            print("[WARNING] No hay resultados cargados")
            return None
        
        comparison_data = []
        
        for name, data in self.results.items():
            model_info = data['model_info']
            efficiency = data['efficiency_metrics']
            
            row = {
                'name': name,
                'architecture': model_info['architecture'],
                'total_params': model_info['total_params'],
                'params_millions': efficiency['params_millions'],
                'd_model': model_info['d_model'],
                'num_layers': model_info['num_layers'],
                'num_heads': model_info['num_heads'],
                'epochs_trained': model_info['epochs'],
                'batch_size': model_info['batch_size'],
                'learning_rate': model_info['lr'],
                'final_ppl': efficiency['final_ppl'],
                'efficiency': efficiency['efficiency'],
                'improvement_per_epoch': efficiency['improvement_per_epoch'],
                'convergence_speed': efficiency['convergence_speed'],
                'converged': efficiency['convergence_analysis']['converged'],
                'convergence_epoch': efficiency['convergence_analysis']['convergence_epoch'],
                'stability': efficiency['convergence_analysis']['stability']
            }
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Ordenar por eficiencia
        df = df.sort_values('efficiency', ascending=False)
        
        return df
    
    def generate_comparison_report(self, output_file=None):
        """Genera reporte de comparación detallado"""
        df = self.create_comparison_dataframe()
        
        if df is None or len(df) == 0:
            print("[ERROR] No hay datos para generar reporte")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_file is None:
            output_file = f"architecture_comparison_report_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE COMPARACION DE ARQUITECTURAS INFINITO V5.2\n")
            f.write("="*80 + "\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Arquitecturas analizadas: {len(df)}\n\n")
            
            # Ranking por eficiencia
            f.write("RANKING POR EFICIENCIA\n")
            f.write("-" * 50 + "\n")
            for i, (_, row) in enumerate(df.iterrows(), 1):
                f.write(f"{i}. {row['architecture']}\n")
                f.write(f"   Parámetros: {row['params_millions']:.1f}M\n")
                f.write(f"   PPL Final: {row['final_ppl']:.2f}\n")
                f.write(f"   Eficiencia: {row['efficiency']:.6f}\n")
                f.write(f"   Convergencia: {'Sí' if row['converged'] else 'No'}")
                if row['converged']:
                    f.write(f" (época {row['convergence_epoch']})")
                f.write("\n\n")
            
            # Mejores en cada categoría
            f.write("MEJORES EN CADA CATEGORIA\n")
            f.write("-" * 50 + "\n")
            
            best_ppl = df.loc[df['final_ppl'].idxmin()]
            best_efficiency = df.loc[df['efficiency'].idxmax()]
            fastest_convergence = df[df['converged']].loc[df[df['converged']]['convergence_speed'].idxmax()] if any(df['converged']) else None
            most_stable = df.loc[df['stability'].idxmax()]
            
            f.write(f"Mejor PPL: {best_ppl['architecture']} (PPL: {best_ppl['final_ppl']:.2f})\n")
            f.write(f"Más Eficiente: {best_efficiency['architecture']} (Eficiencia: {best_efficiency['efficiency']:.6f})\n")
            
            if fastest_convergence is not None:
                f.write(f"Convergencia Más Rápida: {fastest_convergence['architecture']} (época {fastest_convergence['convergence_epoch']})\n")
            else:
                f.write("Convergencia Más Rápida: No disponible\n")
            
            f.write(f"Más Estable: {most_stable['architecture']} (Estabilidad: {most_stable['stability']:.4f})\n\n")
            
            # Estadísticas detalladas
            f.write("ESTADISTICAS DETALLADAS\n")
            f.write("-" * 50 + "\n")
            f.write(df.to_string(index=False))
            
            # Recomendaciones
            f.write("\n\nRECOMENDACIONES\n")
            f.write("-" * 50 + "\n")
            
            if len(df) > 0:
                top_arch = df.iloc[0]
                f.write(f"Arquitectura Recomendada: {top_arch['architecture']}\n")
                f.write(f"Razón: Mejor balance entre eficiencia y rendimiento\n")
                f.write(f"Parámetros: {top_arch['params_millions']:.1f}M\n")
                f.write(f"PPL: {top_arch['final_ppl']:.2f}\n")
                f.write(f"Eficiencia: {top_arch['efficiency']:.6f}\n")
        
        print(f"[SAVE] Reporte guardado en: {output_file}")
        
        # Mostrar resumen en consola
        print("\n" + "="*60)
        print("RESUMEN DE COMPARACION")
        print("="*60)
        print(f"Arquitecturas analizadas: {len(df)}")
        print(f"Mejor eficiencia: {df.iloc[0]['architecture']} ({df.iloc[0]['efficiency']:.6f})")
        print(f"Mejor PPL: {best_ppl['architecture']} ({best_ppl['final_ppl']:.2f})")
        print(f"Reporte completo en: {output_file}")
        
        return output_file
    
    def create_visualization(self, output_dir='analysis_plots'):
        """Crea visualizaciones de comparación"""
        df = self.create_comparison_dataframe()
        
        if df is None or len(df) == 0:
            print("[ERROR] No hay datos para visualizar")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar estilo
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Eficiencia vs Parámetros
        plt.figure()
        plt.scatter(df['params_millions'], df['efficiency'], 
                   s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
        
        for i, row in df.iterrows():
            plt.annotate(row['architecture'], 
                        (row['params_millions'], row['efficiency']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        plt.xlabel('Parámetros (Millones)')
        plt.ylabel('Eficiencia (1/PPL por Millón de Parámetros)')
        plt.title('Eficiencia vs Número de Parámetros')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/efficiency_vs_params.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. PPL vs Parámetros
        plt.figure()
        plt.scatter(df['params_millions'], df['final_ppl'], 
                   s=100, alpha=0.7, c=range(len(df)), cmap='plasma')
        
        for i, row in df.iterrows():
            plt.annotate(row['architecture'], 
                        (row['params_millions'], row['final_ppl']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8)
        
        plt.xlabel('Parámetros (Millones)')
        plt.ylabel('Perplexity Final')
        plt.title('Perplexity vs Número de Parámetros')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig(f'{output_dir}/ppl_vs_params.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Gráfico de barras de eficiencia
        plt.figure()
        colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
        bars = plt.bar(range(len(df)), df['efficiency'], color=colors)
        plt.xlabel('Arquitectura')
        plt.ylabel('Eficiencia')
        plt.title('Comparación de Eficiencia por Arquitectura')
        plt.xticks(range(len(df)), df['architecture'], rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en las barras
        for i, (bar, eff) in enumerate(zip(bars, df['efficiency'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['efficiency'])*0.01,
                    f'{eff:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVE] Visualizaciones guardadas en: {output_dir}/")

def main():
    """Función principal"""
    print("[START] Iniciando análisis de rendimiento de arquitecturas...")
    
    # Crear analizador
    analyzer = ArchitecturePerformanceAnalyzer()
    
    # Cargar todos los historiales disponibles
    analyzer.load_all_histories()
    
    if not analyzer.results:
        print("[WARNING] No se encontraron archivos de historial de entrenamiento")
        print("[INFO] Ejecuta algunos entrenamientos primero para generar datos")
        return
    
    # Generar reporte de comparación
    report_file = analyzer.generate_comparison_report()
    
    # Crear visualizaciones
    try:
        analyzer.create_visualization()
    except ImportError:
        print("[WARNING] matplotlib no disponible, omitiendo visualizaciones")
    except Exception as e:
        print(f"[WARNING] Error creando visualizaciones: {str(e)}")
    
    print("\n[SUCCESS] Análisis de rendimiento completado!")

if __name__ == "__main__":
    main()