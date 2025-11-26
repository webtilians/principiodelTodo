#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparador de Modelos INFINITO V5.2
===================================

Herramienta para comparar múltiples modelos entrenados lado a lado,
analizando rendimiento, calidad de generación y eficiencia.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

class ModelComparator:
    """
    Comparador completo de modelos INFINITO V5.2
    """
    
    def __init__(self, models_directory: str = 'models/checkpoints'):
        """
        Inicializa el comparador
        
        Args:
            models_directory: Directorio donde están los modelos (.pt)
        """
        self.models_directory = models_directory
        self.model_data = {}
        self.comparison_results = {}
        
    def discover_models(self) -> List[str]:
        """Descubre automáticamente todos los modelos disponibles"""
        model_files = []
        
        if os.path.exists(self.models_directory):
            for file in Path(self.models_directory).glob('*.pt'):
                model_files.append(str(file))
        
        # También buscar en directorio raíz
        for file in Path('.').glob('*.pt'):
            model_files.append(str(file))
        
        print(f"[DISCOVER] Encontrados {len(model_files)} modelos:")
        for model in model_files:
            print(f"  - {os.path.basename(model)}")
        
        return model_files
    
    def extract_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Extrae información básica de un modelo sin cargarlo completamente
        
        Args:
            model_path: Path al archivo .pt
            
        Returns:
            Diccionario con información del modelo
        """
        try:
            import torch
            
            # Cargar solo metadatos
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Extraer información básica
            config = checkpoint.get('config', {})
            history = checkpoint.get('history', {})
            
            # Información del modelo
            model_info = {
                'path': model_path,
                'name': os.path.basename(model_path),
                'epoch': checkpoint.get('epoch', 'unknown'),
                'val_loss': checkpoint.get('val_loss', float('inf')),
                'val_ppl': checkpoint.get('val_ppl', float('inf')),
                'vocab_size': config.get('vocab_size', 'unknown'),
                'hidden_dim': config.get('hidden_dim', 'unknown'),
                'num_layers': config.get('num_layers', 'unknown'),
                'num_heads': config.get('num_heads', 'unknown'),
                'use_improved_memory': config.get('use_improved_memory', False),
                'use_improved_iit': config.get('use_improved_iit', False),
                'use_learnable_phi': config.get('use_learnable_phi', False),
                'seed': config.get('seed', 'unknown')
            }
            
            # Información de entrenamiento
            if history:
                train_losses = history.get('train_loss', [])
                val_losses = history.get('val_loss', [])
                val_ppls = history.get('val_perplexity', [])
                
                if train_losses and val_losses:
                    model_info.update({
                        'final_train_loss': train_losses[-1] if train_losses else float('inf'),
                        'final_val_loss': val_losses[-1] if val_losses else float('inf'),
                        'final_val_ppl': val_ppls[-1] if val_ppls else float('inf'),
                        'best_val_loss': min(val_losses) if val_losses else float('inf'),
                        'best_val_ppl': min(val_ppls) if val_ppls else float('inf'),
                        'epochs_trained': len(val_losses),
                        'improvement': (val_losses[0] - val_losses[-1]) / val_losses[0] if len(val_losses) >= 2 else 0
                    })
            
            # Calcular número aproximado de parámetros
            if 'model_state_dict' in checkpoint:
                try:
                    total_params = 0
                    for param_tensor in checkpoint['model_state_dict'].values():
                        if hasattr(param_tensor, 'numel'):
                            total_params += param_tensor.numel()
                    model_info['total_parameters'] = total_params
                except:
                    model_info['total_parameters'] = 'unknown'
            
            return model_info
            
        except Exception as e:
            print(f"[ERROR] No se pudo cargar {model_path}: {str(e)}")
            return {
                'path': model_path,
                'name': os.path.basename(model_path),
                'error': str(e)
            }
    
    def load_evaluation_results(self, evaluation_dir: str = 'evaluation_results') -> Dict[str, Any]:
        """
        Carga resultados de evaluaciones previas de calidad de generación
        
        Args:
            evaluation_dir: Directorio con resultados de evaluación
            
        Returns:
            Diccionario con resultados por modelo
        """
        evaluation_data = {}
        
        if not os.path.exists(evaluation_dir):
            print(f"[WARNING] Directorio de evaluación no existe: {evaluation_dir}")
            return evaluation_data
        
        # Buscar archivos de evaluación
        for eval_file in Path(evaluation_dir).glob('evaluation_*.json'):
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extraer nombre del modelo desde el path
                model_path = data.get('model_info', {}).get('model_path', '')
                model_name = os.path.basename(model_path) if model_path else eval_file.stem
                
                evaluation_data[model_name] = data
                print(f"[LOAD] Evaluación cargada para: {model_name}")
                
            except Exception as e:
                print(f"[ERROR] No se pudo cargar evaluación {eval_file}: {str(e)}")
        
        return evaluation_data
    
    def create_comparison_table(self, models_info: List[Dict], evaluation_data: Dict = None) -> pd.DataFrame:
        """
        Crea tabla comparativa de modelos
        
        Args:
            models_info: Lista de información de modelos
            evaluation_data: Datos de evaluación de calidad (opcional)
            
        Returns:
            DataFrame con comparación
        """
        comparison_data = []
        
        for model_info in models_info:
            if 'error' in model_info:
                continue
            
            row = {
                'Modelo': model_info['name'],
                'Época': model_info.get('epoch', 'N/A'),
                'PPL Final': model_info.get('final_val_ppl', model_info.get('val_ppl', 'N/A')),
                'Pérdida Final': model_info.get('final_val_loss', model_info.get('val_loss', 'N/A')),
                'Parámetros': model_info.get('total_parameters', 'N/A'),
                'Hidden Dim': model_info.get('hidden_dim', 'N/A'),
                'Capas': model_info.get('num_layers', 'N/A'),
                'Cabezas': model_info.get('num_heads', 'N/A'),
                'IIT Memory': model_info.get('use_improved_memory', False),
                'IIT Metrics': model_info.get('use_improved_iit', False),
                'Learnable Phi': model_info.get('use_learnable_phi', False),
                'Mejora %': model_info.get('improvement', 0) * 100 if model_info.get('improvement') else 'N/A'
            }
            
            # Agregar datos de evaluación si están disponibles
            if evaluation_data and model_info['name'] in evaluation_data:
                eval_data = evaluation_data[model_info['name']]
                overall_score = eval_data.get('overall_score', {})
                
                row.update({
                    'Puntuación General': overall_score.get('overall', 'N/A'),
                    'Diversidad': overall_score.get('diversity', 'N/A'),
                    'Anti-Repetición': overall_score.get('repetition', 'N/A'),
                    'Coherencia': overall_score.get('coherence', 'N/A')
                })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Ordenar por PPL (menor es mejor)
        if 'PPL Final' in df.columns:
            df_sorted = df.copy()
            # Convertir PPL a numérico para ordenar correctamente
            df_sorted['PPL_numeric'] = pd.to_numeric(df_sorted['PPL Final'], errors='coerce')
            df_sorted = df_sorted.sort_values('PPL_numeric', ascending=True)
            df_sorted = df_sorted.drop('PPL_numeric', axis=1)
            return df_sorted
        
        return df
    
    def generate_comparison_report(
        self,
        models_info: List[Dict],
        evaluation_data: Dict = None,
        output_file: str = None
    ) -> str:
        """
        Genera reporte completo de comparación
        
        Args:
            models_info: Información de modelos
            evaluation_data: Datos de evaluación (opcional)
            output_file: Archivo de salida (opcional)
            
        Returns:
            Path del archivo generado
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"model_comparison_report_{timestamp}.txt"
        
        # Crear DataFrame
        df = self.create_comparison_table(models_info, evaluation_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REPORTE DE COMPARACION DE MODELOS INFINITO V5.2\n")
            f.write("="*80 + "\n")
            f.write(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Modelos analizados: {len(df)}\n\n")
            
            # Tabla completa
            f.write("TABLA COMPARATIVA COMPLETA\n")
            f.write("-" * 50 + "\n")
            f.write(df.to_string(index=False))
            f.write("\n\n")
            
            # Top 3 modelos por PPL
            if 'PPL Final' in df.columns:
                valid_ppl = df[pd.to_numeric(df['PPL Final'], errors='coerce').notna()]
                if len(valid_ppl) > 0:
                    f.write("TOP 3 MODELOS POR PERPLEXITY (Menor es mejor)\n")
                    f.write("-" * 50 + "\n")
                    top_3_ppl = valid_ppl.head(3)
                    for i, (_, row) in enumerate(top_3_ppl.iterrows(), 1):
                        f.write(f"{i}. {row['Modelo']}\n")
                        f.write(f"   PPL: {row['PPL Final']}\n")
                        f.write(f"   Parámetros: {row['Parámetros']}\n")
                        f.write(f"   Arquitectura: {row['Hidden Dim']}d, {row['Capas']} capas, {row['Cabezas']} cabezas\n\n")
            
            # Top modelos por características IIT
            iit_models = df[df['IIT Memory'] & df['IIT Metrics'] & df['Learnable Phi']]
            if len(iit_models) > 0:
                f.write("MODELOS CON CARACTERÍSTICAS IIT COMPLETAS\n")
                f.write("-" * 50 + "\n")
                for _, row in iit_models.iterrows():
                    f.write(f"- {row['Modelo']}: PPL {row['PPL Final']}\n")
                f.write("\n")
            
            # Análisis de eficiencia (parámetros vs rendimiento)
            valid_params = df[pd.to_numeric(df['Parámetros'], errors='coerce').notna()]
            valid_ppl_params = valid_params[pd.to_numeric(valid_params['PPL Final'], errors='coerce').notna()]
            
            if len(valid_ppl_params) > 0:
                f.write("ANÁLISIS DE EFICIENCIA (Parámetros vs PPL)\n")
                f.write("-" * 50 + "\n")
                
                # Calcular eficiencia (1/PPL por millón de parámetros)
                efficiency_data = []
                for _, row in valid_ppl_params.iterrows():
                    params = pd.to_numeric(row['Parámetros'])
                    ppl = pd.to_numeric(row['PPL Final'])
                    if params > 0 and ppl > 0:
                        efficiency = (1/ppl) / (params/1_000_000)
                        efficiency_data.append((row['Modelo'], efficiency, params, ppl))
                
                # Ordenar por eficiencia
                efficiency_data.sort(key=lambda x: x[1], reverse=True)
                
                f.write("Ranking de eficiencia (mayor es mejor):\n")
                for i, (model, eff, params, ppl) in enumerate(efficiency_data[:5], 1):
                    f.write(f"{i}. {model}\n")
                    f.write(f"   Eficiencia: {eff:.6f}\n")
                    f.write(f"   Parámetros: {params:,.0f}\n")
                    f.write(f"   PPL: {ppl:.2f}\n\n")
            
            # Recomendaciones
            f.write("RECOMENDACIONES\n")
            f.write("-" * 50 + "\n")
            
            if len(df) > 0:
                best_overall = df.iloc[0]
                f.write(f"Mejor modelo general: {best_overall['Modelo']}\n")
                f.write(f"Razón: Mejor perplexity ({best_overall['PPL Final']})\n\n")
                
                # Recomendar por uso
                f.write("Recomendaciones por caso de uso:\n")
                f.write("- Producción: Modelo con mejor PPL y características IIT completas\n")
                f.write("- Experimentación: Modelos más pequeños para iteración rápida\n")
                f.write("- Investigación: Modelos con todas las características IIT activadas\n")
        
        print(f"[SAVE] Reporte de comparación guardado en: {output_file}")
        return output_file
    
    def create_visualizations(self, models_info: List[Dict], output_dir: str = 'comparison_plots'):
        """
        Crea visualizaciones comparativas
        
        Args:
            models_info: Información de modelos
            output_dir: Directorio para guardar gráficos
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear DataFrame
        df = self.create_comparison_table(models_info)
        
        # Configurar estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. PPL vs Parámetros
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Filtrar datos válidos
            valid_data = df[pd.to_numeric(df['PPL Final'], errors='coerce').notna() & 
                           pd.to_numeric(df['Parámetros'], errors='coerce').notna()]
            
            if len(valid_data) > 0:
                x = pd.to_numeric(valid_data['Parámetros']) / 1_000_000  # En millones
                y = pd.to_numeric(valid_data['PPL Final'])
                
                scatter = ax.scatter(x, y, s=100, alpha=0.7, c=range(len(valid_data)), cmap='viridis')
                
                # Añadir etiquetas
                for i, (_, row) in enumerate(valid_data.iterrows()):
                    ax.annotate(
                        row['Modelo'].replace('infinito_v5.2_', '').replace('.pt', ''),
                        (x.iloc[i], y.iloc[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, alpha=0.8
                    )
                
                ax.set_xlabel('Parámetros (Millones)')
                ax.set_ylabel('Perplexity Final')
                ax.set_title('Perplexity vs Número de Parámetros')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/ppl_vs_parameters.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"[SAVE] Gráfico PPL vs Parámetros guardado")
        
        except Exception as e:
            print(f"[WARNING] Error creando gráfico PPL vs Parámetros: {str(e)}")
        
        # 2. Comparación de arquitecturas
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Hidden dimensions
            hidden_dims = pd.to_numeric(df['Hidden Dim'], errors='coerce').dropna()
            if len(hidden_dims) > 0:
                axes[0,0].hist(hidden_dims, bins=10, alpha=0.7, edgecolor='black')
                axes[0,0].set_xlabel('Hidden Dimension')
                axes[0,0].set_ylabel('Frecuencia')
                axes[0,0].set_title('Distribución de Dimensiones Ocultas')
                axes[0,0].grid(True, alpha=0.3)
            
            # Número de capas
            num_layers = pd.to_numeric(df['Capas'], errors='coerce').dropna()
            if len(num_layers) > 0:
                axes[0,1].hist(num_layers, bins=10, alpha=0.7, edgecolor='black')
                axes[0,1].set_xlabel('Número de Capas')
                axes[0,1].set_ylabel('Frecuencia')
                axes[0,1].set_title('Distribución de Capas')
                axes[0,1].grid(True, alpha=0.3)
            
            # Características IIT
            iit_features = ['IIT Memory', 'IIT Metrics', 'Learnable Phi']
            iit_counts = [df[feature].sum() for feature in iit_features]
            
            axes[1,0].bar(range(len(iit_features)), iit_counts, alpha=0.7)
            axes[1,0].set_xticks(range(len(iit_features)))
            axes[1,0].set_xticklabels(['Memory', 'Metrics', 'Learnable Φ'], rotation=45)
            axes[1,0].set_ylabel('Número de Modelos')
            axes[1,0].set_title('Características IIT por Modelo')
            axes[1,0].grid(True, alpha=0.3)
            
            # PPL distribution
            ppls = pd.to_numeric(df['PPL Final'], errors='coerce').dropna()
            if len(ppls) > 0:
                axes[1,1].hist(ppls, bins=10, alpha=0.7, edgecolor='black')
                axes[1,1].set_xlabel('Perplexity Final')
                axes[1,1].set_ylabel('Frecuencia')
                axes[1,1].set_title('Distribución de Perplexity')
                axes[1,1].grid(True, alpha=0.3)
                axes[1,1].set_xscale('log')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/architecture_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[SAVE] Gráfico comparación de arquitecturas guardado")
        
        except Exception as e:
            print(f"[WARNING] Error creando gráfico de arquitecturas: {str(e)}")
    
    def run_complete_comparison(
        self,
        include_evaluations: bool = True,
        create_plots: bool = True,
        output_dir: str = 'model_comparison_results'
    ) -> Dict[str, Any]:
        """
        Ejecuta comparación completa de todos los modelos disponibles
        
        Args:
            include_evaluations: Si incluir datos de evaluación de calidad
            create_plots: Si crear visualizaciones
            output_dir: Directorio para resultados
            
        Returns:
            Diccionario con resultados de comparación
        """
        print(f"[START] Iniciando comparación completa de modelos...")
        
        # Crear directorio de resultados
        os.makedirs(output_dir, exist_ok=True)
        
        # Descubrir modelos
        model_files = self.discover_models()
        
        if not model_files:
            print(f"[WARNING] No se encontraron modelos para comparar")
            return {}
        
        # Extraer información de cada modelo
        print(f"[EXTRACT] Extrayendo información de modelos...")
        models_info = []
        for model_path in model_files:
            print(f"[PROCESS] Procesando {os.path.basename(model_path)}...")
            info = self.extract_model_info(model_path)
            models_info.append(info)
        
        # Cargar evaluaciones si están disponibles
        evaluation_data = {}
        if include_evaluations:
            print(f"[LOAD] Cargando evaluaciones de calidad...")
            evaluation_data = self.load_evaluation_results()
        
        # Generar reporte
        report_file = self.generate_comparison_report(
            models_info,
            evaluation_data,
            os.path.join(output_dir, 'comparison_report.txt')
        )
        
        # Crear visualizaciones
        if create_plots:
            print(f"[PLOT] Creando visualizaciones...")
            try:
                self.create_visualizations(
                    models_info,
                    os.path.join(output_dir, 'plots')
                )
            except Exception as e:
                print(f"[WARNING] Error creando gráficos: {str(e)}")
        
        # Crear CSV para análisis adicional
        df = self.create_comparison_table(models_info, evaluation_data)
        csv_file = os.path.join(output_dir, 'models_comparison.csv')
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"[SAVE] Datos CSV guardados en: {csv_file}")
        
        # Resultados finales
        results = {
            'models_analyzed': len(models_info),
            'report_file': report_file,
            'csv_file': csv_file,
            'models_info': models_info,
            'evaluation_data': evaluation_data,
            'comparison_table': df.to_dict('records')
        }
        
        print(f"\n[SUCCESS] Comparación completada!")
        print(f"Modelos analizados: {len(models_info)}")
        print(f"Reporte: {report_file}")
        print(f"Datos CSV: {csv_file}")
        
        return results


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comparador de Modelos INFINITO V5.2')
    parser.add_argument('--models-dir', type=str, default='models/checkpoints',
                       help='Directorio de modelos')
    parser.add_argument('--output-dir', type=str, default='model_comparison_results',
                       help='Directorio de resultados')
    parser.add_argument('--no-evaluations', action='store_true',
                       help='No incluir evaluaciones de calidad')
    parser.add_argument('--no-plots', action='store_true',
                       help='No crear gráficos')
    
    args = parser.parse_args()
    
    # Crear comparador
    comparator = ModelComparator(args.models_dir)
    
    # Ejecutar comparación completa
    results = comparator.run_complete_comparison(
        include_evaluations=not args.no_evaluations,
        create_plots=not args.no_plots,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()