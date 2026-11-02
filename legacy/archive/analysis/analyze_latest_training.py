#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 ANÁLISIS DEL ENTRENAMIENTO RECIENTE: INFINITO V5.2
=====================================================

Análisis profundo del modelo infinito_v5.2_real_best.pt recién entrenado:
- Comparación de métricas con modelos anteriores
- Evaluación de las características IIT implementadas
- Análisis de la evolución del rendimiento
- Recomendaciones para optimizaciones futuras
"""

import sys
import os

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import glob
from pathlib import Path

def load_model_comparison_data():
    """Carga los datos de comparación de modelos más recientes."""
    try:
        comparison_file = "model_comparison_results_updated/models_comparison.csv"
        if os.path.exists(comparison_file):
            return pd.read_csv(comparison_file)
        else:
            print(f"⚠️  Archivo de comparación no encontrado: {comparison_file}")
            return None
    except Exception as e:
        print(f"❌ Error cargando comparación: {e}")
        return None

def load_evaluation_data():
    """Carga los datos de evaluación más recientes."""
    try:
        eval_dir = "evaluation_results_latest"
        if not os.path.exists(eval_dir):
            print(f"⚠️  Directorio de evaluación no encontrado: {eval_dir}")
            return None
        
        # Buscar el archivo de evaluación más reciente
        eval_files = glob.glob(f"{eval_dir}/evaluation_*.json")
        if not eval_files:
            print(f"⚠️  No se encontraron archivos de evaluación en {eval_dir}")
            return None
        
        latest_file = max(eval_files, key=os.path.getmtime)
        print(f"📊 Cargando evaluación: {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error cargando evaluación: {e}")
        return None

def analyze_model_performance():
    """Analiza el rendimiento del modelo recién entrenado."""
    print(f"\n{'='*70}")
    print(f"🔬 ANÁLISIS DEL MODELO RECIÉN ENTRENADO")
    print(f"{'='*70}")
    
    # 1. Cargar datos de comparación
    comparison_df = load_model_comparison_data()
    if comparison_df is not None:
        # Filtrar el modelo recién entrenado
        recent_model = comparison_df[comparison_df['Modelo'].str.contains('infinito_v5.2_real_best.pt', na=False)]
        
        if not recent_model.empty:
            print(f"\n📈 MÉTRICAS DEL MODELO RECIENTE:")
            print(f"{'='*50}")
            model_data = recent_model.iloc[0]
            
            print(f"  🏷️  Modelo: {model_data['Modelo']}")
            print(f"  📊 PPL Final: {model_data['PPL Final']:.2f}")
            print(f"  🔢 Parámetros: {model_data['Parámetros']:,}")
            print(f"  🧠 Hidden Dim: {model_data['Hidden Dim']}")
            print(f"  ⚙️  IIT Memory: {model_data['IIT Memory']}")
            print(f"  🎯 Época: {model_data['Época']}")
            
            # Calcular ranking
            sorted_by_ppl = comparison_df.sort_values('PPL Final')
            ranking = sorted_by_ppl.reset_index(drop=True).index[
                sorted_by_ppl['Modelo'] == 'infinito_v5.2_real_best.pt'
            ].tolist()
            
            if ranking:
                print(f"  🏆 Ranking por PPL: #{ranking[0] + 1} de {len(comparison_df)} modelos")
        else:
            print("⚠️  Modelo reciente no encontrado en comparación")
    
    # 2. Cargar datos de evaluación de generación
    eval_data = load_evaluation_data()
    if eval_data is not None:
        print(f"\n🎨 ANÁLISIS DE GENERACIÓN DE TEXTO:")
        print(f"{'='*50}")
        
        model_info = eval_data.get('model_info', {})
        global_metrics = eval_data.get('global_metrics', {})
        
        print(f"  🔧 Configuración:")
        config = model_info.get('model_config', {})
        print(f"    - Vocab Size: {config.get('vocab_size', 'N/A'):,}")
        print(f"    - Hidden Dim: {config.get('hidden_dim', 'N/A')}")
        print(f"    - IIT Memory: {config.get('use_improved_memory', 'N/A')}")
        print(f"    - Parámetros: {model_info.get('total_parameters', 'N/A'):,}")
        
        print(f"\n  📊 Métricas de Generación:")
        if 'diversity' in global_metrics:
            diversity = global_metrics['diversity']
            print(f"    - Diversidad TTR: {diversity.get('type_token_ratio', 0):.3f}")
            print(f"    - Tokens únicos: {diversity.get('unique_tokens', 0)}")
            print(f"    - Diversidad intra-texto: {diversity.get('intra_text_diversity', 0):.3f}")
            print(f"    - Diversidad inter-texto: {diversity.get('inter_text_diversity', 0):.3f}")
        
        if 'repetition' in global_metrics:
            repetition = global_metrics['repetition']
            print(f"    - Repetición 2-gram: {repetition.get('avg_repetition_2gram', 0):.3f}")
            print(f"    - Repetición 3-gram: {repetition.get('avg_repetition_3gram', 0):.3f}")
        
        if 'coherence' in global_metrics:
            coherence = global_metrics['coherence']
            print(f"    - Perplexity promedio: {coherence.get('avg_perplexity', 0):,.2f}")
            print(f"    - Puntuación consistencia: {coherence.get('consistency_score', 0):.3f}")
            print(f"    - Puntuación fluidez: {coherence.get('fluency_score', 0):.3f}")

def analyze_training_evolution():
    """Analiza la evolución del entrenamiento comparando con modelos anteriores."""
    print(f"\n{'='*70}")
    print(f"📈 EVOLUCIÓN DEL ENTRENAMIENTO")
    print(f"{'='*70}")
    
    comparison_df = load_model_comparison_data()
    if comparison_df is None:
        return
    
    # Filtrar modelos de la serie v5.2
    v52_models = comparison_df[comparison_df['Modelo'].str.contains('v5.2', na=False)]
    
    if v52_models.empty:
        print("⚠️  No se encontraron modelos v5.2 para comparar")
        return
    
    print(f"\n🔄 COMPARACIÓN MODELOS V5.2:")
    print(f"{'='*50}")
    
    # Ordenar por PPL
    v52_sorted = v52_models.sort_values('PPL Final')
    
    for idx, (_, row) in enumerate(v52_sorted.iterrows()):
        modelo = row['Modelo'].replace('infinito_', '').replace('.pt', '')
        print(f"  {idx+1}. {modelo}")
        print(f"     PPL: {row['PPL Final']:.2f} | Params: {row['Parámetros']:,} | Época: {row['Época']}")
    
    # Análisis de mejoras
    print(f"\n💡 OBSERVACIONES:")
    print(f"{'='*30}")
    
    best_model = v52_sorted.iloc[0]
    if 'real_best' in best_model['Modelo']:
        print(f"  ✅ El modelo recién entrenado ES el mejor de la serie V5.2")
        print(f"     - Mejor PPL: {best_model['PPL Final']:.2f}")
        print(f"     - Configuración: {best_model['Hidden Dim']}d, {best_model['Época']} épocas")
    else:
        print(f"  ⚠️  El modelo recién entrenado NO es el mejor de la serie")
        print(f"     - Mejor modelo: {best_model['Modelo']}")
        print(f"     - Su PPL: {best_model['PPL Final']:.2f}")
    
    # Análisis de eficiencia
    if 'Parámetros' in v52_sorted.columns and 'PPL Final' in v52_sorted.columns:
        v52_sorted['Eficiencia'] = 1 / (v52_sorted['PPL Final'] * v52_sorted['Parámetros'] / 1e6)
        best_efficiency = v52_sorted.loc[v52_sorted['Eficiencia'].idxmax()]
        
        print(f"\n⚡ MODELO MÁS EFICIENTE V5.2:")
        print(f"  🏆 {best_efficiency['Modelo'].replace('infinito_', '').replace('.pt', '')}")
        print(f"  📊 Eficiencia: {best_efficiency['Eficiencia']:.6f}")
        print(f"  📈 PPL: {best_efficiency['PPL Final']:.2f}")
        print(f"  🔢 Parámetros: {best_efficiency['Parámetros']:,}")

def generate_recommendations():
    """Genera recomendaciones para optimizaciones futuras."""
    print(f"\n{'='*70}")
    print(f"💡 RECOMENDACIONES PARA OPTIMIZACIÓN")
    print(f"{'='*70}")
    
    eval_data = load_evaluation_data()
    comparison_df = load_model_comparison_data()
    
    recommendations = []
    
    # Análisis basado en métricas de generación
    if eval_data:
        global_metrics = eval_data.get('global_metrics', {})
        
        # Revisar diversidad
        if 'diversity' in global_metrics:
            ttr = global_metrics['diversity'].get('type_token_ratio', 0)
            if ttr < 0.3:
                recommendations.append({
                    'category': '🎨 Diversidad',
                    'issue': f'TTR bajo ({ttr:.3f})',
                    'suggestion': 'Aumentar temperatura de sampling o usar nucleus sampling'
                })
        
        # Revisar repetición
        if 'repetition' in global_metrics:
            rep_2gram = global_metrics['repetition'].get('avg_repetition_2gram', 0)
            if rep_2gram > 0.08:
                recommendations.append({
                    'category': '🔄 Repetición',
                    'issue': f'Alta repetición 2-gram ({rep_2gram:.3f})',
                    'suggestion': 'Implementar penalty para repeticiones o ajustar dropout'
                })
        
        # Revisar coherencia
        if 'coherence' in global_metrics:
            perplexity = global_metrics['coherence'].get('avg_perplexity', 0)
            if perplexity > 1000:
                recommendations.append({
                    'category': '🧠 Coherencia',
                    'issue': f'Perplexity muy alta ({perplexity:,.0f})',
                    'suggestion': 'Entrenar más épocas o reducir learning rate'
                })
    
    # Análisis basado en comparación de modelos
    if comparison_df is not None:
        recent_model = comparison_df[comparison_df['Modelo'].str.contains('real_best', na=False)]
        if not recent_model.empty:
            model_data = recent_model.iloc[0]
            
            # Comparar con mejor modelo general
            best_overall = comparison_df.loc[comparison_df['PPL Final'].idxmin()]
            if model_data['PPL Final'] > best_overall['PPL Final'] * 1.1:
                recommendations.append({
                    'category': '📊 Rendimiento',
                    'issue': f'PPL subóptima vs mejor modelo ({model_data["PPL Final"]:.2f} vs {best_overall["PPL Final"]:.2f})',
                    'suggestion': f'Probar configuración del modelo {best_overall["Modelo"]}'
                })
    
    # Recomendaciones generales para IIT
    recommendations.extend([
        {
            'category': '🧠 IIT Features',
            'issue': 'Optimización de características IIT',
            'suggestion': 'Ajustar lambda_phi para mejor balance entre LM y IIT loss'
        },
        {
            'category': '⚙️  Arquitectura',
            'issue': 'Exploración de configuraciones',
            'suggestion': 'Probar arquitecturas optimizadas (ultra_efficient, balanced_performance)'
        },
        {
            'category': '📚 Datos',
            'issue': 'Expansión del dataset',
            'suggestion': 'Considerar WikiText-103 o datasets más grandes para mejor generalización'
        }
    ])
    
    # Mostrar recomendaciones
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['category']}")
        print(f"   ⚠️  Problema: {rec['issue']}")
        print(f"   💡 Sugerencia: {rec['suggestion']}")

def create_summary_report():
    """Crea un reporte resumen del análisis."""
    print(f"\n{'='*70}")
    print(f"📋 RESUMEN EJECUTIVO")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"📅 Generado: {timestamp}")
    
    # Estado del modelo
    comparison_df = load_model_comparison_data()
    eval_data = load_evaluation_data()
    
    if comparison_df is not None:
        total_models = len(comparison_df)
        recent_model = comparison_df[comparison_df['Modelo'].str.contains('real_best', na=False)]
        
        if not recent_model.empty:
            model_ppl = recent_model.iloc[0]['PPL Final']
            model_params = recent_model.iloc[0]['Parámetros']
            
            print(f"\n🎯 ESTADO ACTUAL:")
            print(f"  • Total de modelos analizados: {total_models}")
            print(f"  • PPL del modelo reciente: {model_ppl:.2f}")
            print(f"  • Parámetros del modelo reciente: {model_params:,}")
    
    if eval_data:
        global_metrics = eval_data.get('global_metrics', {})
        
        print(f"\n📊 CALIDAD DE GENERACIÓN:")
        if 'diversity' in global_metrics:
            ttr = global_metrics['diversity'].get('type_token_ratio', 0)
            print(f"  • Diversidad (TTR): {ttr:.3f}")
        
        if 'coherence' in global_metrics:
            consistency = global_metrics['coherence'].get('consistency_score', 0)
            fluency = global_metrics['coherence'].get('fluency_score', 0)
            print(f"  • Consistencia: {consistency:.3f}")
            print(f"  • Fluidez: {fluency:.3f}")
    
    print(f"\n🎖️  PRÓXIMOS PASOS:")
    print(f"  1. Analizar resultados en dashboard (puerto 8501/8502)")
    print(f"  2. Considerar entrenar modelos con configuraciones optimizadas")
    print(f"  3. Implementar mejoras sugeridas en generación de texto")
    print(f"  4. Explorar arquitecturas avanzadas (DynamicMemory, HierarchicalAttention)")

def main():
    """Función principal del análisis."""
    print(f"🔬 INICIANDO ANÁLISIS COMPLETO DEL ENTRENAMIENTO RECIENTE")
    print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Ejecutar análisis
        analyze_model_performance()
        analyze_training_evolution()
        generate_recommendations()
        create_summary_report()
        
        print(f"\n{'='*70}")
        print(f"✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n❌ Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()