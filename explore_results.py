#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 EXPLORADOR DE RESULTADOS - INFINITO V5.2
==========================================

Guía completa para examinar todos los resultados de los modelos.
"""

import os
import json
import glob
from datetime import datetime

def show_all_results_locations():
    """Muestra todas las ubicaciones donde están los resultados."""
    
    print("📍 UBICACIONES DE TODOS LOS RESULTADOS")
    print("=" * 70)
    
    # 1. Modelos entrenados
    print("\n🤖 MODELOS ENTRENADOS:")
    checkpoints_dir = "models/checkpoints"
    if os.path.exists(checkpoints_dir):
        model_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.pt')]
        for i, model_file in enumerate(sorted(model_files), 1):
            filepath = os.path.join(checkpoints_dir, model_file)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            print(f"   {i:2d}. {model_file}")
            print(f"       📁 {filepath}")
            print(f"       📊 {size_mb:.1f}MB | {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    
    # 2. Historiales de entrenamiento
    print("📈 HISTORIALES DE ENTRENAMIENTO:")
    training_dir = "results/training"
    if os.path.exists(training_dir):
        history_files = [f for f in os.listdir(training_dir) if f.endswith('.json')]
        for i, history_file in enumerate(sorted(history_files, reverse=True)[:10], 1):
            filepath = os.path.join(training_dir, history_file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            print(f"   {i:2d}. {history_file}")
            print(f"       📁 {filepath}")
            print(f"       🕐 {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
    
    # 3. Comparaciones de modelos
    print("🔬 COMPARACIONES Y ANÁLISIS:")
    results_dir = "results"
    if os.path.exists(results_dir):
        comparison_files = [f for f in os.listdir(results_dir) if 'comparison' in f or 'analysis' in f]
        for i, comp_file in enumerate(sorted(comparison_files, reverse=True), 1):
            filepath = os.path.join(results_dir, comp_file)
            if os.path.isfile(filepath):
                mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                print(f"   {i:2d}. {comp_file}")
                print(f"       📁 {filepath}")
                print(f"       🕐 {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print()

def analyze_specific_result(file_path):
    """Analiza en detalle un archivo de resultados específico."""
    
    print(f"\n🔍 ANÁLISIS DETALLADO: {os.path.basename(file_path)}")
    print("=" * 70)
    
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return
    
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print("📋 CONTENIDO JSON:")
            print()
            
            # Analizar diferentes tipos de archivos JSON
            if 'val_perplexity' in data:
                # Es un historial de entrenamiento
                analyze_training_history(data)
            elif 'generations' in data or any('generated' in str(v) for v in data.values() if isinstance(v, dict)):
                # Es una comparación de modelos
                analyze_model_comparison(data)
            else:
                # Estructura genérica
                print("🗂️ ESTRUCTURA DEL ARCHIVO:")
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"   {key}: Lista con {len(value)} elementos")
                        if value and isinstance(value[0], (int, float)):
                            print(f"      Rango: {min(value):.3f} - {max(value):.3f}")
                    elif isinstance(value, dict):
                        print(f"   {key}: Diccionario con {len(value)} claves")
                    else:
                        print(f"   {key}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
        
        elif file_path.endswith('.pt'):
            # Es un checkpoint de modelo
            analyze_model_checkpoint(file_path)
        
        else:
            print(f"⚠️ Tipo de archivo no reconocido: {file_path}")
    
    except Exception as e:
        print(f"❌ Error analizando archivo: {e}")

def analyze_training_history(data):
    """Analiza un historial de entrenamiento en detalle."""
    
    print("📈 HISTORIAL DE ENTRENAMIENTO:")
    print()
    
    if 'val_perplexity' in data and data['val_perplexity']:
        val_ppls = data['val_perplexity']
        train_ppls = data.get('train_perplexity', [])
        
        print(f"📊 MÉTRICAS GENERALES:")
        print(f"   Épocas completadas: {len(val_ppls)}")
        print(f"   PPL validación inicial: {val_ppls[0]:.2f}")
        print(f"   PPL validación final: {val_ppls[-1]:.2f}")
        print(f"   Mejor PPL validación: {min(val_ppls):.2f} (época {val_ppls.index(min(val_ppls)) + 1})")
        print()
        
        improvement = ((val_ppls[0] - min(val_ppls)) / val_ppls[0]) * 100
        print(f"   📈 Mejora total: {improvement:.1f}%")
        
        # Tendencia final
        if len(val_ppls) >= 3:
            recent_trend = val_ppls[-3:]
            if recent_trend[-1] < recent_trend[0]:
                print("   📉 Tendencia final: MEJORANDO")
            elif recent_trend[-1] > recent_trend[0]:
                print("   📈 Tendencia final: EMPEORANDO (overfitting)")
            else:
                print("   ➡️ Tendencia final: ESTABLE")
        print()
        
        print("📋 PROGRESIÓN ÉPOCA POR ÉPOCA:")
        for i, val_ppl in enumerate(val_ppls, 1):
            train_ppl = train_ppls[i-1] if i-1 < len(train_ppls) else "N/A"
            
            # Indicador de tendencia
            trend = ""
            if i > 1:
                if val_ppl < val_ppls[i-2]:
                    trend = "📉"
                elif val_ppl > val_ppls[i-2]:
                    trend = "📈"
                else:
                    trend = "➡️"
            
            print(f"   Época {i:2d}: Val={val_ppl:7.2f} | Train={train_ppl} {trend}")
        print()
    
    # Métricas IIT si están disponibles
    if 'train_phi' in data and data['train_phi']:
        phi_values = data['train_phi']
        print("🧠 MÉTRICAS IIT:")
        print(f"   PHI inicial: {phi_values[0]:.4f}")
        print(f"   PHI final: {phi_values[-1]:.4f}")
        print(f"   Cambio PHI: {phi_values[-1] - phi_values[0]:+.4f}")
        print()
    
    # Learning rate progression
    if 'learning_rate' in data and data['learning_rate']:
        lr_values = data['learning_rate']
        print("📚 LEARNING RATE:")
        print(f"   LR inicial: {lr_values[0]:.2e}")
        print(f"   LR final: {lr_values[-1]:.2e}")
        if lr_values[-1] < lr_values[0]:
            reduction = ((lr_values[0] - lr_values[-1]) / lr_values[0]) * 100
            print(f"   Reducción: {reduction:.1f}% (scheduler activo)")
        print()

def analyze_model_comparison(data):
    """Analiza una comparación de modelos."""
    
    print("🏆 COMPARACIÓN DE MODELOS:")
    print()
    
    for model_name, model_data in data.items():
        if isinstance(model_data, dict) and 'avg_perplexity' in model_data:
            print(f"🤖 {model_name}:")
            print(f"   Perplexity promedio: {model_data['avg_perplexity']:.2f}")
            
            if 'generations' in model_data:
                print(f"   Generaciones exitosas: {len(model_data['generations'])}")
                
                # Mostrar algunas generaciones
                print("   📝 Ejemplos de generación:")
                for i, gen in enumerate(model_data['generations'][:3], 1):
                    prompt = gen.get('prompt', 'N/A')
                    generated = gen.get('generated', 'N/A')[:80]
                    ppl = gen.get('perplexity', 0)
                    print(f"      {i}. \"{prompt}\" → \"{generated}...\" (PPL: {ppl:.2f})")
            print()

def analyze_model_checkpoint(file_path):
    """Analiza un checkpoint de modelo."""
    
    print("💾 CHECKPOINT DE MODELO:")
    print()
    
    try:
        import torch
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        
        print("📊 INFORMACIÓN DEL CHECKPOINT:")
        print(f"   Época: {checkpoint.get('epoch', 'Desconocida')}")
        print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"   Val PPL: {checkpoint.get('val_ppl', 'N/A')}")
        print()
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("⚙️ CONFIGURACIÓN DEL MODELO:")
            for key, value in config.items():
                print(f"   {key}: {value}")
            print()
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            total_params = sum(p.numel() for p in state_dict.values() if hasattr(p, 'numel'))
            print(f"🔢 PARÁMETROS TOTALES: {total_params:,}")
            print()
            
            # Mostrar algunas capas
            print("🏗️ ESTRUCTURA (primeras capas):")
            for i, (name, tensor) in enumerate(list(state_dict.items())[:10]):
                print(f"   {name}: {list(tensor.shape)}")
            if len(state_dict) > 10:
                print(f"   ... y {len(state_dict) - 10} capas más")
    
    except Exception as e:
        print(f"❌ Error cargando checkpoint: {e}")

def create_results_browser():
    """Crea un navegador interactivo de resultados."""
    
    print("\n🌐 NAVEGADOR DE RESULTADOS")
    print("=" * 70)
    
    all_files = []
    
    # Recopilar todos los archivos
    for root, dirs, files in os.walk("."):
        for file in files:
            if any(keyword in file.lower() for keyword in ['result', 'history', 'comparison', 'analysis']) or file.endswith('.pt'):
                if not file.startswith('.') and 'checkpoint' not in root:
                    full_path = os.path.join(root, file)
                    all_files.append(full_path)
    
    # Mostrar archivos ordenados por fecha
    all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"📁 ARCHIVOS ENCONTRADOS ({len(all_files)}):")
    print()
    
    for i, file_path in enumerate(all_files[:20], 1):  # Mostrar los 20 más recientes
        rel_path = os.path.relpath(file_path)
        size_mb = os.path.getsize(file_path) / (1024*1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        print(f"   {i:2d}. {rel_path}")
        print(f"       📊 {size_mb:.2f}MB | 🕐 {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    if len(all_files) > 20:
        print(f"   ... y {len(all_files) - 20} archivos más")

def main():
    """Función principal del explorador."""
    
    print("📊 EXPLORADOR COMPLETO DE RESULTADOS - INFINITO V5.2")
    print("=" * 70)
    
    # Mostrar ubicaciones
    show_all_results_locations()
    
    # Crear navegador
    create_results_browser()
    
    print("\n💡 CÓMO EXAMINAR RESULTADOS EN DETALLE:")
    print("=" * 70)
    
    print("\n1. 📈 PARA HISTORIALES DE ENTRENAMIENTO:")
    print("   python -c \"")
    print("   import json")
    print("   with open('results/training/ARCHIVO.json', 'r') as f:")
    print("       data = json.load(f)")
    print("   print('Val PPL:', data['val_perplexity'])")
    print("   \"")
    print()
    
    print("2. 🤖 PARA CARGAR UN MODELO:")
    print("   python -c \"")
    print("   import torch")
    print("   checkpoint = torch.load('models/checkpoints/ARCHIVO.pt', map_location='cpu')")
    print("   print('Época:', checkpoint['epoch'])")
    print("   print('Val PPL:', checkpoint['val_ppl'])")
    print("   \"")
    print()
    
    print("3. 🔬 PARA ANÁLISIS ESPECÍFICO:")
    print("   python analyze_specific_result.py --file RUTA_ARCHIVO")
    print()
    
    print("📍 ARCHIVOS MÁS IMPORTANTES:")
    print(f"   🏆 Mejor modelo: models/checkpoints/infinito_v5.2_real_best.pt")
    print(f"   📊 Último entrenamiento: results/training/training_history_real_*.json (más reciente)")
    print(f"   🔬 Comparación modelos: results/model_comparison_*.json")

if __name__ == '__main__':
    main()