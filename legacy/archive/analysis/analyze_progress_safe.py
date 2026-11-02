#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analizador de Progreso del Entrenamiento - Sin Interrupciones
============================================================

Este script analiza el progreso actual del entrenamiento y compara 
resultados históricos SIN interrumpir entrenamientos en curso.
"""

import os
import json
import glob
import torch
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def analyze_current_progress():
    """
    Analiza el progreso actual sin interrumpir entrenamientos
    """
    print("=" * 60)
    print("ANÁLISIS DE PROGRESO ACTUAL (SIN INTERRUPCIONES)")
    print("=" * 60)
    
    # 1. Buscar archivos de historial recientes
    history_files = []
    patterns = ["training_history*.json", "*history*.json", "results/training/*.json"]
    
    for pattern in patterns:
        history_files.extend(glob.glob(pattern))
    
    if not history_files:
        print("[INFO] No se encontraron archivos de historial")
        return
    
    # Ordenar por fecha de modificación (más reciente primero)
    history_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"[INFO] Encontrados {len(history_files)} archivos de historial")
    
    # 2. Analizar archivos más recientes
    for i, file_path in enumerate(history_files[:3]):  # Solo los 3 más recientes
        print(f"\n{'='*50}")
        print(f"ARCHIVO {i+1}: {os.path.basename(file_path)}")
        print(f"{'='*50}")
        
        try:
            with open(file_path, 'r') as f:
                history = json.load(f)
            
            # Información básica
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"[TIME] Última modificación: {mod_time}")
            print(f"[SIZE] Tamaño archivo: {os.path.getsize(file_path)} bytes")
            
            # Analizar métricas
            if 'train_loss' in history and history['train_loss']:
                epochs = len(history['train_loss'])
                final_train_loss = history['train_loss'][-1]
                final_val_loss = history.get('val_loss', [0])[-1] if history.get('val_loss') else None
                final_train_ppl = history.get('train_perplexity', [0])[-1] if history.get('train_perplexity') else None
                final_val_ppl = history.get('val_perplexity', [0])[-1] if history.get('val_perplexity') else None
                
                print(f"[EPOCHS] Épocas completadas: {epochs}")
                print(f"[LOSS] Train Loss: {final_train_loss:.4f}")
                if final_val_loss:
                    print(f"[LOSS] Val Loss: {final_val_loss:.4f}")
                if final_train_ppl:
                    print(f"[PPL] Train PPL: {final_train_ppl:.2f}")
                if final_val_ppl:
                    print(f"[PPL] Val PPL: {final_val_ppl:.2f}")
                
                # Tendencia de mejora
                if len(history['train_loss']) >= 2:
                    improvement = history['train_loss'][0] - history['train_loss'][-1]
                    improvement_pct = (improvement / history['train_loss'][0]) * 100
                    print(f"[TREND] Mejora total: {improvement:.4f} ({improvement_pct:.2f}%)")
                
                # Métricas IIT si están disponibles
                if 'train_phi' in history and history['train_phi']:
                    final_phi = history['train_phi'][-1]
                    print(f"[IIT] Final PHI: {final_phi:.4f}")
                
                if 'learning_rate' in history and history['learning_rate']:
                    final_lr = history['learning_rate'][-1]
                    print(f"[LR] Learning Rate actual: {final_lr:.2e}")
        
        except Exception as e:
            print(f"[ERROR] Error procesando archivo: {e}")
    
    print(f"\n{'='*60}")

def analyze_model_checkpoints():
    """
    Analiza los checkpoints de modelo disponibles
    """
    print("ANÁLISIS DE CHECKPOINTS DISPONIBLES")
    print("=" * 60)
    
    # Buscar archivos de modelo
    model_patterns = ["*.pt", "*.pth", "models/checkpoints/*.pt"]
    model_files = []
    
    for pattern in model_patterns:
        model_files.extend(glob.glob(pattern))
    
    if not model_files:
        print("[INFO] No se encontraron checkpoints")
        return
    
    # Ordenar por tamaño y fecha
    model_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"[INFO] Encontrados {len(model_files)} checkpoints")
    
    for i, model_path in enumerate(model_files[:5]):  # Solo los 5 más recientes
        try:
            stat = os.stat(model_path)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            
            print(f"\n[{i+1}] {os.path.basename(model_path)}")
            print(f"     Tamaño: {size_mb:.1f} MB")
            print(f"     Fecha: {mod_time}")
            
            # Intentar cargar metadata si es posible (sin GPU)
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict):
                    if 'epoch' in checkpoint:
                        print(f"     Época: {checkpoint['epoch']}")
                    if 'val_loss' in checkpoint:
                        print(f"     Val Loss: {checkpoint['val_loss']:.4f}")
                    if 'val_ppl' in checkpoint:
                        print(f"     Val PPL: {checkpoint['val_ppl']:.2f}")
                    if 'config' in checkpoint:
                        config = checkpoint['config']
                        if 'vocab_size' in config:
                            print(f"     Vocab Size: {config['vocab_size']:,}")
                        if 'hidden_dim' in config:
                            print(f"     Hidden Dim: {config['hidden_dim']}")
                else:
                    print(f"     Tipo: Modelo directo (state_dict)")
            except Exception as e:
                print(f"     [Error cargando metadata: {str(e)[:50]}...]")
                
        except Exception as e:
            print(f"[ERROR] Error procesando {model_path}: {e}")

def compare_architectures():
    """
    Compara diferentes arquitecturas basándose en resultados existentes
    """
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE ARQUITECTURAS")
    print("=" * 60)
    
    # Buscar archivos de historial que contengan información de arquitectura
    results = {}
    
    history_files = glob.glob("training_history*.json") + glob.glob("results/training/*.json")
    
    for file_path in history_files:
        try:
            with open(file_path, 'r') as f:
                history = json.load(f)
            
            # Extraer información de arquitectura del nombre del archivo
            filename = os.path.basename(file_path)
            architecture = "unknown"
            
            # Detectar arquitectura por patrones en el nombre
            if "large" in filename.lower():
                architecture = "large_iit"
            elif "small" in filename.lower():
                architecture = "small_iit"
            elif "micro" in filename.lower():
                architecture = "micro_iit"
            elif "tiny" in filename.lower():
                architecture = "tiny_iit"
            
            # Calcular métricas finales
            if 'val_loss' in history and history['val_loss']:
                final_val_loss = history['val_loss'][-1]
                final_val_ppl = history.get('val_perplexity', [float('inf')])[-1]
                epochs = len(history['val_loss'])
                
                if architecture not in results:
                    results[architecture] = []
                
                results[architecture].append({
                    'file': filename,
                    'final_val_loss': final_val_loss,
                    'final_val_ppl': final_val_ppl,
                    'epochs': epochs,
                    'mod_time': os.path.getmtime(file_path)
                })
                
        except Exception as e:
            continue
    
    # Mostrar comparación
    if results:
        print(f"[INFO] Comparando {len(results)} arquitecturas")
        
        for arch, runs in results.items():
            print(f"\n--- {arch.upper()} ---")
            
            # Ordenar por fecha (más reciente primero)
            runs.sort(key=lambda x: x['mod_time'], reverse=True)
            
            for i, run in enumerate(runs[:2]):  # Solo las 2 ejecuciones más recientes
                print(f"  Run {i+1}: {run['file']}")
                print(f"    Val Loss: {run['final_val_loss']:.4f}")
                print(f"    Val PPL: {run['final_val_ppl']:.2f}")
                print(f"    Épocas: {run['epochs']}")
                
                mod_time = datetime.fromtimestamp(run['mod_time'])
                print(f"    Fecha: {mod_time}")
        
        # Ranking por mejor perplexity
        print(f"\n{'='*30}")
        print("RANKING POR MEJOR PERPLEXITY")
        print(f"{'='*30}")
        
        best_runs = []
        for arch, runs in results.items():
            if runs:
                best_run = min(runs, key=lambda x: x['final_val_ppl'])
                best_runs.append((arch, best_run))
        
        # Ordenar por perplexity
        best_runs.sort(key=lambda x: x[1]['final_val_ppl'])
        
        for i, (arch, run) in enumerate(best_runs):
            print(f"{i+1}. {arch}: PPL {run['final_val_ppl']:.2f} (Loss {run['final_val_loss']:.4f})")
    
    else:
        print("[INFO] No se encontraron resultados para comparar")

def generate_quick_visualization():
    """
    Genera una visualización rápida de resultados sin usar archivos temporales
    """
    print("\n" + "=" * 60)
    print("GENERANDO VISUALIZACIÓN RÁPIDA")
    print("=" * 60)
    
    try:
        # Buscar el archivo de historial más reciente
        history_files = glob.glob("training_history*.json")
        if not history_files:
            print("[INFO] No hay archivos de historial para visualizar")
            return
        
        latest_file = max(history_files, key=os.path.getmtime)
        
        with open(latest_file, 'r') as f:
            history = json.load(f)
        
        print(f"[INFO] Visualizando: {os.path.basename(latest_file)}")
        
        # Crear gráfico simple
        if 'train_loss' in history and 'val_loss' in history:
            epochs = list(range(1, len(history['train_loss']) + 1))
            
            plt.figure(figsize=(12, 4))
            
            # Subplot 1: Loss
            plt.subplot(1, 2, 1)
            plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', alpha=0.7)
            if history['val_loss']:
                plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', alpha=0.7)
            plt.xlabel('Época')
            plt.ylabel('Loss')
            plt.title('Evolución del Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Subplot 2: Perplexity
            plt.subplot(1, 2, 2)
            if 'train_perplexity' in history:
                plt.plot(epochs, history['train_perplexity'], 'b-', label='Train PPL', alpha=0.7)
            if 'val_perplexity' in history:
                plt.plot(epochs, history['val_perplexity'], 'r-', label='Val PPL', alpha=0.7)
            plt.xlabel('Época')
            plt.ylabel('Perplexity')
            plt.title('Evolución del Perplexity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Escala logarítmica para mejor visualización
            
            plt.tight_layout()
            
            # Guardar en directorio temporal
            viz_path = f"progress_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[SAVED] Visualización guardada: {viz_path}")
        else:
            print("[INFO] No hay datos suficientes para visualizar")
            
    except Exception as e:
        print(f"[ERROR] Error generando visualización: {e}")

def main():
    """
    Función principal que ejecuta todos los análisis
    """
    print("INICIANDO ANÁLISIS COMPLETO DEL PROGRESO")
    print("=" * 60)
    print("NOTA: Este script NO interrumpe entrenamientos en curso")
    print("=" * 60)
    
    try:
        # 1. Analizar progreso actual
        analyze_current_progress()
        
        # 2. Analizar checkpoints
        analyze_model_checkpoints()
        
        # 3. Comparar arquitecturas
        compare_architectures()
        
        # 4. Generar visualización
        generate_quick_visualization()
        
        print(f"\n{'='*60}")
        print("✅ ANÁLISIS COMPLETADO")
        print("RESUMEN DE RECOMENDACIONES:")
        print("• Revisar los mejores checkpoints para continuar entrenamiento")
        print("• Comparar perplexities entre arquitecturas")
        print("• Analizar tendencias de mejora por época")
        print("• Considerar early stopping si no hay mejoras")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"[ERROR] Error en análisis: {e}")

if __name__ == "__main__":
    main()