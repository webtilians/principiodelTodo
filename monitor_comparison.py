#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor comparativo de entrenamientos INFINITO V5.2
Compara small_iit vs large_iit en tiempo real
"""

import time
import json
import os
from datetime import datetime
import glob

def get_latest_training_files():
    """Encuentra los archivos de entrenamiento más recientes."""
    results_dir = "results/training"
    
    if not os.path.exists(results_dir):
        return None, None
    
    files = glob.glob(os.path.join(results_dir, "training_history_real_*.json"))
    if len(files) < 2:
        return files[0] if files else None, None
    
    # Ordenar por fecha de modificación (más reciente primero)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return files[0], files[1] if len(files) > 1 else None

def load_history(file_path):
    """Carga el historial de entrenamiento."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

def format_time(seconds):
    """Formatea segundos en HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def monitor_comparison():
    """Monitorea y compara entrenamientos small vs large."""
    print("🔬 Monitor Comparativo INFINITO V5.2")
    print("=" * 80)
    print("📊 Configuraciones esperadas:")
    print("  Small IIT: 384 dim, 3 layers, 6 heads → ~45M parámetros")
    print("  Large IIT: 512 dim, 4 layers, 8 heads → ~65M parámetros")
    print("=" * 80)
    
    start_time = time.time()
    
    while True:
        try:
            latest, second_latest = get_latest_training_files()
            
            print(f"\n📊 Comparación ({datetime.now().strftime('%H:%M:%S')}) - Tiempo total: {format_time(time.time() - start_time)}")
            print("=" * 80)
            
            if latest:
                history1 = load_history(latest)
                if history1 and history1['val_loss']:
                    epochs1 = len(history1['val_loss'])
                    val_loss1 = history1['val_loss'][-1]
                    val_ppl1 = history1['val_perplexity'][-1]
                    lr1 = history1['learning_rate'][-1]
                    
                    # Determinar tamaño por número de parámetros aproximado
                    size1 = "Large IIT" if val_ppl1 > 200 else "Small IIT"  # Heurística temporal
                    
                    print(f"🔥 Entrenamiento 1 ({size1}):")
                    print(f"   Épocas: {epochs1}/20 ({epochs1/20*100:.1f}%)")
                    print(f"   Val Loss: {val_loss1:.4f} | Val PPL: {val_ppl1:.2f}")
                    print(f"   Learning Rate: {lr1:.2e}")
                    
                    if 'train_phi' in history1 and history1['train_phi']:
                        phi1 = history1['train_phi'][-1]
                        print(f"   Train PHI: {phi1:.4f}")
                    
                    # Barra de progreso
                    progress1 = epochs1 / 20 * 100
                    bar_length = 30
                    filled1 = int(bar_length * epochs1 / 20)
                    bar1 = '█' * filled1 + '-' * (bar_length - filled1)
                    print(f"   Progreso: [{bar1}] {progress1:.1f}%")
                    
                    # Mejor modelo
                    best_val1 = min(history1['val_loss'])
                    best_epoch1 = history1['val_loss'].index(best_val1) + 1
                    print(f"   🏆 Mejor: Epoch {best_epoch1}, Val Loss {best_val1:.4f}")
            
            print("-" * 40)
            
            if second_latest:
                history2 = load_history(second_latest)
                if history2 and history2['val_loss']:
                    epochs2 = len(history2['val_loss'])
                    val_loss2 = history2['val_loss'][-1]
                    val_ppl2 = history2['val_perplexity'][-1]
                    lr2 = history2['learning_rate'][-1]
                    
                    size2 = "Large IIT" if val_ppl2 > 200 else "Small IIT"
                    
                    print(f"⚡ Entrenamiento 2 ({size2}):")
                    print(f"   Épocas: {epochs2}/20 ({epochs2/20*100:.1f}%)")
                    print(f"   Val Loss: {val_loss2:.4f} | Val PPL: {val_ppl2:.2f}")
                    print(f"   Learning Rate: {lr2:.2e}")
                    
                    if 'train_phi' in history2 and history2['train_phi']:
                        phi2 = history2['train_phi'][-1]
                        print(f"   Train PHI: {phi2:.4f}")
                    
                    progress2 = epochs2 / 20 * 100
                    filled2 = int(bar_length * epochs2 / 20)
                    bar2 = '█' * filled2 + '-' * (bar_length - filled2)
                    print(f"   Progreso: [{bar2}] {progress2:.1f}%")
                    
                    best_val2 = min(history2['val_loss'])
                    best_epoch2 = history2['val_loss'].index(best_val2) + 1
                    print(f"   🏆 Mejor: Epoch {best_epoch2}, Val Loss {best_val2:.4f}")
                    
                    # Comparación directa
                    if history1 and history1['val_loss']:
                        print("\n🔬 Comparación directa:")
                        if best_val1 < best_val2:
                            diff = ((best_val2 - best_val1) / best_val2) * 100
                            print(f"   🥇 {size1} está ganando por {diff:.1f}% en Val Loss")
                        elif best_val2 < best_val1:
                            diff = ((best_val1 - best_val2) / best_val1) * 100
                            print(f"   🥇 {size2} está ganando por {diff:.1f}% en Val Loss")
                        else:
                            print(f"   🤝 Empate técnico en Val Loss")
            
            else:
                print("⏳ Esperando segundo entrenamiento...")
            
        except Exception as e:
            print(f"Error en monitoreo: {e}")
        
        # Esperar 45 segundos
        time.sleep(45)

if __name__ == '__main__':
    try:
        monitor_comparison()
    except KeyboardInterrupt:
        print("\n🛑 Monitor comparativo detenido")