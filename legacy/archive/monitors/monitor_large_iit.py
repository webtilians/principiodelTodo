#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de entrenamiento en tiempo real para INFINITO V5.2
"""

import time
import json
import os
from datetime import datetime

def monitor_training():
    """Monitorea el progreso del entrenamiento en tiempo real."""
    print("🔍 Monitor de entrenamiento INFINITO V5.2 (large_iit)")
    print("=" * 60)
    
    results_dir = "results/training"
    
    while True:
        try:
            # Buscar archivos de historial recientes
            if os.path.exists(results_dir):
                files = [f for f in os.listdir(results_dir) if f.startswith('training_history_real_') and f.endswith('.json')]
                if files:
                    # Obtener el archivo más reciente
                    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
                    file_path = os.path.join(results_dir, latest_file)
                    
                    # Leer historial
                    with open(file_path, 'r') as f:
                        history = json.load(f)
                    
                    # Mostrar progreso
                    print(f"\n📊 Progreso actual ({datetime.now().strftime('%H:%M:%S')}):")
                    print(f"Archivo: {latest_file}")
                    
                    if history['val_loss']:
                        epochs_completed = len(history['val_loss'])
                        latest_val_loss = history['val_loss'][-1]
                        latest_val_ppl = history['val_perplexity'][-1]
                        latest_lr = history['learning_rate'][-1]
                        
                        print(f"Épocas completadas: {epochs_completed}/20")
                        print(f"Val Loss: {latest_val_loss:.4f}")
                        print(f"Val PPL: {latest_val_ppl:.2f}")
                        print(f"Learning Rate: {latest_lr:.2e}")
                        
                        if 'train_phi' in history and history['train_phi']:
                            latest_phi = history['train_phi'][-1]
                            latest_phi_loss = history['train_loss_phi'][-1]
                            print(f"Train PHI: {latest_phi:.4f}")
                            print(f"ΔPhi Loss: {latest_phi_loss:.6f}")
                        
                        # Progreso visual
                        progress = epochs_completed / 20 * 100
                        bar_length = 30
                        filled_length = int(bar_length * epochs_completed / 20)
                        bar = '█' * filled_length + '-' * (bar_length - filled_length)
                        print(f"Progreso: [{bar}] {progress:.1f}%")
                    else:
                        print("Entrenamiento iniciando...")
                else:
                    print("No se encontraron archivos de historial")
            else:
                print("Directorio de resultados no encontrado")
                
        except Exception as e:
            print(f"Error leyendo historial: {e}")
        
        # Esperar 30 segundos antes de la siguiente verificación
        time.sleep(30)

if __name__ == '__main__':
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n🛑 Monitor detenido por usuario")