#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de entrenamiento en tiempo real para INFINITO V5.2 Small IIT
"""

import time
import json
import os
from datetime import datetime

def monitor_small_iit_training():
    """Monitorea el progreso del entrenamiento small_iit en tiempo real."""
    print("🔍 Monitor de entrenamiento INFINITO V5.2 (small_iit)")
    print("=" * 60)
    print("📊 Configuración esperada:")
    print("  • hidden_dim: 384")
    print("  • num_layers: 3") 
    print("  • num_heads: 6")
    print("  • batch_size: 16")
    print("  • learning_rate: 5e-4")
    print("  • epochs: 20")
    print("  • patience: 5")
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
                    print(f"\n📊 Progreso Small IIT ({datetime.now().strftime('%H:%M:%S')}):")
                    print(f"Archivo: {latest_file}")
                    
                    if history['val_loss']:
                        epochs_completed = len(history['val_loss'])
                        latest_train_loss = history['train_loss'][-1]
                        latest_train_ppl = history['train_perplexity'][-1]
                        latest_val_loss = history['val_loss'][-1]
                        latest_val_ppl = history['val_perplexity'][-1]
                        latest_lr = history['learning_rate'][-1]
                        
                        print(f"Épocas completadas: {epochs_completed}/20")
                        print(f"Train Loss: {latest_train_loss:.4f} | Train PPL: {latest_train_ppl:.2f}")
                        print(f"Val Loss: {latest_val_loss:.4f} | Val PPL: {latest_val_ppl:.2f}")
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
                        
                        # Mostrar tendencia de val_loss
                        if len(history['val_loss']) >= 3:
                            recent_losses = history['val_loss'][-3:]
                            if recent_losses[-1] < recent_losses[-2]:
                                trend = "📈 Mejorando"
                            elif recent_losses[-1] > recent_losses[-2]:
                                trend = "📉 Empeorando"
                            else:
                                trend = "➡️ Estable"
                            print(f"Tendencia: {trend}")
                            
                        # Mejor modelo hasta ahora
                        best_val_loss = min(history['val_loss'])
                        best_epoch = history['val_loss'].index(best_val_loss) + 1
                        print(f"🏆 Mejor: Epoch {best_epoch}, Val Loss {best_val_loss:.4f}")
                        
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
        monitor_small_iit_training()
    except KeyboardInterrupt:
        print("\n🛑 Monitor detenido por usuario")