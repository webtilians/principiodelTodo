#!/usr/bin/env python3
"""
Script para monitorear el progreso del entrenamiento base mejorado
"""

import json
import os
import time
from pathlib import Path

def monitor_training():
    """Monitorea el progreso del entrenamiento en tiempo real"""
    
    history_file = Path("models/checkpoints/training_history_improved.json")
    
    print("\n" + "="*70)
    print("📊 MONITOR DE ENTRENAMIENTO - MODELO BASE INFINITO")
    print("="*70)
    print("\nEsperando inicio del entrenamiento...")
    print("(El archivo de historial se creará cuando complete la primera época)\n")
    
    last_epoch = 0
    
    while True:
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                if len(history) > last_epoch:
                    # Nuevas épocas completadas
                    for epoch_data in history[last_epoch:]:
                        epoch = epoch_data['epoch']
                        train_ppl = epoch_data['train_ppl']
                        val_ppl = epoch_data['val_ppl']
                        train_phi = epoch_data.get('train_phi', 0)
                        val_phi = epoch_data.get('val_phi', 0)
                        epoch_time = epoch_data['epoch_time']
                        elapsed = epoch_data['elapsed_time']
                        
                        print(f"\n{'='*70}")
                        print(f"✅ ÉPOCA {epoch} COMPLETADA")
                        print(f"{'='*70}")
                        print(f"Train PPL: {train_ppl:6.2f}  |  Val PPL: {val_ppl:6.2f}")
                        print(f"Train ΔΦ:  {train_phi:6.4f}  |  Val ΔΦ:  {val_phi:6.4f}")
                        print(f"Tiempo época: {epoch_time:.1f}s  |  Total: {elapsed/60:.1f}min")
                        
                        # Calcular mejora vs primera época
                        if epoch > 1:
                            first_val_ppl = history[0]['val_ppl']
                            improvement = ((first_val_ppl - val_ppl) / first_val_ppl) * 100
                            print(f"Mejora vs Época 1: {improvement:+.1f}%")
                        
                        # Proyección tiempo restante
                        if epoch < 30:
                            avg_time_per_epoch = elapsed / epoch
                            remaining_epochs = 30 - epoch
                            estimated_remaining = avg_time_per_epoch * remaining_epochs
                            print(f"⏱️  Tiempo estimado restante: {estimated_remaining/60:.0f} min")
                    
                    last_epoch = len(history)
                    
                    # Mostrar resumen
                    best_val_ppl = min(h['val_ppl'] for h in history)
                    best_epoch = next(h['epoch'] for h in history if h['val_ppl'] == best_val_ppl)
                    
                    print(f"\n📈 PROGRESO ACTUAL:")
                    print(f"   Épocas completadas: {last_epoch}/30")
                    print(f"   Mejor Val PPL: {best_val_ppl:.2f} (época {best_epoch})")
                    print(f"   Val PPL actual: {history[-1]['val_ppl']:.2f}")
                    
                    # Detectar early stopping
                    if last_epoch >= 6:
                        last_5_vals = [h['val_ppl'] for h in history[-5:]]
                        if all(last_5_vals[i] >= last_5_vals[i-1] for i in range(1, 5)):
                            print(f"\n⚠️  ALERTA: Val PPL subiendo últimas 5 épocas")
                            print(f"   Posible early stopping próximo")
            
            else:
                # Archivo aún no existe
                print(".", end="", flush=True)
            
            time.sleep(10)  # Verificar cada 10 segundos
            
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoreo detenido por usuario")
            break
        except Exception as e:
            print(f"\n⚠️  Error leyendo historial: {e}")
            time.sleep(10)


if __name__ == '__main__':
    monitor_training()
