#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de progreso para entrenamiento optimizado
"""

import os
import json
import glob
import time
from datetime import datetime

def monitor_optimized_training():
    """Monitorea el progreso del entrenamiento optimizado."""
    
    print("🔍 MONITOR DE ENTRENAMIENTO OPTIMIZADO")
    print("=" * 60)
    
    results_dir = "results/training"
    
    # Buscar archivos de historial más recientes
    pattern = os.path.join(results_dir, "training_history_real_*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    
    if not files:
        print("❌ No se encontraron archivos de historial")
        return
    
    latest_file = files[0]
    file_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
    
    print(f"📄 Archivo más reciente: {os.path.basename(latest_file)}")
    print(f"🕐 Última modificación: {file_time.strftime('%H:%M:%S')}")
    print()
    
    try:
        with open(latest_file, 'r') as f:
            history = json.load(f)
        
        if 'val_perplexity' in history and history['val_perplexity']:
            epochs_completed = len(history['val_perplexity'])
            
            print(f"📊 PROGRESO ACTUAL:")
            print(f"  Épocas completadas: {epochs_completed}")
            print()
            
            print(f"📈 PERPLEXITY PROGRESSION:")
            for i, ppl in enumerate(history['val_perplexity'], 1):
                trend = ""
                if i > 1:
                    prev_ppl = history['val_perplexity'][i-2]
                    if ppl < prev_ppl:
                        trend = "📉 ↓"
                    elif ppl > prev_ppl:
                        trend = "📈 ↑"
                    else:
                        trend = "➡️ ="
                
                print(f"  Época {i:2d}: {ppl:6.2f} {trend}")
            
            if epochs_completed >= 2:
                initial_ppl = history['val_perplexity'][0]
                current_ppl = history['val_perplexity'][-1]
                improvement = ((initial_ppl - current_ppl) / initial_ppl) * 100
                
                print()
                print(f"💯 MEJORA TOTAL: {improvement:.1f}%")
                print(f"  Inicial: {initial_ppl:.2f}")
                print(f"  Actual:  {current_ppl:.2f}")
                
                # Verificar si está mejorando
                if epochs_completed >= 3:
                    recent_trend = history['val_perplexity'][-3:]
                    if recent_trend[-1] < recent_trend[0]:
                        print("  ✅ Tendencia: MEJORANDO")
                    elif recent_trend[-1] > recent_trend[0]:
                        print("  ⚠️  Tendencia: EMPEORANDO")
                    else:
                        print("  ➡️  Tendencia: ESTABLE")
        
        # Mostrar configuración de entrenamiento
        if 'learning_rate' in history and history['learning_rate']:
            current_lr = history['learning_rate'][-1]
            print()
            print(f"⚙️  CONFIGURACIÓN ACTUAL:")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Verificar si LR ha cambiado (scheduler activo)
            if len(history['learning_rate']) > 1:
                initial_lr = history['learning_rate'][0]
                if current_lr < initial_lr:
                    reduction = ((initial_lr - current_lr) / initial_lr) * 100
                    print(f"  LR Reducción: {reduction:.1f}% (scheduler activo)")
    
    except Exception as e:
        print(f"❌ Error leyendo historial: {e}")

def check_model_files():
    """Verifica archivos de modelos guardados."""
    
    print()
    print("💾 CHECKPOINTS GUARDADOS:")
    
    checkpoints_dir = "models/checkpoints"
    if os.path.exists(checkpoints_dir):
        files = os.listdir(checkpoints_dir)
        recent_files = [f for f in files if 'infinito_v5.2' in f and f.endswith('.pt')]
        
        if recent_files:
            for file in sorted(recent_files, key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True):
                filepath = os.path.join(checkpoints_dir, file)
                size_mb = os.path.getsize(filepath) / (1024*1024)
                mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                print(f"  📁 {file} ({size_mb:.1f}MB, {mod_time.strftime('%H:%M:%S')})")
        else:
            print("  ❌ No se encontraron checkpoints recientes")
    else:
        print("  ❌ Directorio de checkpoints no existe")

if __name__ == '__main__':
    monitor_optimized_training()
    check_model_files()
    
    print()
    print("🔄 Para monitoreo continuo, ejecuta este script cada pocos minutos")