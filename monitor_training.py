#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 MONITOR DE ENTRENAMIENTO - Comparación IIT vs Baseline
=========================================================

Script para monitorear el progreso del entrenamiento de 20 épocas
y comparar con los resultados del baseline.

RESULTADOS ACTUALES:
- Baseline (SIN IIT): 187.08 PPL (5 épocas)
- Modelo IIT: EN PROGRESO (20 épocas)

OBJETIVO:
Verificar si el modelo CON IIT puede superar al baseline con más entrenamiento.
"""

import os
import json
import glob
import time
from datetime import datetime

def check_training_progress():
    """Monitorea el progreso del entrenamiento."""
    
    print(f"\n{'='*70}")
    print(f"📊 MONITOR DE ENTRENAMIENTO - {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")
    
    # Revisar checkpoints
    checkpoint_dir = "models/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        print(f"\n💾 CHECKPOINTS DISPONIBLES:")
        for i, checkpoint in enumerate(checkpoints[:5]):  # Solo los 5 más recientes
            name = os.path.basename(checkpoint)
            size_mb = os.path.getsize(checkpoint) / (1024*1024)
            mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint))
            
            if "baseline" in name:
                status = "✅ COMPLETADO"
                detail = "(Baseline SIN IIT)"
            elif "infinito" in name or "real" in name:
                status = "🔄 EN PROGRESO" if i == 0 else "📁 Guardado"
                detail = "(Modelo CON IIT)"
            else:
                status = "📁 Guardado"
                detail = ""
            
            print(f"  {i+1}. {name}")
            print(f"     {size_mb:.1f} MB - {mod_time.strftime('%H:%M:%S')} - {status} {detail}")
    
    # Revisar logs de entrenamiento
    results_dir = "results/training"
    if os.path.exists(results_dir):
        history_files = glob.glob(os.path.join(results_dir, "*.json"))
        history_files.sort(key=os.path.getmtime, reverse=True)
        
        print(f"\n📈 HISTORIALES DE ENTRENAMIENTO:")
        for i, hist_file in enumerate(history_files[:3]):  # Solo los 3 más recientes
            name = os.path.basename(hist_file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(hist_file))
            
            try:
                with open(hist_file, 'r') as f:
                    history = json.load(f)
                
                if 'val_perplexity' in history and history['val_perplexity']:
                    epochs_completed = len(history['val_perplexity'])
                    last_val_ppl = history['val_perplexity'][-1]
                    best_val_ppl = min(history['val_perplexity'])
                    
                    if "baseline" in name:
                        model_type = "Baseline (SIN IIT)"
                        status = "✅ COMPLETADO"
                    else:
                        model_type = "Modelo (CON IIT)"
                        status = "🔄 EN PROGRESO" if i == 0 else "✅ COMPLETADO"
                    
                    print(f"  {i+1}. {name}")
                    print(f"     {model_type} - {status}")
                    print(f"     Épocas: {epochs_completed} | Último Val PPL: {last_val_ppl:.2f}")
                    print(f"     Mejor Val PPL: {best_val_ppl:.2f} | {mod_time.strftime('%H:%M:%S')}")
                    
            except Exception as e:
                print(f"  {i+1}. {name} - Error leyendo: {e}")
    
    # Comparación actual
    print(f"\n🔬 COMPARACIÓN CIENTÍFICA ACTUAL:")
    print(f"  Baseline (SIN IIT):  187.08 PPL ✅ (5 épocas)")
    print(f"  Modelo A (CON IIT):  216.46 PPL ❌ (6 épocas)")
    print(f"  Modelo B (CON IIT):  207.15 PPL ❌ (5 épocas)")
    print(f"  Nuevo CON IIT:       🔄 EN PROGRESO (20 épocas)")
    
    print(f"\n🎯 OBJETIVOS PARA 20 ÉPOCAS:")
    print(f"  Para que IIT sea beneficioso: Val PPL < 187.08")
    print(f"  Mejora mínima esperada: ~15-20% (Val PPL ~150-160)")
    print(f"  Mejora significativa: ~25-30% (Val PPL ~130-140)")
    
    print(f"\n{'='*70}")

def continuous_monitor():
    """Monitoreo continuo cada 30 segundos."""
    print("🔄 Iniciando monitoreo continuo...")
    print("   Presiona Ctrl+C para detener")
    
    try:
        while True:
            check_training_progress()
            print("\n⏳ Esperando 30 segundos...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoreo detenido por el usuario")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        continuous_monitor()
    else:
        check_training_progress()
        print("\n💡 Usa 'python monitor_training.py --continuous' para monitoreo automático")