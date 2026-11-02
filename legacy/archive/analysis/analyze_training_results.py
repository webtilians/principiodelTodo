#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de resultados de entrenamiento INFINITO V5.2
Compara el rendimiento de diferentes configuraciones
"""

import json
import os
import glob
from datetime import datetime

def load_training_history(file_path):
    """Carga historial de entrenamiento."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None

def analyze_training_results():
    """Analiza todos los resultados de entrenamiento disponibles."""
    
    print("🔬 ANÁLISIS DE RESULTADOS INFINITO V5.2")
    print("=" * 80)
    
    results_dir = "results/training"
    
    # Buscar archivos más recientes
    iit_files = sorted(glob.glob(os.path.join(results_dir, "training_history_real_*.json")), 
                       key=os.path.getmtime, reverse=True)
    baseline_files = sorted(glob.glob(os.path.join(results_dir, "baseline_no_iit_history_*.json")), 
                           key=os.path.getmtime, reverse=True)
    
    print(f"📊 Archivos encontrados:")
    print(f"  IIT models: {len(iit_files)} archivos")
    print(f"  Baseline models: {len(baseline_files)} archivos")
    print()
    
    # Analizar modelo IIT más reciente
    if iit_files:
        print("🔥 MODELO IIT (más reciente):")
        print(f"  Archivo: {os.path.basename(iit_files[0])}")
        
        iit_history = load_training_history(iit_files[0])
        if iit_history and 'val_perplexity' in iit_history:
            epochs = len(iit_history['val_perplexity'])
            initial_ppl = iit_history['val_perplexity'][0]
            final_ppl = iit_history['val_perplexity'][-1]
            best_ppl = min(iit_history['val_perplexity'])
            best_epoch = iit_history['val_perplexity'].index(best_ppl) + 1
            
            print(f"  Épocas entrenadas: {epochs}")
            print(f"  PPL inicial: {initial_ppl:.2f}")
            print(f"  PPL final: {final_ppl:.2f}")
            print(f"  Mejor PPL: {best_ppl:.2f} (época {best_epoch})")
            
            # Calcular mejora
            improvement = ((initial_ppl - best_ppl) / initial_ppl) * 100
            print(f"  Mejora: {improvement:.1f}%")
            
            # Verificar si mejoró con el tiempo
            if final_ppl > best_ppl:
                degradation = ((final_ppl - best_ppl) / best_ppl) * 100
                print(f"  ⚠️  Degradación desde mejor: +{degradation:.1f}%")
            
            # Mostrar últimas 5 épocas para ver tendencia
            if len(iit_history['val_perplexity']) >= 5:
                recent_ppls = iit_history['val_perplexity'][-5:]
                print(f"  Últimas 5 épocas PPL: {[f'{p:.1f}' for p in recent_ppls]}")
                
                # Verificar si está subiendo
                if recent_ppls[-1] > recent_ppls[0]:
                    print("  📈 Tendencia: EMPEORANDO (overfitting)")
                elif recent_ppls[-1] < recent_ppls[0]:
                    print("  📉 Tendencia: MEJORANDO")
                else:
                    print("  ➡️  Tendencia: ESTABLE")
        print()
    
    # Analizar baseline más reciente
    if baseline_files:
        print("🔬 MODELO BASELINE (más reciente):")
        print(f"  Archivo: {os.path.basename(baseline_files[0])}")
        
        baseline_history = load_training_history(baseline_files[0])
        if baseline_history and 'val_perplexity' in baseline_history:
            epochs = len(baseline_history['val_perplexity'])
            final_ppl = baseline_history['val_perplexity'][-1]
            
            print(f"  Épocas entrenadas: {epochs}")
            print(f"  PPL final: {final_ppl:.2f}")
        print()
    
    # Comparación directa si ambos están disponibles
    if iit_files and baseline_files and iit_history and baseline_history:
        print("🥊 COMPARACIÓN DIRECTA:")
        
        if 'val_perplexity' in iit_history and 'val_perplexity' in baseline_history:
            iit_best = min(iit_history['val_perplexity'])
            baseline_ppl = baseline_history['val_perplexity'][-1]
            
            print(f"  IIT mejor PPL: {iit_best:.2f}")
            print(f"  Baseline PPL: {baseline_ppl:.2f}")
            
            if iit_best < baseline_ppl:
                improvement = ((baseline_ppl - iit_best) / baseline_ppl) * 100
                print(f"  🏆 IIT es mejor por {improvement:.1f}%")
            else:
                degradation = ((iit_best - baseline_ppl) / baseline_ppl) * 100
                print(f"  🚨 BASELINE es mejor por {degradation:.1f}%")
                print("  ⚠️  Las características IIT no están ayudando!")
        print()
    
    # Analizar problemas comunes
    print("🔍 DIAGNÓSTICO DE PROBLEMAS:")
    
    if iit_history and 'val_perplexity' in iit_history:
        val_ppls = iit_history['val_perplexity']
        
        # Problema 1: PPL muy alto
        if min(val_ppls) > 200:
            print("  🚨 PPL muy alto (>200)")
            print("     - Modelo no está aprendiendo efectivamente")
            print("     - Posibles causas: LR muy alto, arquitectura problemática")
        
        # Problema 2: No mejora después de épocas iniciales
        if len(val_ppls) >= 5:
            early_avg = sum(val_ppls[:2]) / 2
            late_avg = sum(val_ppls[-3:]) / 3
            
            if late_avg >= early_avg:
                print("  🚨 No mejora en épocas tardías")
                print("     - Posible overfitting o LR muy alto")
                print("     - Considerar: reducir LR, aumentar dropout, early stopping")
        
        # Problema 3: Oscilaciones
        if len(val_ppls) >= 3:
            oscillations = 0
            for i in range(1, len(val_ppls)-1):
                if (val_ppls[i-1] < val_ppls[i] > val_ppls[i+1]) or (val_ppls[i-1] > val_ppls[i] < val_ppls[i+1]):
                    oscillations += 1
            
            if oscillations > len(val_ppls) * 0.3:
                print("  ⚠️  Muchas oscilaciones en validación")
                print("     - LR puede ser muy alto")
                print("     - Batch size muy pequeño")
    
    print()
    print("💡 RECOMENDACIONES:")
    
    # Si IIT no funciona mejor que baseline
    if (iit_files and baseline_files and iit_history and baseline_history and 
        'val_perplexity' in iit_history and 'val_perplexity' in baseline_history):
        
        iit_best = min(iit_history['val_perplexity'])
        baseline_ppl = baseline_history['val_perplexity'][-1]
        
        if iit_best >= baseline_ppl:
            print("  🔧 Las características IIT no están funcionando:")
            print("     1. Reducir lambda_phi (de 0.3 a 0.1)")
            print("     2. Aumentar épocas de warm-up")
            print("     3. Reducir learning rate (de 5e-4 a 1e-4)")
            print("     4. Simplificar arquitectura IIT")
            print("     5. Verificar implementación de métricas IIT")
    
    if iit_history and 'val_perplexity' in iit_history:
        final_ppl = iit_history['val_perplexity'][-1]
        
        if final_ppl > 250:
            print("  🔧 PPL muy alto, ajustes sugeridos:")
            print("     1. Reducir learning rate a 1e-4 o 5e-5")
            print("     2. Aumentar dropout a 0.2-0.3")
            print("     3. Reducir lambda_phi a 0.1")
            print("     4. Usar más épocas con early stopping")

if __name__ == '__main__':
    analyze_training_results()