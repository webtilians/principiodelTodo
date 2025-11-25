#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 VALIDACIÓN CIENTÍFICA - Baseline vs IIT Model
==============================================

Script para entrenar modelo baseline (SIN IIT) y comparar con modelo IIT
para validación científica rigurosa del impacto de características de conciencia.

CONFIGURACIÓN IDÉNTICA:
✅ WikiText-2 REAL, GPT-2 tokenizer
✅ 4 capas, 512 hidden, 8 heads  
✅ Mismos hiperparámetros optimizados
✅ Mismo seed para reproducibilidad

DIFERENCIAS:
❌ SIN IITGuidedMemory
❌ SIN ImprovedIITMetrics
❌ SIN LearnablePhiWeights  
❌ SIN StochasticExploration
"""

import subprocess
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def run_baseline_training():
    """Ejecuta entrenamiento de baseline."""
    print("\n" + "="*80)
    print("🔬 FASE 1: ENTRENAMIENTO BASELINE (SIN IIT)")
    print("="*80)
    
    # Comando para baseline
    cmd_baseline = [
        "python", "train_v5_2_baseline_no_iit.py",
        "--model-size", "large_baseline",
        "--epochs", "10",  # Suficiente para convergencia
        "--patience", "5",
        "--seed", "42",  # Mismo seed que modelo IIT
        "--lr", "1e-4",  # Mismos hiperparámetros optimizados
        "--dropout", "0.25",
        "--batch-size", "16",
        "--seq-len", "256"
    ]
    
    print(f"Comando baseline: {' '.join(cmd_baseline)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd_baseline, 
                              capture_output=True, 
                              text=True,
                              cwd=os.getcwd())
        
        baseline_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Baseline completado en {baseline_time/60:.1f} minutos")
            return True
        else:
            print(f"❌ Error en baseline:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Excepción en baseline: {e}")
        return False

def run_iit_training():
    """Ejecuta entrenamiento de modelo IIT con configuración idéntica."""
    print("\n" + "="*80)
    print("🧠 FASE 2: ENTRENAMIENTO IIT MODEL (CON IIT)")
    print("="*80)
    
    # Comando para IIT model
    cmd_iit = [
        "python", "train_v5_2_wikitext_real.py",
        "--model-size", "large_iit", 
        "--epochs", "10",
        "--patience", "5",
        "--seed", "42",  # Mismo seed para comparación justa
        "--lr", "1e-4",
        "--dropout", "0.25", 
        "--lambda-phi", "0.1",
        "--batch-size", "16",
        "--seq-len", "256"
    ]
    
    print(f"Comando IIT: {' '.join(cmd_iit)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd_iit,
                              capture_output=True,
                              text=True, 
                              cwd=os.getcwd())
        
        iit_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ IIT model completado en {iit_time/60:.1f} minutos")
            return True
        else:
            print(f"❌ Error en IIT model:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Excepción en IIT model: {e}")
        return False

def load_training_history(file_pattern):
    """Cargar historial de entrenamiento más reciente."""
    import glob
    
    files = glob.glob(f"results/training/*{file_pattern}*.json")
    if not files:
        return None
        
    # Obtener el más reciente
    latest_file = max(files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def compare_results():
    """Comparar resultados baseline vs IIT."""
    print("\n" + "="*80)
    print("📊 FASE 3: ANÁLISIS COMPARATIVO")
    print("="*80)
    
    # Cargar historiales
    baseline_history = load_training_history("baseline_no_iit")
    iit_history = load_training_history("training_history_real")
    
    if not baseline_history or not iit_history:
        print("❌ No se encontraron archivos de historial")
        return
    
    # Extraer métricas finales
    baseline_final_ppl = baseline_history['val_perplexity'][-1]
    iit_final_ppl = iit_history['val_perplexity'][-1]
    
    baseline_epochs = len(baseline_history['val_perplexity'])
    iit_epochs = len(iit_history['val_perplexity'])
    
    # Calcular mejora
    improvement_factor = baseline_final_ppl / iit_final_ppl
    improvement_percent = ((baseline_final_ppl - iit_final_ppl) / baseline_final_ppl) * 100
    
    # Resultados
    print("\n📊 RESULTADOS COMPARATIVOS:")
    print(f"  Baseline (SIN IIT):")
    print(f"    Final Val PPL: {baseline_final_ppl:.2f}")
    print(f"    Épocas entrenadas: {baseline_epochs}")
    
    print(f"  IIT Model (CON IIT):")
    print(f"    Final Val PPL: {iit_final_ppl:.2f}")
    print(f"    Épocas entrenadas: {iit_epochs}")
    
    print(f"\n🎯 IMPACTO DE IIT:")
    if improvement_factor > 1:
        print(f"  ✅ IIT MEJORA: {improvement_factor:.2f}x mejor ({improvement_percent:+.1f}%)")
    elif improvement_factor < 1:
        print(f"  ❌ IIT PERJUDICA: {1/improvement_factor:.2f}x peor ({improvement_percent:+.1f}%)")
    else:
        print(f"  ⚪ IIT SIN EFECTO: Similar performance")
    
    # Crear visualización
    create_comparison_plots(baseline_history, iit_history)
    
    # Guardar reporte
    save_scientific_report(baseline_history, iit_history, improvement_factor)

def create_comparison_plots(baseline_history, iit_history):
    """Crear gráficos comparativos."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Val PPL comparison
    epochs_baseline = range(1, len(baseline_history['val_perplexity']) + 1)
    epochs_iit = range(1, len(iit_history['val_perplexity']) + 1)
    
    ax1.plot(epochs_baseline, baseline_history['val_perplexity'], 
             'b-o', linewidth=2, label='Baseline (SIN IIT)')
    ax1.plot(epochs_iit, iit_history['val_perplexity'], 
             'r-o', linewidth=2, label='IIT Model (CON IIT)')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Validation Perplexity')
    ax1.set_title('Convergencia PPL: Baseline vs IIT')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Train PPL comparison  
    ax2.plot(epochs_baseline, baseline_history['train_perplexity'],
             'b-', alpha=0.7, label='Baseline Train')
    ax2.plot(epochs_iit, iit_history['train_perplexity'],
             'r-', alpha=0.7, label='IIT Train')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Train Perplexity')
    ax2.set_title('Training PPL: Baseline vs IIT')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Learning rate
    ax3.plot(epochs_baseline, baseline_history['learning_rate'],
             'b-', label='Baseline LR')
    ax3.plot(epochs_iit, iit_history['learning_rate'],
             'r-', label='IIT LR')
    ax3.set_xlabel('Épocas')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. IIT-specific metrics (if available)
    if 'train_phi' in iit_history:
        ax4.plot(epochs_iit, iit_history['train_phi'],
                 'g-o', label='Train Φ (IIT only)')
        ax4.set_xlabel('Épocas')
        ax4.set_ylabel('Φ (Consciousness)')
        ax4.set_title('IIT Consciousness Metric')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'PHI metrics\nnot available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('IIT Φ Metrics (No Data)')
    
    plt.tight_layout()
    
    # Guardar gráfico
    plot_path = f'results/analysis/baseline_vs_iit_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    os.makedirs('results/analysis', exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Gráficos guardados: {plot_path}")
    plt.close()

def save_scientific_report(baseline_history, iit_history, improvement_factor):
    """Guardar reporte científico."""
    report = {
        "experiment_info": {
            "date": datetime.now().isoformat(),
            "purpose": "Scientific validation of IIT features impact",
            "methodology": "Controlled comparison with identical hyperparameters"
        },
        "model_configurations": {
            "baseline": {
                "architecture": "Standard Transformer (4 layers, 512 hidden, 8 heads)",
                "features": "NO IIT (no consciousness features)",
                "parameters": "~37M"
            },
            "iit_model": {
                "architecture": "INFINITO V5.2 (4 layers, 512 hidden, 8 heads)",
                "features": "IITGuidedMemory, LearnablePhiWeights, ImprovedIITMetrics, StochasticExploration",
                "parameters": "~37M + IIT overhead"
            }
        },
        "training_setup": {
            "dataset": "WikiText-2 real",
            "tokenizer": "GPT-2 (50,257 vocab)",
            "seed": 42,
            "hyperparameters": {
                "learning_rate": 1e-4,
                "dropout": 0.25,
                "batch_size": 16,
                "sequence_length": 256,
                "epochs": "up to 10 (early stopping)"
            }
        },
        "results": {
            "baseline": {
                "final_val_ppl": baseline_history['val_perplexity'][-1],
                "final_train_ppl": baseline_history['train_perplexity'][-1],
                "epochs_trained": len(baseline_history['val_perplexity']),
                "convergence": "stable" if len(baseline_history['val_perplexity']) < 10 else "early_stopped"
            },
            "iit_model": {
                "final_val_ppl": iit_history['val_perplexity'][-1],
                "final_train_ppl": iit_history['train_perplexity'][-1],
                "epochs_trained": len(iit_history['val_perplexity']),
                "convergence": "stable" if len(iit_history['val_perplexity']) < 10 else "early_stopped",
                "phi_metrics": iit_history.get('train_phi', [])
            }
        },
        "analysis": {
            "improvement_factor": improvement_factor,
            "improvement_percentage": ((baseline_history['val_perplexity'][-1] - iit_history['val_perplexity'][-1]) / baseline_history['val_perplexity'][-1]) * 100,
            "conclusion": "IIT features improve performance" if improvement_factor > 1.05 else 
                         "IIT features harm performance" if improvement_factor < 0.95 else
                         "IIT features have neutral impact"
        }
    }
    
    # Guardar reporte
    report_path = f'results/analysis/scientific_validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Reporte científico guardado: {report_path}")

def test_generation_quality():
    """Comparar calidad de generación entre modelos."""
    print("\n" + "="*80)
    print("📝 FASE 4: COMPARACIÓN DE GENERACIÓN DE TEXTO")
    print("="*80)
    
    # Esta fase requeriría cargar ambos modelos y comparar generación
    # Por ahora solo documentamos el plan
    print("🔄 TODO: Implementar comparación de generación de texto")
    print("  - Cargar checkpoint baseline y IIT")
    print("  - Generar texto con mismos prompts")
    print("  - Comparar diversidad y coherencia")
    print("  - Aplicar métricas Dist-1/2/3")

def main():
    """Ejecutar validación científica completa."""
    print("🔬 VALIDACIÓN CIENTÍFICA: BASELINE vs IIT MODEL")
    print("=" * 80)
    print("Objetivo: Cuantificar el impacto real de características IIT")
    print("Método: Entrenamiento controlado con configuración idéntica")
    print("=" * 80)
    
    # FASE 1: Entrenar baseline
    if not run_baseline_training():
        print("❌ Error en entrenamiento baseline, abortando")
        return False
    
    # FASE 2: Entrenar IIT model  
    if not run_iit_training():
        print("❌ Error en entrenamiento IIT, abortando")
        return False
    
    # FASE 3: Comparar resultados
    compare_results()
    
    # FASE 4: Comparar generación
    test_generation_quality()
    
    print("\n" + "="*80)
    print("✅ VALIDACIÓN CIENTÍFICA COMPLETADA")
    print("="*80)
    print("📊 Revisar: results/analysis/ para gráficos y reportes")
    print("🔬 Conclusiones: Impacto cuantificado de características IIT")
    
    return True

if __name__ == '__main__':
    main()