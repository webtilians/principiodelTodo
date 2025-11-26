#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 LANZADOR - Entrenamiento RL v2 con Reward Mejorada
====================================================

Script automatizado para entrenar el agente RL con la reward function v2 mejorada.
Incluye validación pre-entrenamiento y monitoreo.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path


def check_prerequisites():
    """Verificar que todo está listo para entrenar."""
    print("="*70)
    print("🔍 VERIFICANDO PRERREQUISITOS")
    print("="*70)
    
    issues = []
    
    # 1. GPU disponible
    print("\n1. Verificando GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   ✅ GPU disponible: {gpu_name}")
            print(f"   ✅ Memoria: {gpu_memory:.1f} GB")
        else:
            print("   ⚠️ GPU no disponible - usará CPU (MUY LENTO)")
            issues.append("GPU no disponible")
    except Exception as e:
        print(f"   ❌ Error verificando GPU: {e}")
        issues.append("Error GPU")
    
    # 2. Espacio en disco
    print("\n2. Verificando espacio en disco...")
    try:
        import shutil
        disk = shutil.disk_usage(".")
        free_gb = disk.free / (1024**3)
        if free_gb > 5:
            print(f"   ✅ Espacio libre: {free_gb:.1f} GB")
        else:
            print(f"   ⚠️ Espacio libre: {free_gb:.1f} GB (recomendado >5GB)")
            issues.append("Poco espacio en disco")
    except Exception as e:
        print(f"   ❌ Error verificando disco: {e}")
    
    # 3. Dependencias
    print("\n3. Verificando dependencias...")
    deps = ['gymnasium', 'stable_baselines3', 'tensorboard', 'torch', 'transformers']
    for dep in deps:
        try:
            __import__(dep.replace('-', '_'))
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} NO INSTALADO")
            issues.append(f"Falta {dep}")
    
    # 4. Archivos necesarios
    print("\n4. Verificando archivos...")
    files = [
        'src/rl/infinito_rl_env.py',
        'experiments/train_phi_text_scheduler.py',
        'train_v5_2_gpt2_lora.py'
    ]
    for f in files:
        if Path(f).exists():
            print(f"   ✅ {f}")
        else:
            print(f"   ❌ {f} NO ENCONTRADO")
            issues.append(f"Falta {f}")
    
    # 5. Reward function v2
    print("\n5. Verificando reward function v2...")
    try:
        with open('src/rl/infinito_rl_env.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'stability_penalty' in content and 'phi_balance' in content:
                print("   ✅ Reward v2 detectada")
            else:
                print("   ⚠️ Reward v2 no detectada - puede ser v1")
                issues.append("Reward v1 en lugar de v2")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    if issues:
        print("⚠️ PROBLEMAS DETECTADOS:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n¿Continuar de todas formas? (y/n): ", end='')
        response = input().lower()
        return response == 'y'
    else:
        print("✅ TODOS LOS PRERREQUISITOS CUMPLIDOS")
        return True


def run_training(timesteps=50000, inner_steps=5, max_steps=50):
    """Ejecutar entrenamiento RL."""
    print("\n" + "="*70)
    print("🚀 INICIANDO ENTRENAMIENTO RL v2")
    print("="*70)
    
    # Configuración
    config = {
        "timesteps": timesteps,
        "inner_steps": inner_steps,
        "max_steps": max_steps,
        "learning_rate": 3e-4,
        "version": "v2",
        "start_time": datetime.now().isoformat()
    }
    
    print("\n📋 CONFIGURACIÓN:")
    print(f"   Total timesteps: {timesteps:,}")
    print(f"   Inner steps: {inner_steps}")
    print(f"   Max steps/episodio: {max_steps}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Reward function: v2 (mejorada)")
    
    # Duración estimada
    hours_v1 = 2.78  # 10K timesteps tomó 2.78 horas
    estimated_hours = (timesteps / 10000) * hours_v1
    print(f"\n⏱️ Duración estimada: {estimated_hours:.1f} horas")
    
    # Guardar configuración
    config_path = Path("outputs/rl_phi_text_scheduler_v2")
    config_path.mkdir(parents=True, exist_ok=True)
    with open(config_path / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*70)
    print("COMENZANDO ENTRENAMIENTO...")
    print("="*70)
    print("\n💡 TIPS:")
    print("   - Monitorear con: tensorboard --logdir outputs/rl_phi_text_scheduler/tensorboard")
    print("   - Ver checkpoints: outputs/rl_phi_text_scheduler/checkpoints/")
    print("   - Logs: outputs/rl_phi_text_scheduler/logs/")
    print("\n   Presiona Ctrl+C para detener (guardará checkpoint actual)")
    print("\n" + "="*70 + "\n")
    
    # Comando
    cmd = [
        sys.executable,
        "experiments/train_phi_text_scheduler.py",
        "--timesteps", str(timesteps),
        "--inner-steps", str(inner_steps),
        "--max-steps", str(max_steps),
        "--lr", str(config['learning_rate'])
    ]
    
    try:
        # Ejecutar
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "="*70)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("="*70)
        
        # Guardar resultado
        config["end_time"] = datetime.now().isoformat()
        config["status"] = "completed"
        with open(config_path / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⚠️ ENTRENAMIENTO INTERRUMPIDO POR USUARIO")
        print("="*70)
        print("\n📦 Checkpoint guardado automáticamente")
        
        config["end_time"] = datetime.now().isoformat()
        config["status"] = "interrupted"
        with open(config_path / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        return False
        
    except subprocess.CalledProcessError as e:
        print("\n\n" + "="*70)
        print(f"❌ ERROR DURANTE ENTRENAMIENTO: {e}")
        print("="*70)
        
        config["end_time"] = datetime.now().isoformat()
        config["status"] = "error"
        config["error"] = str(e)
        with open(config_path / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        return False


def show_summary():
    """Mostrar resumen de lo que se va a hacer."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           🚀 ENTRENAMIENTO RL v2 - REWARD MEJORADA              ║
╚══════════════════════════════════════════════════════════════════╝

📊 MEJORAS DE REWARD v2:
  ✅ Detecta colapso PHI > 6.0 (previene Fase 2)
  ✅ Detecta colapso PPL < 10 (repeticiones)
  ✅ Penaliza inestabilidad (|ΔΦ| > 1.0)
  ✅ Incentiva rangos óptimos con bonuses

🎯 CONFIGURACIÓN:
  • Timesteps: 50,000 (5× más que v1)
  • Inner steps: 5 (más training/acción)
  • Max steps: 50 (episodios más largos)
  • Learning rate: 3e-4

📈 RESULTADOS ESPERADOS:
  • Recompensa final: +0.15 a +0.25 (vs -0.017 en v1)
  • Uso modo MIXED: 20-30% (vs 0% en v1)
  • PHI estable: 3.5-5.5 (sin colapso)
  • Convergencia: Completa

⏱️ DURACIÓN: ~14 horas

""")


def main():
    """Función principal."""
    show_summary()
    
    print("¿Deseas continuar con el entrenamiento? (y/n): ", end='')
    response = input().lower()
    
    if response != 'y':
        print("\n❌ Entrenamiento cancelado por usuario")
        return
    
    # Verificar prerrequisitos
    if not check_prerequisites():
        print("\n❌ Abortando por problemas en prerrequisitos")
        return
    
    # Entrenar
    success = run_training(timesteps=50000, inner_steps=5, max_steps=50)
    
    if success:
        print("\n" + "="*70)
        print("🎉 SIGUIENTE PASO:")
        print("="*70)
        print("\n1. Analizar resultados:")
        print("   python analyze_rl_results.py")
        print("\n2. Ver visualizaciones:")
        print("   python experiments/run_infinito_with_scheduler.py --episodes 5")
        print("\n3. Comparar con v1:")
        print("   - v1 (10K): reward=-0.017, MIXED=0%")
        print("   - v2 (50K): reward=?, MIXED=?%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
