#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 DIAGNÓSTICO DE ARCHIVOS DE HISTORIAL
======================================

Script para diagnosticar problemas en los archivos de historial
que pueden estar causando errores en el dashboard.
"""

import json
import os
from pathlib import Path

def analyze_training_files():
    """Analiza archivos de entrenamiento para diagnosticar problemas."""
    print("🔍 DIAGNÓSTICO DE ARCHIVOS DE HISTORIAL")
    print("="*50)
    
    # Buscar archivos de historial
    patterns = [
        "training_history*.json", 
        "*history*.json", 
        "*results*.json",
        "results/training/*.json"
    ]
    
    all_files = []
    for pattern in patterns:
        files = list(Path(".").glob(pattern))
        all_files.extend(files)
    
    print(f"\n📁 Encontrados {len(all_files)} archivos:")
    
    for file_path in all_files:
        print(f"\n📄 {file_path}")
        print(f"   Tamaño: {file_path.stat().st_size / 1024:.1f} KB")
        print(f"   Modificado: {file_path.stat().st_mtime}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"   ✅ JSON válido")
            print(f"   📊 Tipo de dato raíz: {type(data).__name__}")
            
            if isinstance(data, dict):
                print(f"   🔑 Claves principales: {list(data.keys())}")
                
                # Buscar datos de entrenamiento
                training_keys = []
                for key in data.keys():
                    if 'loss' in key.lower() or 'history' in key.lower():
                        training_keys.append(key)
                
                if training_keys:
                    print(f"   📈 Claves de entrenamiento: {training_keys}")
                    
                    # Analizar estructura de métricas
                    for key in training_keys:
                        value = data[key]
                        if isinstance(value, dict):
                            print(f"     - {key}: dict con {list(value.keys())}")
                        elif isinstance(value, list):
                            print(f"     - {key}: lista con {len(value)} elementos")
                        else:
                            print(f"     - {key}: {type(value).__name__}")
                else:
                    print(f"   ⚠️  No se encontraron claves de entrenamiento obvias")
            
            elif isinstance(data, list):
                print(f"   📋 Lista con {len(data)} elementos")
                if len(data) > 0:
                    print(f"   📊 Tipo del primer elemento: {type(data[0]).__name__}")
                    if isinstance(data[0], dict):
                        print(f"   🔑 Claves del primer elemento: {list(data[0].keys())}")
                        
        except json.JSONDecodeError as e:
            print(f"   ❌ Error JSON: {e}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def check_current_training():
    """Verifica si hay un entrenamiento activo."""
    print(f"\n🔄 VERIFICANDO ENTRENAMIENTO ACTIVO")
    print("="*40)
    
    # Buscar archivos de log recientes
    current_time = time.time()
    recent_threshold = 3600  # 1 hora
    
    log_patterns = ["*.log", "*output*.txt", "entrenamiento*.txt"]
    recent_files = []
    
    for pattern in log_patterns:
        for file_path in Path(".").glob(pattern):
            if current_time - file_path.stat().st_mtime < recent_threshold:
                recent_files.append(file_path)
    
    if recent_files:
        print(f"📄 Archivos de log recientes encontrados:")
        for file_path in recent_files:
            print(f"   - {file_path}")
    else:
        print("📄 No hay archivos de log recientes")
    
    # Verificar procesos Python activos
    try:
        import psutil
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                    if 'train' in cmdline.lower():
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': cmdline
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if python_processes:
            print(f"\n🐍 Procesos Python de entrenamiento activos:")
            for proc in python_processes:
                print(f"   PID {proc['pid']}: {proc['cmdline'][:100]}...")
        else:
            print("🐍 No hay procesos de entrenamiento activos detectados")
            
    except ImportError:
        print("⚠️  psutil no disponible - no se pueden verificar procesos")

if __name__ == "__main__":
    import time
    
    try:
        analyze_training_files()
        check_current_training()
        
        print(f"\n✅ DIAGNÓSTICO COMPLETADO")
        print("="*30)
        print("💡 Si el dashboard sigue fallando:")
        print("   1. Verifica que los archivos JSON tengan formato correcto")
        print("   2. Reinicia el dashboard: Ctrl+C y python launch_dashboard.py")
        print("   3. Verifica que el entrenamiento esté generando archivos de historial")
        
    except Exception as e:
        print(f"\n❌ Error durante diagnóstico: {e}")
        import traceback
        traceback.print_exc()