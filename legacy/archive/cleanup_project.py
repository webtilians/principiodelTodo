#!/usr/bin/env python3
"""
Script de limpieza inteligente para el proyecto INFINITO
Elimina archivos antiguos y innecesarios de forma segura
"""

import os
import glob
import shutil
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_project():
    """Limpia archivos antiguos del proyecto de forma inteligente"""
    
    base_path = Path(".")
    total_freed = 0
    
    print("🧹 LIMPIEZA INTELIGENTE DEL PROYECTO INFINITO")
    print("=" * 60)
    
    # 1. Limpiar checkpoints antiguos (mantener solo los más recientes)
    print("🔍 Limpiando checkpoints antiguos...")
    checkpoint_files = list(glob.glob("**/*checkpoint*.pt", recursive=True))
    breakthrough_files = list(glob.glob("**/ULTIMATE_BREAKTHROUGH*.pt", recursive=True))
    consciousness_files = list(glob.glob("**/CONSCIOUSNESS_BREAKTHROUGH*.pt", recursive=True))
    interrupted_files = list(glob.glob("**/interrupted_state*.pt", recursive=True))
    
    # Mantener solo los 5 checkpoints más recientes
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    old_checkpoints = checkpoint_files[5:]  # Todos excepto los 5 más nuevos
    
    for file_path in old_checkpoints:
        try:
            size = os.path.getsize(file_path)
            os.remove(file_path)
            total_freed += size
            print(f"  ❌ Eliminado: {file_path} ({size/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"  ⚠️ Error eliminando {file_path}: {e}")
    
    # Mantener solo los 10 breakthrough más recientes
    breakthrough_files.sort(key=os.path.getmtime, reverse=True)
    old_breakthroughs = breakthrough_files[10:]
    
    for file_path in old_breakthroughs:
        try:
            size = os.path.getsize(file_path)
            os.remove(file_path)
            total_freed += size
            print(f"  ❌ Eliminado: {file_path} ({size/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"  ⚠️ Error eliminando {file_path}: {e}")
    
    # Mantener solo los 5 consciousness breakthrough más recientes
    consciousness_files.sort(key=os.path.getmtime, reverse=True)
    old_consciousness = consciousness_files[5:]
    
    for file_path in old_consciousness:
        try:
            size = os.path.getsize(file_path)
            os.remove(file_path)
            total_freed += size
            print(f"  ❌ Eliminado: {file_path} ({size/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"  ⚠️ Error eliminando {file_path}: {e}")
    
    # Eliminar todos los estados interrumpidos excepto el más reciente
    interrupted_files.sort(key=os.path.getmtime, reverse=True)
    old_interrupted = interrupted_files[1:]  # Mantener solo el más reciente
    
    for file_path in old_interrupted:
        try:
            size = os.path.getsize(file_path)
            os.remove(file_path)
            total_freed += size
            print(f"  ❌ Eliminado: {file_path} ({size/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"  ⚠️ Error eliminando {file_path}: {e}")
    
    # 2. Limpiar imágenes y GIFs antiguos
    print("\n🖼️ Limpiando imágenes antiguas...")
    image_files = list(glob.glob("**/*.png", recursive=True)) + \
                  list(glob.glob("**/*.jpg", recursive=True)) + \
                  list(glob.glob("**/*.gif", recursive=True))
    
    # Mantener solo las imágenes de los últimos 7 días
    cutoff_date = datetime.now() - timedelta(days=7)
    
    for file_path in image_files:
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_time < cutoff_date:
                size = os.path.getsize(file_path)
                os.remove(file_path)
                total_freed += size
                print(f"  ❌ Eliminado: {file_path} ({size/1024/1024:.1f} MB)")
        except Exception as e:
            print(f"  ⚠️ Error eliminando {file_path}: {e}")
    
    # 3. Limpiar archivos JSON de experimentos antiguos
    print("\n📊 Limpiando datos de experimentos antiguos...")
    json_files = list(glob.glob("**/experiment_data/*.json", recursive=True)) + \
                 list(glob.glob("**/outputs/*.json", recursive=True))
    
    # Mantener solo los JSONs de experimentos de los últimos 3 días
    cutoff_date = datetime.now() - timedelta(days=3)
    
    for file_path in json_files:
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_time < cutoff_date:
                size = os.path.getsize(file_path)
                os.remove(file_path)
                total_freed += size
                print(f"  ❌ Eliminado: {file_path} ({size/1024:.1f} KB)")
        except Exception as e:
            print(f"  ⚠️ Error eliminando {file_path}: {e}")
    
    # 4. Limpiar logs antiguos
    print("\n📝 Limpiando logs antiguos...")
    log_files = list(glob.glob("**/*.log", recursive=True))
    
    # Mantener solo logs del último día
    cutoff_date = datetime.now() - timedelta(days=1)
    
    for file_path in log_files:
        try:
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_time < cutoff_date:
                size = os.path.getsize(file_path)
                os.remove(file_path)
                total_freed += size
                print(f"  ❌ Eliminado: {file_path} ({size/1024:.1f} KB)")
        except Exception as e:
            print(f"  ⚠️ Error eliminando {file_path}: {e}")
    
    # 5. Limpiar archivos temporales
    print("\n🗑️ Limpiando archivos temporales...")
    temp_patterns = ["**/*.tmp", "**/*.temp", "**/__pycache__/**", "**/*.pyc"]
    
    for pattern in temp_patterns:
        temp_files = glob.glob(pattern, recursive=True)
        for file_path in temp_files:
            try:
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    total_freed += size
                    print(f"  ❌ Eliminado: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"  📁 Eliminado directorio: {file_path}")
            except Exception as e:
                print(f"  ⚠️ Error eliminando {file_path}: {e}")
    
    print("\n" + "=" * 60)
    print(f"✅ LIMPIEZA COMPLETADA")
    print(f"💾 Espacio liberado: {total_freed/1024/1024/1024:.2f} GB")
    print(f"📈 Esto debería resolver los problemas de memoria y espacio")
    
    return total_freed

if __name__ == "__main__":
    cleanup_project()