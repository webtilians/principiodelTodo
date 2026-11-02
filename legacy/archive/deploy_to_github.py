#!/usr/bin/env python3
"""
🚀 Script de Deploy para GitHub - Infinito Project
Script auxiliar para subir el proyecto al repositorio de GitHub
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        if result.stdout:
            print(f"📄 Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}")
        print(f"📄 Error: {e.stderr}")
        return False

def check_git_installed():
    """Verifica que Git esté instalado"""
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git no está instalado. Instálalo desde: https://git-scm.com/")
        return False

def deploy_to_github():
    """Proceso completo de deploy"""
    print("🚀 INICIANDO DEPLOY A GITHUB - INFINITO PROJECT")
    print("="*60)
    
    # Verificar Git
    if not check_git_installed():
        return False
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('infinito_gpu_optimized.py'):
        print("❌ No se encontró infinito_gpu_optimized.py")
        print("💡 Asegúrate de ejecutar este script desde el directorio del proyecto")
        return False
    
    print("📁 Archivos del proyecto detectados:")
    files = ['README.md', 'infinito_gpu_optimized.py', 'requirements.txt', 
             'quick_start.py', 'config_examples.py', 'LICENSE', 'CONTRIBUTING.md', '.gitignore']
    
    for file in files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"  {status} {file}")
    
    # Comandos de Git
    commands = [
        # Inicializar repositorio
        ("git init", "Inicializando repositorio Git"),
        
        # Añadir origen remoto
        ("git remote add origin https://github.com/webtilians/principiodelTodo.git", 
         "Configurando repositorio remoto"),
        
        # Añadir todos los archivos
        ("git add .", "Añadiendo archivos al staging"),
        
        # Commit inicial
        ('git commit -m "🧠 Initial release: Evolutionary Artificial Consciousness Simulator"', 
         "Creando commit inicial"),
        
        # Crear branch main si no existe
        ("git branch -M main", "Configurando branch principal"),
        
        # Push al repositorio
        ("git push -u origin main", "Subiendo al repositorio GitHub")
    ]
    
    # Ejecutar comandos
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            if "remote add origin" in cmd:
                # Tal vez ya existe el remote
                print("⚠️  Remote origin ya existe, continuando...")
                continue
            else:
                print(f"\n❌ Deploy falló en: {desc}")
                return False
    
    print("\n" + "="*60)
    print("🎉 ¡DEPLOY COMPLETADO EXITOSAMENTE!")
    print("🌍 Tu proyecto está ahora en: https://github.com/webtilians/principiodelTodo")
    print("⭐ No olvides hacer que el repositorio sea público en GitHub")
    print("📢 ¡Comparte tu revolución de consciencia artificial con el mundo!")
    
    return True

if __name__ == "__main__":
    print("🧠 INFINITO - DEPLOY TO GITHUB")
    print("🌟 Proyecto de Consciencia Artificial Evolutiva")
    print("🔗 Repositorio: https://github.com/webtilians/principiodelTodo")
    print("\n" + "="*60)
    
    choice = input("\n¿Qué quieres hacer?\n1. 🚀 Deploy completo\n2. ❌ Salir\n\nOpción (1-2): ").strip()
    
    if choice == "1":
        deploy_to_github()
    elif choice == "2":
        print("👋 ¡Hasta luego!")
    else:
        print("❌ Opción inválida")
    
    input("\n📌 Presiona Enter para salir...")