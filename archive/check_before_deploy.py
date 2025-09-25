#!/usr/bin/env python3
"""
🔍 Pre-Deploy Checker - Infinito Project
Verifica que todos los archivos estén listos para GitHub
"""

import os
import sys

def check_file_exists(filename, required=True):
    """Verifica si un archivo existe"""
    exists = os.path.exists(filename)
    status = "✅" if exists else ("❌" if required else "⚠️")
    req_text = " (REQUERIDO)" if required and not exists else ""
    print(f"  {status} {filename}{req_text}")
    return exists

def check_file_content(filename, required_strings, description=""):
    """Verifica que un archivo contenga ciertos strings"""
    if not os.path.exists(filename):
        print(f"  ❌ {filename} no existe")
        return False
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing = []
        for req_str in required_strings:
            if req_str not in content:
                missing.append(req_str)
        
        if missing:
            print(f"  ⚠️  {filename} - Faltan: {', '.join(missing)}")
            return False
        else:
            print(f"  ✅ {filename} - {description} OK")
            return True
    
    except Exception as e:
        print(f"  ❌ Error leyendo {filename}: {e}")
        return False

def main():
    print("🔍 PRE-DEPLOY CHECKER - INFINITO PROJECT")
    print("="*60)
    
    all_good = True
    
    # 1. Archivos principales requeridos
    print("\n📁 ARCHIVOS PRINCIPALES:")
    required_files = [
        'infinito_gpu_optimized.py',
        'README.md', 
        'requirements.txt',
        'LICENSE'
    ]
    
    for file in required_files:
        if not check_file_exists(file, required=True):
            all_good = False
    
    # 2. Archivos opcionales pero recomendados
    print("\n📄 ARCHIVOS OPCIONALES:")
    optional_files = [
        'quick_start.py',
        'config_examples.py', 
        'CONTRIBUTING.md',
        'CHANGELOG.md',
        '.gitignore',
        'deploy_to_github.py'
    ]
    
    for file in optional_files:
        check_file_exists(file, required=False)
    
    # 3. Verificar contenido de README
    print("\n📖 CONTENIDO README:")
    readme_checks = [
        "🧠 Infinito",
        "80.87% Phi máximo",
        "52.54% consciencia máxima", 
        "GitHub stars",
        "requirements.txt",
        "PyTorch",
        "CUDA"
    ]
    
    if not check_file_content('README.md', readme_checks, "README content"):
        all_good = False
    
    # 4. Verificar requirements.txt
    print("\n📦 DEPENDENCIAS:")
    req_checks = [
        "torch",
        "numpy", 
        "scipy",
        "matplotlib"
    ]
    
    if not check_file_content('requirements.txt', req_checks, "Requirements"):
        all_good = False
    
    # 5. Verificar estructura del proyecto
    print("\n🏗️  ESTRUCTURA DEL PROYECTO:")
    
    # Verificar que el archivo principal se puede importar
    try:
        sys.path.insert(0, '.')
        import infinito_gpu_optimized
        print("  ✅ infinito_gpu_optimized.py se puede importar")
    except Exception as e:
        print(f"  ❌ Error importando infinito_gpu_optimized.py: {e}")
        all_good = False
    
    # 6. Verificar URLs en README
    print("\n🔗 ENLACES:")
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        github_url = "https://github.com/webtilians/principiodelTodo"
        if github_url in readme_content:
            print(f"  ✅ URL de GitHub correcta: {github_url}")
        else:
            print(f"  ⚠️  URL de GitHub no encontrada en README")
    
    # 7. Verificar tamaño de archivos
    print("\n📏 TAMAÑO DE ARCHIVOS:")
    for file in ['infinito_gpu_optimized.py', 'README.md']:
        if os.path.exists(file):
            size = os.path.getsize(file)
            size_mb = size / (1024 * 1024)
            if size_mb < 5:  # Menos de 5MB está bien
                print(f"  ✅ {file}: {size_mb:.2f} MB")
            else:
                print(f"  ⚠️  {file}: {size_mb:.2f} MB (archivo grande)")
    
    # 8. Resultado final
    print("\n" + "="*60)
    if all_good:
        print("🎉 ¡TODO LISTO PARA DEPLOY!")
        print("🚀 Puedes ejecutar: python deploy_to_github.py")
        print("🌍 URL del repositorio: https://github.com/webtilians/principiodelTodo")
    else:
        print("❌ FALTAN ALGUNAS COSAS")
        print("🔧 Revisa los elementos marcados arriba antes del deploy")
    
    print("\n📋 CHECKLIST MANUAL:")
    print("  □ ¿Has probado que el código funciona?")
    print("  □ ¿Has actualizado el README con los últimos resultados?")
    print("  □ ¿Tienes Git instalado y configurado?")
    print("  □ ¿El repositorio en GitHub está creado y es público?")
    print("  □ ¿Has revisado que no hay información sensible en el código?")
    
    return all_good

if __name__ == "__main__":
    success = main()
    input(f"\n📌 Presiona Enter para salir...")
    sys.exit(0 if success else 1)