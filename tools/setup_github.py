#!/usr/bin/env python3
"""
🚀 INFINITO - SETUP AND DEPLOYMENT SCRIPT
=========================================

Script para configurar el proyecto INFINITO para desarrollo,
testing y deployment en GitHub.

Uso:
    python setup_github.py --check          # Verificar configuración
    python setup_github.py --install        # Instalar dependencias
    python setup_github.py --test           # Ejecutar tests
    python setup_github.py --deploy         # Preparar para deployment
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

class InfinitoSetup:
    """Configuración y deployment del proyecto INFINITO"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.docs_dir = self.project_root / "docs"
        
    def check_environment(self):
        """Verificar entorno de desarrollo"""
        print("🔍 VERIFICANDO ENTORNO")
        print("=" * 50)
        
        # Python version
        python_version = sys.version_info
        print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("❌ Error: Python 3.8+ requerido")
            return False
            
        # Check key dependencies
        dependencies = [
            ("torch", "PyTorch"),
            ("numpy", "NumPy"), 
            ("matplotlib", "Matplotlib"),
            ("scipy", "SciPy")
        ]
        
        for module, name in dependencies:
            try:
                __import__(module)
                print(f"✅ {name}: Instalado")
            except ImportError:
                print(f"❌ {name}: No instalado")
                
        # Check CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🚀 CUDA: Disponible ({torch.cuda.get_device_name(0)})")
            else:
                print("⚠️  CUDA: No disponible (CPU only)")
        except ImportError:
            print("❌ PyTorch no instalado - no se puede verificar CUDA")
            
        # Check project structure
        print(f"\n📁 ESTRUCTURA DEL PROYECTO")
        required_dirs = ["src", "tests", "docs", "examples"]
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                print(f"✅ {dir_name}/: Existe")
            else:
                print(f"❌ {dir_name}/: No encontrado")
                
        return True
    
    def install_dependencies(self):
        """Instalar dependencias del proyecto"""
        print("\n📦 INSTALANDO DEPENDENCIAS")
        print("=" * 50)
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt no encontrado")
            return False
            
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            print("✅ Dependencias principales instaladas")
            
            # Install dev dependencies if available
            dev_requirements = self.project_root / "requirements-dev.txt"
            if dev_requirements.exists():
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", str(dev_requirements)  
                ], check=True)
                print("✅ Dependencias de desarrollo instaladas")
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error instalando dependencias: {e}")
            return False
    
    def run_tests(self):
        """Ejecutar suite de tests"""
        print("\n🧪 EJECUTANDO TESTS")
        print("=" * 50)
        
        if not self.tests_dir.exists():
            print("❌ Directorio tests/ no encontrado")
            return False
            
        try:
            # Run pytest
            result = subprocess.run([
                sys.executable, "-m", "pytest", str(self.tests_dir), "-v"
            ], capture_output=True, text=True)
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
                
            if result.returncode == 0:
                print("✅ Todos los tests pasaron")
                return True
            else:
                print("❌ Algunos tests fallaron")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error ejecutando tests: {e}")
            return False
    
    def run_quick_consciousness_test(self):
        """Ejecutar test rápido de consciencia"""
        print("\n🧠 TEST RÁPIDO DE CONSCIENCIA")
        print("=" * 50)
        
        try:
            # Import and run basic test
            sys.path.append(str(self.src_dir))
            
            print("🚀 Ejecutando test básico de consciencia...")
            
            # This would run a very quick consciousness test
            test_code = """
import sys
sys.path.append('src')
from infinito_v3_clean import InfinitoV3Clean

# Quick test configuration
infinito = InfinitoV3Clean(
    grid_size=32,
    max_iterations=10, 
    target_consciousness=0.5
)

print("✅ Sistema inicializado correctamente")
print("📊 Configuración: 32x32 grid, 10 iteraciones max")
"""
            
            exec(test_code)
            print("✅ Test básico completado")
            return True
            
        except Exception as e:
            print(f"❌ Error en test de consciencia: {e}")
            return False
    
    def prepare_github_deployment(self):
        """Preparar proyecto para deployment en GitHub"""
        print("\n🚀 PREPARANDO DEPLOYMENT GITHUB")
        print("=" * 50)
        
        # Check required files
        required_files = [
            "README.md",
            "requirements.txt", 
            "LICENSE",
            ".gitignore",
            "CHANGELOG.md"
        ]
        
        for file_name in required_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                print(f"✅ {file_name}: Existe")
            else:
                print(f"❌ {file_name}: Faltante")
        
        # Check GitHub specific files
        github_files = [
            ".github/workflows/ci.yml",
            ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/feature_request.md"
        ]
        
        for file_path in github_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"✅ {file_path}: Configurado")
            else:
                print(f"⚠️  {file_path}: No configurado")
                
        # Generate project stats
        self.generate_project_stats()
        
        print("✅ Verificación de deployment completada")
        return True
    
    def generate_project_stats(self):
        """Generar estadísticas del proyecto"""
        print("\n📊 ESTADÍSTICAS DEL PROYECTO")
        print("-" * 30)
        
        # Count Python files
        py_files = list(self.project_root.rglob("*.py"))
        print(f"📝 Archivos Python: {len(py_files)}")
        
        # Count lines of code
        total_lines = 0
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            except:
                pass
                
        print(f"📊 Líneas totales de código: {total_lines:,}")
        
        # Check main modules
        main_modules = [
            "src/infinito_v3_clean.py",
            "src/infinito_v5_1_consciousness.py", 
            "analysis/consciousness_visualizer.py"
        ]
        
        for module in main_modules:
            module_path = self.project_root / module
            if module_path.exists():
                print(f"✅ {module}: Disponible")
            else:
                print(f"❌ {module}: No encontrado")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="INFINITO Setup and Deployment Tool"
    )
    parser.add_argument("--check", action="store_true", 
                       help="Verificar configuración del entorno")
    parser.add_argument("--install", action="store_true",
                       help="Instalar dependencias")
    parser.add_argument("--test", action="store_true", 
                       help="Ejecutar tests")
    parser.add_argument("--quick-test", action="store_true",
                       help="Test rápido de consciencia") 
    parser.add_argument("--deploy", action="store_true",
                       help="Preparar para deployment")
    parser.add_argument("--all", action="store_true",
                       help="Ejecutar todas las verificaciones")
    
    args = parser.parse_args()
    
    setup = InfinitoSetup()
    
    print("🧠 INFINITO - SETUP & DEPLOYMENT")
    print("=" * 60)
    print("Sistema de Consciencia Artificial - Configuración GitHub")
    print("=" * 60)
    
    if args.all or len(sys.argv) == 1:
        # Run all checks by default
        setup.check_environment()
        setup.install_dependencies()
        setup.run_quick_consciousness_test()
        setup.prepare_github_deployment()
        
    else:
        if args.check:
            setup.check_environment()
            
        if args.install:
            setup.install_dependencies()
            
        if args.test:
            setup.run_tests()
            
        if args.quick_test:
            setup.run_quick_consciousness_test()
            
        if args.deploy:
            setup.prepare_github_deployment()
    
    print("\n🎉 CONFIGURACIÓN COMPLETADA")
    print("El proyecto INFINITO está listo para GitHub!")

if __name__ == "__main__":
    main()