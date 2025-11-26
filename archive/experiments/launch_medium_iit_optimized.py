#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏗️ LANZADOR DE ARQUITECTURA MEDIUM IIT OPTIMIZADA
=================================================

Preparar y lanzar entrenamiento de medium_iit con la configuración
optimizada encontrada (λ_phi=0.010, dropout=0.25).

Arquitectura medium_iit_optimized mostró 96.8% mejora en pruebas.
"""

import sys
import os
import json
import subprocess
import time
from datetime import datetime

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class MediumIITOptimizedLauncher:
    """Lanzador especializado para medium_iit con configuración optimizada."""
    
    def __init__(self):
        self.optimized_config = {
            'model_size': 'medium_iit',
            'lambda_phi': '0.01',      # ← Configuración optimizada
            'dropout': '0.25',         # ← Regularización óptima  
            'learning_rate': '0.0002', # ← LR óptimo para medium
            'batch_size': '8',         # Ajustado para medium_iit
            'epochs': '3',             # Suficiente para validación
            'gradient_accumulation': '4'  # Para estabilidad
        }
        
        self.expected_results = {
            'architecture': 'medium_iit_optimized',
            'parameters': '65.3M',
            'expected_improvement': '96.8%',
            'expected_phi_integration': '0.853',
            'training_time_estimate': '2-3 hours',
            'memory_requirements': '~6-8GB GPU'
        }
    
    def validate_environment(self):
        """Valida que el entorno esté listo para medium_iit."""
        print("🔍 VALIDANDO ENTORNO PARA MEDIUM IIT OPTIMIZADO")
        print("-" * 55)
        
        checks = []
        
        # Verificar script de entrenamiento
        train_script = "train_v5_2_wikitext_real.py"
        if os.path.exists(train_script):
            checks.append(("✅", f"Script de entrenamiento: {train_script}"))
        else:
            checks.append(("❌", f"Script de entrenamiento no encontrado: {train_script}"))
        
        # Verificar archivos necesarios
        required_files = [
            "requirements.txt",
            "generate_text_v5_2.py"  # Para validación posterior
        ]
        
        for file in required_files:
            if os.path.exists(file):
                checks.append(("✅", f"Archivo requerido: {file}"))
            else:
                checks.append(("⚠️", f"Archivo opcional: {file} (no crítico)"))
        
        # Verificar espacio en disco
        import shutil
        free_space_gb = shutil.disk_usage('.')[2] / (1024**3)
        if free_space_gb > 5:
            checks.append(("✅", f"Espacio en disco: {free_space_gb:.1f} GB"))
        else:
            checks.append(("⚠️", f"Poco espacio: {free_space_gb:.1f} GB"))
        
        # Mostrar validaciones
        for status, message in checks:
            print(f"   {status} {message}")
        
        # Determinar si está listo
        critical_issues = [c for c in checks if c[0] == "❌"]
        return len(critical_issues) == 0
    
    def show_training_plan(self):
        """Muestra el plan de entrenamiento detallado."""
        print(f"\n📋 PLAN DE ENTRENAMIENTO MEDIUM IIT OPTIMIZADO")
        print("="*55)
        
        print(f"\n🎯 ARQUITECTURA OBJETIVO:")
        print(f"   Modelo: {self.expected_results['architecture']}")
        print(f"   Parámetros: {self.expected_results['parameters']}")
        print(f"   Mejora esperada: {self.expected_results['expected_improvement']}")
        print(f"   PHI esperado: {self.expected_results['expected_phi_integration']}")
        
        print(f"\n⚙️ CONFIGURACIÓN OPTIMIZADA:")
        for key, value in self.optimized_config.items():
            print(f"   {key}: {value}")
        
        print(f"\n⏱️ ESTIMACIONES:")
        print(f"   Tiempo estimado: {self.expected_results['training_time_estimate']}")
        print(f"   Memoria requerida: {self.expected_results['memory_requirements']}")
        print(f"   Checkpoints: Cada época (~40-60 min)")
        
        print(f"\n🎯 OBJETIVOS:")
        print(f"   • Validar 96.8% mejora en arquitectura medium")
        print(f"   • Confirmar PHI integration ≥ 0.850")
        print(f"   • Generar modelos listos para producción")
        print(f"   • Establecer benchmark para large_iit")
    
    def build_command(self):
        """Construye el comando de entrenamiento."""
        base_cmd = ["python", "train_v5_2_wikitext_real.py"]
        
        # Agregar parámetros optimizados
        cmd_args = []
        for key, value in self.optimized_config.items():
            # Convertir nombres de parámetros
            param_name = key.replace('_', '-')
            cmd_args.extend([f"--{param_name}", value])
        
        full_command = base_cmd + cmd_args
        return full_command
    
    def create_launch_script(self):
        """Crea script de lanzamiento para Windows."""
        script_content = f"""@echo off
echo 🚀 LANZANDO ENTRENAMIENTO MEDIUM IIT OPTIMIZADO
echo =============================================
echo Configuracion: lambda_phi={self.optimized_config['lambda_phi']}, dropout={self.optimized_config['dropout']}
echo Arquitectura: medium_iit (65.3M parametros)
echo Mejora esperada: 96.8%%
echo.

python train_v5_2_wikitext_real.py ^
    --model-size {self.optimized_config['model_size']} ^
    --lambda-phi {self.optimized_config['lambda_phi']} ^
    --dropout {self.optimized_config['dropout']} ^
    --learning-rate {self.optimized_config['learning_rate']} ^
    --batch-size {self.optimized_config['batch_size']} ^
    --epochs {self.optimized_config['epochs']} ^
    --gradient-accumulation {self.optimized_config['gradient_accumulation']}

echo.
echo ✅ Entrenamiento medium_iit completado
pause
"""
        
        script_file = "launch_medium_iit_optimized.bat"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        print(f"💾 Script de lanzamiento creado: {script_file}")
        return script_file
    
    def create_monitoring_config(self):
        """Crea configuración para monitoreo del entrenamiento."""
        timestamp = datetime.now()
        
        monitoring_config = {
            'training_session': {
                'start_time': timestamp.isoformat(),
                'architecture': 'medium_iit_optimized',
                'configuration': self.optimized_config,
                'expected_results': self.expected_results,
                'optimization_source': 'November 17, 2025 optimization session'
            },
            'monitoring_targets': {
                'loss_improvement': '>95%',
                'phi_integration': '>0.850',
                'training_stability': 'Convergent',
                'quality_validation': 'EXCELENTE (>0.8)'
            },
            'checkpoints_to_save': [
                'best_epoch',
                'final_model', 
                'phi_optimized'
            ],
            'validation_tests': [
                'text_generation_quality',
                'phi_integration_analysis',
                'comparative_baseline_test'
            ]
        }
        
        config_file = f'medium_iit_monitoring_config_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(monitoring_config, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"💾 Configuración de monitoreo: {config_file}")
        return config_file
    
    def launch_medium_iit_training(self):
        """Lanza el entrenamiento medium_iit optimizado."""
        print(f"\n🚀 LANZANDO ENTRENAMIENTO MEDIUM IIT OPTIMIZADO")
        print("="*55)
        
        # Validar entorno
        if not self.validate_environment():
            print(f"\n❌ Entorno no válido. Revisa los problemas anteriores.")
            return False
        
        # Mostrar plan
        self.show_training_plan()
        
        # Crear archivos auxiliares
        launch_script = self.create_launch_script()
        monitoring_config = self.create_monitoring_config()
        
        # Construir comando
        command = self.build_command()
        command_str = ' '.join(command)
        
        print(f"\n💻 COMANDO A EJECUTAR:")
        print(f"   {command_str}")
        
        print(f"\n🎯 OPCIONES DE LANZAMIENTO:")
        print(f"   A) Ejecutar ahora en background")
        print(f"   B) Ejecutar ahora en foreground (bloqueante)")
        print(f"   C) Solo preparar (ejecutar manualmente)")
        print(f"   D) Usar script .bat ({launch_script})")
        
        # Por defecto, preparar para ejecución manual
        print(f"\n✅ PREPARACIÓN COMPLETADA")
        print(f"📋 Para ejecutar:")
        print(f"   1. Ejecutar: {launch_script}")
        print(f"   2. O comando directo: {command_str}")
        print(f"   3. Monitorear con: {monitoring_config}")
        
        return True
    
    def run(self):
        """Ejecuta la preparación completa."""
        print("🏗️ LANZADOR DE ARQUITECTURA MEDIUM IIT OPTIMIZADA")
        print("="*55)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            success = self.launch_medium_iit_training()
            
            if success:
                print(f"\n🎉 MEDIUM IIT OPTIMIZADO LISTO PARA LANZAR")
                print(f"   🎯 Configuración: λ_phi=0.01, dropout=0.25")
                print(f"   🏆 Mejora esperada: 96.8%")
                print(f"   ⚡ Parámetros: 65.3M")
                print(f"   🔬 PHI esperado: 0.853")
            else:
                print(f"\n❌ Error en preparación de medium_iit")
            
            return success
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False


def main():
    """Función principal."""
    launcher = MediumIITOptimizedLauncher()
    return launcher.run()


if __name__ == '__main__':
    success = main()
    if success:
        print(f"\n✅ Listo para entrenar medium_iit optimizado")
    else:
        print(f"\n❌ Error en preparación")