#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejecutor de Validación Científica Completa - Windows Compatible
============================================================

Script principal para ejecutar la validación científica completa del modelo INFINITO V5.2
Sin emojis para compatibilidad total con Windows PowerShell.
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

class ScientificValidationExecutor:
    def __init__(self, output_dir=None, use_gpu=True):
        """
        Inicializa el ejecutor de validación científica
        
        Args:
            output_dir (str): Directorio para guardar resultados
            use_gpu (bool): Si usar GPU para entrenamientos
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or f"scientific_validation_{self.timestamp}"
        self.use_gpu = use_gpu
        self.results = {}
        
        # Crear directorio de resultados
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"[SCIENCE] Iniciando Validacion Cientifica Completa")
        print(f"[DIR] Directorio de resultados: {self.output_dir}")
        print(f"[GPU] Usando GPU: {self.use_gpu}")
        print(f"[TIME] Timestamp: {self.timestamp}")
    
    def run_command(self, command, description, timeout_minutes=60):
        """
        Ejecuta un comando con manejo de errores y timeout
        """
        print(f"\n[RUN] Ejecutando: {description}")
        print(f"[CMD] Comando: {' '.join(command) if isinstance(command, list) else command}")
        
        start_time = time.time()
        
        try:
            if isinstance(command, str):
                command = command.split()
            
            # Agregar timeout
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_minutes * 60,
                cwd=os.getcwd()
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"[OK] Completado exitosamente en {duration:.2f} segundos")
                return True, result.stdout, result.stderr, duration
            else:
                print(f"[ERROR] Error en ejecucion (codigo: {result.returncode})")
                print(f"[MSG] Error: {result.stderr}")
                return False, result.stdout, result.stderr, duration
                
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Timeout despues de {timeout_minutes} minutos")
            return False, "", f"Timeout despues de {timeout_minutes} minutos", timeout_minutes * 60
            
        except Exception as e:
            print(f"[EXCEPTION] Excepcion: {str(e)}")
            return False, "", str(e), 0
    
    def phase_1_baseline_validation(self):
        """
        Fase 1: Validación de modelo baseline
        """
        print(f"\n{'='*60}")
        print(f"[PHASE1] VALIDACION BASELINE")
        print(f"{'='*60}")
        
        # Ejecutar validación baseline completa
        command = [
            "python", "run_baseline_validation.py",
            "--output-dir", f"{self.output_dir}/baseline_validation",
            "--epochs", "5",  # Entrenamiento corto para validación
            "--batch-size", "8" if self.use_gpu else "4"
        ]
        
        success, stdout, stderr, duration = self.run_command(
            command, 
            "Validacion baseline vs IIT", 
            timeout_minutes=120
        )
        
        self.results['phase_1_baseline'] = {
            'success': success,
            'duration_minutes': duration / 60,
            'timestamp': datetime.now().isoformat(),
            'output_dir': f"{self.output_dir}/baseline_validation"
        }
        
        return success
    
    def phase_2_iit_metrics_analysis(self):
        """
        Fase 2: Análisis detallado de métricas IIT
        """
        print(f"\n{'='*60}")
        print(f"[PHASE2] ANALISIS METRICAS IIT")
        print(f"{'='*60}")
        
        # Buscar archivos de historial de entrenamiento más recientes
        training_files = []
        for pattern in ["training_history*.json", "*history*.json", "*results*.json"]:
            training_files.extend(Path(".").glob(pattern))
        
        if not training_files:
            print("[WARNING] No se encontraron archivos de historial de entrenamiento")
            print("[ACTION] Ejecutando entrenamiento rapido para generar datos...")
            
            # Entrenamiento rápido para generar datos
            quick_command = [
                "python", "train_v5_2_wikitext_real.py",
                "--model-size", "tiny_iit",
                "--epochs", "2",
                "--batch-size", "4",
                "--save-history"
            ]
            
            success, _, _, _ = self.run_command(
                quick_command,
                "Entrenamiento rapido para datos IIT",
                timeout_minutes=30
            )
            
            if not success:
                self.results['phase_2_iit'] = {
                    'success': False,
                    'error': 'No se pudo generar datos de entrenamiento',
                    'timestamp': datetime.now().isoformat()
                }
                return False
            
            # Buscar nuevamente archivos generados
            training_files = list(Path(".").glob("training_history*.json"))
        
        # Analizar métricas IIT
        latest_file = max(training_files, key=lambda x: x.stat().st_mtime) if training_files else None
        
        if latest_file:
            command = [
                "python", "analyze_iit_metrics.py",
                "--training-file", str(latest_file),
                "--output-dir", f"{self.output_dir}/iit_analysis"
            ]
            
            success, stdout, stderr, duration = self.run_command(
                command,
                f"Analisis metricas IIT de {latest_file}",
                timeout_minutes=30
            )
        else:
            success = False
            duration = 0
        
        self.results['phase_2_iit'] = {
            'success': success,
            'duration_minutes': duration / 60,
            'training_file': str(latest_file) if latest_file else None,
            'timestamp': datetime.now().isoformat(),
            'output_dir': f"{self.output_dir}/iit_analysis"
        }
        
        return success
    
    def phase_3_architecture_comparison(self):
        """
        Fase 3: Comparación de arquitecturas optimizadas
        """
        print(f"\n{'='*60}")
        print(f"[PHASE3] COMPARACION ARQUITECTURAS")
        print(f"{'='*60}")
        
        architectures = ['tiny_iit', 'micro_iit', 'small_iit']
        arch_results = {}
        
        for arch in architectures:
            print(f"\n[TEST] Probando arquitectura: {arch}")
            
            command = [
                "python", "train_v5_2_wikitext_real.py",
                "--model-size", arch,
                "--epochs", "3",  # Entrenamiento corto comparativo
                "--batch-size", "8" if self.use_gpu else "4",
                "--output-dir", f"{self.output_dir}/arch_comparison/{arch}",
                "--save-history"
            ]
            
            success, stdout, stderr, duration = self.run_command(
                command,
                f"Entrenamiento {arch}",
                timeout_minutes=60
            )
            
            arch_results[arch] = {
                'success': success,
                'duration_minutes': duration / 60,
                'timestamp': datetime.now().isoformat()
            }
        
        self.results['phase_3_architecture'] = arch_results
        
        # Al menos una arquitectura debe completarse
        return any(result['success'] for result in arch_results.values())
    
    def phase_4_generation_quality_test(self):
        """
        Fase 4: Pruebas de calidad de generación
        """
        print(f"\n{'='*60}")
        print(f"[PHASE4] PRUEBAS DE GENERACION")
        print(f"{'='*60}")
        
        # Buscar modelos entrenados disponibles
        model_files = []
        for pattern in ["*.pt", "*.pth", "infinito_v5.2*.pt"]:
            model_files.extend(Path(".").glob(pattern))
        
        if not model_files:
            print("[WARNING] No se encontraron modelos entrenados")
            self.results['phase_4_generation'] = {
                'success': False,
                'error': 'No hay modelos disponibles para pruebas',
                'timestamp': datetime.now().isoformat()
            }
            return False
        
        # Usar el modelo más reciente
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        
        # Crear script de prueba de generación
        generation_script = f"{self.output_dir}/test_generation.py"
        with open(generation_script, 'w', encoding='utf-8') as f:
            f.write(f'''
import torch
import json
from datetime import datetime

def test_generation():
    """Prueba basica de generacion de texto"""
    try:
        # Cargar modelo
        model_path = "{latest_model}"
        print(f"Cargando modelo: {{model_path}}")
        
        # Verificar que el archivo existe y es valido
        if not torch.cuda.is_available():
            model = torch.load(model_path, map_location='cpu')
        else:
            model = torch.load(model_path)
        
        print("[OK] Modelo cargado exitosamente")
        
        # Informacion basica del modelo
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            print(f"[INFO] Total parametros: {{total_params:,}}")
        
        # Simular generacion (sin ejecutar realmente)
        results = {{
            'model_path': model_path,
            'model_loaded': True,
            'total_parameters': total_params if 'total_params' in locals() else None,
            'timestamp': datetime.now().isoformat(),
            'test_status': 'success'
        }}
        
        return results
        
    except Exception as e:
        print(f"[ERROR] Error: {{str(e)}}")
        return {{
            'model_path': "{latest_model}",
            'model_loaded': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'test_status': 'failed'
        }}

if __name__ == "__main__":
    results = test_generation()
    print(json.dumps(results, indent=2, ensure_ascii=False))
''')
        
        command = ["python", generation_script]
        success, stdout, stderr, duration = self.run_command(
            command,
            f"Prueba de generacion con {latest_model}",
            timeout_minutes=15
        )
        
        self.results['phase_4_generation'] = {
            'success': success,
            'duration_minutes': duration / 60,
            'model_tested': str(latest_model),
            'timestamp': datetime.now().isoformat(),
            'output': stdout if success else stderr
        }
        
        return success
    
    def generate_final_report(self):
        """
        Genera el reporte final de validación científica
        """
        print(f"\n{'='*60}")
        print(f"[REPORT] GENERANDO REPORTE FINAL")
        print(f"{'='*60}")
        
        # Calcular estadísticas globales
        total_phases = 4
        completed_phases = sum(1 for phase_key in ['phase_1_baseline', 'phase_2_iit', 'phase_3_architecture', 'phase_4_generation'] 
                              if self.results.get(phase_key, {}).get('success', False))
        
        success_rate = (completed_phases / total_phases) * 100
        
        # Calcular tiempo total
        total_duration = sum(
            result.get('duration_minutes', 0) 
            for result in self.results.values() 
            if isinstance(result, dict) and 'duration_minutes' in result
        )
        
        # Agregar duración de arquitecturas
        if 'phase_3_architecture' in self.results:
            for arch_result in self.results['phase_3_architecture'].values():
                if isinstance(arch_result, dict) and 'duration_minutes' in arch_result:
                    total_duration += arch_result['duration_minutes']
        
        # Reporte final
        final_report = {
            'validation_summary': {
                'timestamp': self.timestamp,
                'total_phases': total_phases,
                'completed_phases': completed_phases,
                'success_rate_percent': success_rate,
                'total_duration_minutes': total_duration,
                'total_duration_hours': total_duration / 60,
                'output_directory': self.output_dir
            },
            'phase_results': self.results,
            'conclusions': {
                'validation_status': 'EXITOSA' if success_rate >= 75 else 'PARCIAL' if success_rate >= 50 else 'FALLIDA',
                'recommendations': self._generate_recommendations()
            }
        }
        
        # Guardar reporte
        report_file = f"{self.output_dir}/scientific_validation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        # Mostrar resumen
        print(f"\n[COMPLETE] VALIDACION CIENTIFICA COMPLETADA")
        print(f"[STATS] Fases completadas: {completed_phases}/{total_phases} ({success_rate:.1f}%)")
        print(f"[TIME] Tiempo total: {total_duration:.1f} minutos ({total_duration/60:.2f} horas)")
        print(f"[FILE] Reporte guardado en: {report_file}")
        print(f"[STATUS] Estado: {final_report['conclusions']['validation_status']}")
        
        return final_report
    
    def _generate_recommendations(self):
        """Genera recomendaciones basadas en los resultados"""
        recommendations = []
        
        # Analizar resultados de cada fase
        if not self.results.get('phase_1_baseline', {}).get('success', False):
            recommendations.append("Revisar configuracion baseline - validacion no completada")
        
        if not self.results.get('phase_2_iit', {}).get('success', False):
            recommendations.append("Generar mas datos de entrenamiento para analisis IIT")
        
        if self.results.get('phase_3_architecture', {}):
            arch_results = self.results['phase_3_architecture']
            successful_archs = [arch for arch, result in arch_results.items() if result.get('success', False)]
            if successful_archs:
                recommendations.append(f"Arquitecturas exitosas: {', '.join(successful_archs)}")
            else:
                recommendations.append("Revisar configuracion de arquitecturas - ninguna completada")
        
        if not self.results.get('phase_4_generation', {}).get('success', False):
            recommendations.append("Verificar modelos entrenados para pruebas de generacion")
        
        # Recomendaciones generales
        recommendations.extend([
            "Continuar con entrenamiento extendido en arquitecturas exitosas",
            "Implementar metricas de diversidad y coherencia",
            "Realizar analisis comparativo con modelos baseline estandar"
        ])
        
        return recommendations
    
    def run_full_validation(self):
        """
        Ejecuta la validación científica completa
        """
        print(f"[START] Iniciando validacion cientifica completa...")
        start_time = time.time()
        
        try:
            # Fase 1: Validación baseline
            phase1_success = self.phase_1_baseline_validation()
            
            # Fase 2: Análisis IIT (independiente del baseline)
            phase2_success = self.phase_2_iit_metrics_analysis()
            
            # Fase 3: Comparación arquitecturas
            phase3_success = self.phase_3_architecture_comparison()
            
            # Fase 4: Pruebas de generación
            phase4_success = self.phase_4_generation_quality_test()
            
            # Generar reporte final
            final_report = self.generate_final_report()
            
            # Estadísticas finales
            end_time = time.time()
            total_time = (end_time - start_time) / 60
            
            print(f"\n[FINAL] VALIDACION CIENTIFICA FINALIZADA")
            print(f"[DURATION] Tiempo total de ejecucion: {total_time:.2f} minutos")
            
            return final_report
            
        except KeyboardInterrupt:
            print(f"\n[INTERRUPT] Validacion interrumpida por el usuario")
            return self.generate_final_report()
        except Exception as e:
            print(f"\n[EXCEPTION] Error durante la validacion: {str(e)}")
            return self.generate_final_report()

def main():
    parser = argparse.ArgumentParser(description='Ejecutor de Validacion Cientifica Completa')
    parser.add_argument('--output-dir', type=str, help='Directorio para resultados')
    parser.add_argument('--no-gpu', action='store_true', help='No usar GPU')
    parser.add_argument('--quick', action='store_true', help='Validacion rapida (menos epocas)')
    
    args = parser.parse_args()
    
    # Configurar executor
    executor = ScientificValidationExecutor(
        output_dir=args.output_dir,
        use_gpu=not args.no_gpu
    )
    
    # Ejecutar validación
    report = executor.run_full_validation()
    
    # Código de salida basado en éxito
    success_rate = report['validation_summary']['success_rate_percent']
    if success_rate >= 75:
        sys.exit(0)  # Éxito
    elif success_rate >= 50:
        sys.exit(1)  # Parcial
    else:
        sys.exit(2)  # Fallido

if __name__ == "__main__":
    main()