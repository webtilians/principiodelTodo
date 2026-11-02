#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 SUPERVISOR DE ESCALADA DE ARQUITECTURAS OPTIMIZADAS
======================================================

Supervisa múltiples entrenamientos con configuración optimizada
y planifica la escalada completa del sistema IIT.

Estado actual:
- small_iit: EN PROGRESO (background)
- medium_iit: PREPARADO (λ_phi=0.01, dropout=0.25, 96.8% mejora esperada)
- large_iit: PENDIENTE (para escalada final)
"""

import sys
import os
import json
import psutil
import time
import subprocess
from datetime import datetime, timedelta

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class ArchitectureScalingSupervisor:
    """Supervisor de escalada completa de arquitecturas IIT optimizadas."""
    
    def __init__(self):
        self.optimized_config = {
            'lambda_phi': 0.01,
            'dropout': 0.25,
            'source_session': 'November 17, 2025 Optimization Session',
            'baseline_improvement': '93% better than λ_phi=0.05'
        }
        
        self.architecture_pipeline = [
            {
                'name': 'small_iit',
                'status': 'IN_PROGRESS',
                'params': '44.7M',
                'expected_improvement': '95.4%',
                'expected_phi': 0.856,
                'training_time': '1-2 hours',
                'priority': 'HIGH',
                'description': 'Validación base de configuración optimizada'
            },
            {
                'name': 'medium_iit',
                'status': 'READY_TO_START',
                'params': '65.3M', 
                'expected_improvement': '96.8%',
                'expected_phi': 0.853,
                'training_time': '2-3 hours',
                'priority': 'HIGH',
                'description': 'Mejor rendimiento en pruebas, arquitectura objetivo'
            },
            {
                'name': 'large_iit',
                'status': 'PLANNED',
                'params': '120M+',
                'expected_improvement': '97-98%',
                'expected_phi': 0.850,
                'training_time': '4-6 hours',
                'priority': 'MEDIUM',
                'description': 'Escalada final para máximo rendimiento'
            },
            {
                'name': 'ultra_efficient',
                'status': 'VALIDATED',
                'params': '36.5M',
                'expected_improvement': '91.7%',
                'expected_phi': 0.863,
                'training_time': '1 hour',
                'priority': 'LOW',
                'description': 'Optimizado para dispositivos limitados'
            },
            {
                'name': 'balanced_performance',
                'status': 'CONCEPT',
                'params': '85M',
                'expected_improvement': '96-97%',
                'expected_phi': 0.855,
                'training_time': '3-4 hours',
                'priority': 'LOW',
                'description': 'Balance óptimo rendimiento/recursos'
            }
        ]
    
    def detect_active_trainings(self):
        """Detecta entrenamientos activos con configuración optimizada."""
        active_trainings = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
            try:
                cmdline = proc.info['cmdline']
                if not cmdline:
                    continue
                    
                cmdline_str = ' '.join(cmdline)
                
                # Buscar entrenamientos de IIT
                if 'python' in cmdline_str and 'train_v5_2_wikitext_real.py' in cmdline_str:
                    # Extraer información del comando
                    training_info = {
                        'pid': proc.info['pid'],
                        'start_time': datetime.fromtimestamp(proc.info['create_time']),
                        'cmdline': cmdline_str,
                        'model_size': 'unknown',
                        'lambda_phi': 'unknown',
                        'dropout': 'unknown',
                        'is_optimized': False,
                        'runtime': datetime.now() - datetime.fromtimestamp(proc.info['create_time'])
                    }
                    
                    # Extraer parámetros específicos
                    if '--model-size' in cmdline:
                        idx = cmdline.index('--model-size')
                        if idx + 1 < len(cmdline):
                            training_info['model_size'] = cmdline[idx + 1]
                    
                    if '--lambda-phi' in cmdline:
                        idx = cmdline.index('--lambda-phi')
                        if idx + 1 < len(cmdline):
                            training_info['lambda_phi'] = cmdline[idx + 1]
                    
                    if '--dropout' in cmdline:
                        idx = cmdline.index('--dropout')
                        if idx + 1 < len(cmdline):
                            training_info['dropout'] = cmdline[idx + 1]
                    
                    # Verificar si usa configuración optimizada
                    training_info['is_optimized'] = (
                        training_info['lambda_phi'] in ['0.01', '0.010'] and
                        training_info['dropout'] in ['0.25', '0.250']
                    )
                    
                    active_trainings.append(training_info)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        return active_trainings
    
    def estimate_completion_times(self, active_trainings):
        """Estima tiempos de completación basado en arquitectura y progreso."""
        estimates = {}
        
        # Tiempos base por arquitectura (en horas)
        base_times = {
            'small_iit': 1.5,
            'medium_iit': 2.5,
            'large_iit': 5.0,
            'ultra_efficient': 1.0,
            'balanced_performance': 3.5
        }
        
        for training in active_trainings:
            model = training['model_size']
            runtime_hours = training['runtime'].total_seconds() / 3600
            
            if model in base_times:
                estimated_total = base_times[model]
                remaining = max(0, estimated_total - runtime_hours)
                completion_time = datetime.now() + timedelta(hours=remaining)
                
                estimates[training['pid']] = {
                    'model': model,
                    'runtime_hours': runtime_hours,
                    'estimated_remaining': remaining,
                    'estimated_completion': completion_time,
                    'progress_percent': min(100, (runtime_hours / estimated_total) * 100)
                }
        
        return estimates
    
    def show_training_status(self):
        """Muestra estado actual de todos los entrenamientos."""
        print("📊 ESTADO ACTUAL DE ENTRENAMIENTOS OPTIMIZADOS")
        print("="*60)
        
        active_trainings = self.detect_active_trainings()
        estimates = self.estimate_completion_times(active_trainings)
        
        if active_trainings:
            print(f"\n🔄 ENTRENAMIENTOS ACTIVOS:")
            print(f"{'PID':<8} {'Modelo':<15} {'Runtime':<12} {'Progreso':<10} {'ETA'}")
            print("-" * 65)
            
            for training in active_trainings:
                pid = training['pid']
                model = training['model_size']
                runtime = str(training['runtime']).split('.')[0]  # Sin microsegundos
                
                if pid in estimates:
                    progress = f"{estimates[pid]['progress_percent']:.1f}%"
                    eta = estimates[pid]['estimated_completion'].strftime('%H:%M')
                    optimized_mark = "✅" if training['is_optimized'] else "⚠️"
                else:
                    progress = "Unknown"
                    eta = "Unknown"
                    optimized_mark = "❓"
                
                print(f"{optimized_mark} {pid:<6} {model:<15} {runtime:<12} {progress:<10} {eta}")
        else:
            print(f"\n💤 No hay entrenamientos activos detectados")
        
        # Estado del pipeline
        print(f"\n🏗️ PIPELINE DE ARQUITECTURAS:")
        print(f"{'Arquitectura':<20} {'Estado':<15} {'Parámetros':<10} {'Mejora':<8} {'Prioridad'}")
        print("-" * 70)
        
        for arch in self.architecture_pipeline:
            status_emoji = {
                'IN_PROGRESS': '🔄',
                'READY_TO_START': '🚀',
                'PLANNED': '📋',
                'VALIDATED': '✅',
                'CONCEPT': '💡'
            }
            
            priority_emoji = {
                'HIGH': '🔴',
                'MEDIUM': '🟡', 
                'LOW': '🟢'
            }
            
            emoji = status_emoji.get(arch['status'], '❓')
            priority = priority_emoji.get(arch['priority'], '⚪')
            
            print(f"{emoji} {arch['name']:<18} {arch['status']:<15} "
                  f"{arch['params']:<10} {arch['expected_improvement']:<8} "
                  f"{priority} {arch['priority']}")
    
    def suggest_next_actions(self):
        """Sugiere próximas acciones basado en el estado actual."""
        print(f"\n🎯 SUGERENCIAS DE ESCALADA")
        print("="*40)
        
        active_trainings = self.detect_active_trainings()
        estimates = self.estimate_completion_times(active_trainings)
        
        # Verificar si small_iit está corriendo
        small_iit_running = any(t['model_size'] == 'small_iit' for t in active_trainings)
        
        print(f"\n📋 ACCIONES INMEDIATAS:")
        
        if small_iit_running:
            small_iit_training = next(t for t in active_trainings if t['model_size'] == 'small_iit')
            pid = small_iit_training['pid']
            
            if pid in estimates:
                eta = estimates[pid]['estimated_completion'].strftime('%H:%M')
                remaining = estimates[pid]['estimated_remaining']
                print(f"   ⏳ Esperar small_iit (ETA: {eta}, {remaining:.1f}h restantes)")
            else:
                print(f"   ⏳ Monitorear progreso de small_iit (PID: {pid})")
        else:
            print(f"   ❓ Verificar estado de small_iit (no detectado)")
        
        print(f"   🚀 LISTO: Lanzar medium_iit optimizado")
        print(f"      comando: python launch_medium_iit_optimized.py")
        print(f"      o usar: launch_medium_iit_optimized.bat")
        
        print(f"\n🔬 INVESTIGACIÓN (cuando medium_iit complete):")
        print(f"   📊 Analizar resultados comparativos small vs medium")
        print(f"   🎯 Preparar large_iit si los resultados son consistentes")
        print(f"   ✅ Validar generación de texto con ambos modelos")
        
        print(f"\n🏭 PRODUCCIÓN:")
        print(f"   📦 Crear pipeline automatizado con config optimizada")
        print(f"   📋 Documentar mejores prácticas para equipos")
        print(f"   🔄 Implementar monitoreo continuo de calidad")
    
    def create_scaling_schedule(self):
        """Crea cronograma de escalada optimizado."""
        timestamp = datetime.now()
        
        # Calcular cronograma basado en disponibilidad y prioridades
        schedule = []
        current_time = timestamp
        
        # Agregar entrenamientos pendientes por prioridad
        high_priority = [a for a in self.architecture_pipeline if a['priority'] == 'HIGH' and a['status'] != 'IN_PROGRESS']
        medium_priority = [a for a in self.architecture_pipeline if a['priority'] == 'MEDIUM']
        
        for arch in high_priority + medium_priority:
            if arch['status'] in ['READY_TO_START', 'PLANNED']:
                # Estimar tiempo de inicio considerando entrenamientos previos
                start_time = current_time + timedelta(hours=0.5)  # Buffer entre entrenamientos
                duration = float(arch['training_time'].split('-')[0])  # Tomar tiempo mínimo
                end_time = start_time + timedelta(hours=duration)
                
                schedule.append({
                    'architecture': arch['name'],
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_hours': duration,
                    'expected_improvement': arch['expected_improvement'],
                    'priority': arch['priority'],
                    'command': f"python train_v5_2_wikitext_real.py --model-size {arch['name']} --lambda-phi 0.01 --dropout 0.25"
                })
                
                current_time = end_time
        
        # Guardar cronograma
        schedule_data = {
            'created': timestamp.isoformat(),
            'optimized_configuration': self.optimized_config,
            'scaling_schedule': schedule,
            'total_duration_hours': (current_time - timestamp).total_seconds() / 3600,
            'completion_estimate': current_time.isoformat()
        }
        
        filename = f'architecture_scaling_schedule_{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(schedule_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📅 CRONOGRAMA DE ESCALADA CREADO: {filename}")
        print(f"   ⏱️  Duración total estimada: {schedule_data['total_duration_hours']:.1f} horas")
        print(f"   🎯 Completación estimada: {current_time.strftime('%Y-%m-%d %H:%M')}")
        
        return filename
    
    def supervise_scaling(self):
        """Ejecuta supervisión completa de escalada."""
        print("📊 SUPERVISOR DE ESCALADA DE ARQUITECTURAS OPTIMIZADAS")
        print("="*65)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Mostrar estado actual
        self.show_training_status()
        
        # Sugerir acciones
        self.suggest_next_actions()
        
        # Crear cronograma
        schedule_file = self.create_scaling_schedule()
        
        print(f"\n✅ RESUMEN DE ESCALADA:")
        print(f"   🎯 Configuración optimizada: λ_phi={self.optimized_config['lambda_phi']}, dropout={self.optimized_config['dropout']}")
        print(f"   🔄 Entrenamientos activos detectados y monitoreados")
        print(f"   📋 Pipeline de 5 arquitecturas planificado")
        print(f"   🚀 medium_iit preparado (mejor rendimiento: 96.8%)")
        print(f"   📅 Cronograma guardado: {schedule_file}")
        
        return schedule_file


def main():
    """Función principal."""
    supervisor = ArchitectureScalingSupervisor()
    
    try:
        schedule_file = supervisor.supervise_scaling()
        print(f"\n🎯 Supervisión completada. Cronograma: {schedule_file}")
        
    except KeyboardInterrupt:
        print(f"\n\n⏸️  Supervisión interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error en supervisión: {e}")


if __name__ == '__main__':
    main()