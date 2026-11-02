#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 MONITOR DE ENTRENAMIENTO EXTENDIDO
===================================

Monitor en tiempo real para el entrenamiento extendido con configuración optimizada.
"""

import sys
import os
import time
import json
import glob
from datetime import datetime

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class ExtendedTrainingMonitor:
    """Monitor para entrenamiento extendido."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.workspace_path = "C:\\Users\\ENRIQUE\\universo"
    
    def find_latest_results(self):
        """Encuentra el archivo de resultados más reciente."""
        pattern = os.path.join(self.workspace_path, "optimized_training_results_*.json")
        files = glob.glob(pattern)
        
        if not files:
            return None
        
        # Ordenar por fecha de modificación
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    
    def check_model_file(self):
        """Verifica si el modelo se ha guardado."""
        model_path = os.path.join(self.workspace_path, "infinito_v5.2_optimized_extended.pt")
        return os.path.exists(model_path)
    
    def analyze_progress(self, results_file):
        """Analiza el progreso del entrenamiento."""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metrics = data.get('training_metrics', {})
            config = data.get('optimization_config', {})
            
            print(f"\n📊 ANÁLISIS DE PROGRESO")
            print("="*50)
            print(f"Configuración lambda_phi: {config.get('lambda_phi', 'N/A'):.3f}")
            print(f"Steps completados: {metrics.get('steps', 'N/A')}")
            print(f"Best loss: {metrics.get('best_loss', 'N/A'):.4f}")
            print(f"Final perplexity: {metrics.get('final_perplexity', 'N/A'):.1f}")
            print(f"Avg PHI (last 100): {metrics.get('avg_phi_last_100', 'N/A'):.4f}")
            
            # Analizar curvas de entrenamiento
            curves = metrics.get('training_curves', {})
            if curves:
                losses = curves.get('losses', [])
                perplexities = curves.get('perplexities', [])
                phi_values = curves.get('phi_values', [])
                
                if losses:
                    print(f"\n📈 TENDENCIAS:")
                    print(f"Loss inicial: {losses[0]:.4f}")
                    print(f"Loss final: {losses[-1]:.4f}")
                    improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
                    print(f"Mejora en loss: {improvement:.1f}%")
                    
                    if perplexities:
                        print(f"Perplexity inicial: {perplexities[0]:.1f}")
                        print(f"Perplexity final: {perplexities[-1]:.1f}")
                        ppl_improvement = ((perplexities[0] - perplexities[-1]) / perplexities[0]) * 100
                        print(f"Mejora en perplexity: {ppl_improvement:.1f}%")
                    
                    if phi_values:
                        avg_phi = sum(phi_values) / len(phi_values)
                        print(f"PHI promedio: {avg_phi:.4f}")
            
            # Evaluar generaciones si están disponibles
            generations = data.get('generation_evaluation', [])
            if generations:
                print(f"\n🎨 EVALUACIÓN DE GENERACIÓN:")
                for i, gen in enumerate(generations[:3], 1):
                    prompt = gen.get('prompt', '')
                    continuation = gen.get('clean_continuation', '')[:100]
                    print(f"   {i}. '{prompt}' -> '{continuation}...'")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error analizando resultados: {e}")
            return False
    
    def monitor_training(self, check_interval=30):
        """Monitorea el entrenamiento en curso."""
        print(f"🔍 MONITOR DE ENTRENAMIENTO EXTENDIDO")
        print(f"Iniciado: {self.start_time}")
        print(f"Directorio de trabajo: {self.workspace_path}")
        print(f"Intervalo de verificación: {check_interval}s")
        print("="*60)
        
        iteration = 0
        last_results_time = None
        
        while True:
            iteration += 1
            current_time = datetime.now()
            elapsed = current_time - self.start_time
            
            print(f"\n🔄 Verificación #{iteration} - Tiempo transcurrido: {elapsed}")
            
            # Verificar archivo de resultados
            results_file = self.find_latest_results()
            if results_file:
                file_time = datetime.fromtimestamp(os.path.getmtime(results_file))
                
                # Si es un archivo nuevo o actualizado
                if last_results_time is None or file_time > last_results_time:
                    last_results_time = file_time
                    print(f"✓ Archivo de resultados encontrado: {os.path.basename(results_file)}")
                    print(f"  Última modificación: {file_time}")
                    
                    if self.analyze_progress(results_file):
                        # Si el análisis es exitoso y parece completo, verificar si terminó
                        with open(results_file, 'r') as f:
                            data = json.load(f)
                        
                        steps = data.get('training_metrics', {}).get('steps', 0)
                        if steps >= 2000:  # Si completó los 2000 steps
                            print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
                            break
                else:
                    print(f"📝 Archivo sin cambios desde: {last_results_time}")
            else:
                print("⏳ Esperando archivo de resultados...")
            
            # Verificar modelo guardado
            if self.check_model_file():
                print("✓ Archivo de modelo encontrado: infinito_v5.2_optimized_extended.pt")
                # Si el modelo existe, probablemente el entrenamiento terminó
                if results_file:
                    print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
                    break
            else:
                print("⏳ Esperando archivo de modelo...")
            
            print(f"⏰ Próxima verificación en {check_interval}s...")
            time.sleep(check_interval)
        
        print(f"\n✅ Monitoreo finalizado. Tiempo total: {datetime.now() - self.start_time}")


def main():
    """Función principal."""
    monitor = ExtendedTrainingMonitor()
    
    try:
        monitor.monitor_training(check_interval=45)  # Verificar cada 45 segundos
    except KeyboardInterrupt:
        print(f"\n⏹️ Monitoreo detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error en monitoreo: {e}")


if __name__ == '__main__':
    main()