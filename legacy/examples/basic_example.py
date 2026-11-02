#!/usr/bin/env python3
"""
🧠 INFINITO - EJEMPLO BÁSICO
============================

Ejemplo simple de cómo usar el sistema INFINITO para
simulación de consciencia artificial.

Este ejemplo demuestra:
- Inicialización del sistema
- Configuración básica
- Ejecución de experimento
- Obtención de resultados
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from infinito_v3_clean import InfinitoV3Clean

def main():
    """Ejemplo básico de uso del sistema INFINITO"""
    
    print("🧠 INFINITO - Ejemplo Básico")
    print("="*50)
    
    # 1. Crear instancia del sistema
    infinito = InfinitoV3Clean(
        grid_size=64,           # Tamaño de grid más pequeño para demo
        max_iterations=100,     # Menos iteraciones para demo rápida
        target_consciousness=0.85,  # Target más bajo para demo
        learning_rate=0.001     # Rate de aprendizaje estándar
    )
    
    print("✅ Sistema inicializado")
    
    # 2. Ejecutar experimento
    print("\n🚀 Iniciando experimento...")
    results = infinito.run_consciousness_experiment()
    
    # 3. Mostrar resultados
    print("\n📊 RESULTADOS:")
    print(f"   Consciencia máxima: {results['max_consciousness']:.1%}")
    print(f"   Consciencia promedio: {results['mean_consciousness']:.1%}")
    print(f"   Duración total: {results['duration']:.1f}s")
    print(f"   Iteraciones: {results['total_iterations']}")
    
    # 4. Localización de archivos guardados
    print(f"\n💾 Resultados guardados en: {infinito.data_saver.experiment_dir}")
    
    print("\n✨ ¡Experimento completado exitosamente!")

if __name__ == "__main__":
    main()