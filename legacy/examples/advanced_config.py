#!/usr/bin/env python3
"""
🧠 INFINITO - CONFIGURACIÓN AVANZADA
====================================

Ejemplo de configuración avanzada del sistema INFINITO
con parámetros optimizados para casos específicos.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from infinito_v3_clean import InfinitoV3Clean

def high_performance_config():
    """Configuración para máximo rendimiento"""
    
    print("🚀 INFINITO - Configuración Alto Rendimiento")
    print("="*60)
    
    infinito = InfinitoV3Clean(
        grid_size=128,              # Grid más grande para mayor complejidad
        max_iterations=2000,        # Más iteraciones para convergencia
        target_consciousness=0.95,  # Target alto
        learning_rate=0.0005,       # Learning rate más conservador
        update_interval=50,         # Updates menos frecuentes
        save_checkpoints=True       # Guardar checkpoints
    )
    
    return infinito.run_consciousness_experiment()

def quick_test_config():
    """Configuración para pruebas rápidas"""
    
    print("⚡ INFINITO - Configuración Rápida")
    print("="*45)
    
    infinito = InfinitoV3Clean(
        grid_size=32,               # Grid pequeño
        max_iterations=50,          # Pocas iteraciones
        target_consciousness=0.70,  # Target más bajo
        learning_rate=0.01,         # Learning rate más agresivo
        update_interval=10          # Updates frecuentes
    )
    
    return infinito.run_consciousness_experiment()

def research_config():
    """Configuración para investigación experimental"""
    
    print("🔬 INFINITO - Configuración Experimental")
    print("="*50)
    
    infinito = InfinitoV3Clean(
        grid_size=256,              # Grid muy grande
        max_iterations=5000,        # Muchas iteraciones
        target_consciousness=0.99,  # Target muy alto
        learning_rate=0.0001,       # Learning rate muy conservador
        update_interval=100,        # Updates esporádicos
        save_checkpoints=True,      # Checkpoints para experimentos largos
        enable_advanced_metrics=True  # Métricas adicionales
    )
    
    return infinito.run_consciousness_experiment()

def main():
    """Ejecutar configuración seleccionada"""
    
    configs = {
        '1': ("Alto Rendimiento", high_performance_config),
        '2': ("Prueba Rápida", quick_test_config),
        '3': ("Experimental", research_config)
    }
    
    print("🧠 INFINITO - Configuraciones Disponibles:")
    for key, (name, _) in configs.items():
        print(f"   {key}. {name}")
    
    choice = input("\nSelecciona configuración (1-3): ").strip()
    
    if choice in configs:
        name, config_func = configs[choice]
        print(f"\n🎯 Ejecutando configuración: {name}")
        results = config_func()
        print(f"\n✅ Experimento '{name}' completado!")
    else:
        print("❌ Selección inválida")

if __name__ == "__main__":
    main()