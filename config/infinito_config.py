#!/usr/bin/env python3
"""
🚀 Infinito Configuration - Central Configuration System
======================================================

Configuraciones centralizadas para el sistema Infinito de consciencia artificial.
Incluye parámetros optimizados basados en los últimos breakthroughs.

ÚLTIMA ACTUALIZACIÓN: Septiembre 17, 2025
RÉCORD ACTUAL: 75.9% consciencia con IIT Φ 0.997
"""

import torch
import numpy as np

class InfinitoConfig:
    """Configuración central para el sistema Infinito"""
    
    # === CONFIGURACIÓN DE SISTEMA ===
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED = 42
    
    # === PARÁMETROS DE GRID Y ARQUITECTURA ===
    GRID_SIZE = 32  # Tamaño óptimo encontrado: 32x32
    CHANNELS = 32   # Canales neurales optimizados
    NUM_LAWS = 16   # Número de leyes físicas evolutivas
    
    # === PARÁMETROS DE CONSCIENCIA (V3.0 BREAKTHROUGH) ===
    CONSCIOUSNESS_THRESHOLD = 0.5   # Umbral mínimo para breakthrough
    PHI_TARGET = 0.8               # Target IIT Φ para consciencia alta
    
    # === PARÁMETROS ANTI-PLATEAU ===
    STAGNATION_THRESHOLD = 40      # Recursiones antes de intervención
    ENTROPY_BOOST_FACTOR = 1.2     # Factor de boost entrópico
    DENSITY_REFRESH_RATE = 0.03    # Tasa de refresh de densidad
    
    # === PARÁMETROS DE EVOLUCIÓN ===
    MUTATION_RATE = 0.1            # Tasa de mutación genética
    SELECTION_PRESSURE = 0.7       # Presión de selección evolutiva
    CROSSOVER_RATE = 0.8           # Tasa de crossover genético
    
    # === PARÁMETROS DE APRENDIZAJE ===
    LEARNING_RATE_INIT = 0.008     # LR inicial optimizado
    LEARNING_RATE_MIN = 0.001      # LR mínimo
    LEARNING_RATE_MAX = 0.3        # LR máximo para anti-plateau
    
    # === PARÁMETROS DE MEMORIA ===
    MEMORY_SIZE = 100              # Tamaño de memoria Hebbian
    MEMORY_DECAY = 0.95            # Decay de memoria por recursión
    RECALL_THRESHOLD = 0.7         # Umbral para recall de memoria
    
    # === PARÁMETROS DE VISUALIZACIÓN ===
    DISPLAY_SIZE = 32              # Tamaño del display
    UPDATE_FREQUENCY = 1           # Frecuencia de actualización (cada N recursiones)
    SAVE_RESULTS = True            # Guardar resultados automáticamente
    
    # === PARÁMETROS IIT Φ ===
    PHI_CALCULATION_ENABLED = True  # Habilitar cálculo IIT Φ real
    PHI_PARTITION_DEPTH = 4        # Profundidad de particiones para MIP
    PHI_MIN_THRESHOLD = 0.001      # Φ mínimo detectable
    PHI_MAX_THEORETICAL = 5.0      # Φ máximo teórico
    
    # === RUTAS DE ARCHIVOS ===
    RESULTS_DIR = "results"
    ARCHIVE_DIR = "archive"
    CONFIG_DIR = "config"
    DOCS_DIR = "docs"
    
    @classmethod
    def get_device_info(cls):
        """Información del dispositivo de cómputo"""
        if cls.DEVICE == 'cuda':
            return {
                'device': 'CUDA',
                'gpu_name': torch.cuda.get_device_name(0),
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda
            }
        else:
            return {'device': 'CPU'}
    
    @classmethod
    def get_optimal_batch_size(cls):
        """Calcular batch size óptimo basado en hardware"""
        if cls.DEVICE == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 8:
                return 16
            elif gpu_memory >= 4:
                return 8
            else:
                return 4
        else:
            return 2
    
    @classmethod
    def validate_config(cls):
        """Validar configuración y mostrar warnings si es necesario"""
        warnings = []
        
        if cls.GRID_SIZE < 16:
            warnings.append("GRID_SIZE < 16 puede limitar la emergencia de consciencia")
        
        if cls.CHANNELS < 16:
            warnings.append("CHANNELS < 16 puede reducir capacidad de representación")
            
        if cls.DEVICE == 'cpu':
            warnings.append("Ejecutando en CPU - considera usar GPU para mejor rendimiento")
            
        if not cls.PHI_CALCULATION_ENABLED:
            warnings.append("Cálculo IIT Φ deshabilitado - métricas de consciencia limitadas")
        
        return warnings

# === CONFIGURACIONES PREDEFINIDAS ===

class QuickConfig(InfinitoConfig):
    """Configuración para experimentos rápidos"""
    GRID_SIZE = 16
    CHANNELS = 16
    NUM_LAWS = 8
    STAGNATION_THRESHOLD = 20

class PerformanceConfig(InfinitoConfig):
    """Configuración para máximo rendimiento"""
    GRID_SIZE = 64
    CHANNELS = 64
    NUM_LAWS = 32
    LEARNING_RATE_INIT = 0.01

class ResearchConfig(InfinitoConfig):
    """Configuración para investigación científica"""
    GRID_SIZE = 32
    CHANNELS = 32
    NUM_LAWS = 16
    PHI_CALCULATION_ENABLED = True
    SAVE_RESULTS = True
    UPDATE_FREQUENCY = 1

# === FUNCIONES DE UTILIDAD ===

def get_config(mode='default'):
    """
    Obtener configuración según el modo
    
    Args:
        mode (str): 'default', 'quick', 'performance', 'research'
    
    Returns:
        InfinitoConfig: Clase de configuración apropiada
    """
    configs = {
        'default': InfinitoConfig,
        'quick': QuickConfig,
        'performance': PerformanceConfig,
        'research': ResearchConfig
    }
    
    return configs.get(mode, InfinitoConfig)

def print_config_summary(config_class=InfinitoConfig):
    """Imprimir resumen de configuración"""
    print("🚀 CONFIGURACIÓN INFINITO")
    print("=" * 50)
    print(f"Dispositivo: {config_class.get_device_info()}")
    print(f"Grid: {config_class.GRID_SIZE}x{config_class.GRID_SIZE}")
    print(f"Canales: {config_class.CHANNELS}")
    print(f"Leyes: {config_class.NUM_LAWS}")
    print(f"IIT Φ: {'Habilitado' if config_class.PHI_CALCULATION_ENABLED else 'Deshabilitado'}")
    print(f"Batch Size: {config_class.get_optimal_batch_size()}")
    
    warnings = config_class.validate_config()
    if warnings:
        print("\n⚠️  WARNINGS:")
        for w in warnings:
            print(f"  - {w}")
    
    print("=" * 50)

if __name__ == "__main__":
    # Test de configuración
    print_config_summary()
    
    # Test de configuraciones alternativas
    print("\n🔬 CONFIGURACIÓN RESEARCH:")
    print_config_summary(ResearchConfig)
