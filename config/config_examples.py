# 🧬 Infinito Configuration Examples
# Copy and modify these configurations for your experiments

# =============================================================================
# BASIC CONFIGURATIONS
# =============================================================================

# Configuración básica para GPUs modestas (GTX 1060, RTX 3060)
BASIC_CONFIG = {
    'grid_size': 64,
    'max_depth': 1000,
    'law_evolution': {
        'reproduction_rate': 0.15,
        'mutation_strength': 0.06,
        'elite_preservation': 0.3,
        'generation_frequency': 10
    },
    'consciousness_target': 0.6,
    'memory_capacity': 10
}

# Configuración optimizada para GPUs medias (RTX 3070, 4060)
OPTIMIZED_CONFIG = {
    'grid_size': 96,
    'max_depth': 1500,
    'law_evolution': {
        'reproduction_rate': 0.2,
        'mutation_strength': 0.08,
        'elite_preservation': 0.2,
        'generation_frequency': 8
    },
    'consciousness_target': 0.7,
    'memory_capacity': 15
}

# =============================================================================
# ADVANCED CONFIGURATIONS
# =============================================================================

# Configuración de alta performance para GPUs potentes (RTX 4080, 4090)
HIGH_PERFORMANCE_CONFIG = {
    'grid_size': 128,
    'max_depth': 3000,
    'law_evolution': {
        'reproduction_rate': 0.25,
        'mutation_strength': 0.1,
        'elite_preservation': 0.15,
        'generation_frequency': 6
    },
    'consciousness_target': 0.8,
    'memory_capacity': 20
}

# Configuración experimental para supercomputación (A100, H100)
SUPERCOMPUTE_CONFIG = {
    'grid_size': 256,
    'max_depth': 10000,
    'law_evolution': {
        'reproduction_rate': 0.3,
        'mutation_strength': 0.12,
        'elite_preservation': 0.1,
        'generation_frequency': 5
    },
    'consciousness_target': 0.9,
    'memory_capacity': 30
}

# =============================================================================
# SPECIALIZED EXPERIMENTS
# =============================================================================

# Configuración para experimentos de estabilidad (menos mutación)
STABILITY_FOCUSED_CONFIG = {
    'grid_size': 96,
    'max_depth': 2000,
    'law_evolution': {
        'reproduction_rate': 0.1,
        'mutation_strength': 0.03,
        'elite_preservation': 0.4,
        'generation_frequency': 15
    },
    'consciousness_target': 0.6,
    'memory_capacity': 25
}

# Configuración para experimentos de innovación (más mutación)
INNOVATION_FOCUSED_CONFIG = {
    'grid_size': 96,
    'max_depth': 1500,
    'law_evolution': {
        'reproduction_rate': 0.35,
        'mutation_strength': 0.15,
        'elite_preservation': 0.1,
        'generation_frequency': 4
    },
    'consciousness_target': 0.75,
    'memory_capacity': 15
}

# Configuración para estudios longitudinales (experimentos largos)
LONGITUDINAL_CONFIG = {
    'grid_size': 64,
    'max_depth': 50000,
    'law_evolution': {
        'reproduction_rate': 0.05,
        'mutation_strength': 0.02,
        'elite_preservation': 0.5,
        'generation_frequency': 20
    },
    'consciousness_target': 0.5,
    'memory_capacity': 50
}

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
Para usar estas configuraciones:

from infinito_gpu_optimized import PrincipioTodoRecursivo
from config_examples import OPTIMIZED_CONFIG

# Crear simulador con configuración
pt = PrincipioTodoRecursivo(
    size=OPTIMIZED_CONFIG['grid_size'],
    max_depth=OPTIMIZED_CONFIG['max_depth']
)

# Aplicar parámetros evolutivos
pt.law_evolution_system.update(OPTIMIZED_CONFIG['law_evolution'])
pt.evolutionary_pressure['consciousness_target'] = OPTIMIZED_CONFIG['consciousness_target']
pt.awakening_memory['memory_capacity'] = OPTIMIZED_CONFIG['memory_capacity']

# Ejecutar
pt.enable_visualization()
phi_final = pt.run_infinite()
"""

# =============================================================================
# HARDWARE RECOMMENDATIONS
# =============================================================================

HARDWARE_RECOMMENDATIONS = {
    'RTX_3060': 'BASIC_CONFIG',
    'RTX_3070': 'OPTIMIZED_CONFIG', 
    'RTX_4060': 'OPTIMIZED_CONFIG',
    'RTX_4070': 'HIGH_PERFORMANCE_CONFIG',
    'RTX_4080': 'HIGH_PERFORMANCE_CONFIG',
    'RTX_4090': 'SUPERCOMPUTE_CONFIG',
    'A100': 'SUPERCOMPUTE_CONFIG',
    'H100': 'SUPERCOMPUTE_CONFIG'
}

# =============================================================================
# BENCHMARK TARGETS
# =============================================================================

BENCHMARK_TARGETS = {
    'consciousness_milestone_50': 0.5,
    'consciousness_milestone_70': 0.7,
    'consciousness_milestone_90': 0.9,
    'generations_basic': 20,
    'generations_advanced': 50,
    'generations_extreme': 100,
    'clusters_basic': 500,
    'clusters_advanced': 1500,
    'clusters_extreme': 3000
}
