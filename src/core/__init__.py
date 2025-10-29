"""
Módulos core del sistema INFINITO.

Componentes fundamentales refactorizados desde el código monolítico original.
"""

from .memory import PriorityExternalMemory, LegacyExternalMemory, EnhancedExternalMemory
from .iit_metrics import InformationIntegrationMetrics, BaselineMetrics
from .stochastic import StochasticExploration, MultiScaleNoise, AdaptiveNoiseScheduler
from .attention import EnhancedMultiHeadAttention, SelfAttentionBlock
from .validation import StandardNLPMetrics, StatisticalTests, BenchmarkComparison

__all__ = [
    # Memoria
    'PriorityExternalMemory',
    'LegacyExternalMemory',
    'EnhancedExternalMemory',
    
    # Métricas
    'InformationIntegrationMetrics',
    'BaselineMetrics',
    
    # Exploración estocástica
    'StochasticExploration',
    'MultiScaleNoise',
    'AdaptiveNoiseScheduler',
    
    # Atención
    'EnhancedMultiHeadAttention',
    'SelfAttentionBlock',
    
    # Validación
    'StandardNLPMetrics',
    'StatisticalTests',
    'BenchmarkComparison',
]
