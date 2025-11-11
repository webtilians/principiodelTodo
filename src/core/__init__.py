"""
🧠 INFINITO CORE MODULES - IMPROVED IIT SYSTEM
==============================================

Módulos mejorados con sistema IIT (Integrated Information Theory) completo.

NUEVOS MÓDULOS (FASE 2):
- iit_metrics_improved: Métricas PHI mejoradas (4 componentes)
- phi_learnable: Pesos aprendibles para componentes PHI
- iit_guided_memory: Memoria guiada por PHI
"""

from .memory import PriorityExternalMemory, LegacyExternalMemory, EnhancedExternalMemory
from .iit_metrics import InformationIntegrationMetrics, BaselineMetrics
from .stochastic import StochasticExploration, MultiScaleNoise, AdaptiveNoiseScheduler
from .attention import EnhancedMultiHeadAttention, SelfAttentionBlock
from .validation import StandardNLPMetrics, StatisticalTests, BenchmarkComparison

# NUEVOS IMPORTS - FASE 2 (IIT MEJORADO)
from .iit_metrics_improved import ImprovedIITMetrics, random_baseline_metrics
from .phi_learnable import LearnablePhiWeights, DeltaPhiObjective, LearnableRelevance
from .iit_guided_memory import IITGuidedMemory

__all__ = [
    # Memoria
    'PriorityExternalMemory',
    'LegacyExternalMemory',
    'EnhancedExternalMemory',
    
    # Métricas
    'InformationIntegrationMetrics',
    'BaselineMetrics',
    
    # IIT MEJORADO (FASE 2)
    'ImprovedIITMetrics',
    'random_baseline_metrics',
    'LearnablePhiWeights',
    'DeltaPhiObjective',
    'LearnableRelevance',
    'IITGuidedMemory',
    
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
