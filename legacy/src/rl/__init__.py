"""
Módulo de Aprendizaje por Refuerzo (RL) para INFINITO.

Contiene entornos Gym y utilidades para entrenar agentes que controlen
dinámicamente el balance entre optimización de texto y PHI.
"""

from .infinito_rl_env import InfinitoRLEnv
from .rich_metrics_callback import RichMetricsCallback

__all__ = ["InfinitoRLEnv", "RichMetricsCallback"]
