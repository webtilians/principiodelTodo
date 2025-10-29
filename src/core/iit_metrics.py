"""
📊 MÓDULO DE MÉTRICAS DE INTEGRACIÓN DE INFORMACIÓN
=================================================

Implementación de métricas inspiradas en IIT (Integrated Information Theory).

NOTA IMPORTANTE:
- Estas son APROXIMACIONES SIMPLIFICADAS de métricas IIT
- NO son implementaciones completas de la teoría
- Los valores NO deben interpretarse como "nivel de consciencia"
- Son útiles como proxies de complejidad e integración de información

Referencias:
- Tononi, G. (2004). An information integration theory of consciousness
- Oizumi, M. et al. (2014). From the Phenomenology to the Mechanisms of Consciousness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class InformationIntegrationMetrics(nn.Module):
    """
    Calcula métricas de integración de información.
    
    Estas métricas son APROXIMACIONES y deben usarse con cautela:
    - Phi (Φ): Integración de información (simplificada)
    - Coherencia: Consistencia en patrones de atención
    - Complejidad: Riqueza de patrones emergentes
    """
    
    def __init__(self, hidden_dim=512, epsilon=1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        
        # Proyecciones para cálculo de métricas
        self.phi_projector = nn.Linear(hidden_dim, hidden_dim // 2)
        self.complexity_analyzer = nn.Linear(hidden_dim, 64)
        
    def calculate_phi_approximation(self, hidden_state: torch.Tensor, 
                                     attention_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Calcula una APROXIMACIÓN SIMPLIFICADA de Φ (Phi).
        
        ADVERTENCIA: Esto NO es el Φ completo de IIT. Es una heurística basada en:
        - Entropía de información
        - Integración entre componentes
        - Diversidad de activaciones
        
        Args:
            hidden_state: Tensor [batch, seq_len, hidden_dim]
            attention_weights: Opcional [batch, heads, seq_len, seq_len]
            
        Returns:
            phi_estimate: Tensor [batch] con estimación de integración
        """
        batch_size, seq_len, _ = hidden_state.shape
        
        # Proyectar a espacio reducido
        projected = self.phi_projector(hidden_state)  # [batch, seq_len, hidden_dim//2]
        
        # 1. Componente de entropía (diversidad de estados)
        # Normalizar por token
        normalized = F.normalize(projected, p=2, dim=-1)
        
        # Calcular matriz de covarianza aproximada
        # cov ≈ X^T X / n
        cov_matrix = torch.bmm(normalized.transpose(1, 2), normalized) / seq_len
        
        # Entropía basada en eigenvalues de covarianza
        # (aproximación: usar traza y determinante)
        trace = torch.diagonal(cov_matrix, dim1=-2, dim2=-1).sum(dim=-1)
        
        # 2. Componente de integración (mutual information aproximada)
        # Calcular correlaciones entre diferentes partes
        if seq_len > 1:
            half_len = seq_len // 2
            first_half = normalized[:, :half_len, :]
            second_half = normalized[:, half_len:2*half_len, :]
            
            # Correlación cruzada
            cross_corr = (first_half * second_half).sum(dim=[1, 2]) / (half_len * projected.size(-1))
            integration = torch.abs(cross_corr)
        else:
            integration = torch.ones(batch_size, device=hidden_state.device)
        
        # 3. Componente de complejidad (varianza en activaciones)
        variance = projected.var(dim=[1, 2])
        
        # Combinar componentes (heurística)
        phi_estimate = (
            0.4 * trace +           # Diversidad
            0.4 * integration +     # Integración
            0.2 * variance          # Complejidad
        )
        
        # Normalizar a rango [0, 10] aproximadamente
        phi_estimate = torch.clamp(phi_estimate * 2.0, 0, 10)
        
        return phi_estimate
    
    def calculate_coherence(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Calcula coherencia en patrones de atención.
        
        IMPORTANTE: Esto es simplemente una medida de CONSISTENCIA en la atención,
        no una medida de "coherencia consciente".
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len]
            
        Returns:
            coherence: Tensor [batch] con score de coherencia
        """
        if attention_weights is None:
            return torch.zeros(1, device=next(self.parameters()).device)
        
        batch_size = attention_weights.size(0)
        
        # Promediar sobre heads
        avg_attention = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
        
        # Coherencia = qué tan "enfocada" está la atención
        # Entropía baja = alta coherencia
        entropy = -torch.sum(avg_attention * torch.log(avg_attention + self.epsilon), dim=-1)
        avg_entropy = entropy.mean(dim=-1)
        
        # Normalizar: coherencia alta = entropía baja
        max_entropy = np.log(attention_weights.size(-1))
        coherence = 1.0 - (avg_entropy / max_entropy)
        
        return coherence
    
    def detect_complex_patterns(self, hidden_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detecta patrones complejos en el estado oculto.
        
        ADVERTENCIA: Los "patrones" detectados pueden ser pareidolia estadística.
        Usar solo como heurística, no como evidencia de emergencia real.
        
        Returns:
            Dict con métricas de complejidad:
                - complexity_score: Score de complejidad general
                - pattern_diversity: Diversidad de patrones
                - activation_entropy: Entropía de activaciones
        """
        batch_size, seq_len, _ = hidden_state.shape
        
        # Analizar complejidad
        complexity_features = self.complexity_analyzer(hidden_state)  # [batch, seq_len, 64]
        
        # 1. Score de complejidad (varianza normalizada)
        normalized_features = F.normalize(complexity_features, p=2, dim=-1)
        complexity_score = normalized_features.var(dim=[1, 2])
        
        # 2. Diversidad de patrones (número efectivo de patterns únicos)
        # Usar similitud coseno para agrupar
        similarity_matrix = torch.bmm(normalized_features, normalized_features.transpose(1, 2))
        similarity_matrix = similarity_matrix / (seq_len + self.epsilon)
        
        # Contar patterns "únicos" (similitud < threshold)
        unique_mask = (similarity_matrix < 0.8).float()
        pattern_diversity = unique_mask.sum(dim=[1, 2]) / (seq_len * seq_len)
        
        # 3. Entropía de activaciones
        probs = F.softmax(complexity_features, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + self.epsilon), dim=-1)
        activation_entropy = entropy.mean(dim=-1)
        
        return {
            'complexity_score': complexity_score,
            'pattern_diversity': pattern_diversity,
            'activation_entropy': activation_entropy
        }
    
    def calculate_all_metrics(self, hidden_state: torch.Tensor, 
                              attention_weights: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Calcula todas las métricas en una sola llamada.
        
        USAR ESTO para análisis, con la advertencia de que son métricas
        experimentales y no validadas científicamente.
        """
        phi = self.calculate_phi_approximation(hidden_state, attention_weights)
        coherence = self.calculate_coherence(attention_weights)
        patterns = self.detect_complex_patterns(hidden_state)
        
        return {
            'phi_estimate': phi,
            'coherence': coherence,
            'complexity': patterns['complexity_score'],
            'pattern_diversity': patterns['pattern_diversity'],
            'activation_entropy': patterns['activation_entropy']
        }


class BaselineMetrics:
    """
    Métricas baseline para comparación.
    
    CRÍTICO: Siempre comparar tus métricas contra:
    - Modelo random
    - Modelo pre-entrenado estándar
    - Valores teóricos esperados
    """
    
    @staticmethod
    def random_baseline(batch_size: int, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """Retorna métricas de un modelo completamente random."""
        return {
            'phi_estimate': torch.rand(batch_size, device=device) * 2.0,  # Random [0, 2]
            'coherence': torch.rand(batch_size, device=device) * 0.3,     # Random [0, 0.3]
            'complexity': torch.rand(batch_size, device=device) * 0.1     # Random [0, 0.1]
        }
    
    @staticmethod
    def compare_to_baseline(metrics: Dict[str, torch.Tensor], 
                           baseline: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compara métricas contra baseline.
        
        Retorna ratios: >1.0 significa mejor que baseline.
        """
        comparisons = {}
        for key in metrics:
            if key in baseline:
                metric_val = metrics[key].mean().item()
                baseline_val = baseline[key].mean().item()
                comparisons[f'{key}_ratio'] = metric_val / (baseline_val + 1e-6)
        
        return comparisons
