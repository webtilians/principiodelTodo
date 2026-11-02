"""
🎲 MÓDULO DE EXPLORACIÓN ESTOCÁSTICA
===================================

Sistemas de ruido e exploración para redes neuronales.

NOTA IMPORTANTE:
- Esto NO es computación cuántica real
- Son heurísticas de ruido gaussiano/uniforme
- Útiles para exploración y regularización
- Terminología honesta: "stochastic" en lugar de "quantum"

Anteriormente llamado: "quantum noise injection" (misleading)
Ahora: "stochastic exploration" (honest)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class StochasticExploration(nn.Module):
    """
    Añade ruido estocástico para exploración del espacio de estados.
    
    Esto ayuda a:
    - Evitar mínimos locales
    - Aumentar diversidad de representaciones
    - Regularización implícita
    
    NO simula efectos cuánticos reales.
    """
    
    def __init__(self, noise_scale: float = 0.1, noise_type: str = 'gaussian'):
        """
        Args:
            noise_scale: Escala del ruido a añadir
            noise_type: 'gaussian', 'uniform', o 'dropout'
        """
        super().__init__()
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        
        # Parámetros aprendibles para control adaptativo de ruido
        self.adaptive_scale = nn.Parameter(torch.tensor(noise_scale))
        
    def forward(self, hidden_state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Añade ruido al estado oculto.
        
        Args:
            hidden_state: Tensor a perturbar
            training: Si False, no añade ruido
            
        Returns:
            Tensor perturbado
        """
        if not training:
            return hidden_state
        
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(hidden_state) * self.adaptive_scale
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(hidden_state) - 0.5) * 2 * self.adaptive_scale
        elif self.noise_type == 'dropout':
            # Dropout estocástico
            mask = torch.bernoulli(torch.ones_like(hidden_state) * 0.9)
            return hidden_state * mask
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return hidden_state + noise
    
    def get_noise_statistics(self) -> dict:
        """Retorna estadísticas del ruido para análisis."""
        return {
            'current_scale': self.adaptive_scale.item(),
            'noise_type': self.noise_type
        }


class MultiScaleNoise(nn.Module):
    """
    Ruido multi-escala para diferentes niveles de perturbación.
    
    Combina:
    - Ruido de alta frecuencia (local)
    - Ruido de baja frecuencia (global)
    """
    
    def __init__(self, high_freq_scale: float = 0.05, low_freq_scale: float = 0.15):
        super().__init__()
        self.high_freq_scale = high_freq_scale
        self.low_freq_scale = low_freq_scale
        
    def forward(self, hidden_state: torch.Tensor, training: bool = True) -> torch.Tensor:
        if not training:
            return hidden_state
        
        # Ruido de alta frecuencia (independiente por elemento)
        high_freq_noise = torch.randn_like(hidden_state) * self.high_freq_scale
        
        # Ruido de baja frecuencia (correlacionado en secuencia)
        batch_size, seq_len, hidden_dim = hidden_state.shape
        low_freq_noise = torch.randn(batch_size, 1, hidden_dim, device=hidden_state.device)
        low_freq_noise = low_freq_noise.expand(-1, seq_len, -1) * self.low_freq_scale
        
        return hidden_state + high_freq_noise + low_freq_noise


class AdaptiveNoiseScheduler:
    """
    Scheduler de ruido adaptativo que reduce ruido durante entrenamiento.
    
    Similar a learning rate scheduling pero para ruido de exploración.
    """
    
    def __init__(self, initial_scale: float = 0.3, final_scale: float = 0.05, 
                 total_steps: int = 10000):
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.total_steps = total_steps
        self.current_step = 0
        
    def step(self) -> float:
        """Retorna escala de ruido para el step actual."""
        self.current_step += 1
        progress = min(self.current_step / self.total_steps, 1.0)
        
        # Decaimiento coseno
        scale = self.final_scale + 0.5 * (self.initial_scale - self.final_scale) * \
                (1 + np.cos(np.pi * progress))
        
        return scale
    
    def reset(self):
        """Resetea el scheduler."""
        self.current_step = 0


# ============================================================================
# LEGACY: Mantener por compatibilidad
# ============================================================================

class QuantumNoiseInjection(StochasticExploration):
    """
    DEPRECATED: Nombre misleading.
    Usar StochasticExploration en su lugar.
    
    Mantenido solo para compatibilidad con código existente.
    """
    
    def __init__(self, noise_scale: float = 0.1):
        super().__init__(noise_scale=noise_scale, noise_type='gaussian')
        print("⚠️ WARNING: QuantumNoiseInjection is deprecated. "
              "Use StochasticExploration instead.")


def quantum_superposition(hidden_state: torch.Tensor, alpha_scale: float = 0.3, 
                         beta_scale: float = 0.3) -> torch.Tensor:
    """
    DEPRECATED: Esto NO es superposición cuántica real.
    Es simplemente ruido gaussiano.
    
    Mantenido por compatibilidad. Usar StochasticExploration.
    """
    alpha = torch.randn_like(hidden_state) * alpha_scale
    beta = torch.randn_like(hidden_state) * beta_scale
    return alpha * hidden_state + beta * (1 - hidden_state)


# ============================================================================
# OPTIONAL: Integración con QuTiP si está disponible
# ============================================================================

try:
    import qutip as qt
    QUTIP_AVAILABLE = True
    
    class RealQuantumNoise(nn.Module):
        """
        Ruido cuántico REAL usando QuTiP.
        
        EXPERIMENTAL: Requiere qutip instalado.
        Simula decoherencia cuántica real.
        """
        
        def __init__(self, decoherence_rate: float = 0.1):
            super().__init__()
            self.decoherence_rate = decoherence_rate
            
        def forward(self, hidden_state: torch.Tensor, training: bool = True) -> torch.Tensor:
            if not training:
                return hidden_state
            
            # Simular decoherencia cuántica
            batch_size, seq_len, hidden_dim = hidden_state.shape
            
            # Por cada elemento, simular como qubit con decoherencia
            noise = torch.zeros_like(hidden_state)
            
            for i in range(min(10, seq_len)):  # Limitar por performance
                # Estado inicial
                psi0 = qt.basis(2, 0)
                
                # Operador de decoherencia
                c_ops = [np.sqrt(self.decoherence_rate) * qt.sigmaz()]
                
                # Evolucionar
                times = np.linspace(0, 1, 10)
                result = qt.mesolve(qt.sigmaz(), psi0, times, c_ops)
                
                # Extraer ruido de decoherencia
                final_state = result.states[-1]
                noise_val = np.real(final_state.full()[0, 0])
                
                noise[:, i, :] = noise_val * 0.1
            
            return hidden_state + noise
    
except ImportError:
    QUTIP_AVAILABLE = False
    
    class RealQuantumNoise(nn.Module):
        """Placeholder cuando QuTiP no está disponible."""
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise ImportError("QuTiP not available. Install with: pip install qutip")


# ============================================================================
# UTILIDADES
# ============================================================================

def create_exploration_strategy(strategy: str = 'gaussian', **kwargs):
    """
    Factory function para crear estrategia de exploración.
    
    Args:
        strategy: 'gaussian', 'multiscale', 'quantum_real'
        **kwargs: Argumentos para el constructor
        
    Returns:
        Módulo de exploración estocástica
    """
    if strategy == 'gaussian':
        return StochasticExploration(noise_type='gaussian', **kwargs)
    elif strategy == 'uniform':
        return StochasticExploration(noise_type='uniform', **kwargs)
    elif strategy == 'multiscale':
        return MultiScaleNoise(**kwargs)
    elif strategy == 'quantum_real':
        if not QUTIP_AVAILABLE:
            print("⚠️ QuTiP not available, falling back to gaussian")
            return StochasticExploration(noise_type='gaussian', **kwargs)
        return RealQuantumNoise(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
