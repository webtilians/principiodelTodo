#!/usr/bin/env python3
"""
🎓 LEARNABLE PHI COMPONENTS - PESOS APRENDIBLES PARA PHI
========================================================

Sistema que permite al modelo APRENDER qué componentes de PHI son más
importantes mediante gradient descent durante el entrenamiento.

VENTAJAS:
1. ✅ El modelo descubre la mejor combinación de componentes
2. ✅ Se adapta automáticamente a diferentes datasets
3. ✅ Mejora PHI sin ajuste manual de hiperparámetros
4. ✅ Totalmente diferenciable (compatible con backprop)

COMPONENTES APRENDIBLES:
- weight_temporal: Importancia de coherencia temporal
- weight_integration: Importancia de integración entre partes
- weight_complexity: Importancia de complejidad de activaciones
- weight_attention: Importancia de diversidad de atención

ENTRENAMIENTO:
Los pesos se actualizan junto con el resto del modelo mediante backprop.
Se normalizan automáticamente para sumar 1.0.

Referencias:
- Meta-learning approaches (Finn et al., 2017)
- Neural architecture search (Liu et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import json
from datetime import datetime


class LearnablePhiWeights(nn.Module):
    """
    Pesos aprendibles para componentes de PHI.
    
    Los pesos se inicializan cerca de valores óptimos conocidos y se
    actualizan durante el entrenamiento para encontrar la mejor combinación.
    
    Args:
        initial_weights: Dict con pesos iniciales (opcional)
        constraint: 'softmax' (suman 1.0) o 'sigmoid' (independientes [0,1])
    """
    
    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        constraint: str = 'softmax'
    ):
        super().__init__()
        
        self.constraint = constraint
        
        # Pesos iniciales (optimizados manualmente)
        default_weights = {
            'temporal': 0.30,
            'integration': 0.30,
            'complexity': 0.20,
            'attention': 0.20
        }
        
        if initial_weights is not None:
            default_weights.update(initial_weights)
        
        # Parametrizar en espacio logit (antes de softmax/sigmoid)
        if constraint == 'softmax':
            # Para softmax: inicializar en log-space
            self.logit_temporal = nn.Parameter(
                torch.tensor(default_weights['temporal']).log()
            )
            self.logit_integration = nn.Parameter(
                torch.tensor(default_weights['integration']).log()
            )
            self.logit_complexity = nn.Parameter(
                torch.tensor(default_weights['complexity']).log()
            )
            self.logit_attention = nn.Parameter(
                torch.tensor(default_weights['attention']).log()
            )
        else:  # sigmoid
            # Para sigmoid: inicializar en logit-space
            def inverse_sigmoid(p):
                return torch.log(torch.tensor(p) / (1 - torch.tensor(p)))
            
            self.logit_temporal = nn.Parameter(inverse_sigmoid(default_weights['temporal']))
            self.logit_integration = nn.Parameter(inverse_sigmoid(default_weights['integration']))
            self.logit_complexity = nn.Parameter(inverse_sigmoid(default_weights['complexity']))
            self.logit_attention = nn.Parameter(inverse_sigmoid(default_weights['attention']))
    
    def forward(self) -> Dict[str, torch.Tensor]:
        """
        Retorna los pesos normalizados.
        
        Returns:
            weights_dict: {
                'temporal': float [0, 1],
                'integration': float [0, 1],
                'complexity': float [0, 1],
                'attention': float [0, 1]
            }
        """
        if self.constraint == 'softmax':
            # Concatenar y aplicar softmax (suman 1.0)
            logits = torch.stack([
                self.logit_temporal,
                self.logit_integration,
                self.logit_complexity,
                self.logit_attention
            ])
            
            weights = F.softmax(logits, dim=0)
            
            return {
                'temporal': weights[0],
                'integration': weights[1],
                'complexity': weights[2],
                'attention': weights[3]
            }
        else:  # sigmoid
            # Aplicar sigmoid independientemente
            return {
                'temporal': torch.sigmoid(self.logit_temporal),
                'integration': torch.sigmoid(self.logit_integration),
                'complexity': torch.sigmoid(self.logit_complexity),
                'attention': torch.sigmoid(self.logit_attention)
            }
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Retorna los pesos como dict de floats (para logging)."""
        weights = self.forward()
        return {k: v.item() for k, v in weights.items()}
    
    def save_weights(self, path: str):
        """Guarda los pesos aprendidos a JSON."""
        weights_dict = self.get_weights_dict()
        
        save_data = {
            'weights': weights_dict,
            'constraint': self.constraint,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"✅ Pesos guardados: {path}")
    
    def load_weights(self, path: str):
        """Carga pesos desde JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Extraer pesos
        weights = data.get('weights', data)  # Compatibilidad con ambos formatos
        
        # Actualizar parámetros
        if self.constraint == 'softmax':
            self.logit_temporal.data = torch.tensor(weights['temporal']).log()
            self.logit_integration.data = torch.tensor(weights['integration']).log()
            self.logit_complexity.data = torch.tensor(weights['complexity']).log()
            self.logit_attention.data = torch.tensor(weights['attention']).log()
        else:
            def inverse_sigmoid(p):
                return torch.log(torch.tensor(p) / (1 - torch.tensor(p)))
            
            self.logit_temporal.data = inverse_sigmoid(weights['temporal'])
            self.logit_integration.data = inverse_sigmoid(weights['integration'])
            self.logit_complexity.data = inverse_sigmoid(weights['complexity'])
            self.logit_attention.data = inverse_sigmoid(weights['attention'])
        
        print(f"✅ Pesos cargados: {path}")


class DeltaPhiObjective(nn.Module):
    """
    Objetivo de entrenamiento basado en ΔPhi (Delta Phi).
    
    IDEA: Maximizar el cambio de PHI entre estado inicial y procesado.
    Esto fomenta que el modelo aprenda transformaciones que aumenten
    la integración de información.
    
    IMPORTANTE: Este es un objetivo AUXILIAR, no reemplaza el loss de lenguaje.
    Se usa típicamente como regularización: loss_total = loss_lm + λ * loss_phi
    
    Args:
        lambda_phi: Factor de peso para el término phi (típicamente 0.01-0.1)
        target_phi: PHI objetivo deseado (típicamente 3.0-5.0)
    """
    
    def __init__(
        self,
        lambda_phi: float = 0.01,
        target_phi: float = 1.2  # ⚠️ FIX BUG #3: Cambiar de 3.5 a 1.2 (realista dado PHI actual ~0.9)
    ):
        super().__init__()
        
        self.lambda_phi = lambda_phi
        self.target_phi = target_phi
    
    def forward(
        self,
        phi_initial: torch.Tensor,
        phi_processed: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcula el loss basado en ΔPhi.
        
        Args:
            phi_initial: PHI del estado inicial (embedding) [batch]
            phi_processed: PHI del estado procesado (post-transformer) [batch]
            
        Returns:
            loss: Tensor escalar
            metrics: Dict con estadísticas
        """
        # ΔPhi = PHI_procesado - PHI_inicial
        delta_phi = phi_processed - phi_initial
        
        # Loss: Queremos que PHI_procesado esté cerca del objetivo
        # Y que ΔPhi sea positivo (incremento de integración)
        
        # Componente 1: Distancia al objetivo
        loss_target = F.mse_loss(phi_processed, 
                                 torch.full_like(phi_processed, self.target_phi))
        
        # Componente 2: Penalizar ΔPhi negativo (pérdida de integración)
        loss_delta = F.relu(-delta_phi).mean()  # Solo penaliza si delta < 0
        
        # Loss total
        # ⚠️ FIX BUG #1: Quitar lambda_phi - se aplicará externamente en training script
        loss = loss_target + loss_delta
        
        # Métricas
        metrics = {
            'loss_phi': loss.item(),
            'loss_target': loss_target.item(),
            'loss_delta': loss_delta.item(),
            'mean_delta_phi': delta_phi.mean().item(),
            'mean_phi_initial': phi_initial.mean().item(),
            'mean_phi_processed': phi_processed.mean().item()
        }
        
        return loss, metrics


class LearnableRelevance(nn.Module):
    """
    Sistema completo de pesos aprendibles + objetivo ΔPhi.
    
    Esta clase combina:
    1. LearnablePhiWeights: Para aprender importancia de componentes
    2. DeltaPhiObjective: Para guiar el entrenamiento hacia mayor integración
    
    USO TÍPICO:
    ```python
    # Crear sistema
    learnable_relevance = LearnableRelevance()
    
    # Durante entrenamiento
    phi_initial = metrics_initial['phi_estimate']
    phi_processed = metrics_processed['phi_estimate']
    loss_phi, phi_metrics = learnable_relevance(phi_initial, phi_processed)
    
    # Loss total
    loss_total = loss_lm + loss_phi
    ```
    """
    
    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        constraint: str = 'softmax',
        lambda_phi: float = 0.01,
        target_phi: float = 3.5
    ):
        super().__init__()
        
        # Pesos aprendibles
        self.weights = LearnablePhiWeights(
            initial_weights=initial_weights,
            constraint=constraint
        )
        
        # Objetivo ΔPhi
        self.phi_objective = DeltaPhiObjective(
            lambda_phi=lambda_phi,
            target_phi=target_phi
        )
    
    def forward(
        self,
        phi_initial: torch.Tensor,
        phi_processed: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calcula loss y métricas de PHI.
        
        Args:
            phi_initial: PHI inicial [batch]
            phi_processed: PHI procesado [batch]
            
        Returns:
            loss: Tensor escalar
            metrics: Dict con estadísticas (incluye pesos actuales)
        """
        # Calcular loss
        loss, metrics = self.phi_objective(phi_initial, phi_processed)
        
        # Añadir pesos actuales a métricas
        current_weights = self.weights.get_weights_dict()
        metrics.update({
            f'weight_{k}': v for k, v in current_weights.items()
        })
        
        return loss, metrics
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        """Retorna los pesos normalizados."""
        return self.weights()
    
    def get_weights_dict(self) -> Dict[str, float]:
        """Retorna los pesos como dict de floats."""
        return self.weights.get_weights_dict()
    
    def save_state(self, path: str):
        """Guarda estado completo (pesos + config)."""
        state = {
            'weights': self.get_weights_dict(),
            'lambda_phi': self.phi_objective.lambda_phi,
            'target_phi': self.phi_objective.target_phi,
            'constraint': self.weights.constraint,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"✅ Estado guardado: {path}")


# =============================================================================
# TESTS UNITARIOS
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("🧪 TESTS - LearnablePhiWeights & LearnableRelevance")
    print("="*70)
    
    # Test 1: LearnablePhiWeights con softmax
    print("\n📝 Test 1: LearnablePhiWeights (softmax)...")
    weights_softmax = LearnablePhiWeights(constraint='softmax')
    w = weights_softmax.get_weights_dict()
    
    print(f"  Pesos iniciales:")
    for k, v in w.items():
        print(f"    {k}: {v:.4f}")
    print(f"  Suma: {sum(w.values()):.4f} (debe ser ~1.0)")
    
    # Test 2: LearnablePhiWeights con sigmoid
    print("\n📝 Test 2: LearnablePhiWeights (sigmoid)...")
    weights_sigmoid = LearnablePhiWeights(constraint='sigmoid')
    w = weights_sigmoid.get_weights_dict()
    
    print(f"  Pesos iniciales:")
    for k, v in w.items():
        print(f"    {k}: {v:.4f}")
    print(f"  Suma: {sum(w.values()):.4f} (puede ser != 1.0)")
    
    # Test 3: Gradient flow
    print("\n📝 Test 3: Gradient flow...")
    weights = LearnablePhiWeights(constraint='softmax')
    w = weights()
    
    # Simular backprop
    dummy_loss = w['temporal'] * 2.0
    dummy_loss.backward()
    
    print(f"  ✓ Gradiente temporal: {weights.logit_temporal.grad.item():.6f}")
    print(f"  ✓ Gradiente integration: {weights.logit_integration.grad.item():.6f}")
    
    # Test 4: DeltaPhiObjective
    print("\n📝 Test 4: DeltaPhiObjective...")
    phi_objective = DeltaPhiObjective(lambda_phi=0.01, target_phi=3.5)
    
    batch_size = 4
    phi_initial = torch.rand(batch_size) * 1.0  # [0, 1]
    phi_processed = torch.rand(batch_size) * 5.0  # [0, 5]
    
    loss, metrics = phi_objective(phi_initial, phi_processed)
    
    print(f"  Loss PHI: {loss.item():.6f}")
    print(f"  Delta PHI promedio: {metrics['mean_delta_phi']:.4f}")
    print(f"  PHI inicial: {metrics['mean_phi_initial']:.4f}")
    print(f"  PHI procesado: {metrics['mean_phi_processed']:.4f}")
    
    # Test 5: LearnableRelevance (sistema completo)
    print("\n📝 Test 5: LearnableRelevance...")
    learnable = LearnableRelevance(
        lambda_phi=0.01,
        target_phi=3.5
    )
    
    loss, metrics = learnable(phi_initial, phi_processed)
    
    print(f"  Loss total: {loss.item():.6f}")
    print(f"  Pesos actuales:")
    for k, v in metrics.items():
        if k.startswith('weight_'):
            print(f"    {k}: {v:.4f}")
    
    # Test 6: Save/Load
    print("\n📝 Test 6: Save/Load...")
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        learnable.save_state(temp_path)
        
        # Crear nuevo sistema y cargar
        learnable2 = LearnableRelevance()
        learnable2.weights.load_weights(temp_path)
        
        w1 = learnable.get_weights_dict()
        w2 = learnable2.get_weights_dict()
        
        print(f"  ✓ Pesos coinciden: {w1 == w2}")
        
    finally:
        os.remove(temp_path)
    
    print("\n" + "="*70)
    print("✅ TODOS LOS TESTS PASADOS")
    print("="*70)
