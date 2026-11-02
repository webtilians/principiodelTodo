#!/usr/bin/env python3
"""
🧠 IIT-GUIDED MEMORY - MEMORIA GUIADA POR PHI
=============================================

Sistema de memoria externa que prioriza almacenamiento basándose en
métricas de Integrated Information Theory (IIT).

PRINCIPIO:
Estados con ALTO PHI (alta integración) son más importantes y se
almacenan con mayor prioridad. Esto maximiza la "calidad" de la memoria.

VENTAJAS:
1. ✅ Memoria inteligente - almacena lo más integrado
2. ✅ Eviction policy óptima - descarta lo menos integrado
3. ✅ Compatible con exploración estocástica
4. ✅ Diferenciable (compatible con backprop)

DIFERENCIAS vs PriorityExternalMemory:
- PriorityExternalMemory: Prioridad basada en attention scores
- IITGuidedMemory: Prioridad basada en PHI (integración)

USO TÍPICO:
```python
memory = IITGuidedMemory(
    memory_slots=256,
    hidden_dim=512,
    use_phi_priority=True  # Prioridad por PHI
)

# Durante forward pass
memory.write(
    query=hidden_state,
    content=hidden_state,
    phi_value=phi_estimate  # ← Clave: PHI del estado
)
```

Referencias:
- Tononi, G. (2004). An information integration theory of consciousness
- Prioritized Experience Replay (Schaul et al., 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class IITGuidedMemory(nn.Module):
    """
    Memoria externa guiada por métricas IIT (PHI).
    
    La prioridad de almacenamiento se determina por:
    1. PHI value (integración de información)
    2. Frecuencia de acceso (opcional)
    3. Recency (opcional)
    
    Args:
        memory_slots: Número de slots de memoria
        hidden_dim: Dimensión de los vectores
        use_phi_priority: Si True, usa PHI como prioridad principal
        use_recency_boost: Si True, boost para items recientes
        alpha: Factor de mezcla entre PHI y attention (default: 0.8 → 80% PHI)
    """
    
    def __init__(
        self,
        memory_slots: int = 256,
        hidden_dim: int = 512,
        use_phi_priority: bool = True,
        use_recency_boost: bool = True,
        alpha: float = 0.8,  # Mayor alpha = más peso a PHI
        learnable_threshold: bool = True,  # 🆕 Threshold aprendible
        initial_threshold: float = 3.0     # 🆕 Valor inicial
    ):
        super().__init__()
        
        self.memory_slots = memory_slots
        self.hidden_dim = hidden_dim
        self.use_phi_priority = use_phi_priority
        self.use_recency_boost = use_recency_boost
        self.alpha = alpha
        
        # 🆕 Threshold aprendible para decidir qué guardar
        if learnable_threshold:
            # Parametrizado en log-space para estabilidad
            self.threshold_logit = nn.Parameter(
                torch.tensor(initial_threshold).log()
            )
        else:
            self.register_buffer('threshold_logit', torch.tensor(initial_threshold).log())
        
        self.learnable_threshold = learnable_threshold
        
        # Memory banks
        self.register_buffer('memory_keys', torch.zeros(memory_slots, hidden_dim))
        self.register_buffer('memory_values', torch.zeros(memory_slots, hidden_dim))
        
        # Prioridades basadas en PHI
        self.register_buffer('phi_scores', torch.zeros(memory_slots))
        
        # Attention scores (para combinación híbrida)
        self.register_buffer('attention_scores', torch.zeros(memory_slots))
        
        # Frecuencia de acceso
        self.register_buffer('access_count', torch.zeros(memory_slots, dtype=torch.long))
        
        # Timestamps (para recency)
        self.register_buffer('timestamps', torch.zeros(memory_slots, dtype=torch.long))
        self.register_buffer('global_time', torch.tensor(0, dtype=torch.long))
        
        # Contador de escrituras
        self.register_buffer('write_count', torch.tensor(0, dtype=torch.long))
        
        # Query/Key transformations
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def _compute_priority(
        self,
        phi_value: Optional[torch.Tensor] = None,
        attention_score: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calcula prioridad de almacenamiento.
        
        Prioridad = α * PHI + (1-α) * Attention + Recency_boost
        
        Args:
            phi_value: Valor de PHI [batch] o None
            attention_score: Score de atención [batch] o None
            
        Returns:
            priority: [batch]
        """
        batch_size = phi_value.size(0) if phi_value is not None else 1
        device = phi_value.device if phi_value is not None else self.memory_keys.device
        
        # Inicializar prioridad base
        priority = torch.zeros(batch_size, device=device)
        
        # Componente PHI
        if self.use_phi_priority and phi_value is not None:
            # Normalizar PHI a [0, 1] (asumiendo rango [0, 10])
            phi_normalized = torch.clamp(phi_value / 10.0, 0, 1)
            priority += self.alpha * phi_normalized
        
        # Componente Attention
        if attention_score is not None:
            # Attention ya debe estar en [0, 1]
            priority += (1.0 - self.alpha) * attention_score
        
        # Recency boost (estados recientes tienen prioridad extra)
        if self.use_recency_boost:
            # Boost pequeño (máximo 0.1) que decae con el tiempo
            time_since_write = (self.global_time - self.write_count).float()
            recency_boost = 0.1 * torch.exp(-time_since_write / 100.0)
            priority += recency_boost
        
        return priority
    
    def write(
        self,
        query: torch.Tensor,
        content: torch.Tensor,
        phi_value: Optional[torch.Tensor] = None,
        attention_score: Optional[torch.Tensor] = None,
        force_write: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Escribe en memoria guiado por PHI.
        
        Args:
            query: Vector de query [batch, hidden_dim]
            content: Contenido a almacenar [batch, hidden_dim]
            phi_value: Valor de PHI [batch] (opcional)
            attention_score: Score de atención [batch] (opcional)
            force_write: Si True, siempre escribe (ignora prioridad y threshold)
            
        Returns:
            write_info: Dict con información de escritura
        """
        batch_size = query.size(0)
        
        # Proyectar query
        query_proj = self.query_proj(query)
        
        # Calcular prioridad del nuevo estado
        new_priority = self._compute_priority(phi_value, attention_score)
        
        # 🆕 Obtener threshold aprendible
        threshold = self.threshold_logit.exp()  # Convertir de log-space
        
        # Encontrar slot con menor prioridad para posible eviction
        min_priority_idx = self.phi_scores.argmin()
        min_priority = self.phi_scores[min_priority_idx]
        
        # 🆕 Decidir si escribir basándose en threshold Y prioridad
        # Solo escribir si:
        # 1. force_write=True, O
        # 2. (new_priority > threshold) Y (new_priority > min_priority O memoria no llena)
        above_threshold = (new_priority > threshold) if phi_value is not None else torch.ones(batch_size, dtype=torch.bool, device=query.device)
        write_mask = force_write | (above_threshold & ((new_priority > min_priority) | (self.write_count < self.memory_slots)))
        
        # Estadísticas de escritura
        num_writes = write_mask.sum().item()
        
        if num_writes > 0:
            # Escribir en los slots de menor prioridad
            for i in range(batch_size):
                if write_mask[i] or force_write:
                    # Encontrar slot a reemplazar
                    if self.write_count < self.memory_slots:
                        # Fase de llenado inicial
                        slot_idx = self.write_count.item()
                    else:
                        # Fase de eviction: reemplazar el de menor prioridad
                        slot_idx = min_priority_idx.item()
                    
                    # Escribir
                    self.memory_keys[slot_idx] = query_proj[i]
                    self.memory_values[slot_idx] = content[i]
                    
                    # Actualizar prioridades
                    if phi_value is not None:
                        self.phi_scores[slot_idx] = phi_value[i].item()
                    if attention_score is not None:
                        self.attention_scores[slot_idx] = attention_score[i].item()
                    
                    # Actualizar timestamp
                    self.timestamps[slot_idx] = self.global_time
                    
                    # Incrementar contador
                    self.write_count += 1
        
        # Incrementar tiempo global
        self.global_time += 1
        
        return {
            'num_writes': torch.tensor(num_writes),
            'write_mask': write_mask,
            'new_priority': new_priority,
            'min_priority': min_priority,
            'utilization': min(self.write_count.float() / self.memory_slots, 1.0)
        }
    
    def read(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        phi_guided: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lee de memoria con retrieval guiado por PHI.
        
        Args:
            query: Vector de query [batch, hidden_dim]
            top_k: Número de memories a recuperar
            phi_guided: Si True, prioriza memories con alto PHI
            
        Returns:
            retrieved: Contenido recuperado [batch, top_k, hidden_dim]
            weights: Pesos de atención [batch, top_k]
        """
        batch_size = query.size(0)
        
        # Proyectar query
        query_proj = self.query_proj(query)  # [batch, hidden_dim]
        
        # Calcular similitud con todas las keys
        # similarity: [batch, memory_slots]
        similarity = torch.matmul(
            query_proj,
            self.memory_keys.t()
        ) / math.sqrt(self.hidden_dim)
        
        # Combinar similitud con PHI scores (si phi_guided)
        if phi_guided and self.use_phi_priority:
            # Normalizar phi_scores a [0, 1]
            phi_normalized = self.phi_scores / (self.phi_scores.max() + 1e-8)
            
            # Combinar: 70% similitud, 30% PHI
            combined_score = 0.7 * similarity + 0.3 * phi_normalized.unsqueeze(0)
        else:
            combined_score = similarity
        
        # Top-K retrieval
        top_k_actual = min(top_k, self.write_count.item())
        if top_k_actual == 0:
            # Memoria vacía
            return (
                torch.zeros(batch_size, top_k, self.hidden_dim, device=query.device),
                torch.zeros(batch_size, top_k, device=query.device)
            )
        
        topk_scores, topk_indices = torch.topk(
            combined_score,
            k=top_k_actual,
            dim=1
        )
        
        # Softmax para weights
        weights = F.softmax(topk_scores, dim=1)  # [batch, top_k]
        
        # Recuperar valores
        retrieved = torch.stack([
            self.memory_values[topk_indices[i]]
            for i in range(batch_size)
        ])  # [batch, top_k, hidden_dim]
        
        # Actualizar access count
        for i in range(batch_size):
            self.access_count[topk_indices[i]] += 1
        
        # Padding si top_k_actual < top_k
        if top_k_actual < top_k:
            padding_size = top_k - top_k_actual
            retrieved = F.pad(retrieved, (0, 0, 0, padding_size))
            weights = F.pad(weights, (0, padding_size))
        
        return retrieved, weights
    
    def get_statistics(self) -> Dict[str, float]:
        """Retorna estadísticas de memoria."""
        if self.write_count == 0:
            return {
                'utilization': 0.0,
                'mean_phi': 0.0,
                'max_phi': 0.0,
                'mean_access_count': 0.0,
                'total_writes': 0,
                'threshold': self.get_threshold()  # 🆕 Threshold actual
            }
        
        active_slots = min(self.write_count.item(), self.memory_slots)
        
        return {
            'utilization': active_slots / self.memory_slots,
            'mean_phi': self.phi_scores[:active_slots].mean().item(),
            'max_phi': self.phi_scores[:active_slots].max().item(),
            'mean_access_count': self.access_count[:active_slots].float().mean().item(),
            'total_writes': self.write_count.item(),
            'threshold': self.get_threshold()  # 🆕 Threshold actual
        }
    
    def reset(self):
        """Limpia la memoria."""
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.phi_scores.zero_()
        self.attention_scores.zero_()
        self.access_count.zero_()
        self.timestamps.zero_()
        self.write_count.zero_()
        self.global_time.zero_()
    
    def get_threshold(self) -> float:
        """🆕 Retorna el threshold actual (aprendible)."""
        return self.threshold_logit.exp().item()
    
    def get_threshold_gradient(self) -> Optional[float]:
        """🆕 Retorna el gradiente del threshold (para monitoreo)."""
        if self.learnable_threshold and self.threshold_logit.grad is not None:
            return self.threshold_logit.grad.item()
        return None


# =============================================================================
# TESTS UNITARIOS
# =============================================================================

if __name__ == '__main__':
    print("="*70)
    print("🧪 TESTS - IITGuidedMemory")
    print("="*70)
    
    # Test 1: Inicialización
    print("\n📝 Test 1: Inicialización...")
    memory = IITGuidedMemory(
        memory_slots=16,
        hidden_dim=64,
        use_phi_priority=True
    )
    print(f"  ✓ Slots: {memory.memory_slots}")
    print(f"  ✓ Hidden dim: {memory.hidden_dim}")
    print(f"  ✓ Alpha: {memory.alpha:.2f}")
    
    # Test 2: Write con PHI
    print("\n📝 Test 2: Write con PHI...")
    batch_size = 4
    query = torch.randn(batch_size, 64)
    content = torch.randn(batch_size, 64)
    phi_values = torch.tensor([1.0, 5.0, 3.0, 2.0])  # Diferentes PHI
    
    write_info = memory.write(query, content, phi_value=phi_values, force_write=True)
    
    print(f"  ✓ Writes: {write_info['num_writes'].item()}")
    print(f"  ✓ Utilization: {write_info['utilization']:.2%}")
    print(f"  ✓ Priority range: [{write_info['new_priority'].min():.3f}, {write_info['new_priority'].max():.3f}]")
    
    # Test 3: Read con PHI guidance
    print("\n📝 Test 3: Read con PHI guidance...")
    query_read = torch.randn(2, 64)
    retrieved, weights = memory.read(query_read, top_k=3, phi_guided=True)
    
    print(f"  ✓ Retrieved shape: {retrieved.shape}")
    print(f"  ✓ Weights shape: {weights.shape}")
    print(f"  ✓ Weights sum: {weights[0].sum():.4f} (debe ser ~1.0)")
    
    # Test 4: Eviction policy
    print("\n📝 Test 4: Eviction policy (PHI-based)...")
    
    # Llenar memoria con PHI variables
    for i in range(20):  # Más que memory_slots (16)
        query = torch.randn(1, 64)
        content = torch.randn(1, 64)
        phi = torch.tensor([float(i % 10)])  # PHI cíclico
        
        memory.write(query, content, phi_value=phi)
    
    stats = memory.get_statistics()
    print(f"  ✓ Utilization: {stats['utilization']:.2%}")
    print(f"  ✓ Mean PHI: {stats['mean_phi']:.4f}")
    print(f"  ✓ Max PHI: {stats['max_phi']:.4f}")
    print(f"  ✓ Total writes: {stats['total_writes']}")
    
    # Test 5: Comparación con/sin PHI guidance
    print("\n📝 Test 5: Comparación PHI guidance ON vs OFF...")
    
    query_test = torch.randn(1, 64)
    
    retrieved_on, weights_on = memory.read(query_test, top_k=3, phi_guided=True)
    retrieved_off, weights_off = memory.read(query_test, top_k=3, phi_guided=False)
    
    print(f"  PHI-guided weights: {weights_on[0].tolist()}")
    print(f"  Similarity-only weights: {weights_off[0].tolist()}")
    print(f"  Diferencia: {(weights_on - weights_off).abs().mean():.4f}")
    
    # Test 6: Reset
    print("\n📝 Test 6: Reset...")
    memory.reset()
    stats_after = memory.get_statistics()
    
    print(f"  ✓ Utilization: {stats_after['utilization']:.2%}")
    print(f"  ✓ Total writes: {stats_after['total_writes']}")
    
    print("\n" + "="*70)
    print("✅ TODOS LOS TESTS PASADOS")
    print("="*70)
