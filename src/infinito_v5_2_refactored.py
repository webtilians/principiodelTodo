#!/usr/bin/env python3
"""
🧠 INFINITO V5.2 - REFACTORIZADO CON MÓDULOS CORE 🧠
==================================================

Versión mejorada que usa los módulos refactorizados de src/core/.

MEJORAS vs V5.1:
✅ Memoria con priorización inteligente (PriorityExternalMemory)
✅ Métricas de integración honesta (InformationIntegrationMetrics)
✅ Exploración estocástica en lugar de "quantum noise"
✅ Validación con métricas estándar (perplexity, BLEU)
✅ Código modular y mantenible

COMPATIBILIDAD:
- Mantiene la API de V5.1
- Puede cargar modelos entrenados con V5.1
- Resultados comparables con V5.1

USO:
    from infinito_v5_2_refactored import InfinitoV52Refactored
    
    model = InfinitoV52Refactored(
        hidden_dim=512,
        use_improved_memory=True  # Usa PriorityExternalMemory
    )
"""

import warnings
warnings.filterwarnings('default')
warnings.filterwarnings('ignore', category=UserWarning, module='scipy')

import math
import hashlib
from typing import Optional, Tuple, Dict, List, Any
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer

# 🆕 IMPORTS DE MÓDULOS REFACTORIZADOS
sys.path.insert(0, os.path.dirname(__file__))

from core import (
    PriorityExternalMemory,      # Sistema memoria mejorado
    LegacyExternalMemory,         # Compatibilidad con V5.1
    InformationIntegrationMetrics, # Métricas IIT honestas
    StochasticExploration,        # Exploración (antes "quantum")
    EnhancedMultiHeadAttention,   # Atención mejorada
    StandardNLPMetrics,           # Validación científica
    BenchmarkComparison           # Comparación contra baselines
)

# GloVe embeddings (opcional)
try:
    import gensim.downloader as api
    GLOVE_AVAILABLE = True
    print("🌐 GloVe embeddings available")
except ImportError:
    GLOVE_AVAILABLE = False
    print("⚠️  GloVe not available, using TF-IDF only")


# =============================================================================
# EJEMPLO DE USO DE MÓDULOS REFACTORIZADOS
# =============================================================================

class InfinitoV52Refactored(nn.Module):
    """
    Versión refactorizada del modelo INFINITO usando módulos core.
    
    Esta es una versión SIMPLIFICADA que demuestra cómo usar los nuevos módulos.
    Para el modelo completo, ver infinito_gpt_text_fixed.py
    """
    
    def __init__(
        self,
        vocab_size: int = 5000,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        memory_slots: int = 256,
        use_improved_memory: bool = True,  # 🆕 Usar PriorityExternalMemory
        use_stochastic_exploration: bool = True  # 🆕 Usar exploración estocástica
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_improved_memory = use_improved_memory
        self.use_stochastic_exploration = use_stochastic_exploration
        
        print(f"\n{'='*70}")
        print(f"🚀 INFINITO V5.2 - REFACTORIZADO")
        print(f"{'='*70}")
        print(f"  Memoria mejorada: {use_improved_memory}")
        print(f"  Exploración estocástica: {use_stochastic_exploration}")
        print(f"{'='*70}\n")
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        
        # 🆕 Memoria usando módulo refactorizado
        if use_improved_memory:
            print("  ✓ Usando PriorityExternalMemory (priorización inteligente)")
            self.memory = PriorityExternalMemory(
                memory_slots=memory_slots,
                slot_size=64,
                hidden_dim=hidden_dim
            )
        else:
            print("  ✓ Usando LegacyExternalMemory (compatibilidad V5.1)")
            self.memory = LegacyExternalMemory(
                memory_slots=memory_slots,
                slot_size=64,
                hidden_dim=hidden_dim
            )
        
        # 🆕 Métricas de integración (antes "consciencia")
        self.iit_metrics = InformationIntegrationMetrics(hidden_dim=hidden_dim)
        
        # 🆕 Exploración estocástica (antes "quantum noise")
        if use_stochastic_exploration:
            print("  ✓ Usando StochasticExploration (ruido gaussiano)")
            self.explorer = StochasticExploration(
                noise_scale=0.1,
                noise_type='gaussian'
            )
        else:
            self.explorer = None
        
        # Atención multi-head mejorada
        self.attention_layers = nn.ModuleList([
            EnhancedMultiHeadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1
            )
            for _ in range(num_layers)
        ])
        
        # Feed-forward
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])
        
        # LayerNorms
        self.layer_norms_1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms_2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # 🆕 Métricas estándar para validación
        self.nlp_metrics = StandardNLPMetrics()
        
        print("✅ Modelo inicializado correctamente\n")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_metrics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass con métricas de integración.
        
        Args:
            input_ids: [batch, seq_len] - IDs de tokens
            return_metrics: Si True, retorna métricas de integración
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            metrics: Dict con métricas (opcional)
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding[:, :seq_len, :]
        
        # Variables para métricas
        all_attention_weights = []
        
        # Transformer layers
        for i, (attn, ff, ln1, ln2) in enumerate(
            zip(self.attention_layers, self.ff_layers, 
                self.layer_norms_1, self.layer_norms_2)
        ):
            # Self-attention
            attn_out, attn_weights = attn(hidden, return_attention=True)
            hidden = ln1(hidden + attn_out)
            
            if attn_weights is not None:
                all_attention_weights.append(attn_weights)
            
            # Feed-forward
            ff_out = ff(hidden)
            hidden = ln2(hidden + ff_out)
            
            # 🆕 Exploración estocástica (opcional)
            if self.explorer is not None and self.training:
                hidden = self.explorer(hidden, training=True)
        
        # 🆕 Calcular score de integración (antes "consciencia")
        integration_level = torch.ones(batch_size, device=hidden.device)
        
        if return_metrics and len(all_attention_weights) > 0:
            # Stack attention weights
            stacked_attention = torch.stack(all_attention_weights, dim=1)  # [batch, layers, heads, seq, seq]
            avg_attention = stacked_attention.mean(dim=1)  # Promediar sobre layers
            
            # 🆕 Calcular métricas de integración usando módulo refactorizado
            metrics_dict = self.iit_metrics.calculate_all_metrics(
                hidden,
                avg_attention
            )
            
            # Usar phi_estimate como proxy de integración
            integration_level = metrics_dict['phi_estimate']
        
        # Interacción con memoria
        memory_query = hidden.mean(dim=1)  # [batch, hidden_dim]
        read_content, read_weights = self.memory.read(memory_query, integration_level)
        
        # Escribir en memoria
        memory_content = hidden.mean(dim=1)
        phi_value = integration_level.mean().item() if return_metrics else 0.0
        self.memory.write(memory_query, memory_content, integration_level, phi_value)
        
        # Output
        logits = self.output_projection(hidden)
        
        # Preparar métricas de retorno
        if return_metrics:
            metrics = {
                'integration_phi': metrics_dict['phi_estimate'].mean().item(),
                'coherence': metrics_dict['coherence'].mean().item(),
                'complexity': metrics_dict['complexity'].mean().item(),
                'pattern_diversity': metrics_dict['pattern_diversity'].mean().item(),
                'memory_utilization': self.memory.get_statistics()['utilization'] if hasattr(self.memory, 'get_statistics') else 0.0
            }
            
            # 🆕 Añadir nota sobre interpretación
            metrics['_note'] = (
                "Estas métricas son APROXIMACIONES. "
                "phi_estimate NO es consciencia validada, es integración de información. "
                "Siempre comparar contra baselines."
            )
            
            return logits, metrics
        
        return logits, None
    
    def calculate_perplexity(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor
    ) -> float:
        """
        🆕 Calcula perplexity usando métricas estándar.
        
        Esta es LA métrica que debes reportar en papers.
        """
        logits, _ = self.forward(input_ids, return_metrics=False)
        perplexity = self.nlp_metrics.calculate_perplexity(logits, target_ids)
        return perplexity
    
    def get_memory_statistics(self) -> Dict:
        """🆕 Obtiene estadísticas detalladas de memoria."""
        if hasattr(self.memory, 'get_statistics'):
            return self.memory.get_statistics()
        else:
            return {'note': 'Legacy memory does not provide statistics'}


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

def demo_refactored_model():
    """Demuestra cómo usar el modelo refactorizado."""
    
    print("\n" + "="*70)
    print("📚 DEMO: MODELO INFINITO V5.2 REFACTORIZADO")
    print("="*70 + "\n")
    
    # Crear modelo
    model = InfinitoV52Refactored(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        use_improved_memory=True,
        use_stochastic_exploration=True
    )
    
    # Datos de ejemplo
    batch_size, seq_len = 4, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    target_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward con métricas
    print("🔄 Forward pass...")
    logits, metrics = model(input_ids, return_metrics=True)
    
    print(f"\n📊 Métricas de integración:")
    for key, value in metrics.items():
        if not key.startswith('_'):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print(f"\n📝 {metrics['_note']}")
    
    # 🆕 Calcular perplexity (métrica estándar)
    print(f"\n📏 Métricas estándar:")
    perplexity = model.calculate_perplexity(input_ids, target_ids)
    print(f"  Perplexity: {perplexity:.2f}")
    
    # Estadísticas de memoria
    print(f"\n💾 Estadísticas de memoria:")
    mem_stats = model.get_memory_statistics()
    for key, value in mem_stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETADO")
    print("="*70 + "\n")


if __name__ == '__main__':
    demo_refactored_model()
