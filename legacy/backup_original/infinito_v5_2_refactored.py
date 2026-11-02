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
    InformationIntegrationMetrics, # Métricas IIT honestas (original)
    ImprovedIITMetrics,           # 🆕 Métricas IIT mejoradas (4 componentes)
    LearnablePhiWeights,          # 🆕 Pesos aprendibles para PHI
    DeltaPhiObjective,            # 🆕 Objetivo auxiliar para maximizar ΔPhi
    LearnableRelevance,           # 🆕 Sistema completo de relevancia aprendible
    IITGuidedMemory,              # 🆕 Memoria guiada por PHI
    StochasticExploration,        # Exploración (antes "quantum")
    EnhancedMultiHeadAttention,   # Atención mejorada
    StandardNLPMetrics,           # Validación científica
    BenchmarkComparison           # Comparación contra baselines
)

# GloVe embeddings (opcional)
try:
    import gensim.downloader as api
    GLOVE_AVAILABLE = True
    print("[INFO] GloVe embeddings available")
except ImportError:
    GLOVE_AVAILABLE = False
    print("[WARNING] GloVe not available, using TF-IDF only")


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
        dropout: float = 0.1,  # 🆕 Dropout configurable (default 0.1, usar 0.3 contra overfitting)
        use_improved_memory: bool = True,  # 🆕 Usar IITGuidedMemory
        use_improved_iit: bool = True,     # 🆕 Usar ImprovedIITMetrics
        use_learnable_phi: bool = True,    # 🆕 Usar LearnableRelevance
        use_stochastic_exploration: bool = True,  # 🆕 Usar exploración estocástica
        lambda_phi: float = 0.1,           # 🆕 Peso del objetivo ΔPhi (0.0-1.0)
        seed: int = None  # 🆕 Seed para reproducibilidad
    ):
        super().__init__()
        
        # 🔒 Fijar seeds para reproducibilidad
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            # Para reproducibilidad completa en GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout  # 🆕 Guardar dropout
        self.use_improved_memory = use_improved_memory
        self.use_improved_iit = use_improved_iit
        self.use_learnable_phi = use_learnable_phi
        self.use_stochastic_exploration = use_stochastic_exploration
        self.seed = seed
        
        print(f"\n{'='*70}")
        print(f"INFINITO V5.2 - REFACTORIZADO + IIT MEJORADO")
        print(f"{'='*70}")
        print(f"  Dropout: {dropout} {'⚠️ AGRESIVO (anti-overfitting)' if dropout >= 0.3 else ''}")
        print(f"  Memoria mejorada (IIT-guided): {use_improved_memory}")
        print(f"  IIT Metrics mejorado (4 comp): {use_improved_iit}")
        print(f"  Pesos PHI aprendibles: {use_learnable_phi}")
        print(f"  Exploración estocástica: {use_stochastic_exploration}")
        if seed is not None:
            print(f"  [SEED] Fijado: {seed} (reproducibilidad garantizada)")
        print(f"{'='*70}\n")
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        self.embedding_dropout = nn.Dropout(dropout)  # 🆕 Dropout en embeddings
        
        # Memoria usando módulo refactorizado
        if use_improved_memory:
            print("  [OK] Usando IITGuidedMemory (priorizacion por PHI)")
            self.memory = IITGuidedMemory(
                memory_slots=memory_slots,
                hidden_dim=hidden_dim,
                use_phi_priority=True,
                alpha=0.8,  # 80% peso a PHI, 20% a attention
                learnable_threshold=True,  # 🆕 Threshold aprendible
                initial_threshold=3.0      # 🆕 Valor inicial (se optimizará)
            )
        else:
            print("  [OK] Usando PriorityExternalMemory (legacy)")
            self.memory = PriorityExternalMemory(
                memory_slots=memory_slots,
                slot_size=64,
                hidden_dim=hidden_dim
            )
        
        # Métricas de integración (antes "consciencia")
        if use_improved_iit:
            print("  [OK] Usando ImprovedIITMetrics (4 componentes)")
            self.iit_metrics = ImprovedIITMetrics(hidden_dim=hidden_dim)
        else:
            print("  [OK] Usando InformationIntegrationMetrics (3 componentes)")
            self.iit_metrics = InformationIntegrationMetrics(hidden_dim=hidden_dim)
        
        # Sistema de pesos aprendibles para PHI (opcional)
        if use_learnable_phi:
            print("  [OK] Usando LearnablePhiWeights (pesos componentes aprendibles)")
            self.learnable_phi_weights = LearnablePhiWeights(
                constraint='softmax'  # Los pesos suman 1.0
            )
            # Objetivo auxiliar para maximizar ΔPhi
            self.delta_phi_objective = DeltaPhiObjective(
                lambda_phi=lambda_phi,  # 🆕 Configurable desde constructor
                target_phi=1.2   # ⚠️ FIX BUG #3: Cambiar de 3.5 a 1.2 (realista, PHI actual ~0.9)
            )
        else:
            self.learnable_phi_weights = None
            self.delta_phi_objective = None
        
        # Exploración estocástica (antes "quantum noise")
        if use_stochastic_exploration:
            print("  [OK] Usando StochasticExploration (ruido gaussiano)")
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
                dropout=dropout  # 🆕 Usar dropout configurable
            )
            for _ in range(num_layers)
        ])
        
        # Feed-forward
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),  # 🆕 Usar dropout configurable
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)   # 🆕 Usar dropout configurable
            )
            for _ in range(num_layers)
        ])
        
        # LayerNorms
        self.layer_norms_1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms_2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # --- FIX: Mecanismo de Fusión de Memoria ---
        # Un valor escalar aprendible que empieza en -5.0
        # sigmoid(-5.0) ≈ 0.006 (0.6%) → Prácticamente CERRADO al inicio
        # Al principio: Hidden + 0.006 * Memoria ≈ Hidden (casi igual que Baseline, sin ruido)
        # Con el tiempo: El modelo aprenderá a ABRIR el gate si la memoria es útil
        # CRÍTICO: Empezar cerrado evita que la memoria ruidosa al inicio corrompa el entrenamiento
        self.memory_gate = nn.Parameter(torch.tensor(-5.0)) 
        
        # Una normalización extra para evitar que la suma explote los valores
        self.memory_norm = nn.LayerNorm(hidden_dim)
        # -------------------------------------------
        
        # Métricas estándar para validación
        self.nlp_metrics = StandardNLPMetrics()
        
        print("[OK] Modelo inicializado correctamente\n")
    
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
        hidden = self.embedding_dropout(hidden)  # 🆕 Aplicar dropout a embeddings
        
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
        phi_components = None
        delta_phi_loss = None
        
        if return_metrics and len(all_attention_weights) > 0:
            # Stack attention weights
            stacked_attention = torch.stack(all_attention_weights, dim=1)  # [batch, layers, heads, seq, seq]
            avg_attention = stacked_attention.mean(dim=1)  # Promediar sobre layers
            
            # 🆕 Calcular métricas de integración usando módulo refactorizado
            if self.use_improved_iit:
                # ImprovedIITMetrics con 4 componentes
                phi_estimate = self.iit_metrics.calculate_phi_approximation(hidden, avg_attention)
                temporal_coh = self.iit_metrics.calculate_temporal_coherence(hidden)
                integration = self.iit_metrics.calculate_integration_strength(hidden)
                complexity = self.iit_metrics.calculate_complexity(hidden)
                attn_diversity = self.iit_metrics.calculate_attention_diversity(avg_attention)
                coherence = self.iit_metrics.calculate_coherence(avg_attention)  # Solo attention
                
                metrics_dict = {
                    'phi_estimate': phi_estimate,
                    'temporal_coherence': temporal_coh,
                    'integration_strength': integration,
                    'complexity': complexity,
                    'attention_diversity': attn_diversity,
                    'coherence': coherence,
                    'pattern_diversity': attn_diversity  # Alias
                }
            else:
                # InformationIntegrationMetrics original (3 componentes)
                metrics_dict = self.iit_metrics.calculate_all_metrics(
                    hidden,
                    avg_attention
                )
            
            # Usar phi_estimate como proxy de integración
            integration_level = metrics_dict['phi_estimate']
            
            # 🆕 Si usamos pesos aprendibles, recalcular PHI ponderado
            if self.learnable_phi_weights is not None:
                # ⚠️ FIX BUG #2: Guardar PHI baseline ANTES de ponderar
                phi_baseline_unweighted = integration_level.clone()
                
                # Obtener pesos normalizados
                weights = self.learnable_phi_weights()  # Returns dict of tensors
                
                # Ponderar componentes individuales
                if self.use_improved_iit:
                    # 4 componentes: temporal, integration, complexity, attention
                    weighted_phi = (
                        weights['temporal'] * metrics_dict.get('temporal_coherence', torch.tensor(0.0, device=hidden.device)) +
                        weights['integration'] * metrics_dict.get('integration_strength', torch.tensor(0.0, device=hidden.device)) +
                        weights['complexity'] * metrics_dict.get('complexity', torch.tensor(0.0, device=hidden.device)) +
                        weights['attention'] * metrics_dict.get('attention_diversity', torch.tensor(0.0, device=hidden.device))
                    )
                    # Actualizar integration_level con PHI ponderado
                    integration_level = weighted_phi
                
                # Calcular loss auxiliar ΔPhi (para maximizar integración)
                if self.training and self.delta_phi_objective is not None:
                    # ⚠️ FIX BUG #2: Comparar PHI sin ponderar vs PHI ponderado (delta REAL)
                    delta_phi_loss, _ = self.delta_phi_objective(phi_baseline_unweighted, integration_level)
        
        # Interacción con memoria
        memory_query = hidden.mean(dim=1)  # [batch, hidden_dim]
        
        # 🆕 Leer de memoria (IITGuidedMemory o PriorityExternalMemory)
        if self.use_improved_memory:
            # IITGuidedMemory.read(query, top_k, phi_guided)
            read_content, read_weights = self.memory.read(
                memory_query, 
                top_k=5, 
                phi_guided=True
            )
            # read_content: [batch, top_k, hidden_dim] → tomar mean
            read_content = read_content.mean(dim=1)  # [batch, hidden_dim]
        else:
            # PriorityExternalMemory.read(query, relevance_scores)
            read_content, read_weights = self.memory.read(memory_query, integration_level)
        
        # Escribir en memoria
        memory_content = hidden.mean(dim=1)  # [batch, hidden_dim]
        
        if self.use_improved_memory:
            # IITGuidedMemory.write(query, content, phi_value, attention_score)
            # phi_value debe ser tensor [batch]
            phi_tensor = integration_level if len(all_attention_weights) > 0 else torch.ones(batch_size, device=hidden.device)
            write_info = self.memory.write(
                query=memory_query,
                content=memory_content,
                phi_value=phi_tensor,
                attention_score=None  # Opcional
            )
        else:
            # PriorityExternalMemory.write(query, content, relevance_scores, phi_value)
            phi_value = integration_level.mean().item() if return_metrics else 0.0
            self.memory.write(memory_query, memory_content, integration_level, phi_value)
        
        # --- FIX: FUSIÓN DE MEMORIA CON GATE APRENDIBLE ---
        # Usamos sigmoid en el gate para mantenerlo entre 0 y 1
        # Empieza en sigmoid(0.0) ≈ 0.5, pero inicializamos en 0.0 para empezar neutro
        if read_content is not None:
            # Aplicar gate: empieza cerca de 0, crece si la memoria es útil
            gated_memory = torch.sigmoid(self.memory_gate) * read_content.unsqueeze(1)
            
            # Conexión Residual: Lo que sabías + Lo que recordaste
            hidden = hidden + gated_memory
            
            # Estabilizar con LayerNorm
            hidden = self.memory_norm(hidden)
        # ------------------------------

        # Output (AHORA SÍ CONTIENE LA MEMORIA CON GATE)
        logits = self.output_projection(hidden)
        
        # Preparar métricas de retorno
        if return_metrics:
            metrics = {
                'integration_phi': integration_level.mean().item(),
                'coherence': metrics_dict.get('coherence', torch.tensor(0.0)).mean().item(),
                'complexity': metrics_dict.get('complexity', torch.tensor(0.0)).mean().item(),
                'pattern_diversity': metrics_dict.get('pattern_diversity', torch.tensor(0.0)).mean().item(),
                'memory_utilization': self.memory.get_statistics()['utilization'] if hasattr(self.memory, 'get_statistics') else 0.0
            }
            
            # 🆕 Si hay componentes IIT mejorados, añadirlos
            if self.use_improved_iit:
                metrics['temporal_coherence'] = metrics_dict.get('temporal_coherence', torch.tensor(0.0)).mean().item()
                metrics['integration_strength'] = metrics_dict.get('integration_strength', torch.tensor(0.0)).mean().item()
                metrics['attention_diversity'] = metrics_dict.get('attention_diversity', torch.tensor(0.0)).mean().item()
            
            # 🆕 Si hay pesos aprendibles, reportarlos
            if self.learnable_phi_weights is not None:
                weights = self.learnable_phi_weights.get_weights_dict()
                metrics['phi_weights'] = weights
                if delta_phi_loss is not None:
                    # ⚠️ FIX BUG #4: NO convertir a .item() - conservar tensor con gradientes
                    metrics['delta_phi_loss'] = delta_phi_loss
            
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
    
    def get_auxiliary_loss(self, metrics: Dict) -> Optional[torch.Tensor]:
        """
        🆕 Obtiene el loss auxiliar ΔPhi para backprop.
        
        Este loss incentiva al modelo a maximizar la integración.
        Debe sumarse al loss principal durante el entrenamiento.
        
        Args:
            metrics: Dict retornado por forward() con return_metrics=True
            
        Returns:
            delta_phi_loss: Loss auxiliar (o None si no está activo)
        """
        if metrics and 'delta_phi_loss' in metrics:
            return torch.tensor(metrics['delta_phi_loss'])
        return None
    
    def get_learnable_weights(self) -> Optional[Dict]:
        """🆕 Obtiene los pesos aprendibles actuales de PHI."""
        if self.learnable_phi_weights is not None:
            return self.learnable_phi_weights.get_weights_dict()
        return None
    
    def save_learnable_weights(self, path: str):
        """🆕 Guarda los pesos aprendibles en un archivo JSON."""
        if self.learnable_phi_weights is not None:
            self.learnable_phi_weights.save_weights(path)
            print(f"✅ Pesos PHI guardados en: {path}")
        else:
            print("⚠️ No hay pesos aprendibles para guardar")
    
    def load_learnable_weights(self, path: str):
        """🆕 Carga los pesos aprendibles desde un archivo JSON."""
        if self.learnable_phi_weights is not None:
            self.learnable_phi_weights.load_weights(path)
            print(f"✅ Pesos PHI cargados desde: {path}")
        else:
            print("⚠️ No hay sistema de pesos aprendibles activo")
    
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
    """Demuestra cómo usar el modelo refactorizado con IIT mejorado."""
    
    print("\n" + "="*70)
    print("📚 DEMO: MODELO INFINITO V5.2 CON IIT MEJORADO")
    print("="*70 + "\n")
    
    # Crear modelo CON todas las mejoras IIT
    model = InfinitoV52Refactored(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        use_improved_memory=True,      # ✅ IITGuidedMemory
        use_improved_iit=True,         # ✅ ImprovedIITMetrics (4 componentes)
        use_learnable_phi=True,        # ✅ LearnableRelevance
        use_stochastic_exploration=True
    )
    
    # Datos de ejemplo
    batch_size, seq_len = 4, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    target_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward con métricas
    print("🔄 Forward pass...")
    logits, metrics = model(input_ids, return_metrics=True)
    
    print(f"\n📊 Métricas IIT mejoradas (4 componentes):")
    print(f"  PHI integrado: {metrics['integration_phi']:.4f}")
    print(f"  └─ Temporal coherence: {metrics.get('temporal_coherence', 0):.4f}")
    print(f"  └─ Integration strength: {metrics.get('integration_strength', 0):.4f}")
    print(f"  └─ Complexity: {metrics['complexity']:.4f}")
    print(f"  └─ Attention diversity: {metrics.get('attention_diversity', 0):.4f}")
    
    if 'phi_weights' in metrics:
        print(f"\n⚖️ Pesos PHI aprendibles:")
        for component, weight in metrics['phi_weights'].items():
            print(f"  {component}: {weight:.4f}")
        print(f"\n  ΔPhi Loss: {metrics.get('delta_phi_loss', 0):.6f}")
    
    # 🆕 Calcular perplexity (métrica estándar)
    print(f"\n📏 Métricas estándar:")
    perplexity = model.calculate_perplexity(input_ids, target_ids)
    print(f"  Perplexity: {perplexity:.2f}")
    
    # Estadísticas de memoria IIT-guided
    print(f"\n💾 Estadísticas de memoria IIT-guided:")
    mem_stats = model.get_memory_statistics()
    for key, value in mem_stats.items():
        if isinstance(value, float):
            if key == 'threshold':
                print(f"  {key}: {value:.4f} ← 🎯 APRENDIBLE")
            else:
                print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n📝 {metrics['_note']}")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETADO - SISTEMA IIT MEJORADO FUNCIONANDO")
    print("="*70 + "\n")
    
    print("💡 MEJORAS vs V5.2 original:")
    print("  • PHI con 4 componentes (vs 3)")
    print("  • Pesos PHI aprendibles durante entrenamiento")
    print("  • Memoria guiada por integración (eviction inteligente)")
    print("  • 🆕 Threshold aprendible (decide automáticamente qué guardar)")
    print("  • Objetivo auxiliar ΔPhi para maximizar integración")
    print("")
    
    print("🎯 THRESHOLD APRENDIBLE:")
    print(f"  • Valor inicial: 3.0")
    print(f"  • Durante entrenamiento: Se ajustará automáticamente")
    print(f"  • Función: Solo guarda en memoria si PHI > threshold")
    print(f"  • Beneficio: Filtra ruido ('eh... hmm...') de forma óptima")
    print("")


if __name__ == '__main__':
    demo_refactored_model()
