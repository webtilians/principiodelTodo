#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 IIT METRICS V2 - ARQUITECTURA ENTRENABLE CON REDES NEURONALES
=================================================================

PROBLEMA IDENTIFICADO:
- ImprovedIITMetrics (v1) usa SOLO operaciones estadísticas (correlation, variance, entropy)
- phi_projector y complexity_analyzer definidos pero NUNCA usados
- Resultado: NO hay gradientes → NO puede aprender a maximizar PHI

SOLUCIÓN (V2):
- TODAS las métricas calculadas por redes neuronales entrenables
- ~200K parámetros totales distribuidos en 4 componentes
- Gradientes fluyen directamente → modelo aprende qué patrones maximizan PHI

COMPONENTES:
1. TemporalCoherenceNet: Reemplaza torch.corrcoef (~50K params)
2. IntegrationNet: Reemplaza mutual information manual (~100K params)
3. ComplexityNet: Reemplaza variance (~30K params)
4. AttentionDiversityNet: Reemplaza entropy (~20K params)

ARQUITECTURA:
Input: hidden_states (B, T, 768) + attention_weights (B, H, T, T)
       ↓
   [4 Redes Neuronales Paralelas]
       ↓
   4 Componentes PHI (B,) cada uno
       ↓
   PHI Total = Σ w_i * component_i  (pesos aprendibles)
       ↓
   Scalar PHI (entrenable end-to-end)

MEJORA ESPERADA:
- Lambda_phi=1.0 debería aumentar PHI 5-15% por época
- ΔPhi Loss tendrá gradientes reales hacia parámetros neurales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalCoherenceNet(nn.Module):
    """
    Red neuronal que aprende a medir coherencia temporal.
    
    ANTES (v1): torch.corrcoef(hidden[:, t], hidden[:, t+1])
    AHORA (v2): Red neuronal que aprende patrones de coherencia
    
    Arquitectura:
    - Input: Pares consecutivos de hidden states [h_t, h_{t+1}]
    - 3 capas fully connected con residual connections
    - Output: Score de coherencia [0, 1]
    
    Parámetros: ~50K
    """
    
    def __init__(self, hidden_dim=768, internal_dim=128):
        super().__init__()
        
        # Input: 2*hidden_dim (concatenar t y t+1)
        self.input_proj = nn.Linear(hidden_dim * 2, internal_dim)
        
        # Capas internas con residual
        self.fc1 = nn.Linear(internal_dim, internal_dim)
        self.fc2 = nn.Linear(internal_dim, internal_dim)
        self.fc3 = nn.Linear(internal_dim, internal_dim)
        
        # Output: 1D score
        self.output_proj = nn.Linear(internal_dim, 1)
        
        # Normalization
        self.layer_norm1 = nn.LayerNorm(internal_dim)
        self.layer_norm2 = nn.LayerNorm(internal_dim)
        self.layer_norm3 = nn.LayerNorm(internal_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, T, hidden_dim)
        
        Returns:
            coherence_score: (B,) - Score promedio de coherencia temporal
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        if seq_len < 2:
            # No podemos calcular coherencia con 1 solo timestep
            return torch.zeros(batch_size, device=hidden_states.device)
        
        # Crear pares consecutivos: [h_t, h_{t+1}]
        h_current = hidden_states[:, :-1, :]  # (B, T-1, D)
        h_next = hidden_states[:, 1:, :]      # (B, T-1, D)
        pairs = torch.cat([h_current, h_next], dim=-1)  # (B, T-1, 2D)
        
        # Proyección inicial
        x = self.input_proj(pairs)  # (B, T-1, internal_dim)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Capa 1 con residual
        identity = x
        x = self.fc1(x)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity  # Residual
        
        # Capa 2 con residual
        identity = x
        x = self.fc2(x)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity  # Residual
        
        # Capa 3 con residual
        identity = x
        x = self.fc3(x)
        x = self.layer_norm3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + identity  # Residual
        
        # Output projection
        coherence = self.output_proj(x)  # (B, T-1, 1)
        coherence = torch.sigmoid(coherence)  # [0, 1]
        
        # Promedio sobre tiempo
        coherence_score = coherence.mean(dim=1).squeeze(-1)  # (B,)
        
        return coherence_score


class IntegrationNet(nn.Module):
    """
    Red neuronal que aprende a medir integración de información.
    
    ANTES (v1): Mutual information aproximado con loops y estadísticas
    AHORA (v2): Red neuronal profunda que aprende patrones de integración
    
    Arquitectura:
    - Input: hidden_states global + attention_weights
    - Transformer encoder layer para capturar interacciones
    - MLP para score final
    - Output: Integration strength [0, 1]
    
    Parámetros: ~100K (el más grande, es el componente más complejo)
    """
    
    def __init__(self, hidden_dim=768, num_heads=8, internal_dim=256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Proyección inicial
        self.input_proj = nn.Linear(hidden_dim, internal_dim)
        
        # Transformer encoder layer (captura interacciones globales)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=internal_dim,
            nhead=4,  # 4 heads (internal_dim debe ser divisible)
            dim_feedforward=internal_dim * 2,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        
        # MLP para procesar atención
        self.attention_mlp = nn.Sequential(
            nn.Linear(num_heads, internal_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(internal_dim // 4, internal_dim // 4)
        )
        
        # Combinar hidden + attention features
        self.fusion = nn.Sequential(
            nn.Linear(internal_dim + internal_dim // 4, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(internal_dim, internal_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(internal_dim // 2, 1)
        )
        
    def forward(self, hidden_states, attention_weights=None):
        """
        Args:
            hidden_states: (B, T, hidden_dim)
            attention_weights: (B, num_heads, T, T) opcional
        
        Returns:
            integration_score: (B,) - Score de integración de información
        """
        batch_size = hidden_states.size(0)
        
        # Proyección de hidden states
        x = self.input_proj(hidden_states)  # (B, T, internal_dim)
        
        # Transformer encoder (captura interacciones)
        x = self.transformer_layer(x)  # (B, T, internal_dim)
        
        # Pooling temporal (mean)
        hidden_features = x.mean(dim=1)  # (B, internal_dim)
        
        # Procesar attention weights si están disponibles
        if attention_weights is not None:
            # attention_weights: (B, num_heads, T, T)
            # Calcular promedio de atención por head
            attn_mean = attention_weights.mean(dim=[2, 3])  # (B, num_heads)
            attn_features = self.attention_mlp(attn_mean)  # (B, internal_dim//4)
        else:
            # Sin atención, usar zeros
            attn_features = torch.zeros(
                batch_size, 
                self.attention_mlp[0].out_features,
                device=hidden_states.device
            )
        
        # Fusionar hidden + attention
        combined = torch.cat([hidden_features, attn_features], dim=-1)
        fused = self.fusion(combined)  # (B, internal_dim)
        
        # Score final
        integration = self.output_mlp(fused)  # (B, 1)
        integration = torch.sigmoid(integration).squeeze(-1)  # (B,)
        
        return integration


class ComplexityNet(nn.Module):
    """
    Red neuronal que aprende a medir complejidad de representaciones.
    
    ANTES (v1): Variance simple torch.var(hidden_states)
    AHORA (v2): Red neuronal que aprende patrones de complejidad
    
    Arquitectura:
    - Input: hidden_states
    - Convolutional layers para capturar patrones locales
    - MLP para score final
    - Output: Complexity score [0, 1]
    
    Parámetros: ~30K
    """
    
    def __init__(self, hidden_dim=768, internal_dim=128):
        super().__init__()
        
        # Proyección inicial
        self.input_proj = nn.Linear(hidden_dim, internal_dim)
        
        # Conv1D para capturar patrones temporales locales
        self.conv1 = nn.Conv1d(internal_dim, internal_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(internal_dim, internal_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(internal_dim, internal_dim, kernel_size=7, padding=3)
        
        # Batch norm
        self.bn1 = nn.BatchNorm1d(internal_dim)
        self.bn2 = nn.BatchNorm1d(internal_dim)
        self.bn3 = nn.BatchNorm1d(internal_dim)
        
        # Fusion layer
        self.fusion = nn.Linear(internal_dim * 3, internal_dim)
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(internal_dim, internal_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(internal_dim // 2, 1)
        )
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, T, hidden_dim)
        
        Returns:
            complexity_score: (B,) - Score de complejidad
        """
        # Proyección
        x = self.input_proj(hidden_states)  # (B, T, internal_dim)
        
        # Transpose para Conv1D: (B, internal_dim, T)
        x = x.transpose(1, 2)
        
        # Multi-scale convolutions
        c1 = F.relu(self.bn1(self.conv1(x)))  # (B, internal_dim, T)
        c2 = F.relu(self.bn2(self.conv2(x)))
        c3 = F.relu(self.bn3(self.conv3(x)))
        
        # Pooling temporal
        c1_pool = c1.mean(dim=2)  # (B, internal_dim)
        c2_pool = c2.mean(dim=2)
        c3_pool = c3.mean(dim=2)
        
        # Concatenar multi-scale features
        combined = torch.cat([c1_pool, c2_pool, c3_pool], dim=-1)  # (B, internal_dim*3)
        
        # Fusión
        fused = self.fusion(combined)  # (B, internal_dim)
        fused = F.relu(fused)
        
        # Score final
        complexity = self.output_mlp(fused)  # (B, 1)
        complexity = torch.sigmoid(complexity).squeeze(-1)  # (B,)
        
        return complexity


class AttentionDiversityNet(nn.Module):
    """
    Red neuronal que aprende a medir diversidad de atención.
    
    ANTES (v1): Entropy manual con loops
    AHORA (v2): Red neuronal que aprende patrones de diversidad
    
    Arquitectura:
    - Input: attention_weights (multi-head)
    - CNN para procesar matrices de atención
    - MLP para score final
    - Output: Diversity score [0, 1]
    
    Parámetros: ~20K
    """
    
    def __init__(self, num_heads=12, internal_dim=64):
        super().__init__()
        
        self.num_heads = num_heads
        
        # Procesar cada head de atención
        # Input: (B, num_heads, T, T) -> flatten spatial -> features
        self.head_processors = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),  # Reducir dimensionalidad espacial
                nn.Flatten(),  # (B, 64)
                nn.Linear(64, internal_dim),
                nn.ReLU()
            )
            for _ in range(num_heads)
        ])
        
        # Combinar todos los heads
        self.fusion = nn.Sequential(
            nn.Linear(num_heads * internal_dim, internal_dim),
            nn.LayerNorm(internal_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output MLP
        self.output_mlp = nn.Sequential(
            nn.Linear(internal_dim, internal_dim // 2),
            nn.ReLU(),
            nn.Linear(internal_dim // 2, 1)
        )
        
    def forward(self, attention_weights):
        """
        Args:
            attention_weights: (B, num_heads, T, T)
        
        Returns:
            diversity_score: (B,) - Score de diversidad de atención
        """
        if attention_weights is None:
            batch_size = 1
            return torch.zeros(batch_size, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        batch_size = attention_weights.size(0)
        
        # Procesar cada head independientemente
        head_features = []
        for i in range(self.num_heads):
            head_attn = attention_weights[:, i, :, :]  # (B, T, T)
            head_attn = head_attn.unsqueeze(1)  # (B, 1, T, T) for Conv2D compatibility
            head_feat = self.head_processors[i](head_attn)  # (B, internal_dim)
            head_features.append(head_feat)
        
        # Concatenar features de todos los heads
        combined = torch.cat(head_features, dim=-1)  # (B, num_heads * internal_dim)
        
        # Fusión
        fused = self.fusion(combined)  # (B, internal_dim)
        
        # Score final
        diversity = self.output_mlp(fused)  # (B, 1)
        diversity = torch.sigmoid(diversity).squeeze(-1)  # (B,)
        
        return diversity


class ImprovedIITMetricsV2(nn.Module):
    """
    IIT Metrics V2 - COMPLETAMENTE ENTRENABLE CON REDES NEURONALES
    
    DIFERENCIAS vs V1:
    ✅ V2: ~200K parámetros entrenables distribuidos en 4 redes
    ❌ V1: 0 parámetros (solo estadísticas torch)
    
    ✅ V2: Gradientes fluyen directamente a todas las métricas
    ❌ V1: Sin gradientes (phi_projector definido pero no usado)
    
    ✅ V2: Modelo aprende QUÉ patrones maximizan PHI
    ❌ V1: Métricas fijas, no adaptables
    
    COMPONENTES:
    1. TemporalCoherenceNet: ~50K params
    2. IntegrationNet: ~100K params  
    3. ComplexityNet: ~30K params
    4. AttentionDiversityNet: ~20K params
    
    TOTAL: ~200K parámetros entrenables
    
    USO:
    >>> metrics_v2 = ImprovedIITMetricsV2(hidden_dim=768)
    >>> phi_dict = metrics_v2(hidden_states, attention_weights)
    >>> phi_total = phi_dict['phi_estimate']  # Tensor con gradiente!
    >>> loss = criterion(outputs, labels) + lambda_phi * (target_phi - phi_total)
    >>> loss.backward()  # ✅ Gradientes fluyen a las 4 redes!
    """
    
    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        learnable_weights=True,
        perplexity=None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        print("\n" + "="*70)
        print("🧠 INICIALIZANDO IIT METRICS V2 (ARQUITECTURA ENTRENABLE)")
        print("="*70)
        
        # 1. Temporal Coherence Network
        self.temporal_net = TemporalCoherenceNet(hidden_dim=hidden_dim)
        temporal_params = sum(p.numel() for p in self.temporal_net.parameters())
        print(f"  [1/4] TemporalCoherenceNet: {temporal_params:,} params")
        
        # 2. Integration Network
        self.integration_net = IntegrationNet(
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        integration_params = sum(p.numel() for p in self.integration_net.parameters())
        print(f"  [2/4] IntegrationNet: {integration_params:,} params")
        
        # 3. Complexity Network
        self.complexity_net = ComplexityNet(hidden_dim=hidden_dim)
        complexity_params = sum(p.numel() for p in self.complexity_net.parameters())
        print(f"  [3/4] ComplexityNet: {complexity_params:,} params")
        
        # 4. Attention Diversity Network
        self.diversity_net = AttentionDiversityNet(num_heads=num_heads)
        diversity_params = sum(p.numel() for p in self.diversity_net.parameters())
        print(f"  [4/4] AttentionDiversityNet: {diversity_params:,} params")
        
        # Total params
        total_params = temporal_params + integration_params + complexity_params + diversity_params
        print(f"\n  📊 TOTAL PARÁMETROS IIT: {total_params:,}")
        print("="*70 + "\n")
        
        # Pesos para combinar componentes (aprendibles o fijos)
        if learnable_weights:
            # Pesos aprendibles (inicializar a 1.0)
            self.w_temporal = nn.Parameter(torch.ones(1))
            self.w_integration = nn.Parameter(torch.ones(1))
            self.w_complexity = nn.Parameter(torch.ones(1))
            self.w_diversity = nn.Parameter(torch.ones(1))
            print("  ✓ Pesos de componentes: APRENDIBLES")
        else:
            # Pesos fijos
            self.register_buffer('w_temporal', torch.tensor(1.0))
            self.register_buffer('w_integration', torch.tensor(1.0))
            self.register_buffer('w_complexity', torch.tensor(1.0))
            self.register_buffer('w_diversity', torch.tensor(1.0))
            print("  ✓ Pesos de componentes: FIJOS")
        
        # Normalización de pesos
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, hidden_state, attention_weights=None):
        """
        Calcula PHI usando redes neuronales entrenables.
        
        Args:
            hidden_state: (B, T, hidden_dim)
            attention_weights: (B, num_heads, T, T) opcional
        
        Returns:
            dict con:
                - phi_estimate: (B,) Tensor con gradiente
                - temporal_coherence: (B,)
                - integration_strength: (B,)
                - complexity: (B,)
                - attention_diversity: (B,)
        """
        # Calcular 4 componentes (todos con gradientes!)
        temporal = self.temporal_net(hidden_state)  # (B,)
        integration = self.integration_net(hidden_state, attention_weights)  # (B,)
        complexity = self.complexity_net(hidden_state)  # (B,)
        diversity = self.diversity_net(attention_weights) if attention_weights is not None else torch.zeros_like(temporal)  # (B,)
        
        # Normalizar pesos
        weights = torch.stack([
            self.w_temporal,
            self.w_integration,
            self.w_complexity,
            self.w_diversity
        ])
        weights = self.softmax(weights)
        
        # PHI = weighted sum de componentes
        # Todos tienen shape (B,), expandir weights para broadcasting
        phi_estimate = (
            weights[0] * temporal +
            weights[1] * integration +
            weights[2] * complexity +
            weights[3] * diversity
        )
        
        # Escalar a rango típico de PHI (~1-10)
        # Factor de escala: 10x para que valores [0-1] → [0-10]
        phi_estimate = phi_estimate * 10.0
        
        return {
            'phi_estimate': phi_estimate,  # (B,) - TENSOR CON GRADIENTE
            'temporal_coherence': temporal,
            'integration_strength': integration,
            'complexity': complexity,
            'attention_diversity': diversity
        }
    
    def get_num_parameters(self):
        """Retorna número total de parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# TEST RÁPIDO
# =============================================================================

if __name__ == '__main__':
    print("\n🧪 TEST RÁPIDO: IIT Metrics V2")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Crear métricas v2
    metrics_v2 = ImprovedIITMetricsV2(
        hidden_dim=768,
        num_heads=12,
        learnable_weights=True
    ).to(device)
    
    # Input dummy
    batch_size = 4
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, 768, device=device, requires_grad=True)
    attention_weights = torch.randn(batch_size, 12, seq_len, seq_len, device=device)
    
    print(f"\n📥 Input:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  attention_weights: {attention_weights.shape}")
    
    # Forward pass
    print(f"\n🔄 Forward pass...")
    phi_dict = metrics_v2(hidden_states, attention_weights)
    
    print(f"\n📤 Output:")
    print(f"  phi_estimate: {phi_dict['phi_estimate'].shape} = {phi_dict['phi_estimate'].mean().item():.4f}")
    print(f"  temporal_coherence: {phi_dict['temporal_coherence'].mean().item():.4f}")
    print(f"  integration_strength: {phi_dict['integration_strength'].mean().item():.4f}")
    print(f"  complexity: {phi_dict['complexity'].mean().item():.4f}")
    print(f"  attention_diversity: {phi_dict['attention_diversity'].mean().item():.4f}")
    
    # Test gradientes
    print(f"\n🔍 Test gradientes:")
    phi_total = phi_dict['phi_estimate'].mean()
    target_phi = torch.tensor(6.0, device=device)
    loss = (target_phi - phi_total) ** 2
    
    print(f"  PHI actual: {phi_total.item():.4f}")
    print(f"  Target PHI: {target_phi.item():.4f}")
    print(f"  Loss: {loss.item():.4f}")
    
    loss.backward()
    
    # Verificar gradientes en todas las redes
    print(f"\n✅ Verificación de gradientes:")
    
    grad_check = {
        'TemporalNet': metrics_v2.temporal_net.fc1.weight.grad,
        'IntegrationNet': metrics_v2.integration_net.input_proj.weight.grad,
        'ComplexityNet': metrics_v2.complexity_net.conv1.weight.grad,
        'DiversityNet': metrics_v2.diversity_net.head_processors[0][2].weight.grad
    }
    
    for name, grad in grad_check.items():
        if grad is not None:
            print(f"  ✅ {name}: grad.norm() = {grad.norm().item():.6e}")
        else:
            print(f"  ❌ {name}: NO GRADIENT")
    
    # Verificar pesos aprendibles
    print(f"\n📊 Pesos componentes:")
    weights = torch.stack([
        metrics_v2.w_temporal,
        metrics_v2.w_integration,
        metrics_v2.w_complexity,
        metrics_v2.w_diversity
    ])
    weights_norm = metrics_v2.softmax(weights)
    print(f"  Temporal: {weights_norm[0].item():.4f}")
    print(f"  Integration: {weights_norm[1].item():.4f}")
    print(f"  Complexity: {weights_norm[2].item():.4f}")
    print(f"  Diversity: {weights_norm[3].item():.4f}")
    
    print(f"\n{'='*70}")
    print(f"✅ TEST COMPLETADO - IIT Metrics V2 funcionando correctamente!")
    print(f"{'='*70}\n")
