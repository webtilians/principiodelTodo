#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 IIT TRANSFORMER LAYER - CAPA TRANSFORMER GUIADA POR PHI
============================================================

OBJETIVO FASE 2:
Reemplazar capas GPT-2 con capas custom que maximizan PHI directamente.

DIFERENCIAS vs GPT-2 Layer:
- Self-Attention: Usa scores PHI para ponderar atención
- FFN: Entrenado para maximizar integración de información
- Residual: Gating adaptativo basado en PHI local
- Layer Norm: Normalización consciente de complejidad

ARQUITECTURA:
Input: hidden_states (B, T, 768)
       ↓
  [PHI-Guided Self-Attention]
       ↓ residual + gating
  [Integration-Aware FFN]
       ↓ residual + gating
Output: enhanced_hidden (B, T, 768)

PARÁMETROS: ~7M por capa
- Multi-head attention: ~2.3M
- PHI gating networks: ~1.5M
- FFN: ~2.4M
- Integration networks: ~800K

MEJORA ESPERADA (Fase 2):
- PHI: 3.52 → 4.5-5.0 (+27-42%)
- PPL: 26.4 → 30-35 (+13-32% degradación aceptable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhiGuidedAttention(nn.Module):
    """
    Self-Attention guiada por PHI.
    
    INNOVACIÓN:
    - Attention scores modulados por PHI local de cada token
    - Aprende qué patrones de atención maximizan integración
    - Diversity bonus para evitar collapse atencional
    
    Formula:
    attention_weights = softmax(QK^T / sqrt(d_k) + phi_bonus)
    phi_bonus = learnable_net(phi_local)
    """
    
    def __init__(self, hidden_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim debe ser divisible por num_heads"
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # PHI-aware components
        # Red que calcula PHI local por token
        self.phi_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # PHI local [0, 1]
        )
        
        # Red que convierte PHI local en attention bias
        self.phi_to_bias = nn.Sequential(
            nn.Linear(1, num_heads),
            nn.Tanh()  # Bias [-1, 1]
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (B, T, hidden_dim)
            attention_mask: (B, T) opcional
        
        Returns:
            output: (B, T, hidden_dim)
            attention_weights: (B, num_heads, T, T)
            phi_local: (B, T) - PHI estimado por token
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Estimar PHI local por token
        phi_local = self.phi_estimator(hidden_states).squeeze(-1)  # (B, T)
        
        # 2. Convertir PHI a attention bias
        phi_bias = self.phi_to_bias(phi_local.unsqueeze(-1))  # (B, T, num_heads)
        phi_bias = phi_bias.transpose(1, 2)  # (B, num_heads, T)
        phi_bias = phi_bias.unsqueeze(-1)  # (B, num_heads, T, 1)
        
        # 3. Standard multi-head attention
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (B, num_heads, T, head_dim)
        
        # 4. Attention scores con PHI bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, T, T)
        
        # Añadir PHI bias (broadcast)
        scores = scores + phi_bias  # Tokens con alto PHI reciben bonus
        
        # Mask si es necesario
        if attention_mask is not None:
            # attention_mask: (B, T) -> (B, 1, 1, T)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # 5. Attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (B, num_heads, T, T)
        attention_weights = self.dropout(attention_weights)
        
        # 6. Apply attention
        context = torch.matmul(attention_weights, V)  # (B, num_heads, T, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # 7. Output projection
        output = self.out_proj(context)
        output = self.dropout(output)
        
        return output, attention_weights, phi_local


class IntegrationAwareFFN(nn.Module):
    """
    Feed-Forward Network consciente de integración.
    
    INNOVACIÓN:
    - FFN entrenado para maximizar integración de información
    - Gating basado en complejidad local
    - Expansion ratio adaptativo por token
    """
    
    def __init__(self, hidden_dim=768, ffn_dim=3072, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        
        # Standard FFN
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        
        # Integration gating
        # Red que decide cuánta "integración" aplicar por token
        self.integration_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Gate [0, 1]
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, T, hidden_dim)
        
        Returns:
            output: (B, T, hidden_dim)
            integration_scores: (B, T) - Score de integración por token
        """
        # 1. Calcular integration gate
        integration_scores = self.integration_gate(hidden_states).squeeze(-1)  # (B, T)
        
        # 2. Standard FFN
        x = self.fc1(hidden_states)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        # 3. Modular output por integration gate
        gate = integration_scores.unsqueeze(-1)  # (B, T, 1)
        output = gate * x + (1 - gate) * hidden_states  # Blend adaptativo
        
        return output, integration_scores


class PhiAwareResidual(nn.Module):
    """
    Residual connection con gating basado en PHI.
    
    INNOVACIÓN:
    - Gate aprende cuándo usar nueva info (alto PHI) vs mantener original (bajo PHI)
    - Previene degradación de información integrada
    """
    
    def __init__(self, hidden_dim=768):
        super().__init__()
        
        # Red que decide blend ratio basado en ambas representaciones
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Alpha [0, 1]
        )
        
    def forward(self, original, transformed):
        """
        Args:
            original: (B, T, hidden_dim)
            transformed: (B, T, hidden_dim)
        
        Returns:
            output: (B, T, hidden_dim)
            alpha: (B, T) - Coeficiente de mezcla
        """
        # Concatenar ambas representaciones
        combined = torch.cat([original, transformed], dim=-1)  # (B, T, 2*hidden_dim)
        
        # Calcular gate
        alpha = self.gate_net(combined).squeeze(-1)  # (B, T)
        
        # Blend
        alpha_expanded = alpha.unsqueeze(-1)  # (B, T, 1)
        output = alpha_expanded * transformed + (1 - alpha_expanded) * original
        
        return output, alpha


class IITTransformerLayer(nn.Module):
    """
    Capa Transformer completa guiada por PHI.
    
    ARQUITECTURA COMPLETA:
    Input (B, T, 768)
         ↓
    LayerNorm
         ↓
    PhiGuidedAttention → attention_output, attn_weights, phi_local
         ↓
    PhiAwareResidual(input, attention_output) → x1, alpha1
         ↓
    LayerNorm
         ↓
    IntegrationAwareFFN → ffn_output, integration_scores
         ↓
    PhiAwareResidual(x1, ffn_output) → output, alpha2
         ↓
    Output (B, T, 768)
    
    MÉTRICAS INTERNAS:
    - phi_local: PHI estimado por token
    - integration_scores: Score de integración por token
    - alpha1, alpha2: Coeficientes de mezcla residual
    - attention_weights: Pesos de atención multi-head
    
    PARÁMETROS: ~7M
    """
    
    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        ffn_dim=3072,
        dropout=0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # PHI-guided components
        self.attention = PhiGuidedAttention(hidden_dim, num_heads, dropout)
        self.ffn = IntegrationAwareFFN(hidden_dim, ffn_dim, dropout)
        
        # PHI-aware residuals
        self.residual1 = PhiAwareResidual(hidden_dim)
        self.residual2 = PhiAwareResidual(hidden_dim)
        
        print(f"  [IIT Layer] Inicializada: {hidden_dim}D, {num_heads} heads, {ffn_dim} FFN")
        
    def forward(self, hidden_states, attention_mask=None, return_metrics=False):
        """
        Args:
            hidden_states: (B, T, hidden_dim)
            attention_mask: (B, T) opcional
            return_metrics: Si True, retorna métricas internas
        
        Returns:
            output: (B, T, hidden_dim)
            metrics: dict (si return_metrics=True)
        """
        # 1. Self-attention block
        residual1 = hidden_states
        x = self.ln1(hidden_states)
        attn_output, attn_weights, phi_local = self.attention(x, attention_mask)
        x, alpha1 = self.residual1(residual1, attn_output)
        
        # 2. FFN block
        residual2 = x
        x = self.ln2(x)
        ffn_output, integration_scores = self.ffn(x)
        output, alpha2 = self.residual2(residual2, ffn_output)
        
        if return_metrics:
            metrics = {
                'phi_local': phi_local,  # (B, T)
                'integration_scores': integration_scores,  # (B, T)
                'alpha1': alpha1,  # (B, T)
                'alpha2': alpha2,  # (B, T)
                'attention_weights': attn_weights  # (B, num_heads, T, T)
            }
            return output, metrics
        
        return output


class IITTransformerBlock(nn.Module):
    """
    Bloque de múltiples capas IIT Transformer.
    
    Usado para reemplazar las últimas N capas de GPT-2.
    """
    
    def __init__(
        self,
        num_layers=2,
        hidden_dim=768,
        num_heads=12,
        ffn_dim=3072,
        dropout=0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        print(f"\n{'='*70}")
        print(f"IIT TRANSFORMER BLOCK - {num_layers} CAPAS")
        print(f"{'='*70}")
        
        self.layers = nn.ModuleList([
            IITTransformerLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n📊 Parámetros IIT Transformer Block:")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Por capa: {total_params//num_layers:,}")
        print(f"{'='*70}\n")
        
    def forward(self, hidden_states, attention_mask=None, return_metrics=False):
        """
        Args:
            hidden_states: (B, T, hidden_dim)
            attention_mask: (B, T) opcional
            return_metrics: Si True, retorna métricas de todas las capas
        
        Returns:
            output: (B, T, hidden_dim)
            all_metrics: list[dict] (si return_metrics=True)
        """
        all_metrics = [] if return_metrics else None
        
        x = hidden_states
        for i, layer in enumerate(self.layers):
            if return_metrics:
                x, metrics = layer(x, attention_mask, return_metrics=True)
                metrics['layer_idx'] = i
                all_metrics.append(metrics)
            else:
                x = layer(x, attention_mask, return_metrics=False)
        
        if return_metrics:
            return x, all_metrics
        return x


# =============================================================================
# TEST RÁPIDO
# =============================================================================

if __name__ == '__main__':
    print("\n🧪 TEST RÁPIDO: IIT Transformer Layer")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Crear bloque IIT
    iit_block = IITTransformerBlock(
        num_layers=2,
        hidden_dim=768,
        num_heads=12,
        ffn_dim=3072,
        dropout=0.1
    ).to(device)
    
    # Input dummy
    batch_size = 4
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, 768, device=device, requires_grad=True)
    
    print(f"\n📥 Input:")
    print(f"  hidden_states: {hidden_states.shape}")
    
    # Forward pass
    print(f"\n🔄 Forward pass...")
    output, all_metrics = iit_block(hidden_states, return_metrics=True)
    
    print(f"\n📤 Output:")
    print(f"  output: {output.shape}")
    
    # Métricas por capa
    print(f"\n📊 Métricas por capa:")
    for i, metrics in enumerate(all_metrics):
        print(f"\n  Capa {i}:")
        print(f"    PHI local (mean): {metrics['phi_local'].mean().item():.4f}")
        print(f"    Integration scores (mean): {metrics['integration_scores'].mean().item():.4f}")
        print(f"    Alpha1 (mean): {metrics['alpha1'].mean().item():.4f}")
        print(f"    Alpha2 (mean): {metrics['alpha2'].mean().item():.4f}")
    
    # Test gradientes
    print(f"\n🔍 Test gradientes:")
    loss = output.mean()
    loss.backward()
    
    # Verificar gradientes en componentes clave
    print(f"\n✅ Verificación de gradientes:")
    
    grad_check = {
        'Layer0 Attention Q': iit_block.layers[0].attention.q_proj.weight.grad,
        'Layer0 PHI Estimator': iit_block.layers[0].attention.phi_estimator[0].weight.grad,
        'Layer0 FFN': iit_block.layers[0].ffn.fc1.weight.grad,
        'Layer0 Integration Gate': iit_block.layers[0].ffn.integration_gate[0].weight.grad,
        'Layer0 Residual1 Gate': iit_block.layers[0].residual1.gate_net[0].weight.grad,
        'Layer1 Attention Q': iit_block.layers[1].attention.q_proj.weight.grad,
    }
    
    for name, grad in grad_check.items():
        if grad is not None:
            print(f"  ✅ {name:30s}: grad.norm() = {grad.norm().item():.6e}")
        else:
            print(f"  ❌ {name:30s}: NO GRADIENT")
    
    # Verificar hidden_states
    print(f"\n🔗 Verificación gradiente en hidden_states:")
    if hidden_states.grad is not None:
        print(f"  ✅ hidden_states.grad.norm() = {hidden_states.grad.norm().item():.6e}")
        print(f"  ✅ Gradientes fluyen correctamente!")
    else:
        print(f"  ❌ hidden_states NO tiene gradiente")
    
    print(f"\n{'='*70}")
    print(f"✅ TEST COMPLETADO - IIT Transformer Layer funcionando!")
    print(f"{'='*70}\n")
