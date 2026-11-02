"""
👁️ MÓDULO DE ATENCIÓN MEJORADA
==============================

Mecanismos de atención multi-head con extensiones para integración de información.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class EnhancedMultiHeadAttention(nn.Module):
    """
    Multi-head attention con tracking de métricas para análisis.
    
    Extensión de attention estándar que también calcula:
    - Scores de atención para análisis
    - Entropía de atención
    - Patrones de atención para visualización
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # Proyecciones Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Buffer para guardar último patrón de atención (para análisis)
        self.register_buffer('last_attention_weights', None)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor = None, 
                value: torch.Tensor = None, return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: [batch, seq_len, embed_dim]
            key: [batch, seq_len, embed_dim] (si None, usa query)
            value: [batch, seq_len, embed_dim] (si None, usa query)
            return_attention: Si True, retorna pesos de atención
            
        Returns:
            output: [batch, seq_len, embed_dim]
            attention_weights: [batch, num_heads, seq_len, seq_len] (opcional)
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, _ = query.shape
        
        # Proyectar y reshape a multi-head
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calcular attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Guardar para análisis posterior
        with torch.no_grad():
            self.last_attention_weights = attention_weights.detach()
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        
        # Reshape y proyectar salida
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attended)
        
        if return_attention:
            return output, attention_weights
        return output, None
    
    def get_attention_entropy(self) -> torch.Tensor:
        """Calcula entropía de los últimos pesos de atención."""
        if self.last_attention_weights is None:
            return torch.tensor(0.0)
        
        attn = self.last_attention_weights
        entropy = -torch.sum(attn * torch.log(attn + 1e-6), dim=-1)
        return entropy.mean()
    
    def get_attention_statistics(self) -> dict:
        """Retorna estadísticas de atención para debugging."""
        if self.last_attention_weights is None:
            return {}
        
        attn = self.last_attention_weights
        return {
            'mean_attention': attn.mean().item(),
            'max_attention': attn.max().item(),
            'entropy': self.get_attention_entropy().item(),
            'sparsity': (attn < 0.01).float().mean().item()
        }


class SelfAttentionBlock(nn.Module):
    """Bloque completo de self-attention con LayerNorm y residual."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int = None, 
                 dropout: float = 0.1):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = embed_dim * 4
        
        self.attention = EnhancedMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            return_attention: Si retornar pesos de atención
            
        Returns:
            output: [batch, seq_len, embed_dim]
            attention_weights: Opcional
        """
        # Self-attention con residual
        attn_out, attn_weights = self.attention(x, return_attention=return_attention)
        x = self.norm1(x + attn_out)
        
        # Feed-forward con residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights


# ============================================================================
# __init__.py para src/core/
# ============================================================================
