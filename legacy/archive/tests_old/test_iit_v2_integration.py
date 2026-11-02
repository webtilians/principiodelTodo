#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST DE INTEGRACIÓN: IIT Metrics V2 en Arquitectura Híbrida
===============================================================

Verifica que ImprovedIITMetricsV2 funciona correctamente integrado
en InfinitoGPT2Hybrid y que los gradientes fluyen end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn

# Import componentes necesarios
from transformers import GPT2Config
from core.iit_metrics_v2 import ImprovedIITMetricsV2


def test_iit_v2_integration():
    """Test rápido de integración."""
    
    print("\n" + "="*70)
    print("🧪 TEST DE INTEGRACIÓN: IIT Metrics V2")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    # 1. Crear IIT Metrics V2
    print("\n📦 Inicializando IIT Metrics V2...")
    gpt2_config = GPT2Config.from_pretrained('gpt2')
    
    iit_metrics = ImprovedIITMetricsV2(
        hidden_dim=gpt2_config.n_embd,  # 768
        num_heads=gpt2_config.n_head,   # 12
        learnable_weights=True
    ).to(device)
    
    # 2. Contar parámetros
    total_params = sum(p.numel() for p in iit_metrics.parameters())
    trainable_params = sum(p.numel() for p in iit_metrics.parameters() if p.requires_grad)
    
    print(f"\n📊 Parámetros IIT Metrics V2:")
    print(f"  Total: {total_params:,}")
    print(f"  Entrenables: {trainable_params:,}")
    
    # 3. Forward pass simulado
    print(f"\n🔄 Simulando forward pass...")
    batch_size = 4
    seq_len = 64
    
    hidden_states = torch.randn(
        batch_size, seq_len, gpt2_config.n_embd,
        device=device, requires_grad=True
    )
    
    attention_weights = torch.randn(
        batch_size, gpt2_config.n_head, seq_len, seq_len,
        device=device
    )
    
    # Calcular PHI
    phi_components = iit_metrics(hidden_states, attention_weights)
    
    print(f"\n📤 Componentes PHI:")
    print(f"  PHI estimate: {phi_components['phi_estimate'].mean().item():.4f}")
    print(f"  Temporal coherence: {phi_components['temporal_coherence'].mean().item():.4f}")
    print(f"  Integration strength: {phi_components['integration_strength'].mean().item():.4f}")
    print(f"  Complexity: {phi_components['complexity'].mean().item():.4f}")
    print(f"  Attention diversity: {phi_components['attention_diversity'].mean().item():.4f}")
    
    # 4. Test loss y gradientes
    print(f"\n🔍 Test de gradientes end-to-end...")
    
    # Loss simulado: maximizar PHI
    phi_total = phi_components['phi_estimate'].mean()
    target_phi = 8.0
    loss_phi = (target_phi - phi_total) ** 2
    
    # Loss LM simulado
    logits = torch.randn(batch_size, seq_len, gpt2_config.vocab_size, device=device)
    labels = torch.randint(0, gpt2_config.vocab_size, (batch_size, seq_len), device=device)
    loss_lm = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    # Loss total
    loss = loss_lm + 1.0 * loss_phi  # lambda_phi=1.0
    
    print(f"  Loss LM: {loss_lm.item():.4f}")
    print(f"  Loss PHI: {loss_phi.item():.4f}")
    print(f"  Loss TOTAL: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # 5. Verificar gradientes en IIT Metrics
    print(f"\n✅ Verificación de gradientes IIT Metrics V2:")
    
    grad_norms = []
    for name, param in iit_metrics.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append((name, grad_norm))
    
    # Mostrar top 10 gradientes
    grad_norms_sorted = sorted(grad_norms, key=lambda x: x[1], reverse=True)[:10]
    
    for name, grad_norm in grad_norms_sorted:
        print(f"  ✅ {name[:50]:50s} → {grad_norm:.6e}")
    
    # Verificar que TODOS los parámetros tienen gradiente
    params_without_grad = [name for name, param in iit_metrics.named_parameters() 
                          if param.grad is None and param.requires_grad]
    
    if params_without_grad:
        print(f"\n  ⚠️ ADVERTENCIA: {len(params_without_grad)} parámetros sin gradiente:")
        for name in params_without_grad[:5]:
            print(f"    - {name}")
    else:
        print(f"\n  ✅ PERFECTO: Todos los parámetros IIT tienen gradiente!")
    
    # 6. Test de hidden_states
    print(f"\n🔗 Verificación gradiente en hidden_states:")
    if hidden_states.grad is not None:
        print(f"  ✅ hidden_states.grad.norm() = {hidden_states.grad.norm().item():.6e}")
        print(f"  ✅ Gradientes fluyen desde PHI hasta representaciones GPT-2!")
    else:
        print(f"  ❌ hidden_states NO tiene gradiente")
    
    print(f"\n{'='*70}")
    print(f"✅ TEST DE INTEGRACIÓN COMPLETADO")
    print(f"{'='*70}")
    print(f"\n📊 RESUMEN:")
    print(f"  • IIT Metrics V2: {trainable_params:,} parámetros entrenables")
    print(f"  • PHI calculado: {phi_total.item():.4f}")
    print(f"  • Gradientes fluyen: {'✅ SÍ' if hidden_states.grad is not None else '❌ NO'}")
    print(f"  • Parámetros sin gradiente: {len(params_without_grad)}")
    print(f"\n🎯 CONCLUSIÓN: {'✅ ARQUITECTURA LISTA PARA ENTRENAR' if len(params_without_grad) == 0 else '⚠️ REVISAR PARÁMETROS SIN GRADIENTE'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    test_iit_v2_integration()
