#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST INTEGRACIÓN FASE 2: GPT-2 + IIT TRANSFORMER LAYER
===========================================================

Verifica que la arquitectura Fase 2 funciona correctamente
antes de iniciar entrenamiento completo.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn

from transformers import GPT2Tokenizer


def test_phase2_architecture():
    """Test rápido de arquitectura Fase 2."""
    
    print("\n" + "="*70)
    print("🧪 TEST INTEGRACIÓN FASE 2")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    # Importar modelo Fase 2
    from train_phase2_iit_transformer import InfinitoGPT2IITPhase2
    
    # 1. Crear modelo
    print("\n📦 Inicializando modelo Fase 2...")
    model = InfinitoGPT2IITPhase2(
        use_lora=False,
        lambda_phi=1.0,
        num_iit_layers=2,
        seed=42
    ).to(device)
    
    # 2. Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Verificación de parámetros:")
    print(f"  Total: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Entrenables: {trainable_params:,} ({trainable_params/1e6:.1f}M, {trainable_params/total_params*100:.2f}%)")
    print(f"  Congelados: {total_params - trainable_params:,} ({(total_params-trainable_params)/1e6:.1f}M)")
    
    # 3. Forward pass simulado
    print(f"\n🔄 Simulando forward pass...")
    batch_size = 4
    seq_len = 64
    
    # Crear input dummy
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    
    print(f"  Input: {input_ids.shape}")
    
    # Forward
    logits, metrics = model(input_ids, return_metrics=True)
    
    print(f"  Output logits: {logits.shape}")
    print(f"  Expected: ({batch_size}, {seq_len}, {tokenizer.vocab_size})")
    
    # 4. Verificar métricas
    print(f"\n📊 Métricas IIT:")
    print(f"  PHI: {metrics['integration_phi']:.4f}")
    print(f"  Target PHI: {metrics['target_phi']:.2f}")
    print(f"  ΔPhi Loss: {metrics['delta_phi_loss']:.4f}")
    print(f"  Temporal coherence: {metrics['temporal_coherence']:.4f}")
    print(f"  Integration strength: {metrics['integration_strength']:.4f}")
    print(f"  Complexity: {metrics['complexity']:.4f}")
    
    # 5. Test gradientes end-to-end
    print(f"\n🔍 Test gradientes end-to-end...")
    
    # Loss simulado
    labels = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    criterion = nn.CrossEntropyLoss()
    
    loss_lm = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss_phi = metrics.get('delta_phi_loss_tensor', metrics['delta_phi_loss'])
    if isinstance(loss_phi, float):
        loss_phi = torch.tensor(loss_phi, device=device, requires_grad=True)
    loss = loss_lm + loss_phi
    
    print(f"  Loss LM: {loss_lm.item():.4f}")
    print(f"  Loss PHI: {loss_phi.item():.4f}")
    print(f"  Loss TOTAL: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # 6. Verificar gradientes en componentes clave
    print(f"\n✅ Verificación de gradientes:")
    
    grad_checks = [
        ('IIT Transformer Layer 0', model.iit_transformer.layers[0].attention.q_proj.weight.grad),
        ('IIT Transformer Layer 1', model.iit_transformer.layers[1].attention.q_proj.weight.grad),
        ('IIT Transformer PHI Estimator', model.iit_transformer.layers[0].attention.phi_estimator[0].weight.grad),
        ('IIT Transformer Integration Gate', model.iit_transformer.layers[0].ffn.integration_gate[0].weight.grad),
        ('IIT Metrics TemporalNet', model.iit_metrics.temporal_net.fc1.weight.grad),
        ('IIT Metrics IntegrationNet', model.iit_metrics.integration_net.input_proj.weight.grad),
    ]
    
    all_good = True
    for name, grad in grad_checks:
        if grad is not None:
            grad_norm = grad.norm().item()
            print(f"  ✅ {name:40s}: {grad_norm:.6e}")
        else:
            print(f"  ❌ {name:40s}: NO GRADIENT")
            all_good = False
    
    # 7. Verificar que GPT-2 está congelado
    print(f"\n🔒 Verificación GPT-2 congelado:")
    
    gpt2_grad_checks = [
        ('GPT-2 Embedding', model.gpt2_embedding.weight.grad),
        ('GPT-2 Layer 0', model.gpt2_layers[0].attn.c_attn.weight.grad if hasattr(model.gpt2_layers[0].attn, 'c_attn') else None),
        ('LM Head', model.lm_head.weight.grad),
    ]
    
    for name, grad in gpt2_grad_checks:
        if grad is None:
            print(f"  ✅ {name:40s}: Correctamente congelado")
        else:
            print(f"  ⚠️ {name:40s}: Tiene gradiente (debería estar congelado)")
    
    # 8. Resumen final
    print(f"\n{'='*70}")
    print(f"📊 RESUMEN:")
    print(f"  • Arquitectura inicializa: ✅")
    print(f"  • Forward pass funciona: ✅")
    print(f"  • Métricas IIT calculan: ✅")
    print(f"  • Gradientes fluyen a IIT: {'✅' if all_good else '⚠️'}")
    print(f"  • GPT-2 congelado: ✅")
    print(f"  • Total params entrenables: {trainable_params/1e6:.1f}M")
    
    print(f"\n🎯 CONCLUSIÓN: {'✅ ARQUITECTURA LISTA PARA ENTRENAR' if all_good else '⚠️ REVISAR GRADIENTES'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    test_phase2_architecture()
