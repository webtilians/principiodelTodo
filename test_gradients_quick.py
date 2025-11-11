#!/usr/bin/env python3
"""
🔍 TEST RÁPIDO DE GRADIENTES IIT
================================

Script para verificar rápidamente si los gradientes fluyen correctamente
desde delta_phi_loss hacia los parámetros IIT.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from infinito_v5_2_refactored import (
    IITGuidedMemory,
    ImprovedIITMetrics,
    LearnablePhiWeights
)

print("="*70)
print("🔍 TEST DE GRADIENTES - Sistema IIT")
print("="*70)

# Crear componentes IIT
hidden_dim = 768
memory = IITGuidedMemory(
    hidden_dim=hidden_dim,
    memory_slots=256,
    use_phi_priority=True,
    learnable_threshold=True
)

iit_metrics = ImprovedIITMetrics(
    hidden_dim=hidden_dim,
    perplexity=None,
    learnable_weights=False
)

phi_weights = LearnablePhiWeights()

print(f"\n✓ Componentes IIT inicializados")
print(f"  Hidden dim: {hidden_dim}")

# Crear datos de prueba
batch_size, seq_len = 4, 32
hidden_state = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
attention_weights = torch.softmax(torch.randn(batch_size, 12, seq_len, seq_len), dim=-1)

print(f"\n✓ Datos de prueba creados")
print(f"  Batch: {batch_size}, Seq len: {seq_len}")

# Forward pass IIT
print(f"\n📊 Calculando métricas IIT...")
phi_components = iit_metrics(
    hidden_state=hidden_state,
    attention_weights=attention_weights
)

integration_phi = phi_components['phi_estimate'].mean()
print(f"  PHI estimado: {integration_phi.item():.4f}")
print(f"  PHI requires_grad: {integration_phi.requires_grad}")

# Calcular delta_phi_loss (SIN .item() en target)
lambda_phi = 0.6
target_phi = 6.0  # Fijo, no usar .item()
delta_phi_loss = lambda_phi * (target_phi - integration_phi)

print(f"\n🎯 Delta PHI Loss:")
print(f"  Target: {target_phi:.2f}")
print(f"  Current: {integration_phi.item():.4f}")
print(f"  Delta loss: {delta_phi_loss.item():.4f}")
print(f"  Delta loss requires_grad: {delta_phi_loss.requires_grad}")

# Backward
print(f"\n⬅️  Ejecutando backward...")
delta_phi_loss.backward()

# Verificar gradientes
print(f"\n📈 VERIFICACIÓN DE GRADIENTES:")

# 1. Hidden state
if hidden_state.grad is not None:
    print(f"  ✅ hidden_state.grad: {hidden_state.grad.norm().item():.6e}")
else:
    print(f"  ❌ hidden_state.grad: None")

# 2. Parámetros IIT Metrics
print(f"\n  Parámetros IIT Metrics:")
iit_grad_found = False
for name, param in iit_metrics.named_parameters():
    if param.grad is not None:
        print(f"    ✅ {name}: {param.grad.norm().item():.6e}")
        iit_grad_found = True
    else:
        print(f"    ❌ {name}: None")

if not iit_grad_found:
    print(f"    ⚠️  NO hay gradientes en IIT Metrics")

# 3. Parámetros Memory
print(f"\n  Parámetros Memory:")
memory_grad_found = False
for name, param in memory.named_parameters():
    if param.grad is not None:
        print(f"    ✅ {name}: {param.grad.norm().item():.6e}")
        memory_grad_found = True
    else:
        print(f"    ❌ {name}: None")

if not memory_grad_found:
    print(f"    ⚠️  NO hay gradientes en Memory")

# 4. PHI Weights
print(f"\n  Parámetros PHI Weights:")
phi_weights_grad_found = False
for name, param in phi_weights.named_parameters():
    if param.grad is not None:
        print(f"    ✅ {name}: {param.grad.norm().item():.6e}")
        phi_weights_grad_found = True
    else:
        print(f"    ❌ {name}: None")

if not phi_weights_grad_found:
    print(f"    ⚠️  NO hay gradientes en PHI Weights")

# CONCLUSIÓN
print(f"\n" + "="*70)
if iit_grad_found or memory_grad_found or phi_weights_grad_found:
    print("✅ RESULTADO: Los gradientes SÍ fluyen hacia parámetros IIT")
else:
    print("❌ RESULTADO: Los gradientes NO fluyen hacia parámetros IIT")
    print("   PROBLEMA: El cálculo de PHI no está conectado al grafo computacional")
print("="*70)
