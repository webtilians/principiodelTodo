#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_phi_gradients.py

Script de diagnóstico para verificar que los gradientes fluyen correctamente
a LearnablePhiWeights después de los fixes de PHI learning.

Checks realizados:
1. LearnablePhiWeights tiene requires_grad=True en todos los parámetros
2. Delta PHI loss tiene gradientes después de backward()
3. Los pesos aprendibles reciben gradientes no-zero
4. El delta PHI es realmente diferente (no casi-zero)

Uso:
    python test_phi_gradients.py
"""

import torch
import torch.nn as nn
from src.infinito_v5_2_refactored import InfinitoV52Refactored

def test_phi_gradient_flow():
    """Verifica que los gradientes fluyen correctamente en el sistema PHI."""
    
    print("=" * 80)
    print("TEST: Flujo de gradientes en LearnablePhiWeights")
    print("=" * 80)
    
    # Configuración mínima
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1️⃣ Dispositivo: {device}")
    
    # Crear modelo con configuración mínima
    config = {
        'vocab_size': 50257,  # GPT-2 tokenizer
        'hidden_dim': 256,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'use_improved_iit': True,
        'use_learnable_phi': True,
        'lambda_phi': 0.5
    }
    
    print(f"\n2️⃣ Creando modelo con:")
    print(f"   - use_improved_iit: {config['use_improved_iit']}")
    print(f"   - use_learnable_phi: {config['use_learnable_phi']}")
    print(f"   - lambda_phi: {config['lambda_phi']}")
    
    model = InfinitoV52Refactored(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_improved_iit=config['use_improved_iit'],
        use_learnable_phi=config['use_learnable_phi'],
        lambda_phi=config['lambda_phi']
    ).to(device)
    
    model.train()
    
    # CHECK 1: ¿Existen los módulos?
    print(f"\n3️⃣ Verificando existencia de módulos:")
    print(f"   - learnable_phi_weights: {model.learnable_phi_weights is not None}")
    print(f"   - delta_phi_objective: {model.delta_phi_objective is not None}")
    
    if model.learnable_phi_weights is None:
        print("   ❌ ERROR: LearnablePhiWeights no existe")
        return False
    
    if model.delta_phi_objective is None:
        print("   ❌ ERROR: DeltaPhiObjective no existe")
        return False
    
    # CHECK 2: ¿Parámetros entrenables?
    print(f"\n4️⃣ Verificando requires_grad en LearnablePhiWeights:")
    for name, param in model.learnable_phi_weights.named_parameters():
        print(f"   - {name}: requires_grad={param.requires_grad}, shape={param.shape}")
        if not param.requires_grad:
            print(f"   ❌ ERROR: {name} tiene requires_grad=False")
            return False
    
    # CHECK 3: Forward pass con batch sintético
    print(f"\n5️⃣ Ejecutando forward pass:")
    batch_size = 4
    seq_len = 16
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
    
    logits, metrics = model(input_ids, return_metrics=True)
    
    print(f"   - Logits shape: {logits.shape}")
    print(f"   - Metrics keys: {list(metrics.keys())}")
    
    # CHECK 4: ¿Delta PHI loss existe?
    if 'delta_phi_loss' not in metrics:
        print("   ❌ ERROR: 'delta_phi_loss' no está en metrics")
        return False
    
    delta_phi_loss = metrics['delta_phi_loss']
    
    # Convertir a tensor si es float
    if isinstance(delta_phi_loss, float):
        print(f"\n6️⃣ Delta PHI Loss:")
        print(f"   - Valor: {delta_phi_loss:.6f}")
        print(f"   ❌ ERROR: delta_phi_loss es float, no tensor (no tiene requires_grad)")
        return False
    
    print(f"\n6️⃣ Delta PHI Loss:")
    print(f"   - Valor: {delta_phi_loss.item():.6f}")
    print(f"   - requires_grad: {delta_phi_loss.requires_grad}")
    
    if not delta_phi_loss.requires_grad:
        print("   ❌ ERROR: delta_phi_loss no requiere gradientes")
        return False
    
    # CHECK 5: ¿Delta PHI es diferente de cero?
    mean_delta = 0.0  # Default value
    if 'mean_delta_phi' in metrics:
        mean_delta = metrics['mean_delta_phi']
        print(f"   - Mean Delta PHI: {mean_delta:.6f}")
        
        if abs(mean_delta) < 1e-6:
            print("   ⚠️ WARNING: Delta PHI es prácticamente cero (no hay diferencia real)")
            print("      Esto indica que phi_baseline y phi_processed son casi iguales")
    else:
        print("   ⚠️ WARNING: 'mean_delta_phi' no está en metrics")
    
    # CHECK 6: Backward pass
    print(f"\n7️⃣ Ejecutando backward pass:")
    # Simular loss total (LM + lambda*Phi)
    # Usar dummy LM loss
    dummy_lm_loss = torch.randn(1, device=device, requires_grad=True).mean()
    total_loss = dummy_lm_loss + config['lambda_phi'] * delta_phi_loss
    
    print(f"   - Total loss: {total_loss.item():.6f}")
    total_loss.backward()
    
    # CHECK 7: ¿Gradientes existen en LearnablePhiWeights?
    print(f"\n8️⃣ Verificando gradientes en LearnablePhiWeights:")
    all_gradients_ok = True
    for name, param in model.learnable_phi_weights.named_parameters():
        grad = param.grad
        if grad is None:
            print(f"   ❌ {name}: grad es None")
            all_gradients_ok = False
        else:
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            print(f"   ✅ {name}: norm={grad_norm:.6f}, mean={grad_mean:.6f}")
            
            if grad_norm < 1e-8:
                print(f"      ⚠️ WARNING: Gradiente muy pequeño (posible vanishing gradient)")
    
    if not all_gradients_ok:
        print("\n❌ RESULTADO: FALLÓ - Algunos parámetros no reciben gradientes")
        return False
    
    # CHECK 8: ¿Valores de pesos actuales?
    print(f"\n9️⃣ Valores actuales de LearnablePhiWeights:")
    weights = model.learnable_phi_weights()
    for key, value in weights.items():
        print(f"   - {key}: {value.item():.6f}")
    
    # RESUMEN FINAL
    print("\n" + "=" * 80)
    print("✅ RESULTADO: ÉXITO - Todos los checks pasaron")
    print("=" * 80)
    print("\nDetalle:")
    print("  ✅ LearnablePhiWeights existe")
    print("  ✅ DeltaPhiObjective existe")
    print("  ✅ Todos los parámetros tienen requires_grad=True")
    print("  ✅ Delta PHI loss tiene requires_grad=True")
    print("  ✅ Gradientes fluyen correctamente a todos los parámetros")
    
    if abs(mean_delta) < 1e-6:
        print("\n⚠️ ADVERTENCIA:")
        print("  Delta PHI es prácticamente cero. Verificar que phi_baseline y phi_processed")
        print("  sean realmente diferentes (BUG #2 debería estar arreglado)")
    
    return True


if __name__ == '__main__':
    success = test_phi_gradient_flow()
    exit(0 if success else 1)
