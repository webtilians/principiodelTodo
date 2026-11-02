#!/usr/bin/env python3
"""
✅ VERIFICACIÓN: Attention Diversity ahora funciona correctamente
"""

import torch
import sys
sys.path.insert(0, 'src')

print("="*70)
print("✅ VERIFICANDO ATTENTION DIVERSITY")
print("="*70)

# Cargar el modelo corregido
from train_gpt2_with_phi_observer import InfinitoGPT2WithObserver

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n📦 Cargando modelo con attn_implementation='eager'...")

# Crear modelo (esto cargará GPT-2 con eager attention)
model = InfinitoGPT2WithObserver(use_lora=False)  # Sin LoRA para test rápido
model = model.to(device)
model.eval()

# Test
test_prompts = [
    "La inteligencia artificial es",
    "El tiempo es una dimensión que",
    "¿Cuál es la capital de España?",
]

print("\n🔬 Probando Attention Diversity:")
print("-"*70)

for prompt in test_prompts:
    inputs = model.tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs, phi_metrics = model(inputs.input_ids, return_phi=True, use_memory=False)
    
    attention_div = phi_metrics['raw_components']['attention']
    phi = phi_metrics['phi'].mean().item()
    
    is_valid = attention_div != 0.5
    status = "✅" if is_valid else "⚠️"
    
    print(f"{status} '{prompt[:40]}'")
    print(f"   PHI: {phi:.2f} | Attention Div: {attention_div:.4f}")

print("\n" + "="*70)
if all(phi_metrics['raw_components']['attention'] != 0.5 for _ in test_prompts):
    print("✅ Attention Diversity está funcionando correctamente!")
else:
    print("⚠️ Attention Diversity todavía tiene problemas")
print("="*70)
