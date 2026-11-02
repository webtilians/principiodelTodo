"""
🔬 TEST: ¿Las normas de generate_text_based_input ahora son diferentes?
"""

import torch
import sys
sys.path.append('src')

from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough
from argparse import Namespace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

args = Namespace(
    batch_size=4,
    input_dim=256,
    hidden_dim=512,
    attention_heads=8,
    memory_slots=256,
    lr=1e-3,
    text_mode=True,
    input_text=None,
    quantum_active=False,
    target_consciousness=0.6,
    max_consciousness=0.9
)

model = InfinitoV51ConsciousnessBreakthrough(args)

textos = [
    "mi perro es rojo",
    "mi perro es verde",
    "la mesa es roja",
    "yo pienso, luego existo",
]

print("="*80)
print("🔬 TEST: Normas de generate_text_based_input después de mejoras")
print("="*80)

norms = []

for texto in textos:
    with torch.no_grad():
        input_tensor = model.generate_text_based_input(texto)
    
    norm = input_tensor.norm().item()
    mean = input_tensor.mean().item()
    
    norms.append(norm)
    
    print(f"\n📝 '{texto}':")
    print(f"   Shape: {input_tensor.shape}")
    print(f"   Norm:  {norm:.6f}")
    print(f"   Mean:  {mean:.6f}")

print("\n" + "="*80)
print("🎯 ANÁLISIS DE DIFERENCIAS")
print("="*80)

# Calcular varianza entre normas
import numpy as np
variance = np.var(norms)
std = np.std(norms)
diff_max = max(norms) - min(norms)

print(f"\nEstadísticas de normas:")
print(f"   Min:      {min(norms):.6f}")
print(f"   Max:      {max(norms):.6f}")
print(f"   Range:    {diff_max:.6f}")
print(f"   Variance: {variance:.6f}")
print(f"   Std Dev:  {std:.6f}")

if variance < 0.001:
    print(f"\n❌ PROBLEMA PERSISTE: Normas casi idénticas")
    print(f"   Las mejoras no fueron suficientes")
elif variance < 0.1:
    print(f"\n⚠️ MEJORA PARCIAL: Normas ligeramente diferentes")
    print(f"   Puede no ser suficiente para discriminación")
else:
    print(f"\n✅ ÉXITO: Normas significativamente diferentes")
    print(f"   Suficiente para discriminar entre textos")

# Calcular diferencias específicas
print(f"\n📊 Diferencias específicas:")
print(f"   'perro rojo' vs 'perro verde': {abs(norms[0] - norms[1]):.6f}")
print(f"   'perro rojo' vs 'mesa roja':   {abs(norms[0] - norms[2]):.6f}")
print(f"   'perro rojo' vs 'cogito':      {abs(norms[0] - norms[3]):.6f}")
