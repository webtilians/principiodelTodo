#!/usr/bin/env python3
"""
🔍 INSPECCIÓN DE LOGITS RAW
============================

Ver los valores EXACTOS de los logits para diferentes textos
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough


class Args:
    def __init__(self):
        self.input_dim = 257
        self.batch_size = 4
        self.hidden_dim = 512
        self.attention_heads = 8
        self.memory_slots = 256
        self.lr = 1e-3
        self.seed = 42
        self.input_text = None
        self.text_mode = True
        self.max_iter = 1000
        self.comparative = False
        self.comparative_iterations = 100
        self.bootstrap_samples = 1000


print("🔍 Inspección de Logits RAW")
print("="*70)

infinito = InfinitoV51ConsciousnessBreakthrough(Args())

texts = [
    "mi perro es rojo",
    "mi perro es verde", 
    "mi gato es azul",
    "yo pienso luego existo"
]

print(f"\n📋 Procesando {len(texts)} textos...\n")

all_logits = []

for text in texts:
    # Generate input
    inputs = infinito.generate_text_based_input(
        text=text,
        batch_size=infinito.batch_size,
        seq_len=64
    )
    
    # Forward pass
    with torch.no_grad():
        infinito.model.eval()
        consciousness, phi, debug_info = infinito.model(inputs)
    
    # Get logits
    phi_info = debug_info.get('phi_info', {})
    causal_logits = phi_info.get('causal_logits')
    
    if causal_logits is not None:
        if isinstance(causal_logits, torch.Tensor):
            causal_logits = causal_logits.cpu().detach().numpy()
        
        # Extraer 6 valores
        logits = np.array([
            causal_logits[0, 1],  # visual → auditory
            causal_logits[0, 2],  # visual → motor
            causal_logits[0, 3],  # visual → executive
            causal_logits[1, 2],  # auditory → motor
            causal_logits[1, 3],  # auditory → executive
            causal_logits[2, 3],  # motor → executive
        ])
        
        all_logits.append(logits)
        
        print(f"'{text}':")
        print(f"  visual→auditory  : {logits[0]:+.6f}")
        print(f"  visual→motor     : {logits[1]:+.6f}")
        print(f"  visual→executive : {logits[2]:+.6f}")
        print(f"  auditory→motor   : {logits[3]:+.6f}")
        print(f"  auditory→executive: {logits[4]:+.6f}")
        print(f"  motor→executive  : {logits[5]:+.6f}")
        print(f"  Mean: {logits.mean():+.6f}, Std: {logits.std():.6f}")
        print()

# Comparaciones
print("\n" + "="*70)
print("📊 COMPARACIONES (diferencias absolutas)")
print("="*70 + "\n")

for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        diff = np.abs(all_logits[i] - all_logits[j])
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f"'{texts[i]}' vs '{texts[j]}':")
        print(f"  Diferencia máxima: {max_diff:.6f}")
        print(f"  Diferencia promedio: {mean_diff:.6f}")
        print(f"  Diferencias por conexión:")
        names = ["v→a", "v→m", "v→e", "a→m", "a→e", "m→e"]
        for k, name in enumerate(names):
            print(f"    {name}: {diff[k]:.6f}")
        print()

print("="*70)
print("💡 ANÁLISIS")
print("="*70)
print("Si las diferencias son < 0.01, los logits SON idénticos")
print("Si las diferencias son > 0.5, hay discriminación real")
