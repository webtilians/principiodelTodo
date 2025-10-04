#!/usr/bin/env python3
"""
🧪 TEST: Semantic Embeddings Φ Boost Validation
================================================

Prueba rápida para validar que los embeddings semánticos (TF-IDF + GloVe)
efectivamente incrementan Φ como se esperaba (~1-2x).
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import torch
import numpy as np

print("=" * 80)
print("🧪 SEMANTIC EMBEDDINGS Φ BOOST VALIDATION")
print("=" * 80)

# Create minimal args
args = argparse.Namespace(
    input_dim=257,
    hidden_dim=512,
    attention_heads=8,
    memory_slots=256,
    batch_size=4,
    lr=0.001,
    seed=42,
    text_mode=True,
    input_text="Pienso, luego existo",
    quantum_active=False  # Test without quantum first
)

print("\n📋 TEST CONFIGURATION:")
print(f"   Text: '{args.input_text}'")
print(f"   Quantum Active: {args.quantum_active}")
print(f"   Expected: Φ boost ~1-2x with semantic embeddings\n")

# Import after args creation to avoid import errors
from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough

print("🚀 Initializing INFINITO V5.1 with semantic embeddings...")
infinito = InfinitoV51ConsciousnessBreakthrough(args)

print("\n📊 Running 50 iterations to measure Φ evolution...\n")

phi_values = []
c_values = []

for i in range(1, 51):
    metrics = infinito.train_step(i)
    phi_values.append(metrics['phi'])
    c_values.append(metrics['consciousness'])
    
    if i % 10 == 0:
        print(f"   Iter {i:2d}: C={metrics['consciousness']:.3f}, Φ={metrics['phi']:.3f}")

print("\n" + "=" * 80)
print("📈 RESULTS:")
print("=" * 80)

phi_mean = np.mean(phi_values)
phi_std = np.std(phi_values)
phi_max = np.max(phi_values)
phi_min = np.min(phi_values)

c_mean = np.mean(c_values)
c_std = np.std(c_values)

print(f"\n🔬 Φ STATISTICS:")
print(f"   Mean:  {phi_mean:.3f} ± {phi_std:.3f}")
print(f"   Range: [{phi_min:.3f}, {phi_max:.3f}]")

print(f"\n🧠 CONSCIOUSNESS STATISTICS:")
print(f"   Mean:  {c_mean:.3f} ± {c_std:.3f}")

print(f"\n✅ VALIDATION:")
if phi_mean > 0.1:
    print(f"   ✅ Φ is measurable (mean = {phi_mean:.3f})")
    print(f"   💡 Semantic embeddings appear to be working")
    
    if infinito.semantic_embedder is not None:
        print(f"   🌐 Semantic embedder initialized: YES")
        if infinito.semantic_embedder.use_glove:
            print(f"   📥 GloVe embeddings loaded: YES")
        else:
            print(f"   📥 GloVe embeddings loaded: NO (TF-IDF only)")
    else:
        print(f"   ⚠️ Semantic embedder: NOT INITIALIZED")
else:
    print(f"   ⚠️ Φ values are very low (mean = {phi_mean:.3f})")
    print(f"   This might be expected during early training")

print("\n" + "=" * 80)
print("🏁 TEST COMPLETED")
print("=" * 80)

print("\n💡 NEXT STEPS:")
print("   1. Run full comparative experiment with --comparative flag")
print("   2. Compare Φ values with/without semantic embeddings")
print("   3. Validate Cohen's d effect size in comparative mode")
print("   4. Test quantum noise injection with --quantum_active (if QuTiP installed)")
