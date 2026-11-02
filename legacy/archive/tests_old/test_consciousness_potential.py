"""
🔬 TEST CRÍTICO: ¿analyze_text_consciousness_potential genera valores diferentes?
"""

import sys
sys.path.append('src')

from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough
from argparse import Namespace

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
print("🔬 ANÁLISIS: analyze_text_consciousness_potential")
print("="*80)

for texto in textos:
    analysis = model.analyze_text_consciousness_potential(texto)
    
    print(f"\n📝 '{texto}':")
    print(f"   Word count: {analysis['word_count']}")
    print(f"   Consciousness score: {analysis['consciousness_score']:.6f}")
    print(f"   Dominant modality: {analysis['dominant_modality']}")
    print(f"   Keys disponibles: {list(analysis.keys())}")

print("\n" + "="*80)
print("🎯 CONCLUSIÓN")
print("="*80)

# Verificar si todos tienen el mismo consciousness_score
scores = [model.analyze_text_consciousness_potential(t)['consciousness_score'] for t in textos]

if len(set(scores)) == 1:
    print(f"\n❌ PROBLEMA IDENTIFICADO:")
    print(f"   Todos los textos generan el MISMO consciousness_score: {scores[0]:.6f}")
    print(f"   → Esto hace que visual_intensity, auditory_intensity, etc. sean iguales")
    print(f"   → Por eso las normas de generate_text_based_input son idénticas")
elif len(set(scores)) < len(scores):
    print(f"\n⚠️ PROBLEMA PARCIAL:")
    print(f"   Algunos textos comparten consciousness_score")
    unique_scores = set(scores)
    print(f"   Scores únicos: {unique_scores}")
else:
    print(f"\n✅ OK:")
    print(f"   Cada texto genera un consciousness_score diferente")
    print(f"   Scores: {scores}")
