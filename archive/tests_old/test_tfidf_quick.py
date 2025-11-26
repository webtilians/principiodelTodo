"""
🔬 TEST RÁPIDO: ¿TF-IDF diferencia los textos ahora?
"""

import torch
import sys
sys.path.append('src')

from infinito_gpt_text_fixed import SemanticTextEmbedder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("🔬 TEST RÁPIDO: TF-IDF con vocabulario pre-entrenado")
print("="*80)

embedder = SemanticTextEmbedder(embed_dim=256, use_glove=False)

textos = [
    "mi perro es rojo",
    "mi perro es verde",
    "la mesa es roja",
    "yo pienso, luego existo",
]

embeddings = []

print("\n📊 Embeddings generados:")
print("-"*80)

for texto in textos:
    emb = embedder.text_to_tensor(texto, device)
    embeddings.append(emb)
    print(f"\n'{texto}':")
    print(f"  Norm: {emb.norm().item():.6f}")
    print(f"  Mean: {emb.mean().item():.6f}")
    print(f"  Max:  {emb.max().item():.6f}")

print("\n"+"="*80)
print("📊 MATRIZ DE DISTANCIAS L2")
print("="*80)

for i, texto1 in enumerate(textos):
    for j, texto2 in enumerate(textos):
        if j > i:
            dist = (embeddings[i] - embeddings[j]).norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(embeddings[i], embeddings[j], dim=1).item()
            print(f"\n'{texto1}' <-> '{texto2}':")
            print(f"  L2 distance:       {dist:.6f}")
            print(f"  Cosine similarity: {cos_sim:.6f}")

print("\n"+"="*80)
print("🎯 CONCLUSIÓN")
print("="*80)

# Comparar perro rojo vs perro verde (cambio semántico)
dist_perro = (embeddings[0] - embeddings[1]).norm().item()

# Comparar perro rojo vs mesa roja (cambio conceptual)
dist_concepto = (embeddings[0] - embeddings[2]).norm().item()

print(f"\n🔴 Perro rojo vs verde (semántica):  {dist_perro:.6f}")
print(f"🟡 Perro rojo vs mesa roja (concepto): {dist_concepto:.6f}")

if dist_perro > 0.5 and dist_concepto > 0.5:
    print("\n✅ TF-IDF FUNCIONA CORRECTAMENTE")
    print("   Los embeddings capturan diferencias semánticas y conceptuales")
elif dist_perro < 0.01:
    print("\n❌ PROBLEMA: Embeddings casi idénticos para textos diferentes")
else:
    print("\n⚠️ PARCIAL: Diferencias pequeñas pero detectables")
