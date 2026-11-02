"""
🌙 DEMO: Sistema de Consolidación de Memoria
=============================================

Este script demuestra cómo funciona el aprendizaje continuo
con LoRA + Replay Buffer.

Simula:
1. Añadir interacciones al buffer
2. Entrenar los adapters (consolidación)
3. Ver cómo mejora la predicción de importancia
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import numpy as np
from datetime import datetime

from neural_memory import NeuralMemoryManager, ReplayBuffer
from lora_adapter import LoRAAdapter

print("=" * 70)
print("🌙 DEMO: CONSOLIDACIÓN DE MEMORIA NEURONAL")
print("=" * 70)

# =============================================================================
# 1. CREAR EL NEURAL MEMORY MANAGER
# =============================================================================
print("\n📦 PASO 1: Creando Neural Memory Manager...")

manager = NeuralMemoryManager(
    base_model_path="models/super_golden_seed_54percent.pt",
    hidden_dim=64,
    lora_rank=8,
    auto_consolidate_every=100  # Desactivamos auto para demo manual
)

print(f"\n📊 Estado inicial:")
stats = manager.get_stats()
for k, v in stats.items():
    print(f"   {k}: {v}")

# =============================================================================
# 2. SIMULAR INTERACCIONES (como si el usuario chateara)
# =============================================================================
print("\n" + "=" * 70)
print("💬 PASO 2: Simulando interacciones del usuario...")
print("=" * 70)

# Interacciones simuladas con diferentes niveles de importancia
interacciones = [
    # (texto, importancia, categoría)
    ("Me llamo Enrique García", 0.92, "identity"),
    ("Vivo en Madrid, España", 0.85, "location"),
    ("Hola qué tal", 0.12, "greeting"),
    ("Mi cumpleaños es el 15 de marzo", 0.88, "personal"),
    ("Trabajo como desarrollador de software", 0.82, "professional"),
    ("Buenos días", 0.08, "greeting"),
    ("Me gusta programar en Python", 0.75, "interests"),
    ("Tengo una hija que se llama Luna", 0.90, "family"),
    ("Hace buen tiempo hoy", 0.10, "trivial"),
    ("Mi email es enrique@ejemplo.com", 0.80, "contact"),
    ("Mañana tengo reunión a las 10", 0.78, "schedule"),
    ("Ok", 0.05, "trivial"),
    ("Me encanta el café por las mañanas", 0.65, "habits"),
    ("Estoy aprendiendo machine learning", 0.72, "learning"),
    ("Mi color favorito es el azul", 0.55, "preferences"),
    ("Sí", 0.03, "trivial"),
    ("Tengo dos gatos llamados Pixel y Byte", 0.70, "pets"),
    ("Los fines de semana me gusta hacer senderismo", 0.68, "hobbies"),
    ("Mi película favorita es Inception", 0.60, "entertainment"),
    ("Nací en 1990", 0.85, "identity"),
]

print(f"\n📝 Añadiendo {len(interacciones)} interacciones al buffer...\n")

for i, (texto, importancia, categoria) in enumerate(interacciones):
    # Crear embedding fake (en producción vendría de OpenAI)
    fake_embedding = (torch.randn(64) * 0.1).tolist()
    
    # Añadir al buffer (sin entrenar aún)
    manager.replay_buffer.add(
        text=texto,
        embedding=fake_embedding,
        importance=importancia,
        category=categoria,
        metrics={"phi": importancia * 0.8, "coherence": 0.7}
    )
    
    # Mostrar progreso
    emoji = "🔴" if importancia < 0.3 else "🟡" if importancia < 0.6 else "🟢"
    print(f"   {emoji} [{importancia:.2f}] {texto[:40]}...")

print(f"\n✅ Buffer size: {len(manager.replay_buffer)}")

# =============================================================================
# 3. MOSTRAR DISTRIBUCIÓN DE IMPORTANCIA
# =============================================================================
print("\n" + "=" * 70)
print("📊 PASO 3: Análisis del buffer antes de consolidar")
print("=" * 70)

# Estadísticas
importancias = [e["importance"] for e in manager.replay_buffer.buffer]
print(f"\n   Importancia promedio: {np.mean(importancias):.3f}")
print(f"   Importancia máxima:   {np.max(importancias):.3f}")
print(f"   Importancia mínima:   {np.min(importancias):.3f}")
print(f"   Desviación estándar:  {np.std(importancias):.3f}")

# Distribución por categoría
categorias = {}
for e in manager.replay_buffer.buffer:
    cat = e["category"]
    if cat not in categorias:
        categorias[cat] = []
    categorias[cat].append(e["importance"])

print(f"\n   📂 Por categoría:")
for cat, imps in sorted(categorias.items(), key=lambda x: -np.mean(x[1])):
    print(f"      {cat}: {np.mean(imps):.2f} promedio ({len(imps)} items)")

# =============================================================================
# 4. PROBAR PREDICCIÓN ANTES DE CONSOLIDAR
# =============================================================================
print("\n" + "=" * 70)
print("🔮 PASO 4: Predicción de importancia ANTES de consolidar")
print("=" * 70)

# Crear algunos embeddings de prueba
test_cases = [
    ("Información personal importante", 0.9),  # Debería predecir alto
    ("Saludo trivial", 0.1),  # Debería predecir bajo
]

print("\n   El modelo NO ha aprendido aún de tus datos...")
print("   (Las predicciones serán aleatorias)\n")

for texto, esperado in test_cases:
    fake_emb = torch.randn(1, 64).to(manager.device)
    pred, _ = manager.predict_importance(fake_emb, texto)
    diff = abs(pred - esperado)
    emoji = "✅" if diff < 0.3 else "❌"
    print(f"   {emoji} '{texto}': predicho={pred:.3f}, esperado={esperado:.1f}")

# =============================================================================
# 5. CONSOLIDACIÓN (ENTRENAMIENTO)
# =============================================================================
print("\n" + "=" * 70)
print("🌙 PASO 5: CONSOLIDACIÓN (Entrenando adapters con replay)")
print("=" * 70)

print("\n   Esto es como 'dormir' - el cerebro consolida las memorias")
print("   mezclando experiencias recientes con antiguas...\n")

# Múltiples rondas de consolidación para ver progreso
losses = []
for ronda in range(5):
    print(f"   🔄 Ronda {ronda + 1}/5...")
    stats = manager.consolidate(epochs=3, batch_size=16)
    loss = stats.get("avg_loss", 0)
    losses.append(loss)
    print(f"      Loss: {loss:.4f}")

print(f"\n   📉 Progreso del loss:")
for i, loss in enumerate(losses):
    bar = "█" * int(loss * 50)
    print(f"      Ronda {i+1}: {loss:.4f} {bar}")

mejora = (losses[0] - losses[-1]) / losses[0] * 100 if losses[0] > 0 else 0
print(f"\n   ✅ Mejora total: {mejora:.1f}%")

# =============================================================================
# 6. PROBAR PREDICCIÓN DESPUÉS DE CONSOLIDAR
# =============================================================================
print("\n" + "=" * 70)
print("🔮 PASO 6: Predicción de importancia DESPUÉS de consolidar")
print("=" * 70)

print("\n   El modelo ha aprendido de tus patrones...")
print("   (Las predicciones deberían ser más precisas)\n")

for texto, esperado in test_cases:
    fake_emb = torch.randn(1, 64).to(manager.device)
    pred, _ = manager.predict_importance(fake_emb, texto)
    diff = abs(pred - esperado)
    emoji = "✅" if diff < 0.3 else "⚠️"
    print(f"   {emoji} '{texto}': predicho={pred:.3f}, esperado={esperado:.1f}")

# =============================================================================
# 7. VER MEMORIAS DE ALTA IMPORTANCIA
# =============================================================================
print("\n" + "=" * 70)
print("⭐ PASO 7: Memorias de alta importancia (lo que recordar)")
print("=" * 70)

high_importance = manager.replay_buffer.get_high_importance(threshold=0.7, limit=5)
print(f"\n   Top {len(high_importance)} memorias importantes:\n")
for mem in high_importance:
    print(f"   🟢 [{mem['importance']:.2f}] {mem['text']}")
    print(f"      Categoría: {mem['category']}")

# =============================================================================
# 8. RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("📋 RESUMEN: Cómo funciona la consolidación")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────┐
│                    FLUJO DE CONSOLIDACIÓN                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DURANTE EL DÍA (interacciones)                          │
│     Usuario → Gate decide importancia → Buffer              │
│                                                              │
│  2. CONSOLIDACIÓN (cada N interacciones o manual)           │
│     Buffer → Sample mixto (70% recientes + 30% antiguas)    │
│            → Forward por modelo base (CONGELADO)            │
│            → Ajuste de LoRA adapters (ENTRENABLES)          │
│            → Loss: MSE(predicción, importancia_real)        │
│                                                              │
│  3. RESULTADO                                                │
│     • Modelo aprende TUS patrones de importancia            │
│     • No olvida (modelo base intacto)                       │
│     • Entrenamiento rápido (solo ~8K params)                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
""")

stats = manager.get_stats()
print(f"📊 Estadísticas finales:")
print(f"   • Interacciones procesadas: {len(manager.replay_buffer)}")
print(f"   • Consolidaciones: {len(manager.training_history)}")
print(f"   • Parámetros LoRA: ~8,192")
print(f"   • Device: {manager.device}")

# Guardar estado
manager.replay_buffer.save()
print(f"\n💾 Buffer guardado en: data/replay_buffer.json")

print("\n" + "=" * 70)
print("✅ DEMO COMPLETADA")
print("=" * 70)
