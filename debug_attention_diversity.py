#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO: ¿Por qué Attention Diversity = 0.5?
====================================================
"""

import torch
import sys
sys.path.insert(0, 'src')

# Cargar modelo
print("="*70)
print("🔍 DIAGNÓSTICO ATTENTION DIVERSITY")
print("="*70)

# Cargar el modelo
checkpoint = torch.load('models/infinito_gpt2_spanish_phi.pt', map_location='cpu', weights_only=False)
print(f"✅ Checkpoint cargado")

# Recrear el modelo para inspección
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType

print(f"\n📦 Cargando GPT-2 Spanish...")
tokenizer = GPT2Tokenizer.from_pretrained('datificate/gpt2-small-spanish', use_safetensors=True)

# Probar con attn_implementation="eager" para forzar atención tradicional
print("   → Probando con attn_implementation='eager' para obtener attention weights...")
try:
    model = GPT2LMHeadModel.from_pretrained(
        'datificate/gpt2-small-spanish', 
        use_safetensors=True,
        attn_implementation="eager"  # Forzar atención tradicional
    )
    print("   ✅ Usando atención 'eager'")
except:
    model = GPT2LMHeadModel.from_pretrained('datificate/gpt2-small-spanish', use_safetensors=True)
    print("   ⚠️ Fallback a atención default")

# Test de atención
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

print(f"\n🔬 Ejecutando forward pass con output_attentions=True...")

test_text = "La inteligencia artificial es una tecnología que"
inputs = tokenizer(test_text, return_tensors='pt').to(device)

with torch.no_grad():
    outputs = model(
        input_ids=inputs.input_ids,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True
    )

print(f"\n📊 ANÁLISIS DE OUTPUTS:")
print(f"   hidden_states: {len(outputs.hidden_states)} capas")
print(f"   attentions: {outputs.attentions}")  # ← ¿Es None?

if outputs.attentions is not None:
    print(f"\n   Tipo: {type(outputs.attentions)}")
    print(f"   Número de capas: {len(outputs.attentions)}")
    
    last_attn = outputs.attentions[-1]
    print(f"\n   Última capa de atención:")
    print(f"   - Shape: {last_attn.shape}")  # [B, heads, T, T]
    print(f"   - Min: {last_attn.min().item():.6f}")
    print(f"   - Max: {last_attn.max().item():.6f}")
    print(f"   - Mean: {last_attn.mean().item():.6f}")
    
    # Calcular diversity manualmente
    print(f"\n🧮 CALCULANDO ATTENTION DIVERSITY MANUALMENTE:")
    
    # attention_weights: [B, heads, T, T]
    attn_flat = last_attn.mean(dim=1)  # [B, T, T]
    print(f"   attn_flat shape: {attn_flat.shape}")
    print(f"   attn_flat sum per row: {attn_flat.sum(dim=-1)}")  # Debería ser ~1
    
    # Entropía de Shannon
    entropy = -torch.sum(attn_flat * torch.log(attn_flat + 1e-10), dim=-1)
    print(f"   entropy shape: {entropy.shape}")
    print(f"   entropy values: {entropy}")
    
    max_entropy = torch.log(torch.tensor(attn_flat.size(-1), dtype=torch.float, device=attn_flat.device))
    print(f"   max_entropy: {max_entropy.item():.4f}")
    
    # Normalizar
    diversity = entropy.mean(dim=-1) / max_entropy
    print(f"\n   🎯 ATTENTION DIVERSITY: {diversity.item():.4f}")
    
    if abs(diversity.item() - 0.5) < 0.01:
        print("\n   ⚠️ ¡ES ~0.5! Investigando por qué...")
        
        # ¿Es porque la atención es uniforme?
        seq_len = attn_flat.size(-1)
        uniform_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float))
        print(f"\n   Si atención fuera uniforme:")
        print(f"   - Entropía uniforme: {uniform_entropy.item():.4f}")
        print(f"   - Diversity sería: {uniform_entropy.item() / max_entropy.item():.4f}")
        
        # ¿Es causal mask el problema?
        print(f"\n   Distribución de atención (primer token):")
        print(f"   {attn_flat[0, 0, :].tolist()}")
        
        print(f"\n   Distribución de atención (último token):")
        print(f"   {attn_flat[0, -1, :].tolist()}")
        
else:
    print("\n   ❌ attentions is None!")
    print("   → Este es el problema. GPT-2 no está devolviendo atenciones.")

print("\n" + "="*70)
print("🔍 DIAGNÓSTICO COMPLETADO")
print("="*70)
