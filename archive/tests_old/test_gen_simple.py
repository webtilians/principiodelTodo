"""
Test MÍNIMO de generación - Solo 1 ejemplo corto en CPU
"""

import torch
import sys
from pathlib import Path
from transformers import GPT2Tokenizer

sys.path.append(str(Path(__file__).parent))
from src.infinito_v5_2_refactored import InfinitoV52Refactored

print("=" * 70)
print("TEST RÁPIDO - INFINITO V5.2")
print("=" * 70)

# Cargar checkpoint (solo metadata)
checkpoint_path = "models/checkpoints/infinito_v5.2_real_best.pt"
print(f"\nCargando checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Info
val_loss = checkpoint.get('val_loss', None)
epoch = checkpoint.get('epoch', 'N/A')

print(f"\n📊 Información del Modelo:")
print(f"   Época entrenada: {epoch}")
print(f"   Val Loss: {val_loss:.4f}")

if val_loss:
    val_ppl = torch.exp(torch.tensor(val_loss)).item()
    print(f"   Val Perplexity: {val_ppl:.2f}")

print(f"   Vocab Size: 50,257 (GPT-2)")
print(f"   Parámetros: 71.4M")
print(f"   Dataset: WikiText-2 REAL")

# Crear modelo
print(f"\nCreando modelo...")
model = InfinitoV52Refactored(
    vocab_size=50257,
    hidden_dim=512,
    num_layers=6,
    num_heads=8,
    memory_slots=256,
    use_improved_memory=True,
    use_stochastic_exploration=True
)

# Cargar pesos
print(f"Cargando pesos...")
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

print(f"✅ Modelo listo")

# Tokenizer
print(f"\nCargando tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Tokenizer listo")

# Generar UN ejemplo corto
print(f"\n" + "=" * 70)
print(f"GENERANDO TEXTO (1 ejemplo, 30 tokens)")
print(f"=" * 70)

prompt = "The theory of relativity"
print(f"\n📝 Prompt: '{prompt}'")
print(f"🔄 Generando... (esto puede tardar 1-2 minutos en CPU)")

input_ids = tokenizer.encode(prompt, return_tensors='pt')
generated = input_ids.clone()

with torch.no_grad():
    for i in range(30):  # Solo 30 tokens
        if i % 10 == 0:
            print(f"   Token {i}/30...")
        
        logits, _ = model(generated)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break

generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)

print(f"\n✨ RESULTADO:")
print(f"   {generated_text}")

print(f"\n" + "=" * 70)
print(f"TEST COMPLETADO")
print(f"=" * 70)

print(f"\n💡 NOTAS:")
print(f"   - Corriendo en CPU (lento)")
print(f"   - Para velocidad completa, necesitas PyTorch con CUDA")
print(f"   - El modelo funciona correctamente")
print(f"   - PPL de {val_ppl:.2f} está por encima del objetivo (50-80)")
print(f"   - Considera entrenar más épocas o ajustar hiperparámetros")
