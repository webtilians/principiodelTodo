#!/usr/bin/env python3
"""
🧪 Test de Generación con PHI Observer
========================================

Prueba el modelo entrenado con configuración anti-repetición.
"""

import torch
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/core')

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import json

print("="*60)
print("🧪 TEST DE GENERACIÓN - INFINITO GPT-2 + PHI")
print("="*60)

# Cargar modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Device: {device}")

# Cargar checkpoint
checkpoint_path = 'models/infinito_gpt2_phi_observer.pt'
print(f"📦 Cargando checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
print(f"   Epochs entrenados: {checkpoint.get('epochs', 'N/A')}")

# Reconstruir modelo
print("🔧 Reconstruyendo modelo...")
from train_gpt2_with_phi_observer import InfinitoGPT2WithObserver

model = InfinitoGPT2WithObserver(use_lora=True)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

tokenizer = model.tokenizer

print("✅ Modelo cargado")

# =============================================================================
# GENERACIÓN CON ANTI-REPETICIÓN
# =============================================================================

def generate_with_settings(prompt, max_length=100, **kwargs):
    """Genera texto con configuración robusta."""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Configuración por defecto con anti-repetición
    generation_config = {
        'max_length': max_length,
        'attention_mask': attention_mask,
        'pad_token_id': tokenizer.eos_token_id,
        'do_sample': True,
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 50,
        'repetition_penalty': 1.2,      # Penalizar repeticiones
        'no_repeat_ngram_size': 3,       # No repetir 3-grams
        'early_stopping': True,
    }
    
    # Sobrescribir con kwargs
    generation_config.update(kwargs)
    
    with torch.no_grad():
        output_ids = model.gpt2.generate(input_ids, **generation_config)
    
    # Medir PHI
    outputs = model.gpt2(
        output_ids,
        output_hidden_states=True,
        output_attentions=True
    )
    
    phi_metrics = model.phi_observer(
        outputs.hidden_states[-1],
        outputs.attentions[-1] if outputs.attentions else None
    )
    
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    phi = phi_metrics['phi'].mean().item()
    
    return {
        'text': text,
        'phi': phi,
        'components': phi_metrics['raw_components']
    }


# =============================================================================
# TESTS
# =============================================================================

print("\n" + "="*60)
print("🔮 GENERACIÓN CON CONFIGURACIÓN ANTI-REPETICIÓN")
print("="*60)

prompts = [
    "La inteligencia artificial es",
    "El futuro de la humanidad será",
    "Explica qué es la física cuántica:",
    "Los beneficios de la tecnología incluyen",
    "En el año 2050, el mundo"
]

results = []

for prompt in prompts:
    print(f"\n{'─'*50}")
    print(f"📝 Prompt: '{prompt}'")
    print('─'*50)
    
    result = generate_with_settings(prompt, max_length=80)
    results.append({'prompt': prompt, **result})
    
    print(f"📊 PHI: {result['phi']:.2f}")
    print(f"📄 Output:\n{result['text']}")

# =============================================================================
# COMPARACIÓN: DIFERENTES TEMPERATURAS
# =============================================================================

print("\n" + "="*60)
print("🌡️ COMPARACIÓN DE TEMPERATURAS")
print("="*60)

test_prompt = "La ciencia es importante porque"

for temp in [0.5, 0.7, 0.9, 1.2]:
    print(f"\n{'─'*50}")
    print(f"🌡️ Temperature: {temp}")
    
    result = generate_with_settings(
        test_prompt, 
        max_length=60,
        temperature=temp,
        do_sample=True
    )
    
    print(f"📊 PHI: {result['phi']:.2f}")
    print(f"📄 {result['text']}")

# =============================================================================
# ANÁLISIS DE PHI POR TIPO DE PROMPT
# =============================================================================

print("\n" + "="*60)
print("📈 ANÁLISIS PHI POR TIPO DE PROMPT")
print("="*60)

# Categorizar prompts
categories = {
    'Técnico': [
        "La programación en Python permite",
        "Los algoritmos de machine learning",
        "La arquitectura de redes neuronales"
    ],
    'Filosófico': [
        "El sentido de la vida es",
        "La consciencia humana surge de",
        "La realidad es una construcción"
    ],
    'Casual': [
        "Hola, cómo estás",
        "Qué hora es",
        "Me gusta el"
    ]
}

phi_by_category = {}

for category, prompts in categories.items():
    phis = []
    print(f"\n📂 {category}:")
    
    for prompt in prompts:
        result = generate_with_settings(prompt, max_length=50)
        phis.append(result['phi'])
        print(f"   • PHI: {result['phi']:.2f} | '{prompt[:30]}...'")
    
    avg_phi = sum(phis) / len(phis)
    phi_by_category[category] = avg_phi
    print(f"   📊 Promedio: {avg_phi:.2f}")

print("\n" + "="*60)
print("📊 RESUMEN PHI POR CATEGORÍA")
print("="*60)
for cat, phi in sorted(phi_by_category.items(), key=lambda x: -x[1]):
    bar = "█" * int(phi * 2)
    print(f"   {cat:12} | PHI: {phi:.2f} | {bar}")

# Guardar resultados
with open('results/generation_analysis.json', 'w', encoding='utf-8') as f:
    json.dump({
        'results': results,
        'phi_by_category': phi_by_category
    }, f, indent=2, ensure_ascii=False)

print(f"\n✅ Análisis guardado en: results/generation_analysis.json")
print("="*60)
