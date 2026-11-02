#!/usr/bin/env python3
"""
🔬 ANÁLISIS EXHAUSTIVO DE PHI - INFINITO
=========================================

Este script:
1. Prueba múltiples prompts de diferentes categorías
2. Mide PHI y sus 4 componentes para cada uno
3. Graba datos detallados para análisis de correlación
4. Compara con datos del entrenamiento

Categorías de prompts:
- Técnico/Científico
- Filosófico/Abstracto
- Casual/Cotidiano
- Creativo/Narrativo
- Instrucciones/Comandos
- Preguntas abiertas vs cerradas
"""

import torch
import sys
import json
import os
from datetime import datetime
from collections import defaultdict
import numpy as np

sys.path.insert(0, 'src')
sys.path.insert(0, 'src/core')

print("="*70)
print("🔬 ANÁLISIS EXHAUSTIVO DE PHI - INFINITO")
print("="*70)

# =============================================================================
# CARGAR MODELO
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Device: {device}")

# Cargar modelo entrenado
from train_gpt2_with_phi_observer import InfinitoGPT2WithObserver

checkpoint_path = 'models/infinito_gpt2_spanish_phi.pt'
print(f"📦 Cargando: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model = InfinitoGPT2WithObserver(use_lora=True)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

tokenizer = model.tokenizer
print("✅ Modelo cargado")

# =============================================================================
# FUNCIÓN DE GENERACIÓN CON MÉTRICAS DETALLADAS
# =============================================================================

def analyze_prompt(prompt, max_length=80, temperature=0.7):
    """Genera texto y captura métricas PHI detalladas."""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        # Generar
        output_ids = model.gpt2.generate(
            input_ids,
            max_length=max_length,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        
        # Obtener hidden states y attention del output completo
        outputs = model.gpt2(
            output_ids,
            output_hidden_states=True,
            output_attentions=True
        )
        
        # Calcular PHI con componentes
        last_hidden = outputs.hidden_states[-1]
        last_attention = outputs.attentions[-1] if outputs.attentions else None
        
        phi_metrics = model.phi_observer(last_hidden, last_attention)
        
        # También obtener PHI por capas (para análisis profundo)
        layer_phis = []
        for i, hidden in enumerate(outputs.hidden_states[1:]):  # Skip embedding layer
            layer_phi = model.phi_observer(hidden, None)
            layer_phis.append({
                'layer': i,
                'phi': layer_phi['phi'].mean().item(),
                'temporal': layer_phi['raw_components']['temporal'],
                'integration': layer_phi['raw_components']['integration'],
            })
    
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return {
        'prompt': prompt,
        'output': text,
        'output_length': len(text),
        'tokens_generated': output_ids.shape[1] - input_ids.shape[1],
        'phi': phi_metrics['phi'].mean().item(),
        'components': phi_metrics['raw_components'],
        'layer_phis': layer_phis,
        'temperature': temperature,
    }


# =============================================================================
# BANCO DE PROMPTS POR CATEGORÍA
# =============================================================================

PROMPTS = {
    'tecnico_cientifico': [
        "La inteligencia artificial funciona mediante",
        "Los algoritmos de machine learning aprenden",
        "La física cuántica describe",
        "El ADN contiene información sobre",
        "Las redes neuronales procesan datos",
        "La fotosíntesis es el proceso por el cual",
        "Los agujeros negros son objetos",
        "La teoría de la relatividad explica",
        "Los semiconductores son materiales que",
        "El cambio climático es causado por",
    ],
    
    'filosofico_abstracto': [
        "El sentido de la vida es",
        "La consciencia humana surge de",
        "La realidad es una construcción",
        "El libre albedrío significa que",
        "La felicidad se encuentra cuando",
        "El tiempo es una dimensión que",
        "La verdad absoluta existe porque",
        "La moralidad depende de",
        "El conocimiento se diferencia de la creencia en",
        "La existencia precede a la esencia porque",
    ],
    
    'casual_cotidiano': [
        "Hoy hace buen tiempo para",
        "Me gusta comer",
        "El fin de semana voy a",
        "Mi película favorita es",
        "Para desayunar prefiero",
        "Cuando estoy cansado suelo",
        "Mi color favorito es el",
        "En mi tiempo libre me gusta",
        "La música que más escucho es",
        "Mi lugar favorito para vacacionar es",
    ],
    
    'creativo_narrativo': [
        "Había una vez un dragón que",
        "En un mundo donde los robots",
        "El detective encontró una pista que",
        "La princesa decidió escapar del castillo porque",
        "En el año 3000, la humanidad",
        "El mago pronunció un hechizo que",
        "Bajo el mar, existía una civilización",
        "El astronauta descubrió un planeta donde",
        "En el bosque encantado vivía",
        "La máquina del tiempo llevó al protagonista a",
    ],
    
    'instrucciones_comandos': [
        "Para cocinar arroz primero debes",
        "Los pasos para aprender a programar son",
        "Instrucciones para armar un mueble:",
        "Cómo resolver un problema matemático:",
        "Para mejorar tu salud deberías",
        "Los requisitos para obtener un trabajo son",
        "Pasos para organizar un evento:",
        "Cómo ahorrar dinero efectivamente:",
        "Para aprender un nuevo idioma necesitas",
        "Instrucciones de seguridad en caso de incendio:",
    ],
    
    'preguntas_abiertas': [
        "¿Qué opinas sobre el futuro de",
        "¿Cómo crees que será el mundo en",
        "¿Por qué existen tantas diferencias entre",
        "¿Cuál es la mejor manera de",
        "¿Qué pasaría si pudiéramos",
        "¿Cómo afecta la tecnología a",
        "¿Por qué es importante estudiar",
        "¿Qué significa realmente ser",
        "¿Cuáles son las consecuencias de",
        "¿Cómo podemos mejorar",
    ],
    
    'preguntas_cerradas': [
        "¿Cuántos planetas hay en el sistema solar?",
        "¿En qué año comenzó la Segunda Guerra Mundial?",
        "¿Cuál es la capital de Francia?",
        "¿Quién escribió Don Quijote?",
        "¿Cuántos lados tiene un hexágono?",
        "¿Cuál es el elemento más abundante en la Tierra?",
        "¿En qué continente está Brasil?",
        "¿Cuál es la fórmula del agua?",
        "¿Quién pintó la Mona Lisa?",
        "¿Cuántos días tiene un año bisiesto?",
    ],
    
    'emocional_personal': [
        "Cuando me siento triste suelo",
        "La alegría se experimenta cuando",
        "El amor es un sentimiento que",
        "Me preocupa el futuro porque",
        "Mis mayores miedos son",
        "Lo que más valoro en la vida es",
        "Cuando estoy estresado necesito",
        "La gratitud significa para mí",
        "Mis sueños y aspiraciones son",
        "Lo que me motiva cada día es",
    ],
}

# =============================================================================
# EJECUTAR ANÁLISIS
# =============================================================================

print("\n" + "="*70)
print("🔬 INICIANDO ANÁLISIS EXHAUSTIVO")
print("="*70)

all_results = []
category_stats = defaultdict(list)

total_prompts = sum(len(prompts) for prompts in PROMPTS.values())
current = 0

for category, prompts in PROMPTS.items():
    print(f"\n📂 Categoría: {category.upper()}")
    print("-"*50)
    
    for prompt in prompts:
        current += 1
        print(f"  [{current}/{total_prompts}] Analizando: '{prompt[:40]}...'")
        
        try:
            result = analyze_prompt(prompt)
            result['category'] = category
            result['timestamp'] = datetime.now().isoformat()
            
            all_results.append(result)
            category_stats[category].append(result['phi'])
            
            print(f"       PHI: {result['phi']:.2f} | Tokens: {result['tokens_generated']}")
            
        except Exception as e:
            print(f"       ❌ Error: {e}")

# =============================================================================
# ESTADÍSTICAS POR CATEGORÍA
# =============================================================================

print("\n" + "="*70)
print("📊 ESTADÍSTICAS POR CATEGORÍA")
print("="*70)

category_summary = {}

for category, phis in sorted(category_stats.items(), key=lambda x: -np.mean(x[1])):
    mean_phi = np.mean(phis)
    std_phi = np.std(phis)
    min_phi = np.min(phis)
    max_phi = np.max(phis)
    
    category_summary[category] = {
        'mean': mean_phi,
        'std': std_phi,
        'min': min_phi,
        'max': max_phi,
        'count': len(phis)
    }
    
    bar = "█" * int(mean_phi)
    print(f"\n{category:25} | PHI: {mean_phi:.2f} ± {std_phi:.2f} | {bar}")
    print(f"{'':25} | Range: [{min_phi:.2f} - {max_phi:.2f}]")

# =============================================================================
# ANÁLISIS DE COMPONENTES
# =============================================================================

print("\n" + "="*70)
print("🔍 ANÁLISIS DE COMPONENTES PHI")
print("="*70)

# Agregar componentes por categoría
component_stats = defaultdict(lambda: defaultdict(list))

for result in all_results:
    cat = result['category']
    for comp, val in result['components'].items():
        component_stats[cat][comp].append(val)

print("\nComponentes promedio por categoría:")
print("-"*70)
print(f"{'Categoría':25} | {'Temporal':10} | {'Integration':12} | {'Complexity':11} | {'Attention':10}")
print("-"*70)

for category in sorted(category_stats.keys()):
    temporal = np.mean(component_stats[category]['temporal'])
    integration = np.mean(component_stats[category]['integration'])
    complexity = np.mean(component_stats[category]['complexity'])
    attention = np.mean(component_stats[category]['attention'])
    
    print(f"{category:25} | {temporal:.4f}     | {integration:.4f}       | {complexity:.4f}      | {attention:.4f}")

# =============================================================================
# ANÁLISIS PHI POR CAPA
# =============================================================================

print("\n" + "="*70)
print("📈 PHI POR CAPA (promedio de todos los prompts)")
print("="*70)

layer_phi_avg = defaultdict(list)
for result in all_results:
    for layer_data in result['layer_phis']:
        layer_phi_avg[layer_data['layer']].append(layer_data['phi'])

print("\nCapa | PHI promedio | Visualización")
print("-"*50)
for layer in sorted(layer_phi_avg.keys()):
    avg = np.mean(layer_phi_avg[layer])
    bar = "█" * int(avg)
    print(f"  {layer:2} | {avg:.2f}         | {bar}")

# =============================================================================
# CORRELACIONES INTERESANTES
# =============================================================================

print("\n" + "="*70)
print("🔗 CORRELACIONES")
print("="*70)

# PHI vs longitud de output
phis = [r['phi'] for r in all_results]
lengths = [r['output_length'] for r in all_results]
tokens = [r['tokens_generated'] for r in all_results]

# Correlación Pearson simple
def pearson_corr(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.corrcoef(x, y)[0, 1]

corr_length = pearson_corr(phis, lengths)
corr_tokens = pearson_corr(phis, tokens)

print(f"\n• PHI vs Longitud output: r = {corr_length:.3f}")
print(f"• PHI vs Tokens generados: r = {corr_tokens:.3f}")

# Componentes vs PHI total
temporal = [r['components']['temporal'] for r in all_results]
integration = [r['components']['integration'] for r in all_results]
complexity = [r['components']['complexity'] for r in all_results]
attention = [r['components']['attention'] for r in all_results]

print(f"\n• Temporal → PHI: r = {pearson_corr(temporal, phis):.3f}")
print(f"• Integration → PHI: r = {pearson_corr(integration, phis):.3f}")
print(f"• Complexity → PHI: r = {pearson_corr(complexity, phis):.3f}")
print(f"• Attention → PHI: r = {pearson_corr(attention, phis):.3f}")

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================

output_data = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'model': 'infinito_gpt2_spanish_phi',
        'total_prompts': len(all_results),
        'categories': list(PROMPTS.keys()),
    },
    'category_summary': category_summary,
    'correlations': {
        'phi_vs_length': corr_length,
        'phi_vs_tokens': corr_tokens,
        'temporal_vs_phi': pearson_corr(temporal, phis),
        'integration_vs_phi': pearson_corr(integration, phis),
        'complexity_vs_phi': pearson_corr(complexity, phis),
        'attention_vs_phi': pearson_corr(attention, phis),
    },
    'layer_phi_averages': {str(k): np.mean(v) for k, v in layer_phi_avg.items()},
    'all_results': all_results,
}

output_path = 'results/phi_exhaustive_analysis.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Resultados guardados en: {output_path}")

# =============================================================================
# EJEMPLOS EXTREMOS
# =============================================================================

print("\n" + "="*70)
print("🏆 TOP 5 PHI MÁS ALTO")
print("="*70)

sorted_by_phi = sorted(all_results, key=lambda x: -x['phi'])[:5]
for i, r in enumerate(sorted_by_phi, 1):
    print(f"\n{i}. PHI: {r['phi']:.2f} | {r['category']}")
    print(f"   Prompt: {r['prompt'][:50]}...")
    print(f"   Output: {r['output'][:80]}...")

print("\n" + "="*70)
print("📉 TOP 5 PHI MÁS BAJO")
print("="*70)

sorted_by_phi = sorted(all_results, key=lambda x: x['phi'])[:5]
for i, r in enumerate(sorted_by_phi, 1):
    print(f"\n{i}. PHI: {r['phi']:.2f} | {r['category']}")
    print(f"   Prompt: {r['prompt'][:50]}...")
    print(f"   Output: {r['output'][:80]}...")

print("\n" + "="*70)
print("🎯 ANÁLISIS COMPLETADO")
print("="*70)
print(f"Total prompts analizados: {len(all_results)}")
print(f"Datos guardados en: {output_path}")
print("="*70)
