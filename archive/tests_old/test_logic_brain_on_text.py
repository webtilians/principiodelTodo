#!/usr/bin/env python3
"""
🧪 TEST DE SENSIBILIDAD DEL GATE (Lógica -> Texto)
==================================================

Este script prueba cómo reacciona el Memory Gate del modelo
cuando se le presenta texto humano vs estructuras lógicas.

Ahora con soporte para DYNAMIC GATE que reacciona al contenido!
"""

import torch
import torch.nn as nn
import sys
import os

# Importamos tu clase
sys.path.insert(0, 'src')
from infinito_v5_2_refactored import InfinitoV52Refactored

# --- CONFIGURACIÓN ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_DYNAMIC_GATE = True  # 🆕 Probar el nuevo Dynamic Gate

# --- 1. EL TRADUCTOR (HACK) ---
vocab = {'PAD': 0, '(': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6, 
         '<': 7, '>': 8, 'A': 9, 'B': 10, 'C': 11, 'EOS': 12}

def hacky_tokenizer(text):
    ids = []
    for char in text:
        if char in vocab:
            ids.append(vocab[char])
        elif char == ' ':
            continue
        else:
            noise_idx = (ord(char) % 3) + 9
            ids.append(noise_idx)
    ids.append(vocab['EOS'])
    return torch.tensor([ids]).to(DEVICE)

# --- 2. CREAR MODELO CON DYNAMIC GATE ---
print(f"\n{'='*60}")
print(f"🧪 CREANDO MODELO {'DINÁMICO' if USE_DYNAMIC_GATE else 'ESTÁTICO'}")
print(f"{'='*60}")

model = InfinitoV52Refactored(
    vocab_size=13,
    hidden_dim=64,
    num_layers=2,
    num_heads=4,
    use_improved_memory=True,
    use_improved_iit=True,
    use_learnable_phi=True,
    use_stochastic_exploration=False,  # Apagado para inferencia
    use_dynamic_gate=USE_DYNAMIC_GATE  # 🆕 El nuevo gate dinámico
).to(DEVICE)

model.eval()

# Mostrar estado inicial del gate
if USE_DYNAMIC_GATE:
    gate_bias = model.dynamic_gate.gate_network[2].bias.item()
    print(f"✅ Dynamic Gate creado. Bias inicial: {gate_bias:.4f}")
    print(f"   Apertura inicial: {torch.sigmoid(torch.tensor(gate_bias)).item()*100:.4f}%")
else:
    print(f"✅ Static Gate. Valor: {model.memory_gate.item():.4f}")

# --- 3. EL EXPERIMENTO ---
test_phrases = [
    # CASO 1: Texto plano (puro ruido para el modelo)
    "Hola mundo esto es una prueba",
    
    # CASO 2: Texto con estructura simple
    "El usuario (Enrique) es el creador",
    
    # CASO 3: Estructura lógica pura (esto SÍ entiende)
    "{ [ A B C ] }",
    
    # CASO 4: Estructura anidada perfecta
    "( [ { < A > } ] )",
    
    # CASO 5: Estructura rota/caótica
    "Esto ) es un ( desastre ] lógico",
    
    # CASO 6: Estructura profunda anidada con texto
    "La { casa [ roja ( tiene ) ventanas ] grandes } fin",
    
    # CASO 7: Solo ruido (letras)
    "ABCABCABCABC",
    
    # CASO 8: Solo estructura (sin ruido)
    "((([[[{{{<><><>}}}]]]))))"
]

print(f"\n{'='*60}")
print(f"🧪 TEST: ¿Reacciona el gate según el contenido?")
print(f"{'='*60}")
print(f"Objetivo: Ver si el gate VARÍA entre frases diferentes")
print("-" * 60)

inv_vocab = {v: k for k, v in vocab.items()}
results = []

for text in test_phrases:
    input_tensor = hacky_tokenizer(text)
    
    with torch.no_grad():
        logits, metrics = model(input_tensor, return_metrics=True)
        
        # Obtener valor del gate para ESTE input específico
        if USE_DYNAMIC_GATE:
            # El dynamic gate calcula un valor por cada input
            hidden = model.token_embedding(input_tensor)
            hidden = hidden + model.position_embedding[:, :input_tensor.size(1), :]
            gate_prob, gate_logit = model.dynamic_gate(hidden)
            gate_value = gate_logit.item()
            gate_pct = gate_prob.item() * 100
        else:
            gate_value = model.memory_gate.item()
            gate_pct = torch.sigmoid(model.memory_gate).item() * 100
    
    # Ver qué "leyó" el modelo
    input_ids = input_tensor[0].tolist()
    decoded = [inv_vocab.get(idx, '?') for idx in input_ids[:-1]]
    decoded_str = "".join(decoded)
    
    # Contar símbolos lógicos vs ruido
    logical_chars = sum(1 for c in decoded_str if c in '()[]{}<>')
    noise_chars = sum(1 for c in decoded_str if c in 'ABC')
    logic_ratio = logical_chars / max(1, len(decoded_str)) * 100
    
    results.append({
        'text': text,
        'decoded': decoded_str,
        'gate_value': gate_value,
        'gate_pct': gate_pct,
        'logic_ratio': logic_ratio
    })
    
    print(f"\n📝 '{text[:40]}...' " if len(text) > 40 else f"\n📝 '{text}'")
    print(f"👀 Modelo ve: {decoded_str[:50]}{'...' if len(decoded_str) > 50 else ''}")
    print(f"📊 Lógica: {logic_ratio:.1f}% | Ruido: {100-logic_ratio:.1f}%")
    print(f"🧠 Gate Logit: {gate_value:.4f}")
    print(f"🚪 Apertura: {gate_pct:.4f}%")
    
    if gate_pct > 1.0:
        print("🚨 GATE ABIERTO: El modelo considera esto importante!")
    elif gate_pct > 0.8:
        print("⚠️ REACCIÓN: Algo llamó su atención.")
    else:
        print("💤 GATE CERRADO: Considerado ruido/ignorable.")

# --- ANÁLISIS FINAL ---
print(f"\n{'='*60}")
print("📊 ANÁLISIS DE VARIABILIDAD DEL GATE")
print(f"{'='*60}")

gate_values = [r['gate_pct'] for r in results]
min_gate = min(gate_values)
max_gate = max(gate_values)
variance = max_gate - min_gate

print(f"\n   Gate mínimo: {min_gate:.4f}%")
print(f"   Gate máximo: {max_gate:.4f}%")
print(f"   Varianza: {variance:.4f}%")

if USE_DYNAMIC_GATE:
    if variance > 0.1:
        print(f"\n✅ ¡ÉXITO! El Dynamic Gate REACCIONA al contenido")
        print(f"   Diferencia de {variance:.4f}% entre inputs")
    else:
        print(f"\n⚠️ El gate varía poco. Puede necesitar entrenamiento.")
        print(f"   (Recuerda: está inicializado cerrado, aprenderá con datos)")
else:
    print(f"\n📌 Gate estático: Mismo valor para todos los inputs")
    print(f"   Para gate reactivo, usa use_dynamic_gate=True")

print(f"\n{'='*60}")
