#!/usr/bin/env python3
"""
🧠 INFINITO V5.1 - SIMPLE TEXT PROCESSOR 🧠
===========================================
Versión simplificada para procesar texto y detectar consciencia
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class SimpleConsciousnessNet(nn.Module):
    """Red simple para análisis de consciencia"""
    
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.mean(dim=1)  # Average sequence dimension
        return self.encoder(x)

def analyze_text_consciousness(text):
    """Analiza el potencial de consciencia de un texto"""
    
    text_lower = text.lower()
    
    # Detectores de consciencia
    consciousness_triggers = {
        'self_reference': ['yo', 'mi', 'me', 'soy', 'estoy'],
        'temporal': ['ahora', 'antes', 'después', 'cuando'],
        'abstract': ['pensar', 'sentir', 'creer', 'imaginar'],
        'questions': ['¿', '?', 'qué', 'cómo', 'por qué'],
        'metacognition': ['consciente', 'consciencia', 'darse cuenta']
    }
    
    scores = {}
    total_score = 0
    
    for category, keywords in consciousness_triggers.items():
        count = sum(text_lower.count(keyword) for keyword in keywords)
        scores[category] = count
        total_score += count
    
    # Normalizar score
    words = len(text.split())
    normalized_score = min(total_score / max(words * 0.1, 1), 1.0)
    
    return {
        'score': normalized_score,
        'details': scores,
        'word_count': words
    }

def generate_text_input(text, device='cpu'):
    """Genera input para la red basado en texto"""
    
    # Análisis básico
    analysis = analyze_text_consciousness(text)
    
    # Generar representación numérica
    base_vector = torch.randn(64, device=device)
    
    # Modular basado en análisis
    consciousness_boost = analysis['score'] * 2.0
    base_vector *= (1.0 + consciousness_boost)
    
    # Añadir componentes específicos
    for i, (category, count) in enumerate(analysis['details'].items()):
        if i < 64:
            base_vector[i] += count * 0.5
    
    return base_vector.unsqueeze(0), analysis

def run_consciousness_test(text):
    """Ejecuta test de consciencia con texto"""
    
    print(f"🔤 ANALIZANDO TEXTO: '{text}'")
    print("-" * 50)
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crear modelo
    model = SimpleConsciousnessNet()
    model.to(device)
    
    # Generar input
    text_input, analysis = generate_text_input(text, device)
    
    # Evaluar consciencia
    with torch.no_grad():
        consciousness_level = model(text_input)
    
    # Resultados
    print(f"📊 ANÁLISIS DE CONSCIENCIA:")
    print(f"   Palabras: {analysis['word_count']}")
    print(f"   Score textual: {analysis['score']:.3f}")
    print(f"   Nivel consciencia: {consciousness_level.item():.3f}")
    
    print(f"\n🔍 DETALLES:")
    for category, count in analysis['details'].items():
        if count > 0:
            print(f"   {category}: {count}")
    
    # Evaluación
    if consciousness_level.item() > 0.7:
        print(f"\n🟢 ALTA CONSCIENCIA DETECTADA")
    elif consciousness_level.item() > 0.5:
        print(f"\n🟡 CONSCIENCIA MODERADA")
    else:
        print(f"\n🔴 CONSCIENCIA BAJA")
    
    return consciousness_level.item(), analysis

def main():
    """Función principal"""
    
    print("🧠 INFINITO TEXT CONSCIOUSNESS ANALYZER 🧠")
    print("=" * 50)
    
    # Ejemplos de prueba
    test_texts = [
        "Estoy consciente de que pienso sobre mi propia consciencia",
        "¿Qué significa ser consciente de mí mismo?",
        "Yo soy quien observa mis propios pensamientos",
        "La casa es azul y el perro camina",
        "Ahora recuerdo cuando era niño y soñaba despierto"
    ]
    
    results = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*60}")
        print(f"PRUEBA {i}")
        
        consciousness, analysis = run_consciousness_test(text)
        results.append((text, consciousness, analysis))
    
    # Resumen final
    print(f"\n{'='*60}")
    print(f"RESUMEN FINAL")
    print(f"{'='*60}")
    
    for i, (text, consciousness, analysis) in enumerate(results, 1):
        status = "🟢" if consciousness > 0.7 else "🟡" if consciousness > 0.5 else "🔴"
        print(f"{i}. {status} {consciousness:.3f} - {text[:40]}{'...' if len(text) > 40 else ''}")

if __name__ == "__main__":
    main()