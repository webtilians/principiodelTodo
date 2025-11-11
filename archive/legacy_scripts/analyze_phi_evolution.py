#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 ANÁLISIS DE EVOLUCIÓN PHI
============================

Análisis profundo de por qué PHI no mejoró durante el entrenamiento.
Investiga gradientes, componentes IIT individuales, y threshold.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from infinito_v5_2_refactored import InfinitoV52Refactored

def load_history(path):
    """Carga historial de entrenamiento."""
    with open(path, 'r') as f:
        return json.load(f)

def analyze_phi_components(model, tokenizer, device='cuda'):
    """
    Analiza componentes IIT individuales con muestras de validación.
    """
    model.eval()
    
    # Crear muestra de texto de Wikipedia
    test_texts = [
        "The theory of consciousness is a fundamental question in philosophy and neuroscience.",
        "Artificial intelligence systems are becoming increasingly sophisticated.",
        "The human brain contains approximately 86 billion neurons.",
        "Machine learning models can learn complex patterns from data."
    ]
    
    results = []
    
    with torch.no_grad():
        for text in test_texts:
            # Tokenizar
            tokens = tokenizer.encode(text, return_tensors='pt').to(device)
            
            # Forward pass con métricas
            logits, metrics = model(tokens, return_metrics=True)
            
            if metrics:
                results.append({
                    'text': text[:50] + '...',
                    'phi': metrics.get('integration_phi', 0.0),
                    'temporal_coherence': metrics.get('temporal_coherence', 0.0),
                    'integration_strength': metrics.get('integration_strength', 0.0),
                    'complexity': metrics.get('complexity', 0.0),
                    'attention_diversity': metrics.get('attention_diversity', 0.0),
                    'delta_phi_loss': metrics.get('delta_phi_loss', 0.0)
                })
    
    return results

def analyze_phi_weights(model):
    """Analiza pesos aprendibles de componentes PHI."""
    if hasattr(model, 'phi_weights') and model.phi_weights is not None:
        weights = model.phi_weights.get_weights()
        return {
            'temporal_coherence_weight': weights[0].item(),
            'integration_strength_weight': weights[1].item(),
            'complexity_weight': weights[2].item(),
            'attention_diversity_weight': weights[3].item()
        }
    return None

def analyze_threshold(model):
    """Analiza threshold aprendible de memoria."""
    if hasattr(model.memory, 'get_threshold'):
        threshold = model.memory.get_threshold()
        return threshold
    return None

def plot_phi_evolution(history, save_path='results/analysis'):
    """Genera gráficos de evolución PHI."""
    os.makedirs(save_path, exist_ok=True)
    
    epochs = list(range(1, len(history['train_phi']) + 1))
    
    # Gráfico 1: PHI Evolution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_phi'], 'b-', linewidth=2, label='Train PHI')
    plt.axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label='Baseline (0.90)')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('PHI', fontsize=12)
    plt.title('Evolución de PHI durante Entrenamiento', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Delta PHI Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss_phi'], 'g-', linewidth=2, label='ΔPhi Loss')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('ΔPhi Loss', fontsize=12)
    plt.title('Evolución de ΔPhi Loss', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/phi_evolution.png', dpi=150, bbox_inches='tight')
    print(f"📊 Gráfico guardado: {save_path}/phi_evolution.png")
    plt.close()
    
    # Gráfico 3: PPL vs PHI
    plt.figure(figsize=(10, 6))
    plt.scatter(history['val_perplexity'], history['train_phi'], 
                c=epochs, cmap='viridis', s=100, alpha=0.6, edgecolors='k')
    plt.colorbar(label='Época')
    plt.xlabel('Val PPL', fontsize=12)
    plt.ylabel('Train PHI', fontsize=12)
    plt.title('Relación PPL vs PHI', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/ppl_vs_phi.png', dpi=150, bbox_inches='tight')
    print(f"📊 Gráfico guardado: {save_path}/ppl_vs_phi.png")
    plt.close()

def main():
    print("\n" + "="*70)
    print("🔬 ANÁLISIS DE EVOLUCIÓN PHI")
    print("="*70)
    
    # 1. Cargar historial
    print("\n📂 Cargando historial de entrenamiento...")
    history_path = 'results/training/training_history_real_20251111_145331.json'
    history = load_history(history_path)
    
    print(f"  ✓ Épocas totales: {len(history['train_phi'])}")
    print(f"  ✓ PHI inicial: {history['train_phi'][0]:.6f}")
    print(f"  ✓ PHI final: {history['train_phi'][-1]:.6f}")
    print(f"  ✓ Cambio PHI: {history['train_phi'][-1] - history['train_phi'][0]:.6f} ({((history['train_phi'][-1] / history['train_phi'][0]) - 1) * 100:.2f}%)")
    
    # 2. Análisis estadístico
    print("\n📊 Estadísticas PHI:")
    phi_values = np.array(history['train_phi'])
    print(f"  Media: {phi_values.mean():.6f}")
    print(f"  Desv. estándar: {phi_values.std():.6f}")
    print(f"  Mín: {phi_values.min():.6f} (época {phi_values.argmin() + 1})")
    print(f"  Máx: {phi_values.max():.6f} (época {phi_values.argmax() + 1})")
    
    # 3. Análisis de tendencia
    print("\n📈 Tendencia PHI:")
    phi_diff = np.diff(phi_values)
    mejoras = (phi_diff > 0).sum()
    empeoramientos = (phi_diff < 0).sum()
    print(f"  Épocas con mejora: {mejoras}/{len(phi_diff)} ({mejoras/len(phi_diff)*100:.1f}%)")
    print(f"  Épocas con empeoramiento: {empeoramientos}/{len(phi_diff)} ({empeoramientos/len(phi_diff)*100:.1f}%)")
    
    # 4. Análisis ΔPhi Loss
    print("\n🎯 Análisis ΔPhi Loss:")
    delta_phi_loss = np.array(history['train_loss_phi'])
    print(f"  ΔPhi Loss inicial: {delta_phi_loss[0]:.6f}")
    print(f"  ΔPhi Loss final: {delta_phi_loss[-1]:.6f}")
    print(f"  Cambio: {delta_phi_loss[-1] - delta_phi_loss[0]:.6f} ({((delta_phi_loss[-1] / delta_phi_loss[0]) - 1) * 100:.2f}%)")
    print(f"  Varianza: {delta_phi_loss.std():.6f}")
    
    # 5. Cargar mejor modelo
    print("\n🤖 Cargando mejor modelo para análisis detallado...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load('models/checkpoints/infinito_v5.2_real_best.pt', 
                           map_location=device, weights_only=False)
    
    model = InfinitoV52Refactored(
        vocab_size=50257,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        memory_slots=256,
        dropout=0.1,
        use_improved_memory=True,
        use_improved_iit=True,
        use_learnable_phi=True,
        use_stochastic_exploration=True,
        lambda_phi=0.1,
        seed=42
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # 6. Analizar pesos PHI aprendibles
    print("\n⚖️  Pesos PHI Aprendibles:")
    phi_weights = analyze_phi_weights(model)
    if phi_weights:
        for name, weight in phi_weights.items():
            print(f"  {name}: {weight:.6f}")
        
        # Calcular distribución
        total = sum(phi_weights.values())
        print(f"\n  Distribución normalizada:")
        for name, weight in phi_weights.items():
            pct = (weight / total) * 100
            print(f"  {name}: {pct:.2f}%")
    else:
        print("  ⚠️  No se encontraron pesos aprendibles")
    
    # 7. Analizar threshold
    print("\n🎯 Threshold de Memoria:")
    threshold = analyze_threshold(model)
    if threshold is not None:
        print(f"  Threshold actual: {threshold:.6f}")
        print(f"  ⚠️  PROBLEMA: Threshold se mantuvo constante en 3.0 durante todo el entrenamiento")
    else:
        print("  ⚠️  No se pudo obtener threshold")
    
    # 8. Analizar componentes IIT en muestras
    print("\n🧠 Análisis de Componentes IIT en Muestras:")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    components_results = analyze_phi_components(model, tokenizer, device)
    
    for i, result in enumerate(components_results, 1):
        print(f"\n  Muestra {i}: {result['text']}")
        print(f"    PHI: {result['phi']:.4f}")
        print(f"    └─ Temporal Coherence:    {result['temporal_coherence']:.4f}")
        print(f"    └─ Integration Strength:  {result['integration_strength']:.4f}")
        print(f"    └─ Complexity:            {result['complexity']:.4f}")
        print(f"    └─ Attention Diversity:   {result['attention_diversity']:.4f}")
        print(f"    ΔPhi Loss: {result['delta_phi_loss']:.6f}")
    
    # 9. Generar gráficos
    print("\n📊 Generando visualizaciones...")
    plot_phi_evolution(history)
    
    # 10. Diagnóstico final
    print("\n" + "="*70)
    print("🔍 DIAGNÓSTICO FINAL")
    print("="*70)
    
    print("\n❌ PROBLEMAS IDENTIFICADOS:")
    
    # Problema 1: PHI no mejoró
    if abs(history['train_phi'][-1] - history['train_phi'][0]) < 0.01:
        print("\n1. PHI ESTANCADO:")
        print(f"   - Cambio total: {history['train_phi'][-1] - history['train_phi'][0]:.6f} (<1%)")
        print("   - PHI no está siendo optimizado efectivamente")
    
    # Problema 2: ΔPhi Loss no cambió
    if abs(delta_phi_loss[-1] - delta_phi_loss[0]) < 0.01:
        print("\n2. ΔPhi LOSS CONSTANTE:")
        print(f"   - Cambio total: {delta_phi_loss[-1] - delta_phi_loss[0]:.6f}")
        print("   - Loss auxiliar no tiene impacto")
        print(f"   - Lambda PHI = 0.1 es demasiado bajo (solo 10% del loss total)")
    
    # Problema 3: Threshold no aprendió
    if threshold is not None and abs(threshold - 3.0) < 0.01:
        print("\n3. THRESHOLD CONGELADO:")
        print(f"   - Valor: {threshold:.6f} (constante en 3.0)")
        print("   - Gradientes no fluyen al threshold")
        print("   - Memoria IIT no se adapta")
    
    print("\n✅ RECOMENDACIONES:")
    print("\n1. Aumentar lambda_phi:")
    print("   --lambda-phi 0.5  (5x más influencia)")
    
    print("\n2. Pre-entrenar con foco en PHI:")
    print("   - 5 épocas solo optimizando ΔPhi")
    print("   - Luego fine-tuning con LM + PHI")
    
    print("\n3. Usar modelo pre-entrenado (GPT-2):")
    print("   - Baseline con lenguaje funcional")
    print("   - Comparar PHI antes/después de IIT")
    print("   - Experimento más controlado")
    
    print("\n4. Investigar gradientes:")
    print("   - Verificar que threshold recibe gradientes")
    print("   - Ajustar learning rates específicos por componente")
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)

if __name__ == '__main__':
    main()
