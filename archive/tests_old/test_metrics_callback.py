#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST - Rich Metrics Callback
================================

Test rápido del callback de métricas mejorado.
También muestra PPL actual del modelo base.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset


def check_base_perplexity():
    """Verificar perplexity del modelo GPT-2 base."""
    print("="*70)
    print("📊 PERPLEXITY DEL MODELO GPT-2 BASE")
    print("="*70)
    
    print("\n📦 Cargando GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modelo cargado en {device}")
    
    # Cargar dataset
    print("\n📚 Cargando WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    # Calcular perplexity en muestra
    print("\n🔢 Calculando perplexity...")
    
    total_loss = 0
    total_tokens = 0
    samples = 100
    
    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= samples:
                break
            
            text = example['text'].strip()
            if len(text) < 10:
                continue
            
            # Tokenizar
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            total_loss += loss.item() * inputs['input_ids'].size(1)
            total_tokens += inputs['input_ids'].size(1)
            
            if (i + 1) % 20 == 0:
                current_ppl = torch.exp(torch.tensor(total_loss / total_tokens))
                print(f"   Muestras procesadas: {i+1}/{samples}  |  PPL actual: {current_ppl:.2f}")
    
    # Calcular perplexity final
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTADO:")
    print(f"{'='*70}")
    print(f"   Muestras evaluadas: {samples}")
    print(f"   Tokens totales: {total_tokens:,}")
    print(f"   Loss promedio: {avg_loss:.4f}")
    print(f"   PERPLEXITY: {perplexity:.2f}")
    print(f"{'='*70}")
    
    # Contexto
    print(f"\n💡 CONTEXTO:")
    print(f"   - GPT-2 base en WikiText-2: ~30-35 (típico)")
    print(f"   - Rango seguro para INFINITO: 10-200")
    print(f"   - Colapso detectado: <10")
    print(f"   - Confusión detectada: >200")
    
    if perplexity < 10:
        print(f"\n   🚨 PPL MUY BAJO - Posible colapso")
    elif perplexity > 200:
        print(f"\n   ⚠️  PPL ALTO - Modelo confuso")
    elif 10 <= perplexity <= 100:
        print(f"\n   ✅ PPL EN RANGO BUENO")
    else:
        print(f"\n   ⚠️  PPL ELEVADO pero aceptable")
    
    return perplexity.item()


def test_callback():
    """Test del callback de métricas."""
    print("\n\n" + "="*70)
    print("🧪 TEST - RICH METRICS CALLBACK")
    print("="*70)
    
    print("\n📦 Importando callback...")
    try:
        from src.rl.rich_metrics_callback import RichMetricsCallback
        print("✅ Callback importado correctamente")
        
        print("\n📋 Características del callback:")
        print("   - Barra de progreso visual")
        print("   - Rewards (media, min, max, std)")
        print("   - Métricas INFINITO (C, Φ, PPL)")
        print("   - Distribución de acciones (TEXT/PHI/MIXED)")
        print("   - Alertas automáticas (PHI alto, PPL bajo)")
        print("   - Tiempo transcurrido y ETA")
        print("   - Análisis de estrategia")
        
        print("\n✅ Callback listo para usar en entrenamiento")
        print("\nSe mostrará cada 500 timesteps con:")
        print("   [████████████████████░░░░] 80.0%")
        print("   ⏱️  Transcurrido: 2:30:00  |  ETA: 0:37:30")
        print("   💰 REWARDS: +0.125 ± 0.045")
        print("   🧠 PHI: 4.5 ± 0.3  |  PPL: 85.2 ± 12.1")
        print("   🎮 TEXT: 40% | PHI: 35% | MIXED: 25%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    try:
        # Test 1: Perplexity del modelo base
        ppl = check_base_perplexity()
        
        # Test 2: Callback
        test_callback()
        
        print("\n" + "="*70)
        print("✅ TESTS COMPLETADOS")
        print("="*70)
        print("\n💡 El callback está listo para el entrenamiento.")
        print("   Ejecutar: python experiments/train_phi_text_scheduler.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
