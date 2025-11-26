#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 TEST CREATIVO: INFINITO V5.2
===============================

Test de creatividad y coherencia con prompts más interesantes
para el modelo optimizado.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from transformers import GPT2Tokenizer
from infinito_v5_2_refactored import InfinitoV52Refactored

def load_best_model():
    """Carga el mejor modelo optimizado."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cargar tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Cargar modelo
    checkpoint_path = "models/checkpoints/infinito_v5.2_real_best.pt"
    
    print(f"🤖 Cargando modelo optimizado...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = InfinitoV52Refactored(
        vocab_size=50257,
        hidden_dim=384,
        num_layers=3,
        num_heads=6,
        memory_slots=256,
        dropout=0.0,
        use_improved_memory=True,
        use_improved_iit=True,
        use_learnable_phi=True,
        use_stochastic_exploration=True,
        lambda_phi=0.1,
        seed=42
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"  ✓ Modelo cargado (Época {checkpoint.get('epoch', '?')}, PPL: {checkpoint.get('val_ppl', '?'):.1f})")
    
    return model, tokenizer, device

def generate_creative_text(model, tokenizer, device, prompt, max_length=150, temperature=0.9, top_p=0.85):
    """Genera texto con parámetros optimizados para creatividad."""
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            next_token_logits = logits[0, -1, :] / temperature
            
            # Nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            if generated_ids.size(1) > 1024:
                generated_ids = generated_ids[:, -512:]
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text[len(prompt):].strip()

def main():
    """Test creativo principal."""
    
    print("🎨 TEST CREATIVO - INFINITO V5.2 OPTIMIZADO")
    print("=" * 60)
    
    model, tokenizer, device = load_best_model()
    
    # Prompts más creativos y específicos
    creative_prompts = [
        # Ciencia ficción
        "The last human on Mars discovered",
        "When robots learned to dream, they",
        "The quantum computer gained consciousness and",
        
        # Filosofía
        "The meaning of existence becomes clear when",
        "If time could flow backwards, we would",
        "The boundary between reality and simulation",
        
        # Narrativa
        "In a world where books come alive",
        "The old library contained a secret door that led to",
        "She opened the letter from her future self, which said",
        
        # Conocimiento
        "The greatest scientific breakthrough occurred when",
        "Einstein's final equation revealed that",
        "The universe's biggest secret is hidden in"
    ]
    
    print(f"🎯 Ejecutando {len(creative_prompts)} tests creativos...")
    print()
    
    for i, prompt in enumerate(creative_prompts, 1):
        print(f"{'='*60}")
        print(f"🎨 TEST {i}/12: \"{prompt}\"")
        print(f"{'='*60}")
        
        # Generar 3 variaciones con diferentes temperaturas
        temperatures = [0.7, 0.9, 1.1]
        
        for j, temp in enumerate(temperatures, 1):
            print(f"\n🌡️ Variación {j} (temperatura {temp}):")
            
            try:
                generated = generate_creative_text(
                    model, tokenizer, device, prompt,
                    max_length=100,
                    temperature=temp,
                    top_p=0.85
                )
                
                print(f"  📝 \"{generated}\"")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print()
    
    print("🏆 TEST CREATIVO COMPLETADO")
    print("=" * 60)
    
    print("\n💡 OBSERVACIONES:")
    print("• Temperatura más alta = más creatividad pero menos coherencia")
    print("• Temperatura más baja = más coherente pero menos original")
    print("• El modelo muestra patrones de WikiText-2 (fechas, lugares, nombres)")
    print("• IIT features ayudan a mantener coherencia interna")

if __name__ == '__main__':
    main()