#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 VALIDACIÓN DEL MEJOR MODELO ENTRENADO
=========================================

Genera texto con el checkpoint de época 9 (Val PPL 184.13)
para evaluar calidad real vs métricas.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from transformers import GPT2Tokenizer
from infinito_v5_2_refactored import InfinitoV52Refactored

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=40):
    """Genera texto con top-k sampling."""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenizar prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(generated)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Obtener logits del último token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k filtering
            top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
            next_token_logits_filtered = torch.full_like(next_token_logits, float('-inf'))
            next_token_logits_filtered[top_k_indices] = top_k_values
            
            # Softmax + sample
            probs = torch.softmax(next_token_logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Agregar token generado
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop si genera <eos>
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def main():
    print("=" * 70)
    print("   VALIDACIÓN MODELO MEJOR (Época 9 - Val PPL 184.13)")
    print("=" * 70)
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Cargar tokenizer
    print("\n🔤 Cargando GPT2Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"  ✓ Vocabulario: {len(tokenizer):,} tokens")
    
    # Crear modelo
    print("\n🤖 Creando modelo INFINITO V5.2...")
    model = InfinitoV52Refactored(
        vocab_size=len(tokenizer),
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        memory_slots=256,
        use_improved_memory=True,
        use_improved_iit=True,
        use_learnable_phi=True,
        use_stochastic_exploration=True,
        lambda_phi=0.1,
        seed=42
    ).to(device)
    
    print(f"  ✓ Parámetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Cargar checkpoint
    checkpoint_path = 'models/checkpoints/infinito_v5.2_real_best.pt'
    print(f"\n💾 Cargando checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  ✓ Checkpoint cargado exitosamente")
        print(f"  ✓ Época: {checkpoint['epoch']}")
        print(f"  ✓ Val Loss: {checkpoint['val_loss']:.4f}")
        print(f"  ✓ Val PPL: {checkpoint['val_ppl']:.2f}")
    except Exception as e:
        print(f"  ❌ Error cargando checkpoint: {e}")
        return
    
    print("\n" + "=" * 70)
    print("   GENERACIÓN DE TEXTO - PRUEBAS")
    print("=" * 70)
    
    prompts = [
        "The history of artificial intelligence",
        "In the future, technology will",
        "Once upon a time, there was",
        "The main difference between"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─' * 70}")
        print(f"PRUEBA {i}/4")
        print(f"{'─' * 70}")
        print(f"\n📝 Prompt: '{prompt}'")
        print(f"🎲 Temperatura: 0.8 | Top-k: 40")
        print()
        
        generated = generate_text(
            model, tokenizer, prompt,
            max_length=80,
            temperature=0.8,
            top_k=40
        )
        
        print(f"💬 Texto generado:")
        print(f"   {generated}")
        print()
    
    print("=" * 70)
    print("   VALIDACIÓN COMPLETADA")
    print("=" * 70)


if __name__ == '__main__':
    main()
