#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 GENERACIÓN DE TEXTO - FASE 2 IIT TRANSFORMER
================================================

Genera texto coherente usando el modelo entrenado con IIT Transformer Layer.
PHI: 8.58 | PPL: 1.12
"""

import sys
import os

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from train_phase2_iit_transformer import InfinitoGPT2IITPhase2


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filtrado Top-K y Top-P (nucleus sampling)."""
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    return logits


def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.2,
    device='cuda'
):
    """
    Genera texto usando el modelo Fase 2.
    
    Args:
        model: Modelo InfinitoGPT2IITPhase2
        tokenizer: GPT2Tokenizer
        prompt: Texto inicial
        max_length: Longitud máxima a generar
        temperature: Temperatura de sampling (0.7-1.0 recomendado)
        top_k: Top-K sampling
        top_p: Nucleus sampling
        repetition_penalty: Penalización por repetición
        device: 'cuda' o 'cpu'
    
    Returns:
        Texto generado
    """
    model.eval()
    
    # Tokenizar prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Historial para penalización de repetición
    generated_tokens = input_ids[0].tolist()
    
    print(f"\n{'='*70}")
    print(f"GENERANDO TEXTO - FASE 2 IIT TRANSFORMER")
    print(f"{'='*70}")
    print(f"Prompt: '{prompt}'")
    print(f"Config: temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}")
    print(f"{'='*70}\n")
    
    with torch.no_grad():
        for step in range(max_length):
            # Forward pass
            logits = model(input_ids, return_metrics=False)
            
            # Siguiente token (último logit)
            next_token_logits = logits[0, -1, :].clone()
            
            # Aplicar penalización por repetición
            for token_id in set(generated_tokens):
                next_token_logits[token_id] /= repetition_penalty
            
            # Aplicar temperatura
            next_token_logits = next_token_logits / temperature
            
            # Aplicar top-k y top-p filtering
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            
            # Sampling
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Añadir token generado
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            generated_tokens.append(next_token.item())
            
            # Mostrar progreso cada 10 tokens
            if (step + 1) % 10 == 0:
                current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"[{step+1}/{max_length}] {current_text[-50:]}")
            
            # Detener si genera EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decodificar texto completo
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(f"\n{'='*70}")
    print(f"GENERACIÓN COMPLETADA ({len(generated_tokens)} tokens)")
    print(f"{'='*70}\n")
    
    return generated_text


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generación de texto - Fase 2')
    parser.add_argument('--prompt', type=str, default='The history of artificial intelligence')
    parser.add_argument('--max-length', type=int, default=150)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--repetition-penalty', type=float, default=1.2)
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/infinito_phase2_best.pt')
    parser.add_argument('--num-samples', type=int, default=3, help='Número de muestras a generar')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar tokenizer
    print(f"\n🔤 Cargando tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"  ✓ Vocabulario: {len(tokenizer):,} tokens")
    
    # Cargar modelo
    print(f"\n📦 Cargando modelo Fase 2...")
    model = InfinitoGPT2IITPhase2(
        num_iit_layers=2,
        lambda_phi=1.0
    ).to(device)
    
    # Cargar checkpoint
    print(f"  📂 Cargando checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Mostrar métricas del checkpoint
    print(f"\n📊 Métricas del checkpoint:")
    print(f"  Época: {checkpoint['epoch']}")
    print(f"  Val PHI: {checkpoint['val_phi']:.4f}")
    print(f"  Val PPL: {checkpoint['val_ppl']:.2f}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    
    # Generar múltiples muestras
    for i in range(args.num_samples):
        print(f"\n{'█'*70}")
        print(f"MUESTRA {i+1}/{args.num_samples}")
        print(f"{'█'*70}")
        
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device
        )
        
        print(f"📝 TEXTO GENERADO:\n")
        print(f"{generated_text}\n")
        print(f"{'█'*70}\n")


if __name__ == '__main__':
    main()
