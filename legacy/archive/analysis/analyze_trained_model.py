#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 ANÁLISIS DEL MODELO ENTRENADO - INFINITO V5.2
================================================

Análisis completo del modelo entrenado:
1. Generación de texto con diferentes prompts
2. Métricas IIT (PHI, integración)
3. Características avanzadas (IITGuidedMemory, LearnablePhiWeights)
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
import numpy as np
from transformers import GPT2Tokenizer
from infinito_v5_2_refactored import InfinitoV52Refactored


def print_section(title):
    """Imprime una sección con formato."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def analyze_model_architecture(model, checkpoint):
    """Analiza la arquitectura del modelo."""
    print_section("📐 ARQUITECTURA DEL MODELO")
    
    config = checkpoint.get('config', {})
    
    print(f"Configuración del Modelo:")
    vocab_size = config.get('vocab_size', None)
    print(f"  Vocab size:      {vocab_size:,}" if vocab_size else "  Vocab size:      N/A")
    print(f"  Hidden dim:      {config.get('hidden_dim', 'N/A')}")
    print(f"  Num layers:      {config.get('num_layers', 'N/A')}")
    print(f"  Num heads:       {config.get('num_heads', 'N/A')}")
    print(f"  Memory slots:    {config.get('memory_slots', 'N/A')}")
    print(f"  Dropout:         {config.get('dropout', 'N/A')}")
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParámetros:")
    print(f"  Total:           {total_params:,}")
    print(f"  Entrenables:     {trainable_params:,}")
    print(f"  No entrenables:  {total_params - trainable_params:,}")
    
    # Características especiales
    print(f"\nCaracterísticas Avanzadas:")
    print(f"  IIT Guided Memory:           {hasattr(model, 'memory') and hasattr(model.memory, 'phi_threshold')}")
    print(f"  Improved IIT Metrics:        {hasattr(model, 'iit_metrics')}")
    print(f"  Learnable Phi Weights:       {hasattr(model, 'learnable_phi_weights')}")
    print(f"  Stochastic Exploration:      {hasattr(model, 'exploration_noise')}")
    
    # Verificar si los pesos aprendibles cambiaron
    if hasattr(model, 'learnable_phi_weights'):
        try:
            weights_dict = model.learnable_phi_weights()
            print(f"\n  Learnable Phi Weights:")
            for key, value in weights_dict.items():
                print(f"    {key:15s}: {value.item():.4f}")
            total = sum(v.item() for v in weights_dict.values())
            print(f"    {'Total':15s}: {total:.4f} (debe ser ~1.0)")
        except Exception as e:
            print(f"\n  Learnable Phi Weights: No disponible ({e})")


def analyze_training_results(checkpoint):
    """Analiza los resultados del entrenamiento."""
    print_section("📊 RESULTADOS DEL ENTRENAMIENTO")
    
    print(f"Checkpoint Info:")
    print(f"  Época:           {checkpoint.get('epoch', 'N/A')}")
    print(f"  Train Loss:      {checkpoint.get('train_loss', 'N/A'):.4f}" if 'train_loss' in checkpoint else "  Train Loss:      N/A")
    print(f"  Val Loss:        {checkpoint.get('val_loss', 'N/A'):.4f}" if 'val_loss' in checkpoint else "  Val Loss:        N/A")
    print(f"  Train PPL:       {checkpoint.get('train_ppl', 'N/A'):.2f}" if 'train_ppl' in checkpoint else "  Train PPL:       N/A")
    print(f"  Val PPL:         {checkpoint.get('val_ppl', 'N/A'):.2f}" if 'val_ppl' in checkpoint else "  Val PPL:         N/A")
    print(f"  Learning Rate:   {checkpoint.get('learning_rate', 'N/A'):.6f}" if 'learning_rate' in checkpoint else "  Learning Rate:   N/A")
    
    # Métricas IIT
    if 'train_phi' in checkpoint:
        print(f"\nMétricas IIT:")
        print(f"  PHI (train):     {checkpoint['train_phi']:.4f}")
        print(f"  ΔPhi Loss:       {checkpoint.get('delta_phi_loss', 'N/A'):.4f}" if 'delta_phi_loss' in checkpoint else "")
        print(f"  Memory Threshold: {checkpoint.get('memory_threshold', 'N/A'):.4f}" if 'memory_threshold' in checkpoint else "")


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9, device='cuda'):
    """Genera texto a partir de un prompt."""
    model.eval()
    
    # Tokenizar prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(generated)
            
            # Obtener logits del último token
            # El modelo puede retornar tupla o dict
            if isinstance(outputs, dict):
                logits = outputs['logits'][:, -1, :] / temperature
            else:
                # Asumir que el primer elemento es logits
                logits = outputs[0][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop si es EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def analyze_iit_metrics(model, tokenizer, test_prompts, device='cuda'):
    """Analiza métricas IIT durante la generación."""
    print_section("🧠 ANÁLISIS DE MÉTRICAS IIT")
    
    model.eval()
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'─'*80}")
        print(f"Prompt {i}: '{prompt}'")
        print(f"{'─'*80}")
        
        # Tokenizar
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            
            # El modelo puede retornar tupla o dict
            if isinstance(outputs, dict):
                output_dict = outputs
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                # (logits, metrics_dict)
                logits, output_dict = outputs
            else:
                print(f"  Formato de salida no reconocido: {type(outputs)}")
                continue
            
            # Obtener métricas IIT si están disponibles
            if output_dict and 'phi' in output_dict:
                print(f"\n  PHI:                    {output_dict['phi'].mean().item():.4f}")
            
            if output_dict and 'integration' in output_dict:
                print(f"  Integration:            {output_dict['integration'].mean().item():.4f}")
            
            if output_dict and 'delta_phi' in output_dict:
                print(f"  ΔPhi:                   {output_dict['delta_phi'].mean().item():.4f}")
            
            if output_dict and 'memory_usage' in output_dict:
                print(f"  Memory Usage:           {output_dict['memory_usage'].mean().item():.4f}")
            
            # Analizar atención
            if output_dict and 'attention_weights' in output_dict and output_dict['attention_weights'] is not None:
                attn = output_dict['attention_weights'][-1]  # Última capa
                print(f"\n  Attention Stats (última capa):")
                print(f"    Shape:                {attn.shape}")
                print(f"    Mean:                 {attn.mean().item():.4f}")
                print(f"    Std:                  {attn.std().item():.4f}")
                print(f"    Max:                  {attn.max().item():.4f}")
                print(f"    Min:                  {attn.min().item():.4f}")
            else:
                print(f"\n  No se encontraron métricas IIT en la salida")


def test_generation_quality(model, tokenizer, device='cuda'):
    """Prueba la calidad de generación con varios prompts."""
    print_section("✍️  GENERACIÓN DE TEXTO")
    
    test_prompts = [
        "The meaning of life is",
        "In the beginning",
        "Artificial intelligence",
        "Once upon a time",
        "The future of humanity"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'─'*80}")
        print(f"Prompt {i}: '{prompt}'")
        print(f"{'─'*80}")
        
        # Generar con diferentes configuraciones
        configs = [
            ("Greedy", 0.1, 1, 1.0),
            ("Balanced", 0.8, 50, 0.9),
            ("Creative", 1.2, 40, 0.95)
        ]
        
        for name, temp, top_k, top_p in configs:
            text = generate_text(
                model, tokenizer, prompt,
                max_length=50,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                device=device
            )
            print(f"\n  [{name}] (temp={temp}, top_k={top_k}, top_p={top_p})")
            print(f"  {text}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar modelo entrenado INFINITO V5.2')
    parser.add_argument('--checkpoint', type=str, 
                       default='models/checkpoints/infinito_v5.2_real_best.pt',
                       help='Ruta al checkpoint del modelo')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device a usar')
    
    args = parser.parse_args()
    
    # Verificar que existe el checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: No se encontró el checkpoint: {args.checkpoint}")
        return
    
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    print(f"\n{'='*80}")
    print(f"  🔬 ANÁLISIS COMPLETO DEL MODELO - INFINITO V5.2")
    print(f"{'='*80}")
    print(f"\n  Device:      {device}")
    print(f"  Checkpoint:  {args.checkpoint}")
    
    # Cargar checkpoint
    print(f"\n  Cargando checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Crear modelo
    config = checkpoint.get('config', {})
    model = InfinitoV52Refactored(
        vocab_size=config.get('vocab_size', 50257),
        hidden_dim=config.get('hidden_dim', 512),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 8),
        memory_slots=config.get('memory_slots', 256),
        use_improved_memory=True,
        use_stochastic_exploration=True,
        seed=42
    ).to(device)
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Cargar tokenizer
    print(f"  Cargando GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print(f"  ✓ Modelo cargado exitosamente\n")
    
    # 1. Analizar arquitectura
    analyze_model_architecture(model, checkpoint)
    
    # 2. Analizar resultados de entrenamiento
    analyze_training_results(checkpoint)
    
    # 3. Analizar métricas IIT
    test_prompts = [
        "The",
        "In the beginning",
        "Artificial intelligence"
    ]
    analyze_iit_metrics(model, tokenizer, test_prompts, device)
    
    # 4. Probar generación de texto
    test_generation_quality(model, tokenizer, device)
    
    print_section("✅ ANÁLISIS COMPLETADO")
    
    print(f"Resumen:")
    print(f"  El modelo tiene {sum(p.numel() for p in model.parameters()):,} parámetros")
    print(f"  Val PPL alcanzado: {checkpoint.get('val_ppl', 'N/A'):.2f}")
    print(f"  PHI promedio: {checkpoint.get('train_phi', 'N/A'):.4f}")
    print(f"\nRevisa la generación de texto arriba para evaluar la calidad.")


if __name__ == '__main__':
    main()
