#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
👁️ VISUALIZACIÓN DE ATENCIÓN - FASE 2 IIT TRANSFORMER
======================================================

Visualiza patrones de atención del modelo Fase 2 para entender
por qué colapsa en repeticiones.
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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import GPT2Tokenizer
from train_phase2_iit_transformer import InfinitoGPT2IITPhase2


def visualize_attention_heatmap(attention_weights, tokens, layer_idx, head_idx, save_path=None):
    """
    Visualiza matriz de atención como heatmap.
    
    Args:
        attention_weights: (seq_len, seq_len) - pesos de atención
        tokens: Lista de tokens
        layer_idx: Índice de la capa
        head_idx: Índice de la cabeza de atención
        save_path: Ruta para guardar imagen (opcional)
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Crear heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar=True,
        square=True,
        ax=ax,
        vmin=0,
        vmax=attention_weights.max()
    )
    
    ax.set_title(f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}', fontsize=14, pad=20)
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Guardado: {save_path}")
    
    plt.close()


def visualize_attention_patterns(attention_weights, tokens, layer_idx, save_path=None):
    """
    Visualiza patrones de atención agregados de todas las cabezas.
    
    Args:
        attention_weights: (num_heads, seq_len, seq_len)
        tokens: Lista de tokens
        layer_idx: Índice de la capa
        save_path: Ruta para guardar imagen (opcional)
    """
    # Promedio de todas las cabezas
    avg_attention = attention_weights.mean(axis=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Atención promedio
    sns.heatmap(avg_attention, ax=axes[0], cmap='viridis', cbar=True, square=True,
                xticklabels=tokens, yticklabels=tokens)
    axes[0].set_title(f'Average Attention - Layer {layer_idx}', fontsize=12)
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=7)
    
    # 2. Entropía por posición (diversidad de atención)
    entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-9), axis=1)
    axes[1].plot(entropy, marker='o', color='coral')
    axes[1].set_title('Attention Entropy per Position', fontsize=12)
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('Entropy (bits)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(len(tokens)))
    axes[1].set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
    
    # 3. Suma de atención por posición (qué tokens reciben más atención)
    attention_sum = avg_attention.sum(axis=0)
    axes[2].bar(range(len(tokens)), attention_sum, color='steelblue', alpha=0.7)
    axes[2].set_title('Total Attention Received', fontsize=12)
    axes[2].set_xlabel('Token')
    axes[2].set_ylabel('Total Attention')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].set_xticks(range(len(tokens)))
    axes[2].set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Guardado: {save_path}")
    
    plt.close()


def visualize_phi_components(phi_metrics, tokens, save_path=None):
    """
    Visualiza componentes PHI por token.
    
    Args:
        phi_metrics: Dict con métricas PHI
        tokens: Lista de tokens
        save_path: Ruta para guardar imagen (opcional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    components = [
        ('temporal_coherence', 'Temporal Coherence', axes[0, 0]),
        ('integration_strength', 'Integration Strength', axes[0, 1]),
        ('complexity', 'Complexity', axes[1, 0]),
        ('attention_diversity', 'Attention Diversity', axes[1, 1])
    ]
    
    for key, title, ax in components:
        if key in phi_metrics and phi_metrics[key] is not None:
            values = phi_metrics[key].cpu().numpy()
            if values.ndim == 2:  # (batch, seq_len)
                values = values[0]  # Tomar primer batch
            
            ax.plot(values, marker='o', color='teal', linewidth=2, markersize=6)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Token Position')
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
            
            # Línea de promedio
            mean_val = values.mean()
            ax.axhline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, color='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 Guardado: {save_path}")
    
    plt.close()


def analyze_attention(model, tokenizer, prompt, checkpoint, device='cuda', output_dir='results/attention_viz'):
    """
    Analiza y visualiza patrones de atención del modelo.
    
    Args:
        model: Modelo InfinitoGPT2IITPhase2
        tokenizer: GPT2Tokenizer
        prompt: Texto de entrada
        checkpoint: Dict con métricas del checkpoint
        device: 'cuda' o 'cpu'
        output_dir: Directorio para guardar visualizaciones
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE ATENCIÓN - FASE 2")
    print(f"{'='*70}")
    print(f"Prompt: '{prompt}'")
    print(f"{'='*70}\n")
    
    # Tokenizar
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    tokens = [tokenizer.decode([tok]) for tok in input_ids[0]]
    
    print(f"📝 Tokens ({len(tokens)}): {tokens}\n")
    
    model.eval()
    
    with torch.no_grad():
        # Forward pass manual para capturar atenciones
        batch_size, seq_len = input_ids.shape
        
        # 1. Embeddings
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        hidden_states = model.gpt2_embedding(input_ids) + model.gpt2_position_embedding(position_ids)
        hidden_states = model.gpt2_drop(hidden_states)
        
        # 2. GPT-2 Layers (con atenciones)
        all_attentions = []
        for layer in model.gpt2_layers:
            outputs = layer(hidden_states, output_attentions=True, use_cache=False)
            hidden_states = outputs[0]
            if len(outputs) > 2:  # Verificar que hay atención
                all_attentions.append(outputs[2])  # Attention weights están en índice 2
        
        # 3. IIT Transformer Block
        iit_outputs = model.iit_transformer(hidden_states)
        iit_hidden = iit_outputs[0]
        iit_attentions = iit_outputs[1] if len(iit_outputs) > 1 else None
        
        # PHI metrics - simplificado (sin calcular, solo reportar checkpoint)
        phi_metrics = None
    
    print(f"🧠 Información del checkpoint:")
    print(f"  PHI (validación): {checkpoint['val_phi']:.4f}")
    print(f"  PPL (validación): {checkpoint['val_ppl']:.2f}\n")
    
    # Visualizar atenciones GPT-2 (últimas 3 capas)
    print(f"📊 Generando visualizaciones GPT-2...")
    num_layers_to_viz = min(3, len(all_attentions))
    for i in range(-num_layers_to_viz, 0):
        layer_idx = len(all_attentions) + i
        attention = all_attentions[i][0].cpu().numpy()  # (num_heads, seq_len, seq_len)
        
        # Visualización agregada
        save_path = os.path.join(output_dir, f'gpt2_layer_{layer_idx}_patterns.png')
        visualize_attention_patterns(attention, tokens, layer_idx, save_path)
        
        # Visualización de cabezas individuales (primeras 4)
        for head_idx in range(min(4, attention.shape[0])):
            save_path = os.path.join(output_dir, f'gpt2_layer_{layer_idx}_head_{head_idx}.png')
            visualize_attention_heatmap(attention[head_idx], tokens, layer_idx, head_idx, save_path)
    
    # Visualizar atenciones IIT Transformer
    if iit_attentions is not None:
        print(f"📊 Generando visualizaciones IIT Transformer...")
        for layer_idx, attention in enumerate(iit_attentions):
            attention_np = attention[0].cpu().numpy()  # (num_heads, seq_len, seq_len)
            
            # Visualización agregada
            save_path = os.path.join(output_dir, f'iit_layer_{layer_idx}_patterns.png')
            visualize_attention_patterns(attention_np, tokens, f'IIT-{layer_idx}', save_path)
            
            # Visualización de cabezas individuales (primeras 4)
            for head_idx in range(min(4, attention_np.shape[0])):
                save_path = os.path.join(output_dir, f'iit_layer_{layer_idx}_head_{head_idx}.png')
                visualize_attention_heatmap(attention_np[head_idx], tokens, f'IIT-{layer_idx}', head_idx, save_path)
    
    # Visualizar componentes PHI (solo si tenemos métricas)
    if phi_metrics is not None:
        print(f"📊 Generando visualizaciones PHI...")
        save_path = os.path.join(output_dir, f'phi_components.png')
        visualize_phi_components(phi_metrics, tokens, save_path)
    
    print(f"\n{'='*70}")
    print(f"✅ VISUALIZACIONES COMPLETADAS")
    print(f"  📁 Directorio: {output_dir}")
    print(f"{'='*70}\n")
    
    return {
        'tokens': tokens,
        'phi_metrics': {k: v.cpu() if torch.is_tensor(v) else v for k, v in phi_metrics.items()} if phi_metrics else None,
        'gpt2_attentions': [att.cpu() for att in all_attentions],
        'iit_attentions': [att.cpu() for att in iit_attentions] if iit_attentions else None
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualización de atención - Fase 2')
    parser.add_argument('--prompt', type=str, default='The theory of consciousness')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoints/infinito_phase2_best.pt')
    parser.add_argument('--output-dir', type=str, default='results/attention_viz')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar tokenizer
    print(f"\n🔤 Cargando tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Cargar modelo
    print(f"\n📦 Cargando modelo Fase 2...")
    model = InfinitoGPT2IITPhase2(
        num_iit_layers=2,
        lambda_phi=1.0
    ).to(device)
    
    print(f"  📂 Cargando checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n📊 Métricas del checkpoint:")
    print(f"  Época: {checkpoint['epoch']}")
    print(f"  Val PHI: {checkpoint['val_phi']:.4f}")
    print(f"  Val PPL: {checkpoint['val_ppl']:.2f}")
    
    # Analizar atención
    results = analyze_attention(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        checkpoint=checkpoint,
        device=device,
        output_dir=args.output_dir
    )
    
    # Análisis de patrones problemáticos
    print(f"\n{'='*70}")
    print(f"🔍 ANÁLISIS DE PATRONES PROBLEMÁTICOS")
    print(f"{'='*70}\n")
    
    # Verificar si hay atención colapsada (focalizada en pocos tokens)
    if results['gpt2_attentions'] and len(results['gpt2_attentions']) > 0:
        last_attention = results['gpt2_attentions'][-1].numpy()  # (seq_len, seq_len) o (num_heads, seq_len, seq_len)
        
        # Si tiene dimensión de heads, promediar
        if last_attention.ndim == 3:
            avg_attention = last_attention.mean(axis=0)  # (seq_len, seq_len)
        else:
            avg_attention = last_attention
        
        # Calcular entropía promedio
        entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-9), axis=1)
        avg_entropy = entropy.mean()
        
        print(f"📊 Estadísticas de Atención (última capa GPT-2):")
        print(f"  Entropía promedio: {avg_entropy:.4f} bits")
        print(f"  Entropía mínima: {entropy.min():.4f} bits (posición {entropy.argmin()})")
        print(f"  Entropía máxima: {entropy.max():.4f} bits (posición {entropy.argmax()})")
        
        if avg_entropy < 2.0:
            print(f"\n  ⚠️  ALERTA: Baja entropía detectada (<2.0 bits)")
            print(f"     La atención está colapsada en pocos tokens")
        
        # Verificar tokens con mayor atención
        attention_received = avg_attention.sum(axis=0)
        top_tokens = np.argsort(attention_received)[-3:][::-1]
        
        print(f"\n📌 Tokens con mayor atención recibida:")
        for idx in top_tokens:
            token = results['tokens'][idx]
            score = attention_received[idx]
            print(f"  [{idx}] '{token}': {score:.3f}")


if __name__ == '__main__':
    main()
