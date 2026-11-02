#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ENTRENAMIENTO MEJORADO DEL MODELO BASE
==========================================

Mejoras implementadas:
1. Datos mixtos: Dyck + texto real del replay buffer
2. Curriculum learning: empezar fácil, aumentar dificultad
3. Learning rate scheduling con warmup
4. Gradient clipping para estabilidad
5. Métricas detalladas durante entrenamiento
6. Early stopping si no mejora
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'src')
from infinito_v5_2_refactored import InfinitoV52Refactored

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CONFIG = {
    'epochs': 500,
    'batch_size': 32,
    'lr_start': 1e-4,
    'lr_max': 5e-4,
    'lr_end': 1e-5,
    'warmup_epochs': 50,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'patience': 100,  # early stopping
    'checkpoint_every': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# =============================================================================
# VOCABULARIO EXTENDIDO
# =============================================================================

# Vocabulario base (Dyck)
VOCAB_BASE = {'PAD': 0, '(': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6, '<': 7, '>': 8, 
              'A': 9, 'B': 10, 'C': 11, 'EOS': 12}

# Para texto real, usamos character-level con los primeros 256 códigos ASCII
def text_to_ids(text, max_len=64):
    """Convierte texto a IDs (character-level mod 256)."""
    ids = [ord(c) % 256 for c in text.lower()[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids)


# =============================================================================
# GENERACIÓN DE DATOS
# =============================================================================

def generate_dyck_sample(max_depth=12, noise_len=6):
    """Genera secuencias Dyck con profundidad y ruido."""
    pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
    depth = random.randint(4, max_depth)
    stack = []
    sequence = []
    
    for _ in range(depth):
        pair = random.choice(pairs)
        sequence.append(pair[0])
        stack.append(pair[1])
    
    noise = [random.choice(['A', 'B', 'C']) for _ in range(noise_len)]
    input_str = sequence + noise
    target_str = list(reversed(stack))
    
    return input_str, target_str


def get_dyck_batch(batch_size=32, difficulty=1.0):
    """Genera batch de Dyck con dificultad variable."""
    max_depth = int(4 + 8 * difficulty)  # 4-12 según dificultad
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        inp, tar = generate_dyck_sample(max_depth=max_depth)
        inp_ids = [VOCAB_BASE[c] for c in inp]
        tar_ids = [VOCAB_BASE[c] for c in tar] + [VOCAB_BASE['EOS']]
        
        inputs.append(torch.tensor(inp_ids))
        targets.append(torch.tensor(tar_ids))
    
    inp_tens = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    tar_tens = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inp_tens, tar_tens


def load_replay_buffer():
    """Carga experiencias reales del replay buffer."""
    try:
        with open('data/replay_buffer.json', 'r', encoding='utf-8') as f:
            buffer = json.load(f)
        print(f"📚 Replay buffer cargado: {len(buffer)} experiencias")
        return buffer
    except:
        print("⚠️ No se pudo cargar replay buffer")
        return []


def get_text_batch(replay_buffer, batch_size=16):
    """Genera batch de texto real del replay buffer."""
    if not replay_buffer:
        return None, None
    
    samples = random.choices(replay_buffer, k=min(batch_size, len(replay_buffer)))
    
    inputs = []
    targets = []
    
    for sample in samples:
        text = sample['text']
        # Input: primeros N caracteres, Target: siguientes caracteres
        if len(text) > 10:
            split = len(text) // 2
            inp = text_to_ids(text[:split], max_len=32)
            tar = text_to_ids(text[split:], max_len=32)
            inputs.append(inp)
            targets.append(tar)
    
    if not inputs:
        return None, None
    
    inp_tens = torch.stack(inputs)
    tar_tens = torch.stack(targets)
    return inp_tens, tar_tens


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

def get_lr(epoch, warmup_epochs, total_epochs, lr_start, lr_max, lr_end):
    """Cosine schedule with warmup."""
    if epoch < warmup_epochs:
        # Linear warmup
        return lr_start + (lr_max - lr_start) * (epoch / warmup_epochs)
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr_end + (lr_max - lr_end) * 0.5 * (1 + np.cos(np.pi * progress))


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train():
    print("=" * 60)
    print("🚀 ENTRENAMIENTO MEJORADO DEL MODELO")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print()
    
    device = CONFIG['device']
    
    # Cargar modelo base
    print("📂 Cargando modelo base...")
    checkpoint = torch.load('models/super_golden_seed_54percent.pt', weights_only=True)
    
    model = InfinitoV52Refactored(
        vocab_size=checkpoint['config']['vocab_size'],
        hidden_dim=checkpoint['config']['hidden_dim'],
        num_layers=checkpoint['config']['num_layers'],
        num_heads=checkpoint['config']['num_heads'],
        use_improved_memory=True,
        use_improved_iit=True,
        use_learnable_phi=True,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Modelo cargado: {sum(p.numel() for p in model.parameters())} parámetros")
    
    # Cargar replay buffer
    replay_buffer = load_replay_buffer()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG['lr_start'],
        weight_decay=CONFIG['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Tracking
    best_loss = float('inf')
    best_epoch = 0
    no_improve_count = 0
    history = []
    
    print()
    print("📈 Iniciando entrenamiento...")
    print("-" * 60)
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        
        # Curriculum: aumentar dificultad gradualmente
        difficulty = min(1.0, epoch / 200)
        
        # Learning rate schedule
        lr = get_lr(
            epoch, 
            CONFIG['warmup_epochs'], 
            CONFIG['epochs'],
            CONFIG['lr_start'],
            CONFIG['lr_max'],
            CONFIG['lr_end']
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Batch mixto: 75% Dyck + 25% texto real
        total_loss = 0
        num_batches = 0
        
        # Batch Dyck
        input_ids, target_ids = get_dyck_batch(CONFIG['batch_size'], difficulty)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        
        optimizer.zero_grad()
        logits, metrics = model(input_ids, return_metrics=True)
        min_len = min(logits.shape[1], target_ids.shape[1])
        loss_dyck = criterion(logits[:, :min_len, :].transpose(1, 2), target_ids[:, :min_len])
        
        total_loss += loss_dyck
        num_batches += 1
        
        # Batch de texto real (si hay datos)
        if replay_buffer and epoch % 4 == 0:
            text_inp, text_tar = get_text_batch(replay_buffer, CONFIG['batch_size'] // 2)
            if text_inp is not None:
                # Ajustar IDs para vocab pequeño (mod vocab_size)
                text_inp = (text_inp % checkpoint['config']['vocab_size']).to(device)
                text_tar = (text_tar % checkpoint['config']['vocab_size']).to(device)
                
                logits_text, _ = model(text_inp, return_metrics=True)
                min_len = min(logits_text.shape[1], text_tar.shape[1])
                loss_text = criterion(logits_text[:, :min_len, :].transpose(1, 2), text_tar[:, :min_len])
                
                total_loss += 0.25 * loss_text  # Peso menor para texto
                num_batches += 0.25
        
        # Backward con gradient clipping
        avg_loss = total_loss / num_batches
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        
        # Tracking
        history.append({
            'epoch': epoch,
            'loss': avg_loss.item(),
            'lr': lr,
            'difficulty': difficulty,
            'phi': metrics.get('phi', 0) if isinstance(metrics, dict) else 0,
        })
        
        # Early stopping check
        if avg_loss.item() < best_loss:
            best_loss = avg_loss.item()
            best_epoch = epoch
            no_improve_count = 0
            
            # Guardar mejor modelo
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': checkpoint['config'],
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'history': history[-100:],  # Últimos 100
            }, 'models/improved_model_best.pt')
        else:
            no_improve_count += 1
        
        # Logging
        if epoch % 25 == 0:
            phi = metrics.get('phi', 0) if isinstance(metrics, dict) else 0
            print(f"Epoch {epoch:4d} | Loss: {avg_loss.item():.4f} | LR: {lr:.6f} | "
                  f"Diff: {difficulty:.2f} | PHI: {phi:.4f} | Best: {best_loss:.4f}")
        
        # Checkpoint periódico
        if epoch > 0 and epoch % CONFIG['checkpoint_every'] == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss.item(),
            }, f'models/checkpoints/improved_epoch_{epoch}.pt')
        
        # Early stopping
        if no_improve_count >= CONFIG['patience']:
            print(f"\n⏹️ Early stopping en epoch {epoch} (no mejora en {CONFIG['patience']} epochs)")
            break
    
    print()
    print("=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"Mejor loss: {best_loss:.4f} en epoch {best_epoch}")
    print(f"Modelo guardado: models/improved_model_best.pt")
    
    # Comparar con baseline
    baseline_loss = checkpoint['experiment_results']['baseline_final_loss']
    improvement = (baseline_loss - best_loss) / baseline_loss * 100
    print(f"\n📊 Comparación:")
    print(f"  Baseline original: {baseline_loss:.4f}")
    print(f"  Nuevo mejor:       {best_loss:.4f}")
    print(f"  Mejora:            {improvement:+.1f}%")
    
    return model, history


if __name__ == "__main__":
    os.makedirs('models/checkpoints', exist_ok=True)
    train()
