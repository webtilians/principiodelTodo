#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ENTRENAMIENTO EXTENDIDO CON DATOS REALES
=============================================

Mejoras sobre el entrenamiento original:
1. 1000+ epochs (vs 200)
2. Datos mixtos: Dyck + texto real + patrones de conversacion
3. Curriculum learning
4. Learning rate con warmup + cosine decay
5. Evaluacion periodica
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

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'src')
from infinito_v5_2_refactored import InfinitoV52Refactored

# =============================================================================
# CONFIGURACION
# =============================================================================

CONFIG = {
    'epochs': 1500,
    'batch_size': 32,
    'lr_max': 5e-4,
    'lr_min': 1e-5,
    'warmup_epochs': 100,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'eval_every': 100,
    'save_every': 250,
    'patience': 300,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# =============================================================================
# VOCABULARIO EXTENDIDO
# =============================================================================

# Vocabulario base (Dyck) - mantenemos compatibilidad
VOCAB_DYCK = {
    'PAD': 0, '(': 1, ')': 2, '[': 3, ']': 4, 
    '{': 5, '}': 6, '<': 7, '>': 8,
    'A': 9, 'B': 10, 'C': 11, 'EOS': 12
}

# Para texto real, mapeamos caracteres al rango 0-12
def char_to_id(c):
    """Mapea cualquier caracter al rango del vocabulario."""
    return ord(c.lower()) % 13

def text_to_ids(text, max_len=32):
    """Convierte texto a IDs compatibles con vocab_size=13."""
    ids = [char_to_id(c) for c in text[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids)


# =============================================================================
# DATOS DE ENTRENAMIENTO
# =============================================================================

# Patrones de conversacion en espanol
CONVERSATION_PATTERNS = [
    # Identidad
    ("me llamo", "nombre"),
    ("mi nombre es", "identidad"),
    ("soy", "presentacion"),
    ("tengo anos", "edad"),
    
    # Familia
    ("mi padre", "familia"),
    ("mi madre", "familia"),
    ("mi hermano", "familia"),
    ("mi primo", "familia"),
    ("mi hijo", "familia"),
    
    # Preferencias
    ("me gusta", "preferencia"),
    ("prefiero", "preferencia"),
    ("me encanta", "preferencia"),
    ("odio", "preferencia"),
    
    # Actividades
    ("hoy he", "actividad"),
    ("ayer hice", "actividad"),
    ("manana voy", "plan"),
    ("esta semana", "plan"),
    
    # Recordatorios
    ("recuerda que", "recordatorio"),
    ("no olvides", "recordatorio"),
    ("importante", "recordatorio"),
    
    # Datos tecnicos
    ("mi telefono", "contacto"),
    ("mi email", "contacto"),
    ("vivo en", "ubicacion"),
    ("trabajo en", "trabajo"),
]

# Frases completas para entrenamiento
TRAINING_PHRASES = [
    "me llamo enrique garcia",
    "mi padre se llama juan",
    "mi madre es profesora",
    "tengo un hermano mayor",
    "me gusta el ciclismo",
    "hago descenso en bici",
    "mi bici es una scott",
    "peso setenta y cinco kilos",
    "vivo en madrid",
    "trabajo como ingeniero",
    "mañana tengo cita medico",
    "el viernes vamos en bici",
    "mi primo andres monta bici",
    "hoy he igualado mi tiempo",
    "mi objetivo es mejorar",
    "prefiero el cafe solo",
    "me encanta la montana",
    "odio madrugar mucho",
    "mi telefono es seis seis",
    "recuerda llamar a juan",
]

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
    max_depth = int(4 + 8 * difficulty)
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        inp, tar = generate_dyck_sample(max_depth=max_depth)
        inp_ids = [VOCAB_DYCK[c] for c in inp]
        tar_ids = [VOCAB_DYCK[c] for c in tar] + [VOCAB_DYCK['EOS']]
        
        inputs.append(torch.tensor(inp_ids))
        targets.append(torch.tensor(tar_ids))
    
    inp_tens = nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    tar_tens = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inp_tens, tar_tens


def get_text_batch(batch_size=16):
    """Genera batch de texto real."""
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        # Elegir frase aleatoria
        phrase = random.choice(TRAINING_PHRASES)
        
        # Dividir en input/target (prediccion de siguiente parte)
        if len(phrase) > 10:
            split = len(phrase) // 2
            inp = text_to_ids(phrase[:split], max_len=16)
            tar = text_to_ids(phrase[split:], max_len=16)
        else:
            inp = text_to_ids(phrase, max_len=16)
            tar = text_to_ids(phrase, max_len=16)  # Autoencoder
        
        inputs.append(inp)
        targets.append(tar)
    
    return torch.stack(inputs), torch.stack(targets)


def get_pattern_batch(batch_size=16):
    """Genera batch de patrones de conversacion."""
    inputs = []
    targets = []
    
    for _ in range(batch_size):
        pattern, category = random.choice(CONVERSATION_PATTERNS)
        
        # Input: patron, Target: categoria (comprimida)
        inp = text_to_ids(pattern, max_len=16)
        tar = text_to_ids(category, max_len=16)
        
        inputs.append(inp)
        targets.append(tar)
    
    return torch.stack(inputs), torch.stack(targets)


def load_replay_buffer():
    """Carga experiencias reales del replay buffer."""
    try:
        with open('data/replay_buffer.json', 'r', encoding='utf-8') as f:
            buffer = json.load(f)
        return [item['text'] for item in buffer if 'text' in item]
    except:
        return []


def get_replay_batch(replay_texts, batch_size=16):
    """Genera batch del replay buffer."""
    if not replay_texts:
        return None, None
    
    inputs = []
    targets = []
    
    samples = random.choices(replay_texts, k=min(batch_size, len(replay_texts)))
    
    for text in samples:
        if len(text) > 8:
            split = len(text) // 2
            inp = text_to_ids(text[:split], max_len=20)
            tar = text_to_ids(text[split:], max_len=20)
            inputs.append(inp)
            targets.append(tar)
    
    if not inputs:
        return None, None
    
    return torch.stack(inputs), torch.stack(targets)


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

def get_lr(epoch, warmup, total, lr_max, lr_min):
    """Warmup + Cosine decay."""
    if epoch < warmup:
        return lr_min + (lr_max - lr_min) * (epoch / warmup)
    else:
        progress = (epoch - warmup) / (total - warmup)
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi * progress))


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train():
    print("=" * 70)
    print("  ENTRENAMIENTO EXTENDIDO CON DATOS REALES")
    print("=" * 70)
    print(f"Device: {CONFIG['device']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print()
    
    device = CONFIG['device']
    
    # Semillas del Golden Seed original
    set_seeds(2)
    
    # Crear modelo desde cero con la misma config
    model = InfinitoV52Refactored(
        vocab_size=13,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_improved_memory=True,
        use_improved_iit=True,
        use_learnable_phi=True,
        use_stochastic_exploration=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo inicializado: {total_params:,} parametros")
    
    # Cargar replay buffer
    replay_texts = load_replay_buffer()
    print(f"Replay buffer: {len(replay_texts)} textos")
    print(f"Frases entrenamiento: {len(TRAINING_PHRASES)}")
    print(f"Patrones conversacion: {len(CONVERSATION_PATTERNS)}")
    
    # Crear baseline para comparacion
    set_seeds(2)
    baseline = InfinitoV52Refactored(
        vocab_size=13, hidden_dim=64, num_layers=2, num_heads=4,
        use_improved_memory=False, use_improved_iit=False,
        use_learnable_phi=False, use_stochastic_exploration=False,
    ).to(device)
    
    # Optimizadores
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr_max'], weight_decay=CONFIG['weight_decay'])
    optimizer_base = optim.AdamW(baseline.parameters(), lr=CONFIG['lr_max'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Tracking
    best_loss = float('inf')
    best_improvement = 0
    no_improve = 0
    history = []
    
    # Data seed
    set_seeds(42)
    
    print()
    print("Iniciando entrenamiento...")
    print("-" * 70)
    
    pbar = tqdm(range(CONFIG['epochs']), desc="Training")
    
    for epoch in pbar:
        model.train()
        baseline.train()
        
        # Curriculum: aumentar dificultad
        difficulty = min(1.0, epoch / 500)
        
        # Learning rate schedule
        lr = get_lr(epoch, CONFIG['warmup_epochs'], CONFIG['epochs'], 
                   CONFIG['lr_max'], CONFIG['lr_min'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        for pg in optimizer_base.param_groups:
            pg['lr'] = lr
        
        total_loss_iit = 0
        total_loss_base = 0
        n_batches = 0
        
        # === BATCH 1: Dyck Language (tarea principal) ===
        inp, tar = get_dyck_batch(CONFIG['batch_size'], difficulty)
        inp, tar = inp.to(device), tar.to(device)
        
        # IIT model
        optimizer.zero_grad()
        logits, metrics = model(inp, return_metrics=True)
        min_len = min(logits.shape[1], tar.shape[1])
        loss_iit = criterion(logits[:, :min_len, :].transpose(1, 2), tar[:, :min_len])
        loss_iit.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        
        # Baseline
        optimizer_base.zero_grad()
        logits_b, _ = baseline(inp, return_metrics=True)
        loss_base = criterion(logits_b[:, :min_len, :].transpose(1, 2), tar[:, :min_len])
        loss_base.backward()
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), CONFIG['grad_clip'])
        optimizer_base.step()
        
        total_loss_iit += loss_iit.item()
        total_loss_base += loss_base.item()
        n_batches += 1
        
        # === BATCH 2: Texto real (cada 3 epochs) ===
        if epoch % 3 == 0:
            inp_t, tar_t = get_text_batch(CONFIG['batch_size'] // 2)
            inp_t, tar_t = inp_t.to(device), tar_t.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(inp_t, return_metrics=True)
            min_len = min(logits.shape[1], tar_t.shape[1])
            loss = criterion(logits[:, :min_len, :].transpose(1, 2), tar_t[:, :min_len])
            (0.3 * loss).backward()  # Peso menor
            optimizer.step()
            
            optimizer_base.zero_grad()
            logits_b, _ = baseline(inp_t, return_metrics=True)
            loss_b = criterion(logits_b[:, :min_len, :].transpose(1, 2), tar_t[:, :min_len])
            (0.3 * loss_b).backward()
            optimizer_base.step()
        
        # === BATCH 3: Patrones (cada 5 epochs) ===
        if epoch % 5 == 0:
            inp_p, tar_p = get_pattern_batch(CONFIG['batch_size'] // 2)
            inp_p, tar_p = inp_p.to(device), tar_p.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(inp_p, return_metrics=True)
            min_len = min(logits.shape[1], tar_p.shape[1])
            loss = criterion(logits[:, :min_len, :].transpose(1, 2), tar_p[:, :min_len])
            (0.2 * loss).backward()
            optimizer.step()
        
        # === BATCH 4: Replay buffer (cada 10 epochs) ===
        if epoch % 10 == 0 and replay_texts:
            inp_r, tar_r = get_replay_batch(replay_texts, CONFIG['batch_size'] // 4)
            if inp_r is not None:
                inp_r, tar_r = inp_r.to(device), tar_r.to(device)
                
                optimizer.zero_grad()
                logits, _ = model(inp_r, return_metrics=True)
                min_len = min(logits.shape[1], tar_r.shape[1])
                loss = criterion(logits[:, :min_len, :].transpose(1, 2), tar_r[:, :min_len])
                (0.2 * loss).backward()
                optimizer.step()
        
        # Calcular mejora
        avg_loss_iit = total_loss_iit / n_batches
        avg_loss_base = total_loss_base / n_batches
        
        if avg_loss_base > 0.01:
            improvement = (avg_loss_base - avg_loss_iit) / avg_loss_base * 100
        else:
            improvement = 0
        
        # Tracking
        history.append({
            'epoch': epoch,
            'loss_iit': avg_loss_iit,
            'loss_base': avg_loss_base,
            'improvement': improvement,
            'lr': lr,
        })
        
        # Actualizar mejor
        if avg_loss_iit < best_loss and improvement > 0:
            best_loss = avg_loss_iit
            best_improvement = improvement
            no_improve = 0
            
            # Guardar mejor modelo
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'vocab_size': 13,
                    'hidden_dim': 64,
                    'num_layers': 2,
                    'num_heads': 4,
                    'use_improved_memory': True,
                    'use_improved_iit': True,
                    'use_learnable_phi': True,
                    'use_stochastic_exploration': True,
                },
                'experiment_results': {
                    'iit_final_loss': avg_loss_iit,
                    'baseline_final_loss': avg_loss_base,
                    'improvement_percentage': improvement,
                    'epoch': epoch,
                },
                'base_seed': 2,
                'data_seed': 42,
            }, 'models/extended_training_best.pt')
        else:
            no_improve += 1
        
        # Progress bar
        pbar.set_postfix({
            'loss': f'{avg_loss_iit:.3f}',
            'imp': f'{improvement:.1f}%',
            'best': f'{best_improvement:.1f}%',
        })
        
        # Logging periodico
        if epoch > 0 and epoch % CONFIG['eval_every'] == 0:
            tqdm.write(f"Epoch {epoch:4d} | IIT: {avg_loss_iit:.4f} | Base: {avg_loss_base:.4f} | "
                      f"Mejora: {improvement:.1f}% | Best: {best_improvement:.1f}%")
        
        # Checkpoint periodico
        if epoch > 0 and epoch % CONFIG['save_every'] == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'improvement': improvement,
            }, f'models/checkpoints/extended_epoch_{epoch}.pt')
        
        # Early stopping
        if no_improve >= CONFIG['patience']:
            print(f"\nEarly stopping en epoch {epoch}")
            break
    
    print()
    print("=" * 70)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"Mejor loss IIT: {best_loss:.4f}")
    print(f"Mejor mejora sobre baseline: {best_improvement:.1f}%")
    print()
    
    # Comparar con Golden Seed original
    try:
        orig = torch.load('models/super_golden_seed_54percent.pt', weights_only=True)
        orig_improvement = orig['experiment_results']['improvement_percentage']
        print(f"Comparacion con Golden Seed original:")
        print(f"  Original: {orig_improvement:.1f}%")
        print(f"  Nuevo:    {best_improvement:.1f}%")
        if best_improvement > orig_improvement:
            print(f"  MEJORA: +{best_improvement - orig_improvement:.1f}%")
        else:
            print(f"  Diferencia: {best_improvement - orig_improvement:.1f}%")
    except:
        pass
    
    print()
    print(f"Modelo guardado: models/extended_training_best.pt")
    
    return model, history


if __name__ == "__main__":
    os.makedirs('models/checkpoints', exist_ok=True)
    train()
