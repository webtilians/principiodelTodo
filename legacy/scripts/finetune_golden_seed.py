#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 FINE-TUNING DEL GOLDEN SEED CON DATOS REALES
================================================

Continua desde el modelo Golden Seed 54% y lo mejora
con datos de texto real, sin perder lo aprendido.
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

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, 'src')
from infinito_v5_2_refactored import InfinitoV52Refactored

# =============================================================================
# CONFIGURACION - Learning rate muy bajo para no destruir lo aprendido
# =============================================================================

CONFIG = {
    'epochs': 500,
    'batch_size': 32,
    'lr': 1e-5,  # Muy bajo para fine-tuning
    'weight_decay': 0.001,
    'grad_clip': 0.5,
    'eval_every': 50,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# =============================================================================
# DATOS
# =============================================================================

VOCAB_DYCK = {
    'PAD': 0, '(': 1, ')': 2, '[': 3, ']': 4, 
    '{': 5, '}': 6, '<': 7, '>': 8,
    'A': 9, 'B': 10, 'C': 11, 'EOS': 12
}

# Frases reales para fine-tuning
REAL_PHRASES = [
    # Del replay buffer real
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
    "manana tengo cita medico",
    "el viernes vamos en bici",
    "mi primo andres monta bici",
    "hoy he igualado mi tiempo",
    "mi objetivo es mejorar",
    "prefiero el cafe solo",
    "me encanta la montana",
    "mi telefono es seis seis",
    "recuerda llamar a juan",
    # Mas variedad
    "mi hermana estudia derecho",
    "mi abuelo tiene ochenta anos",
    "me gusta leer por las noches",
    "los fines de semana salgo",
    "mi coche es un seat leon",
    "estudio informatica online",
    "mi perro se llama toby",
    "tengo cita el martes",
    "mi cumpleanos es en mayo",
    "prefiero la playa montana",
]


def text_to_ids(text, max_len=20):
    """Convierte texto a IDs (mod 13 para vocab_size=13)."""
    ids = [ord(c.lower()) % 13 for c in text[:max_len]]
    if len(ids) < max_len:
        ids = ids + [0] * (max_len - len(ids))
    return torch.tensor(ids)


def generate_dyck_sample(max_depth=12):
    pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
    depth = random.randint(4, max_depth)
    stack, sequence = [], []
    for _ in range(depth):
        pair = random.choice(pairs)
        sequence.append(pair[0])
        stack.append(pair[1])
    noise = [random.choice(['A', 'B', 'C']) for _ in range(6)]
    return sequence + noise, list(reversed(stack))


def get_dyck_batch(batch_size=32):
    inputs, targets = [], []
    for _ in range(batch_size):
        inp, tar = generate_dyck_sample()
        inp_ids = [VOCAB_DYCK[c] for c in inp]
        tar_ids = [VOCAB_DYCK[c] for c in tar] + [VOCAB_DYCK['EOS']]
        inputs.append(torch.tensor(inp_ids))
        targets.append(torch.tensor(tar_ids))
    return (nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0),
            nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0))


def get_text_batch(batch_size=16):
    inputs, targets = [], []
    for _ in range(batch_size):
        phrase = random.choice(REAL_PHRASES)
        if len(phrase) > 8:
            split = len(phrase) // 2
            inputs.append(text_to_ids(phrase[:split], 16))
            targets.append(text_to_ids(phrase[split:], 16))
    return torch.stack(inputs), torch.stack(targets)


def load_replay_texts():
    try:
        with open('data/replay_buffer.json', 'r', encoding='utf-8') as f:
            return [item['text'] for item in json.load(f) if 'text' in item]
    except:
        return []


# =============================================================================
# FINE-TUNING
# =============================================================================

def finetune():
    print("=" * 70)
    print("  FINE-TUNING DEL GOLDEN SEED CON DATOS REALES")
    print("=" * 70)
    
    device = CONFIG['device']
    
    # Cargar modelo pre-entrenado
    print("\nCargando Golden Seed 54%...")
    checkpoint = torch.load('models/super_golden_seed_54percent.pt', weights_only=True)
    
    model = InfinitoV52Refactored(
        vocab_size=checkpoint['config']['vocab_size'],
        hidden_dim=checkpoint['config']['hidden_dim'],
        num_layers=checkpoint['config']['num_layers'],
        num_heads=checkpoint['config']['num_heads'],
        use_improved_memory=True,
        use_improved_iit=True,
        use_learnable_phi=True,
        use_stochastic_exploration=True,
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo cargado: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Loss original: {checkpoint['experiment_results']['iit_final_loss']:.4f}")
    print(f"Mejora original: {checkpoint['experiment_results']['improvement_percentage']:.1f}%")
    
    # Replay buffer
    replay_texts = load_replay_texts()
    print(f"Replay buffer: {len(replay_texts)} textos")
    print(f"Frases adicionales: {len(REAL_PHRASES)}")
    
    # Optimizer con learning rate muy bajo
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Evaluar baseline antes de fine-tuning
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for _ in range(20):
            inp, tar = get_dyck_batch(32)
            inp, tar = inp.to(device), tar.to(device)
            logits, _ = model(inp, return_metrics=True)
            min_len = min(logits.shape[1], tar.shape[1])
            loss = criterion(logits[:, :min_len, :].transpose(1, 2), tar[:, :min_len])
            total_loss += loss.item()
        initial_loss = total_loss / 20
    
    print(f"\nLoss inicial (evaluacion): {initial_loss:.4f}")
    
    best_loss = initial_loss
    history = []
    
    print("\nIniciando fine-tuning...")
    print("-" * 70)
    
    for epoch in tqdm(range(CONFIG['epochs']), desc="Fine-tuning"):
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        # Batch principal: Dyck (mantener la tarea original)
        inp, tar = get_dyck_batch(CONFIG['batch_size'])
        inp, tar = inp.to(device), tar.to(device)
        
        optimizer.zero_grad()
        logits, metrics = model(inp, return_metrics=True)
        min_len = min(logits.shape[1], tar.shape[1])
        loss_dyck = criterion(logits[:, :min_len, :].transpose(1, 2), tar[:, :min_len])
        
        epoch_loss += loss_dyck.item()
        n_batches += 1
        
        # Batch secundario: texto real (cada 2 epochs)
        if epoch % 2 == 0:
            inp_t, tar_t = get_text_batch(16)
            inp_t, tar_t = inp_t.to(device), tar_t.to(device)
            
            logits_t, _ = model(inp_t, return_metrics=True)
            min_len = min(logits_t.shape[1], tar_t.shape[1])
            loss_text = criterion(logits_t[:, :min_len, :].transpose(1, 2), tar_t[:, :min_len])
            
            # Peso menor para texto (no destruir Dyck)
            total_loss = loss_dyck + 0.1 * loss_text
        else:
            total_loss = loss_dyck
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        
        avg_loss = epoch_loss / n_batches
        history.append({'epoch': epoch, 'loss': avg_loss})
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': checkpoint['config'],
                'experiment_results': {
                    'iit_final_loss': avg_loss,
                    'baseline_final_loss': checkpoint['experiment_results']['baseline_final_loss'],
                    'improvement_percentage': checkpoint['experiment_results']['improvement_percentage'],
                    'finetuned': True,
                },
                'base_seed': checkpoint['base_seed'],
                'data_seed': checkpoint['data_seed'],
            }, 'models/golden_seed_finetuned.pt')
        
        if epoch > 0 and epoch % CONFIG['eval_every'] == 0:
            tqdm.write(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
    
    # Evaluacion final
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for _ in range(20):
            inp, tar = get_dyck_batch(32)
            inp, tar = inp.to(device), tar.to(device)
            logits, _ = model(inp, return_metrics=True)
            min_len = min(logits.shape[1], tar.shape[1])
            loss = criterion(logits[:, :min_len, :].transpose(1, 2), tar[:, :min_len])
            total_loss += loss.item()
        final_loss = total_loss / 20
    
    print()
    print("=" * 70)
    print("  FINE-TUNING COMPLETADO")
    print("=" * 70)
    print(f"Loss inicial: {initial_loss:.4f}")
    print(f"Loss final:   {final_loss:.4f}")
    print(f"Cambio:       {((final_loss - initial_loss) / initial_loss * 100):+.1f}%")
    print()
    print(f"Modelo guardado: models/golden_seed_finetuned.pt")
    
    # Test con texto real
    print()
    print("Test con frases reales:")
    print("-" * 70)
    model.eval()
    test_phrases = [
        "me llamo enrique",
        "mi padre juan",
        "vivo en madrid",
        "me gusta bici",
    ]
    
    with torch.no_grad():
        for phrase in test_phrases:
            inp = text_to_ids(phrase, 16).unsqueeze(0).to(device)
            logits, metrics = model(inp, return_metrics=True)
            phi = metrics.get('phi', 0) if isinstance(metrics, dict) else 0
            print(f"  '{phrase}' -> PHI: {phi:.4f}")
    
    return model, history


if __name__ == "__main__":
    finetune()
