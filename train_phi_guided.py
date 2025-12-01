#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 ENTRENAMIENTO GUIADO POR PHI
================================

Optimiza el modelo para:
1. Predecir bien (loss de prediccion)
2. Maximizar PHI (integracion de informacion)

Usa el Golden Seed como base y lo mejora.
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
# CONFIGURACION
# =============================================================================

CONFIG = {
    'epochs': 800,
    'batch_size': 32,
    'lr': 2e-5,
    'phi_weight': 0.1,  # Peso de la perdida PHI
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

REAL_PHRASES = [
    "me llamo enrique garcia",
    "mi padre se llama juan",
    "mi madre es profesora",
    "tengo un hermano mayor",
    "me gusta el ciclismo",
    "hago descenso en bici",
    "mi bici es una scott",
    "peso setenta y cinco kilos",
    "vivo en madrid centro",
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
    "mi hermana estudia derecho",
    "mi abuelo tiene ochenta anos",
    "me gusta leer noches",
    "los fines de semana salgo",
    "mi coche es un seat leon",
    "estudio informatica online",
    "mi perro se llama toby",
    "tengo cita el martes diez",
    "mi cumpleanos es en mayo",
    "prefiero la playa montana",
    "el sabado hay partido futbol",
]


def text_to_ids(text, max_len=20):
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
# ENTRENAMIENTO
# =============================================================================

def train():
    print("=" * 70)
    print("  ENTRENAMIENTO GUIADO POR PHI")
    print("=" * 70)
    
    device = CONFIG['device']
    
    # Cargar Golden Seed
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
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    orig_loss = checkpoint['experiment_results']['iit_final_loss']
    orig_improvement = checkpoint['experiment_results']['improvement_percentage']
    print(f"Loss original: {orig_loss:.4f}")
    print(f"Mejora original: {orig_improvement:.1f}%")
    
    # Replay buffer
    replay_texts = load_replay_texts()
    all_texts = REAL_PHRASES + replay_texts
    print(f"Total textos entrenamiento: {len(all_texts)}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Baseline
    baseline = InfinitoV52Refactored(
        vocab_size=13, hidden_dim=64, num_layers=2, num_heads=4,
        use_improved_memory=False, use_improved_iit=False,
    ).to(device)
    baseline.load_state_dict(checkpoint['model_state_dict'], strict=False)
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad = False
    
    # Evaluar inicial
    model.eval()
    with torch.no_grad():
        total_loss, total_phi = 0, 0
        for _ in range(20):
            inp, tar = get_dyck_batch(32)
            inp, tar = inp.to(device), tar.to(device)
            logits, metrics = model(inp, return_metrics=True)
            min_len = min(logits.shape[1], tar.shape[1])
            loss = criterion(logits[:, :min_len, :].transpose(1, 2), tar[:, :min_len])
            total_loss += loss.item()
            total_phi += metrics.get('integration_phi', 0)
        initial_loss = total_loss / 20
        initial_phi = total_phi / 20
    
    print(f"\nMetricas iniciales:")
    print(f"  Loss Dyck: {initial_loss:.4f}")
    print(f"  PHI medio: {initial_phi:.4f}")
    
    best_loss = initial_loss
    best_phi = initial_phi
    history = []
    
    print("\nIniciando entrenamiento guiado por PHI...")
    print(f"  phi_weight = {CONFIG['phi_weight']}")
    print("-" * 70)
    
    for epoch in tqdm(range(CONFIG['epochs']), desc="Training"):
        model.train()
        
        # === Batch Dyck ===
        inp, tar = get_dyck_batch(CONFIG['batch_size'])
        inp, tar = inp.to(device), tar.to(device)
        
        optimizer.zero_grad()
        logits, metrics = model(inp, return_metrics=True)
        min_len = min(logits.shape[1], tar.shape[1])
        
        # Loss de prediccion
        loss_pred = criterion(logits[:, :min_len, :].transpose(1, 2), tar[:, :min_len])
        
        # Loss de PHI (queremos maximizarlo, asi que minimizamos -PHI)
        phi = metrics.get('integration_phi', 0)
        if isinstance(phi, torch.Tensor):
            loss_phi = -phi.mean()  # Negativo porque queremos maximizar
        else:
            loss_phi = torch.tensor(-phi, device=device)
        
        # Loss total
        total_loss = loss_pred + CONFIG['phi_weight'] * loss_phi
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        
        # === Batch texto (cada 3 epochs) ===
        if epoch % 3 == 0:
            inp_t, tar_t = get_text_batch(16)
            inp_t, tar_t = inp_t.to(device), tar_t.to(device)
            
            optimizer.zero_grad()
            logits_t, metrics_t = model(inp_t, return_metrics=True)
            min_len = min(logits_t.shape[1], tar_t.shape[1])
            loss_t = criterion(logits_t[:, :min_len, :].transpose(1, 2), tar_t[:, :min_len])
            
            phi_t = metrics_t.get('integration_phi', 0)
            if isinstance(phi_t, torch.Tensor):
                loss_phi_t = -phi_t.mean()
            else:
                loss_phi_t = torch.tensor(-phi_t, device=device)
            
            total_t = 0.2 * loss_t + 0.05 * loss_phi_t
            total_t.backward()
            optimizer.step()
        
        # Tracking
        history.append({
            'epoch': epoch,
            'loss': loss_pred.item(),
            'phi': phi.item() if isinstance(phi, torch.Tensor) else phi,
        })
        
        # Guardar mejor
        current_phi = phi.item() if isinstance(phi, torch.Tensor) else phi
        if loss_pred.item() < best_loss or current_phi > best_phi:
            if loss_pred.item() < best_loss:
                best_loss = loss_pred.item()
            if current_phi > best_phi:
                best_phi = current_phi
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': checkpoint['config'],
                'experiment_results': {
                    'iit_final_loss': loss_pred.item(),
                    'baseline_final_loss': checkpoint['experiment_results']['baseline_final_loss'],
                    'improvement_percentage': orig_improvement,
                    'phi_optimized': True,
                    'best_phi': best_phi,
                },
                'base_seed': checkpoint['base_seed'],
                'data_seed': checkpoint['data_seed'],
            }, 'models/phi_optimized_model.pt')
        
        if epoch > 0 and epoch % CONFIG['eval_every'] == 0:
            tqdm.write(f"Epoch {epoch:4d} | Loss: {loss_pred.item():.4f} | PHI: {current_phi:.4f} | "
                      f"Best Loss: {best_loss:.4f} | Best PHI: {best_phi:.4f}")
    
    # Evaluacion final
    model.eval()
    with torch.no_grad():
        total_loss, total_phi = 0, 0
        for _ in range(20):
            inp, tar = get_dyck_batch(32)
            inp, tar = inp.to(device), tar.to(device)
            logits, metrics = model(inp, return_metrics=True)
            min_len = min(logits.shape[1], tar.shape[1])
            loss = criterion(logits[:, :min_len, :].transpose(1, 2), tar[:, :min_len])
            total_loss += loss.item()
            total_phi += metrics.get('integration_phi', 0)
        final_loss = total_loss / 20
        final_phi = total_phi / 20
    
    print()
    print("=" * 70)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    print(f"{'Metrica':<20} {'Inicial':>12} {'Final':>12} {'Cambio':>12}")
    print("-" * 70)
    print(f"{'Loss Dyck':<20} {initial_loss:>12.4f} {final_loss:>12.4f} {((final_loss-initial_loss)/initial_loss*100):>+11.1f}%")
    print(f"{'PHI medio':<20} {initial_phi:>12.4f} {final_phi:>12.4f} {((final_phi-initial_phi)/initial_phi*100):>+11.1f}%")
    print()
    print(f"Modelo guardado: models/phi_optimized_model.pt")
    
    # Test con texto real
    print()
    print("Test con frases reales:")
    print("-" * 70)
    test_phrases = ["me llamo enrique", "mi padre juan", "vivo en madrid", "me gusta bici"]
    
    with torch.no_grad():
        for phrase in test_phrases:
            inp = text_to_ids(phrase, 16).unsqueeze(0).to(device)
            _, metrics = model(inp, return_metrics=True)
            phi = metrics.get('integration_phi', 0)
            coh = metrics.get('coherence', 0)
            print(f"  '{phrase}' -> PHI: {phi:.4f}, Coherence: {coh:.4f}")
    
    return model, history


if __name__ == "__main__":
    train()
