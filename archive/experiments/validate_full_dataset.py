#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_full_dataset.py

Validación con DATASET COMPLETO (1 época) para verificar Val PPL realista.
Usa la mejor configuración encontrada: HIGHER_LR (lr 5e-4).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import json
from datetime import datetime

from src.infinito_v5_2_refactored import InfinitoV52Refactored


def prepare_full_data(tokenizer, seq_len=256, batch_size=16):
    """Prepara dataset COMPLETO de WikiText-2."""
    print("Cargando WikiText-2 COMPLETO...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len)
    
    print("Tokenizando dataset completo (esto toma ~2-3 minutos)...")
    tokenized_train = dataset['train'].map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_val = dataset['validation'].map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Filtrar secuencias muy cortas
    tokenized_train = tokenized_train.filter(lambda x: len(x['input_ids']) > 10)
    tokenized_val = tokenized_val.filter(lambda x: len(x['input_ids']) > 10)
    
    print(f"Train samples: {len(tokenized_train)}")
    print(f"Val samples: {len(tokenized_val)}")
    
    def collate_fn(batch):
        input_ids = [torch.tensor(item['input_ids'][:seq_len]) for item in batch]
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = torch.zeros(len(input_ids), max_len, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            padded_ids[i, :len(ids)] = ids
        
        # Labels = shifted input_ids (standard LM objective)
        labels = padded_ids.clone()
        labels[:, :-1] = padded_ids[:, 1:]
        labels[:, -1] = tokenizer.pad_token_id
        
        return padded_ids, labels
    
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device, lambda_phi=0.3):
    """Entrena 1 época completa."""
    model.train()
    total_loss = 0
    total_loss_lm = 0
    total_loss_phi = 0
    num_batches = 0
    
    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits, metrics = model(input_ids, return_metrics=True)
        
        # Loss LM
        loss_lm = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )
        
        # Loss PHI
        loss_phi = 0.0
        if metrics and 'delta_phi_loss' in metrics:
            loss_phi = metrics['delta_phi_loss']
            if isinstance(loss_phi, float):
                loss_phi = torch.tensor(loss_phi, device=device)
        
        # Loss total
        if isinstance(loss_phi, torch.Tensor) and loss_phi.item() > 0:
            loss = loss_lm + lambda_phi * loss_phi
        else:
            loss = loss_lm
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_loss_lm += loss_lm.item()
        if isinstance(loss_phi, torch.Tensor):
            total_loss_phi += loss_phi.item()
        num_batches += 1
        
        # Progress
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = total_loss_lm / num_batches
            avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
            print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f}")
    
    avg_train_loss = total_loss_lm / num_batches
    avg_train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
    
    return avg_train_loss, avg_train_ppl


def validate(model, val_loader, criterion, device):
    """Validación."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits, _ = model(input_ids, return_metrics=True)
            
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_val_loss = total_loss / num_batches
    avg_val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
    
    return avg_val_loss, avg_val_ppl


def main():
    """Validación con dataset completo."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}\n")
    
    # Configuración ganadora: HIGHER_LR
    config = {
        'name': 'HIGHER_LR (lr 5e-4) - DATASET COMPLETO',
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.15,
        'lr': 5e-4,
        'lambda_phi': 0.3,
        'batch_size': 16,
        'seq_len': 256,
        'weight_decay': 0.01
    }
    
    print("="*80)
    print("VALIDACIÓN CON DATASET COMPLETO - 1 ÉPOCA")
    print("="*80)
    print("Configuración:")
    for key, val in config.items():
        if key != 'name':
            print(f"  {key}: {val}")
    print("="*80)
    print("")
    
    # Tokenizer
    print("Cargando tokenizer GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    train_loader, val_loader = prepare_full_data(
        tokenizer,
        seq_len=config['seq_len'],
        batch_size=config['batch_size']
    )
    
    print(f"\nBatches por época: {len(train_loader)} (train), {len(val_loader)} (val)")
    print(f"Tiempo estimado: ~30-45 minutos\n")
    
    # Modelo
    print("Creando modelo...")
    model = InfinitoV52Refactored(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_improved_iit=True,
        use_improved_memory=True,
        use_learnable_phi=True,
        lambda_phi=config['lambda_phi']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parámetros: {total_params:,}\n")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("="*80)
    print("ENTRENANDO 1 ÉPOCA COMPLETA...")
    print("="*80)
    train_loss, train_ppl = train_one_epoch(
        model, train_loader, optimizer, criterion, device, config['lambda_phi']
    )
    
    # Validate
    print("\nVALIDANDO...")
    val_loss, val_ppl = validate(model, val_loader, criterion, device)
    
    # Results
    results = {
        'config': config,
        'train_loss': train_loss,
        'train_ppl': train_ppl,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
        'ratio': val_ppl / train_ppl if train_ppl > 0 else 0,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print("\n" + "="*80)
    print("RESULTADOS FINALES - DATASET COMPLETO")
    print("="*80)
    print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
    print(f"  Ratio:      {results['ratio']:.2f}x")
    print("="*80)
    
    # Comparación con baseline anterior
    print("\nCOMPARACIÓN CON BASELINE:")
    print(f"  Baseline anterior: Val PPL 203.08 (epoch 16, hidden 512, dropout 0.3)")
    print(f"  Este experimento:  Val PPL {val_ppl:.2f}")
    if val_ppl < 203:
        mejora = ((203 - val_ppl) / 203) * 100
        print(f"  MEJORA: {mejora:.1f}% mejor!")
    else:
        print(f"  Peor que baseline (necesita más épocas)")
    
    # Guardar
    results_file = f'full_dataset_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados guardados en: {results_file}")
    
    # Guardar checkpoint
    checkpoint_path = 'infinito_v5.2_validated_1epoch.pt'
    torch.save({
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_ppl': train_ppl,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
        'config': config
    }, checkpoint_path)
    
    print(f"Checkpoint guardado: {checkpoint_path}")
    
    # Recomendación
    print("\n" + "="*80)
    print("RECOMENDACIÓN:")
    print("="*80)
    if val_ppl < 150:
        print("  Val PPL < 150 - Configuración ÓPTIMA")
        print("  Lanzar entrenamiento largo (20 épocas) con esta config")
    elif val_ppl < 200:
        print("  Val PPL entre 150-200 - Configuración BUENA")
        print("  Puede mejorar con más épocas, lanzar entrenamiento largo")
    else:
        print("  Val PPL > 200 - Necesita ajustes")
        print("  Considerar aumentar LR a 1e-3 o reducir dropout a 0.1")


if __name__ == '__main__':
    main()
