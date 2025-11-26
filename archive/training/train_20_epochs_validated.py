#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_20_epochs_validated.py

Entrenamiento de 20 épocas usando la configuración VALIDADA que dio Val PPL 18.99
EXACTAMENTE el mismo código que quick_experiment.py pero con 20 épocas y dataset completo.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from transformers import GPT2Tokenizer
from datasets import load_dataset

from src.infinito_v5_2_refactored import InfinitoV52Refactored


def prepare_data_full(tokenizer, seq_len=256, batch_size=16):
    """Prepara DATASET COMPLETO - igual que en quick_experiment.py pero sin límite de samples."""
    print("Cargando WikiText-2 completo...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len)
    
    print("Tokenizando...")
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
        labels = padded_ids.clone()
        return padded_ids, labels
    
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    return train_loader, val_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device, lambda_phi=0.3):
    """Entrena una época - EXACTO como en quick_experiment.py."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (input_ids, labels) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, metrics = model(input_ids, return_metrics=True)
        
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
        
        if isinstance(loss_phi, torch.Tensor) and loss_phi.item() > 0:
            loss = loss_lm + lambda_phi * loss_phi
        else:
            loss = loss_lm
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = total_loss / num_batches
            avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
            print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f}")
    
    train_loss = total_loss / num_batches
    train_ppl = torch.exp(torch.tensor(train_loss)).item()
    
    return train_loss, train_ppl


def validate(model, val_loader, criterion, device):
    """Valida el modelo - EXACTO como en quick_experiment.py."""
    model.eval()
    val_loss = 0
    val_batches = 0
    
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits, _ = model(input_ids, return_metrics=True)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            val_loss += loss.item()
            val_batches += 1
    
    val_loss = val_loss / val_batches
    val_ppl = torch.exp(torch.tensor(val_loss)).item()
    
    return val_loss, val_ppl


def main():
    # CONFIGURACIÓN EXACTA que dio Val PPL 18.99
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    print("\n" + "="*80)
    print("ENTRENAMIENTO 20 ÉPOCAS - CONFIGURACIÓN VALIDADA")
    print("="*80)
    for key, val in config.items():
        if key != 'name':
            print(f"  {key}: {val}")
    print("="*80 + "\n")
    
    # Tokenizer
    print("Cargando tokenizer GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    train_loader, val_loader = prepare_data_full(
        tokenizer, 
        seq_len=config['seq_len'],
        batch_size=config['batch_size']
    )
    
    # Modelo - EXACTO como en quick_experiment.py
    print("\nCreando modelo...")
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
    print(f"Total parámetros: {total_params/1e6:.1f}M")
    
    # Optimizer - EXACTO como en quick_experiment.py
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Loss - EXACTO como en quick_experiment.py
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n" + "="*80)
    print("INICIANDO ENTRENAMIENTO")
    print("="*80 + "\n")
    
    best_val_ppl = float('inf')
    patience = 5
    patience_counter = 0
    history = []
    
    for epoch in range(1, 21):  # 20 épocas
        print(f"\nÉpoca {epoch}/20")
        print("-" * 80)
        
        # Train
        train_loss, train_ppl = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config['lambda_phi']
        )
        
        # Validate
        val_loss, val_ppl = validate(model, val_loader, criterion, device)
        
        # Log
        ratio = val_ppl / train_ppl if train_ppl > 0 else 0
        print(f"\n{'='*80}")
        print(f"ÉPOCA {epoch} COMPLETADA")
        print(f"{'='*80}")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        print(f"  Ratio:      {ratio:.2f}x")
        print(f"{'='*80}\n")
        
        # Guardar historial
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_ppl': train_ppl,
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'ratio': ratio
        })
        
        # Early stopping y checkpoint
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            patience_counter = 0
            
            checkpoint_path = f'infinito_v5.2_best_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ppl': val_ppl,
                'train_ppl': train_ppl,
                'config': config
            }, checkpoint_path)
            print(f"✓ Mejor modelo guardado: {checkpoint_path} (Val PPL: {val_ppl:.2f})")
        else:
            patience_counter += 1
            print(f"⚠ Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping en época {epoch}")
                break
    
    # Guardar historial completo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = f'training_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump({
            'config': config,
            'history': history,
            'best_val_ppl': best_val_ppl
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"Mejor Val PPL: {best_val_ppl:.2f}")
    print(f"Historial guardado: {history_path}")
    print("="*80)


if __name__ == '__main__':
    main()
