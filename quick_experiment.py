#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
quick_experiment.py

Experimentos RÁPIDOS (1 época) para encontrar configuración óptima.
Prueba diferentes combinaciones de LR, warmup, dropout, etc.
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


def prepare_data_fast(tokenizer, seq_len=128, batch_size=16, max_samples=5000):
    """Prepara subset pequeño de datos para experimentación rápida."""
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len)
    
    # Tomar solo primeros N samples
    train_small = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
    val_small = dataset['validation'].select(range(min(1000, len(dataset['validation']))))
    
    tokenized_train = train_small.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_val = val_small.map(tokenize_function, batched=True, remove_columns=['text'])
    
    tokenized_train = tokenized_train.filter(lambda x: len(x['input_ids']) > 10)
    tokenized_val = tokenized_val.filter(lambda x: len(x['input_ids']) > 10)
    
    def collate_fn(batch):
        input_ids = [torch.tensor(item['input_ids'][:seq_len]) for item in batch]
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = torch.zeros(len(input_ids), max_len, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            padded_ids[i, :len(ids)] = ids
        labels = padded_ids.clone()
        return padded_ids, labels
    
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader


def run_quick_experiment(config, device='cuda'):
    """Ejecuta 1 época con la configuración dada."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENTO: {config['name']}")
    print(f"{'='*80}")
    for key, val in config.items():
        if key != 'name':
            print(f"  {key}: {val}")
    print(f"{'='*80}\n")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    train_loader, val_loader = prepare_data_fast(
        tokenizer, 
        seq_len=config.get('seq_len', 128),
        batch_size=config.get('batch_size', 16),
        max_samples=config.get('max_samples', 5000)
    )
    
    # Modelo
    model = InfinitoV52Refactored(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        use_improved_iit=True,
        use_improved_memory=True,
        use_learnable_phi=config.get('use_learnable_phi', True),
        lambda_phi=config.get('lambda_phi', 0.3)
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Train 1 epoch
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
            loss = loss_lm + config.get('lambda_phi', 0.3) * loss_phi
        else:
            loss = loss_lm
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0 and batch_idx > 0:
            avg_loss = total_loss / num_batches
            avg_ppl = torch.exp(torch.tensor(avg_loss)).item()
            print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f}")
    
    # Train metrics
    train_loss = total_loss / num_batches
    train_ppl = torch.exp(torch.tensor(train_loss)).item()
    
    # Validation
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
    
    # Results
    results = {
        'config': config,
        'train_loss': train_loss,
        'train_ppl': train_ppl,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
        'ratio': val_ppl / train_ppl if train_ppl > 0 else 0
    }
    
    print(f"\n{'='*80}")
    print(f"RESULTADOS: {config['name']}")
    print(f"{'='*80}")
    print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
    print(f"  Ratio:      {results['ratio']:.2f}x")
    print(f"{'='*80}\n")
    
    return results


def main():
    """Ejecuta múltiples experimentos rápidos."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Experimentos a probar
    experiments = [
        {
            'name': 'BASELINE (dropout 0.15, lr 2e-4, hidden 512)',
            'hidden_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.15,
            'lr': 2e-4,
            'lambda_phi': 0.3,
            'batch_size': 16,
            'seq_len': 128,
            'max_samples': 5000,
            'weight_decay': 0.01
        },
        {
            'name': 'HIGHER_LR (lr 5e-4)',
            'hidden_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.15,
            'lr': 5e-4,
            'lambda_phi': 0.3,
            'batch_size': 16,
            'seq_len': 128,
            'max_samples': 5000,
            'weight_decay': 0.01
        },
        {
            'name': 'LOWER_DROPOUT (dropout 0.1)',
            'hidden_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.1,
            'lr': 2e-4,
            'lambda_phi': 0.3,
            'batch_size': 16,
            'seq_len': 128,
            'max_samples': 5000,
            'weight_decay': 0.01
        },
        {
            'name': 'BIGGER_MODEL (hidden 640)',
            'hidden_dim': 640,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.15,
            'lr': 2e-4,
            'lambda_phi': 0.3,
            'batch_size': 16,
            'seq_len': 128,
            'max_samples': 5000,
            'weight_decay': 0.01
        },
        {
            'name': 'OPTIMAL_COMBO (hidden 640, dropout 0.1, lr 3e-4)',
            'hidden_dim': 640,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.1,
            'lr': 3e-4,
            'lambda_phi': 0.3,
            'batch_size': 16,
            'seq_len': 128,
            'max_samples': 5000,
            'weight_decay': 0.01
        }
    ]
    
    # Ejecutar experimentos
    all_results = []
    for exp in experiments:
        try:
            result = run_quick_experiment(exp, device)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR en experimento {exp['name']}: {e}")
            continue
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'quick_experiments_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Resumen
    print(f"\n{'='*80}")
    print("RESUMEN DE TODOS LOS EXPERIMENTOS")
    print(f"{'='*80}")
    print(f"{'Experimento':<50} {'Val PPL':<12} {'Ratio':<8}")
    print(f"{'-'*80}")
    
    for res in all_results:
        name = res['config']['name']
        val_ppl = res['val_ppl']
        ratio = res['ratio']
        print(f"{name:<50} {val_ppl:<12.2f} {ratio:<8.2f}x")
    
    print(f"{'='*80}")
    print(f"Resultados guardados en: {results_file}")
    
    # Mejor experimento
    best = min(all_results, key=lambda x: x['val_ppl'])
    print(f"\n MEJOR CONFIGURACIÓN:")
    print(f"  {best['config']['name']}")
    print(f"  Val PPL: {best['val_ppl']:.2f}")
    print(f"  Ratio: {best['ratio']:.2f}x")


if __name__ == '__main__':
    main()
