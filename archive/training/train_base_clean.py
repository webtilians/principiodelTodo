#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento base INFINITO - 30 épocas optimizado
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
import time
import json
from pathlib import Path

from train_v5_2_gpt2_lora import InfinitoGPT2Hybrid

def train_base_model(
    epochs=30,
    batch_size=16,
    lr=4e-4,
    lambda_phi=0.1,
    patience=5,
    checkpoint_dir="models/checkpoints",
    log_file="training_improved_base.log"
):
    """Entrena el modelo base INFINITO por 30 épocas"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("ENTRENAMIENTO MODELO BASE INFINITO - 30 EPOCAS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Lambda PHI: {lambda_phi}")
    print("="*70)
    print()
    
    # Crear directorio para checkpoints
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar modelo
    print("Cargando modelo INFINITO V5.2...")
    model = InfinitoGPT2Hybrid(
        use_lora=True,
        lora_r=4,
        lora_alpha=16,
        lambda_phi=lambda_phi
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 2. Cargar datos
    print("\nCargando WikiText-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    val_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )
    
    print("   Tokenizando dataset...")
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    train_loader = DataLoader(tokenized, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # 3. Optimizer y scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\nOptimizer configurado:")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Warmup steps: {warmup_steps:,}")
    
    # 4. Training loop
    print("\n" + "="*70)
    print("INICIANDO ENTRENAMIENTO")
    print("="*70)
    print()
    
    training_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        total_text_loss = 0
        total_phi_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, metrics = model(input_ids=input_ids, return_metrics=True)
            
            # Calculate losses
            labels = input_ids
            text_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            
            # PHI loss
            delta_phi = metrics['delta_phi_loss']
            if isinstance(delta_phi, float):
                delta_phi = torch.tensor(delta_phi, device=device)
            
            phi_loss = lambda_phi * delta_phi
            loss = text_loss + phi_loss
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            total_text_loss += text_loss.item()
            total_phi_loss += phi_loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_text_loss = 0
        val_phi_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                
                logits, metrics = model(input_ids=input_ids, return_metrics=True)
                
                labels = input_ids
                text_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
                
                delta_phi = metrics['delta_phi_loss']
                if isinstance(delta_phi, float):
                    delta_phi = torch.tensor(delta_phi, device=device)
                
                phi_loss = lambda_phi * delta_phi
                loss = text_loss + phi_loss
                
                val_loss += loss.item()
                val_text_loss += text_loss.item()
                val_phi_loss += phi_loss.item()
        
        # Metrics
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_ppl = torch.exp(torch.tensor(total_text_loss / len(train_loader))).item()
        avg_val_ppl = torch.exp(torch.tensor(val_text_loss / len(val_loader))).item()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{epochs} completado:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train PPL: {avg_train_ppl:.2f}")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val PPL: {avg_val_ppl:.2f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_ppl': avg_train_ppl,
            'val_ppl': avg_val_ppl,
            'time': epoch_time
        })
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_ppl': avg_val_ppl
            }, f"{checkpoint_dir}/infinito_base_improved_best.pt")
            
            print(f"  [BEST MODEL] Guardado en {checkpoint_dir}/infinito_base_improved_best.pt")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_ppl': avg_val_ppl
            }, f"{checkpoint_dir}/infinito_base_improved_epoch{epoch+1}.pt")
        
        # Save history
        with open('training_history_improved.json', 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print()
    
    total_time = time.time() - start_time
    
    print("="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"Total epochs: {len(training_history)}")
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val PPL: {min(h['val_ppl'] for h in training_history):.2f}")
    print(f"Modelo guardado en: {checkpoint_dir}/infinito_base_improved_best.pt")
    print("="*70)
    
    return training_history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo base INFINITO mejorado')
    parser.add_argument('--epochs', type=int, default=30, help='Numero de epocas')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--lambda-phi', type=float, default=0.1, help='Peso de PHI')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    
    history = train_base_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_phi=args.lambda_phi,
        patience=args.patience
    )
