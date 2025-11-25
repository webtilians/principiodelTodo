#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrenamiento OFFLINE - sin dependencias de red HuggingFace
"""
import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from pathlib import Path

# Importar modelo
from src.infinito_v5_2_refactored import InfinitoV52Refactored
from transformers import GPT2Tokenizer
from datasets import load_dataset


class LabelSmoothingLoss(nn.Module):
    """Cross Entropy con Label Smoothing."""
    
    def __init__(self, vocab_size, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        log_probs = torch.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-7):
    """Cosine decay con warmup mejorado."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup lineal desde min_lr hasta base_lr
            progress = current_step / max(1, num_warmup_steps)
            return min_lr + (1.0 - min_lr) * progress
        # Cosine decay después del warmup
        progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prepare_data_offline(tokenizer, seq_len=256, batch_size=16):
    """Prepara datos desde caché offline."""
    print(" Cargando WikiText-2 (modo offline)...")
    
    # Cargar desde caché usando load_dataset en modo offline
    dataset = load_dataset(
        'wikitext', 
        'wikitext-2-raw-v1',
        cache_dir=str(Path.home() / '.cache' / 'huggingface' / 'datasets')
    )
    print(f"   ✓ Dataset cargado desde caché local")
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len)
    
    print(" Tokenizando...")
    tokenized_train = dataset['train'].map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_val = dataset['validation'].map(tokenize_function, batched=True, remove_columns=['text'])
    
    def collate(batch):
        input_ids = [item['input_ids'] for item in batch]
        max_len = min(seq_len, max(len(ids) for ids in input_ids))
        
        padded_inputs = []
        padded_targets = []
        
        for ids in input_ids:
            if len(ids) < 2:
                continue
            ids = ids[:max_len]
            padded_ids = ids + [tokenizer.pad_token_id] * (max_len - len(ids))
            padded_inputs.append(padded_ids[:-1])
            padded_targets.append(padded_ids[1:])
        
        if not padded_inputs:
            return None, None
            
        return (
            torch.tensor(padded_inputs, dtype=torch.long),
            torch.tensor(padded_targets, dtype=torch.long)
        )
    
    train_loader = DataLoader(
        tokenized_train, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate,
        num_workers=0
    )
    
    val_loader = DataLoader(
        tokenized_val, 
        batch_size=batch_size, 
        collate_fn=collate,
        num_workers=0
    )
    
    # Calcular tamaños reales
    train_samples = sum(1 for _ in train_loader) * batch_size
    val_samples = sum(1 for _ in val_loader) * batch_size
    print(f" Train samples: {train_samples}")
    print(f" Val samples: {val_samples}")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, grad_accum_steps=2):
    """Entrena una época."""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if inputs is None:
            continue
            
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _, _ = model(inputs)
        
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss = loss / grad_accum_steps
        loss.backward()
        
        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Valida el modelo."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            if inputs is None:
                continue
                
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _, _ = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--lambda-phi', type=float, default=0.3)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Cargar tokenizer desde caché
    print(" Cargando tokenizer GPT-2 (offline)...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Preparar datos
    train_loader, val_loader = prepare_data_offline(tokenizer, args.seq_len, args.batch_size)
    
    # Crear modelo
    print("\n Creando modelo...")
    model = InfinitoV52Refactored(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        dropout=args.dropout,
        lambda_phi=args.lambda_phi
    ).to(device)
    
    # Cargar checkpoint si existe
    start_epoch = 1
    best_val_ppl = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        print(f"\n Cargando checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_ppl = checkpoint.get('val_ppl', float('inf'))
        print(f"   Continuando desde época {start_epoch}")
        print(f"   Mejor Val PPL anterior: {best_val_ppl:.2f}")
    
    # Parámetros
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Total parámetros: {total_params/1e6:.1f}M")
    
    # Optimizador y scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = LabelSmoothingLoss(tokenizer.vocab_size, smoothing=args.label_smoothing)
    
    # Training loop
    print(f"\n INICIANDO ENTRENAMIENTO")
    print("=" * 70)
    
    patience_counter = 0
    history = []
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, args.grad_accum)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Métricas
        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        epoch_time = time.time() - epoch_start
        
        # Log
        print(f"\nÉpoca {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        print(f"  Ratio: {val_ppl/train_ppl:.2f}x")
        
        # Guardar historial
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_ppl': train_ppl,
            'val_ppl': val_ppl,
            'lr': scheduler.get_last_lr()[0]
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
                'config': vars(args)
            }, checkpoint_path)
            print(f"  ✓ Mejor modelo guardado: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"\n Early stopping en época {epoch}")
                break
    
    # Guardar historial
    history_path = f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n ENTRENAMIENTO COMPLETADO")
    print(f"Mejor Val PPL: {best_val_ppl:.2f}")
    print(f"Historial: {history_path}")


if __name__ == '__main__':
    main()
