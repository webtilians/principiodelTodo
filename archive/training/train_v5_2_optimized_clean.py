#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_v5_2_optimized.py

Entrenamiento OPTIMIZADO para reducir Val PPL de 208  100-150

MEJORAS IMPLEMENTADAS:
1. Dropout reducido: 0.3  0.15 (mejor balance regularizacin/aprendizaje)
2. Hidden dim aumentado: 512  640 (ms capacidad sin explotar memoria)
3. Label smoothing: 0.1 (reduce overconfidence)
4. Gradient accumulation: 2 steps (simula batch_size=32)
5. Warmup mejorado: 500 steps (estabiliza inicio)
6. Weight decay optimizado: 0.01  0.05
7. Learning rate ajustado: 1e-4  2e-4 con warmup
8. Early stopping con patience=5 (evita overfitting)

BASELINE:
- Train PPL: 31, Val PPL: 203 (ratio 6.5x = overfitting)

OBJETIVO:
- Val PPL: 100-150 (ratio <3x = overfitting controlado)
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
from datasets import load_dataset


class LabelSmoothingLoss(nn.Module):
    """Cross Entropy con Label Smoothing para reducir overconfidence."""
    
    def __init__(self, vocab_size, smoothing=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        pred: [batch*seq, vocab_size] logits
        target: [batch*seq] tokens
        """
        # Log softmax para estabilidad numrica
        log_probs = torch.log_softmax(pred, dim=-1)
        
        # True label loss
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        # Smooth loss (distribucin uniforme)
        smooth_loss = -log_probs.mean(dim=-1)
        
        # Combinar
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class OptimizedTrainer:
    """Trainer con optimizaciones anti-overfitting."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=2e-4,
        device='cuda',
        warmup_steps=500,
        label_smoothing=0.1,
        grad_accumulation_steps=2,
        weight_decay=0.05,
        patience=5,
        vocab_size=50257,
        lambda_phi=0.3
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_accumulation_steps = grad_accumulation_steps
        self.patience = patience
        self.lambda_phi = lambda_phi  # Guardar para usar en loss
        
        # Optimizer con weight decay ms fuerte
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=weight_decay
        )
        
        # Loss con label smoothing
        self.criterion = LabelSmoothingLoss(
            vocab_size=vocab_size,
            smoothing=label_smoothing
        )
        
        # Warmup + Cosine decay
        self.warmup_steps = warmup_steps
        self.total_steps = 0
        
        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'train_ppl': [],
            'val_loss': [],
            'val_ppl': [],
            'train_phi': [],
            'learning_rate': []
        }
        
    def get_lr(self):
        """Learning rate con warmup + cosine decay."""
        base_lr = self.optimizer.param_groups[0]['lr']
        
        if self.total_steps < self.warmup_steps:
            # Linear warmup: empezar en 1e-7, subir hasta base_lr
            min_lr = 1e-7
            warmup_progress = self.total_steps / max(1, self.warmup_steps)
            return min_lr + (base_lr - min_lr) * warmup_progress
        else:
            # Cosine decay despus de warmup
            progress = (self.total_steps - self.warmup_steps) / (len(self.train_loader) * 20 - self.warmup_steps)
            progress = min(1.0, progress)
            return self.optimizer.param_groups[0]['lr'] * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
    def train_epoch(self, epoch):
        """Entrena una poca con gradient accumulation."""
        self.model.train()
        total_loss = 0
        total_loss_lm = 0
        total_loss_phi = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (input_ids, labels) in enumerate(self.train_loader):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            logits, metrics = self.model(input_ids, return_metrics=True)
            
            # Loss LM con label smoothing
            loss_lm = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            
            # Loss PHI (si est disponible)
            loss_phi = 0.0
            if metrics and 'delta_phi_loss' in metrics:
                loss_phi = metrics['delta_phi_loss']
                if isinstance(loss_phi, float):
                    loss_phi = torch.tensor(loss_phi, device=self.device)
            
            # Loss total con lambda-phi
            if isinstance(loss_phi, torch.Tensor) and loss_phi.item() > 0:
                loss = loss_lm + self.lambda_phi * loss_phi
            else:
                loss = loss_lm
            
            # Normalizar por gradient accumulation
            loss = loss / self.grad_accumulation_steps
            loss.backward()
            
            # Actualizar cada N steps
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update con LR schedule
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.get_lr()
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.total_steps += 1
            
            # Tracking
            total_loss += loss.item() * self.grad_accumulation_steps
            total_loss_lm += loss_lm.item()
            if isinstance(loss_phi, torch.Tensor):
                total_loss_phi += loss_phi.item()
            num_batches += 1
            
            # Log cada 50 batches
            if batch_idx % 50 == 0:
                avg_loss = total_loss / num_batches
                avg_ppl = torch.exp(torch.tensor(total_loss_lm / num_batches)).item()
                current_lr = self.get_lr()
                print(f"  Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {avg_loss:.4f} | PPL: {avg_ppl:.2f} | LR: {current_lr:.2e}")
        
        # Promedios finales
        avg_train_loss = total_loss / num_batches
        avg_train_loss_lm = total_loss_lm / num_batches
        avg_train_ppl = torch.exp(torch.tensor(avg_train_loss_lm)).item()
        
        # Calcular Train PHI promedio
        avg_train_phi = 0.0
        if hasattr(self.model, 'learnable_phi_weights') and self.model.learnable_phi_weights is not None:
            # Hacer un forward pass de validacin para obtener PHI
            with torch.no_grad():
                sample_input, _ = next(iter(self.train_loader))
                sample_input = sample_input.to(self.device)
                _, sample_metrics = self.model(sample_input, return_metrics=True)
                if 'phi_estimate' in sample_metrics:
                    avg_train_phi = sample_metrics['phi_estimate'].mean().item()
        
        return avg_train_loss, avg_train_ppl, avg_train_phi
    
    def validate(self):
        """Validacin."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, labels in self.val_loader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                logits, _ = self.model(input_ids, return_metrics=True)
                
                # Solo loss LM en validacin (sin PHI)
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        avg_val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
        
        return avg_val_loss, avg_val_ppl
    
    def train(self, epochs):
        """Loop de entrenamiento completo."""
        print(f"\n{'='*80}")
        print(f"ENTRENAMIENTO OPTIMIZADO - {epochs} pocas")
        print(f"{'='*80}")
        print(f"Configuracin:")
        print(f"  - Dropout: {self.model.dropout}")
        print(f"  - Hidden dim: {self.model.hidden_dim}")
        print(f"  - Label smoothing: {self.criterion.smoothing}")
        print(f"  - Weight decay: {self.optimizer.param_groups[0]['weight_decay']}")
        print(f"  - Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"  - Warmup steps: {self.warmup_steps}")
        print(f"  - Gradient accumulation: {self.grad_accumulation_steps}")
        print(f"  - Early stopping patience: {self.patience}")
        print(f"{'='*80}\n")
        
        for epoch in range(epochs):
            print(f"\n{'='*80}")
            print(f"POCA {epoch+1}/{epochs}")
            print(f"{'='*80}")
            
            # Entrenar
            train_loss, train_ppl, train_phi = self.train_epoch(epoch)
            
            # Validar
            val_loss, val_ppl = self.validate()
            
            # Guardar mtricas
            self.history['train_loss'].append(train_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_loss'].append(val_loss)
            self.history['val_ppl'].append(val_ppl)
            self.history['train_phi'].append(train_phi)
            self.history['learning_rate'].append(self.get_lr())
            
            # Log
            print(f"\n Resultados poca {epoch+1}:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
            print(f"  Train PHI:  {train_phi:.4f}")
            print(f"  LR:         {self.get_lr():.2e}")
            print(f"  Ratio PPL:  {val_ppl/train_ppl:.2f}x")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                # Guardar mejor modelo
                checkpoint_path = 'infinito_v5.2_optimized_best.pt'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'train_phi': train_phi
                }, checkpoint_path)
                print(f"   Mejor modelo guardado (Val PPL: {val_ppl:.2f})")
            else:
                self.epochs_without_improvement += 1
                print(f"   Sin mejora ({self.epochs_without_improvement}/{self.patience})")
                
                if self.epochs_without_improvement >= self.patience:
                    print(f"\n Early stopping activado (patience={self.patience})")
                    break
        
        # Guardar historial
        history_path = f'training_history_optimized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f" ENTRENAMIENTO COMPLETADO")
        print(f"{'='*80}")
        print(f"Mejor Val PPL: {torch.exp(torch.tensor(self.best_val_loss)).item():.2f}")
        print(f"Historial guardado: {history_path}")
        

def prepare_data(tokenizer, seq_len=256, batch_size=16):
    """Prepara datasets WikiText-2."""
    print(" Cargando WikiText-2...")
    # Intentar cargar desde caché local primero
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', download_mode='reuse_cache_if_exists')
    except:
        # Si falla, intentar sin especificar versión específica
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=seq_len)
    
    print(" Tokenizando...")
    tokenized_train = dataset['train'].map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_val = dataset['validation'].map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Filtrar secuencias vacas
    tokenized_train = tokenized_train.filter(lambda x: len(x['input_ids']) > 10)
    tokenized_val = tokenized_val.filter(lambda x: len(x['input_ids']) > 10)
    
    # Convertir a tensores
    def collate_fn(batch):
        input_ids = [torch.tensor(item['input_ids'][:seq_len]) for item in batch]
        # Pad
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = torch.zeros(len(input_ids), max_len, dtype=torch.long)
        for i, ids in enumerate(input_ids):
            padded_ids[i, :len(ids)] = ids
        
        # Labels = input_ids shifted
        labels = padded_ids.clone()
        
        return padded_ids, labels
    
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f" Train samples: {len(tokenized_train)}")
    print(f" Val samples: {len(tokenized_val)}")
    
    return train_loader, val_loader


def main():
    import argparse
    from transformers import GPT2Tokenizer
    
    parser = argparse.ArgumentParser(description='Entrenamiento optimizado INFINITO V5.2')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--dropout', type=float, default=0.15, help='Reducido de 0.3')
    parser.add_argument('--hidden-dim', type=int, default=640, help='Aumentado de 512')
    parser.add_argument('--num-layers', type=int, default=6)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--lambda-phi', type=float, default=0.3)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--grad-accum', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint para resumir')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")
    
    # Tokenizer
    print(" Cargando tokenizer GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    train_loader, val_loader = prepare_data(tokenizer, args.seq_len, args.batch_size)
    
    # Modelo
    print(f"\n Creando modelo optimizado...")
    model = InfinitoV52Refactored(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_improved_iit=True,
        use_improved_memory=True,
        use_learnable_phi=True,
        lambda_phi=args.lambda_phi
    )
    
    # Resumir si se especifica
    if args.resume:
        print(f" Cargando checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        # Cargar con strict=False para ignorar keys incompatibles
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"   poca {checkpoint.get('epoch', 'N/A')}, Val PPL: {checkpoint.get('val_ppl', 'N/A'):.2f}")
        print("   [OK] Checkpoint cargado (strict=False)")
    
    # Contar parmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parmetros: {total_params:,}")
    print(f"  Entrenables: {trainable_params:,}")
    
    # Trainer
    trainer = OptimizedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        device=device,
        warmup_steps=args.warmup_steps,
        label_smoothing=args.label_smoothing,
        grad_accumulation_steps=args.grad_accum,
        weight_decay=args.weight_decay,
        patience=args.patience,
        vocab_size=tokenizer.vocab_size,
        lambda_phi=args.lambda_phi
    )
    
    # Entrenar
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()

