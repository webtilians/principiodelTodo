#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ENTRENAMIENTO MEJORADO: INFINITO V5.2 con WikiText-2 REAL
==============================================================

Script de entrenamiento MEJORADO usando:
- WikiText-2 REAL (HuggingFace datasets)
- GPT2Tokenizer (50,257 tokens)
- Optimizaciones de rendimiento
- Mejor logging y checkpointing

Mejoras vs versión anterior:
✅ Vocabulario real (50k tokens vs 100)
✅ Datos reales de Wikipedia (vs sintéticos)
✅ Tokenización BPE profesional
✅ Mejor perplexity esperado (50-80 vs 99)
"""

import sys
import os

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import math

# HuggingFace libraries
from datasets import load_dataset
from transformers import GPT2Tokenizer

# Import modelo V5.2
from infinito_v5_2_refactored import InfinitoV52Refactored


# =============================================================================
# DATASET WIKITEXT-2 REAL CON GPT2TOKENIZER
# =============================================================================

class WikiText2RealDataset(Dataset):
    """
    Dataset WikiText-2 REAL usando GPT2Tokenizer.
    
    Características:
    - Datos reales de Wikipedia (HuggingFace datasets)
    - Tokenización BPE profesional (GPT-2)
    - Vocabulario: 50,257 tokens
    - Soporte de caching para velocidad
    """
    
    def __init__(self, split='train', seq_len=256, tokenizer=None):
        """
        Args:
            split: 'train', 'validation', o 'test'
            seq_len: Longitud de las secuencias
            tokenizer: GPT2Tokenizer instance (si None, se crea uno nuevo)
        """
        self.seq_len = seq_len
        
        # Cargar tokenizer
        if tokenizer is None:
            print(f"\n🔤 Cargando GPT2Tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            self.tokenizer = tokenizer
        
        self.vocab_size = len(self.tokenizer)
        print(f"  ✓ Vocabulario: {self.vocab_size:,} tokens")
        
        # Cargar WikiText-2 real
        print(f"\n📚 Cargando WikiText-2 REAL ({split})...")
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            print(f"  ✓ Dataset cargado: {len(dataset):,} ejemplos")
            
            # Concatenar todo el texto
            text = '\n'.join([example['text'] for example in dataset if example['text'].strip()])
            print(f"  ✓ Caracteres totales: {len(text):,}")
            
        except Exception as e:
            print(f"  ❌ Error cargando dataset: {e}")
            raise
        
        # Tokenizar todo el texto
        print(f"  📝 Tokenizando con GPT-2 BPE...")
        self.tokens = self.tokenizer.encode(text)
        print(f"  ✓ Total tokens: {len(self.tokens):,}")
        
        # Calcular número de secuencias
        self.num_sequences = len(self.tokens) // seq_len
        print(f"  ✓ Secuencias disponibles: {self.num_sequences:,}")
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """
        Retorna una secuencia de tokens.
        
        Returns:
            input_ids: tensor de shape (seq_len,)
            labels: tensor de shape (seq_len,) - shifted input_ids
        """
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1  # +1 para el label
        
        # Obtener secuencia
        sequence = self.tokens[start_idx:end_idx]
        
        # Si no hay suficientes tokens, hacer padding
        if len(sequence) < self.seq_len + 1:
            sequence = sequence + [self.tokenizer.eos_token_id] * (self.seq_len + 1 - len(sequence))
        
        # Input y labels (shifted)
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, labels


# =============================================================================
# TRAINER
# =============================================================================

class InfinitoTrainer:
    """Entrenador para INFINITO V5.2 con WikiText-2 real."""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        batch_size=32,
        learning_rate=1e-4,
        device='cuda'
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.device = device
        
        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=True if device == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=20,  # epochs
            eta_min=1e-6
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_perplexity': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch):
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Época {epoch}')
        
        for input_ids, labels in pbar:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(input_ids)
            
            # Manejar tupla si es necesario
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # Calcular loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Actualizar métricas
            total_loss += loss.item()
            num_batches += 1
            
            # Actualizar progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{math.exp(loss.item()):.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        avg_ppl = math.exp(avg_loss)
        
        return avg_loss, avg_ppl
    
    def validate(self):
        """Valida el modelo."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validación')
            
            for input_ids, labels in pbar:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                output = self.model(input_ids)
                
                # Manejar tupla si es necesario
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                # Calcular loss
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        avg_ppl = math.exp(avg_loss)
        
        return avg_loss, avg_ppl
    
    def train(self, num_epochs, save_dir='models/checkpoints'):
        """Ejecuta el entrenamiento completo."""
        print(f"\n{'='*70}")
        print(f"INICIANDO ENTRENAMIENTO - INFINITO V5.2 (WikiText-2 REAL)")
        print(f"{'='*70}")
        print(f"  Épocas: {num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Device: {self.device}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"{'='*70}\n")
        
        # Crear directorio para checkpoints
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"ÉPOCA {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            # Entrenar
            train_loss, train_ppl = self.train_epoch(epoch)
            
            # Validar
            val_loss, val_ppl = self.validate()
            
            # Learning rate actual
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Guardar métricas
            self.history['train_loss'].append(train_loss)
            self.history['train_perplexity'].append(train_ppl)
            self.history['val_loss'].append(val_loss)
            self.history['val_perplexity'].append(val_ppl)
            self.history['learning_rate'].append(current_lr)
            
            # Mostrar resultados
            print(f"\n📊 Resultados Época {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:,.2f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:,.2f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    epoch,
                    val_loss,
                    val_ppl,
                    os.path.join(save_dir, 'infinito_v5.2_real_best.pt')
                )
                print(f"  ✅ MEJOR MODELO guardado (val_loss: {val_loss:.4f})")
            
            # Guardar checkpoint cada 5 épocas
            if epoch % 5 == 0:
                self.save_checkpoint(
                    epoch,
                    val_loss,
                    val_ppl,
                    os.path.join(save_dir, f'infinito_v5.2_real_epoch_{epoch}.pt')
                )
            
            # Actualizar learning rate
            self.scheduler.step()
        
        # Guardar historial
        self.save_history()
        
        print(f"\n{'='*70}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*70}")
        print(f"  Mejor Val Loss: {best_val_loss:.4f}")
        print(f"  Mejor Val PPL: {math.exp(best_val_loss):,.2f}")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, epoch, val_loss, val_ppl, path):
        """Guarda un checkpoint del modelo."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_ppl': val_ppl,
            'history': self.history,
            'config': {
                'vocab_size': self.train_dataset.vocab_size,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
                'memory_slots': self.model.memory_slots
            }
        }
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Guarda el historial de entrenamiento."""
        os.makedirs('results/training', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_path = f'results/training/training_history_real_{timestamp}.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n💾 Historial guardado: {history_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Función principal de entrenamiento."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenamiento INFINITO V5.2 con WikiText-2 REAL')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Número de épocas (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Tamaño del batch (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--seq-len', type=int, default=256,
                       help='Longitud de secuencia (default: 256)')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Dimensión oculta (default: 512)')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Número de capas (default: 6)')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Número de cabezas de atención (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed para reproducibilidad (default: 42)')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Crear tokenizer compartido
    print(f"\n🔤 Inicializando GPT2Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    vocab_size = len(tokenizer)
    print(f"  ✓ Vocabulario: {vocab_size:,} tokens")
    
    # Cargar datasets
    print(f"\n📚 Cargando datasets...")
    train_dataset = WikiText2RealDataset(
        split='train',
        seq_len=args.seq_len,
        tokenizer=tokenizer
    )
    
    val_dataset = WikiText2RealDataset(
        split='validation',
        seq_len=args.seq_len,
        tokenizer=tokenizer
    )
    
    # Crear modelo
    print(f"\n🤖 Creando modelo INFINITO V5.2...")
    model = InfinitoV52Refactored(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        memory_slots=256,
        use_improved_memory=True,
        use_stochastic_exploration=True,
        seed=args.seed
    ).to(device)
    
    # Contar parámetros
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parámetros: {num_params:,}")
    
    # Crear trainer
    trainer = InfinitoTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device
    )
    
    # Entrenar
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
