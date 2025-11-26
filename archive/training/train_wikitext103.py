#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ENTRENAMIENTO CON WIKITEXT-103
=================================

Dataset: WikiText-103 (103M tokens vs 2.4M de WikiText-2)
Objetivo: Reducir overfitting y mejorar calidad de generación
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json
from pathlib import Path

from transformers import GPT2Tokenizer
from datasets import load_dataset

from infinito_v5_2_refactored import InfinitoV52Refactored


class WikiText103Dataset(Dataset):
    """Dataset WikiText-103 tokenizado."""
    
    def __init__(self, texts, tokenizer, seq_len=256, split='train'):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        
        print(f"\n📚 Procesando WikiText-103 ({split})...")
        print(f"  Ejemplos: {len(texts):,}")
        
        # Tokenizar todo el texto
        all_text = '\n'.join(texts)
        print(f"  Caracteres: {len(all_text):,}")
        
        print(f"  Tokenizando...")
        encoded = tokenizer.encode(all_text)
        print(f"  Tokens: {len(encoded):,}")
        
        # Crear secuencias
        self.sequences = []
        for i in range(0, len(encoded) - seq_len, seq_len // 2):  # 50% overlap
            seq = encoded[i:i + seq_len + 1]
            if len(seq) == seq_len + 1:
                self.sequences.append(seq)
        
        print(f"  Secuencias creadas: {len(self.sequences):,}")
        print(f"  Vocabulario: {len(tokenizer):,} tokens\n")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Input: [:-1], Labels: [1:] (shifted)
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        return input_ids, labels


def collate_fn(batch):
    """Collate function para el DataLoader."""
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels


class Trainer:
    """Entrenador para INFINITO V5.2."""
    
    def __init__(self, model, train_loader, val_loader, args, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        
        # Optimizador
        self.optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Historial
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_ppl': [],
            'val_ppl': [],
            'learning_rate': [],
            'train_phi': [],
            'epoch': []
        }
    
    def train_epoch(self, epoch):
        """Entrena una época."""
        self.model.train()
        total_loss = 0
        total_phi = 0
        
        pbar = tqdm(self.train_loader, desc=f'Época {epoch}')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            outputs = self.model(inputs)
            
            # Obtener logits y métricas
            if isinstance(outputs, tuple):
                logits, metrics = outputs
            else:
                logits = outputs
                metrics = {}
            
            # Loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Agregar loss de PHI si está disponible
            lambda_phi = getattr(self.model, 'lambda_phi', 0.3)
            if metrics and 'delta_phi_loss' in metrics:
                loss = loss + lambda_phi * metrics['delta_phi_loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Métricas
            total_loss += loss.item()
            if metrics and 'phi' in metrics:
                total_phi += metrics['phi'].mean().item()
            
            # Actualizar barra
            ppl = np.exp(min(loss.item(), 10))
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{ppl:.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_ppl = np.exp(min(avg_loss, 10))
        avg_phi = total_phi / len(self.train_loader) if total_phi > 0 else 0
        
        return avg_loss, avg_ppl, avg_phi
    
    @torch.no_grad()
    def validate(self):
        """Valida el modelo."""
        self.model.eval()
        total_loss = 0
        
        pbar = tqdm(self.val_loader, desc='Validación')
        
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(inputs)
            
            if isinstance(outputs, tuple):
                logits, _ = outputs
            else:
                logits = outputs
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        avg_ppl = np.exp(min(avg_loss, 10))
        
        return avg_loss, avg_ppl
    
    def train(self):
        """Loop principal de entrenamiento."""
        print(f"\n{'='*70}")
        print(f"INICIANDO ENTRENAMIENTO - WIKITEXT-103")
        print(f"{'='*70}")
        print(f"  Épocas: {self.args.epochs}")
        print(f"  Batch size: {self.args.batch_size}")
        print(f"  Learning rate: {self.args.lr:.2e}")
        print(f"  Device: {self.device}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.args.epochs + 1):
            print(f"\n{'='*70}")
            print(f"ÉPOCA {epoch}/{self.args.epochs}")
            print(f"{'='*70}")
            
            # Entrenar
            train_loss, train_ppl, train_phi = self.train_epoch(epoch)
            
            # Validar
            val_loss, val_ppl = self.validate()
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Guardar historial
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_ppl'].append(val_ppl)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['train_phi'].append(train_phi)
            self.history['epoch'].append(epoch)
            
            # Imprimir resultados
            print(f"\n📊 Resultados Época {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            if train_phi > 0:
                print(f"  🧠 Train PHI: {train_phi:.4f}")
            
            # Guardar mejor modelo
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                checkpoint_path = Path('models/checkpoints/infinito_v5.2_wikitext103_best.pt')
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_ppl': train_ppl,
                    'val_ppl': val_ppl,
                    'train_phi': train_phi,
                    'config': vars(self.args),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }, checkpoint_path)
                
                print(f"  ✅ MEJOR MODELO guardado (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.args.patience:
                print(f"\n⏹️  EARLY STOPPING activado en época {epoch}")
                print(f"  Val Loss no mejoró durante {self.args.patience} épocas")
                print(f"  Mejor Val Loss: {self.best_val_loss:.4f}")
                break
        
        # Guardar historial
        history_path = Path(f'results/training/wikitext103_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n💾 Historial guardado: {history_path}")
        
        print(f"\n{'='*70}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*70}")
        print(f"  Mejor Val Loss: {self.best_val_loss:.4f}")
        print(f"  Mejor Val PPL: {np.exp(min(self.best_val_loss, 10)):.2f}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Entrenar INFINITO V5.2 con WikiText-103')
    
    # Arquitectura
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Entrenamiento
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--lambda-phi', type=float, default=0.3)
    
    # Sistema
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=0)
    
    args = parser.parse_args()
    
    # Device
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"🚀 ENTRENAMIENTO WIKITEXT-103 - INFINITO V5.2")
    print(f"{'='*70}")
    print(f"\n🖥️  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar tokenizer
    print(f"\n🔤 Cargando GPT-2 Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"  ✓ Vocabulario: {len(tokenizer):,} tokens (GPT-2 (50k))")
    
    # Cargar dataset
    print(f"\n📚 Cargando WikiText-103...")
    try:
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')
    except:
        print(f"  Intentando cargar desde caché local...")
        from datasets import load_from_disk
        cache_dir = r"C:\Users\ENRIQUE\.cache\huggingface\datasets\wikitext\wikitext-103-raw-v1"
        try:
            dataset = load_from_disk(cache_dir)
        except:
            print(f"  ❌ No se pudo cargar WikiText-103")
            print(f"  Usando WikiText-2 en su lugar (más pequeño pero funcional)...")
            dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')
    
    # Filtrar líneas vacías
    train_texts = [text for text in dataset['train']['text'] if text.strip()]
    val_texts = [text for text in dataset['validation']['text'] if text.strip()]
    
    # Crear datasets
    train_dataset = WikiText103Dataset(train_texts, tokenizer, args.seq_len, 'train')
    val_dataset = WikiText103Dataset(val_texts, tokenizer, args.seq_len, 'validation')
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Crear modelo
    print(f"\n🤖 Creando modelo INFINITO V5.2...")
    model = InfinitoV52Refactored(
        vocab_size=len(tokenizer),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        memory_slots=256,
        dropout=args.dropout,
        use_improved_memory=True,
        use_improved_iit=True,
        use_learnable_phi=True,
        use_stochastic_exploration=True,
        lambda_phi=args.lambda_phi,
        seed=42
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parámetros: {total_params:,}")
    
    # Entrenar
    trainer = Trainer(model, train_loader, val_loader, args, device)
    trainer.train()


if __name__ == '__main__':
    main()
