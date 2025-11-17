#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 BASELINE TRANSFORMER - SIN CARACTERÍSTICAS IIT
===================================================

Script para entrenar transformer ESTÁNDAR sin características IIT
para comparación científica con InfinitoV52Refactored.

ARQUITECTURA:
- Transformer decoder estándar
- hidden_dim=512, num_layers=4, num_heads=8
- SIN IITGuidedMemory
- SIN LearnablePhiWeights
- SIN StochasticExploration
- SIN ImprovedIITMetrics

OBJETIVO:
Validar si características IIT aportan beneficio vs transformer baseline.
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

# =============================================================================
# TRANSFORMER BASELINE (SIN IIT)
# =============================================================================

class TransformerBaseline(nn.Module):
    """
    Transformer Decoder estándar SIN características IIT.
    
    Arquitectura:
    - Embedding layer
    - N layers de TransformerDecoderLayer estándar
    - Output projection a vocabulario
    
    NO incluye:
    - IITGuidedMemory
    - LearnablePhiWeights
    - StochasticExploration
    - ImprovedIITMetrics
    """
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.15,
        max_seq_len=512
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN como GPT-2
        )
        
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # Output projection
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie weights (compartir embeddings de entrada/salida)
        self.output_proj.weight = self.token_embedding.weight
        
        # Inicialización
        self.apply(self._init_weights)
        
        # Contar parámetros
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def _init_weights(self, module):
        """Inicialización estilo GPT-2"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, labels=None):
        """
        Forward pass estándar.
        
        Args:
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] (opcional, para calcular loss)
            
        Returns:
            Si labels es None: logits [batch_size, seq_len, vocab_size]
            Si labels no es None: (loss, logits)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_emb = self.position_embedding(positions).unsqueeze(0)
        
        x = token_emb + position_emb
        
        # Causal mask (evitar ver tokens futuros)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )
        
        # Transformer decoder
        # Como usamos decoder-only, memory=tgt (autoregresivo)
        x = self.transformer(
            tgt=x,
            memory=x,
            tgt_mask=causal_mask,
            memory_mask=causal_mask
        )
        
        # Output projection
        x = self.ln_f(x)
        logits = self.output_proj(x)
        
        # Calcular loss si se proveen labels
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
            return loss, logits
        
        return logits


# =============================================================================
# DATASET WIKITEXT-2 (IGUAL QUE MODELO CON IIT)
# =============================================================================

class WikiText2Dataset(Dataset):
    """Dataset WikiText-2 con shift para language modeling"""
    
    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"📊 Tokenizando {len(texts)} textos...")
        
        # Tokenizar en batches para mayor eficiencia
        batch_size = 1000
        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizando batches"):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t for t in batch_texts if t.strip()]
            
            # Tokenizar batch
            batch_encoded = tokenizer(
                batch_texts,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length + 1,
                return_attention_mask=False,
                padding=False
            )['input_ids']
            
            # Agregar solo secuencias válidas
            for tokens in batch_encoded:
                if len(tokens) >= 2:
                    self.examples.append(tokens)
        
        print(f"✅ Total de secuencias: {len(self.examples)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        
        # Shift para language modeling
        # input_ids: primeros N-1 tokens
        # labels: últimos N-1 tokens (shifted)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


def collate_fn(batch):
    """Collate function con padding"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Padding
    input_ids = nn.utils.rnn.pad_sequence(
        input_ids, 
        batch_first=True, 
        padding_value=0
    )
    labels = nn.utils.rnn.pad_sequence(
        labels, 
        batch_first=True, 
        padding_value=-100  # Ignorar en loss
    )
    
    return {
        'input_ids': input_ids,
        'labels': labels
    }


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping para detener cuando val_loss no mejora"""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            return True
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return False
            return True


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Entrenar una época"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Época {epoch}")
    
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        loss, logits = model(input_ids, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(min(avg_loss, 20))  # Cap para evitar overflow
    
    # Step scheduler
    if scheduler is not None:
        scheduler.step()
    
    return avg_loss, avg_ppl


def evaluate(model, dataloader, device):
    """Evaluar modelo en validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validando"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            loss, logits = model(input_ids, labels)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_ppl = math.exp(min(avg_loss, 20))
    
    return avg_loss, avg_ppl


def main():
    """Entrenamiento principal"""
    
    print("\n" + "="*70)
    print("🔬 ENTRENAMIENTO BASELINE TRANSFORMER (SIN IIT)")
    print("="*70)
    
    # Configuración
    config = {
        'vocab_size': 50257,
        'hidden_dim': 512,
        'num_layers': 4,
        'num_heads': 8,
        'dropout': 0.15,
        'max_seq_len': 512,
        'seq_len': 256,
        'batch_size': 16,
        'learning_rate': 5e-4,
        'weight_decay': 0.01,
        'epochs': 15,
        'patience': 5,
        'warmup_steps': 500,
    }
    
    print("\n📋 CONFIGURACIÓN:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar tokenizer
    print("\n📝 Cargando GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Cargar dataset WikiText-2
    print("\n📊 Cargando WikiText-2...")
    try:
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=['train', 'validation'])
    except:
        # Usar caché local si falla
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', cache_dir=cache_dir, split=['train', 'validation'])
    
    train_texts = [text for text in dataset[0]['text'] if text.strip()]
    val_texts = [text for text in dataset[1]['text'] if text.strip()]
    
    print(f"   Train texts: {len(train_texts)}")
    print(f"   Val texts: {len(val_texts)}")
    
    # Crear datasets
    train_dataset = WikiText2Dataset(train_texts, tokenizer, config['seq_len'])
    val_dataset = WikiText2Dataset(val_texts, tokenizer, config['seq_len'])
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"\n   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Crear modelo
    print("\n🤖 Creando Transformer Baseline (SIN IIT)...")
    model = TransformerBaseline(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len']
    )
    model = model.to(device)
    
    print(f"   Parámetros: {model.num_parameters:,}")
    
    # Optimizer y scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])
    
    # Training loop
    print("\n" + "="*70)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("="*70)
    
    history = {
        'train_loss': [],
        'train_ppl': [],
        'val_loss': [],
        'val_ppl': [],
        'learning_rates': []
    }
    
    best_val_ppl = float('inf')
    best_epoch = 0
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"ÉPOCA {epoch}/{config['epochs']}")
        print(f"{'='*70}")
        
        # Entrenar
        train_loss, train_ppl = train_epoch(
            model, train_loader, optimizer, None, device, epoch
        )
        
        # Validar
        val_loss, val_ppl = evaluate(model, val_loader, device)
        
        # Learning rate actual
        current_lr = optimizer.param_groups[0]['lr']
        
        # Guardar historia
        history['train_loss'].append(train_loss)
        history['train_ppl'].append(train_ppl)
        history['val_loss'].append(val_loss)
        history['val_ppl'].append(val_ppl)
        history['learning_rates'].append(current_lr)
        
        # Imprimir resumen
        print(f"\n📊 Resumen Época {epoch}:")
        print(f"   Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"   Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
        print(f"   Learning Rate: {current_lr:.2e}")
        
        # Step scheduler (basado en val_loss)
        scheduler.step(val_loss)
        
        # Guardar mejor modelo
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_ppl': train_ppl,
                'val_ppl': val_ppl,
                'config': config,
                'history': history
            }
            
            os.makedirs('models/checkpoints', exist_ok=True)
            checkpoint_path = 'models/checkpoints/baseline_no_iit_best.pt'
            torch.save(checkpoint, checkpoint_path)
            
            print(f"\n✅ Mejor modelo guardado: {checkpoint_path}")
            print(f"   Val PPL: {val_ppl:.2f} (época {epoch})")
        
        # Early stopping check
        if not early_stopping(val_loss):
            print(f"\n⏹️  Early stopping activado después de época {epoch}")
            print(f"   Mejor Val PPL: {best_val_ppl:.2f} (época {best_epoch})")
            break
    
    # Resumen final
    print("\n" + "="*70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\n📊 MEJORES RESULTADOS:")
    print(f"   Época: {best_epoch}")
    print(f"   Val PPL: {best_val_ppl:.2f}")
    print(f"   Checkpoint: models/checkpoints/baseline_no_iit_best.pt")
    
    # Guardar historia completa
    os.makedirs('results/training', exist_ok=True)
    history_path = f'results/training/baseline_no_iit_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n📁 Historia guardada: {history_path}")
    
    print("\n" + "="*70)
    print("🔬 PRÓXIMO PASO: Comparar con modelo CON IIT")
    print("="*70)
    print("\nComparar Val PPL:")
    print(f"   Baseline (SIN IIT):  {best_val_ppl:.2f}")
    print(f"   Model A (CON IIT):   216.46")
    print(f"   Model B (CON IIT):   207.15")
    print("\nSi baseline > 216.46: IIT aporta beneficio ✅")
    print("Si baseline ≈ 216.46: IIT sin efecto significativo ⚠️")
    print("Si baseline < 216.46: IIT perjudica rendimiento ❌")
    

if __name__ == '__main__':
    main()
