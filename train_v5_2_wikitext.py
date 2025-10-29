#!/usr/bin/env python3
"""
🎓 ENTRENAMIENTO: INFINITO V5.2 con WikiText-2
==============================================

Script completo de entrenamiento para INFINITO V5.2 usando el dataset WikiText-2.

Características:
- Dataset: WikiText-2 (artículos de Wikipedia)
- Epochs: 10-20 configurables
- Learning rate: 1e-4 con scheduler
- Batch size: 32
- Checkpointing automático
- Logging detallado de métricas
- Reproducibilidad garantizada (seed fijado)
"""

import sys
import os
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

# Import modelo V5.2
from infinito_v5_2_refactored import InfinitoV52Refactored


# =============================================================================
# DATASET WIKITEXT-2
# =============================================================================

class WikiText2Dataset(Dataset):
    """
    Dataset WikiText-2 con tokenización simple basada en vocabulario.
    
    En producción, usar tokenizers más sofisticados (BPE, WordPiece, etc.)
    """
    
    def __init__(self, split='train', seq_len=256, vocab_size=10000):
        """
        Args:
            split: 'train', 'valid', o 'test'
            seq_len: Longitud de las secuencias
            vocab_size: Tamaño del vocabulario
        """
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        print(f"\n📚 Cargando WikiText-2 ({split})...")
        
        # Intentar cargar usando datasets de HuggingFace
        try:
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            text = '\n'.join(dataset['text'])
            print(f"  ✓ Cargado desde HuggingFace datasets")
        except Exception as e:
            print(f"  ⚠️  No se pudo cargar desde HuggingFace: {e}")
            print(f"  ℹ️  Generando texto sintético de ejemplo...")
            # Texto sintético si no hay conexión
            text = self._generate_synthetic_text()
        
        # Tokenización simple (por palabras)
        print(f"  📝 Tokenizando...")
        self.tokens = self._tokenize(text)
        
        print(f"  ✓ Total tokens: {len(self.tokens):,}")
        print(f"  ✓ Vocabulario: {self.vocab_size:,} tokens")
        print(f"  ✓ Secuencias de {seq_len} tokens")
        
    def _generate_synthetic_text(self):
        """Genera texto sintético para pruebas si no hay dataset real."""
        words = [
            'the', 'of', 'and', 'to', 'a', 'in', 'is', 'that', 'it', 'was',
            'for', 'on', 'are', 'as', 'with', 'his', 'they', 'be', 'at', 'one',
            'have', 'this', 'from', 'or', 'had', 'by', 'not', 'word', 'but', 'what',
            'some', 'we', 'can', 'out', 'other', 'were', 'all', 'there', 'when', 'up',
            'use', 'your', 'how', 'said', 'an', 'each', 'she', 'which', 'do', 'their',
            'time', 'if', 'will', 'way', 'about', 'many', 'then', 'them', 'write', 'would',
            'like', 'so', 'these', 'her', 'long', 'make', 'thing', 'see', 'him', 'two',
            'has', 'look', 'more', 'day', 'could', 'go', 'come', 'did', 'number', 'sound',
            'no', 'most', 'people', 'my', 'over', 'know', 'water', 'than', 'call', 'first',
            'who', 'may', 'down', 'side', 'been', 'now', 'find', 'any', 'new', 'work'
        ]
        
        # Generar 50K palabras sintéticas
        sentences = []
        for _ in range(5000):
            sentence_len = np.random.randint(10, 30)
            sentence = ' '.join(np.random.choice(words, sentence_len))
            sentences.append(sentence)
        
        return ' . '.join(sentences)
    
    def _tokenize(self, text):
        """Tokenización simple por palabras con vocabulario limitado."""
        # Separar por espacios y limpiar
        words = text.lower().split()
        
        # Crear vocabulario de las palabras más frecuentes
        from collections import Counter
        word_counts = Counter(words)
        most_common = word_counts.most_common(self.vocab_size - 3)  # -3 para tokens especiales
        
        # Vocabulario: <pad>=0, <unk>=1, <eos>=2, luego palabras comunes
        self.vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)
        
        # Crear mapping inverso
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        
        # Convertir texto a tokens
        tokens = []
        for word in words:
            tokens.append(self.vocab.get(word, 1))  # 1 = <unk>
        
        return tokens
    
    def __len__(self):
        """Número de secuencias en el dataset."""
        return max(1, (len(self.tokens) - self.seq_len) // self.seq_len)
    
    def __getitem__(self, idx):
        """
        Retorna una secuencia y su target (shifted).
        
        Returns:
            input_ids: [seq_len] - secuencia de entrada
            target_ids: [seq_len] - secuencia target (shifted)
        """
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1  # +1 para el target
        
        # Si llegamos al final, tomar los últimos tokens
        if end_idx > len(self.tokens):
            end_idx = len(self.tokens)
            start_idx = max(0, end_idx - self.seq_len - 1)
        
        sequence = self.tokens[start_idx:end_idx]
        
        # Pad si es necesario
        if len(sequence) < self.seq_len + 1:
            sequence = sequence + [0] * (self.seq_len + 1 - len(sequence))
        
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids


# =============================================================================
# TRAINER
# =============================================================================

class InfinitoTrainer:
    """Trainer para INFINITO V5.2 con logging y checkpointing."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1e-4,
        device='cpu',
        checkpoint_dir='models/checkpoints',
        log_dir='results/training'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Optimizer y scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 20,  # 20 epochs
            eta_min=1e-6
        )
        
        # Crear directorios
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Métricas
        self.history = {
            'train_loss': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_perplexity': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        
        print(f"\n🎓 Trainer inicializado:")
        print(f"  Device: {device}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Optimizer: AdamW")
        print(f"  Scheduler: CosineAnnealingLR")
    
    def train_epoch(self, epoch):
        """Entrena una época."""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids)
            
            # Calcular loss
            batch_size, seq_len, vocab_size = logits.shape
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
                ignore_index=0  # Ignorar padding
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Métricas
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        avg_perplexity = math.exp(min(avg_loss, 20))  # Cap para evitar overflow
        
        return avg_loss, avg_perplexity
    
    def validate(self):
        """Valida el modelo."""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in tqdm(self.val_loader, desc='Validación'):
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Calcular loss
                batch_size, seq_len, vocab_size = logits.shape
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1),
                    ignore_index=0
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_perplexity = math.exp(min(avg_loss, 20))
        
        return avg_loss, avg_perplexity
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Guarda checkpoint del modelo."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'history': self.history
        }
        
        # Guardar checkpoint regular
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'infinito_v5.2_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Guardar mejor modelo
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir,
                'infinito_v5.2_best.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"  💾 Mejor modelo guardado (val_loss: {val_loss:.4f})")
    
    def save_history(self):
        """Guarda historial de métricas."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(
            self.log_dir,
            f'training_history_{timestamp}.json'
        )
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n💾 Historial guardado en: {history_path}")
    
    def train(self, num_epochs=10, early_stopping_patience=3):
        """
        Loop de entrenamiento completo.
        
        Args:
            num_epochs: Número de épocas
            early_stopping_patience: Paciencia para early stopping
        """
        print(f"\n{'='*70}")
        print(f"🚀 COMENZANDO ENTRENAMIENTO")
        print(f"{'='*70}")
        print(f"  Épocas: {num_epochs}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print(f"{'='*70}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n📅 Época {epoch}/{num_epochs}")
            
            # Entrenar
            train_loss, train_perplexity = self.train_epoch(epoch)
            
            # Validar
            val_loss, val_perplexity = self.validate()
            
            # Guardar métricas
            self.history['train_loss'].append(train_loss)
            self.history['train_perplexity'].append(train_perplexity)
            self.history['val_loss'].append(val_loss)
            self.history['val_perplexity'].append(val_perplexity)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            # Imprimir resultados
            print(f"\n📊 Resultados Época {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Perplexity: {train_perplexity:.2f}")
            print(f"  Val Loss:   {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")
            print(f"  Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Guardar checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping activado (patience: {early_stopping_patience})")
                break
        
        # Guardar historial final
        self.save_history()
        
        print(f"\n{'='*70}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"{'='*70}")
        print(f"  Mejor val_loss: {self.best_val_loss:.4f}")
        print(f"  Mejor val_perplexity: {math.exp(min(self.best_val_loss, 20)):.2f}")
        print(f"{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Función principal de entrenamiento."""
    
    # Configuración
    config = {
        'vocab_size': 10000,
        'hidden_dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'memory_slots': 256,
        'seq_len': 256,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 15,
        'seed': 42,  # 🔒 Seed fijado para reproducibilidad
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\n{'='*70}")
    print(f"🎓 ENTRENAMIENTO INFINITO V5.2 - WikiText-2")
    print(f"{'='*70}")
    print(f"  Configuración:")
    for key, value in config.items():
        print(f"    {key}: {value}")
    print(f"{'='*70}\n")
    
    # Crear datasets
    train_dataset = WikiText2Dataset(
        split='train',
        seq_len=config['seq_len'],
        vocab_size=config['vocab_size']
    )
    
    val_dataset = WikiText2Dataset(
        split='validation',
        seq_len=config['seq_len'],
        vocab_size=config['vocab_size']
    )
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # 0 para evitar problemas en Windows
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print(f"\n📊 Datasets:")
    print(f"  Train: {len(train_dataset):,} secuencias")
    print(f"  Val:   {len(val_dataset):,} secuencias")
    print(f"  Batches por época: {len(train_loader):,}")
    
    # Crear modelo
    model = InfinitoV52Refactored(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        memory_slots=config['memory_slots'],
        use_improved_memory=True,
        use_stochastic_exploration=True,
        seed=config['seed']  # 🔒 Reproducibilidad
    )
    
    print(f"\n📏 Parámetros del modelo:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Entrenables: {trainable_params:,}")
    
    # Crear trainer
    trainer = InfinitoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        device=config['device']
    )
    
    # Entrenar
    trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=3
    )


if __name__ == '__main__':
    main()
