#!/usr/bin/env python3
"""
🚀 DEMO DE ENTRENAMIENTO RÁPIDO
================================

Versión rápida del entrenamiento para demostración:
- 3 épocas (vs 15)
- Modelo pequeño (256 hidden, 3 layers)
- Batch size 16
- Dataset reducido

Para entrenamiento completo, usar: train_v5_2_wikitext.py
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


# Copiar clases del script completo
class WikiText2Dataset(Dataset):
    """Dataset WikiText-2 simplificado."""
    
    def __init__(self, split='train', seq_len=128, vocab_size=5000, max_samples=1000):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.max_samples = max_samples
        
        print(f"\n📚 Cargando WikiText-2 ({split}) - DEMO...")
        
        # Intentar cargar desde HuggingFace
        try:
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            # Limitar dataset para demo
            text_lines = dataset['text'][:max_samples * 10]
            text = '\n'.join(text_lines)
            print(f"  ✓ Cargado desde HuggingFace (limitado para demo)")
        except Exception as e:
            print(f"  ⚠️  Usando texto sintético: {e}")
            text = self._generate_synthetic_text()
        
        # Tokenización
        self.tokens = self._tokenize(text)
        print(f"  ✓ Total tokens: {len(self.tokens):,}")
        print(f"  ✓ Vocabulario: {self.vocab_size:,} tokens")
    
    def _generate_synthetic_text(self):
        """Genera texto sintético."""
        words = [
            'the', 'of', 'and', 'to', 'a', 'in', 'is', 'that', 'it', 'was',
            'for', 'on', 'are', 'as', 'with', 'his', 'they', 'be', 'at', 'one',
            'have', 'this', 'from', 'or', 'had', 'by', 'not', 'word', 'but', 'what'
        ]
        sentences = []
        for _ in range(500):  # Reducido para demo
            sentence = ' '.join(np.random.choice(words, np.random.randint(10, 20)))
            sentences.append(sentence)
        return ' . '.join(sentences)
    
    def _tokenize(self, text):
        """Tokenización simple."""
        words = text.lower().split()
        from collections import Counter
        word_counts = Counter(words)
        most_common = word_counts.most_common(self.vocab_size - 3)
        
        self.vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        for word, _ in most_common:
            self.vocab[word] = len(self.vocab)
        
        self.idx2word = {idx: word for word, idx in self.vocab.items()}
        
        tokens = [self.vocab.get(word, 1) for word in words]
        return tokens
    
    def __len__(self):
        return min(self.max_samples, max(1, (len(self.tokens) - self.seq_len) // self.seq_len))
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        
        if end_idx > len(self.tokens):
            end_idx = len(self.tokens)
            start_idx = max(0, end_idx - self.seq_len - 1)
        
        sequence = self.tokens[start_idx:end_idx]
        
        if len(sequence) < self.seq_len + 1:
            sequence = sequence + [0] * (self.seq_len + 1 - len(sequence))
        
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, target_ids


def train_demo():
    """Entrenamiento de demostración rápido."""
    
    # Configuración DEMO
    config = {
        'vocab_size': 5000,
        'hidden_dim': 256,  # Reducido
        'num_layers': 3,    # Reducido
        'num_heads': 4,     # Reducido
        'memory_slots': 128,  # Reducido
        'seq_len': 128,     # Reducido
        'batch_size': 16,   # Reducido
        'learning_rate': 1e-4,
        'num_epochs': 3,    # Solo 3 épocas para demo
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"\n{'='*70}")
    print(f"🚀 DEMO DE ENTRENAMIENTO - INFINITO V5.2")
    print(f"{'='*70}")
    print(f"  🎯 Configuración RÁPIDA para demostración:")
    print(f"     - Épocas: {config['num_epochs']} (vs 15 completo)")
    print(f"     - Hidden dim: {config['hidden_dim']} (vs 512 completo)")
    print(f"     - Batch size: {config['batch_size']} (vs 32 completo)")
    print(f"     - Seq len: {config['seq_len']} (vs 256 completo)")
    print(f"  📝 Para entrenamiento completo: train_v5_2_wikitext.py")
    print(f"{'='*70}\n")
    
    # Datasets
    train_dataset = WikiText2Dataset(
        split='train',
        seq_len=config['seq_len'],
        vocab_size=config['vocab_size'],
        max_samples=200  # Limitado para demo
    )
    
    val_dataset = WikiText2Dataset(
        split='validation',
        seq_len=config['seq_len'],
        vocab_size=config['vocab_size'],
        max_samples=50  # Limitado para demo
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"\n📊 Datasets (DEMO - reducidos):")
    print(f"  Train: {len(train_dataset):,} secuencias")
    print(f"  Val:   {len(val_dataset):,} secuencias")
    print(f"  Batches por época: {len(train_loader):,}")
    
    # Modelo
    model = InfinitoV52Refactored(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        memory_slots=config['memory_slots'],
        use_improved_memory=True,
        use_stochastic_exploration=True,
        seed=config['seed']
    ).to(config['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📏 Parámetros: {total_params:,}")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"🚀 COMENZANDO ENTRENAMIENTO DEMO")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_ppl': [], 'val_ppl': []}
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        model.train()
        train_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Época {epoch}/{config["num_epochs"]}')
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(config['device'])
            target_ids = target_ids.to(config['device'])
            
            optimizer.zero_grad()
            output = model(input_ids)
            # El modelo retorna solo logits cuando return_metrics=False (default)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            loss = nn.functional.cross_entropy(
                logits.view(-1, config['vocab_size']),
                target_ids.view(-1),
                ignore_index=0
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / num_batches
        train_ppl = math.exp(min(avg_train_loss, 20))
        
        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(config['device'])
                target_ids = target_ids.to(config['device'])
                
                output = model(input_ids)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                
                loss = nn.functional.cross_entropy(
                    logits.view(-1, config['vocab_size']),
                    target_ids.view(-1),
                    ignore_index=0
                )
                
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        val_ppl = math.exp(min(avg_val_loss, 20))
        
        # Guardar métricas
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)
        
        # Imprimir
        print(f"\n📊 Época {epoch}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Perplexity: {train_ppl:.2f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Perplexity: {val_ppl:.2f}")
        
        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('models/checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }, 'models/checkpoints/infinito_v5.2_demo_best.pt')
            print(f"  💾 Mejor modelo guardado!")
    
    # Resumen final
    print(f"\n{'='*70}")
    print(f"✅ ENTRENAMIENTO DEMO COMPLETADO")
    print(f"{'='*70}")
    print(f"  Mejor val_loss: {best_val_loss:.4f}")
    print(f"  Mejor val_perplexity: {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"\n📈 Progreso:")
    print(f"  Época 1 → {config['num_epochs']}: {history['val_ppl'][0]:.2f} → {history['val_ppl'][-1]:.2f}")
    
    if history['val_ppl'][-1] < history['val_ppl'][0]:
        improvement = (history['val_ppl'][0] - history['val_ppl'][-1]) / history['val_ppl'][0] * 100
        print(f"  ✅ Mejora: {improvement:.1f}%")
    
    print(f"\n📝 Para entrenamiento completo (15 épocas, modelo grande):")
    print(f"   python train_v5_2_wikitext.py")
    print(f"{'='*70}\n")
    
    return history


if __name__ == '__main__':
    train_demo()
