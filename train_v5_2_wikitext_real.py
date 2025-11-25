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
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """
    Early stopping para detener el entrenamiento cuando val_loss no mejora.
    
    Args:
        patience: Número de épocas sin mejora antes de detener
        min_delta: Mínima mejora para considerar que hubo progreso
        mode: 'min' para loss (menor es mejor), 'max' para accuracy
    """
    def __init__(self, patience=3, min_delta=0.01, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Actualiza el contador y decide si debe hacer early stop.
        
        Returns:
            True si debe continuar, False si debe detener
        """
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            return True
        
        # Comprueba si hubo mejora significativa
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
# DATASET WIKITEXT-2 REAL CON LLAMA 3 TOKENIZER 🦙
# =============================================================================

class WikiText2RealDataset(Dataset):
    """
    Dataset WikiText-2 REAL usando Llama 3 Tokenizer.
    
    🆕 MEJORAS vs GPT-2:
    - Vocabulario: 128,256 tokens (vs 50,257 GPT-2) → +155% más grande
    - Compresión: ~25% más eficiente (menos tokens por texto)
    - Multilenguaje: Mejor cobertura internacional
    - Modernidad: Tokenizer de 2024 (vs 2019)
    
    Características:
    - Datos reales de Wikipedia (HuggingFace datasets)
    - Tokenización BPE moderna (Llama 3)
    - Soporte de caching para velocidad
    """
    
    def __init__(self, split='train', seq_len=256, tokenizer=None, use_llama3=True):
        """
        Args:
            split: 'train', 'validation', o 'test'
            seq_len: Longitud de las secuencias
            tokenizer: Tokenizer instance (si None, se crea uno nuevo)
            use_llama3: Si True, usa Llama 3 tokenizer; si False, GPT-2
        """
        self.seq_len = seq_len
        self.use_llama3 = use_llama3
        
        # Cargar tokenizer
        if tokenizer is None:
            if use_llama3:
                print(f"\n🦙 Cargando Llama 3 Tokenizer...")
                from transformers import AutoTokenizer
                try:
                    # Llama 3 tokenizer (128k vocab)
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        'meta-llama/Meta-Llama-3-8B',
                        use_fast=True
                    )
                except Exception as e:
                    print(f"  ⚠️  Error cargando Llama 3, usando GPT-2 como fallback: {e}")
                    from transformers import GPT2Tokenizer
                    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                    self.use_llama3 = False
            else:
                print(f"\n🔤 Cargando GPT-2 Tokenizer...")
                from transformers import GPT2Tokenizer
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            self.tokenizer = tokenizer
        
        self.vocab_size = len(self.tokenizer)
        tokenizer_name = "Llama 3 (128k)" if self.use_llama3 else "GPT-2 (50k)"
        print(f"  ✓ Vocabulario: {self.vocab_size:,} tokens ({tokenizer_name})")
        
        # Cargar WikiText-2 real
        print(f"\n📚 Cargando WikiText-2 REAL ({split})...")
        try:
            # Nuevo path del dataset (sin trust_remote_code que está deprecated)
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
            print(f"  ✓ Dataset cargado: {len(dataset):,} ejemplos")
            
            # Concatenar todo el texto
            text = '\n'.join([example['text'] for example in dataset if example['text'].strip()])
            print(f"  ✓ Caracteres totales: {len(text):,}")
            
        except Exception as e:
            print(f"  ❌ Error cargando dataset: {e}")
            raise
        
        # Tokenizar todo el texto
        tokenizer_type = "Llama 3" if self.use_llama3 else "GPT-2 BPE"
        print(f"  📝 Tokenizando con {tokenizer_type}...")
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
        device='cuda',
        start_epoch=0,  # Parámetro para continuar entrenamiento
        use_early_stopping=True,  # 🆕 Activar early stopping
        patience=3,  # 🆕 Paciencia para early stopping
        use_plateau_scheduler=True  # 🆕 Usar ReduceLROnPlateau
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.device = device
        self.start_epoch = start_epoch  # Guardar época inicial
        self.use_early_stopping = use_early_stopping
        self.use_plateau_scheduler = use_plateau_scheduler
        
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
            betas=(0.9, 0.999),  # ✅ ÓPTIMO (match con Model A)
            eps=1e-9,
            weight_decay=0.01  # ✅ ÓPTIMO (Regularización L2)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        if use_plateau_scheduler:
            print("  🔧 Usando ReduceLROnPlateau (adaptativo)")
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,  # Reducir LR a la mitad
                patience=2,  # Esperar 2 épocas sin mejora
                min_lr=1e-6,
                verbose=True
            )
        else:
            print("  🔧 Usando CosineAnnealingLR (fijo)")
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=20,  # epochs
                eta_min=1e-6
            )
        
        # Early Stopping
        if use_early_stopping:
            print(f"  ⏹️  Early Stopping activado (patience={patience})")
            self.early_stopping = EarlyStopping(patience=patience, min_delta=0.01, mode='min')
        else:
            self.early_stopping = None
        
        # History
        self.history = {
            'train_loss': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_perplexity': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch):
        """Entrena una época con IIT mejorado."""
        self.model.train()
        total_loss = 0
        total_loss_lm = 0
        total_loss_phi = 0
        num_batches = 0
        
        # Acumuladores para métricas IIT
        phi_values = []
        
        pbar = tqdm(self.train_loader, desc=f'Época {epoch}')
        
        for input_ids, labels in pbar:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass con métricas IIT
            self.optimizer.zero_grad()
            logits, metrics = self.model(input_ids, return_metrics=True)
            
            # Loss de language modeling
            loss_lm = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            
            # 🆕 Loss auxiliar ΔPhi (si está disponible)
            loss_phi = 0.0
            if metrics and 'delta_phi_loss' in metrics:
                loss_phi = metrics['delta_phi_loss']
                # Convertir a tensor si es float
                if isinstance(loss_phi, float):
                    loss_phi = torch.tensor(loss_phi, device=self.device)
            
            # Loss total (LM + lambda*ΔPhi)
            # Obtener lambda_phi del modelo (puede estar como atributo o en config)
            lambda_phi = getattr(self.model, 'lambda_phi', 0.3)
            
            if isinstance(loss_phi, torch.Tensor) and loss_phi.item() > 0:
                loss = loss_lm + lambda_phi * loss_phi
            else:
                loss = loss_lm
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Actualizar métricas
            total_loss += loss.item()
            total_loss_lm += loss_lm.item()
            if isinstance(loss_phi, torch.Tensor):
                total_loss_phi += loss_phi.item()
            num_batches += 1
            
            # Acumular PHI
            if metrics and 'integration_phi' in metrics:
                phi_values.append(metrics['integration_phi'])
            
            # Actualizar progress bar
            postfix = {
                'loss': f'{loss.item():.4f}',
                'ppl': f'{math.exp(loss_lm.item()):.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            }
            if metrics and 'integration_phi' in metrics:
                postfix['phi'] = f'{metrics["integration_phi"]:.3f}'
            pbar.set_postfix(postfix)
        
        avg_loss = total_loss / num_batches
        avg_loss_lm = total_loss_lm / num_batches
        avg_loss_phi = total_loss_phi / num_batches if total_loss_phi > 0 else 0.0
        avg_ppl = math.exp(avg_loss_lm)
        avg_phi = sum(phi_values) / len(phi_values) if phi_values else 0.0
        
        return avg_loss, avg_ppl, avg_phi, avg_loss_phi
    
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
        print(f"  Época inicial: {self.start_epoch + 1}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Device: {self.device}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"{'='*70}\n")
        
        # Crear directorio para checkpoints
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.start_epoch + 1, num_epochs + 1):  # Continuar desde start_epoch
            print(f"\n{'='*70}")
            print(f"ÉPOCA {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            # Entrenar
            train_loss, train_ppl, train_phi, train_loss_phi = self.train_epoch(epoch)
            
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
            
            # 🆕 Guardar métricas IIT
            if 'train_phi' not in self.history:
                self.history['train_phi'] = []
                self.history['train_loss_phi'] = []
            self.history['train_phi'].append(train_phi)
            self.history['train_loss_phi'].append(train_loss_phi)
            
            # Mostrar resultados
            print(f"\n📊 Resultados Época {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:,.2f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:,.2f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  🧠 Train PHI: {train_phi:.4f} | ΔPhi Loss: {train_loss_phi:.6f}")
            
            # 🆕 Mostrar threshold aprendible
            if hasattr(self.model.memory, 'get_threshold'):
                threshold = self.model.memory.get_threshold()
                print(f"  🎯 Memory Threshold: {threshold:.4f} (aprendible)")
            
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
            if self.use_plateau_scheduler:
                self.scheduler.step(val_loss)  # ReduceLROnPlateau necesita la métrica
            else:
                self.scheduler.step()
            
            # 🆕 Early Stopping
            if self.early_stopping is not None:
                should_continue = self.early_stopping(val_loss)
                if not should_continue:
                    print(f"\n⏹️  EARLY STOPPING activado en época {epoch}")
                    print(f"  Val Loss no mejoró durante {self.early_stopping.patience} épocas")
                    print(f"  Mejor Val Loss: {best_val_loss:.4f}")
                    break
        
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
                'use_improved_memory': self.model.use_improved_memory,
                'use_stochastic_exploration': self.model.use_stochastic_exploration,
                'seed': self.model.seed
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
# MODEL CONFIGURATIONS (PRESETS)
# =============================================================================

MODEL_CONFIGS = {
    "large_iit": {
        # Configuración actual optimizada (Model A)
        "hidden_dim": 512,
        "num_layers": 4,
        "num_heads": 8,
        "batch_size": 16,
        "learning_rate": 5e-4,
        "seq_len": 256,
        "dropout": 0.15,
        "lambda_phi": 0.3,
        "vocab_size": None,  # Se calculará dinámicamente según el tokenizer
        "description": "Configuración optimizada para rendimiento máximo con IIT features"
    },
    "small_iit": {
        # Configuración más pequeña y eficiente
        "hidden_dim": 384,
        "num_layers": 3,
        "num_heads": 6,
        "batch_size": 16,
        "learning_rate": 5e-4,
        "seq_len": 256,
        "dropout": 0.15,
        "lambda_phi": 0.3,
        "vocab_size": None,  # Usar vocabulario completo para evitar problemas de indexing
        "description": "Configuración compacta para experimentación rápida"
    },
    "micro_iit": {
        # 🆕 Configuración micro para reducir ratio parámetros/datos
        "hidden_dim": 384,
        "num_layers": 3,
        "num_heads": 6,
        "batch_size": 16,
        "learning_rate": 1e-4,  # LR más conservador para modelo pequeño
        "seq_len": 256,
        "dropout": 0.2,  # Dropout ligeramente mayor para regularización
        "lambda_phi": 0.1,  # Peso PHI reducido para no dominar
        "vocab_size": None,
        "description": "Configuración micro (~28M parámetros) para ratio óptimo parámetros/datos ~12:1"
    },
    "tiny_iit": {
        # 🆕 Configuración ultra-pequeña para experimentación extrema
        "hidden_dim": 256,
        "num_layers": 2,
        "num_heads": 4,
        "batch_size": 16,
        "learning_rate": 2e-4,
        "seq_len": 256,
        "dropout": 0.25,  # Dropout alto para forzar generalización
        "lambda_phi": 0.05,  # Peso PHI muy bajo
        "vocab_size": None,
        "description": "Configuración tiny (~12M parámetros) para ratio extremo parámetros/datos ~5:1"
    }
}


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Función principal de entrenamiento."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenamiento INFINITO V5.2 con WikiText-2 REAL - CONFIGURACIÓN ÓPTIMA (Model A)')
    
    # 🆕 PRESET DE CONFIGURACIÓN
    parser.add_argument('--model-size', type=str, default='large_iit',
                       choices=['large_iit', 'small_iit', 'micro_iit', 'tiny_iit'],
                       help='Preset de configuración del modelo (default: large_iit)')
    
    parser.add_argument('--epochs', type=int, default=5,
                       help='Número de épocas (default: 5 para prueba rápida, luego 20)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Tamaño del batch (si no se especifica, usa el preset)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (si no se especifica, usa el preset)')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout rate (si no se especifica, usa el preset)')
    parser.add_argument('--seq-len', type=int, default=None,
                       help='Longitud de secuencia (si no se especifica, usa el preset)')
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Dimensión oculta (si no se especifica, usa el preset)')
    parser.add_argument('--num-layers', type=int, default=None,
                       help='Número de capas (si no se especifica, usa el preset)')
    parser.add_argument('--num-heads', type=int, default=None,
                       help='Número de cabezas de atención (si no se especifica, usa el preset)')
    parser.add_argument('--lambda-phi', type=float, default=None,
                       help='Peso del objetivo ΔPhi (si no se especifica, usa el preset)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed para reproducibilidad (default: 42)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Paciencia para early stopping (default: 5 - ÓPTIMO)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Desactivar early stopping')
    parser.add_argument('--no-plateau-scheduler', action='store_true',
                       help='Usar CosineAnnealingLR en vez de ReduceLROnPlateau')
    parser.add_argument('--use-llama3', action='store_true',
                       help='🦙 Usar Llama 3 tokenizer (128k vocab) en vez de GPT-2 (50k)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path al checkpoint para continuar entrenamiento')
    
    args = parser.parse_args()
    
    # 🚀 CARGAR CONFIGURACIÓN DESDE PRESET
    config = MODEL_CONFIGS[args.model_size].copy()
    print(f"\n🔧 Using preset: {args.model_size} -> {config['description']}")
    print(f"📋 Configuration: {config}")
    
    # Permitir override manual de cualquier parámetro
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        print(f"  ⚠️  Override: batch_size = {args.batch_size}")
    if args.lr is not None:
        config['learning_rate'] = args.lr
        print(f"  ⚠️  Override: learning_rate = {args.lr}")
    if args.dropout is not None:
        config['dropout'] = args.dropout
        print(f"  ⚠️  Override: dropout = {args.dropout}")
    if args.seq_len is not None:
        config['seq_len'] = args.seq_len
        print(f"  ⚠️  Override: seq_len = {args.seq_len}")
    if args.hidden_dim is not None:
        config['hidden_dim'] = args.hidden_dim
        print(f"  ⚠️  Override: hidden_dim = {args.hidden_dim}")
    if args.num_layers is not None:
        config['num_layers'] = args.num_layers
        print(f"  ⚠️  Override: num_layers = {args.num_layers}")
    if args.num_heads is not None:
        config['num_heads'] = args.num_heads
        print(f"  ⚠️  Override: num_heads = {args.num_heads}")
    if args.lambda_phi is not None:
        config['lambda_phi'] = args.lambda_phi
        print(f"  ⚠️  Override: lambda_phi = {args.lambda_phi}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar datasets con tokenizer correspondiente
    print(f"\n📚 Cargando datasets...")
    
    train_dataset = WikiText2RealDataset(
        split='train',
        seq_len=config['seq_len'],
        tokenizer=None,  # Se creará internamente
        use_llama3=args.use_llama3
    )
    
    val_dataset = WikiText2RealDataset(
        split='validation',
        seq_len=config['seq_len'],
        tokenizer=train_dataset.tokenizer,  # Compartir tokenizer
        use_llama3=args.use_llama3
    )
    
    # Determinar vocab_size (del dataset o del preset)
    if config['vocab_size'] is None:
        vocab_size = train_dataset.vocab_size
        print(f"  ✓ Vocabulario dinámico: {vocab_size:,} tokens (desde tokenizer)")
    else:
        vocab_size = min(config['vocab_size'], train_dataset.vocab_size)
        print(f"  ✓ Vocabulario limitado: {vocab_size:,} tokens (preset: {config['vocab_size']}, disponible: {train_dataset.vocab_size})")
    
    # Actualizar config con vocab_size final
    config['vocab_size'] = vocab_size
    
    # Crear modelo
    print(f"\n🤖 Creando modelo INFINITO V5.2 CON IIT MEJORADO...")
    print(f"  📋 Configuración final: hidden_dim={config['hidden_dim']}, layers={config['num_layers']}, heads={config['num_heads']}, vocab={vocab_size:,}")
    
    model = InfinitoV52Refactored(
        vocab_size=vocab_size,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        memory_slots=256,
        dropout=config['dropout'],           # 🆕 Dropout desde config
        use_improved_memory=True,      # ✅ IITGuidedMemory
        use_improved_iit=True,         # ✅ ImprovedIITMetrics (4 componentes)
        use_learnable_phi=True,        # ✅ LearnablePhiWeights
        use_stochastic_exploration=True,
        lambda_phi=config['lambda_phi'],    # 🆕 Peso desde config
        seed=args.seed
    ).to(device)
    
    # Cargar checkpoint si se especifica --resume
    start_epoch = 0
    if args.resume:
        print(f"\n📂 Cargando checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"  ✓ Continuando desde época {start_epoch}")
    
    # Contar parámetros
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Parámetros: {num_params:,}")
    
    # Crear trainer
    trainer = InfinitoTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        device=device,
        start_epoch=start_epoch,
        use_early_stopping=not args.no_early_stopping,  # 🆕 Early stopping
        patience=args.patience,                         # 🆕 Paciencia configurable
        use_plateau_scheduler=not args.no_plateau_scheduler  # 🆕 ReduceLROnPlateau
    )
    
    # Entrenar
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
