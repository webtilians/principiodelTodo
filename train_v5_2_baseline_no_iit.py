#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 BASELINE TRANSFORMER - COMPARACIÓN CIENTÍFICA CON INFINITO V5.2
=================================================================

Script para entrenar transformer ESTÁNDAR sin características IIT
usando EXACTAMENTE la misma configuración que train_v5_2_wikitext_real.py

OBJETIVO:
Comparación científica directa entre:
- Model CON IIT: InfinitoV52Refactored (features IIT activas)
- Model SIN IIT: TransformerBaseline (transformer estándar)

CONFIGURACIÓN IDÉNTICA:
✅ WikiText-2 REAL (HuggingFace datasets)  
✅ GPT2Tokenizer (50,257 tokens)
✅ hidden_dim=512, num_layers=4, num_heads=8
✅ dropout=0.15, learning_rate=5e-4, batch_size=16
✅ lambda_phi=0.3 (ignorado en baseline)
✅ Early stopping, ReduceLROnPlateau, etc.

DIFERENCIAS:
❌ SIN IITGuidedMemory
❌ SIN ImprovedIITMetrics  
❌ SIN LearnablePhiWeights
❌ SIN StochasticExploration
❌ SIN DeltaPhiObjective
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
    
    Arquitectura IDÉNTICA a InfinitoV52Refactored pero sin:
    - IITGuidedMemory
    - LearnablePhiWeights  
    - StochasticExploration
    - ImprovedIITMetrics
    - DeltaPhiObjective
    """
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.15,
        max_seq_len=512,
        seed=42
    ):
        super().__init__()
        
        # 🔒 Fijar seeds para reproducibilidad (IGUAL que InfinitoV52)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.seed = seed
        
        print(f"\n{'='*70}")
        print(f"TRANSFORMER BASELINE (SIN CARACTERÍSTICAS IIT)")
        print(f"{'='*70}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Num layers: {num_layers}")
        print(f"  Num heads: {num_heads}")
        print(f"  Dropout: {dropout}")
        print(f"  Vocab size: {vocab_size:,}")
        print(f"  Max seq len: {max_seq_len}")
        if seed is not None:
            print(f"  [SEED] Fijado: {seed} (reproducibilidad garantizada)")
        print(f"{'='*70}\n")
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Transformer decoder layers estándar
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
        
        # Inicialización estilo GPT-2
        self.apply(self._init_weights)
        
        # Contar parámetros
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def _init_weights(self, module):
        """Inicialización estilo GPT-2 (IGUAL que InfinitoV52)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, return_metrics=False):
        """
        Forward pass estándar.
        
        Args:
            input_ids: [batch_size, seq_len]
            return_metrics: Si True, retorna (logits, metrics) para compatibilidad
            
        Returns:
            Si return_metrics=False: logits [batch_size, seq_len, vocab_size]
            Si return_metrics=True: (logits, None) - sin métricas IIT
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
        logits = x @ self.output_proj.weight.T  # Usar tied weights
        
        if return_metrics:
            # Compatibilidad con InfinitoTrainer - sin métricas IIT
            return logits, None
        
        return logits


# =============================================================================
# EARLY STOPPING (COPIADO DE train_v5_2_wikitext_real.py)
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
# DATASET (COPIADO EXACTO DE train_v5_2_wikitext_real.py)
# =============================================================================

class WikiText2RealDataset(Dataset):
    """
    Dataset WikiText-2 REAL - IDÉNTICO al usado en InfinitoV52.
    
    Características:
    - Datos reales de Wikipedia (HuggingFace datasets)
    - Tokenización GPT-2 BPE (50,257 tokens)
    - Secuencias de 256 tokens con shift correcto
    """
    
    def __init__(self, split='train', seq_len=256, tokenizer=None, use_llama3=False):
        """
        Args:
            split: 'train', 'validation', o 'test'
            seq_len: Longitud de las secuencias  
            tokenizer: Tokenizer instance (si None, se crea uno nuevo)
            use_llama3: Si True, usa Llama 3; si False, GPT-2 (default False para baseline)
        """
        self.seq_len = seq_len
        self.use_llama3 = use_llama3
        
        # Cargar tokenizer (siempre GPT-2 para baseline, para comparación justa)
        if tokenizer is None:
            print(f"\n🔤 Cargando GPT-2 Tokenizer (BASELINE)...")
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.use_llama3 = False  # Forzar GPT-2
        else:
            self.tokenizer = tokenizer
        
        self.vocab_size = len(self.tokenizer)
        print(f"  ✓ Vocabulario: {self.vocab_size:,} tokens (GPT-2 BPE)")
        
        # Cargar WikiText-2 real (IDÉNTICO proceso)
        print(f"\n📚 Cargando WikiText-2 REAL ({split})...")
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, trust_remote_code=True)
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
# TRAINER (ADAPTADO DE InfinitoTrainer pero SIN MÉTRICAS IIT)
# =============================================================================

class BaselineTrainer:
    """Entrenador para Transformer Baseline (SIN IIT) - API idéntica a InfinitoTrainer."""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        batch_size=16,
        learning_rate=5e-4,
        device='cuda',
        start_epoch=0,
        use_early_stopping=True,
        patience=5,
        use_plateau_scheduler=True
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.device = device
        self.start_epoch = start_epoch
        self.use_early_stopping = use_early_stopping
        self.use_plateau_scheduler = use_plateau_scheduler
        
        # DataLoaders (IDÉNTICOS)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        # Optimizer (IDÉNTICO a InfinitoTrainer)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),  # ✅ ÓPTIMO (match con Model A)
            eps=1e-9,
            weight_decay=0.01  # ✅ ÓPTIMO (Regularización L2)
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler (IDÉNTICO)
        if use_plateau_scheduler:
            print("  🔧 Usando ReduceLROnPlateau (adaptativo)")
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=True
            )
        else:
            print("  🔧 Usando CosineAnnealingLR (fijo)")
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=20,
                eta_min=1e-6
            )
        
        # Early Stopping (IDÉNTICO)
        if use_early_stopping:
            print(f"  ⏹️  Early Stopping activado (patience={patience})")
            self.early_stopping = EarlyStopping(patience=patience, min_delta=0.01, mode='min')
        else:
            self.early_stopping = None
        
        # History (SIN métricas IIT)
        self.history = {
            'train_loss': [],
            'train_perplexity': [],
            'val_loss': [],
            'val_perplexity': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch):
        """Entrena una época SIN métricas IIT."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Época {epoch}')
        
        for input_ids, labels in pbar:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass SIN métricas IIT
            logits = self.model(input_ids, return_metrics=False)
            
            # Loss de language modeling únicamente
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
        """Valida el modelo SIN métricas IIT."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validación')
            
            for input_ids, labels in pbar:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
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
        print(f"INICIANDO ENTRENAMIENTO - BASELINE (SIN IIT)")
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
        
        for epoch in range(self.start_epoch + 1, num_epochs + 1):
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
            
            # Mostrar resultados (SIN métricas IIT)
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
                    os.path.join(save_dir, 'baseline_no_iit_best.pt')
                )
                print(f"  ✅ MEJOR MODELO guardado (val_loss: {val_loss:.4f})")
            
            # Guardar checkpoint cada 5 épocas
            if epoch % 5 == 0:
                self.save_checkpoint(
                    epoch,
                    val_loss,
                    val_ppl,
                    os.path.join(save_dir, f'baseline_no_iit_epoch_{epoch}.pt')
                )
            
            # Actualizar learning rate
            if self.use_plateau_scheduler:
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Early Stopping
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
        print(f"✅ ENTRENAMIENTO BASELINE COMPLETADO")
        print(f"{'='*70}")
        print(f"  Mejor Val Loss: {best_val_loss:.4f}")
        print(f"  Mejor Val PPL: {math.exp(best_val_loss):,.2f}")
        print(f"{'='*70}\n")
        
        # Comparación con resultados conocidos de InfinitoV52
        print(f"\n🔬 COMPARACIÓN CIENTÍFICA:")
        print(f"  Baseline (SIN IIT):  {math.exp(best_val_loss):.2f} PPL")
        print(f"  Model A (CON IIT):   216.46 PPL")
        print(f"  Model B (CON IIT):   207.15 PPL")
        print()
        
        baseline_ppl = math.exp(best_val_loss)
        if baseline_ppl > 216.46:
            print("✅ CONCLUSIÓN: IIT aporta beneficio significativo")
        elif baseline_ppl > 207.15:
            print("🟡 CONCLUSIÓN: IIT aporta beneficio moderado")
        else:
            print("❌ CONCLUSIÓN: IIT no mejora o perjudica performance")
    
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
                'dropout': self.model.dropout,
                'seed': self.model.seed,
                'model_type': 'baseline_no_iit'
            }
        }
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Guarda el historial de entrenamiento."""
        os.makedirs('results/training', exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_path = f'results/training/baseline_no_iit_history_{timestamp}.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n💾 Historial guardado: {history_path}")


# =============================================================================
# BASELINE MODEL CONFIGURATIONS (PRESETS)
# =============================================================================

MODEL_CONFIGS = {
    "large_baseline": {
        # Configuración baseline equivalente a large_iit
        "hidden_dim": 512,
        "num_layers": 4,
        "num_heads": 8,
        "dropout": 0.15,
        "seq_len": 256,
        "batch_size": 16,
        "lr": 5e-4,
        "description": "Baseline transformer grande para comparación con large_iit"
    },
    "small_baseline": {
        # Configuración baseline equivalente a small_iit
        "hidden_dim": 384,
        "num_layers": 3,
        "num_heads": 6,
        "dropout": 0.15,
        "seq_len": 256,
        "batch_size": 16,
        "lr": 5e-4,
        "description": "Baseline transformer pequeño para comparación con small_iit"
    }
}


# =============================================================================
# MAIN (API IDÉNTICA A train_v5_2_wikitext_real.py)
# =============================================================================

def main():
    """Función principal de entrenamiento."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BASELINE Transformer (SIN IIT) - Comparación científica con INFINITO V5.2')
    
    # 🆕 PRESET DE CONFIGURACIÓN BASELINE
    parser.add_argument('--model-size', type=str, default='large_baseline',
                       choices=['large_baseline', 'small_baseline'],
                       help='Preset de configuración del modelo baseline (default: large_baseline)')
    
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
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed para reproducibilidad (default: 42)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Paciencia para early stopping (default: 5 - ÓPTIMO)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Desactivar early stopping')
    parser.add_argument('--no-plateau-scheduler', action='store_true',
                       help='Usar CosineAnnealingLR en vez de ReduceLROnPlateau')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path al checkpoint para continuar entrenamiento')
    
    args = parser.parse_args()
    
    # 🚀 CARGAR CONFIGURACIÓN DESDE PRESET BASELINE
    config = MODEL_CONFIGS[args.model_size].copy()
    print(f"\n🔧 Using baseline preset: {args.model_size} -> {config['description']}")
    print(f"📋 Configuration: {config}")
    
    # Permitir override manual de cualquier parámetro
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
        print(f"  ⚠️  Override: batch_size = {args.batch_size}")
    if args.lr is not None:
        config['lr'] = args.lr
        print(f"  ⚠️  Override: lr = {args.lr}")
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
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Cargar datasets (IDÉNTICOS a InfinitoV52)
    print(f"\n📊 Cargando datasets...")
    
    train_dataset = WikiText2RealDataset(
        split='train',
        seq_len=config['seq_len'],
        tokenizer=None,
        use_llama3=False  # Siempre GPT-2 para baseline
    )
    
    val_dataset = WikiText2RealDataset(
        split='validation',
        seq_len=config['seq_len'],
        tokenizer=train_dataset.tokenizer,
        use_llama3=False
    )
    
    vocab_size = train_dataset.vocab_size
    
    # Crear modelo BASELINE (sin IIT)
    print(f"\n🤖 Creando modelo BASELINE (SIN IIT)...")
    print(f"  📋 Configuración final: hidden_dim={config['hidden_dim']}, layers={config['num_layers']}, heads={config['num_heads']}, vocab={vocab_size:,}")
    
    model = TransformerBaseline(
        vocab_size=vocab_size,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        max_seq_len=512,
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
    trainer = BaselineTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config['batch_size'],
        learning_rate=config['lr'],
        device=device,
        start_epoch=start_epoch,
        use_early_stopping=not args.no_early_stopping,
        patience=args.patience,
        use_plateau_scheduler=not args.no_plateau_scheduler
    )
    
    # Entrenar
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()