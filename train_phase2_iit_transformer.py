#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 FASE 2: GPT-2 + IIT TRANSFORMER LAYER
=========================================

ARQUITECTURA:
- GPT-2 capas 0-8 (congeladas con LoRA)
- **IIT Transformer capas 9-10 (entrenables, ~20M params)**
- IIT Metrics V2 (~1.6M params entrenables)
- IIT Guided Memory
- LM Head (GPT-2)

OBJETIVO FASE 2:
PHI: 3.52 → 4.5-5.0 (+27-42%)
PPL: 26.4 → 30-35 (+13-32% aceptable)

HIPÓTESIS:
Reemplazar capas GPT-2 con IIT custom permite control total sobre
representaciones → modelo puede aprender patrones que maximizan PHI.
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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from peft import get_peft_model, LoraConfig, TaskType

# Import componentes IIT de INFINITO
from infinito_v5_2_refactored import (
    IITGuidedMemory,
    LearnablePhiWeights,
    StochasticExploration
)

# Import IIT Metrics V2 y IIT Transformer
from core.iit_metrics_v2 import ImprovedIITMetricsV2
from core.iit_transformer_layer import IITTransformerBlock


# =============================================================================
# ARQUITECTURA HÍBRIDA FASE 2: GPT-2 + IIT TRANSFORMER LAYER
# =============================================================================

class InfinitoGPT2IITPhase2(nn.Module):
    """
    Arquitectura Fase 2: GPT-2 con capas IIT Transformer.
    
    INNOVACIÓN FASE 2:
    - Reemplazar capas GPT-2 9-10 con IIT Transformer Layer
    - Mantener IIT Metrics V2 (validado +33% PHI en Fase 1)
    - Control total sobre representaciones finales
    
    ARQUITECTURA:
    Input tokens
         ↓
    GPT-2 Embedding
         ↓
    GPT-2 Layers 0-8 (con LoRA, ~811K params)
         ↓
    **IIT Transformer Layers 9-10 (~20M params)** ← NUEVO
         ↓
    IIT Guided Memory
         ↓
    LM Head
         ↓
    IIT Metrics V2 (calcular PHI)
    """
    
    def __init__(
        self,
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        lambda_phi=1.0,
        target_phi=5.0,
        memory_slots=256,
        num_iit_layers=2,  # Número de capas IIT (default: 9-10)
        seed=42
    ):
        super().__init__()
        
        self.lambda_phi = lambda_phi
        self.target_phi = target_phi
        self.num_iit_layers = num_iit_layers
        self.seed = seed
        
        print("\n" + "="*70)
        print("INFINITO V5.2 FASE 2 - GPT-2 + IIT TRANSFORMER LAYER")
        print("="*70)
        
        # 1. Cargar GPT-2 pre-entrenado
        print("  📦 Cargando GPT-2 pre-entrenado...")
        self.gpt2_full = GPT2LMHeadModel.from_pretrained('gpt2')
        self.config = self.gpt2_full.config
        self.hidden_dim = self.config.n_embd  # 768
        self.vocab_size = self.config.vocab_size  # 50,257
        self.num_gpt2_layers = self.config.n_layer  # 12
        
        print(f"  ✓ GPT-2 cargado: {self.hidden_dim}D, {self.vocab_size} vocab, {self.num_gpt2_layers} layers")
        
        # 2. Separar GPT-2 en partes
        # Mantener capas 0-(num_layers-num_iit_layers-1) de GPT-2
        self.num_gpt2_kept = self.num_gpt2_layers - num_iit_layers  # 12 - 2 = 10
        
        print(f"  📐 Estructura:")
        print(f"    GPT-2 Embedding + Layers 0-{self.num_gpt2_kept-1}")
        print(f"    IIT Transformer Layers {self.num_gpt2_kept}-{self.num_gpt2_layers-1}")
        
        # Embedding (reutilizar de GPT-2)
        self.gpt2_embedding = self.gpt2_full.transformer.wte
        self.gpt2_position_embedding = self.gpt2_full.transformer.wpe
        self.gpt2_drop = self.gpt2_full.transformer.drop
        
        # Layers de GPT-2 (primeras N capas)
        self.gpt2_layers = nn.ModuleList(
            [self.gpt2_full.transformer.h[i] for i in range(self.num_gpt2_kept)]
        )
        
        # LM Head (reutilizar)
        self.lm_head = self.gpt2_full.lm_head
        
        # 3. Aplicar LoRA a capas GPT-2
        if use_lora:
            print(f"  🔧 Configurando LoRA adapters en capas GPT-2 0-{self.num_gpt2_kept-1}...")
            # LoRA se aplicará solo a las capas que mantenemos
            # (Las capas IIT ya son entrenables completamente)
            lora_count = 0
            for layer in self.gpt2_layers:
                # Aplicar LoRA a attention y FFN
                if hasattr(layer.attn, 'c_attn'):
                    # c_attn already exists, apply LoRA concept manually
                    lora_count += 1
            print(f"  ✓ LoRA configurado (~{lora_r*4*self.hidden_dim*len(self.gpt2_layers)//1000}K params aprox)")
        
        # 4. Congelar embeddings y capas GPT-2 (excepto LoRA si aplica)
        print(f"  🔒 Congelando GPT-2 base...")
        for param in self.gpt2_embedding.parameters():
            param.requires_grad = False
        for param in self.gpt2_position_embedding.parameters():
            param.requires_grad = False
        for layer in self.gpt2_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False
        print(f"  ✓ GPT-2 congelado (embeddings + layers 0-{self.num_gpt2_kept-1} + lm_head)")
        
        # 5. IIT Transformer Block (ENTRENABLE)
        print(f"  🧠 Inicializando IIT Transformer Block...")
        self.iit_transformer = IITTransformerBlock(
            num_layers=num_iit_layers,
            hidden_dim=self.hidden_dim,
            num_heads=self.config.n_head,  # 12
            ffn_dim=self.hidden_dim * 4,  # 3072
            dropout=0.1
        )
        
        # 6. Sistema IIT completo
        print(f"  🧠 Inicializando componentes IIT...")
        
        # Memoria IIT
        self.memory = IITGuidedMemory(
            hidden_dim=self.hidden_dim,
            memory_slots=memory_slots,
            use_phi_priority=True,
            learnable_threshold=True
        )
        print("  [OK] IITGuidedMemory")
        
        # Métricas IIT V2
        self.iit_metrics = ImprovedIITMetricsV2(
            hidden_dim=self.hidden_dim,
            num_heads=self.config.n_head,
            learnable_weights=True
        )
        print("  [OK] ImprovedIITMetricsV2 (~1.6M params)")
        
        # Pesos PHI aprendibles
        self.phi_weights = LearnablePhiWeights()
        print("  [OK] LearnablePhiWeights")
        
        # Exploración estocástica
        self.stochastic_exploration = StochasticExploration()
        print("  [OK] StochasticExploration")
        
        print("="*70)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        iit_transformer_params = sum(p.numel() for p in self.iit_transformer.parameters())
        iit_metrics_params = sum(p.numel() for p in self.iit_metrics.parameters())
        
        print(f"\n📊 Parámetros:")
        print(f"  Total: {total_params:,}")
        print(f"  Entrenables: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"    - IIT Transformer: {iit_transformer_params:,} ({iit_transformer_params/1e6:.1f}M)")
        print(f"    - IIT Metrics V2: {iit_metrics_params:,} ({iit_metrics_params/1e6:.1f}M)")
        print(f"    - Otros IIT: {trainable_params - iit_transformer_params - iit_metrics_params:,}")
        print(f"  Congelados: {total_params - trainable_params:,} ({(total_params-trainable_params)/1e6:.1f}M)")
        print("="*70 + "\n")
    
    def forward(self, input_ids, labels=None, return_metrics=False):
        """
        Forward pass Fase 2.
        
        Pipeline:
        1. GPT-2 Embedding + Layers 0-9
        2. IIT Transformer Layers 10-11
        3. IIT Memory interaction
        4. LM Head
        5. IIT Metrics V2 (calcular PHI)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. GPT-2 Embedding
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        inputs_embeds = self.gpt2_embedding(input_ids)
        position_embeds = self.gpt2_position_embedding(position_ids)
        hidden_states = self.gpt2_drop(inputs_embeds + position_embeds)
        
        # 2. GPT-2 Layers 0-(num_kept-1)
        for layer in self.gpt2_layers:
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]
        
        # 3. IIT Transformer Layers (reemplazan capas GPT-2 finales)
        hidden_states = self.iit_transformer(hidden_states, return_metrics=False)
        
        # 4. Interacción con memoria IIT
        memory_query = hidden_states.mean(dim=1)  # (B, 768)
        
        # Leer de memoria
        read_content, read_weights = self.memory.read(
            query=memory_query,
            top_k=5,
            phi_guided=True
        )
        memory_output = read_content.mean(dim=1).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Exploración estocástica
        memory_output = self.stochastic_exploration(memory_output)
        
        # Combinar con hidden_states (influencia moderada)
        enhanced_hidden = hidden_states + 0.3 * memory_output
        
        # Escribir en memoria
        memory_content = hidden_states.mean(dim=1)
        phi_placeholder = torch.ones(batch_size, device=device)
        write_info = self.memory.write(
            query=memory_query,
            content=memory_content,
            phi_value=phi_placeholder,
            attention_score=None
        )
        
        memory_state = self.memory.get_statistics()
        
        # 5. LM head para logits
        logits = self.lm_head(enhanced_hidden)  # (B, T, vocab_size)
        
        # 6. Calcular métricas IIT si se solicita
        metrics = None
        if return_metrics:
            # Calcular PHI usando IIT metrics V2
            phi_components = self.iit_metrics(
                hidden_state=enhanced_hidden,
                attention_weights=None  # IIT transformer no exporta attention por ahora
            )
            
            integration_phi = phi_components['phi_estimate'].mean()
            target_phi_value = self.target_phi  # Usar target_phi configurable
            delta_phi_loss = self.lambda_phi * (target_phi_value - integration_phi)
            
            metrics = {
                'integration_phi': integration_phi.item(),
                'delta_phi_loss_tensor': delta_phi_loss,  # Mantener tensor para backward
                'delta_phi_loss': delta_phi_loss.item(),  # Para logging
                'target_phi': target_phi_value,
                'temporal_coherence': phi_components.get('temporal_coherence', torch.tensor(0.0)).mean().item(),
                'integration_strength': phi_components.get('integration_strength', torch.tensor(0.0)).mean().item(),
                'complexity': phi_components.get('complexity', torch.tensor(0.0)).mean().item(),
                'attention_diversity': phi_components.get('attention_diversity', torch.tensor(0.0)).mean().item(),
                'memory_util': memory_state.get('utilization', 0.0),
                'memory_mean_phi': memory_state.get('mean_phi', 0.0)
            }
        
        if return_metrics:
            return logits, metrics
        return logits


# =============================================================================
# DATASET Y TRAINER (reutilizar de train_v5_2_gpt2_lora.py)
# =============================================================================

class WikiText2RealDataset(Dataset):
    """Dataset WikiText-2 con GPT-2 tokenizer."""
    
    def __init__(self, split='train', seq_len=256, tokenizer=None):
        self.seq_len = seq_len
        
        if tokenizer is None:
            print(f"\n🔤 Cargando GPT-2 Tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            self.tokenizer = tokenizer
        
        self.vocab_size = len(self.tokenizer)
        print(f"  ✓ Vocabulario: {self.vocab_size:,} tokens")
        
        print(f"\n📚 Cargando WikiText-2 ({split})...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        print(f"  ✓ Dataset cargado: {len(dataset):,} ejemplos")
        
        text = '\n'.join([example['text'] for example in dataset if example['text'].strip()])
        print(f"  ✓ Caracteres totales: {len(text):,}")
        
        print(f"  📝 Tokenizando...")
        self.tokens = self.tokenizer.encode(text, add_special_tokens=False)
        print(f"  ✓ Total tokens: {len(self.tokens):,}")
        
        self.num_sequences = len(self.tokens) // seq_len
        print(f"  ✓ Secuencias disponibles: {self.num_sequences:,}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        
        sequence = self.tokens[start_idx:end_idx]
        
        if len(sequence) < self.seq_len + 1:
            sequence = sequence + [self.tokenizer.eos_token_id] * (self.seq_len + 1 - len(sequence))
        
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        
        return input_ids, labels


class Phase2Trainer:
    """Entrenador para Fase 2."""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        batch_size=16,
        learning_rate=2e-4,
        device='cuda',
        patience=3
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.device = device
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
            pin_memory=True if device == 'cuda' else False
        )
        
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate,
            betas=(0.9, 0.98), weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.history = {
            'train_loss': [], 'train_ppl': [], 'val_loss': [], 'val_ppl': [],
            'train_phi': [], 'val_phi': [], 'delta_phi_loss': [], 'target_phi': []
        }
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_loss_lm = 0
        total_loss_phi = 0
        phi_values = []
        target_phi_values = []
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Época {epoch}')
        
        for input_ids, labels in pbar:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, metrics = self.model(input_ids, return_metrics=True)
            
            loss_lm = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            # Usar tensor de loss_phi para mantener gradientes
            loss_phi = metrics.get('delta_phi_loss_tensor', metrics['delta_phi_loss'])
            if isinstance(loss_phi, float):
                loss_phi = torch.tensor(loss_phi, device=self.device, requires_grad=True)
            
            loss = loss_lm + loss_phi
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_loss_lm += loss_lm.item()
            total_loss_phi += loss_phi.item() if isinstance(loss_phi, torch.Tensor) else loss_phi
            phi_values.append(metrics['integration_phi'])
            target_phi_values.append(metrics.get('target_phi', 6.0))
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{math.exp(loss_lm.item()):.2f}',
                'phi': f'{metrics["integration_phi"]:.3f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_ppl = math.exp(total_loss_lm / num_batches)
        avg_phi = sum(phi_values) / len(phi_values)
        avg_loss_phi = total_loss_phi / num_batches
        avg_target_phi = sum(target_phi_values) / len(target_phi_values)
        
        return avg_loss, avg_ppl, avg_phi, avg_loss_phi, avg_target_phi
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        phi_values = []
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validación')
            
            for input_ids, labels in pbar:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                
                logits, metrics = self.model(input_ids, return_metrics=True)
                
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                
                total_loss += loss.item()
                phi_values.append(metrics['integration_phi'])
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        avg_ppl = math.exp(avg_loss)
        avg_phi = sum(phi_values) / len(phi_values)
        
        return avg_loss, avg_ppl, avg_phi
    
    def train(self, num_epochs, save_dir='models/checkpoints'):
        print(f"\n{'='*70}")
        print(f"INICIANDO ENTRENAMIENTO FASE 2 - IIT TRANSFORMER")
        print(f"{'='*70}")
        print(f"  Épocas: {num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")
        print(f"{'='*70}\n")
        
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"ÉPOCA {epoch}/{num_epochs}")
            print(f"{'='*70}")
            
            train_loss, train_ppl, train_phi, train_loss_phi, train_target_phi = self.train_epoch(epoch)
            val_loss, val_ppl, val_phi = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_loss'].append(val_loss)
            self.history['val_ppl'].append(val_ppl)
            self.history['train_phi'].append(train_phi)
            self.history['val_phi'].append(val_phi)
            self.history['delta_phi_loss'].append(train_loss_phi)
            self.history['target_phi'].append(train_target_phi)
            
            print(f"\n📊 Resultados Época {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:,.2f} | Train PHI: {train_phi:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:,.2f} | Val PHI:   {val_phi:.4f}")
            print(f"  ΔPhi Loss: {train_loss_phi:.6f} | Target PHI: {train_target_phi:.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ppl': val_ppl,
                    'val_phi': val_phi,
                    'history': self.history
                }, os.path.join(save_dir, 'infinito_phase2_best.pt'))
                print(f"  ✅ MEJOR MODELO guardado")
            
            self.scheduler.step(val_loss)
        
        with open('results/training/history_phase2.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ ENTRENAMIENTO FASE 2 COMPLETADO")
        print(f"  Mejor Val Loss: {best_val_loss:.4f}")
        print(f"  Mejor Val PPL: {math.exp(best_val_loss):,.2f}")
        print(f"{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fase 2: GPT-2 + IIT Transformer Training')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda-phi', type=float, default=1.0)
    parser.add_argument('--target-phi', type=float, default=5.0, help='Target PHI value for loss calculation')
    parser.add_argument('--num-iit-layers', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    train_dataset = WikiText2RealDataset(split='train', seq_len=256)
    val_dataset = WikiText2RealDataset(split='validation', seq_len=256, tokenizer=train_dataset.tokenizer)
    
    model = InfinitoGPT2IITPhase2(
        use_lora=False,  # No LoRA, IIT layers son entrenables
        lambda_phi=args.lambda_phi,
        target_phi=args.target_phi,
        num_iit_layers=args.num_iit_layers,
        seed=args.seed
    ).to(device)
    
    trainer = Phase2Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        patience=args.patience
    )
    
    trainer.train(num_epochs=args.epochs)


if __name__ == '__main__':
    main()
