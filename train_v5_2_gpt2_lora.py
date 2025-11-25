#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ENTRENAMIENTO HÍBRIDO: GPT-2 + INFINITO IIT con LoRA
========================================================

Script de entrenamiento que combina:
- GPT-2 pre-entrenado (baseline de lenguaje)
- Sistema IIT completo de INFINITO V5.2
- Adaptadores LoRA (eficiente)

OBJETIVO CIENTÍFICO:
Comparar PHI antes/después de añadir sistema IIT a modelo pre-entrenado.

HIPÓTESIS:
PHI_GPT2+IIT > PHI_GPT2_vanilla → Sistema IIT aumenta consciencia
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

# Import IIT Metrics V2 (ENTRENABLE) - ya configurado sys.path arriba
from core.iit_metrics_v2 import ImprovedIITMetricsV2


# =============================================================================
# ARQUITECTURA HÍBRIDA: GPT-2 + IIT
# =============================================================================

class InfinitoGPT2Hybrid(nn.Module):
    """
    Arquitectura híbrida que combina GPT-2 pre-entrenado con sistema IIT.
    
    Componentes:
    - GPT-2 base (con LoRA adapters opcionales)
    - IITGuidedMemory: Memoria guiada por PHI
    - ImprovedIITMetrics: 4 componentes IIT
    - LearnablePhiWeights: Pesos aprendibles
    - StochasticExploration: Exploración estocástica
    
    Modos:
    - freeze_base=True: Solo entrena IIT + LoRA (rápido)
    - freeze_base=False: Fine-tuning completo (lento)
    """
    
    def __init__(
        self,
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        freeze_base=True,
        lambda_phi=0.3,
        memory_slots=256,
        seed=42
    ):
        super().__init__()
        
        self.use_lora = use_lora
        self.freeze_base = freeze_base
        self.lambda_phi = lambda_phi
        self.seed = seed
        
        # RL: Pesos de loss dinámicos (para control por agente RL)
        self.loss_weights = {
            "text": 1.0,
            "phi": lambda_phi,  # Inicializar con lambda_phi
        }
        
        # Métricas actuales (para RL environment)
        self.current_metrics = {
            "consciousness": 0.0,
            "phi": 0.0,
            "loss_text": 0.0,
            "loss_phi": 0.0,
            "memory_utilization": 0.0,
            "perplexity": 0.0,
        }
        
        print("\n" + "="*70)
        print("INFINITO V5.2 + GPT-2 - ARQUITECTURA HÍBRIDA (RL-Ready)")
        print("="*70)
        
        # 1. Cargar GPT-2 pre-entrenado
        print("  📦 Cargando GPT-2 pre-entrenado...")
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.config = self.gpt2.config
        self.hidden_dim = self.config.n_embd  # 768 para GPT-2 small
        self.vocab_size = self.config.vocab_size  # 50,257
        
        print(f"  ✓ GPT-2 cargado: {self.hidden_dim}D, {self.vocab_size} vocab")
        
        # 2. Aplicar LoRA si está activado
        if use_lora:
            print(f"  🔧 Configurando LoRA adapters (r={lora_r}, alpha={lora_alpha})...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["c_attn", "c_proj"],  # Attention + projection
                bias="none"
            )
            self.gpt2 = get_peft_model(self.gpt2, lora_config)
            self.gpt2.print_trainable_parameters()
        
        # 3. Congelar base si es necesario
        if freeze_base and not use_lora:
            print("  🔒 Congelando GPT-2 base...")
            for param in self.gpt2.parameters():
                param.requires_grad = False
            print(f"  ✓ GPT-2 congelado (solo entrenar IIT)")
        
        # 4. Sistema IIT completo
        print("  🧠 Inicializando sistema IIT...")
        
        # Memoria IIT
        self.memory = IITGuidedMemory(
            hidden_dim=self.hidden_dim,
            memory_slots=memory_slots,
            use_phi_priority=True,
            learnable_threshold=True
        )
        print("  [OK] IITGuidedMemory (priorización por PHI)")
        
        # Métricas IIT V2 (ENTRENABLE con ~1.6M parámetros)
        self.iit_metrics = ImprovedIITMetricsV2(
            hidden_dim=self.hidden_dim,
            num_heads=self.config.n_head,  # 12 para GPT-2
            learnable_weights=True
        )
        print("  [OK] ImprovedIITMetricsV2 (~1.6M params entrenables)")
        
        # Pesos PHI aprendibles
        self.phi_weights = LearnablePhiWeights()
        print("  [OK] LearnablePhiWeights")
        
        # Exploración estocástica
        self.stochastic_exploration = StochasticExploration()
        print("  [OK] StochasticExploration")
        
        # Proyección de salida (si es necesario)
        # GPT-2 ya tiene lm_head, así que no necesitamos otra
        
        print("="*70)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n📊 Parámetros:")
        print(f"  Total: {total_params:,}")
        print(f"  Entrenables: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  Congelados: {total_params - trainable_params:,}")
        print("="*70 + "\n")
    
    def forward(self, input_ids, labels=None, return_metrics=False):
        """
        Forward pass con sistema IIT integrado.
        
        Args:
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len) opcional para calcular loss
            return_metrics: Si True, retorna métricas IIT
        
        Returns:
            Si return_metrics=False: logits
            Si return_metrics=True: (logits, metrics_dict)
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. GPT-2 transformer (sin head final)
        transformer_outputs = self.gpt2.transformer(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=True
        )
        
        hidden_states = transformer_outputs.last_hidden_state  # (B, T, 768)
        all_hidden_states = transformer_outputs.hidden_states  # Tuple de capas
        attentions = transformer_outputs.attentions  # Tuple de attention weights
        
        # 2. Interacción con memoria IIT (read + write)
        # Memory query: usar hidden state promedio
        memory_query = hidden_states.mean(dim=1)  # (B, 768)
        
        # Leer de memoria (top-5 memories guiadas por PHI)
        read_content, read_weights = self.memory.read(
            query=memory_query,
            top_k=5,
            phi_guided=True
        )
        # read_content: (B, 5, 768) -> mean -> (B, 768)
        memory_output = read_content.mean(dim=1).unsqueeze(1)  # (B, 1, 768)
        
        # Expandir a seq_len para residual connection
        memory_output = memory_output.expand(-1, seq_len, -1)  # (B, T, 768)
        
        # 3. Exploración estocástica
        memory_output = self.stochastic_exploration(memory_output)
        
        # 4. Combinar con hidden_states original (residual)
        # 🔥 AUMENTADO: 0.4 (4x más influencia IIT vs 0.1 original)
        enhanced_hidden = hidden_states + 0.4 * memory_output
        
        # 5. Escribir en memoria (para próximas iteraciones)
        # Usar mean del hidden state como contenido
        memory_content = hidden_states.mean(dim=1)  # (B, 768)
        # PHI placeholder (se calculará después)
        phi_placeholder = torch.ones(batch_size, device=input_ids.device)
        write_info = self.memory.write(
            query=memory_query,
            content=memory_content,
            phi_value=phi_placeholder,
            attention_score=None
        )
        
        # memory_state para IIT metrics: usar estadísticas de memoria
        memory_state = self.memory.get_statistics()
        
        # 5. LM head para logits
        logits = self.gpt2.lm_head(enhanced_hidden)  # (B, T, vocab_size)
        
        # 6. Calcular métricas IIT si se solicita
        metrics = None
        if return_metrics:
            # Calcular PHI usando IIT metrics
            # Usar enhanced_hidden como estado principal y última capa de attention
            last_attention = attentions[-1] if len(attentions) > 0 else None
            
            phi_components = self.iit_metrics(
                hidden_state=enhanced_hidden,  # (B, T, 768)
                attention_weights=last_attention  # (B, num_heads, T, T)
            )
            
            # Integrar PHI total (phi_components es dict con phi_estimate)
            integration_phi = phi_components['phi_estimate'].mean()  # Tensor scalar
            
            # ΔPhi Loss: MAXIMIZAR PHI
            # Target adaptativo SIN .item() para mantener gradiente
            target_phi_value = 6.0  # Valor fijo para evitar romper grafo
            delta_phi_loss = self.lambda_phi * (target_phi_value - integration_phi)
            
            metrics = {
                'integration_phi': integration_phi.item(),
                'delta_phi_loss': delta_phi_loss.item(),
                'target_phi': target_phi_value,  # Guardar target para monitoreo
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
    
    # =========================================================================
    # MÉTODOS PARA CONTROL RL
    # =========================================================================
    
    def set_loss_weights(self, w_text: float, w_phi: float) -> None:
        """
        Actualiza los pesos de las componentes de loss.
        Usado por el agente RL para cambiar entre modo TEXTO, PHI o MIXTO.
        
        Args:
            w_text: Peso para loss de texto (típicamente 0.1-1.0)
            w_phi: Peso para loss de PHI (típicamente 0.0-1.0)
        """
        self.loss_weights["text"] = float(w_text)
        self.loss_weights["phi"] = float(w_phi)
        # Actualizar lambda_phi para mantener compatibilidad
        self.lambda_phi = float(w_phi)
    
    def get_current_metrics(self) -> dict:
        """
        Devuelve métricas actuales del sistema para observación del agente RL.
        
        Returns:
            Dict con métricas: consciousness, phi, loss_text, loss_phi, memory_utilization, perplexity
        """
        return {
            "consciousness": float(self.current_metrics.get("consciousness", 0.0)),
            "phi": float(self.current_metrics.get("phi", 0.0)),
            "loss_text": float(self.current_metrics.get("loss_text", 0.0)),
            "loss_phi": float(self.current_metrics.get("loss_phi", 0.0)),
            "memory_utilization": float(self.current_metrics.get("memory_utilization", 0.0)),
            "perplexity": float(self.current_metrics.get("perplexity", 0.0)),
        }
    
    def update_current_metrics(self, loss_text: float, loss_phi: float, phi: float, 
                               memory_util: float, perplexity: float = 0.0):
        """
        Actualiza las métricas internas (llamado durante entrenamiento).
        
        Args:
            loss_text: Loss de lenguaje actual
            loss_phi: Loss de PHI actual
            phi: Valor de PHI actual
            memory_util: Utilización de memoria
            perplexity: Perplejidad actual (opcional)
        """
        # Consciousness como proxy: PHI normalizado
        consciousness = min(phi / 10.0, 1.0)  # Normalizar Φ≈10 → 1.0
        
        self.current_metrics.update({
            "consciousness": float(consciousness),
            "phi": float(phi),
            "loss_text": float(loss_text),
            "loss_phi": float(loss_phi),
            "memory_utilization": float(memory_util),
            "perplexity": float(perplexity),
        })


# =============================================================================
# DATASET (reutilizamos WikiText2RealDataset)
# =============================================================================

class WikiText2RealDataset(Dataset):
    """Dataset WikiText-2 con GPT-2 tokenizer."""
    
    def __init__(self, split='train', seq_len=256, tokenizer=None):
        self.seq_len = seq_len
        
        # Cargar tokenizer
        if tokenizer is None:
            print(f"\n🔤 Cargando GPT-2 Tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        else:
            self.tokenizer = tokenizer
        
        self.vocab_size = len(self.tokenizer)
        print(f"  ✓ Vocabulario: {self.vocab_size:,} tokens")
        
        # Cargar WikiText-2
        print(f"\n📚 Cargando WikiText-2 ({split})...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        print(f"  ✓ Dataset cargado: {len(dataset):,} ejemplos")
        
        # Concatenar texto
        text = '\n'.join([example['text'] for example in dataset if example['text'].strip()])
        print(f"  ✓ Caracteres totales: {len(text):,}")
        
        # Tokenizar (más eficiente con add_special_tokens=False)
        print(f"  📝 Tokenizando...")
        self.tokens = self.tokenizer.encode(text, add_special_tokens=False)
        print(f"  ✓ Total tokens: {len(self.tokens):,}")
        
        # Calcular secuencias
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


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.01):
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
# TRAINER
# =============================================================================

class HybridTrainer:
    """Entrenador para modelo híbrido GPT-2 + IIT."""
    
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
        
        # DataLoaders
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
        
        # Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.98),
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience)
        
        # History
        self.history = {
            'train_loss': [],
            'train_ppl': [],
            'val_loss': [],
            'val_ppl': [],
            'train_phi': [],
            'val_phi': [],
            'delta_phi_loss': [],
            'target_phi': []  # Añadir target_phi al historial
        }
    
    def train_epoch(self, epoch, debug_gradients=False):
        self.model.train()
        total_loss = 0
        total_loss_lm = 0
        total_loss_phi = 0
        phi_values = []
        target_phi_values = []  # Añadir para rastrear target
        num_batches = 0
        
        # Para debugging de gradientes
        phi_gradients = []
        
        pbar = tqdm(self.train_loader, desc=f'Época {epoch}')
        
        for input_ids, labels in pbar:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward con métricas IIT
            logits, metrics = self.model(input_ids, return_metrics=True)
            
            # Loss LM
            loss_lm = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            
            # Loss PHI (mantener tensor para gradientes)
            loss_phi_value = metrics['delta_phi_loss']
            if isinstance(loss_phi_value, float):
                loss_phi = torch.tensor(loss_phi_value, device=self.device, requires_grad=True)
            else:
                loss_phi = loss_phi_value
            
            # RL: Loss total con pesos dinámicos
            w_text = self.model.loss_weights.get("text", 1.0)
            w_phi = self.model.loss_weights.get("phi", 1.0)
            loss = w_text * loss_lm + w_phi * loss_phi
            
            # Actualizar métricas internas del modelo (para RL)
            memory_util = metrics.get('memory_util', 0.0)
            perplexity = math.exp(loss_lm.item()) if loss_lm.item() < 10 else 1000.0
            self.model.update_current_metrics(
                loss_text=loss_lm.item(),
                loss_phi=loss_phi.item() if isinstance(loss_phi, torch.Tensor) else loss_phi,
                phi=metrics['integration_phi'],
                memory_util=memory_util,
                perplexity=perplexity
            )
            
            # Backward
            loss.backward()
            
            # 🔍 DEBUG GRADIENTES: Verificar si PHI tiene gradiente
            if debug_gradients and num_batches < 5:  # Solo primeros 5 batches
                # Verificar gradiente de loss_phi
                if hasattr(loss_phi, 'grad') and loss_phi.grad is not None:
                    phi_grad_norm = loss_phi.grad.norm().item()
                    phi_gradients.append(phi_grad_norm)
                    print(f"\n  [BATCH {num_batches}] loss_phi.grad norm: {phi_grad_norm:.6e}")
                else:
                    print(f"\n  [BATCH {num_batches}] loss_phi.grad: None")
                
                # Verificar gradientes de parámetros IIT
                iit_grad_norms = []
                for name, param in self.model.named_parameters():
                    if 'iit_metrics' in name or 'memory' in name or 'phi_weights' in name:
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            iit_grad_norms.append((name, grad_norm))
                
                if iit_grad_norms:
                    print(f"  Gradientes IIT (top 5):")
                    for name, grad_norm in sorted(iit_grad_norms, key=lambda x: x[1], reverse=True)[:5]:
                        print(f"    {name}: {grad_norm:.6e}")
                else:
                    print(f"  ⚠️ NO hay gradientes en parámetros IIT")
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Acumular
            total_loss += loss.item()
            total_loss_lm += loss_lm.item()
            total_loss_phi += loss_phi.item() if isinstance(loss_phi, torch.Tensor) else loss_phi
            phi_values.append(metrics['integration_phi'])
            target_phi_values.append(metrics.get('target_phi', 6.0))  # Capturar target
            num_batches += 1
            
            # Progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{math.exp(loss_lm.item()):.2f}',
                'phi': f'{metrics["integration_phi"]:.3f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_ppl = math.exp(total_loss_lm / num_batches)
        avg_phi = sum(phi_values) / len(phi_values)
        avg_loss_phi = total_loss_phi / num_batches
        avg_target_phi = sum(target_phi_values) / len(target_phi_values)  # Promedio target
        
        # Reporte de gradientes si debug activado
        if debug_gradients and phi_gradients:
            avg_phi_grad = sum(phi_gradients) / len(phi_gradients)
            print(f"\n  📊 Promedio gradiente PHI (primeros 5 batches): {avg_phi_grad:.6e}")
        
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
                
                # Forward con métricas
                logits, metrics = self.model(input_ids, return_metrics=True)
                
                # Loss
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1)
                )
                
                total_loss += loss.item()
                phi_values.append(metrics['integration_phi'])
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        avg_ppl = math.exp(avg_loss)
        avg_phi = sum(phi_values) / len(phi_values)
        
        return avg_loss, avg_ppl, avg_phi
    
    def train(self, num_epochs, save_dir='models/checkpoints', debug_gradients=False):
        print(f"\n{'='*70}")
        print(f"INICIANDO ENTRENAMIENTO - GPT-2 + IIT")
        if debug_gradients:
            print(f"🔍 MODO DEBUG: Verificación de gradientes activada")
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
            
            # Train (con debug en época 1 si está activado)
            train_loss, train_ppl, train_phi, train_loss_phi, train_target_phi = self.train_epoch(
                epoch, debug_gradients=(debug_gradients and epoch == 1)
            )
            
            # Validate
            val_loss, val_ppl, val_phi = self.validate()
            
            # Guardar métricas
            self.history['train_loss'].append(train_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['val_loss'].append(val_loss)
            self.history['val_ppl'].append(val_ppl)
            self.history['train_phi'].append(train_phi)
            self.history['val_phi'].append(val_phi)
            self.history['delta_phi_loss'].append(train_loss_phi)
            self.history['target_phi'].append(train_target_phi)  # Guardar target
            
            # Resultados
            print(f"\n📊 Resultados Época {epoch}:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:,.2f} | Train PHI: {train_phi:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:,.2f} | Val PHI:   {val_phi:.4f}")
            print(f"  ΔPhi Loss: {train_loss_phi:.6f} | Target PHI: {train_target_phi:.2f} (adaptativo)")

            
            # Guardar mejor modelo
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
                }, os.path.join(save_dir, 'infinito_gpt2_best.pt'))
                print(f"  ✅ MEJOR MODELO guardado")
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Early stopping
            if not self.early_stopping(val_loss):
                print(f"\n⏹️  EARLY STOPPING en época {epoch}")
                break
        
        # Guardar historial
        with open('results/training/history_gpt2_iit.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"  Mejor Val Loss: {best_val_loss:.4f}")
        print(f"  Mejor Val PPL: {math.exp(best_val_loss):,.2f}")
        print(f"{'='*70}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GPT-2 + IIT Hybrid Training')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lambda-phi', type=float, default=1.5)  # 🔥🔥 AGRESIVO: Loss_PHI > Loss_LM
    parser.add_argument('--use-lora', action='store_true', default=True)
    parser.add_argument('--no-lora', action='store_true')
    parser.add_argument('--freeze-base', action='store_true', default=True)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug-gradients', action='store_true', help='Activar verificación de gradientes (Opción B)')
    parser.add_argument('--train-only-iit', action='store_true', help='Congelar LoRA y entrenar solo IIT (Opción E)')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Datasets
    train_dataset = WikiText2RealDataset(split='train', seq_len=256)
    val_dataset = WikiText2RealDataset(split='validation', seq_len=256, tokenizer=train_dataset.tokenizer)
    
    # Modelo híbrido
    use_lora = args.use_lora and not args.no_lora
    model = InfinitoGPT2Hybrid(
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        freeze_base=args.freeze_base,
        lambda_phi=args.lambda_phi,
        seed=args.seed
    ).to(device)
    
    # 🔬 OPCIÓN E: Congelar LoRA y entrenar solo IIT
    if args.train_only_iit:
        print("\n" + "="*70)
        print("🔬 EXPERIMENTO OPCIÓN E: ENTRENAR SOLO IIT")
        print("="*70)
        print("  Congelando parámetros LoRA...")
        
        frozen_count = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = False
                frozen_count += 1
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"  ✓ {frozen_count} parámetros LoRA congelados")
        print(f"  Parámetros entrenables: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print("="*70 + "\n")
    
    # Trainer
    trainer = HybridTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        patience=args.patience
    )
    
    # Train
    trainer.train(num_epochs=args.epochs, debug_gradients=args.debug_gradients)


if __name__ == '__main__':
    main()
