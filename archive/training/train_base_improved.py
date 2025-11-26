#!/usr/bin/env python3
"""
Script para entrenar el modelo base INFINITO con 30 épocas
Objetivo: Reducir PPL de ~95 a ~40-60 para mejores demos
"""

import sys
import os

# Añadir src/ al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
import time
import json
from pathlib import Path

from train_v5_2_gpt2_lora import InfinitoGPT2Hybrid

def train_base_model(
    epochs=30,
    batch_size=16,
    lr=4e-4,  # LR óptimo del experimento 3
    lambda_phi=0.1,
    patience=5,
    checkpoint_dir="models/checkpoints",
    log_file="training_improved_base.log"
):
    """
    Entrena el modelo base INFINITO por 30 épocas
    
    Configuración optimizada basada en experimentos previos:
    - LR: 4e-4 (mejor que 2e-4 según EXPERIMENTOS_HIPERPARAMETROS.md)
    - Batch size: 16 (buen balance GPU/calidad)
    - Lambda PHI: 0.1 (valor óptimo)
    - Patience: 5 (early stopping conservador)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"🚀 ENTRENAMIENTO MODELO BASE INFINITO - 30 ÉPOCAS")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Lambda PHI: {lambda_phi}")
    print(f"{'='*70}\n")
    
    # Crear directorio para checkpoints
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar modelo
    print("📦 Cargando modelo INFINITO V5.2...")
    model = InfinitoGPT2Hybrid(
        use_lora=True,
        lora_r=4,
        lora_alpha=16,
        lambda_phi=lambda_phi
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 2. Cargar datos
    print("\n📚 Cargando WikiText-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=128,
            padding='max_length',
            return_tensors='pt'
        )
    
    print("   Tokenizando dataset...")
    tokenized_train = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    tokenized_val = dataset['validation'].map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Filtrar ejemplos vacíos
    tokenized_train = tokenized_train.filter(lambda x: len(x['input_ids']) > 0)
    tokenized_val = tokenized_val.filter(lambda x: len(x['input_ids']) > 0)
    
    tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    tokenized_val.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    train_loader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_val, batch_size=batch_size)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # 3. Configurar optimizador y scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"\n🔧 Optimizer configurado:")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Warmup steps: {warmup_steps:,}")
    
    # 4. Variables para early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    # 5. Loop de entrenamiento
    print(f"\n{'='*70}")
    print("🏋️ INICIANDO ENTRENAMIENTO")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # TRAIN
        model.train()
        train_loss = 0.0
        train_text_loss = 0.0
        train_phi_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass con métricas
            logits, metrics = model(
                input_ids=input_ids,
                return_metrics=True
            )
            
            # Loss components
            labels = input_ids.clone()  # Autoregresivo
            # Ignorar padding tokens en el loss (pad_token_id = eos_token_id = 50256)
            labels[labels == tokenizer.pad_token_id] = -100
            text_loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100  # Ignorar padding
            )
            
            # Loss PHI desde métricas
            delta_phi = metrics['delta_phi_loss']
            if isinstance(delta_phi, float):
                delta_phi = torch.tensor(delta_phi, device=device)
            
            phi_loss = lambda_phi * delta_phi
            total_loss = text_loss + phi_loss
            
            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Acumular losses
            train_loss += total_loss.item()
            train_text_loss += text_loss.item()
            train_phi_loss += delta_phi.item()
            
            # Log cada 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                print(f"   Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Promedios de entrenamiento
        avg_train_loss = train_loss / len(train_loader)
        avg_train_text = train_text_loss / len(train_loader)
        avg_train_phi = train_phi_loss / len(train_loader)
        train_ppl = torch.exp(torch.tensor(avg_train_text)).item()
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_text_loss = 0.0
        val_phi_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                
                logits, metrics = model(
                    input_ids=input_ids,
                    return_metrics=True
                )
                
                labels = input_ids.clone()
                # Ignorar padding tokens
                labels[labels == tokenizer.pad_token_id] = -100
                text_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100
                )
                delta_phi = metrics['delta_phi_loss']
                if isinstance(delta_phi, float):
                    delta_phi = torch.tensor(delta_phi, device=device)
                
                val_loss += (text_loss + lambda_phi * delta_phi).item()
                val_text_loss += text_loss.item()
                val_phi_loss += delta_phi.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_text = val_text_loss / len(val_loader)
        avg_val_phi = val_phi_loss / len(val_loader)
        val_ppl = torch.exp(torch.tensor(avg_val_text)).item()
        
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        
        # Log epoch
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_ppl': train_ppl,
            'train_phi': avg_train_phi,
            'val_loss': avg_val_loss,
            'val_ppl': val_ppl,
            'val_phi': avg_val_phi,
            'epoch_time': epoch_time,
            'elapsed_time': elapsed_time
        }
        training_history.append(epoch_log)
        
        print(f"\n{'='*70}")
        print(f"📊 EPOCH {epoch+1}/{epochs} COMPLETADO")
        print(f"{'='*70}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:.2f} | Train ΔΦ: {avg_train_phi:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val PPL:   {val_ppl:.2f} | Val ΔΦ:   {avg_val_phi:.4f}")
        print(f"Time: {epoch_time:.1f}s | Total: {elapsed_time/60:.1f}min")
        
        # Comparación con baseline
        if epoch == 0:
            baseline_val_ppl = val_ppl
        else:
            improvement = ((baseline_val_ppl - val_ppl) / baseline_val_ppl) * 100
            print(f"Mejora vs Epoch 1: {improvement:+.1f}%")
        
        # Guardar mejor modelo
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(checkpoint_dir, 'infinito_base_improved_best.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_ppl': val_ppl,
                'training_history': training_history
            }, checkpoint_path)
            
            print(f"✅ Mejor modelo guardado (Val Loss: {avg_val_loss:.4f}, PPL: {val_ppl:.2f})")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping en epoch {epoch+1}")
                break
        
        print(f"{'='*70}\n")
        
        # Guardar checkpoint cada 5 épocas
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'infinito_base_improved_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_ppl': val_ppl,
                'training_history': training_history
            }, checkpoint_path)
            print(f"💾 Checkpoint epoch {epoch+1} guardado\n")
    
    # Guardar historial de entrenamiento
    history_path = os.path.join(checkpoint_dir, 'training_history_improved.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"🎉 ENTRENAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"Total epochs: {len(training_history)}")
    print(f"Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val PPL: {min(h['val_ppl'] for h in training_history):.2f}")
    print(f"Modelo guardado en: {checkpoint_dir}/infinito_base_improved_best.pt")
    print(f"{'='*70}\n")
    
    return training_history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo base INFINITO mejorado')
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=4e-4, help='Learning rate')
    parser.add_argument('--lambda-phi', type=float, default=0.1, help='Peso de PHI')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    
    history = train_base_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_phi=args.lambda_phi,
        patience=args.patience
    )
