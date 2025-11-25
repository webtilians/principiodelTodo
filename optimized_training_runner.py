#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 APLICACIÓN DE HIPERPARÁMETROS OPTIMIZADOS
===========================================

Script para aplicar la configuración óptima encontrada en la optimización
de hiperparámetros IIT y entrenar un modelo con mejor rendimiento.
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
import json
import argparse
from datetime import datetime
import math
from tqdm import tqdm

from datasets import load_dataset
from transformers import GPT2Tokenizer
from infinito_v5_2_refactored import InfinitoV52Refactored
from improved_text_generation import ImprovedTextGenerator


class OptimizedTrainer:
    """Entrenador con hiperparámetros optimizados."""
    
    def __init__(self, optimized_config, device='cuda'):
        self.device = device
        self.config = optimized_config
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer)
        
        print(f"🎯 Configuración optimizada cargada:")
        print(f"   Lambda PHI: {self.config['lambda_phi']:.3f}")
        print(f"   Hidden Dim: {self.config['hidden_dim']}")
        print(f"   Layers: {self.config['num_layers']}")
        print(f"   Heads: {self.config['num_heads']}")
        
    def prepare_dataset(self, max_tokens=50000):
        """Prepara un dataset más grande para entrenamiento real."""
        print(f"📚 Preparando dataset de entrenamiento ({max_tokens} tokens)...")
        
        try:
            # Intentar cargar dataset real
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            texts = []
            total_chars = 0
            max_chars = max_tokens * 4  # Aproximadamente 4 chars por token
            
            for example in dataset:
                if total_chars >= max_chars:
                    break
                    
                if isinstance(example, dict) and 'text' in example:
                    text = example['text'].strip()
                elif isinstance(example, str):
                    text = example.strip()
                else:
                    continue
                
                if text and len(text) > 10:  # Solo textos meaningful
                    texts.append(text)
                    total_chars += len(text)
            
            full_text = '\n\n'.join(texts)
            tokens = self.tokenizer.encode(full_text)[:max_tokens]
            print(f"✓ Dataset real cargado: {len(tokens)} tokens de WikiText-2")
            
        except Exception as e:
            print(f"⚠️  Error cargando dataset real, usando sintético extendido...")
            # Dataset sintético más extenso
            templates = [
                "The concept of {} has been fundamental in understanding {}.",
                "Recent advances in {} have shown that {} can be improved significantly.",
                "Researchers discovered that {} is closely related to {}.",
                "The relationship between {} and {} has implications for {}.",
                "Studies indicate that {} plays a crucial role in {}.",
                "The development of {} has revolutionized how we approach {}.",
                "Scientists believe that {} may be the key to understanding {}.",
                "The integration of {} with {} offers new possibilities for {}.",
                "Experimental results suggest that {} influences {} in unexpected ways.",
                "The complexity of {} requires sophisticated approaches to {}."
            ]
            
            topics = [
                "artificial intelligence", "machine learning", "neural networks",
                "natural language processing", "computer vision", "deep learning",
                "data science", "computational linguistics", "pattern recognition",
                "information theory", "cognitive science", "automation",
                "robotics", "algorithm optimization", "statistical modeling"
            ]
            
            texts = []
            for _ in range(1000):
                template = np.random.choice(templates)
                topic1 = np.random.choice(topics)
                topic2 = np.random.choice(topics)
                topic3 = np.random.choice(topics)
                text = template.format(topic1, topic2, topic3)
                texts.append(text)
            
            full_text = '\n\n'.join(texts)
            tokens = self.tokenizer.encode(full_text)[:max_tokens]
            print(f"✓ Dataset sintético generado: {len(tokens)} tokens")
        
        self.tokens = tokens
        return tokens
    
    def create_model(self):
        """Crea modelo con configuración optimizada."""
        model = InfinitoV52Refactored(
            vocab_size=self.vocab_size,
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            memory_slots=256,
            dropout=self.config['dropout'],
            use_improved_memory=True,
            use_improved_iit=True,
            use_learnable_phi=True,
            use_stochastic_exploration=True,
            lambda_phi=self.config['lambda_phi'],  # Configuración optimizada!
            seed=42
        ).to(self.device)
        
        return model
    
    def train_optimized(self, steps=500, seq_len=128, batch_size=4):
        """Entrenamiento con configuración optimizada."""
        print(f"\n🚀 ENTRENAMIENTO CON CONFIGURACIÓN OPTIMIZADA")
        print(f"Steps: {steps}, Seq length: {seq_len}, Batch size: {batch_size}")
        print("="*60)
        
        model = self.create_model()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        # Métricas de seguimiento
        losses = []
        lm_losses = []
        phi_losses = []
        phi_values = []
        perplexities = []
        
        best_loss = float('inf')
        best_model_state = None
        
        progress_bar = tqdm(range(steps), desc="Entrenamiento optimizado")
        
        for step in progress_bar:
            # Crear batch
            start_indices = [
                np.random.randint(0, len(self.tokens) - seq_len - 1) 
                for _ in range(batch_size)
            ]
            
            input_batch = []
            label_batch = []
            
            for start_idx in start_indices:
                input_seq = self.tokens[start_idx:start_idx + seq_len]
                label_seq = self.tokens[start_idx + 1:start_idx + seq_len + 1]
                
                # Padding si es necesario
                if len(input_seq) < seq_len:
                    input_seq.extend([self.tokenizer.pad_token_id] * (seq_len - len(input_seq)))
                if len(label_seq) < seq_len:
                    label_seq.extend([self.tokenizer.pad_token_id] * (seq_len - len(label_seq)))
                
                input_batch.append(input_seq)
                label_batch.append(label_seq)
            
            input_ids = torch.tensor(input_batch, device=self.device)
            labels = torch.tensor(label_batch, device=self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, metrics = model(input_ids, return_metrics=True)
            
            # Calcular losses
            loss_lm = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            loss_phi = 0.0
            if metrics and 'delta_phi_loss' in metrics:
                loss_phi = metrics['delta_phi_loss']
                if isinstance(loss_phi, float):
                    loss_phi = torch.tensor(loss_phi, device=self.device)
            
            # Loss total con configuración optimizada
            if isinstance(loss_phi, torch.Tensor) and loss_phi.item() > 0:
                loss = loss_lm + self.config['lambda_phi'] * loss_phi
                phi_loss_val = loss_phi.item()
            else:
                loss = loss_lm
                phi_loss_val = 0.0
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Guardar métricas
            loss_val = loss.item()
            lm_loss_val = loss_lm.item()
            current_ppl = math.exp(lm_loss_val)
            
            losses.append(loss_val)
            lm_losses.append(lm_loss_val)
            phi_losses.append(phi_loss_val)
            perplexities.append(current_ppl)
            
            if metrics and 'integration_phi' in metrics:
                phi_values.append(metrics['integration_phi'])
            
            # Guardar mejor modelo
            if loss_val < best_loss:
                best_loss = loss_val
                best_model_state = model.state_dict().copy()
            
            # Actualizar progress bar
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'Loss': f'{loss_val:.3f}',
                'PPL': f'{current_ppl:.1f}',
                'PHI': f'{phi_values[-1]:.3f}' if phi_values else '0.000',
                'LR': f'{current_lr:.2e}'
            })
            
            # Log periódico
            if (step + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                avg_ppl = np.mean(perplexities[-100:])
                avg_phi = np.mean(phi_values[-100:]) if phi_values else 0.0
                
                print(f"\n📊 Step {step+1}/{steps}:")
                print(f"   Avg Loss (last 100): {avg_loss:.4f}")
                print(f"   Avg Perplexity: {avg_ppl:.1f}")
                print(f"   Avg PHI: {avg_phi:.4f}")
                print(f"   Best Loss: {best_loss:.4f}")
        
        # Cargar mejor modelo
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # Resultados finales
        final_metrics = {
            'steps': steps,
            'final_loss': losses[-1],
            'best_loss': best_loss,
            'final_perplexity': perplexities[-1],
            'avg_loss_last_100': np.mean(losses[-100:]) if len(losses) >= 100 else np.mean(losses),
            'avg_perplexity_last_100': np.mean(perplexities[-100:]) if len(perplexities) >= 100 else np.mean(perplexities),
            'avg_phi_last_100': np.mean(phi_values[-100:]) if len(phi_values) >= 100 else (np.mean(phi_values) if phi_values else 0.0),
            'optimization_config': self.config,
            'training_curves': {
                'losses': losses,
                'lm_losses': lm_losses,
                'phi_losses': phi_losses,
                'perplexities': perplexities,
                'phi_values': phi_values
            }
        }
        
        return model, final_metrics
    
    def evaluate_generation(self, model, num_samples=5):
        """Evalúa la calidad de generación del modelo optimizado."""
        print(f"\n🎨 EVALUACIÓN DE GENERACIÓN")
        print("="*50)
        
        generator = ImprovedTextGenerator(model, self.tokenizer, device=self.device)
        
        prompts = [
            "The future of artificial intelligence",
            "Machine learning algorithms can",
            "Deep neural networks are designed to",
            "Natural language processing enables",
            "The integration of technology and"
        ]
        
        generation_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"\n🔸 Prompt {i+1}: '{prompt}'")
            
            generated = generator.generate_with_advanced_sampling(
                prompt=prompt,
                max_length=100,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
                frequency_penalty=0.1
            )
            
            # Limpiar texto generado
            clean_text = generated.replace(prompt, "").strip()
            if len(clean_text) > 200:
                clean_text = clean_text[:200] + "..."
            
            print(f"   Generated: {clean_text}")
            
            generation_results.append({
                'prompt': prompt,
                'generated_text': generated,
                'clean_continuation': clean_text
            })
        
        return generation_results


def load_optimization_results(results_file):
    """Carga los resultados de optimización de hiperparámetros."""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    recommendation = data['recommendation']
    config = recommendation['config']
    
    print(f"📋 Configuración optimizada cargada de: {results_file}")
    print(f"   Lambda PHI: {config['lambda_phi']:.3f}")
    print(f"   Perplexity objetivo: {recommendation['perplexity']:.1f}")
    print(f"   PHI Integration: {recommendation['avg_phi']:.4f}")
    
    return config


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Entrenamiento con configuración optimizada')
    
    parser.add_argument('--results-file', type=str, default=None,
                       help='Archivo de resultados de optimización')
    parser.add_argument('--steps', type=int, default=500,
                       help='Número de pasos de entrenamiento')
    parser.add_argument('--tokens', type=int, default=50000,
                       help='Número de tokens del dataset')
    parser.add_argument('--output', type=str, default=None,
                       help='Archivo para guardar resultados')
    parser.add_argument('--save-model', type=str, default=None,
                       help='Ruta para guardar el modelo entrenado')
    
    args = parser.parse_args()
    
    # Cargar configuración optimizada
    if args.results_file and os.path.exists(args.results_file):
        config = load_optimization_results(args.results_file)
    else:
        # Configuración por defecto basada en optimización
        print("⚠️  Usando configuración por defecto optimizada")
        config = {
            'hidden_dim': 256,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.25,
            'lr': 2e-4,
            'lambda_phi': 0.010  # Valor optimizado encontrado
        }
    
    # Crear trainer optimizado
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = OptimizedTrainer(config, device)
    
    # Preparar dataset
    trainer.prepare_dataset(args.tokens)
    
    # Entrenar
    model, metrics = trainer.train_optimized(steps=args.steps)
    
    # Evaluar generación
    generation_results = trainer.evaluate_generation(model)
    
    # Guardar resultados
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'optimization_config': config,
        'training_metrics': metrics,
        'generation_evaluation': generation_results,
        'training_args': {
            'steps': args.steps,
            'tokens': args.tokens
        }
    }
    
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'optimized_training_results_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Guardar modelo si se especifica
    if args.save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics,
            'tokenizer': trainer.tokenizer
        }, args.save_model)
        print(f"💾 Modelo guardado en: {args.save_model}")
    
    print(f"\n✅ ENTRENAMIENTO OPTIMIZADO COMPLETADO!")
    print(f"📊 Mejores métricas:")
    print(f"   Best Loss: {metrics['best_loss']:.4f}")
    print(f"   Final Perplexity: {metrics['final_perplexity']:.1f}")
    print(f"   Avg PHI (last 100): {metrics['avg_phi_last_100']:.4f}")
    print(f"💾 Resultados guardados en: {output_file}")


if __name__ == '__main__':
    main()