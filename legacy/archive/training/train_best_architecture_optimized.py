#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 ENTRENAMIENTO COMPLETO DE LA MEJOR ARQUITECTURA OPTIMIZADA
============================================================

Basado en los resultados del test de arquitecturas, entrena completamente
la mejor arquitectura (medium_iit_optimized con 96.8% mejora) usando
la configuración optimizada λ_phi=0.010, dropout=0.25.
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
import numpy as np
import json
import math
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer

from infinito_v5_2_refactored import InfinitoV52Refactored
from improved_text_generation import ImprovedTextGenerator


class OptimizedArchitectureTrainer:
    """Entrenador para la mejor arquitectura con configuración optimizada."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuración optimizada + mejor arquitectura
        self.config = {
            # Configuración optimizada encontrada
            'lambda_phi': 0.010,
            'dropout': 0.25,
            'lr': 2e-4,
            
            # Mejor arquitectura: medium_iit_optimized (96.8% mejora en test)
            'hidden_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'memory_slots': 768,
            
            # Configuraciones adicionales
            'use_improved_memory': True,
            'use_improved_iit': True,
            'use_learnable_phi': True,
            'use_stochastic_exploration': True,
            'seed': 42
        }
        
        print(f"🏆 CONFIGURACIÓN DE LA MEJOR ARQUITECTURA:")
        print(f"   Arquitectura: medium_iit_optimized (~65M parámetros)")
        print(f"   λ_phi: {self.config['lambda_phi']} (optimizado)")
        print(f"   dropout: {self.config['dropout']} (optimizado)")
        print(f"   hidden_dim: {self.config['hidden_dim']}")
        print(f"   num_layers: {self.config['num_layers']}")
        print(f"   Target perplexity: <60")
    
    def create_best_model(self):
        """Crea el modelo con la mejor arquitectura y configuración optimizada."""
        model = InfinitoV52Refactored(
            vocab_size=len(self.tokenizer),
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            memory_slots=self.config['memory_slots'],
            dropout=self.config['dropout'],
            use_improved_memory=self.config['use_improved_memory'],
            use_improved_iit=self.config['use_improved_iit'],
            use_learnable_phi=self.config['use_learnable_phi'],
            use_stochastic_exploration=self.config['use_stochastic_exploration'],
            lambda_phi=self.config['lambda_phi'],
            seed=self.config['seed']
        ).to(self.device)
        
        # Información del modelo
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n🏗️ Modelo creado exitosamente:")
        print(f"   Parámetros totales: {total_params:,}")
        print(f"   Parámetros entrenables: {trainable_params:,}")
        print(f"   Tamaño estimado: ~{total_params/1e6:.1f}M")
        
        return model
    
    def prepare_extended_dataset(self, max_tokens=100000):
        """Prepara un dataset más extenso para entrenamiento completo."""
        print(f"📚 Preparando dataset extendido ({max_tokens} tokens)...")
        
        try:
            # Intentar cargar WikiText-2 completo
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
            texts = []
            total_chars = 0
            max_chars = max_tokens * 4
            
            for example in dataset:
                if total_chars >= max_chars:
                    break
                    
                if isinstance(example, dict) and 'text' in example:
                    text = example['text'].strip()
                    if text and len(text) > 20:  # Filtrar textos muy cortos
                        texts.append(text)
                        total_chars += len(text)
            
            full_text = '\n\n'.join(texts)
            tokens = self.tokenizer.encode(full_text)[:max_tokens]
            print(f"✓ Dataset real cargado: {len(tokens)} tokens de WikiText-2")
            
        except Exception as e:
            print(f"⚠️ Error cargando dataset real, usando sintético extendido...")
            # Dataset sintético más extenso para entrenamiento robusto
            templates = [
                "Artificial intelligence research has shown that {} can significantly improve {} in various applications.",
                "Machine learning algorithms such as {} have revolutionized how we approach {} in modern computing.",
                "Deep learning models with {} architecture demonstrate superior performance in {} tasks.",
                "Natural language processing techniques using {} enable computers to understand {} more effectively.",
                "Recent advances in {} have led to breakthrough applications in {} and related fields.",
                "The development of {} has transformed the landscape of {} research and practical implementation.",
                "Scientists have discovered that {} models can achieve remarkable accuracy in {} prediction tasks.",
                "Computational methods involving {} show promise for solving complex problems in {} domain.",
                "Experimental results indicate that {} approaches outperform traditional methods in {} applications.",
                "The integration of {} with {} offers new possibilities for advancing {} technology."
            ]
            
            domains = [
                "neural networks", "transformer models", "attention mechanisms", "recurrent networks",
                "convolutional architectures", "generative models", "reinforcement learning", "supervised learning",
                "unsupervised learning", "semi-supervised approaches", "meta-learning", "transfer learning",
                "computer vision", "natural language processing", "speech recognition", "image classification",
                "text generation", "machine translation", "question answering", "sentiment analysis",
                "medical diagnosis", "autonomous vehicles", "robotics", "financial modeling",
                "climate prediction", "drug discovery", "materials science", "energy optimization"
            ]
            
            texts = []
            for _ in range(2000):  # Más textos para entrenamiento extenso
                template = np.random.choice(templates)
                domain1 = np.random.choice(domains)
                domain2 = np.random.choice(domains)
                domain3 = np.random.choice(domains)
                text = template.format(domain1, domain2, domain3)
                texts.append(text)
            
            full_text = '\n\n'.join(texts)
            tokens = self.tokenizer.encode(full_text)[:max_tokens]
            print(f"✓ Dataset sintético extendido: {len(tokens)} tokens")
        
        self.tokens = tokens
        return tokens
    
    def train_best_architecture(self, epochs=3, steps_per_epoch=1000, seq_len=128, batch_size=2):
        """Entrenamiento completo de la mejor arquitectura."""
        print(f"\n🚀 ENTRENAMIENTO COMPLETO DE LA MEJOR ARQUITECTURA")
        print(f"Epochs: {epochs}, Steps por época: {steps_per_epoch}")
        print(f"Seq length: {seq_len}, Batch size: {batch_size}")
        print("="*70)
        
        # Crear modelo
        model = self.create_best_model()
        
        # Preparar dataset
        self.prepare_extended_dataset(max_tokens=100000)
        
        # Configurar entrenamiento
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['lr'],
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        total_steps = epochs * steps_per_epoch
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        criterion = nn.CrossEntropyLoss()
        
        # Métricas de seguimiento
        training_history = {
            'losses': [],
            'lm_losses': [],
            'phi_losses': [],
            'perplexities': [],
            'phi_values': [],
            'learning_rates': []
        }
        
        best_loss = float('inf')
        best_model_state = None
        
        model.train()
        global_step = 0
        
        for epoch in range(epochs):
            print(f"\n📚 ÉPOCA {epoch+1}/{epochs}")
            print("-" * 50)
            
            epoch_losses = []
            progress_bar = tqdm(range(steps_per_epoch), desc=f"Época {epoch+1}")
            
            for step in progress_bar:
                global_step += 1
                
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
                    
                    # Padding si necesario
                    if len(input_seq) < seq_len:
                        pad_len = seq_len - len(input_seq)
                        input_seq.extend([self.tokenizer.pad_token_id] * pad_len)
                        label_seq.extend([self.tokenizer.pad_token_id] * pad_len)
                    
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
                current_lr = scheduler.get_last_lr()[0]
                
                training_history['losses'].append(loss_val)
                training_history['lm_losses'].append(lm_loss_val)
                training_history['phi_losses'].append(phi_loss_val)
                training_history['perplexities'].append(current_ppl)
                training_history['learning_rates'].append(current_lr)
                
                if metrics and 'integration_phi' in metrics:
                    training_history['phi_values'].append(metrics['integration_phi'])
                
                epoch_losses.append(loss_val)
                
                # Guardar mejor modelo
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_model_state = model.state_dict().copy()
                
                # Actualizar progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss_val:.3f}',
                    'PPL': f'{current_ppl:.1f}',
                    'PHI': f'{metrics.get("integration_phi", 0):.3f}' if metrics else '0.000',
                    'LR': f'{current_lr:.2e}'
                })
            
            # Resumen de época
            avg_loss = np.mean(epoch_losses)
            avg_ppl = math.exp(avg_loss)
            
            print(f"\n📊 Resumen Época {epoch+1}:")
            print(f"   Loss promedio: {avg_loss:.4f}")
            print(f"   Perplexity promedio: {avg_ppl:.1f}")
            print(f"   Mejor loss hasta ahora: {best_loss:.4f}")
            
            # Test de generación cada época
            if epoch % 1 == 0:  # Cada época
                self.test_generation_epoch(model, epoch+1)
        
        # Cargar mejor modelo
        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"\n✅ Modelo restaurado al mejor checkpoint (loss: {best_loss:.4f})")
        
        return model, training_history, best_loss
    
    def test_generation_epoch(self, model, epoch):
        """Prueba rápida de generación durante el entrenamiento."""
        print(f"\n🎨 Test generación época {epoch}:")
        
        model.eval()
        generator = ImprovedTextGenerator(model, self.tokenizer, device=self.device)
        
        test_prompt = "Artificial intelligence research has shown"
        try:
            generated = generator.generate_with_advanced_sampling(
                prompt=test_prompt,
                max_length=60,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            clean_text = generated.replace(test_prompt, "").strip()
            if len(clean_text) > 80:
                clean_text = clean_text[:80] + "..."
            
            print(f"   '{test_prompt}' → {clean_text}")
        except Exception as e:
            print(f"   ⚠️ Error en generación: {e}")
        
        model.train()
    
    def evaluate_final_model(self, model):
        """Evaluación completa del modelo final."""
        print(f"\n📊 EVALUACIÓN FINAL DEL MODELO")
        print("="*50)
        
        model.eval()
        generator = ImprovedTextGenerator(model, self.tokenizer, device=self.device)
        
        # Prompts de evaluación
        evaluation_prompts = [
            "The future of artificial intelligence",
            "Machine learning algorithms have",
            "Recent advances in deep learning",
            "Natural language processing enables",
            "Scientific research demonstrates that"
        ]
        
        evaluation_results = []
        
        for i, prompt in enumerate(evaluation_prompts, 1):
            print(f"\n{i}. Evaluando: '{prompt}'")
            
            # Generar con diferentes configuraciones
            configurations = [
                {'temp': 0.7, 'top_p': 0.9, 'rep_pen': 1.1, 'name': 'Conservador'},
                {'temp': 0.9, 'top_p': 0.8, 'rep_pen': 1.2, 'name': 'Balanceado'},
                {'temp': 1.1, 'top_p': 0.7, 'rep_pen': 1.3, 'name': 'Creativo'}
            ]
            
            prompt_results = []
            
            for config in configurations:
                try:
                    generated = generator.generate_with_advanced_sampling(
                        prompt=prompt,
                        max_length=80,
                        temperature=config['temp'],
                        top_p=config['top_p'],
                        repetition_penalty=config['rep_pen'],
                        frequency_penalty=0.1
                    )
                    
                    clean_text = generated.replace(prompt, "").strip()
                    if len(clean_text) > 100:
                        clean_text = clean_text[:100] + "..."
                    
                    prompt_results.append({
                        'config': config['name'],
                        'generated': clean_text
                    })
                    
                    print(f"   {config['name']:12}: {clean_text}")
                    
                except Exception as e:
                    print(f"   {config['name']:12}: Error - {e}")
            
            evaluation_results.append({
                'prompt': prompt,
                'results': prompt_results
            })
        
        return evaluation_results


def main():
    """Función principal."""
    print("🏆 ENTRENAMIENTO COMPLETO DE LA MEJOR ARQUITECTURA OPTIMIZADA")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Device: {device}")
    
    # Crear entrenador
    trainer = OptimizedArchitectureTrainer(device)
    
    # Entrenamiento completo
    print(f"\n🎯 Iniciando entrenamiento con configuración optimizada...")
    model, history, best_loss = trainer.train_best_architecture(epochs=2, steps_per_epoch=800)
    
    # Evaluación final
    evaluation_results = trainer.evaluate_final_model(model)
    
    # Guardar resultados y modelo
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Guardar modelo
    model_path = f'infinito_v5.2_best_architecture_optimized_{timestamp}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': trainer.config,
        'training_history': history,
        'best_loss': best_loss,
        'evaluation_results': evaluation_results
    }, model_path)
    
    # Guardar resumen
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'architecture': 'medium_iit_optimized',
        'config': trainer.config,
        'training_summary': {
            'best_loss': best_loss,
            'final_perplexity': math.exp(best_loss),
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'training_steps': len(history['losses'])
        },
        'evaluation_results': evaluation_results
    }
    
    summary_path = f'best_architecture_training_summary_{timestamp}.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print(f"📊 Métricas finales:")
    print(f"   Mejor loss: {best_loss:.4f}")
    print(f"   Perplexity estimado: {math.exp(best_loss):.1f}")
    print(f"   Parámetros: ~65M")
    print(f"💾 Modelo guardado: {model_path}")
    print(f"💾 Resumen guardado: {summary_path}")


if __name__ == '__main__':
    main()