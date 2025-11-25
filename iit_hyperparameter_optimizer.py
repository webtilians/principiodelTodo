#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 OPTIMIZACIÓN DE HIPERPARÁMETROS IIT
====================================

Herramienta para encontrar la configuración óptima de lambda_phi
y otros hiperparámetros IIT para maximizar el rendimiento.

Objetivo: Encontrar el balance perfecto entre LM loss e IIT loss
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


class QuickTrainer:
    """Entrenador rápido para probar hiperparámetros."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = len(self.tokenizer)
        
        # Preparar dataset pequeño para pruebas rápidas
        print("📚 Preparando dataset de prueba...")
        try:
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
            texts = []
            for i, example in enumerate(dataset):
                if i >= 100:
                    break
                if isinstance(example, dict) and 'text' in example:
                    text = example['text'].strip()
                elif isinstance(example, str):
                    text = example.strip()
                else:
                    continue
                
                if text:
                    texts.append(text)
            
            full_text = '\n'.join(texts)
            self.tokens = self.tokenizer.encode(full_text)[:5000]  # Solo 5k tokens para rapidez
            print(f"✓ Dataset preparado: {len(self.tokens)} tokens")
        except Exception as e:
            print(f"⚠️  Error cargando dataset online, usando texto sintético...")
            # Texto de ejemplo para pruebas
            sample_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "Natural language processing enables computers to understand human language.",
                "Deep learning models use neural networks with multiple layers.",
                "Transformers have revolutionized the field of natural language processing.",
                "Attention mechanisms allow models to focus on relevant parts of the input."
            ] * 20  # Repetir para tener más texto
            
            full_text = '\n'.join(sample_texts)
            self.tokens = self.tokenizer.encode(full_text)[:5000]
            print(f"✓ Dataset sintético preparado: {len(self.tokens)} tokens")
    
    def create_model(self, config):
        """Crea un modelo con la configuración especificada."""
        model = InfinitoV52Refactored(
            vocab_size=self.vocab_size,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 2),
            num_heads=config.get('num_heads', 4),
            memory_slots=256,
            dropout=config.get('dropout', 0.1),
            use_improved_memory=True,
            use_improved_iit=True,
            use_learnable_phi=True,
            use_stochastic_exploration=True,
            lambda_phi=config.get('lambda_phi', 0.1),
            seed=42
        ).to(self.device)
        
        return model
    
    def quick_train(self, config, steps=100):
        """Entrenamiento rápido para evaluar configuración."""
        model = self.create_model(config)
        optimizer = optim.AdamW(model.parameters(), lr=config.get('lr', 2e-4), weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        total_loss = 0
        total_loss_lm = 0
        total_loss_phi = 0
        phi_values = []
        
        seq_len = 128
        batch_size = 4
        
        for step in tqdm(range(steps), desc=f"Lambda={config['lambda_phi']:.3f}"):
            # Crear batch aleatorio
            start_idx = np.random.randint(0, len(self.tokens) - seq_len - 1)
            input_ids = torch.tensor(
                self.tokens[start_idx:start_idx + seq_len], 
                device=self.device
            ).unsqueeze(0).repeat(batch_size, 1)
            labels = torch.tensor(
                self.tokens[start_idx + 1:start_idx + seq_len + 1], 
                device=self.device
            ).unsqueeze(0).repeat(batch_size, 1)
            
            optimizer.zero_grad()
            
            # Forward pass con métricas IIT
            logits, metrics = model(input_ids, return_metrics=True)
            
            # Loss de language modeling
            loss_lm = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            # Loss auxiliar ΔPhi
            loss_phi = 0.0
            if metrics and 'delta_phi_loss' in metrics:
                loss_phi = metrics['delta_phi_loss']
                if isinstance(loss_phi, float):
                    loss_phi = torch.tensor(loss_phi, device=self.device)
            
            # Loss total
            lambda_phi = config['lambda_phi']
            if isinstance(loss_phi, torch.Tensor) and loss_phi.item() > 0:
                loss = loss_lm + lambda_phi * loss_phi
            else:
                loss = loss_lm
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Acumular métricas
            total_loss += loss.item()
            total_loss_lm += loss_lm.item()
            if isinstance(loss_phi, torch.Tensor):
                total_loss_phi += loss_phi.item()
            
            if metrics and 'integration_phi' in metrics:
                phi_values.append(metrics['integration_phi'])
        
        # Calcular métricas finales
        avg_loss = total_loss / steps
        avg_loss_lm = total_loss_lm / steps
        avg_loss_phi = total_loss_phi / steps if total_loss_phi > 0 else 0.0
        avg_ppl = math.exp(avg_loss_lm)
        avg_phi = sum(phi_values) / len(phi_values) if phi_values else 0.0
        
        # Limpiar memoria
        del model
        del optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'total_loss': avg_loss,
            'lm_loss': avg_loss_lm,
            'phi_loss': avg_loss_phi,
            'perplexity': avg_ppl,
            'avg_phi': avg_phi,
            'phi_contribution': (avg_loss_phi * config['lambda_phi']) / avg_loss if avg_loss > 0 else 0.0
        }
    
    def grid_search_lambda_phi(self, base_config, lambda_range, steps=100):
        """Búsqueda en grilla para lambda_phi óptimo."""
        results = []
        
        print(f"\n🔍 BÚSQUEDA DE LAMBDA_PHI ÓPTIMO")
        print(f"Rango: {lambda_range}")
        print(f"Configuración base: {base_config}")
        print("="*60)
        
        for lambda_phi in lambda_range:
            config = base_config.copy()
            config['lambda_phi'] = lambda_phi
            
            result = self.quick_train(config, steps)
            result['lambda_phi'] = lambda_phi
            result['config'] = config
            results.append(result)
            
            print(f"λ={lambda_phi:.3f} | "
                  f"Loss={result['total_loss']:.4f} | "
                  f"PPL={result['perplexity']:.1f} | "
                  f"PHI={result['avg_phi']:.4f} | "
                  f"PHI%={result['phi_contribution']*100:.1f}%")
        
        return results
    
    def analyze_results(self, results):
        """Analiza los resultados y encuentra la configuración óptima."""
        # Ordenar por perplexity (menor es mejor)
        sorted_by_ppl = sorted(results, key=lambda x: x['perplexity'])
        
        # Calcular score combinado (perplexity + diversidad phi)
        for result in results:
            # Score que balancea perplexity baja con phi diversity alta
            ppl_score = 1.0 / result['perplexity'] if result['perplexity'] > 0 else 0
            phi_score = result['avg_phi'] * 10  # Escalar phi para que tenga peso similar
            combined_score = ppl_score + phi_score
            result['combined_score'] = combined_score
        
        sorted_by_combined = sorted(results, key=lambda x: x['combined_score'], reverse=True)
        
        print(f"\n📊 ANÁLISIS DE RESULTADOS")
        print("="*60)
        print("🏆 TOP 5 POR PERPLEXITY (menor es mejor):")
        for i, result in enumerate(sorted_by_ppl[:5], 1):
            print(f"  {i}. λ={result['lambda_phi']:.3f} | PPL={result['perplexity']:.1f} | PHI={result['avg_phi']:.4f}")
        
        print(f"\n🎯 TOP 5 POR SCORE COMBINADO:")
        for i, result in enumerate(sorted_by_combined[:5], 1):
            print(f"  {i}. λ={result['lambda_phi']:.3f} | Score={result['combined_score']:.4f} | PPL={result['perplexity']:.1f} | PHI={result['avg_phi']:.4f}")
        
        # Encontrar configuración recomendada
        best_combined = sorted_by_combined[0]
        best_ppl = sorted_by_ppl[0]
        
        # Preferir configuración que esté en top 3 de ambas métricas
        recommendation = None
        for result in sorted_by_combined[:3]:
            if result in sorted_by_ppl[:3]:
                recommendation = result
                break
        
        if recommendation is None:
            recommendation = best_combined
        
        print(f"\n💡 CONFIGURACIÓN RECOMENDADA:")
        print(f"   Lambda PHI: {recommendation['lambda_phi']:.3f}")
        print(f"   Perplexity: {recommendation['perplexity']:.1f}")
        print(f"   PHI Integration: {recommendation['avg_phi']:.4f}")
        print(f"   PHI Contribution: {recommendation['phi_contribution']*100:.1f}% del loss total")
        print(f"   Score combinado: {recommendation['combined_score']:.4f}")
        
        return recommendation, results


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Optimización de hiperparámetros IIT')
    
    parser.add_argument('--model-size', type=str, default='tiny_iit',
                       choices=['tiny_iit', 'micro_iit', 'small_iit'],
                       help='Configuración base del modelo')
    parser.add_argument('--lambda-min', type=float, default=0.01,
                       help='Valor mínimo de lambda_phi')
    parser.add_argument('--lambda-max', type=float, default=0.5,
                       help='Valor máximo de lambda_phi')
    parser.add_argument('--lambda-steps', type=int, default=10,
                       help='Número de pasos en la búsqueda')
    parser.add_argument('--train-steps', type=int, default=100,
                       help='Pasos de entrenamiento por configuración')
    parser.add_argument('--output', type=str, default=None,
                       help='Archivo para guardar resultados')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device para entrenamiento')
    
    args = parser.parse_args()
    
    # Configuraciones base
    base_configs = {
        'tiny_iit': {
            'hidden_dim': 256,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.25,
            'lr': 2e-4
        },
        'micro_iit': {
            'hidden_dim': 384,
            'num_layers': 3,
            'num_heads': 6,
            'dropout': 0.2,
            'lr': 1e-4
        },
        'small_iit': {
            'hidden_dim': 384,
            'num_layers': 3,
            'num_heads': 6,
            'dropout': 0.15,
            'lr': 5e-4
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device
    
    # Crear trainer
    trainer = QuickTrainer(device)
    
    # Configuración base
    base_config = base_configs[args.model_size]
    
    # Rango de lambda_phi
    lambda_range = np.linspace(args.lambda_min, args.lambda_max, args.lambda_steps)
    
    print(f"🎯 OPTIMIZACIÓN DE HIPERPARÁMETROS IIT")
    print(f"Configuración: {args.model_size}")
    print(f"Device: {device}")
    print(f"Pasos por configuración: {args.train_steps}")
    
    # Ejecutar búsqueda
    results = trainer.grid_search_lambda_phi(base_config, lambda_range, args.train_steps)
    
    # Analizar resultados
    recommendation, all_results = trainer.analyze_results(results)
    
    # Guardar resultados
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model_size': args.model_size,
            'base_config': base_config,
            'lambda_range': lambda_range.tolist(),
            'train_steps': args.train_steps
        },
        'results': all_results,
        'recommendation': recommendation
    }
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Resultados guardados en: {args.output}")
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'iit_hyperparameter_optimization_{args.model_size}_{timestamp}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Resultados guardados en: {output_file}")
    
    print(f"\n✅ Optimización completada!")
    print(f"🎯 Lambda PHI recomendado: {recommendation['lambda_phi']:.3f}")


if __name__ == '__main__':
    main()