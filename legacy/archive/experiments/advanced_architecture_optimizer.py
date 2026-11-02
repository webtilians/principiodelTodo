#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 APLICACIÓN DE OPTIMIZACIONES A ARQUITECTURAS AVANZADAS
========================================================

Script para aplicar la configuración optimizada encontrada
(λ_phi=0.010, dropout=0.25) a arquitecturas más grandes y sofisticadas.
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
import argparse
from datetime import datetime
from tqdm import tqdm
from transformers import GPT2Tokenizer
from datasets import load_dataset

from infinito_v5_2_refactored import InfinitoV52Refactored
from improved_text_generation import ImprovedTextGenerator


class AdvancedArchitectureOptimizer:
    """Optimizador para arquitecturas avanzadas con configuración optimizada."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuración optimizada encontrada
        self.optimized_base_config = {
            'lambda_phi': 0.010,  # ← CRÍTICO: Valor optimizado
            'dropout': 0.25,      # ← Regularización óptima
            'lr': 2e-4,
            'use_improved_memory': True,
            'use_improved_iit': True,
            'use_learnable_phi': True,
            'use_stochastic_exploration': True,
            'seed': 42
        }
        
        print(f"🎯 Configuración base optimizada cargada:")
        print(f"   λ_phi: {self.optimized_base_config['lambda_phi']}")
        print(f"   dropout: {self.optimized_base_config['dropout']}")
    
    def get_advanced_architectures(self):
        """Define las arquitecturas avanzadas a probar."""
        architectures = {
            'small_iit_optimized': {
                'description': 'Small IIT con optimizaciones aplicadas',
                'hidden_dim': 384,
                'num_layers': 3,
                'num_heads': 6,
                'memory_slots': 512,
                'expected_params': '~45M',
                'target_perplexity': 80
            },
            
            'medium_iit_optimized': {
                'description': 'Medium IIT balanceado con optimizaciones',
                'hidden_dim': 512,
                'num_layers': 4,
                'num_heads': 8,
                'memory_slots': 768,
                'expected_params': '~85M',
                'target_perplexity': 60
            },
            
            'large_iit_optimized': {
                'description': 'Large IIT con configuración optimizada',
                'hidden_dim': 768,
                'num_layers': 6,
                'num_heads': 12,
                'memory_slots': 1024,
                'expected_params': '~180M',
                'target_perplexity': 45
            },
            
            'ultra_efficient_optimized': {
                'description': 'Ultra eficiente con optimizaciones aplicadas',
                'hidden_dim': 320,
                'num_layers': 3,
                'num_heads': 8,
                'memory_slots': 640,
                'expected_params': '~35M',
                'target_perplexity': 85
            },
            
            'balanced_performance_optimized': {
                'description': 'Rendimiento balanceado optimizado',
                'hidden_dim': 448,
                'num_layers': 4,
                'num_heads': 8,
                'memory_slots': 896,
                'expected_params': '~65M',
                'target_perplexity': 70
            }
        }
        
        return architectures
    
    def create_optimized_model(self, arch_name, arch_config):
        """Crea un modelo con arquitectura avanzada y configuración optimizada."""
        # Combinar configuración de arquitectura con optimizaciones
        model_config = {**self.optimized_base_config, **arch_config}
        
        model = InfinitoV52Refactored(
            vocab_size=len(self.tokenizer),
            hidden_dim=arch_config['hidden_dim'],
            num_layers=arch_config['num_layers'],
            num_heads=arch_config['num_heads'],
            memory_slots=arch_config['memory_slots'],
            dropout=model_config['dropout'],
            use_improved_memory=model_config['use_improved_memory'],
            use_improved_iit=model_config['use_improved_iit'],
            use_learnable_phi=model_config['use_learnable_phi'],
            use_stochastic_exploration=model_config['use_stochastic_exploration'],
            lambda_phi=model_config['lambda_phi'],
            seed=model_config['seed']
        ).to(self.device)
        
        # Calcular parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n🏗️ Modelo {arch_name} creado:")
        print(f"   Arquitectura: {arch_config['description']}")
        print(f"   Parámetros totales: {total_params:,}")
        print(f"   Parámetros entrenables: {trainable_params:,}")
        print(f"   Tamaño estimado: {arch_config['expected_params']}")
        print(f"   Target perplexity: {arch_config['target_perplexity']}")
        
        return model, model_config
    
    def quick_training_test(self, model, arch_name, steps=200):
        """Realiza un test rápido de entrenamiento para validar la arquitectura."""
        print(f"\n🚀 Test rápido de entrenamiento: {arch_name}")
        
        # Preparar datos de prueba
        sample_texts = [
            "Artificial intelligence and machine learning are transforming technology.",
            "Deep learning models can understand complex patterns in data.",
            "Natural language processing enables computers to understand human language.",
            "The future of AI lies in developing more efficient and capable models.",
            "Scientific research benefits from advanced computational methods."
        ] * 20
        
        full_text = '\n'.join(sample_texts)
        tokens = self.tokenizer.encode(full_text)[:10000]  # 10K tokens para test rápido
        
        # Configuración de entrenamiento
        optimizer = optim.AdamW(model.parameters(), lr=self.optimized_base_config['lr'], weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        
        model.train()
        losses = []
        phi_values = []
        
        seq_len = 128
        batch_size = 2 if 'large' in arch_name else 4
        
        progress_bar = tqdm(range(steps), desc=f"Entrenando {arch_name}")
        
        for step in progress_bar:
            # Crear batch
            start_indices = [np.random.randint(0, len(tokens) - seq_len - 1) for _ in range(batch_size)]
            
            input_batch = []
            label_batch = []
            
            for start_idx in start_indices:
                input_seq = tokens[start_idx:start_idx + seq_len]
                label_seq = tokens[start_idx + 1:start_idx + seq_len + 1]
                input_batch.append(input_seq)
                label_batch.append(label_seq)
            
            input_ids = torch.tensor(input_batch, device=self.device)
            labels = torch.tensor(label_batch, device=self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, metrics = model(input_ids, return_metrics=True)
            
            # Calcular loss
            loss_lm = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            
            loss_phi = 0.0
            if metrics and 'delta_phi_loss' in metrics:
                loss_phi = metrics['delta_phi_loss']
                if isinstance(loss_phi, float):
                    loss_phi = torch.tensor(loss_phi, device=self.device)
            
            # Loss total con configuración optimizada
            if isinstance(loss_phi, torch.Tensor) and loss_phi.item() > 0:
                loss = loss_lm + self.optimized_base_config['lambda_phi'] * loss_phi
            else:
                loss = loss_lm
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Guardar métricas
            losses.append(loss.item())
            if metrics and 'integration_phi' in metrics:
                phi_values.append(metrics['integration_phi'])
            
            # Actualizar progress bar
            if len(losses) >= 10:
                recent_loss = np.mean(losses[-10:])
                recent_phi = np.mean(phi_values[-10:]) if phi_values else 0
                progress_bar.set_postfix({
                    'Loss': f'{recent_loss:.3f}',
                    'PHI': f'{recent_phi:.3f}'
                })
        
        # Resultados finales
        final_loss = losses[-1]
        avg_loss_final = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
        avg_phi = np.mean(phi_values) if phi_values else 0
        improvement = ((losses[0] - final_loss) / losses[0] * 100) if losses[0] > 0 else 0
        
        results = {
            'architecture': arch_name,
            'steps': steps,
            'initial_loss': losses[0],
            'final_loss': final_loss,
            'avg_loss_final_50': avg_loss_final,
            'improvement_percent': improvement,
            'avg_phi': avg_phi,
            'convergence_quality': 'good' if improvement > 20 else 'slow',
            'losses': losses,
            'phi_values': phi_values
        }
        
        print(f"\n📊 Resultados {arch_name}:")
        print(f"   Loss inicial: {losses[0]:.4f}")
        print(f"   Loss final: {final_loss:.4f}")
        print(f"   Mejora: {improvement:.1f}%")
        print(f"   PHI promedio: {avg_phi:.4f}")
        print(f"   Convergencia: {results['convergence_quality']}")
        
        # Limpiar memoria
        del model
        del optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return results
    
    def test_generation_quality(self, arch_name, arch_config, quick_training=False):
        """Prueba la calidad de generación de una arquitectura optimizada."""
        print(f"\n🎨 Test de generación: {arch_name}")
        
        # Crear modelo
        model, model_config = self.create_optimized_model(arch_name, arch_config)
        
        # Entrenamiento rápido si se solicita
        if quick_training:
            print(f"   Realizando entrenamiento rápido...")
            training_results = self.quick_training_test(model, arch_name, steps=100)
        else:
            training_results = None
        
        # Test de generación
        generator = ImprovedTextGenerator(model, self.tokenizer, device=self.device)
        
        test_prompts = [
            "The advancement of artificial intelligence",
            "Modern machine learning techniques",
            "Scientific research in the digital age"
        ]
        
        generation_results = []
        
        for prompt in test_prompts:
            generated = generator.generate_with_advanced_sampling(
                prompt=prompt,
                max_length=80,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.1,
                frequency_penalty=0.1
            )
            
            clean_text = generated.replace(prompt, "").strip()
            if len(clean_text) > 120:
                clean_text = clean_text[:120] + "..."
            
            generation_results.append({
                'prompt': prompt,
                'generated': clean_text
            })
        
        # Limpiar memoria
        del model
        del generator
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'architecture': arch_name,
            'config': model_config,
            'training_results': training_results,
            'generation_samples': generation_results
        }
    
    def run_comprehensive_architecture_test(self, selected_archs=None):
        """Ejecuta test comprensivo de todas las arquitecturas avanzadas."""
        print(f"\n🏗️ TEST COMPRENSIVO DE ARQUITECTURAS AVANZADAS")
        print("="*60)
        
        architectures = self.get_advanced_architectures()
        
        if selected_archs:
            architectures = {k: v for k, v in architectures.items() if k in selected_archs}
        
        print(f"🎯 Probando {len(architectures)} arquitecturas con configuración optimizada")
        
        all_results = []
        
        for arch_name, arch_config in architectures.items():
            print(f"\n" + "="*50)
            print(f"🔬 ARQUITECTURA: {arch_name.upper()}")
            print(f"   {arch_config['description']}")
            print("="*50)
            
            try:
                # Test completo con entrenamiento rápido
                result = self.test_generation_quality(arch_name, arch_config, quick_training=True)
                all_results.append(result)
                
                print(f"✅ {arch_name} completado exitosamente")
                
            except Exception as e:
                print(f"❌ Error en {arch_name}: {e}")
                continue
        
        return all_results
    
    def analyze_architecture_results(self, results):
        """Analiza los resultados de todas las arquitecturas."""
        print(f"\n📊 ANÁLISIS DE RESULTADOS DE ARQUITECTURAS")
        print("="*60)
        
        if not results:
            print("❌ No hay resultados para analizar")
            return
        
        # Analizar rendimiento de entrenamiento
        print(f"\n🏋️ Rendimiento de entrenamiento:")
        print(f"{'Arquitectura':<25} {'Mejora%':<10} {'PHI':<8} {'Convergencia'}")
        print("-" * 55)
        
        for result in results:
            if result['training_results']:
                tr = result['training_results']
                arch_name = result['architecture'][:24]
                improvement = tr['improvement_percent']
                phi = tr['avg_phi']
                convergence = tr['convergence_quality']
                
                print(f"{arch_name:<25} {improvement:<10.1f} {phi:<8.3f} {convergence}")
        
        # Analizar calidad de generación
        print(f"\n🎨 Calidad de generación (muestra):")
        for result in results:
            print(f"\n🔸 {result['architecture']}:")
            if result['generation_samples']:
                sample = result['generation_samples'][0]
                clean_text = sample['generated'][:80] + "..." if len(sample['generated']) > 80 else sample['generated']
                print(f"   '{sample['prompt'][:30]}...' → {clean_text}")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        
        best_training = max(results, key=lambda x: x['training_results']['improvement_percent'] if x['training_results'] else 0)
        best_phi = max(results, key=lambda x: x['training_results']['avg_phi'] if x['training_results'] else 0)
        
        print(f"   🏆 Mejor entrenamiento: {best_training['architecture']}")
        if best_training['training_results']:
            print(f"      Mejora: {best_training['training_results']['improvement_percent']:.1f}%")
        
        print(f"   🧠 Mejor integración IIT: {best_phi['architecture']}")
        if best_phi['training_results']:
            print(f"      PHI: {best_phi['training_results']['avg_phi']:.3f}")
        
        # Configuración aplicada
        print(f"\n⚙️ Configuración optimizada aplicada a todas:")
        print(f"   λ_phi: {self.optimized_base_config['lambda_phi']} (valor optimizado)")
        print(f"   dropout: {self.optimized_base_config['dropout']} (regularización óptima)")
        print(f"   Todas las mejoras IIT activadas")
        
        return results


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description='Test de arquitecturas avanzadas con optimizaciones')
    
    parser.add_argument('--architectures', nargs='+', 
                       choices=['small_iit_optimized', 'medium_iit_optimized', 'large_iit_optimized', 
                               'ultra_efficient_optimized', 'balanced_performance_optimized'],
                       help='Arquitecturas específicas a probar (por defecto: todas)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device para entrenamiento')
    parser.add_argument('--output', type=str, default=None,
                       help='Archivo para guardar resultados')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() and args.device == 'auto' else args.device
    
    print("🚀 APLICACIÓN DE OPTIMIZACIONES A ARQUITECTURAS AVANZADAS")
    print("="*70)
    print(f"Device: {device}")
    print(f"Configuración optimizada: λ_phi=0.010, dropout=0.25")
    
    # Crear optimizador
    optimizer = AdvancedArchitectureOptimizer(device)
    
    # Ejecutar tests
    results = optimizer.run_comprehensive_architecture_test(args.architectures)
    
    # Analizar resultados
    optimizer.analyze_architecture_results(results)
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output:
        output_file = args.output
    else:
        output_file = f'advanced_architectures_optimization_{timestamp}.json'
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'optimized_config': optimizer.optimized_base_config,
        'architectures_tested': args.architectures or 'all',
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Resultados guardados en: {output_file}")
    print(f"✅ Test de arquitecturas avanzadas completado!")


if __name__ == '__main__':
    main()