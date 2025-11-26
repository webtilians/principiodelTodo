#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 PROBADOR DE MODELOS INFINITO V5.2
=====================================

Script para probar la coherencia y calidad de texto generado
por diferentes modelos entrenados.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import json
import math
from datetime import datetime

from infinito_v5_2_refactored import InfinitoV52Refactored

class ModelTester:
    """Probador de modelos INFINITO V5.2."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Cargar tokenizer
        print("🔤 Cargando GPT-2 Tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = len(self.tokenizer)
        
        # Configurar pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"  ✓ Vocabulario: {self.vocab_size:,} tokens")
    
    def load_model(self, checkpoint_path, config=None):
        """Carga un modelo desde checkpoint."""
        
        print(f"\n📂 Cargando modelo: {os.path.basename(checkpoint_path)}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Obtener configuración del checkpoint o usar la proporcionada
            if config is None and 'config' in checkpoint:
                model_config = checkpoint['config']
                print(f"  📋 Config desde checkpoint: {model_config}")
            else:
                # Configuración por defecto basada en los experimentos
                model_config = config or {
                    'vocab_size': self.vocab_size,
                    'hidden_dim': 384,
                    'num_layers': 3,
                    'num_heads': 6,
                    'memory_slots': 256,
                    'dropout': 0.25,
                }
            
            # Crear modelo
            model = InfinitoV52Refactored(
                vocab_size=model_config.get('vocab_size', self.vocab_size),
                hidden_dim=model_config.get('hidden_dim', 384),
                num_layers=model_config.get('num_layers', 3),
                num_heads=model_config.get('num_heads', 6),
                memory_slots=model_config.get('memory_slots', 256),
                dropout=model_config.get('dropout', 0.0),  # Sin dropout en inferencia
                use_improved_memory=model_config.get('use_improved_memory', True),
                use_improved_iit=model_config.get('use_improved_iit', True),
                use_learnable_phi=model_config.get('use_learnable_phi', True),
                use_stochastic_exploration=model_config.get('use_stochastic_exploration', True),
                lambda_phi=0.1,  # Valor por defecto para inferencia
                seed=42
            ).to(self.device)
            
            # Cargar pesos
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.eval()
            
            # Información del modelo
            num_params = sum(p.numel() for p in model.parameters())
            epoch = checkpoint.get('epoch', 'Desconocida')
            val_ppl = checkpoint.get('val_ppl', 'Desconocida')
            
            print(f"  ✓ Modelo cargado exitosamente")
            print(f"  📊 Época: {epoch} | Val PPL: {val_ppl}")
            print(f"  🔢 Parámetros: {num_params:,}")
            
            return model
            
        except Exception as e:
            print(f"  ❌ Error cargando modelo: {e}")
            return None
    
    def generate_text(self, model, prompt, max_length=200, temperature=0.8, top_p=0.9, do_sample=True):
        """Genera texto usando el modelo."""
        
        # Tokenizar prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = model(generated_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Obtener logits del último token
                next_token_logits = logits[0, -1, :] / temperature
                
                if do_sample:
                    # Nucleus (top-p) sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Agregar token generado
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                
                # Parar si genera EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Limitar longitud de contexto para eficiencia
                if generated_ids.size(1) > 1024:
                    generated_ids = generated_ids[:, -512:]
        
        # Decodificar texto generado
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extraer solo la parte generada (después del prompt)
        generated_only = generated_text[len(prompt):].strip()
        
        return generated_only
    
    def evaluate_perplexity(self, model, text):
        """Calcula la perplexity de un texto."""
        
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        if input_ids.size(1) < 2:
            return float('inf')
        
        with torch.no_grad():
            outputs = model(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calcular loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            perplexity = math.exp(loss.item())
            
        return perplexity

def main():
    """Función principal para probar modelos."""
    
    print("🧠 PROBADOR DE MODELOS INFINITO V5.2")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    
    tester = ModelTester(device)
    
    # Modelos disponibles
    models_to_test = {
        "IIT Optimizado (Reciente)": "models/checkpoints/infinito_v5.2_real_best.pt",
        "IIT Baseline": "models/checkpoints/baseline_no_iit_best.pt",
        "IIT V5.2 Epoch 10": "models/checkpoints/infinito_v5.2_real_epoch_10.pt",
    }
    
    # Prompts de prueba
    test_prompts = [
        "The future of artificial intelligence is",
        "In the year 2050, humanity will",
        "The most important discovery in science was",
        "Technology has changed the way we",
        "The mystery of consciousness lies in"
    ]
    
    print(f"\n📝 Prompts de prueba: {len(test_prompts)}")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  {i}. \"{prompt}\"")
    
    print(f"\n🤖 Modelos a probar: {len(models_to_test)}")
    
    results = {}
    
    for model_name, model_path in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"🧪 PROBANDO: {model_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"  ❌ Archivo no encontrado: {model_path}")
            continue
        
        model = tester.load_model(model_path)
        if model is None:
            continue
        
        model_results = {
            'generations': [],
            'perplexities': [],
            'avg_perplexity': 0.0
        }
        
        total_perplexity = 0
        valid_generations = 0
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Prompt {i}: \"{prompt}\"")
            
            try:
                # Generar texto
                generated = tester.generate_text(
                    model, prompt, 
                    max_length=100, 
                    temperature=0.8, 
                    top_p=0.9
                )
                
                # Calcular perplexity
                full_text = prompt + " " + generated
                perplexity = tester.evaluate_perplexity(model, full_text)
                
                print(f"  🎯 Generado: \"{generated[:150]}{'...' if len(generated) > 150 else ''}\"")
                print(f"  📊 Perplexity: {perplexity:.2f}")
                
                model_results['generations'].append({
                    'prompt': prompt,
                    'generated': generated,
                    'perplexity': perplexity
                })
                
                total_perplexity += perplexity
                valid_generations += 1
                
            except Exception as e:
                print(f"  ❌ Error generando: {e}")
                continue
        
        if valid_generations > 0:
            model_results['avg_perplexity'] = total_perplexity / valid_generations
            print(f"\n📊 RESUMEN {model_name}:")
            print(f"  📈 Perplexity promedio: {model_results['avg_perplexity']:.2f}")
            print(f"  ✅ Generaciones exitosas: {valid_generations}/{len(test_prompts)}")
        
        results[model_name] = model_results
    
    # Resumen comparativo
    print(f"\n{'='*60}")
    print("🏆 COMPARACIÓN FINAL")
    print(f"{'='*60}")
    
    best_model = None
    best_perplexity = float('inf')
    
    for model_name, model_results in results.items():
        if model_results['avg_perplexity'] > 0:
            print(f"\n🤖 {model_name}:")
            print(f"  📊 Perplexity promedio: {model_results['avg_perplexity']:.2f}")
            
            if model_results['avg_perplexity'] < best_perplexity:
                best_perplexity = model_results['avg_perplexity']
                best_model = model_name
    
    if best_model:
        print(f"\n🏆 MEJOR MODELO: {best_model}")
        print(f"  📊 Mejor Perplexity: {best_perplexity:.2f}")
    
    # Guardar resultados
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'results/model_comparison_{timestamp}.json'
    
    os.makedirs('results', exist_ok=True)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados: {results_file}")

if __name__ == '__main__':
    main()