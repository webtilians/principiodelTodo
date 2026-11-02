#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 GENERACIÓN DE TEXTO MEJORADA - INFINITO V5.2
==============================================

Módulo de generación de texto con técnicas avanzadas para mejorar
diversidad, coherencia y reducir repetición.

Mejoras implementadas:
- Nucleus Sampling (top-p)
- Temperature scaling
- Repetition penalty 
- Frequency penalty
- Length penalty
- Beam search alternativo

Objetivo: Mejorar TTR de 0.229 a >0.4 y reducir repetición
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
import torch.nn.functional as F
import numpy as np
import argparse
from transformers import GPT2Tokenizer
from infinito_v5_2_refactored import InfinitoV52Refactored


class ImprovedTextGenerator:
    """Generador de texto con técnicas avanzadas para mejor calidad."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        
        # Configuraciones por defecto optimizadas
        self.default_config = {
            'temperature': 0.8,
            'top_p': 0.9,
            'top_k': 40,
            'repetition_penalty': 1.1,
            'frequency_penalty': 0.1,
            'length_penalty': 1.0,
            'max_length': 100,
            'min_length': 20,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
    
    def nucleus_sampling(self, logits, top_p=0.9, temperature=1.0, filter_value=-float('inf')):
        """
        Implementación de Nucleus Sampling (top-p).
        
        Args:
            logits: Tensor de logits sin normalizar
            top_p: Probabilidad acumulada para el núcleo
            temperature: Temperatura para suavizar distribución
            filter_value: Valor para filtrar tokens
            
        Returns:
            Token seleccionado según nucleus sampling
        """
        # Aplicar temperatura
        logits = logits / temperature
        
        # Convertir a probabilidades
        probs = F.softmax(logits, dim=-1)
        
        # Ordenar probabilidades en orden descendente
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        
        # Calcular probabilidad acumulada
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Crear máscara para tokens que exceden top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Mantener al menos el primer token (el más probable)
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Crear máscara para indices originales
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        
        # Aplicar máscara
        logits[indices_to_remove] = filter_value
        
        return logits
    
    def apply_repetition_penalty(self, logits, input_ids, penalty=1.1):
        """
        Aplica penalización por repetición a los logits.
        
        Args:
            logits: Tensor de logits
            input_ids: Secuencia de input hasta el momento
            penalty: Factor de penalización (>1.0 penaliza repetición)
            
        Returns:
            Logits con penalización aplicada
        """
        # Obtener tokens únicos en el input
        unique_ids = torch.unique(input_ids)
        
        # Aplicar penalización: dividir logits de tokens repetidos
        for token_id in unique_ids:
            if penalty != 1.0:
                logits[token_id] = logits[token_id] / penalty
                
        return logits
    
    def apply_frequency_penalty(self, logits, input_ids, penalty=0.1):
        """
        Aplica penalización por frecuencia (más frecuente = más penalizado).
        
        Args:
            logits: Tensor de logits
            input_ids: Secuencia de input hasta el momento
            penalty: Factor de penalización por frecuencia
            
        Returns:
            Logits con penalización por frecuencia aplicada
        """
        if penalty == 0.0:
            return logits
            
        # Contar frecuencias
        unique_ids, counts = torch.unique(input_ids, return_counts=True)
        
        # Aplicar penalización proporcional a la frecuencia
        for token_id, count in zip(unique_ids, counts):
            freq_penalty = penalty * (count.float() - 1)
            logits[token_id] = logits[token_id] - freq_penalty
            
        return logits
    
    def generate_with_advanced_sampling(self, prompt, **kwargs):
        """
        Genera texto usando técnicas avanzadas de sampling.
        
        Args:
            prompt: Texto de entrada
            **kwargs: Parámetros de generación (sobrescriben defaults)
            
        Returns:
            Texto generado
        """
        # Combinar configuración
        config = {**self.default_config, **kwargs}
        
        # Tokenizar prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        generated_ids = input_ids.clone()
        max_length = config['max_length']
        min_length = config['min_length']
        
        with torch.no_grad():
            for step in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.model(generated_ids)
                
                # Obtener logits del último token
                if isinstance(outputs, tuple):
                    logits = outputs[0][:, -1, :]
                else:
                    logits = outputs[:, -1, :]
                
                # Aplicar penalizaciones
                if config['repetition_penalty'] != 1.0:
                    logits = self.apply_repetition_penalty(
                        logits.squeeze(0), 
                        generated_ids.squeeze(0), 
                        config['repetition_penalty']
                    ).unsqueeze(0)
                
                if config['frequency_penalty'] != 0.0:
                    logits = self.apply_frequency_penalty(
                        logits.squeeze(0), 
                        generated_ids.squeeze(0), 
                        config['frequency_penalty']
                    ).unsqueeze(0)
                
                # Aplicar nucleus sampling
                if config['do_sample']:
                    logits = self.nucleus_sampling(
                        logits.squeeze(0),
                        top_p=config['top_p'],
                        temperature=config['temperature']
                    ).unsqueeze(0)
                    
                    # Sampling
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Agregar token generado
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Verificar condiciones de parada
                if next_token.item() == config['eos_token_id']:
                    if generated_ids.shape[1] >= len(input_ids[0]) + min_length:
                        break
                
                # Verificar longitud máxima
                if generated_ids.shape[1] >= len(input_ids[0]) + max_length:
                    break
        
        # Decodificar resultado
        generated_text = self.tokenizer.decode(
            generated_ids[0][len(input_ids[0]):], 
            skip_special_tokens=True
        )
        
        return generated_text
    
    def generate_multiple_samples(self, prompt, num_samples=5, **kwargs):
        """
        Genera múltiples muestras de texto con diferentes configuraciones.
        
        Args:
            prompt: Texto de entrada
            num_samples: Número de muestras a generar
            **kwargs: Parámetros de generación
            
        Returns:
            Lista de textos generados
        """
        samples = []
        
        for i in range(num_samples):
            # Variar ligeramente la temperatura para más diversidad
            varied_config = kwargs.copy()
            if 'temperature' not in varied_config:
                varied_config['temperature'] = 0.7 + (i * 0.1)  # 0.7, 0.8, 0.9, 1.0, 1.1
            
            if 'top_p' not in varied_config:
                varied_config['top_p'] = 0.85 + (i * 0.03)  # 0.85, 0.88, 0.91, 0.94, 0.97
            
            generated_text = self.generate_with_advanced_sampling(prompt, **varied_config)
            samples.append(generated_text)
        
        return samples
    
    def compare_generation_methods(self, prompt):
        """
        Compara diferentes métodos de generación para análisis.
        
        Args:
            prompt: Texto de entrada
            
        Returns:
            Diccionario con diferentes métodos y sus resultados
        """
        methods = {
            'greedy': {
                'do_sample': False,
                'temperature': 1.0
            },
            'temperature_low': {
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 1.0,
                'repetition_penalty': 1.0
            },
            'temperature_high': {
                'do_sample': True,
                'temperature': 1.2,
                'top_p': 1.0,
                'repetition_penalty': 1.0
            },
            'nucleus_conservative': {
                'do_sample': True,
                'temperature': 0.8,
                'top_p': 0.85,
                'repetition_penalty': 1.1
            },
            'nucleus_aggressive': {
                'do_sample': True,
                'temperature': 1.0,
                'top_p': 0.95,
                'repetition_penalty': 1.2
            },
            'optimized': {  # Configuración basada en análisis previo
                'do_sample': True,
                'temperature': 0.85,
                'top_p': 0.9,
                'repetition_penalty': 1.15,
                'frequency_penalty': 0.05
            }
        }
        
        results = {}
        print(f"\n🎨 COMPARANDO MÉTODOS DE GENERACIÓN")
        print(f"📝 Prompt: '{prompt}'")
        print("="*60)
        
        for method_name, config in methods.items():
            print(f"\n🔧 {method_name.upper()}:")
            generated = self.generate_with_advanced_sampling(prompt, **config)
            results[method_name] = generated
            print(f"   {generated}")
        
        return results


def load_model_and_tokenizer(model_path, device='auto'):
    """Carga modelo y tokenizer desde checkpoint."""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔧 Cargando modelo desde: {model_path}")
    print(f"🖥️  Device: {device}")
    
    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    # Crear modelo
    model = InfinitoV52Refactored(
        vocab_size=config.get('vocab_size', 50257),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 2),
        num_heads=config.get('num_heads', 4),
        memory_slots=256,
        dropout=0.1,
        use_improved_memory=config.get('use_improved_memory', True),
        use_improved_iit=True,
        use_learnable_phi=True,
        use_stochastic_exploration=config.get('use_stochastic_exploration', True),
        lambda_phi=0.05,
        seed=42
    ).to(device)
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Cargar tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    print(f"✅ Modelo cargado: {sum(p.numel() for p in model.parameters()):,} parámetros")
    
    return model, tokenizer, device


def main():
    """Función principal para prueba y demostración."""
    parser = argparse.ArgumentParser(description='Generación de texto mejorada para INFINITO V5.2')
    
    parser.add_argument('model_path', help='Path al modelo (.pt)')
    parser.add_argument('--prompt', type=str, default='The future of artificial intelligence',
                       help='Texto de entrada para generación')
    parser.add_argument('--method', type=str, default='optimized',
                       choices=['greedy', 'temperature_low', 'temperature_high', 
                               'nucleus_conservative', 'nucleus_aggressive', 'optimized', 'compare'],
                       help='Método de generación')
    parser.add_argument('--num-samples', type=int, default=3,
                       help='Número de muestras a generar')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Longitud máxima de generación')
    parser.add_argument('--temperature', type=float, default=0.85,
                       help='Temperatura para sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Nucleus sampling top-p')
    parser.add_argument('--repetition-penalty', type=float, default=1.15,
                       help='Penalización por repetición')
    parser.add_argument('--frequency-penalty', type=float, default=0.05,
                       help='Penalización por frecuencia')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device para inferencia')
    
    args = parser.parse_args()
    
    # Cargar modelo
    model, tokenizer, device = load_model_and_tokenizer(args.model_path, args.device)
    
    # Crear generador
    generator = ImprovedTextGenerator(model, tokenizer, device)
    
    print(f"\n🎯 GENERACIÓN DE TEXTO MEJORADA")
    print("="*50)
    
    if args.method == 'compare':
        # Comparar todos los métodos
        results = generator.compare_generation_methods(args.prompt)
        
        print(f"\n📊 ANÁLISIS DE DIVERSIDAD:")
        for method, text in results.items():
            words = text.split()
            unique_words = set(words)
            ttr = len(unique_words) / len(words) if words else 0
            print(f"   {method}: TTR = {ttr:.3f}, Palabras = {len(words)}, Únicas = {len(unique_words)}")
    
    else:
        # Generar con método específico
        config = {
            'temperature': args.temperature,
            'top_p': args.top_p,
            'repetition_penalty': args.repetition_penalty,
            'frequency_penalty': args.frequency_penalty,
            'max_length': args.max_length
        }
        
        print(f"🔧 Método: {args.method}")
        print(f"📋 Configuración: {config}")
        print(f"📝 Prompt: '{args.prompt}'")
        print("-" * 50)
        
        samples = generator.generate_multiple_samples(
            args.prompt, 
            num_samples=args.num_samples,
            **config
        )
        
        for i, sample in enumerate(samples, 1):
            print(f"\n📄 Muestra {i}:")
            print(f"   {sample}")
        
        # Calcular métricas básicas
        all_words = []
        for sample in samples:
            all_words.extend(sample.split())
        
        unique_words = set(all_words)
        avg_ttr = len(unique_words) / len(all_words) if all_words else 0
        
        print(f"\n📊 Métricas globales:")
        print(f"   TTR promedio: {avg_ttr:.3f}")
        print(f"   Total palabras: {len(all_words)}")
        print(f"   Palabras únicas: {len(unique_words)}")
    
    print(f"\n✅ Generación completada!")


if __name__ == '__main__':
    main()