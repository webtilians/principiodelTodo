#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 GENERACIÓN DE TEXTO - INFINITO V5.2
======================================

Script para generar texto usando el modelo INFINITO V5.2 entrenado.

Características:
- Carga del mejor checkpoint entrenado
- Generación con diferentes estrategias (greedy, sampling, top-k, top-p)
- Control de temperatura para creatividad
- Visualización de probabilidades
- Múltiples prompts de prueba
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
from transformers import GPT2Tokenizer
from infinito_v5_2_refactored import InfinitoV52Refactored


class TextGenerator:
    """Generador de texto usando INFINITO V5.2."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Args:
            checkpoint_path: Ruta al checkpoint del modelo
            device: 'cuda' o 'cpu'
        """
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        print(f"\n{'='*70}")
        print(f"GENERADOR DE TEXTO - INFINITO V5.2")
        print(f"{'='*70}")
        print(f"  Device: {self.device}")
        print(f"  Checkpoint: {checkpoint_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Obtener configuración del modelo
        config = checkpoint.get('config', {})
        
        # Crear modelo
        self.model = InfinitoV52Refactored(
            vocab_size=config.get('vocab_size', 10000),
            hidden_dim=config.get('hidden_dim', 512),
            num_layers=config.get('num_layers', 6),
            num_heads=config.get('num_heads', 8),
            memory_slots=config.get('memory_slots', 256),
            use_improved_memory=True,
            use_stochastic_exploration=True,
            seed=42
        ).to(self.device)
        
        # Cargar pesos (strict=False para ignorar claves extra como last_attention_weights)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        model_vocab_size = config.get('vocab_size', 10000)
        
        print(f"  Epoca entrenada: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        if 'val_ppl' in checkpoint:
            print(f"  Val PPL: {checkpoint['val_ppl']:.2f}")
        print(f"  Vocab size del modelo: {model_vocab_size:,}")
        print(f"{'='*70}\n")
        
        # Usar GPT2Tokenizer real
        print(f"Cargando GPT2Tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.vocab_size = len(self.tokenizer)
        print(f"  Vocabulario tokenizer: {self.vocab_size:,} tokens")
        
        # Verificar compatibilidad
        if model_vocab_size != self.vocab_size:
            print(f"\n{'='*70}")
            print(f"⚠️  ADVERTENCIA: INCOMPATIBILIDAD DE VOCABULARIO")
            print(f"{'='*70}")
            print(f"  Modelo entrenado con: {model_vocab_size:,} tokens")
            print(f"  Tokenizer GPT-2 tiene: {self.vocab_size:,} tokens")
            print(f"\n  Este modelo NO es compatible con GPT-2 tokenizer.")
            print(f"  El texto generado será incoherente y tendrá muchos <unk>.")
            print(f"\n  Solución: Usa un modelo entrenado con vocab_size=50257")
            print(f"  Ejemplo: models/checkpoints/infinito_v5.2_real_best.pt")
            print(f"{'='*70}\n")
            raise ValueError(f"Vocabulario incompatible: modelo={model_vocab_size}, tokenizer={self.vocab_size}")
        else:
            print(f"  ✅ Vocabularios compatibles!")
        
        print(f"{'='*70}\n")
    
    def tokenize(self, text):
        """Tokeniza texto a IDs usando GPT2Tokenizer."""
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def detokenize(self, ids):
        """Convierte IDs a texto usando GPT2Tokenizer."""
        return self.tokenizer.decode(ids)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        strategy: str = 'greedy',
        repetition_penalty: float = 1.2
    ):
        """
        Genera texto a partir de un prompt.
        
        Args:
            prompt: Texto inicial
            max_length: Longitud máxima de generación
            temperature: Temperatura (mayor = más creativo, menor = más conservador)
            top_k: Top-k sampling (0 = desactivado)
            top_p: Nucleus sampling (0 = desactivado)
            strategy: 'greedy', 'sample', 'top_k', 'top_p'
            repetition_penalty: Penalización por repetición (>1.0 = menos repeticiones)
        
        Returns:
            Texto generado
        """
        # Tokenizar prompt
        input_ids = self.tokenize(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated_ids = input_ids.copy()
        
        print(f"\nPrompt: '{prompt}'")
        print(f"Estrategia: {strategy} (temp={temperature}, top_k={top_k}, top_p={top_p})")
        print(f"Repetition penalty: {repetition_penalty}")
        print(f"\nGenerando...")
        
        for _ in range(max_length):
            # Forward pass
            output = self.model(input_tensor)
            
            # Manejar tupla si es necesario
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            
            # Obtener logits del último token
            next_token_logits = logits[0, -1, :].clone()
            
            # APLICAR REPETITION PENALTY (tokens ya generados)
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[-20:]):  # últimos 20 tokens únicos
                    # Si el logit es positivo, dividir; si negativo, multiplicar
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty
            
            # Aplicar temperatura
            next_token_logits = next_token_logits / temperature
            
            # Aplicar estrategia de muestreo
            if strategy == 'greedy':
                next_token_id = torch.argmax(next_token_logits).item()
            
            elif strategy == 'sample':
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            elif strategy == 'top_k' and top_k > 0:
                # Top-k sampling
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = F.softmax(top_k_values, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1).item()
                next_token_id = top_k_indices[next_token_idx].item()
            
            elif strategy == 'top_p' and top_p > 0:
                # Nucleus (top-p) sampling
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remover tokens con cumulative probability > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            else:
                # Fallback a greedy
                next_token_id = torch.argmax(next_token_logits).item()
            
            # Agregar a generados
            generated_ids.append(next_token_id)
            
            # Actualizar input (ventana deslizante)
            input_ids.append(next_token_id)
            if len(input_ids) > 256:  # max seq_len
                input_ids = input_ids[-256:]
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            # Detener si generamos <eos> (id=2)
            if next_token_id == 2:
                break
        
        # Detokenizar
        generated_text = self.detokenize(generated_ids)
        
        return generated_text
    
    def generate_multiple(self, prompt, num_samples=3, **kwargs):
        """Genera múltiples muestras con el mismo prompt."""
        print(f"\n{'='*70}")
        print(f"GENERANDO {num_samples} MUESTRAS")
        print(f"{'='*70}")
        
        samples = []
        for i in range(num_samples):
            print(f"\n--- Muestra {i+1}/{num_samples} ---")
            text = self.generate(prompt, **kwargs)
            samples.append(text)
            print(f"Resultado: {text}")
        
        return samples
    
    def interactive_mode(self):
        """Modo interactivo para generar texto."""
        print(f"\n{'='*70}")
        print(f"MODO INTERACTIVO")
        print(f"{'='*70}")
        print(f"\nEscribe un prompt y el modelo generara texto.")
        print(f"Comandos especiales:")
        print(f"  'exit' o 'quit' - Salir")
        print(f"  'config' - Configurar parametros")
        print(f"{'='*70}\n")
        
        # Configuración por defecto
        config = {
            'max_length': 50,
            'temperature': 1.0,
            'strategy': 'sample'
        }
        
        while True:
            try:
                prompt = input("\nPrompt: ").strip()
                
                if prompt.lower() in ['exit', 'quit']:
                    print("Saliendo...")
                    break
                
                if prompt.lower() == 'config':
                    print(f"\nConfiguracion actual:")
                    for key, value in config.items():
                        print(f"  {key}: {value}")
                    continue
                
                if not prompt:
                    continue
                
                # Generar
                generated = self.generate(prompt, **config)
                print(f"\n[GENERADO]: {generated}")
                
            except KeyboardInterrupt:
                print("\n\nInterrumpido por usuario.")
                break
            except Exception as e:
                print(f"\nError: {e}")


def demo_generation():
    """Demo de generación con varios prompts."""
    
    checkpoint_path = 'models/checkpoints/infinito_v5.2_best.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] No se encontro checkpoint: {checkpoint_path}")
        print(f"Ejecuta primero: python train_v5_2_wikitext.py")
        return
    
    # Crear generador
    generator = TextGenerator(checkpoint_path)
    
    # Prompts de prueba
    prompts = [
        "the quick brown",
        "in the beginning",
        "once upon a time",
        "the first thing",
        "when we look at"
    ]
    
    print(f"\n{'='*70}")
    print(f"DEMO: GENERACION CON DIFERENTES ESTRATEGIAS")
    print(f"{'='*70}\n")
    
    # 1. Generación Greedy (determinista)
    print(f"\n{'='*70}")
    print(f"ESTRATEGIA 1: GREEDY (determinista)")
    print(f"{'='*70}")
    
    for prompt in prompts[:2]:
        text = generator.generate(
            prompt=prompt,
            max_length=30,
            strategy='greedy'
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Generado: {text}")
    
    # 2. Generación con Sampling
    print(f"\n{'='*70}")
    print(f"ESTRATEGIA 2: SAMPLING (aleatorio con temperatura)")
    print(f"{'='*70}")
    
    for prompt in prompts[2:4]:
        text = generator.generate(
            prompt=prompt,
            max_length=30,
            temperature=0.8,
            strategy='sample'
        )
        print(f"\nPrompt: '{prompt}'")
        print(f"Generado: {text}")
    
    # 3. Generación con Top-k
    print(f"\n{'='*70}")
    print(f"ESTRATEGIA 3: TOP-K SAMPLING (k=40)")
    print(f"{'='*70}")
    
    text = generator.generate(
        prompt=prompts[4],
        max_length=40,
        temperature=0.9,
        top_k=40,
        strategy='top_k'
    )
    print(f"\nPrompt: '{prompts[4]}'")
    print(f"Generado: {text}")
    
    # 4. Múltiples muestras del mismo prompt
    print(f"\n{'='*70}")
    print(f"ESTRATEGIA 4: MULTIPLES MUESTRAS (diversidad)")
    print(f"{'='*70}")
    
    generator.generate_multiple(
        prompt="the first",
        num_samples=3,
        max_length=25,
        temperature=1.2,
        strategy='sample'
    )
    
    # 5. Estadísticas del modelo
    print(f"\n{'='*70}")
    print(f"ESTADISTICAS DEL MODELO")
    print(f"{'='*70}")
    
    stats = generator.model.get_memory_statistics()
    print(f"\nMemoria externa:")
    print(f"  Utilizacion: {stats['utilization']:.2%}")
    print(f"  Slots ocupados: {stats['occupied_slots']}")
    print(f"  Importancia promedio: {stats.get('avg_importance', 'N/A')}")
    
    print(f"\n{'='*70}")
    print(f"DEMO COMPLETADO")
    print(f"{'='*70}")
    print(f"\nPara modo interactivo, ejecuta:")
    print(f"  python generate_text_v5_2.py --interactive")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generador de texto INFINITO V5.2')
    parser.add_argument('--checkpoint', type=str, 
                       default='models/checkpoints/infinito_v5.2_real_best.pt',
                       help='Ruta al checkpoint del modelo (vocab 50,257 tokens)')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Prompt para generar texto')
    parser.add_argument('--max-length', type=int, default=50,
                       help='Longitud maxima de generacion')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperatura (creatividad)')
    parser.add_argument('--strategy', type=str, default='sample',
                       choices=['greedy', 'sample', 'top_k', 'top_p'],
                       help='Estrategia de muestreo')
    parser.add_argument('--top-k', type=int, default=0,
                       help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.0,
                       help='Nucleus (top-p) sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.5,
                       help='Penalizacion por repeticion (>1.0 = menos repeticiones)')
    parser.add_argument('--interactive', action='store_true',
                       help='Modo interactivo')
    parser.add_argument('--demo', action='store_true',
                       help='Ejecutar demo con varios prompts')
    
    args = parser.parse_args()
    
    # Verificar checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"\n[ERROR] No se encontro checkpoint: {args.checkpoint}")
        print(f"Ejecuta primero: python train_v5_2_wikitext.py")
        return
    
    # Crear generador
    generator = TextGenerator(args.checkpoint)
    
    # Modo demo
    if args.demo:
        demo_generation()
        return
    
    # Modo interactivo
    if args.interactive:
        generator.interactive_mode()
        return
    
    # Generación simple
    if args.prompt:
        text = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            strategy=args.strategy,
            repetition_penalty=args.repetition_penalty
        )
        print(f"\n[RESULTADO]:")
        print(f"{text}\n")
    else:
        # Si no hay prompt, ejecutar demo
        demo_generation()


if __name__ == '__main__':
    main()
