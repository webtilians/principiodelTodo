#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador de Generación de Texto - INFINITO V5.2
===============================================

Herramientas para evaluar la calidad, coherencia y diversidad 
de la generación de texto de los modelos INFINITO V5.2.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import re
from collections import Counter
import math

# Importar transformers para métricas de evaluación
try:
    from transformers import GPT2Tokenizer, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARNING] Transformers no disponible, usando métricas básicas")

# Importar modelo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from infinito_v5_2_refactored import InfinitoV52Refactored

class TextGenerationEvaluator:
    """
    Evaluador completo de generación de texto con múltiples métricas
    """
    
    def __init__(self, tokenizer_type='gpt2'):
        """
        Inicializa el evaluador
        
        Args:
            tokenizer_type: 'gpt2' o 'llama3'
        """
        self.tokenizer_type = tokenizer_type
        
        # Cargar tokenizer
        if TRANSFORMERS_AVAILABLE:
            if tokenizer_type == 'gpt2':
                self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                print(f"[TOKENIZER] GPT-2 cargado (vocab: {len(self.tokenizer):,})")
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
                    print(f"[TOKENIZER] Llama-3 cargado (vocab: {len(self.tokenizer):,})")
                except:
                    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                    print(f"[TOKENIZER] Fallback a GPT-2 (vocab: {len(self.tokenizer):,})")
        else:
            self.tokenizer = None
    
    def load_model(self, model_path: str, device: str = 'cuda') -> InfinitoV52Refactored:
        """
        Carga un modelo desde checkpoint
        
        Args:
            model_path: Ruta al archivo .pt
            device: dispositivo ('cuda' o 'cpu')
        """
        print(f"[MODEL] Cargando modelo: {model_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            # Checkpoint completo con configuración
            config = checkpoint['config']
            model = InfinitoV52Refactored(
                vocab_size=config['vocab_size'],
                hidden_dim=config['hidden_dim'],
                num_layers=4,  # Default
                num_heads=8,   # Default
                memory_slots=256,
                use_improved_memory=config.get('use_improved_memory', True),
                use_improved_iit=True,
                use_learnable_phi=True,
                use_stochastic_exploration=True,
                seed=config.get('seed', 42)
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[CONFIG] Vocab: {config['vocab_size']:,}, Hidden: {config['hidden_dim']}")
            
        else:
            # Solo state_dict - asumir configuración estándar
            model = InfinitoV52Refactored(
                vocab_size=50257,  # GPT-2 vocab size
                hidden_dim=512,
                num_layers=4,
                num_heads=8,
                memory_slots=256,
                use_improved_memory=True,
                use_improved_iit=True,
                use_learnable_phi=True,
                use_stochastic_exploration=True,
            ).to(device)
            
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[CONFIG] Configuración por defecto aplicada")
        
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] Modelo cargado: {total_params:,} parámetros")
        
        return model
    
    def generate_text(
        self, 
        model: InfinitoV52Refactored, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        num_samples: int = 5,
        device: str = 'cuda'
    ) -> List[str]:
        """
        Genera múltiples muestras de texto
        
        Args:
            model: Modelo INFINITO cargado
            prompt: Texto inicial
            max_length: Longitud máxima de generación
            temperature: Temperatura para sampling
            top_k: Top-k filtering
            top_p: Nucleus sampling
            num_samples: Número de muestras a generar
            device: dispositivo
            
        Returns:
            Lista de textos generados
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer no disponible")
        
        print(f"[GEN] Generando {num_samples} muestras (max_len={max_length})")
        print(f"[GEN] Parámetros: T={temperature}, top_k={top_k}, top_p={top_p}")
        
        # Tokenizar prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_length = input_ids.shape[1]
        
        generated_texts = []
        
        with torch.no_grad():
            for sample_idx in range(num_samples):
                print(f"  Generando muestra {sample_idx + 1}/{num_samples}...", end=' ')
                
                # Generar secuencia
                generated_ids = input_ids.clone()
                
                for _ in range(max_length):
                    # Forward pass
                    output = model(generated_ids)
                    
                    # Manejar salida (puede ser tuple)
                    if isinstance(output, tuple):
                        logits = output[0]
                    else:
                        logits = output
                    
                    # Aplicar temperatura y filtros
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                        filtered_logits.scatter_(1, top_k_indices, top_k_logits)
                        next_token_logits = filtered_logits
                    
                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Agregar token
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    # Detener si encontramos EOS
                    if hasattr(self.tokenizer, 'eos_token_id') and next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                # Decodificar texto
                generated_text = self.tokenizer.decode(generated_ids[0, prompt_length:], skip_special_tokens=True)
                generated_texts.append(generated_text)
                print("✓")
        
        return generated_texts
    
    def calculate_perplexity(self, model: InfinitoV52Refactored, text: str, device: str = 'cuda') -> float:
        """
        Calcula perplexity de un texto
        """
        if not self.tokenizer:
            return float('inf')
        
        # Tokenizar
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(device)
        
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Calcular loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def calculate_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Calcula métricas de diversidad entre textos generados
        """
        if not texts:
            return {}
        
        # Tokenizar todos los textos
        all_tokens = []
        for text in texts:
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
            else:
                tokens = text.lower().split()
            all_tokens.extend(tokens)
        
        # Métricas básicas
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        
        # Type-Token Ratio (TTR)
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Calcular n-gramas únicos
        bigrams = []
        trigrams = []
        
        for text in texts:
            if self.tokenizer:
                tokens = self.tokenizer.encode(text)
            else:
                tokens = text.lower().split()
            
            # Bigrams
            for i in range(len(tokens) - 1):
                bigrams.append((tokens[i], tokens[i + 1]))
            
            # Trigrams
            for i in range(len(tokens) - 2):
                trigrams.append((tokens[i], tokens[i + 1], tokens[i + 2]))
        
        total_bigrams = len(bigrams)
        unique_bigrams = len(set(bigrams))
        total_trigrams = len(trigrams)
        unique_trigrams = len(set(trigrams))
        
        bigram_ttr = unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0
        trigram_ttr = unique_trigrams / total_trigrams if total_trigrams > 0 else 0.0
        
        # Self-BLEU (diversidad interna)
        self_bleu_scores = []
        for i, text1 in enumerate(texts):
            references = [texts[j] for j in range(len(texts)) if j != i]
            if references:
                # Métrica simple de similitud (inversamente proporcional a diversidad)
                avg_similarity = self._calculate_average_similarity(text1, references)
                self_bleu_scores.append(1.0 - avg_similarity)  # Convertir a diversidad
        
        avg_self_bleu = np.mean(self_bleu_scores) if self_bleu_scores else 0.0
        
        # Longitud promedio
        avg_length = np.mean([len(text.split()) for text in texts])
        
        return {
            'ttr': ttr,
            'bigram_ttr': bigram_ttr,
            'trigram_ttr': trigram_ttr,
            'self_bleu': avg_self_bleu,
            'avg_length': avg_length,
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens,
            'num_samples': len(texts)
        }
    
    def _calculate_average_similarity(self, text: str, references: List[str]) -> float:
        """
        Calcula similitud promedio entre un texto y referencias
        """
        if not references:
            return 0.0
        
        # Métrica simple de Jaccard similarity
        text_tokens = set(text.lower().split())
        similarities = []
        
        for ref in references:
            ref_tokens = set(ref.lower().split())
            intersection = len(text_tokens.intersection(ref_tokens))
            union = len(text_tokens.union(ref_tokens))
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def calculate_coherence_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Calcula métricas de coherencia textual
        """
        if not texts:
            return {}
        
        coherence_scores = []
        
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                coherence_scores.append(0.0)
                continue
            
            # Coherencia basada en repetición de palabras entre oraciones
            sentence_similarities = []
            for i in range(len(sentences) - 1):
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i + 1].lower().split())
                
                intersection = len(words1.intersection(words2))
                union = len(words1.union(words2))
                similarity = intersection / union if union > 0 else 0.0
                sentence_similarities.append(similarity)
            
            avg_coherence = np.mean(sentence_similarities) if sentence_similarities else 0.0
            coherence_scores.append(avg_coherence)
        
        return {
            'avg_coherence': np.mean(coherence_scores),
            'coherence_std': np.std(coherence_scores),
            'min_coherence': min(coherence_scores) if coherence_scores else 0.0,
            'max_coherence': max(coherence_scores) if coherence_scores else 0.0
        }
    
    def evaluate_model(
        self, 
        model_path: str, 
        prompts: List[str],
        output_dir: str = 'evaluation_results',
        **generation_kwargs
    ) -> Dict:
        """
        Evaluación completa de un modelo
        
        Args:
            model_path: Ruta al checkpoint del modelo
            prompts: Lista de prompts para evaluar
            output_dir: Directorio para guardar resultados
            **generation_kwargs: Parámetros de generación
            
        Returns:
            Diccionario con todas las métricas calculadas
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"\n{'='*60}")
        print(f"EVALUANDO MODELO: {os.path.basename(model_path)}")
        print(f"{'='*60}")
        
        # Cargar modelo
        model = self.load_model(model_path, device)
        
        # Crear directorio de resultados
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results = {
            'model_path': model_path,
            'evaluation_timestamp': timestamp,
            'device': device,
            'generation_params': generation_kwargs,
            'prompt_results': []
        }
        
        for prompt_idx, prompt in enumerate(prompts):
            print(f"\n--- PROMPT {prompt_idx + 1}/{len(prompts)} ---")
            print(f"Prompt: \"{prompt[:50]}...\"")
            
            # Generar textos
            generated_texts = self.generate_text(model, prompt, device=device, **generation_kwargs)
            
            # Calcular métricas
            prompt_results = {
                'prompt': prompt,
                'generated_texts': generated_texts,
                'metrics': {}
            }
            
            # Perplexity del prompt
            prompt_ppl = self.calculate_perplexity(model, prompt, device)
            prompt_results['metrics']['prompt_perplexity'] = prompt_ppl
            print(f"[PPL] Prompt perplexity: {prompt_ppl:.2f}")
            
            # Perplexity de textos generados
            gen_ppls = []
            for text in generated_texts:
                full_text = prompt + " " + text
                ppl = self.calculate_perplexity(model, full_text, device)
                gen_ppls.append(ppl)
            
            avg_ppl = np.mean(gen_ppls)
            prompt_results['metrics']['generated_perplexity'] = {
                'average': avg_ppl,
                'std': np.std(gen_ppls),
                'min': min(gen_ppls),
                'max': max(gen_ppls),
                'individual': gen_ppls
            }
            print(f"[PPL] Generated avg: {avg_ppl:.2f} (±{np.std(gen_ppls):.2f})")
            
            # Métricas de diversidad
            diversity_metrics = self.calculate_diversity_metrics(generated_texts)
            prompt_results['metrics']['diversity'] = diversity_metrics
            print(f"[DIV] TTR: {diversity_metrics.get('ttr', 0):.3f}, Self-BLEU: {diversity_metrics.get('self_bleu', 0):.3f}")
            
            # Métricas de coherencia
            coherence_metrics = self.calculate_coherence_metrics(generated_texts)
            prompt_results['metrics']['coherence'] = coherence_metrics
            print(f"[COH] Avg coherence: {coherence_metrics.get('avg_coherence', 0):.3f}")
            
            results['prompt_results'].append(prompt_results)
        
        # Guardar resultados
        results_file = os.path.join(
            output_dir, 
            f"evaluation_{os.path.basename(model_path).replace('.pt', '')}_{timestamp}.json"
        )
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] Resultados guardados: {results_file}")
        
        return results

def main():
    """Función principal para evaluación interactiva"""
    parser = argparse.ArgumentParser(description='Evaluador de Generación - INFINITO V5.2')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Ruta al checkpoint del modelo (.pt)')
    parser.add_argument('--prompts-file', type=str, default=None,
                       help='Archivo JSON con prompts de prueba')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directorio para resultados')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Número de muestras por prompt')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Longitud máxima de generación')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperatura para sampling')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k filtering')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Nucleus sampling threshold')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       choices=['gpt2', 'llama3'],
                       help='Tipo de tokenizer')
    
    args = parser.parse_args()
    
    # Cargar prompts
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompt_data = json.load(f)
            prompts = prompt_data.get('prompts', prompt_data)
    else:
        # Prompts por defecto
        prompts = [
            "The future of artificial intelligence",
            "In a world where consciousness can be measured",
            "The relationship between quantum mechanics and",
            "Once upon a time in a digital realm",
            "The philosophical implications of integrated information"
        ]
        print(f"[INFO] Usando {len(prompts)} prompts por defecto")
    
    # Crear evaluador
    evaluator = TextGenerationEvaluator(tokenizer_type=args.tokenizer)
    
    # Evaluar modelo
    generation_kwargs = {
        'num_samples': args.num_samples,
        'max_length': args.max_length,
        'temperature': args.temperature,
        'top_k': args.top_k,
        'top_p': args.top_p
    }
    
    results = evaluator.evaluate_model(
        model_path=args.model,
        prompts=prompts,
        output_dir=args.output_dir,
        **generation_kwargs
    )
    
    print(f"\n{'='*60}")
    print(f"✅ EVALUACIÓN COMPLETADA")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()