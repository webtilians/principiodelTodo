#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluador de Calidad de Generación de Texto
===========================================

Herramienta completa para evaluar la calidad, coherencia y diversidad
de los modelos INFINITO V5.2 entrenados.

Métricas implementadas:
- Perplexity en diferentes contextos
- Diversidad léxica y semántica
- Coherencia de texto generado
- Repetitividad y naturalidad
- Comparación entre modelos
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime
import math
from typing import List, Dict, Tuple, Optional

# Importar modelo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from infinito_v5_2_refactored import InfinitoV52Refactored

# Importar tokenizer
from transformers import GPT2Tokenizer

class TextGenerationEvaluator:
    """
    Evaluador completo de calidad de generación de texto
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Inicializa el evaluador
        
        Args:
            model_path: Path al modelo entrenado (.pt)
            device: 'cuda', 'cpu', o 'auto'
        """
        self.model_path = model_path
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar modelo
        self.model, self.tokenizer, self.model_config = self._load_model()
        
        # Prompts de prueba predefinidos
        self.test_prompts = [
            "The future of artificial intelligence",
            "In a world where technology",
            "Science has always been",
            "The relationship between humans and",
            "Once upon a time in",
            "The most important discovery",
            "Climate change represents",
            "Education in the 21st century",
            "Space exploration has",
            "The evolution of language"
        ]
        
        # Métricas acumuladas
        self.evaluation_results = {}
    
    def _load_model(self):
        """Carga el modelo desde checkpoint"""
        print(f"[LOAD] Cargando modelo desde: {self.model_path}")
        
        try:
            # Cargar checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extraer configuración
            config = checkpoint.get('config', {})
            vocab_size = config.get('vocab_size', 50257)
            hidden_dim = config.get('hidden_dim', 512)
            
            print(f"[CONFIG] Vocab: {vocab_size}, Hidden: {hidden_dim}")
            
            # Crear modelo
            model = InfinitoV52Refactored(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_layers=config.get('num_layers', 4),
                num_heads=config.get('num_heads', 8),
                memory_slots=256,
                dropout=0.1,  # Evaluación sin dropout
                use_improved_memory=config.get('use_improved_memory', True),
                use_improved_iit=config.get('use_improved_iit', True),
                use_learnable_phi=config.get('use_learnable_phi', True),
                use_stochastic_exploration=False,  # Sin exploración en evaluación
                seed=config.get('seed', 42)
            )
            
            # Cargar pesos
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(self.device)
            model.eval()
            
            # Cargar tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print(f"[OK] Modelo cargado exitosamente")
            print(f"[PARAMS] {sum(p.numel() for p in model.parameters()):,} parámetros")
            
            return model, tokenizer, config
            
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo: {str(e)}")
            raise
    
    def generate_text(
        self, 
        prompt: str, 
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_samples: int = 3
    ) -> List[str]:
        """
        Genera texto usando diferentes estrategias de sampling
        
        Args:
            prompt: Texto inicial
            max_length: Longitud máxima del texto generado
            temperature: Temperatura para sampling (0.1=conservador, 2.0=creativo)
            top_k: Top-k filtering
            top_p: Nucleus sampling
            num_samples: Número de muestras a generar
            
        Returns:
            Lista de textos generados
        """
        # Tokenizar prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        generated_texts = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Generar secuencia
                generated = input_ids.clone()
                
                for _ in range(max_length):
                    # Forward pass
                    outputs = self.model(generated)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # Obtener logits del último token
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                        filtered_logits[top_k_indices] = top_k_logits
                        next_token_logits = filtered_logits
                    
                    # Top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Encontrar tokens que mantienen p <= top_p
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                        sorted_indices_to_remove[0] = 0
                        
                        # Aplicar filtro
                        indices_to_remove = sorted_indices_to_remove.gather(-1, sorted_indices.argsort(-1))
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sampling
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Agregar token
                    generated = torch.cat([generated, next_token.unsqueeze(0)], dim=-1)
                    
                    # Parar en EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                
                # Decodificar texto
                generated_text = self.tokenizer.decode(
                    generated[0], 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                generated_texts.append(generated_text)
        
        return generated_texts
    
    def calculate_perplexity(self, text: str) -> float:
        """Calcula perplexity del texto dado"""
        # Tokenizar
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
        
        if input_ids.shape[1] < 2:
            return float('inf')
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(input_ids[:, :-1])
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Calcular loss
            labels = input_ids[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction='mean'
            )
            
            perplexity = torch.exp(loss).item()
        
        return perplexity
    
    def calculate_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """
        Calcula métricas de diversidad léxica y semántica
        
        Args:
            texts: Lista de textos generados
            
        Returns:
            Diccionario con métricas de diversidad
        """
        if not texts:
            return {}
        
        # Tokenizar todos los textos
        all_tokens = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        
        # Métricas básicas
        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)
        
        # Type-Token Ratio (TTR)
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # Diversidad intra-texto (promedio de TTR por texto)
        intra_diversity = []
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > 0:
                text_ttr = len(set(tokens)) / len(tokens)
                intra_diversity.append(text_ttr)
        
        avg_intra_diversity = np.mean(intra_diversity) if intra_diversity else 0
        
        # Diversidad inter-texto (diferencias entre textos)
        inter_diversity_scores = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                tokens_i = set(self.tokenizer.encode(texts[i], add_special_tokens=False))
                tokens_j = set(self.tokenizer.encode(texts[j], add_special_tokens=False))
                
                # Jaccard similarity
                intersection = len(tokens_i.intersection(tokens_j))
                union = len(tokens_i.union(tokens_j))
                jaccard = intersection / union if union > 0 else 0
                
                # Diversidad = 1 - similitud
                inter_diversity_scores.append(1 - jaccard)
        
        avg_inter_diversity = np.mean(inter_diversity_scores) if inter_diversity_scores else 0
        
        return {
            'type_token_ratio': ttr,
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens,
            'intra_text_diversity': avg_intra_diversity,
            'inter_text_diversity': avg_inter_diversity,
            'vocabulary_coverage': unique_tokens / self.tokenizer.vocab_size
        }
    
    def calculate_repetition_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Calcula métricas de repetición y naturalidad"""
        metrics = {
            'avg_repetition_2gram': 0,
            'avg_repetition_3gram': 0,
            'avg_repetition_4gram': 0,
            'max_repeated_sequence': 0,
            'avg_sentence_length': 0
        }
        
        if not texts:
            return metrics
        
        repetition_scores_2 = []
        repetition_scores_3 = []
        repetition_scores_4 = []
        max_sequences = []
        sentence_lengths = []
        
        for text in texts:
            # Tokenizar
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # N-gramas repetidos
            for n in [2, 3, 4]:
                ngrams = []
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    ngrams.append(ngram)
                
                if ngrams:
                    ngram_counts = Counter(ngrams)
                    repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
                    repetition_ratio = repeated_ngrams / len(ngrams)
                    
                    if n == 2:
                        repetition_scores_2.append(repetition_ratio)
                    elif n == 3:
                        repetition_scores_3.append(repetition_ratio)
                    elif n == 4:
                        repetition_scores_4.append(repetition_ratio)
            
            # Secuencia más larga repetida
            max_repeat = self._find_longest_repeated_sequence(tokens)
            max_sequences.append(max_repeat)
            
            # Longitud promedio de oraciones
            sentences = text.split('.')
            valid_sentences = [s.strip() for s in sentences if s.strip()]
            if valid_sentences:
                avg_length = np.mean([len(s.split()) for s in valid_sentences])
                sentence_lengths.append(avg_length)
        
        # Promedios
        metrics['avg_repetition_2gram'] = np.mean(repetition_scores_2) if repetition_scores_2 else 0
        metrics['avg_repetition_3gram'] = np.mean(repetition_scores_3) if repetition_scores_3 else 0
        metrics['avg_repetition_4gram'] = np.mean(repetition_scores_4) if repetition_scores_4 else 0
        metrics['max_repeated_sequence'] = max(max_sequences) if max_sequences else 0
        metrics['avg_sentence_length'] = np.mean(sentence_lengths) if sentence_lengths else 0
        
        return metrics
    
    def _find_longest_repeated_sequence(self, tokens: List[int]) -> int:
        """Encuentra la secuencia más larga que se repite"""
        max_length = 0
        
        for length in range(2, len(tokens) // 2 + 1):
            sequences = {}
            for i in range(len(tokens) - length + 1):
                seq = tuple(tokens[i:i+length])
                if seq in sequences:
                    max_length = length
                    break
                sequences[seq] = i
        
        return max_length
    
    def evaluate_coherence(self, texts: List[str]) -> Dict[str, float]:
        """
        Evalúa coherencia de los textos usando heurísticas
        """
        metrics = {
            'avg_perplexity': 0,
            'consistency_score': 0,
            'fluency_score': 0
        }
        
        if not texts:
            return metrics
        
        perplexities = []
        consistency_scores = []
        fluency_scores = []
        
        for text in texts:
            # Perplexity como medida de coherencia
            ppl = self.calculate_perplexity(text)
            if not math.isinf(ppl):
                perplexities.append(ppl)
            
            # Puntuación de consistencia (basada en longitud y estructura)
            sentences = text.split('.')
            valid_sentences = [s.strip() for s in sentences if s.strip()]
            
            if valid_sentences:
                # Consistencia = variación en longitud de oraciones (menor variación = más consistente)
                lengths = [len(s.split()) for s in valid_sentences]
                if len(lengths) > 1:
                    consistency = 1 / (1 + np.std(lengths))
                else:
                    consistency = 1.0
                consistency_scores.append(consistency)
                
                # Fluidez = presencia de palabras de conexión
                connectors = ['and', 'but', 'however', 'therefore', 'moreover', 'furthermore', 'meanwhile']
                connector_count = sum(1 for word in text.lower().split() if word in connectors)
                fluency = min(connector_count / len(valid_sentences), 1.0)
                fluency_scores.append(fluency)
        
        metrics['avg_perplexity'] = np.mean(perplexities) if perplexities else float('inf')
        metrics['consistency_score'] = np.mean(consistency_scores) if consistency_scores else 0
        metrics['fluency_score'] = np.mean(fluency_scores) if fluency_scores else 0
        
        return metrics
    
    def comprehensive_evaluation(
        self,
        save_results: bool = True,
        output_dir: str = 'evaluation_results'
    ) -> Dict[str, any]:
        """
        Ejecuta evaluación completa del modelo
        
        Args:
            save_results: Si guardar resultados en archivos
            output_dir: Directorio para guardar resultados
            
        Returns:
            Diccionario con todos los resultados de evaluación
        """
        print(f"[EVAL] Iniciando evaluación completa del modelo")
        print(f"[EVAL] Modelo: {os.path.basename(self.model_path)}")
        
        # Crear directorio de resultados
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        # Resultados por prompt
        prompt_results = {}
        all_generated_texts = []
        
        # Evaluar cada prompt
        for i, prompt in enumerate(self.test_prompts):
            print(f"[EVAL] Evaluando prompt {i+1}/{len(self.test_prompts)}: '{prompt[:30]}...'")
            
            # Generar textos con diferentes configuraciones
            results = {}
            
            # Configuración conservadora
            conservative_texts = self.generate_text(
                prompt, 
                max_length=50, 
                temperature=0.7, 
                top_p=0.8,
                num_samples=3
            )
            
            # Configuración creativa
            creative_texts = self.generate_text(
                prompt, 
                max_length=80, 
                temperature=1.0, 
                top_p=0.9,
                num_samples=3
            )
            
            # Combinar textos para análisis
            all_texts = conservative_texts + creative_texts
            all_generated_texts.extend(all_texts)
            
            # Calcular métricas para este prompt
            diversity_metrics = self.calculate_diversity_metrics(all_texts)
            repetition_metrics = self.calculate_repetition_metrics(all_texts)
            coherence_metrics = self.evaluate_coherence(all_texts)
            
            prompt_results[prompt] = {
                'conservative_texts': conservative_texts,
                'creative_texts': creative_texts,
                'diversity_metrics': diversity_metrics,
                'repetition_metrics': repetition_metrics,
                'coherence_metrics': coherence_metrics
            }
        
        # Métricas globales
        print(f"[EVAL] Calculando métricas globales...")
        
        global_diversity = self.calculate_diversity_metrics(all_generated_texts)
        global_repetition = self.calculate_repetition_metrics(all_generated_texts)
        global_coherence = self.evaluate_coherence(all_generated_texts)
        
        # Compilar resultados finales
        final_results = {
            'model_info': {
                'model_path': self.model_path,
                'model_config': self.model_config,
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'device': self.device
            },
            'evaluation_config': {
                'num_prompts': len(self.test_prompts),
                'total_generated_texts': len(all_generated_texts),
                'timestamp': datetime.now().isoformat()
            },
            'global_metrics': {
                'diversity': global_diversity,
                'repetition': global_repetition,
                'coherence': global_coherence
            },
            'prompt_results': prompt_results
        }
        
        # Calcular puntuación final
        overall_score = self._calculate_overall_score(
            global_diversity, 
            global_repetition, 
            global_coherence
        )
        final_results['overall_score'] = overall_score
        
        # Guardar resultados
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            results_file = f"{output_dir}/evaluation_{model_name}_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            print(f"[SAVE] Resultados guardados en: {results_file}")
        
        # Mostrar resumen
        self._print_evaluation_summary(final_results)
        
        return final_results
    
    def _calculate_overall_score(
        self, 
        diversity: Dict, 
        repetition: Dict, 
        coherence: Dict
    ) -> Dict[str, float]:
        """Calcula puntuación general del modelo"""
        
        # Normalizar métricas (0-1, donde 1 es mejor)
        diversity_score = min(diversity.get('type_token_ratio', 0) * 2, 1.0)  # TTR * 2, max 1
        
        # Para repetición, menos es mejor
        repetition_penalty = (
            repetition.get('avg_repetition_2gram', 0) * 0.3 +
            repetition.get('avg_repetition_3gram', 0) * 0.5 +
            repetition.get('avg_repetition_4gram', 0) * 0.2
        )
        repetition_score = max(0, 1 - repetition_penalty)
        
        # Para coherencia
        ppl = coherence.get('avg_perplexity', float('inf'))
        if ppl == float('inf'):
            coherence_score = 0
        else:
            # Normalizar perplexity (menor es mejor)
            coherence_score = max(0, min(1, 1 - (math.log(ppl) - 3) / 5))  # Escala aproximada
        
        consistency_score = coherence.get('consistency_score', 0)
        fluency_score = coherence.get('fluency_score', 0)
        
        # Puntuación final ponderada
        final_score = (
            diversity_score * 0.3 +
            repetition_score * 0.3 +
            coherence_score * 0.25 +
            consistency_score * 0.1 +
            fluency_score * 0.05
        )
        
        return {
            'overall': final_score,
            'diversity': diversity_score,
            'repetition': repetition_score,
            'coherence': coherence_score,
            'consistency': consistency_score,
            'fluency': fluency_score
        }
    
    def _print_evaluation_summary(self, results: Dict):
        """Imprime resumen de evaluación"""
        print(f"\n{'='*70}")
        print(f"RESUMEN DE EVALUACION")
        print(f"{'='*70}")
        
        model_info = results['model_info']
        global_metrics = results['global_metrics']
        overall_score = results['overall_score']
        
        print(f"Modelo: {os.path.basename(model_info['model_path'])}")
        print(f"Parámetros: {model_info['total_parameters']:,}")
        
        print(f"\n--- PUNTUACIONES ---")
        print(f"Puntuación General: {overall_score['overall']:.3f}")
        print(f"  Diversidad: {overall_score['diversity']:.3f}")
        print(f"  Anti-Repetición: {overall_score['repetition']:.3f}")
        print(f"  Coherencia: {overall_score['coherence']:.3f}")
        print(f"  Consistencia: {overall_score['consistency']:.3f}")
        print(f"  Fluidez: {overall_score['fluency']:.3f}")
        
        print(f"\n--- MÉTRICAS DETALLADAS ---")
        diversity = global_metrics['diversity']
        repetition = global_metrics['repetition']
        coherence = global_metrics['coherence']
        
        print(f"Type-Token Ratio: {diversity.get('type_token_ratio', 0):.3f}")
        print(f"Vocabulario único: {diversity.get('unique_tokens', 0):,}")
        print(f"Repetición 2-gram: {repetition.get('avg_repetition_2gram', 0):.3f}")
        print(f"Repetición 3-gram: {repetition.get('avg_repetition_3gram', 0):.3f}")
        print(f"Perplexity promedio: {coherence.get('avg_perplexity', 0):.2f}")
        
        # Clasificar calidad
        score = overall_score['overall']
        if score >= 0.8:
            quality = "EXCELENTE"
        elif score >= 0.6:
            quality = "BUENA"
        elif score >= 0.4:
            quality = "REGULAR"
        else:
            quality = "NECESITA MEJORAS"
        
        print(f"\nCLASIFICACIÓN: {quality}")
        print(f"{'='*70}\n")


def main():
    """Función principal para evaluar modelos"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluador de Calidad de Generación')
    parser.add_argument('model_path', type=str, help='Path al modelo (.pt)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directorio para resultados')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device para evaluación')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"[ERROR] Modelo no encontrado: {args.model_path}")
        return
    
    # Crear evaluador
    evaluator = TextGenerationEvaluator(args.model_path, args.device)
    
    # Ejecutar evaluación completa
    results = evaluator.comprehensive_evaluation(
        save_results=True,
        output_dir=args.output_dir
    )
    
    print(f"[SUCCESS] Evaluación completada!")

if __name__ == "__main__":
    main()