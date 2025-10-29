"""
📊 MÓDULO DE VALIDACIÓN Y BENCHMARKING
====================================

Métricas estándar de NLP para validación científica rigurosa.

Incluye:
- Perplexity (métrica estándar de modelos de lenguaje)
- BLEU scores (para generación de texto)
- Comparación contra baselines
- Tests estadísticos rigurosos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


class StandardNLPMetrics:
    """
    Métricas estándar de NLP para comparación con estado del arte.
    
    CRÍTICO: Estas son las métricas que DEBES reportar en papers.
    Las métricas custom (Phi, etc.) son complementarias, no reemplazan estas.
    """
    
    @staticmethod
    def calculate_perplexity(logits: torch.Tensor, targets: torch.Tensor, 
                            pad_token_id: int = -100) -> float:
        """
        Calcula perplexity, LA métrica estándar para modelos de lenguaje.
        
        Perplexity = exp(cross_entropy_loss)
        Menor es mejor.
        
        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
            pad_token_id: ID del token de padding a ignorar
            
        Returns:
            perplexity: Valor de perplejidad
        """
        # Reshape para calcular loss
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        
        # Calcular cross entropy (ignorando padding)
        loss = F.cross_entropy(
            logits_flat, 
            targets_flat, 
            ignore_index=pad_token_id,
            reduction='mean'
        )
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
        
        return perplexity
    
    @staticmethod
    def calculate_bleu(predictions: List[str], references: List[str], 
                      n_gram: int = 4) -> float:
        """
        Calcula BLEU score para generación de texto.
        
        NOTA: Para producción, usar sacrebleu. Esta es versión simplificada.
        
        Args:
            predictions: Lista de textos generados
            references: Lista de textos de referencia
            n_gram: Máximo n-grama a considerar (default 4)
            
        Returns:
            bleu_score: Score BLEU [0, 1], mayor es mejor
        """
        from collections import Counter
        
        def get_ngrams(tokens: List[str], n: int) -> Counter:
            """Extrae n-gramas de lista de tokens."""
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(tuple(tokens[i:i+n]))
            return Counter(ngrams)
        
        def precision_n(pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
            """Calcula precisión para n-gramas."""
            pred_ngrams = get_ngrams(pred_tokens, n)
            ref_ngrams = get_ngrams(ref_tokens, n)
            
            if sum(pred_ngrams.values()) == 0:
                return 0.0
            
            clipped_counts = 0
            for ngram, count in pred_ngrams.items():
                clipped_counts += min(count, ref_ngrams.get(ngram, 0))
            
            precision = clipped_counts / sum(pred_ngrams.values())
            return precision
        
        # Calcular para todos los ejemplos
        precisions = {n: [] for n in range(1, n_gram + 1)}
        brevity_penalties = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # Precisiones por n-grama
            for n in range(1, n_gram + 1):
                p_n = precision_n(pred_tokens, ref_tokens, n)
                precisions[n].append(p_n)
            
            # Brevity penalty
            bp = min(1.0, np.exp(1 - len(ref_tokens) / (len(pred_tokens) + 1e-6)))
            brevity_penalties.append(bp)
        
        # Promediar
        avg_precisions = [np.mean(precisions[n]) for n in range(1, n_gram + 1)]
        avg_bp = np.mean(brevity_penalties)
        
        # BLEU = BP * exp(sum(log(p_n) / N))
        if all(p > 0 for p in avg_precisions):
            bleu = avg_bp * np.exp(np.mean([np.log(p) for p in avg_precisions]))
        else:
            bleu = 0.0
        
        return bleu
    
    @staticmethod
    def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor, 
                          pad_token_id: int = -100) -> float:
        """
        Calcula accuracy de predicción de tokens.
        
        Args:
            predictions: [batch, seq_len, vocab_size] o [batch, seq_len]
            targets: [batch, seq_len]
            
        Returns:
            accuracy: Porcentaje de tokens correctos
        """
        if predictions.dim() == 3:
            predictions = predictions.argmax(dim=-1)
        
        # Máscara para ignorar padding
        mask = (targets != pad_token_id)
        correct = (predictions == targets) & mask
        
        accuracy = correct.sum().item() / mask.sum().item()
        return accuracy


class StatisticalTests:
    """
    Tests estadísticos rigurosos para validación científica.
    
    NO usar thresholds arbitrarios. Usar p-values y tests de hipótesis.
    """
    
    @staticmethod
    def test_reproducibility(results_group_a: np.ndarray, 
                           results_group_b: np.ndarray,
                           alpha: float = 0.05) -> Dict[str, any]:
        """
        Test de reproducibilidad usando t-test.
        
        H0: No hay diferencia significativa entre grupos
        H1: Hay diferencia significativa
        
        Args:
            results_group_a: Resultados con seed A (array de métricas)
            results_group_b: Resultados con seed B
            alpha: Nivel de significancia (default 0.05)
            
        Returns:
            Dict con resultados del test
        """
        # T-test de dos muestras
        statistic, pvalue = stats.ttest_ind(results_group_a, results_group_b)
        
        # Cálculo de effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(results_group_a) + np.var(results_group_b)) / 2)
        cohens_d = abs(np.mean(results_group_a) - np.mean(results_group_b)) / (pooled_std + 1e-6)
        
        return {
            't_statistic': statistic,
            'p_value': pvalue,
            'is_reproducible': pvalue > alpha,  # No rechazamos H0
            'cohens_d': cohens_d,
            'effect_size_interpretation': _interpret_cohens_d(cohens_d),
            'mean_group_a': np.mean(results_group_a),
            'mean_group_b': np.mean(results_group_b),
            'variance_group_a': np.var(results_group_a),
            'variance_group_b': np.var(results_group_b)
        }
    
    @staticmethod
    def test_improvement(baseline_scores: np.ndarray, 
                        improved_scores: np.ndarray,
                        alpha: float = 0.05) -> Dict[str, any]:
        """
        Test si una mejora es estadísticamente significativa.
        
        Usa paired t-test (muestras pareadas).
        
        Args:
            baseline_scores: Scores del modelo baseline
            improved_scores: Scores del modelo mejorado
            alpha: Nivel de significancia
            
        Returns:
            Dict con resultados del test
        """
        # Paired t-test (one-tailed)
        statistic, pvalue = stats.ttest_rel(improved_scores, baseline_scores)
        pvalue_one_tailed = pvalue / 2  # One-tailed: improved > baseline
        
        # Effect size
        differences = improved_scores - baseline_scores
        cohens_d = np.mean(differences) / (np.std(differences) + 1e-6)
        
        return {
            't_statistic': statistic,
            'p_value': pvalue_one_tailed,
            'is_significant_improvement': (pvalue_one_tailed < alpha) and (statistic > 0),
            'cohens_d': cohens_d,
            'mean_improvement': np.mean(differences),
            'improvement_percentage': (np.mean(improved_scores) / np.mean(baseline_scores) - 1) * 100
        }
    
    @staticmethod
    def multiple_comparison_correction(p_values: List[float], 
                                      method: str = 'bonferroni') -> List[float]:
        """
        Corrección por comparaciones múltiples.
        
        CRÍTICO: Si haces N tests, debes corregir p-values.
        
        Args:
            p_values: Lista de p-values
            method: 'bonferroni' o 'holm'
            
        Returns:
            p_values corregidos
        """
        n = len(p_values)
        
        if method == 'bonferroni':
            # Bonferroni: p_corrected = p * n
            return [min(p * n, 1.0) for p in p_values]
        
        elif method == 'holm':
            # Holm-Bonferroni (más potente)
            sorted_indices = np.argsort(p_values)
            corrected = [0.0] * n
            
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_values[idx] * (n - i), 1.0)
            
            return corrected
        
        else:
            raise ValueError(f"Unknown method: {method}")


class BenchmarkComparison:
    """
    Comparación sistemática contra modelos baseline.
    
    SIEMPRE compara contra:
    1. Modelo random
    2. Modelo pre-entrenado estándar (GPT-2, etc.)
    3. Tu mejor modelo anterior
    """
    
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.results = {
            'random': {},
            'baseline': {},
            'current': {}
        }
    
    def add_random_baseline(self, metrics: Dict[str, float]):
        """Añade resultados de modelo completamente random."""
        self.results['random'] = metrics
    
    def add_baseline(self, metrics: Dict[str, float]):
        """Añade resultados de modelo baseline (ej: GPT-2)."""
        self.results['baseline'] = metrics
    
    def add_current_model(self, metrics: Dict[str, float]):
        """Añade resultados de tu modelo actual."""
        self.results['current'] = metrics
    
    def generate_comparison_report(self) -> str:
        """Genera reporte de comparación."""
        report = "=" * 60 + "\n"
        report += "BENCHMARK COMPARISON REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for metric in self.metric_names:
            report += f"\n{metric}:\n"
            report += f"  Random:  {self.results['random'].get(metric, 'N/A')}\n"
            report += f"  Baseline: {self.results['baseline'].get(metric, 'N/A')}\n"
            report += f"  Current:  {self.results['current'].get(metric, 'N/A')}\n"
            
            # Calcular mejora vs baseline
            if metric in self.results['baseline'] and metric in self.results['current']:
                baseline_val = self.results['baseline'][metric]
                current_val = self.results['current'][metric]
                improvement = ((current_val - baseline_val) / baseline_val) * 100
                report += f"  Improvement: {improvement:+.2f}%\n"
        
        return report


def _interpret_cohens_d(d: float) -> str:
    """Interpreta Cohen's d según estándares."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
