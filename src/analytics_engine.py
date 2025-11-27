"""
📊 ANALYTICS ENGINE - Sistema de Análisis de Métricas INFINITO
================================================================

Recopila, almacena y analiza métricas del sistema para encontrar
correlaciones y patrones que ayuden a optimizar el Gate.

Métricas capturadas:
- PHI (Φ) - Información integrada
- Coherence - Coherencia temporal
- Complexity - Complejidad del input
- Importance - Score del Gate
- Category - Tipo de información
- Triviality Score - Score del TrivialityGate
- Response Time - Tiempo de procesamiento
- Token Count - Longitud del input
- Saved - Si se guardó o no
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict

# Archivo de métricas
METRICS_FILE = "data/metrics_log.jsonl"  # JSON Lines para append eficiente
ANALYSIS_DIR = "data/analysis"


class MetricsLogger:
    """Logger de métricas para análisis posterior."""
    
    def __init__(self, metrics_file: str = METRICS_FILE):
        self.metrics_file = metrics_file
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.buffer = []
        self.buffer_size = 10  # Flush cada N registros
    
    def log(self, metrics: Dict[str, Any]):
        """Registra una interacción con todas sus métricas."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            **metrics
        }
        
        self.buffer.append(entry)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def flush(self):
        """Escribe el buffer a disco."""
        if not self.buffer:
            return
        
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            for entry in self.buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        self.buffer = []
    
    def __del__(self):
        self.flush()


class MetricsAnalyzer:
    """Analizador de métricas para encontrar correlaciones y patrones."""
    
    def __init__(self, metrics_file: str = METRICS_FILE):
        self.metrics_file = metrics_file
        self.df = None
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    def load_data(self) -> pd.DataFrame:
        """Carga todas las métricas del archivo JSONL."""
        if not os.path.exists(self.metrics_file):
            print(f"⚠️ No existe {self.metrics_file}")
            return pd.DataFrame()
        
        data = []
        with open(self.metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        self.df = pd.DataFrame(data)
        print(f"📊 Cargadas {len(self.df)} interacciones")
        return self.df
    
    def basic_stats(self) -> Dict:
        """Estadísticas básicas de las métricas."""
        if self.df is None or self.df.empty:
            return {}
        
        numeric_cols = ['phi', 'coherence', 'complexity', 'importance', 
                        'combined', 'triviality_score', 'token_count']
        
        # Filtrar solo columnas que existen
        existing_cols = [c for c in numeric_cols if c in self.df.columns]
        
        stats_dict = {}
        for col in existing_cols:
            stats_dict[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std(),
                'min': self.df[col].min(),
                'max': self.df[col].max(),
                'median': self.df[col].median()
            }
        
        # Stats por categoría
        if 'category' in self.df.columns:
            stats_dict['category_counts'] = self.df['category'].value_counts().to_dict()
        
        # Tasa de guardado
        if 'saved' in self.df.columns:
            stats_dict['save_rate'] = self.df['saved'].mean()
        
        return stats_dict
    
    def correlation_matrix(self, save_plot: bool = True) -> pd.DataFrame:
        """Calcula la matriz de correlación entre métricas numéricas."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        numeric_cols = ['phi', 'coherence', 'complexity', 'importance', 
                        'combined', 'triviality_score', 'token_count',
                        'category_bonus']
        
        existing_cols = [c for c in numeric_cols if c in self.df.columns]
        
        if len(existing_cols) < 2:
            print("⚠️ No hay suficientes columnas numéricas para correlación")
            return pd.DataFrame()
        
        corr_matrix = self.df[existing_cols].corr()
        
        if save_plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       fmt='.2f', square=True)
            plt.title('Matriz de Correlación - Métricas INFINITO')
            plt.tight_layout()
            plt.savefig(f'{ANALYSIS_DIR}/correlation_matrix.png', dpi=150)
            plt.close()
            print(f"📈 Guardado: {ANALYSIS_DIR}/correlation_matrix.png")
        
        return corr_matrix
    
    def find_key_correlations(self, threshold: float = 0.5) -> List[Dict]:
        """Encuentra correlaciones significativas."""
        corr = self.correlation_matrix(save_plot=False)
        if corr.empty:
            return []
        
        correlations = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                val = corr.iloc[i, j]
                if abs(val) >= threshold:
                    correlations.append({
                        'var1': corr.columns[i],
                        'var2': corr.columns[j],
                        'correlation': val,
                        'strength': 'strong' if abs(val) > 0.7 else 'moderate'
                    })
        
        return sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    def analyze_saved_vs_not_saved(self) -> Dict:
        """Compara métricas entre mensajes guardados y no guardados."""
        if self.df is None or 'saved' not in self.df.columns:
            return {}
        
        saved = self.df[self.df['saved'] == True]
        not_saved = self.df[self.df['saved'] == False]
        
        numeric_cols = ['phi', 'coherence', 'complexity', 'importance', 
                        'combined', 'triviality_score', 'token_count']
        existing_cols = [c for c in numeric_cols if c in self.df.columns]
        
        comparison = {}
        for col in existing_cols:
            if col in saved.columns and col in not_saved.columns:
                saved_mean = saved[col].mean()
                not_saved_mean = not_saved[col].mean()
                
                # T-test para significancia
                if len(saved) > 1 and len(not_saved) > 1:
                    t_stat, p_value = stats.ttest_ind(
                        saved[col].dropna(), 
                        not_saved[col].dropna()
                    )
                else:
                    t_stat, p_value = 0, 1
                
                comparison[col] = {
                    'saved_mean': saved_mean,
                    'not_saved_mean': not_saved_mean,
                    'difference': saved_mean - not_saved_mean,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return comparison
    
    def category_analysis(self) -> Dict:
        """Analiza métricas por categoría."""
        if self.df is None or 'category' not in self.df.columns:
            return {}
        
        numeric_cols = ['phi', 'coherence', 'complexity', 'importance', 'combined']
        existing_cols = [c for c in numeric_cols if c in self.df.columns]
        
        analysis = {}
        for category in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == category]
            analysis[category] = {
                'count': len(cat_data),
                'save_rate': cat_data['saved'].mean() if 'saved' in cat_data.columns else None,
                **{col: cat_data[col].mean() for col in existing_cols if col in cat_data.columns}
            }
        
        return analysis
    
    def triviality_threshold_analysis(self) -> Dict:
        """Analiza el impacto de diferentes thresholds de trivialidad."""
        if self.df is None or 'triviality_score' not in self.df.columns:
            return {}
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        analysis = {}
        
        for thresh in thresholds:
            trivial = self.df[self.df['triviality_score'] < thresh]
            important = self.df[self.df['triviality_score'] >= thresh]
            
            # Calcular precision/recall si tenemos ground truth
            if 'saved' in self.df.columns:
                # Asumiendo que 'saved' es el ground truth de importancia
                true_positives = len(important[important['saved'] == True])
                false_positives = len(important[important['saved'] == False])
                false_negatives = len(trivial[trivial['saved'] == True])
                true_negatives = len(trivial[trivial['saved'] == False])
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                analysis[thresh] = {
                    'trivial_count': len(trivial),
                    'important_count': len(important),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
        
        return analysis
    
    def temporal_patterns(self) -> Dict:
        """Analiza patrones temporales (hora del día, día de semana)."""
        if self.df is None or 'timestamp' not in self.df.columns:
            return {}
        
        df = self.df.copy()
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Patrones por hora
        hourly = df.groupby('hour').agg({
            'importance': 'mean',
            'phi': 'mean',
            'saved': 'mean'
        }).to_dict()
        
        # Patrones por día
        daily = df.groupby('day_of_week').agg({
            'importance': 'mean',
            'phi': 'mean',
            'saved': 'mean'
        }).to_dict()
        
        return {
            'hourly': hourly,
            'daily': daily
        }
    
    def generate_report(self, save_to_file: bool = True) -> str:
        """Genera un reporte completo de análisis."""
        self.load_data()
        
        if self.df is None or self.df.empty:
            return "⚠️ No hay datos para analizar"
        
        report = []
        report.append("=" * 60)
        report.append("📊 REPORTE DE ANÁLISIS - INFINITO METRICS")
        report.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Total interacciones: {len(self.df)}")
        report.append("=" * 60)
        
        # Estadísticas básicas
        report.append("\n📈 ESTADÍSTICAS BÁSICAS")
        report.append("-" * 40)
        stats = self.basic_stats()
        for metric, values in stats.items():
            if isinstance(values, dict) and 'mean' in values:
                report.append(f"  {metric}:")
                report.append(f"    Media: {values['mean']:.4f}")
                report.append(f"    Std: {values['std']:.4f}")
                report.append(f"    Min/Max: {values['min']:.4f} / {values['max']:.4f}")
        
        if 'save_rate' in stats:
            report.append(f"\n  Tasa de guardado: {stats['save_rate']*100:.1f}%")
        
        if 'category_counts' in stats:
            report.append("\n  Distribución por categoría:")
            for cat, count in stats['category_counts'].items():
                report.append(f"    {cat}: {count}")
        
        # Correlaciones clave
        report.append("\n🔗 CORRELACIONES SIGNIFICATIVAS")
        report.append("-" * 40)
        correlations = self.find_key_correlations(threshold=0.4)
        if correlations:
            for corr in correlations[:10]:
                report.append(f"  {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f} ({corr['strength']})")
        else:
            report.append("  No se encontraron correlaciones significativas")
        
        # Guardado vs No guardado
        report.append("\n💾 ANÁLISIS: GUARDADO vs NO GUARDADO")
        report.append("-" * 40)
        comparison = self.analyze_saved_vs_not_saved()
        for metric, data in comparison.items():
            if data.get('significant'):
                report.append(f"  {metric}: ⭐ SIGNIFICATIVO")
            else:
                report.append(f"  {metric}:")
            report.append(f"    Guardado: {data['saved_mean']:.4f}")
            report.append(f"    No guardado: {data['not_saved_mean']:.4f}")
            report.append(f"    Diferencia: {data['difference']:+.4f} (p={data['p_value']:.4f})")
        
        # Análisis por categoría
        report.append("\n📁 ANÁLISIS POR CATEGORÍA")
        report.append("-" * 40)
        cat_analysis = self.category_analysis()
        for cat, data in cat_analysis.items():
            report.append(f"  {cat}:")
            report.append(f"    N={data['count']}, Save rate={data.get('save_rate', 0)*100:.0f}%")
            if 'phi' in data:
                report.append(f"    PHI medio: {data['phi']:.4f}")
        
        # Recomendaciones
        report.append("\n💡 RECOMENDACIONES")
        report.append("-" * 40)
        recommendations = self._generate_recommendations(stats, correlations, comparison)
        for rec in recommendations:
            report.append(f"  • {rec}")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        if save_to_file:
            report_path = f"{ANALYSIS_DIR}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Reporte guardado en: {report_path}")
        
        # Generar gráficos
        self.correlation_matrix(save_plot=True)
        self._plot_distributions()
        
        return report_text
    
    def _generate_recommendations(self, stats: Dict, correlations: List, comparison: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis."""
        recommendations = []
        
        # Basado en tasa de guardado
        if 'save_rate' in stats:
            rate = stats['save_rate']
            if rate > 0.7:
                recommendations.append("Tasa de guardado alta (>70%). Considera subir el threshold para ser más selectivo.")
            elif rate < 0.3:
                recommendations.append("Tasa de guardado baja (<30%). El sistema es muy selectivo, ¿demasiado?")
        
        # Basado en correlaciones
        for corr in correlations[:3]:
            if corr['var1'] == 'phi' or corr['var2'] == 'phi':
                other = corr['var2'] if corr['var1'] == 'phi' else corr['var1']
                if corr['correlation'] > 0:
                    recommendations.append(f"PHI correlaciona fuerte con {other}. Usar {other} como proxy puede ser útil.")
        
        # Basado en comparación saved/not_saved
        for metric, data in comparison.items():
            if data.get('significant') and abs(data['difference']) > 0.1:
                if data['difference'] > 0:
                    recommendations.append(f"{metric} es significativamente mayor en mensajes guardados. Buen discriminador.")
                else:
                    recommendations.append(f"{metric} es menor en mensajes guardados. Revisar lógica del Gate.")
        
        if not recommendations:
            recommendations.append("Recopilar más datos para obtener insights significativos.")
        
        return recommendations
    
    def _plot_distributions(self):
        """Genera gráficos de distribución de métricas."""
        if self.df is None or self.df.empty:
            return
        
        numeric_cols = ['phi', 'coherence', 'complexity', 'importance', 'combined']
        existing_cols = [c for c in numeric_cols if c in self.df.columns]
        
        if not existing_cols:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(existing_cols):
            if i < len(axes):
                # Histograma con separación por saved
                if 'saved' in self.df.columns:
                    saved = self.df[self.df['saved'] == True][col]
                    not_saved = self.df[self.df['saved'] == False][col]
                    
                    axes[i].hist(saved, bins=20, alpha=0.5, label='Guardado', color='green')
                    axes[i].hist(not_saved, bins=20, alpha=0.5, label='No guardado', color='red')
                    axes[i].legend()
                else:
                    axes[i].hist(self.df[col], bins=20, color='blue', alpha=0.7)
                
                axes[i].set_title(f'Distribución de {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frecuencia')
        
        # Ocultar ejes vacíos
        for i in range(len(existing_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{ANALYSIS_DIR}/distributions.png', dpi=150)
        plt.close()
        print(f"📈 Guardado: {ANALYSIS_DIR}/distributions.png")


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

_logger = None

def get_logger() -> MetricsLogger:
    """Obtiene el logger singleton."""
    global _logger
    if _logger is None:
        _logger = MetricsLogger()
    return _logger


def log_interaction(
    text: str,
    phi: float,
    coherence: float,
    complexity: float,
    importance: float,
    combined: float,
    category: str,
    category_bonus: float,
    triviality_score: float,
    is_question: bool,
    saved: bool,
    response_time_ms: Optional[float] = None,
    **extra_metrics
):
    """Registra una interacción completa."""
    logger = get_logger()
    
    metrics = {
        'text': text[:100],  # Truncar para privacidad
        'text_hash': hash(text),  # Para identificar duplicados
        'token_count': len(text.split()),
        'char_count': len(text),
        'phi': phi,
        'coherence': coherence,
        'complexity': complexity,
        'importance': importance,
        'combined': combined,
        'category': category,
        'category_bonus': category_bonus,
        'triviality_score': triviality_score,
        'is_question': is_question,
        'saved': saved,
        'response_time_ms': response_time_ms,
        **extra_metrics
    }
    
    logger.log(metrics)


def analyze_and_report():
    """Ejecuta análisis completo y genera reporte."""
    analyzer = MetricsAnalyzer()
    return analyzer.generate_report()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        print(analyze_and_report())
    else:
        print("📊 INFINITO Analytics Engine")
        print("=" * 40)
        print("Uso:")
        print("  python analytics_engine.py report  - Generar reporte completo")
        print("")
        print("Desde código:")
        print("  from analytics_engine import log_interaction, analyze_and_report")
        print("  log_interaction(text, phi, coherence, ...)")
        print("  analyze_and_report()")
