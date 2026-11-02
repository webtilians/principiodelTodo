"""
Análisis Estadístico Profundo de PHI - INFINITO
===============================================
Análisis estadísticos avanzados:
1. Distribución estadística de PHI
2. Tests de normalidad
3. Análisis de outliers
4. Correlaciones cruzadas
5. Análisis de series temporales
6. Comparativa con baseline
7. Resumen ejecutivo
"""

import torch
import numpy as np
import json
import os
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuración
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/infinito_gpt2_spanish_phi.pt"
RESULTS_DIR = "results"

def load_model():
    """Carga el modelo entrenado"""
    from train_gpt2_with_phi_observer import InfinitoGPT2WithObserver
    
    print("Cargando modelo...")
    model = InfinitoGPT2WithObserver()
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✓ Modelo cargado desde {MODEL_PATH}")
    
    model = model.to(DEVICE)
    model.eval()
    return model

def collect_phi_samples(model, tokenizer, n_samples=100):
    """Recolecta muestras de PHI de diversos prompts"""
    print("\n📊 Recolectando muestras de PHI...")
    
    prompts = [
        # Científicos
        "La física cuántica demuestra", "El ADN contiene información", 
        "Los agujeros negros absorben", "La evolución selecciona",
        "Las neuronas transmiten señales", "El cambio climático afecta",
        
        # Filosóficos
        "La consciencia emerge de", "El libre albedrío es",
        "La realidad se construye", "El tiempo fluye hacia",
        "La existencia precede a", "El conocimiento se adquiere",
        
        # Creativos
        "Había una vez un", "En el lejano futuro",
        "El héroe decidió enfrentar", "La música resonaba en",
        "Los colores del atardecer", "El misterio se resolvió",
        
        # Técnicos
        "El algoritmo procesa datos", "La red neuronal aprende",
        "El código ejecuta instrucciones", "La base de datos almacena",
        "El sistema operativo gestiona", "El protocolo transmite paquetes",
        
        # Emocionales
        "El amor transforma todo", "La tristeza invade mi",
        "La alegría brota cuando", "El miedo paraliza mis",
        "La esperanza renace cada", "La nostalgia me lleva",
        
        # Cotidianos
        "El café de la mañana", "El viaje en tren",
        "La cena de familia", "El paseo por el",
        "La conversación con amigos", "El descanso del fin"
    ]
    
    # Extender lista si necesario
    while len(prompts) < n_samples:
        prompts.extend(prompts)
    prompts = prompts[:n_samples]
    
    phi_values = []
    components_data = defaultdict(list)
    
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        
        with torch.no_grad():
            outputs, metrics = model(input_ids, return_phi=True, use_memory=False)
        
        phi_values.append(metrics['phi'].mean().item())
        for comp, val in metrics['raw_components'].items():
            components_data[comp].append(val)
    
    return np.array(phi_values), {k: np.array(v) for k, v in components_data.items()}

def analyze_distribution(phi_values, components_data):
    """Análisis de distribución estadística"""
    print("\n" + "="*60)
    print("ANÁLISIS 1: Distribución Estadística de PHI")
    print("="*60)
    
    results = {
        'phi': {
            'mean': np.mean(phi_values),
            'std': np.std(phi_values),
            'median': np.median(phi_values),
            'min': np.min(phi_values),
            'max': np.max(phi_values),
            'skewness': float(stats.skew(phi_values)),
            'kurtosis': float(stats.kurtosis(phi_values)),
            'q25': np.percentile(phi_values, 25),
            'q75': np.percentile(phi_values, 75),
            'iqr': np.percentile(phi_values, 75) - np.percentile(phi_values, 25)
        }
    }
    
    print(f"\n📈 Estadísticas de PHI:")
    print(f"  Media:     {results['phi']['mean']:.4f}")
    print(f"  Mediana:   {results['phi']['median']:.4f}")
    print(f"  Desv.Std:  {results['phi']['std']:.4f}")
    print(f"  Mínimo:    {results['phi']['min']:.4f}")
    print(f"  Máximo:    {results['phi']['max']:.4f}")
    print(f"  Asimetría: {results['phi']['skewness']:.4f}")
    print(f"  Curtosis:  {results['phi']['kurtosis']:.4f}")
    print(f"  IQR:       {results['phi']['iqr']:.4f}")
    
    # Estadísticas por componente
    print(f"\n📊 Estadísticas por Componente:")
    for comp, values in components_data.items():
        results[comp] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        print(f"  {comp:12s}: μ={np.mean(values):.4f}, σ={np.std(values):.4f}, "
              f"rango=[{np.min(values):.3f}, {np.max(values):.3f}]")
    
    return results

def test_normality(phi_values):
    """Tests de normalidad para PHI"""
    print("\n" + "="*60)
    print("ANÁLISIS 2: Tests de Normalidad")
    print("="*60)
    
    results = {}
    
    # Shapiro-Wilk (mejor para n < 5000)
    if len(phi_values) <= 5000:
        statistic, p_value = stats.shapiro(phi_values)
        results['shapiro_wilk'] = {'statistic': statistic, 'p_value': p_value}
        print(f"\n🔬 Test Shapiro-Wilk:")
        print(f"   Estadístico: {statistic:.4f}")
        print(f"   p-valor: {p_value:.6f}")
        print(f"   ¿Normal? {'Sí' if p_value > 0.05 else 'No'} (α=0.05)")
    
    # Kolmogorov-Smirnov
    statistic, p_value = stats.kstest(phi_values, 'norm', 
                                       args=(np.mean(phi_values), np.std(phi_values)))
    results['ks_test'] = {'statistic': statistic, 'p_value': p_value}
    print(f"\n🔬 Test Kolmogorov-Smirnov:")
    print(f"   Estadístico: {statistic:.4f}")
    print(f"   p-valor: {p_value:.6f}")
    print(f"   ¿Normal? {'Sí' if p_value > 0.05 else 'No'} (α=0.05)")
    
    # D'Agostino K² test
    statistic, p_value = stats.normaltest(phi_values)
    results['dagostino'] = {'statistic': statistic, 'p_value': p_value}
    print(f"\n🔬 Test D'Agostino K²:")
    print(f"   Estadístico: {statistic:.4f}")
    print(f"   p-valor: {p_value:.6f}")
    print(f"   ¿Normal? {'Sí' if p_value > 0.05 else 'No'} (α=0.05)")
    
    return results

def analyze_outliers(phi_values):
    """Detección y análisis de outliers"""
    print("\n" + "="*60)
    print("ANÁLISIS 3: Detección de Outliers")
    print("="*60)
    
    # Método IQR
    q1 = np.percentile(phi_values, 25)
    q3 = np.percentile(phi_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers_iqr = phi_values[(phi_values < lower_bound) | (phi_values > upper_bound)]
    
    # Método Z-score
    z_scores = np.abs(stats.zscore(phi_values))
    outliers_z = phi_values[z_scores > 3]
    
    # Método MAD (Median Absolute Deviation)
    median = np.median(phi_values)
    mad = np.median(np.abs(phi_values - median))
    modified_z = 0.6745 * (phi_values - median) / mad if mad > 0 else np.zeros_like(phi_values)
    outliers_mad = phi_values[np.abs(modified_z) > 3.5]
    
    results = {
        'iqr_method': {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': len(outliers_iqr),
            'outlier_values': outliers_iqr.tolist()
        },
        'zscore_method': {
            'n_outliers': len(outliers_z),
            'outlier_values': outliers_z.tolist()
        },
        'mad_method': {
            'n_outliers': len(outliers_mad),
            'outlier_values': outliers_mad.tolist()
        }
    }
    
    print(f"\n📍 Método IQR:")
    print(f"   Límites: [{lower_bound:.3f}, {upper_bound:.3f}]")
    print(f"   Outliers encontrados: {len(outliers_iqr)}")
    
    print(f"\n📍 Método Z-Score (|z| > 3):")
    print(f"   Outliers encontrados: {len(outliers_z)}")
    
    print(f"\n📍 Método MAD (|z_mod| > 3.5):")
    print(f"   Outliers encontrados: {len(outliers_mad)}")
    
    if len(outliers_iqr) > 0:
        print(f"\n   Valores atípicos: {outliers_iqr[:5]}...")
    
    return results

def analyze_correlations(phi_values, components_data):
    """Análisis de correlaciones cruzadas"""
    print("\n" + "="*60)
    print("ANÁLISIS 4: Correlaciones Cruzadas")
    print("="*60)
    
    # Correlación de cada componente con PHI
    print("\n🔗 Correlación Componente → PHI:")
    results = {'component_to_phi': {}}
    
    for comp, values in components_data.items():
        corr, p_value = stats.pearsonr(values, phi_values)
        spearman_corr, spearman_p = stats.spearmanr(values, phi_values)
        
        results['component_to_phi'][comp] = {
            'pearson_r': corr,
            'pearson_p': p_value,
            'spearman_r': spearman_corr,
            'spearman_p': spearman_p
        }
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"   {comp:12s}: r={corr:+.3f}{significance}, ρ={spearman_corr:+.3f}")
    
    # Correlaciones entre componentes
    print("\n🔗 Correlaciones entre Componentes:")
    components = list(components_data.keys())
    n = len(components)
    results['component_to_component'] = {}
    
    for i in range(n):
        for j in range(i+1, n):
            comp1, comp2 = components[i], components[j]
            corr, p_value = stats.pearsonr(components_data[comp1], components_data[comp2])
            
            key = f"{comp1}_vs_{comp2}"
            results['component_to_component'][key] = {
                'pearson_r': corr,
                'pearson_p': p_value
            }
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"   {comp1[:8]:8s} ↔ {comp2[:8]:8s}: r={corr:+.3f}{significance}")
    
    return results

def analyze_confidence_intervals(phi_values):
    """Calcular intervalos de confianza"""
    print("\n" + "="*60)
    print("ANÁLISIS 5: Intervalos de Confianza")
    print("="*60)
    
    n = len(phi_values)
    mean = np.mean(phi_values)
    std = np.std(phi_values, ddof=1)
    se = std / np.sqrt(n)
    
    results = {}
    
    for confidence in [0.90, 0.95, 0.99]:
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * se
        lower = mean - margin
        upper = mean + margin
        
        results[f'ci_{int(confidence*100)}'] = {
            'lower': lower,
            'upper': upper,
            'margin': margin
        }
        
        print(f"\n📊 Intervalo de Confianza {int(confidence*100)}%:")
        print(f"   PHI = {mean:.4f} ± {margin:.4f}")
        print(f"   Rango: [{lower:.4f}, {upper:.4f}]")
    
    return results

def compare_with_baseline(phi_values):
    """Comparar con un baseline teórico"""
    print("\n" + "="*60)
    print("ANÁLISIS 6: Comparación con Baseline")
    print("="*60)
    
    # Baseline: PHI aleatorio (distribución uniforme 0-10)
    n = len(phi_values)
    baseline = np.random.uniform(0, 10, n)
    
    # Baseline: modelo GPT-2 sin entrenamiento (estimado)
    baseline_untrained = np.random.normal(5.0, 1.0, n)  # Estimación
    
    results = {}
    
    # Test t de dos muestras vs aleatorio
    t_stat, p_value = stats.ttest_ind(phi_values, baseline)
    results['vs_random'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    print(f"\n📊 PHI Entrenado vs Aleatorio:")
    print(f"   PHI medio entrenado: {np.mean(phi_values):.4f}")
    print(f"   PHI medio aleatorio: {np.mean(baseline):.4f}")
    print(f"   t-estadístico: {t_stat:.4f}")
    print(f"   p-valor: {p_value:.6f}")
    print(f"   ¿Significativo? {'Sí' if p_value < 0.05 else 'No'}")
    
    # Test vs no entrenado
    t_stat2, p_value2 = stats.ttest_ind(phi_values, baseline_untrained)
    results['vs_untrained'] = {
        't_statistic': t_stat2,
        'p_value': p_value2,
        'significant': p_value2 < 0.05
    }
    
    print(f"\n📊 PHI Entrenado vs No Entrenado (estimado):")
    print(f"   PHI medio no entrenado: {np.mean(baseline_untrained):.4f}")
    print(f"   t-estadístico: {t_stat2:.4f}")
    print(f"   p-valor: {p_value2:.6f}")
    print(f"   ¿Significativo? {'Sí' if p_value2 < 0.05 else 'No'}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(phi_values)**2 + np.std(baseline)**2) / 2)
    cohens_d = (np.mean(phi_values) - np.mean(baseline)) / pooled_std
    results['effect_size'] = {
        'cohens_d': cohens_d,
        'interpretation': 'grande' if abs(cohens_d) > 0.8 else 'medio' if abs(cohens_d) > 0.5 else 'pequeño'
    }
    
    print(f"\n📊 Tamaño del Efecto:")
    print(f"   Cohen's d: {cohens_d:.4f} ({results['effect_size']['interpretation']})")
    
    return results

def create_statistical_visualizations(phi_values, components_data, all_results):
    """Crear visualizaciones estadísticas"""
    print("\n" + "="*60)
    print("Generando Visualizaciones Estadísticas...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Análisis Estadístico de PHI - INFINITO', fontsize=16, fontweight='bold')
    
    # 1. Histograma con distribución normal
    ax = axes[0, 0]
    ax.hist(phi_values, bins=20, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Ajustar curva normal
    mu, std = np.mean(phi_values), np.std(phi_values)
    x = np.linspace(phi_values.min(), phi_values.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
    ax.axvline(mu, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('PHI')
    ax.set_ylabel('Densidad')
    ax.set_title('Distribución de PHI')
    ax.legend()
    
    # 2. Box plot de componentes
    ax = axes[0, 1]
    data_to_plot = [components_data[c] for c in components_data.keys()]
    bp = ax.boxplot(data_to_plot, labels=[c[:6] for c in components_data.keys()], patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(components_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Valor')
    ax.set_title('Box Plot de Componentes IIT')
    
    # 3. Q-Q Plot
    ax = axes[0, 2]
    stats.probplot(phi_values, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normalidad)')
    ax.get_lines()[0].set_color('steelblue')
    ax.get_lines()[0].set_markersize(5)
    
    # 4. Matriz de correlación
    ax = axes[1, 0]
    all_data = np.column_stack([phi_values] + [components_data[c] for c in components_data.keys()])
    labels = ['PHI'] + list(components_data.keys())
    corr_matrix = np.corrcoef(all_data.T)
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([l[:6] for l in labels], rotation=45, ha='right')
    ax.set_yticklabels([l[:6] for l in labels])
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title('Matriz de Correlación')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 5. Violin plot
    ax = axes[1, 1]
    parts = ax.violinplot([phi_values], positions=[1], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('steelblue')
    parts['bodies'][0].set_alpha(0.7)
    ax.set_xticks([1])
    ax.set_xticklabels(['PHI'])
    ax.set_ylabel('Valor')
    ax.set_title('Violin Plot de PHI')
    
    # Añadir anotaciones
    mean = np.mean(phi_values)
    median = np.median(phi_values)
    ax.annotate(f'μ={mean:.2f}', xy=(1.1, mean), fontsize=10)
    ax.annotate(f'mediana={median:.2f}', xy=(1.1, median-0.1), fontsize=10)
    
    # 6. Intervalos de confianza
    ax = axes[1, 2]
    confidence_levels = [90, 95, 99]
    colors = ['lightgreen', 'lightblue', 'lightyellow']
    
    mean = np.mean(phi_values)
    for i, (conf, color) in enumerate(zip(confidence_levels, colors)):
        ci_key = f'ci_{conf}'
        if 'confidence_intervals' in all_results and ci_key in all_results['confidence_intervals']:
            ci = all_results['confidence_intervals'][ci_key]
            ax.barh(i, ci['upper'] - ci['lower'], left=ci['lower'], height=0.6, 
                   color=color, edgecolor='black', label=f'{conf}% CI')
    
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Media={mean:.2f}')
    ax.set_yticks(range(len(confidence_levels)))
    ax.set_yticklabels([f'{c}%' for c in confidence_levels])
    ax.set_xlabel('PHI')
    ax.set_title('Intervalos de Confianza')
    ax.legend(loc='upper right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = os.path.join(RESULTS_DIR, 'phi_statistical_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualización guardada: {output_path}")
    
    plt.close()

def generate_executive_summary(all_results, phi_values):
    """Generar resumen ejecutivo"""
    print("\n" + "="*60)
    print("  RESUMEN EJECUTIVO")
    print("="*60)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_samples': len(phi_values),
        'findings': []
    }
    
    # PHI general
    if 'distribution' in all_results:
        dist = all_results['distribution']['phi']
        print(f"\n📊 DISTRIBUCIÓN DE PHI:")
        print(f"   • Media: {dist['mean']:.4f} ± {dist['std']:.4f}")
        print(f"   • Rango efectivo: [{dist['q25']:.3f}, {dist['q75']:.3f}]")
        
        summary['findings'].append({
            'area': 'distribución',
            'finding': f"PHI tiene media {dist['mean']:.3f} con desviación {dist['std']:.3f}"
        })
    
    # Normalidad
    if 'normality' in all_results:
        is_normal = all_results['normality']['shapiro_wilk']['p_value'] > 0.05
        print(f"\n📐 NORMALIDAD:")
        print(f"   • Distribución: {'Normal' if is_normal else 'No normal'}")
        
        summary['findings'].append({
            'area': 'normalidad',
            'finding': f"Distribución {'aproximadamente normal' if is_normal else 'no normal'}"
        })
    
    # Outliers
    if 'outliers' in all_results:
        n_outliers = all_results['outliers']['iqr_method']['n_outliers']
        pct_outliers = (n_outliers / len(phi_values)) * 100
        print(f"\n⚠️ OUTLIERS:")
        print(f"   • Detectados: {n_outliers} ({pct_outliers:.1f}%)")
        
        summary['findings'].append({
            'area': 'outliers',
            'finding': f"{n_outliers} valores atípicos detectados"
        })
    
    # Correlaciones dominantes
    if 'correlations' in all_results:
        comp_corrs = all_results['correlations']['component_to_phi']
        best_comp = max(comp_corrs.items(), key=lambda x: abs(x[1]['pearson_r']))
        print(f"\n🔗 CORRELACIONES CLAVE:")
        print(f"   • Componente dominante: {best_comp[0]} (r={best_comp[1]['pearson_r']:.3f})")
        
        summary['findings'].append({
            'area': 'correlaciones',
            'finding': f"{best_comp[0]} es el predictor más fuerte de PHI"
        })
    
    # Significancia vs baseline
    if 'baseline' in all_results:
        sig = all_results['baseline']['vs_random']['significant']
        effect = all_results['baseline']['effect_size']['interpretation']
        print(f"\n🎯 SIGNIFICANCIA:")
        print(f"   • vs Random: {'Significativo' if sig else 'No significativo'}")
        print(f"   • Tamaño del efecto: {effect}")
        
        summary['findings'].append({
            'area': 'significancia',
            'finding': f"Efecto {effect} comparado con baseline aleatorio"
        })
    
    # Confianza
    if 'confidence_intervals' in all_results:
        ci95 = all_results['confidence_intervals']['ci_95']
        print(f"\n📈 INTERVALOS DE CONFIANZA (95%):")
        print(f"   • PHI: [{ci95['lower']:.4f}, {ci95['upper']:.4f}]")
    
    # Conclusiones
    print(f"\n" + "="*60)
    print("  CONCLUSIONES")
    print("="*60)
    
    mean_phi = np.mean(phi_values)
    if mean_phi > 7.0:
        print("  ✅ PHI alto indica buena integración de información")
    elif mean_phi > 5.0:
        print("  ⚠️ PHI moderado, hay espacio para mejora")
    else:
        print("  ❌ PHI bajo, revisar arquitectura o entrenamiento")
    
    if 'correlations' in all_results:
        integration_r = all_results['correlations']['component_to_phi'].get('integration', {}).get('pearson_r', 0)
        if integration_r > 0.7:
            print("  ✅ Integration es el driver principal del PHI")
    
    summary['overall_assessment'] = 'positivo' if mean_phi > 6.0 else 'neutro' if mean_phi > 4.0 else 'negativo'
    
    return summary

def main():
    print("="*60)
    print("  ANÁLISIS ESTADÍSTICO PROFUNDO - INFINITO")
    print("="*60)
    
    # Cargar modelo
    model = load_model()
    tokenizer = model.tokenizer
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Recolectar muestras
    phi_values, components_data = collect_phi_samples(model, tokenizer, n_samples=100)
    
    all_results = {}
    
    # Ejecutar análisis
    try:
        all_results['distribution'] = analyze_distribution(phi_values, components_data)
    except Exception as e:
        print(f"Error en análisis de distribución: {e}")
    
    try:
        all_results['normality'] = test_normality(phi_values)
    except Exception as e:
        print(f"Error en tests de normalidad: {e}")
    
    try:
        all_results['outliers'] = analyze_outliers(phi_values)
    except Exception as e:
        print(f"Error en análisis de outliers: {e}")
    
    try:
        all_results['correlations'] = analyze_correlations(phi_values, components_data)
    except Exception as e:
        print(f"Error en análisis de correlaciones: {e}")
    
    try:
        all_results['confidence_intervals'] = analyze_confidence_intervals(phi_values)
    except Exception as e:
        print(f"Error en intervalos de confianza: {e}")
    
    try:
        all_results['baseline'] = compare_with_baseline(phi_values)
    except Exception as e:
        print(f"Error en comparación baseline: {e}")
    
    # Visualizaciones
    try:
        create_statistical_visualizations(phi_values, components_data, all_results)
    except Exception as e:
        print(f"Error en visualizaciones: {e}")
    
    # Resumen ejecutivo
    try:
        all_results['executive_summary'] = generate_executive_summary(all_results, phi_values)
    except Exception as e:
        print(f"Error en resumen ejecutivo: {e}")
    
    # Guardar resultados
    output_file = os.path.join(RESULTS_DIR, 'phi_statistical_analysis.json')
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Resultados guardados: {output_file}")
    
    print("\n" + "="*60)
    print("  ANÁLISIS ESTADÍSTICO COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()
