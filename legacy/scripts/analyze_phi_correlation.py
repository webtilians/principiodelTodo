#!/usr/bin/env python3
"""
📊 CORRELACIÓN PHI: Generación vs Entrenamiento
================================================

Compara los patrones de PHI durante generación con los del entrenamiento.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

print("="*70)
print("📊 ANÁLISIS DE CORRELACIÓN PHI")
print("="*70)

# =============================================================================
# CARGAR DATOS
# =============================================================================

# Datos de generación
with open('results/phi_exhaustive_analysis.json', 'r', encoding='utf-8') as f:
    gen_data = json.load(f)

# Datos de entrenamiento
with open('results/phi_training_log.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

print(f"✅ Datos de generación: {len(gen_data['all_results'])} prompts")
print(f"✅ Datos de entrenamiento: {len(train_data)} batches")

# =============================================================================
# ESTADÍSTICAS GENERALES
# =============================================================================

print("\n" + "="*70)
print("📈 COMPARACIÓN GENERAL")
print("="*70)

# PHI en generación
gen_phis = [r['phi'] for r in gen_data['all_results']]
gen_mean = np.mean(gen_phis)
gen_std = np.std(gen_phis)

# PHI en entrenamiento
train_phis = [r['phi'] for r in train_data]
train_mean = np.mean(train_phis)
train_std = np.std(train_phis)

print(f"\n{'Métrica':<25} | {'Entrenamiento':>15} | {'Generación':>15}")
print("-"*60)
print(f"{'PHI promedio':<25} | {train_mean:>15.2f} | {gen_mean:>15.2f}")
print(f"{'PHI std':<25} | {train_std:>15.2f} | {gen_std:>15.2f}")
print(f"{'PHI min':<25} | {min(train_phis):>15.2f} | {min(gen_phis):>15.2f}")
print(f"{'PHI max':<25} | {max(train_phis):>15.2f} | {max(gen_phis):>15.2f}")

# =============================================================================
# EVOLUCIÓN PHI DURANTE ENTRENAMIENTO
# =============================================================================

print("\n" + "="*70)
print("📉 EVOLUCIÓN PHI DURANTE ENTRENAMIENTO")
print("="*70)

# Agrupar por epoch
epoch_phis = defaultdict(list)
for entry in train_data:
    epoch_phis[entry['epoch']].append(entry['phi'])

print(f"\n{'Epoch':<10} | {'PHI promedio':>15} | {'PHI std':>12} | {'Samples':>10}")
print("-"*55)
for epoch in sorted(epoch_phis.keys()):
    phis = epoch_phis[epoch]
    print(f"{epoch:<10} | {np.mean(phis):>15.2f} | {np.std(phis):>12.2f} | {len(phis):>10}")

# =============================================================================
# COMPONENTES PHI: ENTRENAMIENTO VS GENERACIÓN
# =============================================================================

print("\n" + "="*70)
print("🔬 COMPONENTES PHI")
print("="*70)

# Componentes de entrenamiento
train_components = defaultdict(list)
for entry in train_data:
    if 'components' in entry:
        for comp, val in entry['components'].items():
            train_components[comp].append(val)

# Componentes de generación
gen_components = defaultdict(list)
for entry in gen_data['all_results']:
    for comp, val in entry['components'].items():
        gen_components[comp].append(val)

print(f"\n{'Componente':<15} | {'Train Mean':>12} | {'Gen Mean':>12} | {'Diferencia':>12}")
print("-"*60)
for comp in ['temporal', 'integration', 'complexity', 'attention']:
    train_val = np.mean(train_components[comp]) if train_components[comp] else 0
    gen_val = np.mean(gen_components[comp]) if gen_components[comp] else 0
    diff = gen_val - train_val
    print(f"{comp:<15} | {train_val:>12.4f} | {gen_val:>12.4f} | {diff:>+12.4f}")

# =============================================================================
# PHI POR CATEGORÍA VS DISTRIBUCIÓN DE ENTRENAMIENTO
# =============================================================================

print("\n" + "="*70)
print("📊 PHI POR CATEGORÍA (ordenado)")
print("="*70)

category_summary = gen_data['category_summary']

print(f"\n{'Categoría':<25} | {'PHI Mean':>10} | {'vs Train':>12} | {'Interpretación':<20}")
print("-"*75)

for cat in sorted(category_summary.keys(), key=lambda x: -category_summary[x]['mean']):
    cat_mean = category_summary[cat]['mean']
    diff = cat_mean - train_mean
    
    if diff > 0.1:
        interp = "↑ Más integrado"
    elif diff < -0.1:
        interp = "↓ Menos integrado"
    else:
        interp = "≈ Similar"
    
    print(f"{cat:<25} | {cat_mean:>10.2f} | {diff:>+12.2f} | {interp:<20}")

# =============================================================================
# INSIGHTS CLAVE
# =============================================================================

print("\n" + "="*70)
print("💡 INSIGHTS CLAVE")
print("="*70)

# 1. Comparación entrenamiento vs generación
phi_diff = gen_mean - train_mean
print(f"\n1. PHI Generación vs Entrenamiento: {phi_diff:+.2f}")
if phi_diff < -1:
    print("   → El modelo genera con MENOS integración que durante entrenamiento")
    print("   → Posible: El modelo aprendió patrones pero los aplica de forma diferente")
elif phi_diff > 1:
    print("   → El modelo genera con MÁS integración que durante entrenamiento")
else:
    print("   → Integración SIMILAR entre entrenamiento y generación")

# 2. Correlaciones importantes
print(f"\n2. Correlaciones encontradas:")
for key, val in gen_data['correlations'].items():
    if not np.isnan(val):
        strength = "fuerte" if abs(val) > 0.7 else "moderada" if abs(val) > 0.3 else "débil"
        print(f"   • {key}: r = {val:.3f} ({strength})")

# 3. Categoría más integrada
best_cat = max(category_summary.items(), key=lambda x: x[1]['mean'])
worst_cat = min(category_summary.items(), key=lambda x: x[1]['mean'])
print(f"\n3. Categoría con MAYOR integración: {best_cat[0]} (PHI={best_cat[1]['mean']:.2f})")
print(f"   Categoría con MENOR integración: {worst_cat[0]} (PHI={worst_cat[1]['mean']:.2f})")

# 4. Componente dominante
correlations = gen_data['correlations']
comp_corrs = {
    'temporal': correlations.get('temporal_vs_phi', 0),
    'integration': correlations.get('integration_vs_phi', 0),
    'complexity': correlations.get('complexity_vs_phi', 0),
}
dominant = max(comp_corrs.items(), key=lambda x: abs(x[1]) if not np.isnan(x[1]) else 0)
print(f"\n4. Componente DOMINANTE en PHI: {dominant[0]} (r={dominant[1]:.3f})")
print("   → Este componente explica la mayor variación en PHI")

# 5. PHI por capa
layer_phis = gen_data['layer_phi_averages']
peak_layer = max(layer_phis.items(), key=lambda x: x[1])
print(f"\n5. Capa con PHI máximo: Layer {peak_layer[0]} (PHI={peak_layer[1]:.2f})")
print("   → La integración de información ocurre principalmente en capas intermedias")

# =============================================================================
# GUARDAR RESUMEN
# =============================================================================

summary = {
    'training': {
        'phi_mean': train_mean,
        'phi_std': train_std,
        'total_batches': len(train_data),
    },
    'generation': {
        'phi_mean': gen_mean,
        'phi_std': gen_std,
        'total_prompts': len(gen_data['all_results']),
    },
    'difference': phi_diff,
    'category_ranking': {cat: summary['mean'] for cat, summary in sorted(
        category_summary.items(), key=lambda x: -x[1]['mean']
    )},
    'insights': {
        'dominant_component': dominant[0],
        'peak_layer': int(peak_layer[0]),
        'best_category': best_cat[0],
        'worst_category': worst_cat[0],
    }
}

with open('results/phi_correlation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Resumen guardado en: results/phi_correlation_summary.json")

# =============================================================================
# VISUALIZACIÓN
# =============================================================================

print("\n" + "="*70)
print("📊 GENERANDO VISUALIZACIONES...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. PHI por categoría
ax1 = axes[0, 0]
categories = list(category_summary.keys())
means = [category_summary[c]['mean'] for c in categories]
stds = [category_summary[c]['std'] for c in categories]
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
bars = ax1.barh(categories, means, xerr=stds, color=colors, capsize=3)
ax1.axvline(train_mean, color='red', linestyle='--', label=f'Train mean ({train_mean:.2f})')
ax1.set_xlabel('PHI')
ax1.set_title('PHI por Categoría de Prompt')
ax1.legend()

# 2. Evolución PHI en entrenamiento
ax2 = axes[0, 1]
batches = [i for i in range(len(train_phis))]
ax2.plot(batches, train_phis, alpha=0.5, linewidth=0.5)
# Moving average
window = 50
if len(train_phis) > window:
    ma = np.convolve(train_phis, np.ones(window)/window, mode='valid')
    ax2.plot(range(window-1, len(train_phis)), ma, color='red', linewidth=2, label=f'MA({window})')
ax2.set_xlabel('Batch')
ax2.set_ylabel('PHI')
ax2.set_title('Evolución PHI durante Entrenamiento')
ax2.legend()

# 3. PHI por capa
ax3 = axes[1, 0]
layers = [int(k) for k in layer_phis.keys()]
layer_values = [layer_phis[str(l)] for l in sorted(layers)]
ax3.plot(sorted(layers), layer_values, 'o-', markersize=8, linewidth=2)
ax3.fill_between(sorted(layers), layer_values, alpha=0.3)
ax3.set_xlabel('Capa')
ax3.set_ylabel('PHI')
ax3.set_title('PHI por Capa del Transformer')
ax3.set_xticks(sorted(layers))

# 4. Componentes PHI
ax4 = axes[1, 1]
components = ['temporal', 'integration', 'complexity']
train_vals = [np.mean(train_components[c]) for c in components]
gen_vals = [np.mean(gen_components[c]) for c in components]
x = np.arange(len(components))
width = 0.35
ax4.bar(x - width/2, train_vals, width, label='Entrenamiento', alpha=0.8)
ax4.bar(x + width/2, gen_vals, width, label='Generación', alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(components)
ax4.set_ylabel('Valor')
ax4.set_title('Componentes PHI: Entrenamiento vs Generación')
ax4.legend()

plt.tight_layout()
plt.savefig('results/phi_analysis_visualization.png', dpi=150)
print("✅ Gráficos guardados en: results/phi_analysis_visualization.png")

plt.show()

print("\n" + "="*70)
print("🎯 ANÁLISIS DE CORRELACIÓN COMPLETADO")
print("="*70)
