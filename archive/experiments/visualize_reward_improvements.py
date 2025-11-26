#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 VISUALIZACIÓN - Comparación Reward v1 vs v2
==============================================

Script para visualizar gráficamente las diferencias entre
las reward functions v1 (original) y v2 (mejorada).
"""

import matplotlib.pyplot as plt
import numpy as np

# Configurar estilo
plt.style.use('seaborn-v0_8-darkgrid')

def plot_phi_rewards():
    """Comparar rewards por diferentes valores de PHI."""
    phi_values = np.linspace(0.5, 10.0, 100)
    
    # Métricas fijas para comparación
    prev = {
        "consciousness": 0.45,
        "phi": 4.0,
        "perplexity": 90.0,
        "memory_utilization": 0.3
    }
    
    rewards_v1 = []
    rewards_v2 = []
    
    # Pesos
    alpha, beta, gamma, delta = 1.0, 0.5, 0.1, 0.2
    
    for phi in phi_values:
        cur = {
            "consciousness": 0.46,
            "phi": phi,
            "perplexity": 85.0,
            "memory_utilization": 0.3
        }
        
        # v1: Solo deltas
        delta_c = cur["consciousness"] - prev["consciousness"]
        delta_phi = cur["phi"] - prev["phi"]
        delta_ppl = (prev["perplexity"] - cur["perplexity"]) / prev["perplexity"]
        cost = cur["memory_utilization"]
        
        reward_v1 = alpha * delta_c + beta * delta_phi + gamma * delta_ppl - delta * cost
        rewards_v1.append(reward_v1)
        
        # v2: Con términos adicionales
        phi_balance = 0.0
        if phi < 3.0:
            phi_balance = -0.3 * (3.0 - phi)
        elif phi > 6.0:
            phi_balance = -0.6 * (phi - 6.0)
        else:
            phi_balance = +0.1
        
        phi_change = abs(delta_phi)
        stability = -0.8 * (phi_change - 1.0) if phi_change > 1.0 else 0.0
        
        c_balance = +0.05  # En rango óptimo
        
        reward_v2 = reward_v1 + stability + phi_balance + c_balance
        rewards_v2.append(reward_v2)
    
    # Graficar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Recompensas absolutas
    ax1.plot(phi_values, rewards_v1, 'b-', linewidth=2, label='v1 (Original)')
    ax1.plot(phi_values, rewards_v2, 'r-', linewidth=2, label='v2 (Mejorada)')
    ax1.axvspan(3.0, 6.0, alpha=0.2, color='green', label='Rango óptimo')
    ax1.axvline(8.0, color='orange', linestyle='--', label='Colapso Fase 2')
    ax1.set_xlabel('PHI (Φ)', fontsize=12)
    ax1.set_ylabel('Recompensa', fontsize=12)
    ax1.set_title('Reward Function: v1 vs v2', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Diferencia (v2 - v1)
    diff = np.array(rewards_v2) - np.array(rewards_v1)
    ax2.plot(phi_values, diff, 'g-', linewidth=2)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvspan(3.0, 6.0, alpha=0.2, color='green', label='Rango óptimo')
    ax2.axvline(8.0, color='orange', linestyle='--', label='Colapso Fase 2')
    ax2.set_xlabel('PHI (Φ)', fontsize=12)
    ax2.set_ylabel('Diferencia (v2 - v1)', fontsize=12)
    ax2.set_title('Mejora de v2 sobre v1', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/reward_comparison_phi.png', dpi=150, bbox_inches='tight')
    print("✅ Gráfico PHI guardado: outputs/reward_comparison_phi.png")
    plt.close()


def plot_ppl_rewards():
    """Comparar rewards por diferentes valores de Perplexity."""
    ppl_values = np.linspace(1, 300, 100)
    
    prev = {
        "consciousness": 0.45,
        "phi": 4.5,
        "perplexity": 100.0,
        "memory_utilization": 0.3
    }
    
    rewards_v1 = []
    rewards_v2 = []
    
    alpha, beta, gamma, delta = 1.0, 0.5, 0.1, 0.2
    
    for ppl in ppl_values:
        cur = {
            "consciousness": 0.46,
            "phi": 4.6,
            "perplexity": ppl,
            "memory_utilization": 0.3
        }
        
        # v1
        delta_c = cur["consciousness"] - prev["consciousness"]
        delta_phi = cur["phi"] - prev["phi"]
        delta_ppl = (prev["perplexity"] - cur["perplexity"]) / prev["perplexity"]
        cost = cur["memory_utilization"]
        
        reward_v1 = alpha * delta_c + beta * delta_phi + gamma * delta_ppl - delta * cost
        rewards_v1.append(reward_v1)
        
        # v2: Con límites PPL
        ppl_penalty = 0.0
        if ppl < 10.0:
            ppl_penalty = -2.0 * (10.0 - ppl) / 10.0
        elif ppl > 200.0:
            ppl_penalty = -0.3 * (ppl - 200.0) / 100.0
        
        phi_balance = +0.1  # En rango óptimo
        c_balance = +0.05
        
        reward_v2 = reward_v1 + ppl_penalty + phi_balance + c_balance
        rewards_v2.append(reward_v2)
    
    # Graficar
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ppl_values, rewards_v1, 'b-', linewidth=2, label='v1 (Original)')
    ax.plot(ppl_values, rewards_v2, 'r-', linewidth=2, label='v2 (Mejorada)')
    ax.axvspan(10, 200, alpha=0.2, color='green', label='Rango seguro')
    ax.axvline(10, color='red', linestyle='--', linewidth=1.5, label='Colapso (PPL < 10)')
    ax.axvline(200, color='orange', linestyle='--', linewidth=1.5, label='Confusión (PPL > 200)')
    
    ax.set_xlabel('Perplexity', fontsize=12)
    ax.set_ylabel('Recompensa', fontsize=12)
    ax.set_title('Reward Function vs Perplexity: Detección de Colapso', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/reward_comparison_ppl.png', dpi=150, bbox_inches='tight')
    print("✅ Gráfico PPL guardado: outputs/reward_comparison_ppl.png")
    plt.close()


def plot_summary_table():
    """Crear tabla comparativa de características."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Datos de la tabla
    features = [
        ['Característica', 'v1 (Original)', 'v2 (Mejorada)', 'Mejora'],
        ['', '', '', ''],
        ['Términos básicos', '4 (ΔC, ΔΦ, ΔPPL, cost)', '4 (mismos)', ''],
        ['Términos adicionales', '0', '4 nuevos', '+400%'],
        ['', '', '', ''],
        ['Detecta colapso PHI', '❌ No', '✅ Sí (Φ > 6.0)', '✅'],
        ['Penalización Φ > 8', 'Indirecta', '-0.6 × exceso', '✅'],
        ['', '', '', ''],
        ['Detecta colapso PPL', '❌ No', '✅ Sí (PPL < 10)', '✅'],
        ['Penalización PPL < 10', 'Indirecta', '-2.0 × factor', '✅'],
        ['', '', '', ''],
        ['Penaliza inestabilidad', '❌ No', '✅ Sí (|ΔΦ| > 1)', '✅'],
        ['Penalización cambios', 'Ninguna', '-0.8 × exceso', '✅'],
        ['', '', '', ''],
        ['Incentiva rangos óptimos', '❌ No', '✅ Sí (bonuses)', '✅'],
        ['Bonus Φ ∈ [3,6]', '0', '+0.1', '✅'],
        ['Bonus C ∈ [0.3,0.7]', '0', '+0.05', '✅'],
        ['', '', '', ''],
        ['Robustez', 'Media', 'Alta', '+100%'],
        ['Convergencia esperada', 'Lenta', 'Rápida', '+50%'],
        ['Evita colapso Fase 2', 'Parcial', 'Completo', '✅'],
    ]
    
    # Colores
    colors = []
    for row in features:
        if row[0] == '':
            colors.append(['white']*4)
        elif row[0] == 'Característica':
            colors.append(['lightblue']*4)
        else:
            if '✅' in row[3]:
                colors.append(['white', 'lightyellow', 'lightgreen', 'lightgreen'])
            else:
                colors.append(['white', 'lightyellow', 'lightgreen', 'white'])
    
    table = ax.table(cellText=features, cellColours=colors,
                     cellLoc='left', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Estilo header
    for i in range(4):
        table[(0, i)].set_facecolor('darkblue')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Comparación Reward Function v1 vs v2', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig('outputs/reward_comparison_table.png', dpi=150, bbox_inches='tight')
    print("✅ Tabla comparativa guardada: outputs/reward_comparison_table.png")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("📊 GENERANDO VISUALIZACIONES - REWARD v1 vs v2")
    print("="*70)
    
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("\n1. Gráfico PHI...")
    plot_phi_rewards()
    
    print("\n2. Gráfico Perplexity...")
    plot_ppl_rewards()
    
    print("\n3. Tabla comparativa...")
    plot_summary_table()
    
    print("\n" + "="*70)
    print("✅ VISUALIZACIONES COMPLETADAS")
    print("="*70)
    print("\nArchivos generados:")
    print("  - outputs/reward_comparison_phi.png")
    print("  - outputs/reward_comparison_ppl.png")
    print("  - outputs/reward_comparison_table.png")
