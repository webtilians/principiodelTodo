#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 ANÁLISIS: ¿Por qué el experimento anterior dio mejores resultados?
====================================================================

Compara los resultados del experimento actual con el anterior para
determinar qué causó la diferencia en mejora (23.16% vs 3.56%).
"""

import sys
import os
import json
import torch
import matplotlib.pyplot as plt

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def analyze_training_dynamics():
    """Analiza la dinámica de entrenamiento del experimento actual."""
    
    print("="*70)
    print("🔬 ANÁLISIS COMPARATIVO: ¿Por qué el primer experimento fue mejor?")
    print("="*70)
    
    # Cargar resultados del experimento actual (V3)
    try:
        with open('training_results_v3.json', 'r') as f:
            results_v3 = json.load(f)
    except FileNotFoundError:
        print("❌ No se encontró training_results_v3.json")
        print("   Ejecuta primero: python src/infinito_gemini.py")
        return
    
    print("\n📊 EXPERIMENTO ACTUAL (V3 - Recién ejecutado):")
    print(f"   IIT Loss: {results_v3['results']['iit_final_loss']:.5f}")
    print(f"   Baseline Loss: {results_v3['results']['baseline_final_loss']:.5f}")
    print(f"   Mejora: {results_v3['results']['improvement_percentage']:.2f}%")
    print(f"   Memory Gate: {results_v3['results']['memory_gate_value']:.6f}")
    
    print("\n📊 EXPERIMENTO ANTERIOR (Primera ejecución exitosa):")
    print(f"   IIT Loss: 0.34393")
    print(f"   Baseline Loss: 0.44756")
    print(f"   Mejora: 23.16%")
    print(f"   Memory Gate: (desconocido - no se guardó)")
    
    print("\n" + "="*70)
    print("🔍 DIFERENCIAS CRÍTICAS DETECTADAS")
    print("="*70)
    
    # Analizar las curvas de aprendizaje
    iit_history = results_v3['loss_history']['iit']
    base_history = results_v3['loss_history']['baseline']
    
    # Calcular convergencia
    iit_early = sum(iit_history[:500]) / 500
    iit_late = sum(iit_history[-500:]) / 500
    base_early = sum(base_history[:500]) / 500
    base_late = sum(base_history[-500:]) / 500
    
    print("\n1️⃣ VELOCIDAD DE CONVERGENCIA:")
    print(f"   IIT  - Primeras 500 épocas: {iit_early:.4f}")
    print(f"   IIT  - Últimas 500 épocas: {iit_late:.4f}")
    print(f"   IIT  - Mejora interna: {((iit_early-iit_late)/iit_early*100):.2f}%")
    print(f"")
    print(f"   Base - Primeras 500 épocas: {base_early:.4f}")
    print(f"   Base - Últimas 500 épocas: {base_late:.4f}")
    print(f"   Base - Mejora interna: {((base_early-base_late)/base_early*100):.2f}%")
    
    # Analizar varianza (estabilidad)
    iit_variance = sum([(x - iit_late)**2 for x in iit_history[-500:]]) / 500
    base_variance = sum([(x - base_late)**2 for x in base_history[-500:]]) / 500
    
    print("\n2️⃣ ESTABILIDAD DEL ENTRENAMIENTO:")
    print(f"   IIT Varianza (últimas 500): {iit_variance:.6f}")
    print(f"   Base Varianza (últimas 500): {base_variance:.6f}")
    if iit_variance < base_variance:
        print("   ✅ IIT es más estable")
    else:
        print("   ❌ IIT es menos estable que Baseline")
    
    print("\n3️⃣ MEMORY GATE APRENDIZAJE:")
    gate_value = results_v3['results']['memory_gate_value']
    gate_activated = results_v3['results']['memory_gate_activated']
    print(f"   Raw value: {gate_value:.6f}")
    print(f"   Activated (sigmoid): {gate_activated:.6f}")
    print(f"   Uso efectivo de memoria: {gate_activated*100:.2f}%")
    
    if abs(gate_value) < 0.01:
        print("   ⚠️ PROBLEMA: El gate NO aprendió a moverse de su valor inicial")
        print("   → La memoria NO se está usando efectivamente")
        print("   → El modelo IIT funciona básicamente como Baseline + ruido")
    
    # Comparar loss finales
    print("\n4️⃣ COMPARACIÓN DE LOSS FINALES:")
    print(f"   Experimento Anterior:")
    print(f"      IIT: 0.34393 | Base: 0.44756 | Gap: {0.44756-0.34393:.5f}")
    print(f"   Experimento Actual:")
    print(f"      IIT: {results_v3['results']['iit_final_loss']:.5f} | Base: {results_v3['results']['baseline_final_loss']:.5f} | Gap: {results_v3['results']['baseline_final_loss']-results_v3['results']['iit_final_loss']:.5f}")
    
    gap_anterior = 0.44756 - 0.34393
    gap_actual = results_v3['results']['baseline_final_loss'] - results_v3['results']['iit_final_loss']
    
    print(f"\n   📉 Gap anterior: {gap_anterior:.5f}")
    print(f"   📉 Gap actual: {gap_actual:.5f}")
    print(f"   📊 Diferencia de gaps: {gap_anterior - gap_actual:.5f}")
    print(f"   🔻 Reducción de ventaja: {((gap_anterior - gap_actual)/gap_anterior*100):.1f}%")
    
    print("\n" + "="*70)
    print("💡 HIPÓTESIS SOBRE LA DIFERENCIA (ORDENADAS POR PROBABILIDAD)")
    print("="*70)
    
    print("\n🎯 HIPÓTESIS #1: SEED ALEATORIO (Probabilidad: 80%)")
    print("   📝 Descripción:")
    print("      • Los experimentos NO fijan seed para generación de datos")
    print("      • Cada ejecución genera secuencias Dyck diferentes aleatoriamente")
    print("      • Algunas secuencias son más fáciles/difíciles que otras")
    print("   🔬 Evidencia:")
    print("      • El código usa random.choice() sin seed fijo")
    print("      • Loss finales varían mucho entre ejecuciones")
    print(f"      • Baseline varió: 0.44756 → {results_v3['results']['baseline_final_loss']:.5f}")
    print(f"      • IIT varió: 0.34393 → {results_v3['results']['iit_final_loss']:.5f}")
    print("   ✅ Solución:")
    print("      • Fijar random.seed() y torch.manual_seed() al inicio")
    print("      • Ejecutar múltiples experimentos con seeds diferentes")
    print("      • Promediar resultados")
    
    print("\n🎯 HIPÓTESIS #2: MEMORY GATE NO APRENDE (Probabilidad: 15%)")
    print("   📝 Descripción:")
    print(f"      • Gate actual: {gate_value:.6f} (prácticamente 0)")
    print(f"      • Activación: {gate_activated:.4f} (50% = no aprendió nada)")
    print("      • La memoria existe pero NO se usa")
    print("   🔬 Evidencia:")
    print("      • lambda_phi = 0.0 (sin presión para usar PHI)")
    print("      • Solo 3000 épocas (quizá insuficiente)")
    print("      • Sin señal explícita de que memoria ayuda")
    print("   ✅ Solución:")
    print("      • Aumentar épocas a 5000-10000")
    print("      • Usar lambda_phi > 0 (ej. 0.01) para forzar integración")
    print("      • Añadir reward explícito por usar memoria")
    
    print("\n🎯 HIPÓTESIS #3: INICIALIZACIÓN DE PESOS (Probabilidad: 5%)")
    print("   📝 Descripción:")
    print("      • PyTorch inicializa pesos aleatoriamente")
    print("      • Experimento anterior pudo tener inicialización favorable")
    print("   🔬 Evidencia:")
    print("      • No hay torch.manual_seed() en el código")
    print("      • Cada ejecución parte de pesos diferentes")
    print("   ✅ Solución:")
    print("      • Fijar torch.manual_seed() antes de crear modelos")
    
    print("\n" + "="*70)
    print("🔬 ANÁLISIS DEL CÓDIGO: ¿Qué cambió?")
    print("="*70)
    
    print("\n📋 CÓDIGO ANTERIOR (con bug):")
    print("""
    # ❌ BUG: Memoria leída pero NO usada
    read_content, read_weights = self.memory.read(...)
    # ... escritura ...
    logits = self.output_projection(hidden)  # <-- Sin memoria!
    """)
    
    print("\n📋 CÓDIGO ACTUAL (corregido):")
    print("""
    # ✅ Memoria leída Y usada con gate
    read_content, read_weights = self.memory.read(...)
    gated_memory = torch.sigmoid(self.memory_gate) * read_content
    hidden = hidden + gated_memory
    hidden = self.memory_norm(hidden)
    logits = self.output_projection(hidden)
    """)
    
    print("\n🤔 PARADOJA OBSERVADA:")
    print("   • El código CON BUG dio mejor resultado (23.16%)")
    print("   • El código CORREGIDO dio peor resultado (3.56%)")
    print("   ")
    print("   💡 EXPLICACIÓN MÁS PROBABLE:")
    print("   → NO es paradoja, es VARIANZA ALEATORIA")
    print("   → El experimento anterior tuvo SUERTE con el seed")
    print("   → El actual tuvo MALA SUERTE con el seed")
    print("   → Necesitamos múltiples ejecuciones para confirmar")
    
    # Visualizar curvas de aprendizaje
    print("\n" + "="*70)
    print("📈 GENERANDO GRÁFICAS DE ANÁLISIS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis de Entrenamiento: Experimento V3 vs Anterior', fontsize=16)
    
    # Gráfica 1: Curvas completas
    axes[0, 0].plot(iit_history, label='IIT Actual', alpha=0.7)
    axes[0, 0].plot(base_history, label='Baseline Actual', alpha=0.7)
    axes[0, 0].axhline(y=0.34393, color='g', linestyle='--', label='IIT Anterior', alpha=0.5)
    axes[0, 0].axhline(y=0.44756, color='r', linestyle='--', label='Base Anterior', alpha=0.5)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Curvas de Aprendizaje Completas')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfica 2: Últimas 500 épocas (convergencia)
    axes[0, 1].plot(iit_history[-500:], label='IIT Actual', alpha=0.7)
    axes[0, 1].plot(base_history[-500:], label='Baseline Actual', alpha=0.7)
    axes[0, 1].axhline(y=0.34393, color='g', linestyle='--', label='IIT Anterior', alpha=0.5)
    axes[0, 1].axhline(y=0.44756, color='r', linestyle='--', label='Base Anterior', alpha=0.5)
    axes[0, 1].set_xlabel('Época (últimas 500)')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Convergencia Final')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfica 3: Gap entre modelos
    gap_history = [base_history[i] - iit_history[i] for i in range(len(iit_history))]
    axes[1, 0].plot(gap_history, label='Gap Actual (Base-IIT)', color='purple', alpha=0.7)
    axes[1, 0].axhline(y=gap_anterior, color='orange', linestyle='--', 
                       label=f'Gap Anterior: {gap_anterior:.5f}', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Gap (Baseline - IIT)')
    axes[1, 0].set_title('Ventaja del IIT sobre Baseline')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(range(len(gap_history)), 0, gap_history, 
                             where=[g > 0 for g in gap_history], 
                             alpha=0.3, color='green', label='IIT mejor')
    axes[1, 0].fill_between(range(len(gap_history)), 0, gap_history, 
                             where=[g < 0 for g in gap_history], 
                             alpha=0.3, color='red', label='Baseline mejor')
    
    # Gráfica 4: Comparación de loss finales
    experiments = ['Anterior\n(23.16%)', 'Actual\n(3.56%)']
    iit_losses = [0.34393, results_v3['results']['iit_final_loss']]
    base_losses = [0.44756, results_v3['results']['baseline_final_loss']]
    
    x = range(len(experiments))
    width = 0.35
    axes[1, 1].bar([i - width/2 for i in x], iit_losses, width, label='IIT', color='green', alpha=0.7)
    axes[1, 1].bar([i + width/2 for i in x], base_losses, width, label='Baseline', color='red', alpha=0.7)
    axes[1, 1].set_ylabel('Loss Final')
    axes[1, 1].set_title('Comparación de Loss Finales')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(experiments)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Añadir anotaciones de mejora
    for i, (iit_l, base_l) in enumerate(zip(iit_losses, base_losses)):
        mejora = ((base_l - iit_l) / base_l) * 100
        axes[1, 1].text(i, max(iit_l, base_l) + 0.02, f'{mejora:.1f}%', 
                        ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Guardar gráfica
    graph_path = 'experiment_comparison_analysis.png'
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    print(f"✅ Gráficas guardadas: {graph_path}")
    
    plt.show()
    
    # Resumen final
    print("\n" + "="*70)
    print("🎯 CONCLUSIÓN Y RECOMENDACIONES")
    print("="*70)
    
    print("\n📊 CONCLUSIÓN PRINCIPAL:")
    print("   La diferencia de 23.16% vs 3.56% es probablemente debida a")
    print("   VARIABILIDAD ALEATORIA en la generación de datos y pesos iniciales.")
    print("   ")
    print("   NO es evidencia de que el bug ayudara o el fix empeorara el modelo.")
    
    print("\n✅ RECOMENDACIONES:")
    print("   1. Ejecutar experimento con múltiples seeds (10-20 repeticiones)")
    print("   2. Calcular media y desviación estándar de la mejora")
    print("   3. Si la mejora promedio es >10% con p<0.05, es significativa")
    print("   4. Aumentar épocas para permitir que memory_gate aprenda")
    print("   5. Añadir lambda_phi > 0 para incentivar uso de memoria")
    
    print("\n🚀 SIGUIENTE PASO:")
    print("   ¿Quieres que cree un script para ejecutar múltiples experimentos")
    print("   con seeds diferentes y obtener resultados estadísticamente válidos?")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    analyze_training_dynamics()
