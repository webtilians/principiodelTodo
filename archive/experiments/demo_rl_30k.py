#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎬 DEMO - Generación con Modelo RL 30K
======================================

Script de demostración que muestra las capacidades del modelo RL 30K
con múltiples ejemplos de generación.

Muestra:
- Diferentes tipos de prompts (científico, creativo, filosófico)
- Control adaptativo en acción
- Comparación de métricas
- Análisis de estrategias del agente
"""

from generate_with_rl_30k import RLTextGenerator
from datetime import datetime
import json

def print_header(title):
    """Imprimir encabezado bonito."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_section(title):
    """Imprimir sección."""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")

def analyze_strategy(result):
    """Analizar la estrategia del agente."""
    stats = result['stats']
    dist = stats['actions_distribution']
    
    text_pct = dist['TEXT']['percentage']
    phi_pct = dist['PHI']['percentage']
    mixed_pct = dist['MIXED']['percentage']
    
    print("\n🎯 Análisis de estrategia:")
    
    # Dominancia
    if max(text_pct, phi_pct, mixed_pct) == text_pct:
        print(f"  Estrategia dominante: TEXT ({text_pct:.1f}%)")
        print("  → Prioriza calidad del lenguaje")
    elif max(text_pct, phi_pct, mixed_pct) == phi_pct:
        print(f"  Estrategia dominante: PHI ({phi_pct:.1f}%)")
        print("  → Prioriza integración de información")
    else:
        print(f"  Estrategia dominante: MIXED ({mixed_pct:.1f}%)")
        print("  → Balance adaptativo equilibrado")
    
    # Balance
    balance = abs(text_pct - phi_pct)
    if balance < 10:
        print(f"  Balance TEXT/PHI: Muy equilibrado ({balance:.1f}% diff)")
    elif balance < 20:
        print(f"  Balance TEXT/PHI: Equilibrado ({balance:.1f}% diff)")
    else:
        print(f"  Balance TEXT/PHI: Desbalanceado ({balance:.1f}% diff)")
    
    # Exploración
    if mixed_pct > 25:
        print(f"  Exploración: Alta ({mixed_pct:.1f}% MIXED)")
    elif mixed_pct > 15:
        print(f"  Exploración: Media ({mixed_pct:.1f}% MIXED)")
    else:
        print(f"  Exploración: Baja ({mixed_pct:.1f}% MIXED)")
    
    # Métricas
    phi_ok = stats['phi_in_optimal_range_pct'] > 70
    ppl_ok = stats['perplexity_safe_pct'] > 90
    
    print(f"\n  Estado del sistema:")
    print(f"    PHI óptimo: {stats['phi_in_optimal_range_pct']:5.1f}% {'✅' if phi_ok else '⚠️'}")
    print(f"    PPL seguro: {stats['perplexity_safe_pct']:5.1f}% {'✅' if ppl_ok else '⚠️'}")
    
    if phi_ok and ppl_ok:
        print(f"    Evaluación: ✅ EXCELENTE - Sistema estable y óptimo")
    elif phi_ok or ppl_ok:
        print(f"    Evaluación: ⚠️ ACEPTABLE - Revisión recomendada")
    else:
        print(f"    Evaluación: ❌ PROBLEMAS - Requiere ajustes")

def demo_single_generation(generator, prompt, max_length=200, description=""):
    """Demo de una generación individual."""
    print_section(f"Demo: {description}")
    print(f"\nPrompt: '{prompt}'")
    print(f"Max length: {max_length} tokens")
    print()
    
    result = generator.generate(
        prompt=prompt,
        max_length=max_length,
        verbose=False,
        return_metrics=True
    )
    
    # Mostrar texto generado
    print("\n📄 Texto generado:")
    print("─" * 70)
    print(result['text'])
    print("─" * 70)
    
    # Estadísticas compactas
    stats = result['stats']
    print(f"\n📊 Estadísticas rápidas:")
    print(f"  Tokens: {stats['tokens_generated']}")
    print(f"  Tiempo: {stats['duration_seconds']:.2f}s")
    print(f"  PHI medio: {stats['phi_mean']:.3f} (en [3-6]: {stats['phi_in_optimal_range_pct']:.0f}%)")
    print(f"  Reward total: {stats['total_reward']:+.3f}")
    
    # Distribución visual
    dist = stats['actions_distribution']
    print(f"\n  Distribución:")
    for action in ['TEXT', 'PHI', 'MIXED']:
        pct = dist[action]['percentage']
        bar = '█' * int(pct / 3)
        print(f"    {action:5s}: {pct:5.1f}% {bar}")
    
    # Análisis
    analyze_strategy(result)
    
    return result

def demo_comparative(generator):
    """Demo comparativo con múltiples prompts."""
    print_header("DEMO COMPARATIVO - Diferentes Tipos de Texto")
    
    test_cases = [
        {
            'prompt': "The nature of consciousness",
            'description': "Filosófico/Abstracto",
            'max_length': 150
        },
        {
            'prompt': "Machine learning algorithms can",
            'description': "Técnico/Científico",
            'max_length': 150
        },
        {
            'prompt': "Once upon a time in a distant galaxy",
            'description': "Narrativo/Creativo",
            'max_length': 150
        }
    ]
    
    results = []
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n\n{'='*70}")
        print(f"  TEST {i}/3: {case['description']}")
        print(f"{'='*70}")
        
        result = demo_single_generation(
            generator,
            prompt=case['prompt'],
            max_length=case['max_length'],
            description=case['description']
        )
        
        results.append({
            'case': case,
            'result': result
        })
    
    # Comparación final
    print("\n\n" + "="*70)
    print("  COMPARACIÓN FINAL")
    print("="*70)
    
    print("\n📊 Resumen comparativo:")
    print()
    print(f"{'Tipo':<20} {'TEXT%':>7} {'PHI%':>7} {'MIXED%':>7} {'Φ medio':>8} {'Reward':>9}")
    print("─" * 70)
    
    for item in results:
        desc = item['case']['description']
        stats = item['result']['stats']
        dist = stats['actions_distribution']
        
        print(f"{desc:<20} "
              f"{dist['TEXT']['percentage']:>6.1f}% "
              f"{dist['PHI']['percentage']:>6.1f}% "
              f"{dist['MIXED']['percentage']:>6.1f}% "
              f"{stats['phi_mean']:>8.3f} "
              f"{stats['total_reward']:>+9.3f}")
    
    # Análisis de patrones
    print("\n💡 Observaciones:")
    
    avg_text = sum(r['result']['stats']['actions_distribution']['TEXT']['percentage'] for r in results) / len(results)
    avg_phi = sum(r['result']['stats']['actions_distribution']['PHI']['percentage'] for r in results) / len(results)
    avg_mixed = sum(r['result']['stats']['actions_distribution']['MIXED']['percentage'] for r in results) / len(results)
    
    print(f"\n  Promedio global:")
    print(f"    TEXT:  {avg_text:5.1f}%")
    print(f"    PHI:   {avg_phi:5.1f}%")
    print(f"    MIXED: {avg_mixed:5.1f}%")
    
    if avg_mixed > 20:
        print(f"\n  ✅ El agente usa bien el modo MIXED (adaptativo)")
    else:
        print(f"\n  ⚠️ El agente usa poco MIXED (menos de 20%)")
    
    # PHI
    phi_values = [r['result']['stats']['phi_mean'] for r in results]
    phi_avg = sum(phi_values) / len(phi_values)
    phi_std = (sum((x - phi_avg)**2 for x in phi_values) / len(phi_values)) ** 0.5
    
    print(f"\n  PHI medio global: {phi_avg:.3f} ± {phi_std:.3f}")
    if 3.0 <= phi_avg <= 6.0:
        print(f"  ✅ PHI en rango óptimo [3.0-6.0]")
    else:
        print(f"  ⚠️ PHI fuera de rango óptimo")

def main():
    """Función principal del demo."""
    print("="*70)
    print("  🎬 DEMO - MODELO RL 30K ÓPTIMO")
    print("="*70)
    print(f"  Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Modelo: PPO 30K steps (óptimo)")
    print("="*70)
    
    try:
        # Crear generador
        print("\n📦 Inicializando generador...")
        generator = RLTextGenerator()
        
        # Cargar modelo
        print("⏳ Cargando modelo RL 30K...")
        generator.load()
        print("✅ Modelo listo")
        
        # Demo 1: Generación simple
        print_header("DEMO 1 - Generación Simple")
        demo_single_generation(
            generator,
            prompt="The future of artificial intelligence",
            max_length=200,
            description="Demo básico"
        )
        
        # Demo 2: Comparativo
        print("\n\n")
        demo_comparative(generator)
        
        # Demo 3: Diferentes configuraciones
        print("\n\n")
        print_header("DEMO 3 - Diferentes Temperaturas")
        
        temps = [0.5, 0.8, 1.2]
        prompt = "Consciousness emerges when"
        
        print(f"\nPrompt fijo: '{prompt}'")
        print(f"Temperaturas: {temps}")
        
        for temp in temps:
            print(f"\n\n{'─'*70}")
            print(f"  Temperature = {temp}")
            print(f"{'─'*70}")
            
            result = generator.generate(
                prompt=prompt,
                max_length=100,
                temperature=temp,
                verbose=False
            )
            
            print(f"\nTexto generado:")
            print(result['text'])
            print(f"\nReward total: {result['stats']['total_reward']:+.3f}")
            print(f"PHI medio: {result['stats']['phi_mean']:.3f}")
        
        # Resumen final
        print("\n\n" + "="*70)
        print("  ✅ DEMO COMPLETADO")
        print("="*70)
        
        print("\n💡 Conclusiones:")
        print("  1. El modelo RL 30K ajusta automáticamente TEXT/PHI/MIXED")
        print("  2. Mantiene PHI en rango óptimo [3.0-6.0] >90% del tiempo")
        print("  3. Evita colapsos (PPL < 10) y confusión (PPL > 200)")
        print("  4. Adapta estrategia según el tipo de texto")
        print("  5. Mayor temperatura = más exploración pero más variabilidad")
        
        print("\n📚 Para más información:")
        print("  - README_PRODUCCION_RL.md (guía de uso)")
        print("  - MODELO_30K_GUIA.md (detalles técnicos)")
        print("  - RESUMEN_EJECUTIVO_RL_V2.md (resultados)")
        
        # Cerrar
        print("\n🧹 Liberando recursos...")
        generator.close()
        
        print("\n✅ Demo finalizado exitosamente")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
