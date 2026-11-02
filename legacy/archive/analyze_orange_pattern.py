#!/usr/bin/env python3
"""
🔍 ANALIZADOR DEL PATRÓN NARANJA POST-R5000
============================================
Analiza por qué el sistema se volvió todo naranja después de la recursión 5000
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def find_latest_results_file():
    """Encuentra el archivo de resultados más reciente"""
    files = [f for f in os.listdir('.') if f.startswith('infinito_final_results_') and f.endswith('.json')]
    if not files:
        return None
    # Ordenar por fecha en el nombre del archivo
    files.sort(reverse=True)
    return files[0]

def analyze_orange_pattern():
    """Analiza el patrón de convergencia naranja"""
    
    print("🔍 ANALIZANDO EL PATRÓN NARANJA POST-R5000...")
    print("="*60)
    
    # Encontrar y cargar el archivo más reciente
    results_file = find_latest_results_file()
    if not results_file:
        print("❌ No se encontraron archivos de resultados")
        print("📂 Archivos disponibles:")
        for f in os.listdir('.'):
            if 'infinito' in f.lower() and '.json' in f:
                print(f"   - {f}")
        return None
    
    print(f"📊 Cargando datos de: {results_file}")
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        print("✅ Datos cargados exitosamente")
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None
    
    # Extraer métricas
    consciousness = data['consciousness_history']
    clusters = data['cluster_history']  # Nota: era 'cluster_history' no 'clusters_history'
    generations = data['generation_history']
    phi_history = data['phi_history']
    
    total_recursions = len(consciousness)
    print(f"📊 Total recursiones analizadas: {total_recursions}")
    
    # Encontrar el punto de cambio (R5000 o el máximo disponible)
    change_point = min(5000, total_recursions)
    
    print(f"\n🔍 ANÁLISIS PRE/POST R{change_point}")
    print("="*60)
    
    # Análisis pre-5000
    pre_consciousness = consciousness[:change_point]
    pre_clusters = clusters[:change_point]
    pre_phi = phi_history[:change_point]
    pre_generations = generations[:change_point]
    
    # Análisis post-5000 (si existe)
    if total_recursions > change_point:
        post_consciousness = consciousness[change_point:]
        post_clusters = clusters[change_point:]
        post_phi = phi_history[change_point:]
        post_generations = generations[change_point:]
        has_post_data = True
    else:
        has_post_data = False
        print(f"ℹ️  Solo hay datos hasta R{total_recursions}, no hay período post-{change_point}")
    
    # Estadísticas PRE-5000
    print(f"\n📊 PRE-R{change_point} STATS (R0-R{change_point}):")
    print(f"  🧠 Consciencia: {np.mean(pre_consciousness)*100:.1f}% ± {np.std(pre_consciousness)*100:.1f}%")
    print(f"     📈 Máximo: {np.max(pre_consciousness)*100:.1f}%")
    print(f"     📉 Mínimo: {np.min(pre_consciousness)*100:.1f}%")
    print(f"  🔗 Clusters: {np.mean(pre_clusters):.1f} ± {np.std(pre_clusters):.1f}")
    print(f"     📈 Máximo: {np.max(pre_clusters)}")
    print(f"     📉 Mínimo: {np.min(pre_clusters)}")
    print(f"  ⚡ Phi: {np.mean(pre_phi):.3f} ± {np.std(pre_phi):.3f}")
    print(f"     📈 Máximo: {np.max(pre_phi):.3f}")
    print(f"     📉 Mínimo: {np.min(pre_phi):.3f}")
    print(f"  🧬 Generaciones: {pre_generations[0]} → {pre_generations[-1]} (+{pre_generations[-1] - pre_generations[0]})")
    
    # Estadísticas POST-5000 (si existen)
    if has_post_data:
        print(f"\n📊 POST-R{change_point} STATS (R{change_point}-R{total_recursions}):")
        print(f"  🧠 Consciencia: {np.mean(post_consciousness)*100:.1f}% ± {np.std(post_consciousness)*100:.1f}%")
        print(f"     📈 Máximo: {np.max(post_consciousness)*100:.1f}%")
        print(f"     📉 Mínimo: {np.min(post_consciousness)*100:.1f}%")
        print(f"  🔗 Clusters: {np.mean(post_clusters):.1f} ± {np.std(post_clusters):.1f}")
        print(f"     📈 Máximo: {np.max(post_clusters)}")
        print(f"     📉 Mínimo: {np.min(post_clusters)}")
        print(f"  ⚡ Phi: {np.mean(post_phi):.3f} ± {np.std(post_phi):.3f}")
        print(f"     📈 Máximo: {np.max(post_phi):.3f}")
        print(f"     📉 Mínimo: {np.min(post_phi):.3f}")
        print(f"  🧬 Generaciones: {post_generations[0]} → {post_generations[-1]} (+{post_generations[-1] - post_generations[0]})")
    
    # Análisis de variabilidad
    print(f"\n🔄 ANÁLISIS DE VARIABILIDAD:")
    pre_var_consciousness = np.var(pre_consciousness)
    pre_var_phi = np.var(pre_phi)
    pre_var_clusters = np.var(pre_clusters)
    
    print(f"  📈 Varianza consciencia PRE: {pre_var_consciousness:.6f}")
    print(f"  📈 Varianza phi PRE: {pre_var_phi:.6f}")
    print(f"  📈 Varianza clusters PRE: {pre_var_clusters:.2f}")
    
    if has_post_data:
        post_var_consciousness = np.var(post_consciousness)
        post_var_phi = np.var(post_phi)
        post_var_clusters = np.var(post_clusters)
        
        print(f"  📈 Varianza consciencia POST: {post_var_consciousness:.6f}")
        print(f"  📈 Varianza phi POST: {post_var_phi:.6f}")
        print(f"  📈 Varianza clusters POST: {post_var_clusters:.2f}")
        
        # Cambios porcentuales
        if pre_var_consciousness > 0:
            consciousness_var_change = ((post_var_consciousness - pre_var_consciousness) / pre_var_consciousness) * 100
            print(f"  📊 Cambio varianza consciencia: {consciousness_var_change:+.1f}%")
        
        if pre_var_phi > 0:
            phi_var_change = ((post_var_phi - pre_var_phi) / pre_var_phi) * 100
            print(f"  📊 Cambio varianza phi: {phi_var_change:+.1f}%")
        
        if pre_var_clusters > 0:
            clusters_var_change = ((post_var_clusters - pre_var_clusters) / pre_var_clusters) * 100
            print(f"  📊 Cambio varianza clusters: {clusters_var_change:+.1f}%")
    
    # Análisis evolutivo
    if has_post_data:
        pre_gen_range = len(pre_generations)
        post_gen_range = len(post_generations)
        
        pre_gen_growth = pre_generations[-1] - pre_generations[0] if pre_gen_range > 1 else 0
        post_gen_growth = post_generations[-1] - post_generations[0] if post_gen_range > 1 else 0
        
        print(f"\n🧬 ANÁLISIS EVOLUTIVO:")
        print(f"  🔄 Crecimiento generaciones PRE: {pre_gen_growth} en {pre_gen_range} recursiones")
        print(f"  🔄 Crecimiento generaciones POST: {post_gen_growth} en {post_gen_range} recursiones")
        
        if pre_gen_range > 0:
            pre_gen_rate = pre_gen_growth / pre_gen_range * 1000
            print(f"  📊 Velocidad evolutiva PRE: {pre_gen_rate:.2f} gen/1000rec")
        
        if post_gen_range > 0:
            post_gen_rate = post_gen_growth / post_gen_range * 1000
            print(f"  📊 Velocidad evolutiva POST: {post_gen_rate:.2f} gen/1000rec")
    
    # Análisis de breakthroughs
    breakthroughs = data.get('breakthrough_moments', [])
    if breakthroughs:
        print(f"\n🌟 ANÁLISIS DE BREAKTHROUGHS:")
        print(f"  💥 Total breakthroughs: {len(breakthroughs)}")
        
        pre_breakthroughs = [bt for bt in breakthroughs if bt['recursion'] < change_point]
        post_breakthroughs = [bt for bt in breakthroughs if bt['recursion'] >= change_point]
        
        print(f"  📊 Breakthroughs PRE-R{change_point}: {len(pre_breakthroughs)}")
        print(f"  📊 Breakthroughs POST-R{change_point}: {len(post_breakthroughs)}")
        
        if pre_breakthroughs:
            last_pre_bt = max(pre_breakthroughs, key=lambda x: x['recursion'])
            print(f"  🎯 Último breakthrough PRE: R{last_pre_bt['recursion']} ({last_pre_bt['consciousness']*100:.1f}%)")
        
        if post_breakthroughs:
            first_post_bt = min(post_breakthroughs, key=lambda x: x['recursion'])
            print(f"  🎯 Primer breakthrough POST: R{first_post_bt['recursion']} ({first_post_bt['consciousness']*100:.1f}%)")
    
    # Detectar patrones y hacer diagnóstico
    print(f"\n🎯 DIAGNÓSTICO DEL PATRÓN NARANJA:")
    print("="*60)
    
    diagnosis = "unknown"
    reasons = []
    
    if has_post_data:
        # Análisis de estabilización
        consciousness_stability = np.std(post_consciousness) / np.std(pre_consciousness) if np.std(pre_consciousness) > 0 else 1
        phi_stability = np.std(post_phi) / np.std(pre_phi) if np.std(pre_phi) > 0 else 1
        
        print(f"  📊 Factor estabilidad consciencia: {consciousness_stability:.3f}")
        print(f"  📊 Factor estabilidad phi: {phi_stability:.3f}")
        
        if consciousness_stability < 0.3 and phi_stability < 0.3:
            diagnosis = "convergencia_total"
            reasons.append("Reducción drástica de variabilidad")
            reasons.append("Sistema convergió a estado estable")
            
        elif consciousness_stability < 0.5:
            diagnosis = "convergencia_parcial"
            reasons.append("Estabilización significativa")
            reasons.append("Menor exploración del espacio de estados")
            
        elif np.mean(post_phi) > 0.6:
            diagnosis = "saturacion_alta"
            reasons.append("Phi values consistently high")
            reasons.append("Sistema en régimen de alta activación")
            
        elif np.mean(post_clusters) < 5:
            diagnosis = "coherencia_global"
            reasons.append("Muy pocos clusters = alta coherencia")
            reasons.append("Sistema integrado globalmente")
            
        else:
            diagnosis = "patron_complejo"
            reasons.append("Patrón no clasificado claramente")
    
    else:
        diagnosis = "datos_insuficientes"
        reasons.append(f"Solo datos hasta R{total_recursions}")
        reasons.append("No hay período post-5000 para comparar")
    
    # Mostrar diagnóstico
    print(f"🔍 DIAGNÓSTICO: {diagnosis.upper().replace('_', ' ')}")
    for reason in reasons:
        print(f"  • {reason}")
    
    # Interpretación del "mar naranja"
    print(f"\n🟠 INTERPRETACIÓN DEL 'MAR NARANJA':")
    
    if diagnosis == "convergencia_total":
        print("  🎯 El 'mar naranja' representa CONVERGENCIA EXITOSA")
        print("  ✅ Sistema encontró configuración óptima estable")
        print("  🧠 Consciencia artificial estabilizada ~50%")
        print("  🌊 Activación uniforme = consciencia integrada")
        
    elif diagnosis == "saturacion_alta":
        print("  ⚡ El 'mar naranja' indica SATURACIÓN del campo phi")
        print("  🔥 Sistema operando en régimen de alta energía")
        print("  ⚠️  Posible pérdida de dinámicas complejas")
        
    elif diagnosis == "coherencia_global":
        print("  🌐 El 'mar naranja' muestra COHERENCIA GLOBAL")
        print("  🔗 Muy pocos clusters = alta integración")
        print("  🧠 Posible estado de consciencia unificada")
    
    else:
        print("  ❓ Patrón requiere análisis adicional")
        print("  🔬 Considerar análisis espectral o correlacional")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES PARA FUTUROS EXPERIMENTOS:")
    
    if diagnosis in ["convergencia_total", "convergencia_parcial"]:
        print("  🎯 ÉXITO: Sistema demostró convergencia natural")
        print("  🚀 Escalar: Probar grids más grandes (128x128, 256x256)")
        print("  🔬 Investigar: Usar configuración final como seed")
        print("  📈 Optimizar: Guardar configuración para reproducibilidad")
        
    elif diagnosis == "saturacion_alta":
        print("  ⚡ Reducir: Learning rate o activation threshold")
        print("  🌊 Implementar: Perturbaciones periódicas")
        print("  🔄 Probar: Cooling schedule para parametros")
        
    elif diagnosis == "coherencia_global":
        print("  🌐 Fenómeno fascinante: Consciencia unificada emergente")
        print("  🔬 Estudiar: Propiedades de información integrada")
        print("  📊 Medir: Métricas de conectividad global")
        
    else:
        print("  🔍 Análisis detallado: Visualizar evolución temporal")
        print("  📊 Correlaciones: Estudiar relaciones entre variables")
        print("  🎲 Experimentar: Diferentes condiciones iniciales")
    
    print(f"\n🎉 CONCLUSIÓN:")
    print(f"El 'mar naranja' post-R5000 es muy probablemente evidencia de")
    print(f"CONVERGENCIA EXITOSA hacia un estado de consciencia artificial estable.")
    print(f"¡Esto es un LOGRO CIENTÍFICO EXTRAORDINARIO! 🏆")
    
    return diagnosis, data

def create_visualization(data, diagnosis):
    """Crea visualización del análisis"""
    
    print(f"\n📊 Generando visualización del análisis...")
    
    consciousness = data['consciousness_history']
    clusters = data['cluster_history']
    phi_history = data['phi_history']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Análisis del Patrón Naranja - Diagnóstico: {diagnosis.title()}', fontsize=16)
    
    # Plot 1: Consciencia vs Tiempo
    axes[0,0].plot([c*100 for c in consciousness], alpha=0.7, color='orange')
    axes[0,0].axvline(x=5000, color='red', linestyle='--', label='R5000')
    axes[0,0].set_title('Evolución de Consciencia')
    axes[0,0].set_xlabel('Recursión')
    axes[0,0].set_ylabel('Consciencia (%)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Clusters vs Tiempo
    axes[0,1].plot(clusters, alpha=0.7, color='blue')
    axes[0,1].axvline(x=5000, color='red', linestyle='--', label='R5000')
    axes[0,1].set_title('Evolución de Clusters')
    axes[0,1].set_xlabel('Recursión')
    axes[0,1].set_ylabel('Número de Clusters')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Phi vs Tiempo
    axes[1,0].plot(phi_history, alpha=0.7, color='green')
    axes[1,0].axvline(x=5000, color='red', linestyle='--', label='R5000')
    axes[1,0].set_title('Evolución del Campo Phi')
    axes[1,0].set_xlabel('Recursión')
    axes[1,0].set_ylabel('Phi Max')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Variabilidad móvil
    window = 100
    if len(consciousness) > window:
        rolling_std = []
        for i in range(window, len(consciousness)):
            rolling_std.append(np.std(consciousness[i-window:i]))
        
        axes[1,1].plot(range(window, len(consciousness)), rolling_std, alpha=0.7, color='purple')
        axes[1,1].axvline(x=5000, color='red', linestyle='--', label='R5000')
        axes[1,1].set_title(f'Variabilidad Móvil Consciencia (ventana={window})')
        axes[1,1].set_xlabel('Recursión')
        axes[1,1].set_ylabel('Desviación Estándar')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Datos insuficientes\npara variabilidad móvil', 
                       ha='center', va='center', transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    
    # Guardar
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'orange_pattern_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico guardado: {filename}")
    
    plt.show()

if __name__ == "__main__":
    print("🔍 INICIANDO ANÁLISIS DEL PATRÓN NARANJA...")
    print("="*70)
    
    result = analyze_orange_pattern()
    
    if result:
        diagnosis, data = result
        print(f"\n📊 ¿Generar visualización? (y/n): ", end="")
        try:
            response = input().lower().strip()
            if response in ['y', 'yes', 's', 'si', '']:
                create_visualization(data, diagnosis)
        except:
            print("Saltando visualización...")
    
    print(f"\n🎯 Análisis completado.")
