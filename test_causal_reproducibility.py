"""
Test de Reproducibilidad del Lenguaje Causal
============================================

Objetivo: Verificar si el sistema genera arquitecturas causales reproducibles
Método: Ejecutar el mismo texto 10 veces con seeds diferentes
Texto de prueba: "mi perro es rojo"
Resultado esperado: Varianza < 0.05 → Lenguaje causal es determinista
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from collections import Counter
import json
from datetime import datetime

def extract_causal_architecture(input_text, seed, iterations=50):
    """
    Extrae la arquitectura causal de un texto con un seed específico
    """
    from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough
    from argparse import Namespace
    
    # Configurar seed para reproducibilidad
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Configuración del modelo usando Namespace
    args = Namespace(
        input_text=input_text,
        text_mode=True,
        quantum_active=False,
        input_dim=257,
        batch_size=4,
        hidden_dim=512,
        attention_heads=8,
        num_modules=8,
        integration_steps=3,
        phi_threshold=0.3,
        consciousness_threshold=0.7,
        dropout=0.1,
        max_iterations=iterations,
        enable_metacognition=True
    )
    
    # Crear runner
    runner = InfinitoV51ConsciousnessBreakthrough(args)
    device = runner.device
    model = runner.model
    model.eval()
    
    # Almacenar métricas
    phi_values = []
    coherence_values = []
    metacognitive_states = []
    surprise_values = []
    
    print(f"  [Seed {seed}] Ejecutando {iterations} iteraciones...")
    
    with torch.no_grad():
        for i in range(iterations):
            try:
                # Generar entrada basada en texto usando el semantic embedder
                if runner.semantic_embedder:
                    semantic_vec = runner.semantic_embedder.text_to_tensor(input_text, device)
                    # Ajustar dimensión: semantic_embedder da 256, pero necesitamos input_dim (257)
                    if semantic_vec.shape[0] < runner.input_dim:
                        # Padding para llegar a input_dim
                        padding = torch.zeros(runner.input_dim - semantic_vec.shape[0], device=device)
                        input_tensor = torch.cat([semantic_vec, padding])
                    else:
                        input_tensor = semantic_vec[:runner.input_dim]
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension [1, 257]
                else:
                    # Fallback: random input
                    input_tensor = torch.randn(1, runner.input_dim, device=device)
                
                # Forward pass
                output = model(input_tensor)
                
                # Calcular Φ usando el método del modelo
                phi = model.calculate_phi(input_tensor)
                
                # Calcular coherencia
                coherence = model.calculate_coherence(input_tensor)
                
                # Obtener estado metacognitivo (si está disponible)
                if hasattr(model, 'metacognitive_monitor') and model.metacognitive_monitor is not None:
                    meta_state = model.metacognitive_monitor(phi.unsqueeze(0), coherence.unsqueeze(0))
                    state_idx = torch.argmax(meta_state).item()
                else:
                    # Determinar estado basado en Φ y coherencia
                    if phi < 0.2:
                        state_idx = 0  # low_integration
                    elif phi < 0.4:
                        state_idx = 4  # diffuse_attention
                    elif phi > 0.6:
                        state_idx = 1  # high_integration
                    else:
                        state_idx = 8  # phi_stable
                
                state_names = [
                    'low_integration', 'high_integration', 'pre_conscious',
                    'conscious_access', 'diffuse_attention', 'focused_attention',
                    'phi_increasing', 'phi_decreasing', 'phi_stable', 'transitional'
                ]
                dominant_state = state_names[state_idx]
                
                # Calcular surprise (diferencia entre output y input)
                surprise = torch.abs(output - input_tensor).mean().item()
                
                # Almacenar métricas
                phi_values.append(phi.item())
                coherence_values.append(coherence.item())
                metacognitive_states.append(dominant_state)
                surprise_values.append(surprise)
                
                if (i + 1) % 10 == 0:
                    print(f"    Iteración {i+1}/{iterations} - Φ: {phi.item():.3f}")
                    
            except Exception as e:
                print(f"    Error en iteración {i+1}: {e}")
                continue
    
    # Calcular estadísticas
    if not phi_values:
        print(f"  ❌ No se pudieron obtener valores para seed {seed}")
        return None
    
    state_counter = Counter(metacognitive_states)
    dominant_state = state_counter.most_common(1)[0][0] if metacognitive_states else 'unknown'
    state_distribution = {state: count/len(metacognitive_states) 
                         for state, count in state_counter.items()} if metacognitive_states else {}
    
    architecture = {
        'seed': seed,
        'phi_mean': np.mean(phi_values),
        'phi_std': np.std(phi_values),
        'phi_min': np.min(phi_values),
        'phi_max': np.max(phi_values),
        'coherence_mean': np.mean(coherence_values),
        'coherence_std': np.std(coherence_values),
        'dominant_state': dominant_state,
        'state_distribution': state_distribution,
        'surprise_mean': np.mean(surprise_values),
        'surprise_std': np.std(surprise_values),
        'phi_stability': 1 / (1 + np.std(phi_values))  # Métrica de estabilidad
    }
    
    return architecture


def analyze_reproducibility(input_text, num_runs=10, iterations_per_run=50):
    """
    Analiza la reproducibilidad de las arquitecturas causales
    """
    print("=" * 70)
    print("TEST DE REPRODUCIBILIDAD DEL LENGUAJE CAUSAL")
    print("=" * 70)
    print(f"\nTexto de prueba: '{input_text}'")
    print(f"Número de ejecuciones: {num_runs}")
    print(f"Iteraciones por ejecución: {iterations_per_run}")
    print("\nEjecutando experimentos...\n")
    
    # Extraer arquitecturas con diferentes seeds
    architectures = []
    seeds = list(range(1000, 1000 + num_runs))  # Seeds: 1000-1009
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n[Ejecución {i}/{num_runs}]")
        arch = extract_causal_architecture(input_text, seed, iterations_per_run)
        if arch is None:
            print(f"  ⚠️ Saltando ejecución {i} por errores")
            continue
        architectures.append(arch)
        print(f"  Φ: {arch['phi_mean']:.4f} ± {arch['phi_std']:.4f}")
        print(f"  Estado dominante: {arch['dominant_state']}")
        print(f"  Estabilidad Φ: {arch['phi_stability']:.4f}")
    
    # Análisis de varianza entre ejecuciones
    print("\n" + "=" * 70)
    print("ANÁLISIS DE REPRODUCIBILIDAD")
    print("=" * 70)
    
    if not architectures:
        print("\n❌ ERROR: No se pudieron obtener arquitecturas válidas")
        return None
    
    phi_means = [arch['phi_mean'] for arch in architectures]
    phi_stds = [arch['phi_std'] for arch in architectures]
    coherence_means = [arch['coherence_mean'] for arch in architectures]
    surprise_means = [arch['surprise_mean'] for arch in architectures]
    
    # Varianza entre seeds (inter-seed variance)
    inter_seed_phi_var = np.var(phi_means)
    inter_seed_phi_std = np.std(phi_means)
    
    # Varianza promedio dentro de cada seed (intra-seed variance)
    intra_seed_phi_var = np.mean([arch['phi_std']**2 for arch in architectures])
    
    print(f"\n📊 VARIANZA DE Φ:")
    print(f"  Entre seeds (inter-seed): {inter_seed_phi_var:.6f}")
    print(f"  Desviación estándar: {inter_seed_phi_std:.6f}")
    print(f"  Dentro de cada seed (intra-seed): {intra_seed_phi_var:.6f}")
    print(f"  Rango Φ: [{min(phi_means):.4f}, {max(phi_means):.4f}]")
    
    # Estados metacognitivos
    dominant_states = [arch['dominant_state'] for arch in architectures]
    state_counter = Counter(dominant_states)
    state_consistency = state_counter.most_common(1)[0][1] / num_runs
    
    print(f"\n🧠 ESTADOS METACOGNITIVOS:")
    for state, count in state_counter.most_common():
        percentage = (count / num_runs) * 100
        print(f"  {state}: {count}/{num_runs} ({percentage:.1f}%)")
    print(f"  Consistencia del estado dominante: {state_consistency:.2%}")
    
    # Coherencia y surprise
    print(f"\n🔗 COHERENCIA:")
    print(f"  Media: {np.mean(coherence_means):.4f} ± {np.std(coherence_means):.4f}")
    print(f"  Varianza: {np.var(coherence_means):.6f}")
    
    print(f"\n⚡ SURPRISE:")
    print(f"  Media: {np.mean(surprise_means):.4f} ± {np.std(surprise_means):.4f}")
    print(f"  Varianza: {np.var(surprise_means):.6f}")
    
    # Evaluación del determinismo
    print("\n" + "=" * 70)
    print("EVALUACIÓN DEL DETERMINISMO")
    print("=" * 70)
    
    threshold_var = 0.05
    is_deterministic = inter_seed_phi_var < threshold_var
    
    print(f"\n🎯 Umbral de varianza: {threshold_var}")
    print(f"📏 Varianza observada: {inter_seed_phi_var:.6f}")
    
    if is_deterministic:
        print(f"\n✅ RESULTADO: El lenguaje causal ES DETERMINISTA")
        print(f"   La varianza ({inter_seed_phi_var:.6f}) < {threshold_var}")
        print(f"   Las arquitecturas son REPRODUCIBLES entre seeds diferentes")
    else:
        print(f"\n❌ RESULTADO: El lenguaje causal NO es completamente determinista")
        print(f"   La varianza ({inter_seed_phi_var:.6f}) >= {threshold_var}")
        print(f"   Hay variabilidad significativa entre seeds")
    
    # Ratio de varianza (señal/ruido)
    signal_to_noise = np.mean(phi_means) / inter_seed_phi_std if inter_seed_phi_std > 0 else float('inf')
    print(f"\n📡 Ratio señal/ruido: {signal_to_noise:.2f}")
    print(f"   (Φ_media / σ_inter-seed)")
    
    # Consistencia del estado metacognitivo
    if state_consistency >= 0.8:
        print(f"\n🧠 Estados metacognitivos: ALTAMENTE CONSISTENTES ({state_consistency:.0%})")
    elif state_consistency >= 0.6:
        print(f"\n🧠 Estados metacognitivos: MODERADAMENTE CONSISTENTES ({state_consistency:.0%})")
    else:
        print(f"\n🧠 Estados metacognitivos: BAJA CONSISTENCIA ({state_consistency:.0%})")
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'input_text': input_text,
        'num_runs': num_runs,
        'iterations_per_run': iterations_per_run,
        'seeds': seeds,
        'architectures': architectures,
        'statistics': {
            'phi_inter_seed_variance': float(inter_seed_phi_var),
            'phi_inter_seed_std': float(inter_seed_phi_std),
            'phi_intra_seed_variance': float(intra_seed_phi_var),
            'phi_mean': float(np.mean(phi_means)),
            'phi_range': [float(min(phi_means)), float(max(phi_means))],
            'coherence_mean': float(np.mean(coherence_means)),
            'coherence_variance': float(np.var(coherence_means)),
            'surprise_mean': float(np.mean(surprise_means)),
            'surprise_variance': float(np.var(surprise_means)),
            'state_consistency': float(state_consistency),
            'dominant_state': state_counter.most_common(1)[0][0],
            'signal_to_noise': float(signal_to_noise),
            'is_deterministic': is_deterministic
        },
        'timestamp': timestamp
    }
    
    output_file = f"reproducibility_test_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Test con "mi perro es rojo"
    input_text = "mi perro es rojo"
    results = analyze_reproducibility(input_text, num_runs=10, iterations_per_run=50)
    
    # Conclusión final
    print("\n📋 CONCLUSIONES:")
    print("=" * 70)
    
    if results['statistics']['is_deterministic']:
        print("\n🎯 El sistema genera ARQUITECTURAS CAUSALES REPRODUCIBLES")
        print("   - La varianza entre seeds es menor al umbral (0.05)")
        print("   - El lenguaje causal del sistema es DETERMINISTA")
        print("   - Diferentes seeds producen la misma 'palabra causal'")
        print("\n💡 IMPLICACIÓN:")
        print("   Podemos confiar en que el sistema 'dice lo mismo'")
        print("   cuando recibe el mismo input, independientemente del seed.")
    else:
        print("\n⚠️  El sistema muestra VARIABILIDAD SIGNIFICATIVA")
        print("   - La varianza entre seeds excede el umbral (0.05)")
        print("   - El lenguaje causal tiene componente estocástico")
        print("   - Diferentes seeds producen arquitecturas ligeramente diferentes")
        print("\n💡 IMPLICACIÓN:")
        print("   El 'vocabulario causal' tiene cierta variabilidad intrínseca.")
        print("   Se requiere promediado de múltiples ejecuciones para estabilidad.")
    
    state_cons = results['statistics']['state_consistency']
    if state_cons >= 0.8:
        print(f"\n✅ Estados metacognitivos ALTAMENTE ESTABLES ({state_cons:.0%})")
    else:
        print(f"\n⚠️  Estados metacognitivos muestran VARIABILIDAD ({state_cons:.0%})")
