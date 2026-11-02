#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 EXPERIMENTO ESTADÍSTICO: 10 Runs con Seeds Fijos
===================================================

Ejecuta 10 experimentos independientes con seeds diferentes para
obtener resultados estadísticamente robustos y determinar si el
modelo IIT supera consistentemente al Baseline.
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Configurar encoding UTF-8 para Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Importar modelo
sys.path.insert(0, os.path.dirname(__file__))
from infinito_v5_2_refactored import InfinitoV52Refactored


# =============================================================================
# GENERACIÓN DE DATOS (COPIADO DE infinito_gemini.py)
# =============================================================================

def generate_dyck_sample(max_depth=12, noise_len=6):
    """Genera secuencias Dyck con profundidad y ruido."""
    pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
    depth = random.randint(4, max_depth)
    stack = []
    sequence = []
    
    for _ in range(depth):
        pair = random.choice(pairs)
        sequence.append(pair[0])
        stack.append(pair[1])
    
    noise = [random.choice(['A', 'B', 'C']) for _ in range(noise_len)]
    input_str = sequence + noise
    target_str = list(reversed(stack))
    
    return input_str, target_str


vocab = {'PAD': 0, '(': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6, '<': 7, '>': 8, 
         'A': 9, 'B': 10, 'C': 11, 'EOS': 12}
idx_to_char = {v: k for k, v in vocab.items()}


def get_batch(batch_size=32):
    """Genera un batch de datos."""
    inputs = []
    targets = []
    for _ in range(batch_size):
        inp, tar = generate_dyck_sample()
        inp_ids = [vocab[c] for c in inp]
        tar_ids = [vocab[c] for c in tar] + [vocab['EOS']]
        
        inputs.append(torch.tensor(inp_ids))
        targets.append(torch.tensor(tar_ids))
        
    inp_tens = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    tar_tens = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inp_tens, tar_tens


# =============================================================================
# FUNCIÓN DE ENTRENAMIENTO PARA UN SEED
# =============================================================================

def train_with_seed(seed, epochs=3000, device='cuda'):
    """
    Entrena un par de modelos (IIT + Baseline) con un seed fijo.
    
    Returns:
        dict con resultados del experimento
    """
    # Fijar seeds para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"\n{'='*70}")
    print(f"🌱 SEED {seed} - Entrenamiento iniciado")
    print(f"{'='*70}")
    
    # Configuración
    HIDDEN_DIM = 64
    LAYERS = 2
    HEADS = 4
    
    # Crear modelos
    model_iit = InfinitoV52Refactored(
        vocab_size=len(vocab),
        hidden_dim=HIDDEN_DIM,
        num_layers=LAYERS,
        num_heads=HEADS,
        use_improved_memory=True,  
        use_improved_iit=True,
        use_learnable_phi=True,
        use_stochastic_exploration=True,  # <--- ACTIVADO (igual que v3)
        lambda_phi=0.0,
        seed=seed  # Fijar seed en el modelo
    ).to(device)
    
    model_base = InfinitoV52Refactored(
        vocab_size=len(vocab),
        hidden_dim=HIDDEN_DIM,
        num_layers=LAYERS,
        num_heads=HEADS,
        use_improved_memory=False,
        use_improved_iit=False,
        use_learnable_phi=False,
        use_stochastic_exploration=False,
        seed=seed  # Fijar seed en el modelo
    ).to(device)
    
    # Optimizadores
    opt_iit = optim.AdamW(model_iit.parameters(), lr=0.0005)
    opt_base = optim.AdamW(model_base.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Entrenamiento
    history_iit = []
    history_base = []
    
    pbar = tqdm(range(epochs), desc=f"Seed {seed}")
    for epoch in pbar:
        input_ids, target_ids = get_batch(32)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        
        # Baseline
        opt_base.zero_grad()
        logits_base, _ = model_base(input_ids)
        min_len = min(logits_base.shape[1], target_ids.shape[1])
        loss_base = criterion(logits_base[:, :min_len, :].transpose(1, 2), target_ids[:, :min_len])
        loss_base.backward()
        opt_base.step()
        
        # Infinito IIT
        opt_iit.zero_grad()
        logits_iit, metrics = model_iit(input_ids, return_metrics=True)
        loss_main = criterion(logits_iit[:, :min_len, :].transpose(1, 2), target_ids[:, :min_len])
        
        loss_phi = 0
        if metrics and 'delta_phi_loss' in metrics:
            loss_phi = metrics['delta_phi_loss']
        
        total_loss_iit = loss_main + (0.0 * loss_phi)
        total_loss_iit.backward()
        opt_iit.step()
        
        history_base.append(loss_base.item())
        history_iit.append(loss_main.item())
        
        if epoch % 500 == 0:
            pbar.set_description(f"Seed {seed} | Base: {loss_base.item():.4f} | IIT: {loss_main.item():.4f}")
    
    # Calcular métricas finales
    loss_iit_final = history_iit[-1]
    loss_base_final = history_base[-1]
    improvement = ((loss_base_final - loss_iit_final) / loss_base_final) * 100
    
    # Memory gate final
    memory_gate_value = model_iit.memory_gate.item()
    memory_gate_activated = torch.sigmoid(model_iit.memory_gate).item()
    
    # Promedios de las últimas 500 épocas
    avg_iit_late = sum(history_iit[-500:]) / 500
    avg_base_late = sum(history_base[-500:]) / 500
    
    # Imprimir resumen
    print(f"\n📊 Resultados Seed {seed}:")
    print(f"   IIT Loss: {loss_iit_final:.5f}")
    print(f"   Base Loss: {loss_base_final:.5f}")
    print(f"   Mejora: {improvement:.2f}%")
    print(f"   Memory Gate: {memory_gate_value:.6f} → {memory_gate_activated*100:.2f}%")
    print(f"   {'✅ IIT GANA' if improvement > 0 else '❌ Baseline GANA'}")
    
    # Retornar resultados
    return {
        'seed': seed,
        'iit_loss_final': loss_iit_final,
        'baseline_loss_final': loss_base_final,
        'improvement_percentage': improvement,
        'memory_gate_value': memory_gate_value,
        'memory_gate_activated': memory_gate_activated,
        'iit_avg_late': avg_iit_late,
        'baseline_avg_late': avg_base_late,
        'iit_wins': improvement > 0,
        'history_iit': history_iit,
        'history_baseline': history_base
    }


# =============================================================================
# ANÁLISIS ESTADÍSTICO
# =============================================================================

def analyze_results(results):
    """Analiza estadísticamente los resultados de múltiples experimentos."""
    
    print("\n" + "="*70)
    print("📊 ANÁLISIS ESTADÍSTICO DE RESULTADOS")
    print("="*70)
    
    # Extraer datos
    improvements = [r['improvement_percentage'] for r in results]
    iit_losses = [r['iit_loss_final'] for r in results]
    base_losses = [r['baseline_loss_final'] for r in results]
    gate_values = [r['memory_gate_value'] for r in results]
    gate_activations = [r['memory_gate_activated'] for r in results]
    iit_wins = sum([r['iit_wins'] for r in results])
    
    # Calcular estadísticas
    mean_improvement = np.mean(improvements)
    std_improvement = np.std(improvements)
    median_improvement = np.median(improvements)
    
    mean_iit_loss = np.mean(iit_losses)
    std_iit_loss = np.std(iit_losses)
    
    mean_base_loss = np.mean(base_losses)
    std_base_loss = np.std(base_losses)
    
    mean_gate = np.mean(gate_values)
    std_gate = np.std(gate_values)
    
    mean_gate_activation = np.mean(gate_activations)
    
    # Imprimir resultados
    print(f"\n🎯 MEJORA DEL IIT vs BASELINE:")
    print(f"   Media: {mean_improvement:.2f}%")
    print(f"   Desviación Estándar: {std_improvement:.2f}%")
    print(f"   Mediana: {median_improvement:.2f}%")
    print(f"   Rango: [{min(improvements):.2f}%, {max(improvements):.2f}%]")
    print(f"   Victorias del IIT: {iit_wins}/10 ({iit_wins*10}%)")
    
    print(f"\n📉 LOSS FINAL (IIT):")
    print(f"   Media: {mean_iit_loss:.5f} ± {std_iit_loss:.5f}")
    print(f"   Rango: [{min(iit_losses):.5f}, {max(iit_losses):.5f}]")
    
    print(f"\n📉 LOSS FINAL (Baseline):")
    print(f"   Media: {mean_base_loss:.5f} ± {std_base_loss:.5f}")
    print(f"   Rango: [{min(base_losses):.5f}, {max(base_losses):.5f}]")
    
    print(f"\n🚪 MEMORY GATE:")
    print(f"   Valor medio: {mean_gate:.6f} ± {std_gate:.6f}")
    print(f"   Activación media: {mean_gate_activation*100:.2f}%")
    print(f"   Rango: [{min(gate_values):.6f}, {max(gate_values):.6f}]")
    
    # Análisis de significancia estadística (t-test pareado)
    from scipy import stats
    t_statistic, p_value = stats.ttest_rel(base_losses, iit_losses)
    
    print(f"\n🔬 SIGNIFICANCIA ESTADÍSTICA (t-test pareado):")
    print(f"   t-statistic: {t_statistic:.4f}")
    print(f"   p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        if mean_improvement > 0:
            print(f"   ✅ SIGNIFICATIVO: El IIT ES MEJOR que Baseline (p < 0.05)")
        else:
            print(f"   ❌ SIGNIFICATIVO: El Baseline ES MEJOR que IIT (p < 0.05)")
    else:
        print(f"   ⚠️ NO SIGNIFICATIVO: No hay diferencia estadística (p ≥ 0.05)")
    
    print(f"\n💡 INTERPRETACIÓN:")
    if mean_improvement > 5 and p_value < 0.05:
        print(f"   🏆 El modelo IIT supera consistentemente al Baseline")
        print(f"   → Mejora promedio: {mean_improvement:.2f}%")
        print(f"   → Con significancia estadística (p={p_value:.4f})")
    elif mean_improvement > 0 and p_value >= 0.05:
        print(f"   🤷 El IIT muestra ligera ventaja pero no es consistente")
        print(f"   → Mejora promedio: {mean_improvement:.2f}%")
        print(f"   → Sin significancia estadística (p={p_value:.4f})")
        print(f"   → La varianza entre seeds es muy alta")
    elif mean_improvement < -5 and p_value < 0.05:
        print(f"   ❌ El Baseline es significativamente mejor que el IIT")
        print(f"   → La arquitectura IIT necesita ajustes")
    else:
        print(f"   🔄 Ambos modelos son equivalentes estadísticamente")
        print(f"   → La diferencia se debe principalmente al azar")
    
    # Guardar resultados completos
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(results),
        'statistics': {
            'mean_improvement': mean_improvement,
            'std_improvement': std_improvement,
            'median_improvement': median_improvement,
            'iit_wins': iit_wins,
            'iit_win_rate': iit_wins / len(results),
            'mean_iit_loss': mean_iit_loss,
            'std_iit_loss': std_iit_loss,
            'mean_baseline_loss': mean_base_loss,
            'std_baseline_loss': std_base_loss,
            'mean_gate_value': mean_gate,
            'mean_gate_activation': mean_gate_activation,
            't_statistic': t_statistic,
            'p_value': p_value,
            'is_significant': p_value < 0.05
        },
        'individual_results': results
    }
    
    output_file = f'statistical_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n💾 Resultados guardados: {output_file}")
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Ejecuta 10 experimentos con seeds fijos."""
    
    print("="*70)
    print("🔬 EXPERIMENTO ESTADÍSTICO: 10 Seeds Fijos")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nConfiguración:")
    print(f"  • Seeds: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    print(f"  • Épocas por seed: 3000")
    print(f"  • Arquitectura: InfinitoV52 vs Baseline")
    print(f"  • Memory Gate inicial: -5.0 (~0.6% apertura)")
    print(f"  • Lambda PHI: 0.0")
    print(f"\n⏱️ Tiempo estimado: ~15-20 minutos")
    print("="*70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Usando dispositivo: {device}")
    
    # Seeds a probar
    seeds = list(range(1, 11))
    
    # Ejecutar experimentos
    results = []
    for seed in seeds:
        try:
            result = train_with_seed(seed, epochs=3000, device=device)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error en seed {seed}: {e}")
            continue
    
    # Análisis estadístico
    if len(results) >= 5:  # Mínimo 5 experimentos exitosos
        summary = analyze_results(results)
        
        print("\n" + "="*70)
        print("✅ EXPERIMENTO COMPLETADO")
        print("="*70)
        print(f"Total de experimentos exitosos: {len(results)}/10")
        print(f"Resultados guardados para análisis posterior")
        
    else:
        print(f"\n❌ Error: Solo {len(results)} experimentos exitosos")
        print("   Se necesitan al menos 5 para análisis estadístico válido")


if __name__ == '__main__':
    main()
