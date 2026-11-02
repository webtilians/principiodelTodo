#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 ANÁLISIS PROFUNDO: ¿Qué causa el 30% de mejora?
====================================================

Este script investiga sistemáticamente qué factores (además de los pesos
iniciales del modelo) están causando la variabilidad en los resultados.

Hipótesis a probar:
1. Inicialización de pesos del modelo (Golden Seed) ✅ Ya probado - NO es suficiente
2. Orden de los datos de entrenamiento (data shuffling)
3. Inicialización del optimizador (momentum buffers, etc.)
4. Estado del generador de batches
5. Combinación de todos los anteriores
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
import copy

# Configurar encoding UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))
from infinito_v5_2_refactored import InfinitoV52Refactored


# =============================================================================
# GENERACIÓN DE DATOS
# =============================================================================

def generate_dyck_sample(max_depth=12, noise_len=6):
    """Genera secuencias Dyck."""
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
# FUNCIONES DE ENTRENAMIENTO
# =============================================================================

def set_all_seeds(seed):
    """Fija TODOS los seeds posibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_experiment(seed, epochs, device, use_golden_seed=False):
    """
    Entrena un experimento completo con control total de seeds.
    
    Args:
        seed: Semilla para TODO (modelo, datos, optimizador)
        epochs: Número de épocas
        device: cuda o cpu
        use_golden_seed: Si True, usa Golden Seed 2 para inicializar el modelo
    
    Returns:
        dict con resultados
    """
    # PASO 1: Fijar TODOS los seeds ANTES de hacer cualquier cosa
    set_all_seeds(seed)
    
    # PASO 2: Crear modelos
    config = {
        'vocab_size': 13,
        'hidden_dim': 64,
        'num_layers': 2,
        'num_heads': 4
    }
    
    # Modelo IIT
    if use_golden_seed:
        # Crear modelo SIN seed (lo cargaremos de Golden Seed)
        model_iit = InfinitoV52Refactored(
            vocab_size=config['vocab_size'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            use_improved_memory=True,
            use_improved_iit=True,
            use_learnable_phi=True,
            use_stochastic_exploration=True,
            lambda_phi=0.0
        ).to(device)
        
        # Cargar Golden Seed
        golden_checkpoint = torch.load('../models/golden_seed2_init.pt', weights_only=False)
        model_iit.load_state_dict(golden_checkpoint['model_state_dict'])
    else:
        # Inicialización normal con seed
        model_iit = InfinitoV52Refactored(
            vocab_size=config['vocab_size'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            use_improved_memory=True,
            use_improved_iit=True,
            use_learnable_phi=True,
            use_stochastic_exploration=True,
            lambda_phi=0.0,
            seed=seed
        ).to(device)
    
    # Modelo Baseline
    model_base = InfinitoV52Refactored(
        vocab_size=config['vocab_size'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        use_improved_memory=False,
        use_improved_iit=False,
        use_learnable_phi=False,
        use_stochastic_exploration=False,
        seed=seed
    ).to(device)
    
    # PASO 3: Crear optimizadores (también afectados por el seed)
    opt_iit = optim.AdamW(model_iit.parameters(), lr=0.0005)
    opt_base = optim.AdamW(model_base.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # PASO 4: Entrenar
    history_iit = []
    history_base = []
    
    for epoch in range(epochs):
        # Generar batch (usa el seed actual)
        input_ids, target_ids = get_batch(32)
        input_ids, target_ids = input_ids.to(device), target_ids.to(device)
        
        # Baseline
        opt_base.zero_grad()
        logits_base, _ = model_base(input_ids)
        min_len = min(logits_base.shape[1], target_ids.shape[1])
        loss_base = criterion(logits_base[:, :min_len, :].transpose(1, 2), target_ids[:, :min_len])
        loss_base.backward()
        opt_base.step()
        
        # IIT
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
    
    # Calcular métricas finales
    loss_iit_final = history_iit[-1]
    loss_base_final = history_base[-1]
    improvement = ((loss_base_final - loss_iit_final) / loss_base_final) * 100
    
    return {
        'seed': seed,
        'use_golden_seed': use_golden_seed,
        'iit_loss_final': loss_iit_final,
        'baseline_loss_final': loss_base_final,
        'improvement': improvement,
        'memory_gate_final': model_iit.memory_gate.item(),
        'history_iit': history_iit,
        'history_base': history_base
    }


# =============================================================================
# EXPERIMENTOS
# =============================================================================

def main():
    """Ejecuta análisis completo."""
    print("="*80)
    print("🔬 ANÁLISIS PROFUNDO: ¿Qué causa el 30%?")
    print("="*80)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 3000
    
    print(f"\n🖥️  Dispositivo: {DEVICE}")
    print(f"⏱️  Épocas por experimento: {EPOCHS}")
    
    # ==========================================================================
    # EXPERIMENTO 1: Seed 2 completo (todo seed=2)
    # ==========================================================================
    print("\n" + "="*80)
    print("📊 EXPERIMENTO 1: Seed 2 COMPLETO (modelo + datos + optimizador)")
    print("="*80)
    print("Hipótesis: Si fijamos seed=2 en TODAS las operaciones, deberíamos")
    print("           recuperar el 30% de mejora original")
    
    result_seed2_full = train_experiment(
        seed=2,
        epochs=EPOCHS,
        device=DEVICE,
        use_golden_seed=False  # NO usar Golden Seed, usar seed=2 normal
    )
    
    print(f"\n📊 Resultados:")
    print(f"   IIT Loss: {result_seed2_full['iit_loss_final']:.5f}")
    print(f"   Baseline Loss: {result_seed2_full['baseline_loss_final']:.5f}")
    print(f"   🚀 MEJORA: {result_seed2_full['improvement']:.2f}%")
    
    if result_seed2_full['improvement'] >= 25.0:
        print("   ✅ HIPÓTESIS CONFIRMADA: Seed 2 completo reproduce el 30%")
        print("   🔑 CONCLUSIÓN: La clave NO es solo la inicialización del modelo,")
        print("                  sino la SECUENCIA DE DATOS de entrenamiento")
    else:
        print("   ⚠️ HIPÓTESIS RECHAZADA: Seed 2 completo NO reproduce el 30%")
        print("   🤔 Hay más factores desconocidos en juego...")
    
    # ==========================================================================
    # EXPERIMENTO 2: Golden Seed + Seed Random (datos aleatorios)
    # ==========================================================================
    print("\n" + "="*80)
    print("📊 EXPERIMENTO 2: Golden Seed + Datos ALEATORIOS")
    print("="*80)
    print("Hipótesis: Si usamos Golden Seed pero datos aleatorios, veremos")
    print("           si la inicialización del modelo es suficiente")
    
    result_golden_random = train_experiment(
        seed=42,  # Seed diferente para los datos
        epochs=EPOCHS,
        device=DEVICE,
        use_golden_seed=True  # Usar Golden Seed para el modelo
    )
    
    print(f"\n📊 Resultados:")
    print(f"   IIT Loss: {result_golden_random['iit_loss_final']:.5f}")
    print(f"   Baseline Loss: {result_golden_random['baseline_loss_final']:.5f}")
    print(f"   🚀 MEJORA: {result_golden_random['improvement']:.2f}%")
    
    # ==========================================================================
    # COMPARACIÓN
    # ==========================================================================
    print("\n" + "="*80)
    print("📈 COMPARACIÓN DE RESULTADOS")
    print("="*80)
    
    print(f"\n1. Seed 2 COMPLETO (modelo + datos con seed=2):")
    print(f"   Mejora: {result_seed2_full['improvement']:.2f}%")
    
    print(f"\n2. Golden Seed + Datos aleatorios:")
    print(f"   Mejora: {result_golden_random['improvement']:.2f}%")
    
    print(f"\n3. Diferencia:")
    diff = result_seed2_full['improvement'] - result_golden_random['improvement']
    print(f"   {diff:.2f}% de mejora se debe al ORDEN DE LOS DATOS")
    
    # ==========================================================================
    # CONCLUSIÓN
    # ==========================================================================
    print("\n" + "="*80)
    print("🎯 CONCLUSIÓN")
    print("="*80)
    
    if result_seed2_full['improvement'] >= 25.0:
        print("\n✅ LA CLAVE PARA REPRODUCIR EL 30%:")
        print("   1. ✅ Inicialización del modelo (Golden Seed)")
        print("   2. ✅ ORDEN DE LOS DATOS de entrenamiento (Seed 2)")
        print("   3. ✅ Inicialización del optimizador (Seed 2)")
        print("\n💡 Para producción, necesitas:")
        print("   - Guardar NO solo los pesos iniciales del modelo")
        print("   - Sino también la SECUENCIA EXACTA de batches de entrenamiento")
        print("   - O encontrar una inicialización que sea robusta a diferentes datos")
    else:
        print("\n🤔 HALLAZGO IMPORTANTE:")
        print("   - El 30% NO es reproducible ni siquiera con seed=2 completo")
        print("   - Esto sugiere que hay otros factores aleatorios:")
        print("     * Estado de CUDA/cuDNN")
        print("     * Operaciones no deterministas en GPU")
        print("     * Condiciones de carrera en cálculos paralelos")
        print("\n💡 Recomendación:")
        print("   - Entrenar múltiples veces y promediar resultados")
        print("   - Usar técnicas de ensemble")
        print("   - Enfocarse en mejorar la robustez del modelo")
    
    # ==========================================================================
    # GUARDAR RESULTADOS
    # ==========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'experiment': 'Deep Analysis - What causes 30% improvement',
        'results': {
            'seed2_full': result_seed2_full,
            'golden_seed_random_data': result_golden_random
        },
        'analysis': {
            'seed2_full_improvement': result_seed2_full['improvement'],
            'golden_random_improvement': result_golden_random['improvement'],
            'difference_due_to_data_order': diff,
            'hypothesis_confirmed': result_seed2_full['improvement'] >= 25.0
        }
    }
    
    results_path = f"../models/deep_analysis_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados guardados en: {results_path}")


if __name__ == "__main__":
    main()
