#!/usr/bin/env python3
"""
🔍 Diagnóstico de problemas en entrenamiento
"""

import torch
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/core')

print('='*60)
print('🔍 DIAGNÓSTICO DE PROBLEMAS')
print('='*60)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

try:
    from train_gpt2_with_phi_observer import InfinitoGPT2WithObserver
    
    print('\n1. Creando modelo...')
    model = InfinitoGPT2WithObserver(use_lora=True)
    model = model.to(device)
    print('   ✅ Modelo creado')
    
    print('\n2. Test forward pass...')
    test_input = model.tokenizer('Hola mundo, esto es una prueba', return_tensors='pt').to(device)
    
    # Forward con memoria
    outputs, phi = model(test_input.input_ids, return_phi=True, use_memory=True)
    
    print('   ✅ Forward OK')
    print(f'   PHI (escala 0-10): {phi["phi"].mean().item():.2f}')
    print(f'   Temporal: {phi["raw_components"]["temporal"]:.4f}')
    print(f'   Integration: {phi["raw_components"]["integration"]:.4f}')
    print(f'   Complexity: {phi["raw_components"]["complexity"]:.4f}')
    print(f'   Attention: {phi["raw_components"]["attention"]:.4f}')
    
    print('\n3. Estado de memoria IIT...')
    print(f'   threshold_logit: {model.iit_memory.threshold_logit.item():.4f}')
    print(f'   threshold (exp): {model.iit_memory.threshold_logit.exp().item():.4f}')
    print(f'   Slot count: {model.iit_memory.memory_slots}')
    
    # Verificar qué pasa con la prioridad
    print('\n4. Diagnóstico de prioridad...')
    phi_value = phi["phi"].detach()
    print(f'   PHI value: {phi_value.mean().item():.4f}')
    
    # Calcular prioridad como lo hace la memoria
    phi_normalized = torch.clamp(phi_value / 10.0, 0, 1)
    print(f'   PHI normalized (0-1): {phi_normalized.mean().item():.4f}')
    
    # Priority con alpha=0.8
    priority = 0.8 * phi_normalized
    print(f'   Priority (alpha=0.8): {priority.mean().item():.4f}')
    
    threshold = model.iit_memory.threshold_logit.exp()
    print(f'   Threshold: {threshold.item():.4f}')
    
    above_threshold = priority > threshold
    print(f'   ¿Priority > Threshold? {above_threshold.any().item()}')
    
    # Verificar memoria info
    print('\n5. Memory info del forward...')
    if 'memory' in phi:
        mem = phi['memory']
        print(f'   Utilization: {mem["utilization"]*100:.1f}%')
        print(f'   Mean PHI: {mem["mean_phi"]:.4f}')
        print(f'   Writes this batch: {mem["writes_this_batch"]}')
        print(f'   Threshold: {mem["threshold"]:.4f}')
    else:
        print('   ❌ No hay memory info')
    
    # Problema identificado?
    print('\n6. Verificando problema...')
    if priority.mean() < threshold:
        print(f'   ⚠️ PROBLEMA: Priority ({priority.mean().item():.4f}) < Threshold ({threshold.item():.4f})')
        print(f'   → La memoria NUNCA se escribirá con estos valores')
        print(f'   → Solución: Bajar threshold inicial o cambiar escala')
    else:
        print(f'   ✅ Priority > Threshold, memoria debería escribirse')
    
    print('\n✅ DIAGNÓSTICO COMPLETADO')
    
except Exception as e:
    print(f'\n❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
