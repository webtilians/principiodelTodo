#!/usr/bin/env python3
"""
🔍 Diagnóstico de NaN en PHI durante entrenamiento
"""

import torch
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/core')
from torch.amp import autocast

from train_gpt2_with_phi_observer import InfinitoGPT2WithObserver

device = 'cuda'
print("Cargando modelo...")
model = InfinitoGPT2WithObserver(use_lora=True).to(device)
model.train()

# Simular batch de entrenamiento
inputs = model.tokenizer('Esto es una prueba de entrenamiento', return_tensors='pt').to(device)

print('\n1. Test SIN autocast:')
outputs, phi = model(inputs.input_ids, labels=inputs.input_ids, return_phi=True)
phi_val = phi['phi'].mean().item()
print(f'   PHI: {phi_val:.4f}')
print(f'   Loss: {outputs.loss.item():.4f}')
print(f'   PHI is NaN? {torch.isnan(phi["phi"]).any().item()}')

print('\n2. Test CON autocast:')
with autocast(device_type='cuda'):
    outputs, phi = model(inputs.input_ids, labels=inputs.input_ids, return_phi=True)
    phi_val = phi['phi'].mean().item()
    print(f'   PHI: {phi_val:.4f}')
    print(f'   Loss: {outputs.loss.item():.4f}')
    print(f'   PHI is NaN? {torch.isnan(phi["phi"]).any().item()}')

# Verificar componentes
print('\n3. Componentes PHI:')
for k, v in phi['raw_components'].items():
    print(f'   {k}: {v}')
    if isinstance(v, float) and (v != v):  # NaN check
        print(f'      ⚠️ {k} es NaN!')

# Verificar attentions
print('\n4. Attentions:')
with torch.no_grad():
    test_outputs = model.gpt2(
        inputs.input_ids, 
        output_attentions=True,
        return_dict=True
    )
    print(f'   attentions type: {type(test_outputs.attentions)}')
    if test_outputs.attentions:
        print(f'   attentions length: {len(test_outputs.attentions)}')
        last_attn = test_outputs.attentions[-1]
        if last_attn is not None:
            print(f'   last_attn shape: {last_attn.shape}')
            print(f'   last_attn has NaN? {torch.isnan(last_attn).any().item()}')
        else:
            print('   ⚠️ last_attn is None!')
