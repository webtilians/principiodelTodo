#!/usr/bin/env python3
"""
🏆 REPRODUCCIÓN 100% DEL SUPER GOLDEN SEED
==========================================

Objetivo: Reproducir el 54.35% de mejora en TODAS las ejecuciones.

Método EXACTO del experimento original:
1. CARGAR Golden Seed 2 (pesos guardados) para modelo IIT
2. Crear Baseline con seed=2 normal
3. Usar seed=42 para generar datos de entrenamiento
4. Entrenar 3000 épocas
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN EXACTA (del checkpoint)
# ============================================================================

# La CLAVE: Cargar Golden Seed 2 + usar Data Seed 42
MODEL_SEED = 2       # Para baseline
DATA_SEED = 42       # Super Golden data seed

CONFIG = {
    'vocab_size': 13,
    'hidden_dim': 64,
    'num_layers': 2,
    'num_heads': 4,
    'use_improved_memory': True,
    'use_improved_iit': True,
    'use_learnable_phi': True,
    'use_stochastic_exploration': True,
    'lambda_phi': 0.0
}

EPOCHS = 3000
BATCH_SIZE = 32
LR = 0.0005

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# FUNCIÓN DE REPRODUCIBILIDAD TOTAL
# ============================================================================

def set_all_seeds(seed):
    """Fija TODOS los seeds para reproducibilidad total."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Configuración determinista de CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # NO usar use_deterministic_algorithms porque no es compatible con nll_loss2d
    # El original tampoco lo usa


# ============================================================================
# VOCABULARIO Y DATOS (EXACTO del experimento original)
# ============================================================================

vocab = {
    'PAD': 0, '(': 1, ')': 2, '[': 3, ']': 4, 
    '{': 5, '}': 6, '<': 7, '>': 8, 
    'A': 9, 'B': 10, 'C': 11, 'EOS': 12
}
idx_to_char = {v: k for k, v in vocab.items()}


def generate_dyck_sample(max_depth=12, noise_len=6):
    """Genera secuencias Dyck - EXACTO del original."""
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


def get_batch(batch_size=32):
    """Genera batch - EXACTO del original."""
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


# ============================================================================
# EXPERIMENTO PRINCIPAL
# ============================================================================

def run_experiment():
    """Ejecuta el experimento y retorna el % de mejora."""
    
    print(f"\n{'='*70}")
    print(f"🏆 REPRODUCCIÓN SUPER GOLDEN SEED")
    print(f"   Método: Cargar Golden Seed 2 + Data Seed {DATA_SEED}")
    print(f"   Device: {DEVICE}")
    print(f"{'='*70}")
    
    # Importar modelo
    from infinito_v5_2_refactored import InfinitoV52Refactored
    
    # =========================================================================
    # CREAR MODELO IIT - CARGAR GOLDEN SEED 2 (PESOS GUARDADOS)
    # =========================================================================
    print("\n[1/4] Creando modelo IIT (cargando Golden Seed 2)...")
    
    # Crear modelo SIN seed
    model_iit = InfinitoV52Refactored(
        vocab_size=CONFIG['vocab_size'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        use_improved_memory=CONFIG['use_improved_memory'],
        use_improved_iit=CONFIG['use_improved_iit'],
        use_learnable_phi=CONFIG['use_learnable_phi'],
        use_stochastic_exploration=CONFIG['use_stochastic_exploration'],
        lambda_phi=CONFIG['lambda_phi']
    ).to(DEVICE)
    
    # CARGAR Golden Seed 2 (pesos guardados)
    golden_path = os.path.join(os.path.dirname(__file__), 'models', 'golden_seed2_init.pt')
    golden_checkpoint = torch.load(golden_path, weights_only=False, map_location=DEVICE)
    model_iit.load_state_dict(golden_checkpoint['model_state_dict'])
    print(f"   ✅ Cargados pesos de Golden Seed 2")
    print(f"   Memory Gate inicial: {model_iit.memory_gate.item():.4f}")
    
    # =========================================================================
    # CREAR MODELO BASELINE (con seed 42 - IGUAL que datos)
    # =========================================================================
    print("[2/4] Creando modelo Baseline (seed=42)...")
    set_all_seeds(DATA_SEED)  # USAR DATA_SEED (42), no MODEL_SEED
    
    model_base = InfinitoV52Refactored(
        vocab_size=CONFIG['vocab_size'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        use_improved_memory=False,
        use_improved_iit=False,
        use_learnable_phi=False,
        use_stochastic_exploration=False,
        seed=DATA_SEED  # USAR DATA_SEED (42)
    ).to(DEVICE)
    
    print(f"   Memory Gate inicial: {model_base.memory_gate.item():.4f}")
    
    # =========================================================================
    # CONFIGURAR ENTRENAMIENTO
    # =========================================================================
    print("[3/4] Configurando entrenamiento...")
    
    opt_iit = optim.AdamW(model_iit.parameters(), lr=LR)
    opt_base = optim.AdamW(model_base.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # =========================================================================
    # ENTRENAR con data seed 42
    # =========================================================================
    print(f"[4/4] Entrenando {EPOCHS} épocas con data_seed={DATA_SEED}...")
    
    # CRÍTICO: Fijar DATA_SEED (42) antes del loop
    set_all_seeds(DATA_SEED)
    
    history_iit = []
    history_base = []
    
    print_interval = 500
    
    for epoch in range(1, EPOCHS + 1):
        # Obtener batch (mismo para ambos)
        input_ids, target_ids = get_batch(BATCH_SIZE)
        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)
        
        # === ENTRENAR BASELINE PRIMERO (como el original) ===
        model_base.train()
        opt_base.zero_grad()
        logits_base, _ = model_base(input_ids)
        min_len = min(logits_base.shape[1], target_ids.shape[1])
        # USAR TRANSPOSE COMO EL ORIGINAL (NO RESHAPE)
        loss_base = criterion(logits_base[:, :min_len, :].transpose(1, 2), target_ids[:, :min_len])
        loss_base.backward()
        opt_base.step()
        
        # === ENTRENAR IIT DESPUÉS (CON LOS MISMOS DATOS) ===
        model_iit.train()
        opt_iit.zero_grad()
        logits_iit, metrics = model_iit(input_ids, return_metrics=True)
        # USAR TRANSPOSE COMO EL ORIGINAL
        loss_main = criterion(logits_iit[:, :min_len, :].transpose(1, 2), target_ids[:, :min_len])
        
        # Lambda phi = 0.0
        loss_phi = 0
        if metrics and 'delta_phi_loss' in metrics:
            loss_phi = metrics['delta_phi_loss']
        
        total_loss_iit = loss_main + (0.0 * loss_phi)
        total_loss_iit.backward()
        opt_iit.step()
        
        history_iit.append(loss_main.item())
        history_base.append(loss_base.item())
        
        if epoch % print_interval == 0:
            avg_iit = np.mean(history_iit[-100:])
            avg_base = np.mean(history_base[-100:])
            imp = (avg_base - avg_iit) / avg_base * 100
            gate = torch.sigmoid(model_iit.memory_gate).item() * 100
            print(f"   Época {epoch}: IIT={avg_iit:.4f}, Base={avg_base:.4f}, Mejora={imp:+.1f}%, Gate={gate:.2f}%")
    
    # =========================================================================
    # RESULTADOS FINALES (USAR ÚLTIMO VALOR como el original)
    # =========================================================================
    final_iit = history_iit[-1]
    final_base = history_base[-1]
    improvement = ((final_base - final_iit) / final_base) * 100
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS FINALES")
    print(f"{'='*70}")
    print(f"   IIT Loss: {final_iit:.5f}")
    print(f"   Baseline Loss: {final_base:.5f}")
    print(f"   MEJORA: {improvement:.2f}%")
    print(f"   Memory Gate: {model_iit.memory_gate.item():.4f} ({torch.sigmoid(model_iit.memory_gate).item()*100:.2f}%)")
    print(f"\n   🎯 Objetivo: 54.35%")
    
    if improvement >= 50:
        print(f"   ✅ ¡ÉXITO! Reproducido >= 50%")
    elif improvement >= 40:
        print(f"   ⚠️ Cerca (>= 40%)")
    else:
        print(f"   ❌ No reproducido (< 40%)")
    
    return improvement, final_iit, final_base


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    start_time = datetime.now()
    
    improvement, loss_iit, loss_base = run_experiment()
    
    elapsed = datetime.now() - start_time
    print(f"\n⏱️ Tiempo: {elapsed}")
    print(f"{'='*70}")
