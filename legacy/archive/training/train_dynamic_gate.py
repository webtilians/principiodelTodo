#!/usr/bin/env python3
"""
🧠 ENTRENAMIENTO CON DYNAMIC GATE
=================================

Entrena el modelo IIT con el DynamicGatingModule activado.
El gate aprenderá a abrirse/cerrarse según el contenido del input.

Base: Golden Seed 2 + Data Seed 42 (misma fórmula del 54.35%)
Nuevo: use_dynamic_gate=True
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
# CONFIGURACIÓN
# ============================================================================

DATA_SEED = 42  # Super Golden data seed

CONFIG = {
    'vocab_size': 13,
    'hidden_dim': 64,
    'num_layers': 2,
    'num_heads': 4,
    'use_improved_memory': True,
    'use_improved_iit': True,
    'use_learnable_phi': True,
    'use_stochastic_exploration': True,
    'use_dynamic_gate': True,  # ¡NUEVO! Dynamic Gate activado
    'lambda_phi': 0.0
}

EPOCHS = 3000
BATCH_SIZE = 32
LR = 0.0005

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================================
# FUNCIÓN DE REPRODUCIBILIDAD
# ============================================================================

def set_all_seeds(seed):
    """Fija TODOS los seeds para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# VOCABULARIO Y DATOS
# ============================================================================

vocab = {
    'PAD': 0, '(': 1, ')': 2, '[': 3, ']': 4, 
    '{': 5, '}': 6, '<': 7, '>': 8, 
    'A': 9, 'B': 10, 'C': 11, 'EOS': 12
}
idx_to_char = {v: k for k, v in vocab.items()}


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


def get_batch(batch_size=32):
    """Genera batch."""
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
# EXPERIMENTO CON DYNAMIC GATE
# ============================================================================

def run_experiment():
    """Ejecuta el experimento con Dynamic Gate."""
    
    print(f"\n{'='*70}")
    print(f"🧠 ENTRENAMIENTO CON DYNAMIC GATE")
    print(f"   El gate aprenderá a reaccionar al contenido")
    print(f"   Device: {DEVICE}")
    print(f"{'='*70}")
    
    from infinito_v5_2_refactored import InfinitoV52Refactored
    
    # =========================================================================
    # CREAR MODELO IIT CON DYNAMIC GATE
    # =========================================================================
    print("\n[1/4] Creando modelo IIT con Dynamic Gate...")
    
    model_iit = InfinitoV52Refactored(
        vocab_size=CONFIG['vocab_size'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        use_improved_memory=CONFIG['use_improved_memory'],
        use_improved_iit=CONFIG['use_improved_iit'],
        use_learnable_phi=CONFIG['use_learnable_phi'],
        use_stochastic_exploration=CONFIG['use_stochastic_exploration'],
        use_dynamic_gate=CONFIG['use_dynamic_gate'],  # ¡ACTIVADO!
        lambda_phi=CONFIG['lambda_phi']
    ).to(DEVICE)
    
    # Cargar Golden Seed 2 (solo pesos compatibles)
    golden_path = os.path.join(os.path.dirname(__file__), 'models', 'golden_seed2_init.pt')
    golden_checkpoint = torch.load(golden_path, weights_only=False, map_location=DEVICE)
    
    # Cargar solo los pesos que existen (el dynamic_gate es nuevo)
    pretrained_dict = golden_checkpoint['model_state_dict']
    model_dict = model_iit.state_dict()
    
    # Filtrar pesos compatibles
    compatible_dict = {k: v for k, v in pretrained_dict.items() 
                       if k in model_dict and v.shape == model_dict[k].shape}
    
    # Actualizar y cargar
    model_dict.update(compatible_dict)
    model_iit.load_state_dict(model_dict)
    
    loaded_count = len(compatible_dict)
    total_count = len(model_dict)
    new_count = total_count - loaded_count
    
    print(f"   ✅ Cargados {loaded_count}/{total_count} parámetros de Golden Seed 2")
    print(f"   🆕 {new_count} parámetros nuevos (Dynamic Gate)")
    
    # Mostrar estado inicial del gate
    if hasattr(model_iit, 'dynamic_gate') and model_iit.dynamic_gate is not None:
        # El dynamic gate necesita input para calcular
        test_input = torch.randn(1, 10, CONFIG['hidden_dim']).to(DEVICE)
        with torch.no_grad():
            gate_prob, gate_logit = model_iit.dynamic_gate(test_input)
        print(f"   Dynamic Gate inicial (test): {gate_prob.mean().item()*100:.2f}%")
        print(f"   (logit: {gate_logit.mean().item():.2f})")
    else:
        print(f"   Static Memory Gate: {model_iit.memory_gate.item():.4f}")
    
    # =========================================================================
    # CREAR MODELO BASELINE
    # =========================================================================
    print("[2/4] Creando modelo Baseline (seed=42)...")
    set_all_seeds(DATA_SEED)
    
    model_base = InfinitoV52Refactored(
        vocab_size=CONFIG['vocab_size'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        use_improved_memory=False,
        use_improved_iit=False,
        use_learnable_phi=False,
        use_stochastic_exploration=False,
        use_dynamic_gate=False,  # Baseline sin dynamic gate
        seed=DATA_SEED
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
    # ENTRENAR
    # =========================================================================
    print(f"[4/4] Entrenando {EPOCHS} épocas con Dynamic Gate...")
    
    set_all_seeds(DATA_SEED)
    
    history_iit = []
    history_base = []
    gate_history = []  # Track gate evolution
    
    print_interval = 500
    
    for epoch in range(1, EPOCHS + 1):
        input_ids, target_ids = get_batch(BATCH_SIZE)
        input_ids = input_ids.to(DEVICE)
        target_ids = target_ids.to(DEVICE)
        
        # === ENTRENAR BASELINE ===
        model_base.train()
        opt_base.zero_grad()
        logits_base, _ = model_base(input_ids)
        min_len = min(logits_base.shape[1], target_ids.shape[1])
        loss_base = criterion(logits_base[:, :min_len, :].transpose(1, 2), target_ids[:, :min_len])
        loss_base.backward()
        opt_base.step()
        
        # === ENTRENAR IIT CON DYNAMIC GATE ===
        model_iit.train()
        opt_iit.zero_grad()
        logits_iit, metrics = model_iit(input_ids, return_metrics=True)
        loss_main = criterion(logits_iit[:, :min_len, :].transpose(1, 2), target_ids[:, :min_len])
        
        loss_phi = 0
        if metrics and 'delta_phi_loss' in metrics:
            loss_phi = metrics['delta_phi_loss']
        
        total_loss_iit = loss_main + (0.0 * loss_phi)
        total_loss_iit.backward()
        opt_iit.step()
        
        history_iit.append(loss_main.item())
        history_base.append(loss_base.item())
        
        # Track gate value
        if metrics and 'gate_value' in metrics:
            gate_history.append(metrics['gate_value'])
        
        if epoch % print_interval == 0:
            avg_iit = np.mean(history_iit[-100:])
            avg_base = np.mean(history_base[-100:])
            imp = (avg_base - avg_iit) / avg_base * 100
            
            # Get current gate value
            gate_str = "N/A"
            if gate_history:
                gate_str = f"{gate_history[-1]*100:.2f}%"
            
            print(f"   Época {epoch}: IIT={avg_iit:.4f}, Base={avg_base:.4f}, Mejora={imp:+.1f}%, Gate={gate_str}")
    
    # =========================================================================
    # RESULTADOS FINALES
    # =========================================================================
    final_iit = history_iit[-1]
    final_base = history_base[-1]
    improvement = ((final_base - final_iit) / final_base) * 100
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS FINALES (Dynamic Gate)")
    print(f"{'='*70}")
    print(f"   IIT Loss: {final_iit:.5f}")
    print(f"   Baseline Loss: {final_base:.5f}")
    print(f"   MEJORA: {improvement:.2f}%")
    
    if gate_history:
        print(f"\n📈 Evolución del Dynamic Gate:")
        print(f"   Inicio: {gate_history[0]*100:.2f}%")
        print(f"   Medio:  {gate_history[len(gate_history)//2]*100:.2f}%")
        print(f"   Final:  {gate_history[-1]*100:.2f}%")
        print(f"   Min:    {min(gate_history)*100:.2f}%")
        print(f"   Max:    {max(gate_history)*100:.2f}%")
    
    print(f"\n🎯 Comparación:")
    print(f"   Static Gate (original): 54.35%")
    print(f"   Dynamic Gate (nuevo):   {improvement:.2f}%")
    
    # Guardar checkpoint
    save_path = os.path.join(os.path.dirname(__file__), 'models', 'dynamic_gate_trained.pt')
    torch.save({
        'model_state_dict': model_iit.state_dict(),
        'improvement': improvement,
        'config': CONFIG,
        'epochs': EPOCHS,
        'gate_history': gate_history
    }, save_path)
    print(f"\n💾 Modelo guardado en: {save_path}")
    
    return improvement, final_iit, final_base, gate_history


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    start_time = datetime.now()
    
    improvement, loss_iit, loss_base, gate_history = run_experiment()
    
    elapsed = datetime.now() - start_time
    print(f"\n⏱️ Tiempo: {elapsed}")
    print(f"{'='*70}")
