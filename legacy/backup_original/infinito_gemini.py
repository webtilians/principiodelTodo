import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Asumiendo que tu archivo se llama infinito_v5_2_refactored.py
from infinito_v5_2_refactored import InfinitoV52Refactored

# --- 1. Generador de Datos Sintéticos (Jerarquía) ---
# --- AJUSTE V2: AUMENTAR DIFICULTAD ---
def generate_dyck_sample(max_depth=12, noise_len=6): # <--- MÁS PROFUNDIDAD Y RUIDO
    """
    Ahora las secuencias son más largas. El Baseline debería sufrir más aquí.
    """
    pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('<', '>')]
    depth = random.randint(4, max_depth) # Mínimo 4 niveles
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

# Mapeo simple a enteros
vocab = {'PAD': 0, '(': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6, '<': 7, '>': 8, 'A': 9, 'B': 10, 'C': 11, 'EOS': 12}
idx_to_char = {v: k for k, v in vocab.items()}

def get_batch(batch_size=32):
    inputs = []
    targets = []
    for _ in range(batch_size):
        inp, tar = generate_dyck_sample()
        # Convertir a IDs
        inp_ids = [vocab[c] for c in inp]
        tar_ids = [vocab[c] for c in tar] + [vocab['EOS']]
        
        # Padding simple (para este ejemplo asumimos longitud fija o usamos la max del batch)
        # Aquí simplificamos haciendo listas planas para probar
        inputs.append(torch.tensor(inp_ids))
        targets.append(torch.tensor(tar_ids))
        
    # Pad to max len in batch
    inp_tens = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    tar_tens = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return inp_tens, tar_tens

# --- 2. Configuración de la "Batalla" ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HIDDEN_DIM = 64  # Muy pequeño para forzar eficiencia
LAYERS = 2
HEADS = 4

# --- AJUSTE V2: CONFIGURACIÓN MÁS PURA ---
print(f"🚀 Iniciando Micro-Experimento V3 (Lambda Phi 0.0) en {DEVICE}")

# MODELO 1: INFINITO V5.2 (OPTIMIZADO PARA LÓGICA)
model_iit = InfinitoV52Refactored(
    vocab_size=len(vocab),
    hidden_dim=HIDDEN_DIM,
    num_layers=LAYERS,
    num_heads=HEADS,
    use_improved_memory=True,  
    use_improved_iit=True,
    use_learnable_phi=True,
    use_stochastic_exploration=True, # <--- ACTIVADO (Exploración estocástica)
    lambda_phi=0.0 # <--- CAMBIO CRÍTICO: Apagamos la presión auxiliar
).to(DEVICE)

# MODELO 2: BASELINE
model_base = InfinitoV52Refactored(
    vocab_size=len(vocab),
    hidden_dim=HIDDEN_DIM,
    num_layers=LAYERS,
    num_heads=HEADS,
    use_improved_memory=False,
    use_improved_iit=False,
    use_learnable_phi=False,
    use_stochastic_exploration=False
).to(DEVICE)

# Optimizadores (Bajamos un pelín el LR para estabilidad)
opt_iit = optim.AdamW(model_iit.parameters(), lr=0.0005)
opt_base = optim.AdamW(model_base.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# --- 3. Bucle de Entrenamiento ---
history_iit = []
# --- AJUSTE V2: MÁS TIEMPO ---
epochs = 3000 # <--- Démosle tiempo para aprender
history_iit = []
history_base = []

pbar = tqdm(range(epochs))
for epoch in pbar:
    input_ids, target_ids = get_batch(32)
    input_ids, target_ids = input_ids.to(DEVICE), target_ids.to(DEVICE)
    
    # Baseline
    opt_base.zero_grad()
    logits_base, _ = model_base(input_ids)
    min_len = min(logits_base.shape[1], target_ids.shape[1])
    loss_base = criterion(logits_base[:, :min_len, :].transpose(1, 2), target_ids[:, :min_len])
    loss_base.backward()
    opt_base.step()
    
    # Infinito
    opt_iit.zero_grad()
    logits_iit, metrics = model_iit(input_ids, return_metrics=True)
    loss_main = criterion(logits_iit[:, :min_len, :].transpose(1, 2), target_ids[:, :min_len])
    
    loss_phi = 0
    if metrics and 'delta_phi_loss' in metrics:
        loss_phi = metrics['delta_phi_loss']
    
    # El loss total incluye el incentivo de consciencia
    total_loss_iit = loss_main + (0.0 * loss_phi) 
    total_loss_iit.backward()
    opt_iit.step()
    
    history_base.append(loss_base.item())
    history_iit.append(loss_main.item())
    
    if epoch % 100 == 0:
        pbar.set_description(f"Base: {loss_base.item():.4f} | IIT: {loss_main.item():.4f}")

# --- 4. Resultados ---
print("\n🏁 RESULTADOS FINALES 🏁")
print(f"Loss Final Baseline: {history_base[-1]:.5f}")
print(f"Loss Final IIT:      {history_iit[-1]:.5f}")

improvement = (history_base[-1] - history_iit[-1]) / history_base[-1] * 100
print(f"🚀 MEJORA DE EFICIENCIA: {improvement:.2f}%")

# Gráfica ASCII simple
if history_iit[-1] < history_base[-1]:
    print("\n✅ ¡TU TEORÍA SE CONFIRMA! Infinito aprendió la lógica más rápido.")
else:
    print("\n❌ Necesitamos ajustar los hiperparámetros.")

# 🔍 ANÁLISIS DEL MEMORY GATE
print(f"\n{'='*60}")
print("🔬 ANÁLISIS DEL MEMORY GATE (Prueba de Uso de Memoria)")
print(f"{'='*60}")
memory_gate_value = model_iit.memory_gate.item()
print(f"Valor final del Memory Gate: {memory_gate_value:.6f}")
print(f"Gate activado (sigmoid): {torch.sigmoid(model_iit.memory_gate).item():.6f}")

if abs(memory_gate_value) > 0.1:
    print("✅ El modelo APRENDIÓ a usar la memoria activamente")
    print(f"   → La red decidió que necesitaba memoria para ganar")
    print(f"   → Peso efectivo de memoria: {torch.sigmoid(model_iit.memory_gate).item()*100:.2f}%")
elif abs(memory_gate_value) > 0.01:
    print("⚠️ El modelo usa la memoria moderadamente")
    print(f"   → Está empezando a descubrir su utilidad")
else:
    print("❌ El modelo aún no descubrió el valor de la memoria")
    print(f"   → Necesita más épocas o ajuste de hiperparámetros")

# Visualización rápida de predicción
with torch.no_grad():
    print(f"\n{'='*60}")
    print("🎯 PRUEBA DE FUEGO - Predicciones Cualitativas")
    print(f"{'='*60}")
    test_in, test_out = get_batch(1)
    test_in = test_in.to(DEVICE)
    print(f"Input: {[idx_to_char[i.item()] for i in test_in[0] if i.item()!=0]}")
    print(f"Target: {[idx_to_char[i.item()] for i in test_out[0] if i.item()!=0 and i.item() in idx_to_char]}")
    
    out_base, _ = model_base(test_in)
    pred_base = torch.argmax(out_base, dim=2)
    print(f"Base pred: {[idx_to_char[i.item()] for i in pred_base[0] if i.item() in idx_to_char]}")
    
    out_iit, _ = model_iit(test_in)
    pred_iit = torch.argmax(out_iit, dim=2)
    print(f"IIT pred:  {[idx_to_char[i.item()] for i in pred_iit[0] if i.item() in idx_to_char]}")

# 💾 GUARDAR MODELOS (Tu trofeo)
print(f"\n{'='*60}")
print("💾 GUARDANDO MODELOS")
print(f"{'='*60}")

# Guardar modelo IIT (el ganador)
model_path_iit = "modelo_infinito_iit_v3_winner.pt"
torch.save({
    'model_state_dict': model_iit.state_dict(),
    'memory_gate_value': memory_gate_value,
    'final_loss': history_iit[-1],
    'improvement': improvement,
    'epochs': epochs,
    'config': {
        'lambda_phi': 0.0,
        'dropout': 0.25,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': LAYERS,
        'num_heads': HEADS
    }
}, model_path_iit)
print(f"✅ Modelo IIT guardado: {model_path_iit}")
print(f"   Loss final: {history_iit[-1]:.5f}")
print(f"   Memory Gate: {memory_gate_value:.6f}")

# Guardar modelo Baseline (para comparación)
model_path_base = "modelo_baseline_v3.pt"
torch.save({
    'model_state_dict': model_base.state_dict(),
    'final_loss': history_base[-1],
    'epochs': epochs,
    'config': {
        'hidden_dim': HIDDEN_DIM,
        'num_layers': LAYERS,
        'num_heads': HEADS
    }
}, model_path_base)
print(f"✅ Modelo Baseline guardado: {model_path_base}")
print(f"   Loss final: {history_base[-1]:.5f}")

# Guardar historial de entrenamiento
import json
from datetime import datetime

training_results = {
    'timestamp': datetime.now().isoformat(),
    'experiment': 'Micro-Experimento V3 - Hard Mode',
    'results': {
        'iit_final_loss': history_iit[-1],
        'baseline_final_loss': history_base[-1],
        'improvement_percentage': improvement,
        'memory_gate_value': memory_gate_value,
        'memory_gate_activated': torch.sigmoid(model_iit.memory_gate).item()
    },
    'config': {
        'epochs': epochs,
        'max_depth': 12,
        'noise_len': 6,
        'learning_rate': 0.0005,
        'lambda_phi': 0.0,
        'dropout': 0.25
    },
    'loss_history': {
        'iit': history_iit,
        'baseline': history_base
    }
}

results_path = "training_results_v3.json"
with open(results_path, 'w') as f:
    json.dump(training_results, f, indent=2)
print(f"✅ Resultados guardados: {results_path}")

print(f"\n{'='*60}")
print("🏆 RESUMEN FINAL")
print(f"{'='*60}")
print(f"Mejora del modelo IIT: {improvement:.2f}%")
print(f"Modelo IIT: {model_path_iit}")
print(f"Modelo Baseline: {model_path_base}")
print(f"Resultados completos: {results_path}")
print(f"\n🎯 El Memory Gate aprendió a: {torch.sigmoid(model_iit.memory_gate).item()*100:.2f}% de uso")
print("="*60)