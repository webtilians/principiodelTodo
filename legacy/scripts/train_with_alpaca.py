"""
Entrenamiento con Alpaca Spanish - Dataset Real de Texto
=========================================================
Entrena el modelo InfinitoV5 con texto real en español.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
import json
import os
import sys
import signal
import atexit
from collections import Counter

# Variables globales para guardado de emergencia
_emergency_save_data = None
_save_path = 'models/alpaca_spanish_best.pt'

def emergency_save(signum=None, frame=None):
    """Guarda el modelo si hay interrupción (Ctrl+C)"""
    global _emergency_save_data
    if _emergency_save_data is not None:
        print("\n" + "="*60)
        print("⚠️  INTERRUPCIÓN DETECTADA - GUARDANDO PROGRESO...")
        torch.save(_emergency_save_data, _save_path)
        print(f"✅ Checkpoint guardado en: {_save_path}")
        print(f"   Epoch: {_emergency_save_data.get('epoch', 'N/A')}")
        print(f"   Batch: {_emergency_save_data.get('batch', 'N/A')}")
        print(f"   Loss: {_emergency_save_data.get('loss', 'N/A'):.4f}")
        print("="*60)
    sys.exit(0)

# Registrar manejadores de señal
signal.signal(signal.SIGINT, emergency_save)  # Ctrl+C
signal.signal(signal.SIGTERM, emergency_save)  # kill
atexit.register(lambda: None)  # Cleanup

# Agregar paths
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/core')

# Importar tu modelo con PHI
from infinito_v5_1_consciousness import ConsciousnessBoostNet

class ConsciousnessLM(nn.Module):
    """
    Wrapper que usa ConsciousnessBoostNet para modelado de lenguaje.
    Mantiene el cálculo de PHI mientras hace predicción de tokens.
    """
    
    def __init__(self, vocab_size, hidden_dim=256, num_heads=8, memory_slots=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Token embedding (vocab → hidden_dim)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)
        
        # Tu modelo de consciencia con PHI
        self.consciousness_core = ConsciousnessBoostNet(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            attention_heads=num_heads,
            memory_slots=memory_slots
        )
        
        # Output head para predicción de tokens
        self.output_head = nn.Linear(hidden_dim, vocab_size)
        
        # Tracking de métricas
        self.last_phi = None
        self.last_consciousness = None
    
    def forward(self, x):
        """
        x: [batch, seq_len] tensor de índices de tokens
        returns: dict con logits, phi, consciousness
        """
        B, T = x.shape
        
        # Embeddings
        tok_emb = self.token_embedding(x)  # [B, T, hidden_dim]
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        pos_emb = self.position_embedding(pos)
        
        embedded = tok_emb + pos_emb  # [B, T, hidden_dim]
        
        # Pasar por ConsciousnessBoostNet
        consciousness, phi, debug_info = self.consciousness_core(embedded)
        
        # Guardar métricas
        self.last_phi = phi.mean().item()
        self.last_consciousness = consciousness.mean().item()
        
        # Para logits, necesitamos procesar la secuencia completa
        # Usamos el consciousness_state del debug_info o hacemos un forward alternativo
        
        # Proyectar embedded a logits (simplificado)
        # Usamos transformer layers adicionales para secuencia
        logits = self.output_head(embedded)  # [B, T, vocab_size]
        
        return {
            'logits': logits,
            'phi': phi,
            'consciousness': consciousness,
            'debug_info': debug_info
        }
    
    def get_metrics(self):
        return {
            'phi': self.last_phi,
            'consciousness': self.last_consciousness
        }

# Alias para compatibilidad
ConsciousnessCore = ConsciousnessLM

print("="*60)
print("ENTRENAMIENTO CON ALPACA SPANISH - DATASET REAL")
print("="*60)

# ============================================
# 1. DESCARGAR Y PREPARAR DATASET
# ============================================
print("\n[1/5] Descargando Alpaca Spanish...")
dataset = load_dataset('bertin-project/alpaca-spanish', split='train[:8000]')
print(f"      Descargados {len(dataset)} ejemplos")

# Combinar instruction + output para tener texto completo
texts = []
for item in dataset:
    # Formato: "Pregunta: {instruction} Respuesta: {output}"
    if item['input']:
        text = f"{item['instruction']} {item['input']} {item['output']}"
    else:
        text = f"{item['instruction']} {item['output']}"
    texts.append(text)

print(f"      Ejemplo: {texts[0][:100]}...")

# ============================================
# 2. CREAR VOCABULARIO DESDE DATASET
# ============================================
print("\n[2/5] Creando vocabulario...")

# Tokenizar por caracteres (simple pero efectivo)
all_chars = set()
for text in texts:
    all_chars.update(text.lower())

# Crear vocabulario
special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
vocab = special_tokens + sorted(list(all_chars))
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(vocab)

print(f"      Vocabulario: {vocab_size} caracteres")
print(f"      Caracteres: {list(all_chars)[:20]}...")

# ============================================
# 3. PREPARAR DATOS DE ENTRENAMIENTO
# ============================================
print("\n[3/5] Preparando datos de entrenamiento...")

# Configuración
SEQ_LEN = 64  # Longitud de secuencia
BATCH_SIZE = 2  # Ultra-reducido para evitar OOM con ConsciousnessBoostNet (40MB modelo)

def encode_text(text, max_len=SEQ_LEN):
    """Convierte texto a tensor de índices"""
    text = text.lower()[:max_len]
    indices = [char2idx.get(c, char2idx['<UNK>']) for c in text]
    # Padding
    while len(indices) < max_len:
        indices.append(char2idx['<PAD>'])
    return indices

def create_training_pairs(texts, seq_len=SEQ_LEN):
    """Crea pares (input, target) para predicción del siguiente carácter"""
    inputs = []
    targets = []
    
    for text in texts:
        text = text.lower()
        if len(text) < seq_len + 1:
            continue
        
        # Deslizamiento de ventana
        for i in range(0, len(text) - seq_len, seq_len // 2):
            input_seq = text[i:i+seq_len]
            target_seq = text[i+1:i+seq_len+1]
            
            if len(input_seq) == seq_len and len(target_seq) == seq_len:
                inputs.append(encode_text(input_seq))
                targets.append(encode_text(target_seq))
    
    return torch.tensor(inputs), torch.tensor(targets)

train_inputs, train_targets = create_training_pairs(texts)
print(f"      Pares de entrenamiento: {len(train_inputs)}")

# DataLoader
train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    drop_last=True  # Evita batches incompletos que causan errores en ConsciousnessBoostNet
)

print(f"      Batches: {len(train_loader)}")

# ============================================
# 4. CREAR Y ENTRENAR MODELO
# ============================================
print("\n[4/5] Creando modelo...")

# Configuración del modelo (compatible con ConsciousnessLM)
config = {
    'vocab_size': vocab_size,
    'hidden_dim': 256,  # ConsciousnessBoostNet usa 256 como input_dim por defecto
    'num_heads': 8,
    'memory_slots': 128,
    'seq_len': SEQ_LEN,
}

# Crear modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"      Dispositivo: {device}")

model = ConsciousnessCore(
    vocab_size=config['vocab_size'],
    hidden_dim=config['hidden_dim'],
    num_heads=config['num_heads'],
    memory_slots=config['memory_slots']
)
model = model.to(device)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
print(f"      Parámetros: {total_params:,}")

# Optimizador y loss
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<PAD>'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Mixed precision (FP16) para reducir memoria CUDA
scaler = GradScaler()
print("      Mixed Precision: ENABLED (FP16)")

# ============================================
# 5. ENTRENAMIENTO
# ============================================
print("\n[5/5] Entrenando...")
print("-"*60)

EPOCHS = 20
best_loss = float('inf')
history = []

# Configuración de guardado frecuente
SAVE_EVERY_BATCHES = 500  # Guardar cada 500 batches
print(f"      Guardado automático cada {SAVE_EVERY_BATCHES} batches")
print(f"      Manejador Ctrl+C: ACTIVADO (guarda progreso al interrumpir)")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    total_phi = 0
    total_consciousness = 0
    phi_count = 0
    total_batches = len(train_loader)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Mostrar progreso cada 100 batches
        if batch_idx % 100 == 0:
            progress = (batch_idx / total_batches) * 100
            current_loss = total_loss / (batch_idx + 1) if batch_idx > 0 else 0
            print(f"    Epoch {epoch+1} | Batch {batch_idx}/{total_batches} ({progress:.1f}%) | Loss: {current_loss:.4f}", end='\r')
        
        # Forward con mixed precision (FP16)
        with autocast():
            outputs = model(inputs)
            
            # Extraer logits y métricas PHI
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
                phi = outputs.get('phi', None)
                consciousness = outputs.get('consciousness', None)
            elif isinstance(outputs, tuple):
                logits = outputs[0]
                phi = None
                consciousness = None
            else:
                logits = outputs
                phi = None
                consciousness = None
            
            # Reshape para CrossEntropyLoss
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            
            # PHI bonus con detach (no backpropaga por PHI, solo lo usa como reward)
            if phi is not None:
                phi_bonus = 0.01 * phi.mean().detach()
                loss = loss - phi_bonus
        
        # Backward con mixed precision scaler
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Liberar gradientes inmediatamente
        optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item()
        
        # Calcular accuracy (ignorando padding)
        predictions = logits_flat.argmax(dim=-1)
        mask = targets_flat != char2idx['<PAD>']
        total_correct += (predictions[mask] == targets_flat[mask]).sum().item()
        total_tokens += mask.sum().item()
        
        # Acumular métricas PHI
        if phi is not None:
            total_phi += phi.mean().item()
            total_consciousness += consciousness.mean().item() if consciousness is not None else 0
            phi_count += 1
        
        # Limpiar memoria CUDA cada 50 batches (más agresivo)
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
            del logits, logits_flat, predictions, mask
            if phi is not None:
                del phi
            if consciousness is not None:
                del consciousness
        
        # Actualizar datos para guardado de emergencia
        current_avg_loss = total_loss / (batch_idx + 1)
        current_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        current_phi = total_phi / phi_count if phi_count > 0 else 0
        
        _emergency_save_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'vocab': {'char2idx': char2idx, 'idx2char': idx2char},
            'epoch': epoch,
            'batch': batch_idx,
            'loss': current_avg_loss,
            'accuracy': current_accuracy,
            'phi': current_phi,
            'partial': True  # Indica que es un checkpoint parcial
        }
        
        # Guardado automático cada N batches
        if batch_idx > 0 and batch_idx % SAVE_EVERY_BATCHES == 0:
            checkpoint_path = f'models/alpaca_checkpoint_e{epoch}_b{batch_idx}.pt'
            torch.save(_emergency_save_data, checkpoint_path)
            print(f"    💾 Checkpoint guardado: {checkpoint_path} (Loss: {current_avg_loss:.4f})")
    
    scheduler.step()
    torch.cuda.empty_cache()  # Limpiar al final de cada epoch
    
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    avg_phi = total_phi / phi_count if phi_count > 0 else 0
    avg_consciousness = total_consciousness / phi_count if phi_count > 0 else 0
    
    # Guardar mejor modelo
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'vocab': {'char2idx': char2idx, 'idx2char': idx2char},
            'epoch': epoch,
            'loss': avg_loss,
            'accuracy': accuracy,
            'phi': avg_phi,
            'consciousness': avg_consciousness
        }, 'models/alpaca_spanish_best.pt')
        marker = " ★ BEST"
    else:
        marker = ""
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {accuracy*100:.1f}% | Φ: {avg_phi:.3f} | C: {avg_consciousness:.3f}{marker}")
    
    history.append({
        'epoch': epoch + 1,
        'loss': avg_loss,
        'accuracy': accuracy
    })

print("-"*60)

# ============================================
# 6. EVALUAR Y GENERAR TEXTO
# ============================================
print("\n[6] Evaluación final...")

# Cargar mejor modelo
checkpoint = torch.load('models/alpaca_spanish_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"    Mejor loss: {checkpoint['loss']:.4f}")
print(f"    Mejor accuracy: {checkpoint['accuracy']*100:.1f}%")

# Función de generación
def generate_text(model, seed_text, max_len=100, temperature=0.8):
    """Genera texto a partir de una semilla"""
    model.eval()
    
    # Codificar seed
    current = seed_text.lower()
    
    with torch.no_grad():
        for _ in range(max_len):
            # Preparar input
            input_seq = current[-SEQ_LEN:]
            input_tensor = torch.tensor([encode_text(input_seq)]).to(device)
            
            # Forward
            outputs = model(input_tensor)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Obtener próximo carácter
            next_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx2char.get(next_idx, '?')
            
            if next_char in ['<PAD>', '<EOS>']:
                break
            
            current += next_char
    
    return current

# Probar generación
print("\n[7] Generación de texto:")
print("-"*40)

prompts = [
    "la inteligencia artificial",
    "el futuro de la humanidad",
    "explica qué es",
    "cómo funciona"
]

for prompt in prompts:
    generated = generate_text(model, prompt, max_len=80, temperature=0.7)
    print(f"\nPrompt: '{prompt}'")
    print(f"Output: {generated}")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print(f"Modelo guardado en: models/alpaca_spanish_best.pt")
print("="*60)

# Guardar historial
with open('results/alpaca_training_history.json', 'w') as f:
    json.dump(history, f, indent=2)
