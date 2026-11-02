#!/usr/bin/env python3
"""
🧠 INFINITO: GPT-2 + PHI Observer + IIT-Guided Memory
======================================================

Arquitectura que diferencia INFINITO de cualquier GPT vanilla:

1. GPT-2 genera texto normalmente (LoRA fine-tuning)
2. PHI Observer MIDE integración de información
3. IIT-Guided Memory ALMACENA estados con PHI alto (aprende durante training)

¿QUÉ NOS DIFERENCIA?
- GPT normal: Caja negra → "funciona o no"
- INFINITO: Caja con sensores + memoria inteligente

COMPONENTES:
1. GPT-2 + LoRA: Generación de texto (backbone)
2. PHI Observer: Mide 4 componentes IIT
3. IIT-Guided Memory: Almacena estados priorizados por PHI (256 slots)

La memoria APRENDE durante el entrenamiento qué patrones tienen PHI alto.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
import json
import os
import sys
import signal
from datetime import datetime
from collections import deque

# Paths
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/core')

# Importar IIT-Guided Memory
from iit_guided_memory import IITGuidedMemory

# =============================================================================
# PHI OBSERVER - No modifica la generación, solo observa
# =============================================================================

class PhiObserver(nn.Module):
    """
    🔬 Observador de PHI - Mide integración sin afectar generación.
    
    4 Componentes IIT:
    1. Temporal Coherence: Consistencia entre posiciones consecutivas
    2. Integration Strength: Correlación cruzada entre primera/segunda mitad
    3. Complexity: Varianza normalizada de activaciones
    4. Attention Diversity: Entropía de distribución de atención
    """
    
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Proyector para análisis (no afecta generación)
        self.phi_projector = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Historial para análisis temporal
        self.phi_history = deque(maxlen=1000)
        self.pattern_memory = {}  # Guarda patrones PHI por tipo de prompt
        
    def calculate_temporal_coherence(self, hidden_state):
        """Mide qué tan consistente es la secuencia temporalmente."""
        if hidden_state.size(1) < 2:
            return torch.ones(hidden_state.size(0), device=hidden_state.device) * 0.5
        
        # Correlación entre posiciones consecutivas
        h1 = hidden_state[:, :-1, :]  # [B, T-1, D]
        h2 = hidden_state[:, 1:, :]   # [B, T-1, D]
        
        # Normalizar
        h1_norm = h1 / (h1.norm(dim=-1, keepdim=True) + 1e-8)
        h2_norm = h2 / (h2.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Correlación promedio
        correlation = (h1_norm * h2_norm).sum(dim=-1).mean(dim=-1)
        
        return (correlation + 1) / 2  # Normalizar a [0, 1]
    
    def calculate_integration_strength(self, hidden_state):
        """Mide integración entre primera y segunda mitad de la secuencia."""
        seq_len = hidden_state.size(1)
        if seq_len < 4:
            return torch.ones(hidden_state.size(0), device=hidden_state.device) * 0.5
        
        half = seq_len // 2
        first_half = hidden_state[:, :half, :].mean(dim=1)
        second_half = hidden_state[:, half:2*half, :].mean(dim=1)
        
        # Normalizar
        f_norm = first_half / (first_half.norm(dim=-1, keepdim=True) + 1e-8)
        s_norm = second_half / (second_half.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Similitud como proxy de integración
        integration = (f_norm * s_norm).sum(dim=-1)
        
        return (integration + 1) / 2
    
    def calculate_complexity(self, hidden_state):
        """Mide complejidad como varianza normalizada."""
        variance = hidden_state.var(dim=[1, 2])
        # Normalizar con sigmoid para rango [0, 1]
        return torch.sigmoid(variance - 0.5)
    
    def calculate_attention_diversity(self, attention_weights):
        """Mide diversidad de atención usando entropía."""
        if attention_weights is None:
            return None
        
        # Verificar si es una tupla con Nones (Flash Attention no devuelve pesos)
        if isinstance(attention_weights, tuple):
            if all(a is None for a in attention_weights):
                return None
            # Tomar el último attention que no sea None
            for attn in reversed(attention_weights):
                if attn is not None:
                    attention_weights = attn
                    break
            else:
                return None
        
        # attention_weights: [B, heads, T, T]
        # IMPORTANTE: Convertir a float32 para evitar underflow con FP16
        attn_float = attention_weights.float()
        
        # Calcular entropía por cabeza y promediar
        attn_flat = attn_float.mean(dim=1)  # [B, T, T]
        
        # Clamp valores muy pequeños para evitar log(0)
        attn_clamped = torch.clamp(attn_flat, min=1e-8)
        
        # Entropía de Shannon
        entropy = -torch.sum(attn_clamped * torch.log(attn_clamped), dim=-1)
        max_entropy = torch.log(torch.tensor(attn_flat.size(-1), dtype=torch.float32, device=attn_flat.device))
        
        # Normalizar
        diversity = entropy.mean(dim=-1) / max_entropy
        
        return diversity
    
    def forward(self, hidden_state, attention_weights=None):
        """
        Calcula PHI completo sin afectar gradientes de generación.
        
        Returns:
            dict con todas las métricas PHI
        """
        with torch.no_grad():  # NO afecta backprop de generación
            # Proyectar para análisis
            projected = self.phi_projector(hidden_state)
            
            # Calcular 4 componentes
            temporal = self.calculate_temporal_coherence(projected)
            integration = self.calculate_integration_strength(projected)
            complexity = self.calculate_complexity(projected)
            
            if attention_weights is not None:
                attention_div = self.calculate_attention_diversity(attention_weights)
            else:
                attention_div = torch.ones_like(temporal) * 0.5
            
            # PHI combinado (pesos estándar IIT)
            phi = (
                0.30 * temporal +
                0.30 * integration +
                0.20 * complexity +
                0.20 * attention_div
            )
            
            # Escalar a rango típico [0, 10]
            phi_scaled = phi * 10.0
            
            return {
                'phi': phi_scaled,
                'temporal_coherence': temporal,
                'integration_strength': integration,
                'complexity': complexity,
                'attention_diversity': attention_div,
                'raw_components': {
                    'temporal': temporal.mean().item(),
                    'integration': integration.mean().item(),
                    'complexity': complexity.mean().item(),
                    'attention': attention_div.mean().item() if attention_div is not None else 0.5
                }
            }
    
    def log_phi(self, phi_data, prompt_type="general"):
        """Guarda datos PHI para análisis posterior."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'prompt_type': prompt_type,
            'phi': phi_data['phi'].mean().item(),
            'components': phi_data['raw_components']
        }
        self.phi_history.append(entry)
        
        # Actualizar estadísticas por tipo de prompt
        if prompt_type not in self.pattern_memory:
            self.pattern_memory[prompt_type] = []
        self.pattern_memory[prompt_type].append(entry['phi'])
    
    def get_analysis(self):
        """Retorna análisis de patrones PHI."""
        analysis = {}
        for prompt_type, phis in self.pattern_memory.items():
            if len(phis) > 0:
                analysis[prompt_type] = {
                    'mean_phi': sum(phis) / len(phis),
                    'max_phi': max(phis),
                    'min_phi': min(phis),
                    'samples': len(phis)
                }
        return analysis


# =============================================================================
# MODELO PRINCIPAL: GPT-2 + Observer + IIT Memory
# =============================================================================

class InfinitoGPT2WithObserver(nn.Module):
    """
    GPT-2 con LoRA + PHI Observer + IIT-Guided Memory.
    
    Componentes:
    - GPT-2 + LoRA: Backbone de generación
    - PHI Observer: Mide integración de información
    - IIT Memory: Almacena estados con PHI alto (aprende durante training)
    """
    
    def __init__(self, use_lora=True, lora_r=8, lora_alpha=32, memory_slots=256):
        super().__init__()
        
        print("="*60)
        print("🧠 INFINITO: GPT-2 Spanish + PHI Observer + IIT Memory")
        print("="*60)
        
        # 1. Cargar GPT-2 en español (con safetensors para evitar CVE-2025-32434)
        # IMPORTANTE: Usamos attn_implementation="eager" para obtener attention weights reales
        print("📦 Cargando GPT-2 Spanish (datificate/gpt2-small-spanish)...")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(
            'datificate/gpt2-small-spanish',
            use_safetensors=True,  # Evita vulnerabilidad torch.load
            attn_implementation="eager"  # CRÍTICO: Para obtener attention weights
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained('datificate/gpt2-small-spanish')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.hidden_dim = 768  # GPT-2 small
        
        # 2. LoRA adapters
        if use_lora:
            print(f"🔧 Aplicando LoRA (r={lora_r}, alpha={lora_alpha})...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj"],
            )
            self.gpt2 = get_peft_model(self.gpt2, lora_config)
            self.gpt2.print_trainable_parameters()
        
        # 3. PHI Observer
        print("🔬 Inicializando PHI Observer...")
        self.phi_observer = PhiObserver(hidden_dim=self.hidden_dim)
        
        # 4. IIT-Guided Memory (NUEVA!)
        print(f"🧠 Inicializando IIT-Guided Memory ({memory_slots} slots)...")
        self.iit_memory = IITGuidedMemory(
            memory_slots=memory_slots,
            hidden_dim=self.hidden_dim,
            use_phi_priority=True,      # Prioridad por PHI
            use_recency_boost=True,     # Boost para items recientes
            alpha=0.8,                  # 80% PHI, 20% attention
            learnable_threshold=True,   # Threshold aprendible
            initial_threshold=0.3       # Threshold bajo para permitir escrituras iniciales
        )
        
        # 5. Proyección para combinar memoria con hidden states
        self.memory_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # 6. Estadísticas
        self.training_phi_log = []
        self.memory_stats_log = []
        
        print("="*60)
        print("✅ Modelo listo")
        print("   - Generación: GPT-2 + LoRA")
        print("   - Observación: PHI Observer (4 componentes IIT)")
        print(f"   - Memoria: IIT-Guided ({memory_slots} slots, threshold aprendible)")
        print("="*60)
    
    def forward(self, input_ids, labels=None, return_phi=True, use_memory=True):
        """
        Forward pass con observación PHI y memoria IIT.
        
        Returns:
            outputs, phi_metrics (si return_phi=True)
        """
        batch_size = input_ids.size(0)
        
        # GPT-2 forward con hidden states y attention
        outputs = self.gpt2(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )
        
        phi_metrics = None
        memory_info = None
        
        if return_phi:
            # Obtener hidden states y attention
            last_hidden = outputs.hidden_states[-1]  # [B, T, 768]
            last_attention = outputs.attentions[-1] if outputs.attentions else None
            
            # 1. Calcular PHI
            phi_metrics = self.phi_observer(last_hidden, last_attention)
            
            # 2. Interactuar con IIT Memory
            if use_memory:
                # Query: promedio del hidden state
                memory_query = last_hidden.mean(dim=1)  # [B, 768]
                memory_content = memory_query.detach()  # Lo que vamos a almacenar (detach para evitar memory leak)
                
                # PHI value para priorización (detach porque es solo para decisión)
                phi_value = phi_metrics['phi'].detach()  # [B]
                
                # Leer de memoria (top-5 más relevantes por PHI)
                read_result = self.iit_memory.read(
                    query=memory_query.detach(),
                    top_k=5,
                    phi_guided=True
                )
                
                # Escribir en memoria (solo si PHI es suficiente)
                write_info = self.iit_memory.write(
                    query=memory_query.detach(),
                    content=memory_content,
                    phi_value=phi_value,
                    attention_score=None
                )
                
                # AUTO-AJUSTE del threshold basado en utilización
                # Si la memoria está muy vacía, bajar threshold
                # Si está muy llena, subir threshold
                current_util = write_info['utilization'].item()
                with torch.no_grad():
                    if current_util < 0.3:  # Memoria muy vacía → bajar threshold
                        self.iit_memory.threshold_logit.data -= 0.02  # Bajar más rápido
                    elif current_util > 0.9:  # Memoria muy llena → subir threshold
                        self.iit_memory.threshold_logit.data += 0.01
                    # Clamp threshold a rango razonable [0.05, 0.7] en escala priority
                    # (reducido porque priority típica es ~0.48)
                    device = self.iit_memory.threshold_logit.device
                    self.iit_memory.threshold_logit.data.clamp_(
                        min=torch.tensor(0.05, device=device).log(),  # Mínimo 0.05
                        max=torch.tensor(0.7, device=device).log()    # Máximo 0.7
                    )
                
                # Estadísticas de memoria
                memory_stats = self.iit_memory.get_statistics()
                memory_info = {
                    'utilization': memory_stats.get('utilization', 0.0),
                    'mean_phi': memory_stats.get('mean_phi', 0.0),
                    'writes_this_batch': write_info['num_writes'].item(),
                    'threshold': self.iit_memory.threshold_logit.exp().item()
                }
                
                # Agregar a phi_metrics
                phi_metrics['memory'] = memory_info
        
        return outputs, phi_metrics
    
    def generate_with_phi(self, prompt, max_length=100, **kwargs):
        """Genera texto y mide PHI."""
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        
        # Generar
        with torch.no_grad():
            output_ids = self.gpt2.generate(
                input_ids,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Medir PHI del output
        outputs = self.gpt2(
            output_ids,
            output_hidden_states=True,
            output_attentions=True
        )
        
        phi_metrics = self.phi_observer(
            outputs.hidden_states[-1],
            outputs.attentions[-1]
        )
        
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return {
            'text': generated_text,
            'phi': phi_metrics['phi'].mean().item(),
            'components': phi_metrics['raw_components']
        }


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

# Variables globales para guardado de emergencia
_emergency_save_data = None
_save_path = 'models/infinito_gpt2_spanish_phi.pt'  # Nuevo nombre para versión española

def emergency_save(signum=None, frame=None):
    """Guarda el modelo si hay interrupción."""
    global _emergency_save_data
    if _emergency_save_data is not None:
        print("\n" + "="*60)
        print("⚠️  INTERRUPCIÓN DETECTADA - GUARDANDO PROGRESO...")
        torch.save(_emergency_save_data, _save_path)
        print(f"✅ Checkpoint guardado en: {_save_path}")
        print("="*60)
    sys.exit(0)

# Solo registrar signal en el thread principal (evita error en Streamlit)
try:
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, emergency_save)
except Exception:
    pass  # Ignorar en threads secundarios (ej: Streamlit)


def train():
    """Entrenamiento principal."""
    global _emergency_save_data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Dispositivo: {device}")
    
    # Crear modelo
    model = InfinitoGPT2WithObserver(use_lora=True)
    model = model.to(device)
    
    # Cargar dataset
    print("\n📚 Cargando dataset Alpaca Spanish...")
    dataset = load_dataset('bertin-project/alpaca-spanish', split='train[:5000]')
    
    # Preparar textos
    texts = []
    for item in dataset:
        if item['input']:
            text = f"{item['instruction']} {item['input']} {item['output']}"
        else:
            text = f"{item['instruction']} {item['output']}"
        texts.append(text[:512])  # Limitar longitud
    
    # Tokenizar
    print("🔤 Tokenizando...")
    tokenizer = model.tokenizer
    
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # Dataset simple
    class TextDataset(Dataset):
        def __init__(self, encodings):
            self.input_ids = encodings['input_ids']
            self.attention_mask = encodings['attention_mask']
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx]
            }
    
    train_dataset = TextDataset(encodings)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scaler = GradScaler()
    
    # Training
    EPOCHS = 3
    SAVE_EVERY = 200
    
    print(f"\n🚀 Iniciando entrenamiento ({EPOCHS} epochs)...")
    print("-"*60)
    
    phi_log = []
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_phi = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            
            # Forward con FP16
            with autocast('cuda'):
                outputs, phi_metrics = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    return_phi=True
                )
                loss = outputs.loss
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            if phi_metrics:
                phi_val = phi_metrics['phi'].mean().item()
                total_phi += phi_val
                
                # Log PHI + Memory
                log_entry = {
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'phi': phi_val,
                    'components': phi_metrics['raw_components']
                }
                
                # Agregar stats de memoria si están disponibles
                if 'memory' in phi_metrics:
                    log_entry['memory'] = phi_metrics['memory']
                
                phi_log.append(log_entry)
            
            # Actualizar datos de emergencia
            _emergency_save_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'batch': batch_idx,
                'loss': total_loss / (batch_idx + 1),
                'phi_log': phi_log
            }
            
            # Progreso (incluir stats de memoria)
            if batch_idx % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_phi = total_phi / (batch_idx + 1)
                mem_info = ""
                if phi_metrics and 'memory' in phi_metrics:
                    mem = phi_metrics['memory']
                    mem_info = f" | Mem: {mem['utilization']*100:.1f}% (th:{mem['threshold']:.2f})"
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {avg_loss:.4f} | PHI: {avg_phi:.2f}{mem_info}")
            
            # Checkpoint periódico
            if batch_idx > 0 and batch_idx % SAVE_EVERY == 0:
                torch.save(_emergency_save_data, f'models/gpt2_phi_checkpoint_e{epoch}_b{batch_idx}.pt')
                print(f"💾 Checkpoint guardado")
        
        avg_loss = total_loss / len(train_loader)
        avg_phi = total_phi / len(train_loader)
        print(f"\n✅ Epoch {epoch+1} completado | Loss: {avg_loss:.4f} | PHI promedio: {avg_phi:.2f}")
    
    # Guardar modelo final
    torch.save({
        'model_state_dict': model.state_dict(),
        'phi_log': phi_log,
        'epochs': EPOCHS,
        'memory_stats': model.iit_memory.get_statistics() if hasattr(model, 'iit_memory') else None
    }, _save_path)
    
    print(f"\n🎉 Entrenamiento completado!")
    print(f"   Modelo guardado en: {_save_path}")
    
    # Análisis PHI
    print("\n📊 ANÁLISIS PHI:")
    print("-"*40)
    if phi_log:
        phis = [p['phi'] for p in phi_log]
        print(f"   PHI promedio: {sum(phis)/len(phis):.2f}")
        print(f"   PHI máximo: {max(phis):.2f}")
        print(f"   PHI mínimo: {min(phis):.2f}")
    
    # Análisis Memoria
    print("\n🧠 ANÁLISIS MEMORIA IIT:")
    print("-"*40)
    if hasattr(model, 'iit_memory'):
        mem_stats = model.iit_memory.get_statistics()
        print(f"   Utilización: {mem_stats.get('utilization', 0)*100:.1f}%")
        print(f"   PHI promedio en memoria: {mem_stats.get('mean_phi', 0):.2f}")
        print(f"   Threshold aprendido: {model.iit_memory.threshold_logit.exp().item():.2f}")
    
    # Guardar log PHI
    with open('results/phi_training_log.json', 'w') as f:
        json.dump(phi_log, f, indent=2)
    print(f"   Log PHI guardado en: results/phi_training_log.json")
    
    # Demo de generación
    print("\n🔮 DEMO DE GENERACIÓN CON PHI:")
    print("-"*40)
    
    model.eval()
    prompts = [
        "La inteligencia artificial es",
        "El futuro de la humanidad",
        "Explica qué es la física cuántica"
    ]
    
    for prompt in prompts:
        result = model.generate_with_phi(prompt, max_length=50)
        print(f"\nPrompt: '{prompt}'")
        print(f"PHI: {result['phi']:.2f}")
        print(f"Output: {result['text'][:100]}...")


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    train()
