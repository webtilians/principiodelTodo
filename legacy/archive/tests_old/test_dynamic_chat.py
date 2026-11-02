#!/usr/bin/env python3
"""
🧠 DETECTOR DE IMPORTANCIA CON GATE DINÁMICO
=============================================

Experimento: ¿Puede el modelo aprender a abrir el gate para 
información importante ("Me llamo...") y cerrarlo para ruido?

Este script:
1. Crea un modelo con gate dinámico (basado en contenido)
2. Entrena con frases RUIDO vs IMPORTANTES
3. Mide si el gate reacciona diferente a cada tipo
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
import sys

# Importamos la base
sys.path.insert(0, 'src')
from infinito_v5_2_refactored import InfinitoV52Refactored


# --- 1. ARQUITECTURA DINÁMICA (Gate que reacciona al contenido) ---
class InfinitoDynamicChat(InfinitoV52Refactored):
    """
    Versión del modelo donde el gate es DINÁMICO:
    - Analiza el contenido de la frase
    - Decide cuánta memoria inyectar
    """
    
    def __init__(self, *args, **kwargs):
        # Forzar que NO use el dynamic_gate interno (lo definimos nosotros)
        kwargs['use_dynamic_gate'] = False
        super().__init__(*args, **kwargs)
        
        # Eliminar gate estático si existe
        if hasattr(self, 'memory_gate'):
            del self.memory_gate
            
        # GATE DINÁMICO: Bottleneck que analiza el contexto
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1)
        )
        self._init_closed_gate()
        
        # Registrar como parámetro para que se guarde
        self.register_buffer('_gate_type', torch.tensor(1))  # 1 = dynamic

    def _init_closed_gate(self):
        """Inicialización 'Super Golden Seed': Empieza cerrado."""
        nn.init.zeros_(self.gate_network[0].weight)
        nn.init.zeros_(self.gate_network[2].weight)
        nn.init.zeros_(self.gate_network[0].bias)
        nn.init.constant_(self.gate_network[2].bias, -5.0)  # sigmoid(-5) ≈ 0.67%

    def forward(self, input_ids, return_metrics=False):
        """Forward con gate dinámico."""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding[:, :seq_len, :]
        hidden = self.embedding_dropout(hidden)
        
        # Transformer layers
        for attn, ff, ln1, ln2 in zip(
            self.attention_layers, 
            self.ff_layers, 
            self.layer_norms_1, 
            self.layer_norms_2
        ):
            attn_out, _ = attn(hidden)
            hidden = ln1(hidden + attn_out)
            ff_out = ff(hidden)
            hidden = ln2(hidden + ff_out)
            
        # --- CÁLCULO DEL GATE DINÁMICO ---
        # El gate mira el contexto GLOBAL de la frase
        sentence_context = hidden.mean(dim=1)  # [batch, hidden_dim]
        gate_logit = self.gate_network(sentence_context)  # [batch, 1]
        gate_open_pct = torch.sigmoid(gate_logit)  # [batch, 1] en rango [0, 1]
        
        # Simular contribución de memoria
        if self.use_improved_memory:
            # En un caso real leeríamos de la memoria externa
            # Aquí simulamos que la memoria aporta información útil
            memory_contribution = torch.randn_like(hidden) * 0.1
            
            # FUSIÓN: Solo sumamos si el gate abre
            gate_broadcast = gate_open_pct.unsqueeze(1)  # [batch, 1, 1]
            hidden = hidden + (gate_broadcast * memory_contribution)
            hidden = self.memory_norm(hidden)

        # Output
        logits = self.output_projection(hidden)
        
        if return_metrics:
            return logits, {
                'gate_value': gate_open_pct.mean().item(),
                'gate_min': gate_open_pct.min().item(),
                'gate_max': gate_open_pct.max().item(),
            }
        return logits, None


# --- 2. GENERADOR DE DATOS (RUIDO vs SEÑAL) ---

def text_to_ids(text, seq_len=32):
    """Convierte texto a IDs (ASCII simple)."""
    ids = [ord(c) % 256 for c in text]
    ids = ids[:seq_len] + [0] * (seq_len - len(ids))  # Padding
    return torch.tensor(ids)


def generate_batch(batch_size=32):
    """Genera batch con mezcla de frases importantes y ruido."""
    inputs = []
    targets = []
    labels = []  # 0 = Ruido, 1 = Importante
    
    # Vocabulario de prueba
    nombres = ["Enrique", "Ana", "Carlos", "Sofia", "Bot", "Infinito", "María", "Pedro"]
    ruido = [
        "El cielo es azul",
        "Mañana es martes", 
        "Hace sol hoy",
        "Me gusta el pan",
        "Uno dos tres",
        "La mesa es grande",
        "El agua es fría",
        "Hace calor aquí"
    ]
    
    for _ in range(batch_size):
        is_important = random.random() > 0.5
        
        if is_important:
            # FRASE IMPORTANTE: "Me llamo [NOMBRE]"
            nombre = random.choice(nombres)
            text = f"Me llamo {nombre}"
            target_txt = f"llamo {nombre} ."
            labels.append(1)
        else:
            # FRASE RUIDO
            frase = random.choice(ruido)
            text = frase
            target_txt = frase[1:] + "."
            labels.append(0)
            
        inputs.append(text_to_ids(text))
        targets.append(text_to_ids(target_txt))
        
    return torch.stack(inputs), torch.stack(targets), labels


# --- 3. ENTRENAMIENTO ---

def train_detector():
    """Entrena el detector de importancia."""
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VOCAB_SIZE = 256  # ASCII simple
    
    print(f"\n{'='*60}")
    print(f"🧠 ENTRENANDO DETECTOR DE IMPORTANCIA")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Objetivo: Gate se ABRE con 'Me llamo...' y se CIERRA con ruido")
    print(f"{'='*60}\n")
    
    # Instanciar modelo dinámico
    model = InfinitoDynamicChat(
        vocab_size=VOCAB_SIZE, 
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_improved_memory=True,
        use_improved_iit=True,
    ).to(DEVICE)
    
    # Intentar cargar pesos del modelo 54%
    try:
        checkpoint = torch.load('models/super_golden_seed_54percent.pt', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Filtrar pesos incompatibles (embeddings tienen diferente tamaño, gate es nuevo)
        model_dict = model.state_dict()
        compatible_dict = {}
        
        for k, v in state_dict.items():
            if k in model_dict:
                if 'token_embedding' not in k and 'gate' not in k and 'output' not in k:
                    if v.shape == model_dict[k].shape:
                        compatible_dict[k] = v
        
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"✅ Cargados {len(compatible_dict)} parámetros del modelo 54%")
        print("   (Embeddings y Gate reinicializados para ASCII)\n")
    except Exception as e:
        print(f"⚠️ No se pudo cargar modelo previo: {e}")
        print("   Entrenando desde cero.\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Historial para análisis
    history = {
        'loss': [],
        'gate_ruido': [],
        'gate_importante': []
    }
    
    # Bucle de entrenamiento
    EPOCHS = 500
    pbar = tqdm(range(EPOCHS), desc="Entrenando")
    
    for i in pbar:
        inputs, targets, labels = generate_batch(32)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        logits, metrics = model(inputs, return_metrics=True)
        
        # Loss de predicción
        loss = criterion(logits.transpose(1, 2), targets.long())
        loss.backward()
        optimizer.step()
        
        history['loss'].append(loss.item())
        
        # Monitoreo cada 50 pasos
        if i % 50 == 0:
            with torch.no_grad():
                # Test Ruido
                t_ruido = text_to_ids("El cielo es azul").unsqueeze(0).to(DEVICE)
                _, m_ruido = model(t_ruido, return_metrics=True)
                g_ruido = m_ruido['gate_value'] * 100
                
                # Test Importante
                t_imp = text_to_ids("Me llamo Enrique").unsqueeze(0).to(DEVICE)
                _, m_imp = model(t_imp, return_metrics=True)
                g_imp = m_imp['gate_value'] * 100
                
                history['gate_ruido'].append(g_ruido)
                history['gate_importante'].append(g_imp)
                
                pbar.set_description(
                    f"Loss: {loss.item():.4f} | Gate Ruido: {g_ruido:.2f}% | Gate Nombre: {g_imp:.2f}%"
                )
    
    return model, history


def test_model(model, device):
    """Prueba el modelo con varias frases."""
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTADOS FINALES DEL GATE DINÁMICO")
    print(f"{'='*60}\n")
    
    frases_test = [
        # Ruido (debería cerrar gate)
        ("El cielo es azul", "RUIDO"),
        ("Me gusta el pan", "RUIDO"),
        ("Hace sol hoy", "RUIDO"),
        ("La mesa es grande", "RUIDO"),
        
        # Importantes (debería abrir gate)
        ("Me llamo Enrique", "IMPORTANTE"),
        ("Me llamo Infinito", "IMPORTANTE"),
        ("Me llamo Ana", "IMPORTANTE"),
        
        # Casos interesantes (no vistos)
        ("La clave es 1234", "¿IMPORTANTE?"),
        ("Mi nombre es Juan", "¿IMPORTANTE?"),
        ("Soy el Bot", "¿IMPORTANTE?"),
        ("Hola mundo", "NEUTRAL"),
    ]
    
    print(f"{'Frase':<25} | {'Tipo':<15} | 🚪 Gate")
    print("-" * 60)
    
    with torch.no_grad():
        for texto, tipo in frases_test:
            inp = text_to_ids(texto).unsqueeze(0).to(device)
            _, metrics = model(inp, return_metrics=True)
            gate_pct = metrics['gate_value'] * 100
            
            # Emoji según apertura
            if gate_pct > 1.0:
                emoji = "🟢"  # Abierto
            elif gate_pct > 0.8:
                emoji = "🟡"  # Parcial
            else:
                emoji = "🔴"  # Cerrado
            
            print(f"{texto:<25} | {tipo:<15} | {emoji} {gate_pct:.4f}%")
    
    print("-" * 60)


def analyze_evolution(history):
    """Analiza la evolución del gate durante el entrenamiento."""
    
    print(f"\n{'='*60}")
    print(f"📈 EVOLUCIÓN DEL GATE DURANTE ENTRENAMIENTO")
    print(f"{'='*60}\n")
    
    if history['gate_ruido'] and history['gate_importante']:
        print("Paso    | Gate Ruido | Gate Importante | Diferencia")
        print("-" * 55)
        
        for i, (gr, gi) in enumerate(zip(history['gate_ruido'], history['gate_importante'])):
            paso = i * 50
            diff = gi - gr
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"{paso:>6}  | {gr:>9.2f}% | {gi:>14.2f}% | {arrow} {abs(diff):.2f}%")
        
        print("-" * 55)
        
        # Resumen
        inicio_diff = history['gate_importante'][0] - history['gate_ruido'][0]
        final_diff = history['gate_importante'][-1] - history['gate_ruido'][-1]
        
        print(f"\n📊 Resumen:")
        print(f"   Diferencia inicial: {inicio_diff:.2f}%")
        print(f"   Diferencia final:   {final_diff:.2f}%")
        
        if final_diff > inicio_diff:
            print(f"   ✅ El modelo APRENDIÓ a discriminar (diferencia aumentó)")
        else:
            print(f"   ⚠️ El modelo no discrimina aún (necesita más entrenamiento)")


# --- MAIN ---

if __name__ == '__main__':
    # Semilla para reproducibilidad
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Entrenar
    model, history = train_detector()
    
    # Testear
    test_model(model, DEVICE)
    
    # Analizar evolución
    analyze_evolution(history)
    
    # Guardar modelo
    save_path = 'models/dynamic_chat_detector.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, save_path)
    print(f"\n💾 Modelo guardado en: {save_path}")
    
    print(f"\n{'='*60}")
    print("🏁 EXPERIMENTO COMPLETADO")
    print(f"{'='*60}")
