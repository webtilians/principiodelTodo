#!/usr/bin/env python3
"""
🧠 DETECTOR DE IMPORTANCIA CON GATE DINÁMICO v2
================================================

Versión mejorada con LOSS EXPLÍCITO para entrenar el gate:
- Gate ABIERTO para frases importantes ("Me llamo...")
- Gate CERRADO para ruido ("El cielo es azul")

La clave: Añadir un loss de discriminación que guíe al gate.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, 'src')
from infinito_v5_2_refactored import InfinitoV52Refactored


class InfinitoDynamicChat(InfinitoV52Refactored):
    """Modelo con gate dinámico que reacciona al contenido."""
    
    def __init__(self, *args, **kwargs):
        kwargs['use_dynamic_gate'] = False
        super().__init__(*args, **kwargs)
        
        if hasattr(self, 'memory_gate'):
            del self.memory_gate
            
        # Gate dinámico con capacidad de aprender
        self.gate_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, 1)
        )
        self._init_gate()

    def _init_gate(self):
        """Inicialización balanceada (no tan cerrado)."""
        # Inicializar con Xavier para mejor flujo de gradientes
        for layer in self.gate_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        # Bias final a -2 (sigmoid(-2) ≈ 12%, empieza algo abierto)
        nn.init.constant_(self.gate_network[-1].bias, -2.0)

    def forward(self, input_ids, return_metrics=False):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding[:, :seq_len, :]
        hidden = self.embedding_dropout(hidden)
        
        # Transformer
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
            
        # Gate dinámico
        sentence_context = hidden.mean(dim=1)
        gate_logit = self.gate_network(sentence_context)
        gate_prob = torch.sigmoid(gate_logit)  # [batch, 1]
        
        # Memoria simulada
        if self.use_improved_memory:
            memory_contribution = torch.randn_like(hidden) * 0.1
            gate_broadcast = gate_prob.unsqueeze(1)
            hidden = hidden + (gate_broadcast * memory_contribution)
            hidden = self.memory_norm(hidden)

        logits = self.output_projection(hidden)
        
        if return_metrics:
            return logits, {
                'gate_value': gate_prob.mean().item(),
                'gate_prob': gate_prob,  # Tensor para loss
                'gate_logit': gate_logit,
            }
        return logits, None


def text_to_ids(text, seq_len=32):
    ids = [ord(c) % 256 for c in text]
    ids = ids[:seq_len] + [0] * (seq_len - len(ids))
    return torch.tensor(ids)


def generate_batch_with_labels(batch_size=32):
    """Genera batch CON ETIQUETAS de importancia."""
    inputs = []
    targets = []
    importance_labels = []  # 1.0 = importante, 0.0 = ruido
    
    nombres = ["Enrique", "Ana", "Carlos", "Sofia", "Bot", "Infinito", "María", "Pedro", "Luis", "Eva"]
    patrones_importantes = [
        "Me llamo {}",
        "Mi nombre es {}",
        "Soy {}",
        "La contraseña es {}",
        "El código es {}",
        "Recuerda que {}",
    ]
    ruido = [
        "El cielo es azul",
        "Mañana es martes", 
        "Hace sol hoy",
        "Me gusta el pan",
        "Uno dos tres",
        "La mesa es grande",
        "El agua es fría",
        "Hace calor aquí",
        "Es de noche",
        "El perro ladra",
    ]
    
    for _ in range(batch_size):
        is_important = random.random() > 0.5
        
        if is_important:
            patron = random.choice(patrones_importantes)
            valor = random.choice(nombres + ["1234", "secreto", "42"])
            text = patron.format(valor)
            target_txt = text[1:] + "."
            importance_labels.append(1.0)  # Debería ABRIR gate
        else:
            text = random.choice(ruido)
            target_txt = text[1:] + "."
            importance_labels.append(0.0)  # Debería CERRAR gate
            
        inputs.append(text_to_ids(text))
        targets.append(text_to_ids(target_txt))
        
    return (
        torch.stack(inputs), 
        torch.stack(targets), 
        torch.tensor(importance_labels).unsqueeze(1)  # [batch, 1]
    )


def train_detector():
    """Entrena con loss de discriminación explícito."""
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    VOCAB_SIZE = 256
    
    print(f"\n{'='*65}")
    print(f"🧠 DETECTOR DE IMPORTANCIA v2 (con Loss de Discriminación)")
    print(f"{'='*65}")
    print(f"Device: {DEVICE}")
    print(f"Objetivo: Gate ABIERTO para nombres/claves, CERRADO para ruido")
    print(f"{'='*65}\n")
    
    model = InfinitoDynamicChat(
        vocab_size=VOCAB_SIZE, 
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        use_improved_memory=True,
        use_improved_iit=True,
    ).to(DEVICE)
    
    # Cargar pesos compatibles
    try:
        checkpoint = torch.load('models/super_golden_seed_54percent.pt', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        compatible = {k: v for k, v in state_dict.items() 
                     if k in model_dict 
                     and 'token_embedding' not in k 
                     and 'gate' not in k 
                     and 'output' not in k
                     and v.shape == model_dict[k].shape}
        
        model_dict.update(compatible)
        model.load_state_dict(model_dict, strict=False)
        print(f"✅ Cargados {len(compatible)} parámetros del modelo 54%\n")
    except Exception as e:
        print(f"⚠️ Entrenando desde cero: {e}\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()  # Para el gate
    
    history = {'loss': [], 'gate_loss': [], 'gate_ruido': [], 'gate_importante': []}
    
    EPOCHS = 1000
    GATE_WEIGHT = 1.0  # Peso del loss de discriminación
    
    pbar = tqdm(range(EPOCHS), desc="Entrenando")
    
    for i in pbar:
        inputs, targets, importance = generate_batch_with_labels(32)
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        importance = importance.to(DEVICE)
        
        optimizer.zero_grad()
        logits, metrics = model(inputs, return_metrics=True)
        
        # Loss 1: Predicción de secuencia
        loss_pred = ce_criterion(logits.transpose(1, 2), targets.long())
        
        # Loss 2: Discriminación del gate
        # El gate (metrics['gate_prob']) debería = importance
        gate_prob = metrics['gate_prob']  # [batch, 1]
        loss_gate = bce_criterion(gate_prob, importance)
        
        # Loss total
        total_loss = loss_pred + GATE_WEIGHT * loss_gate
        total_loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        history['loss'].append(loss_pred.item())
        history['gate_loss'].append(loss_gate.item())
        
        if i % 100 == 0:
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
                
                diff = g_imp - g_ruido
                arrow = "↑" if diff > 5 else "↓" if diff < -5 else "="
                
                pbar.set_description(
                    f"Pred:{loss_pred.item():.3f} Gate:{loss_gate.item():.3f} | "
                    f"R:{g_ruido:.1f}% I:{g_imp:.1f}% {arrow}"
                )
    
    return model, history


def test_model(model, device):
    """Prueba exhaustiva del modelo."""
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS FINALES DEL GATE DINÁMICO")
    print(f"{'='*70}\n")
    
    frases_test = [
        # Ruido (debería cerrar gate ≈ 0%)
        ("El cielo es azul", "RUIDO", 0),
        ("Me gusta el pan", "RUIDO", 0),
        ("Hace sol hoy", "RUIDO", 0),
        ("El perro ladra", "RUIDO", 0),
        
        # Importantes (debería abrir gate ≈ 100%)
        ("Me llamo Enrique", "IMPORTANTE", 1),
        ("Me llamo Infinito", "IMPORTANTE", 1),
        ("Mi nombre es Ana", "IMPORTANTE", 1),
        ("Soy Carlos", "IMPORTANTE", 1),
        ("La contraseña es 1234", "IMPORTANTE", 1),
        
        # Casos ambiguos (no vistos exactamente)
        ("El secreto es azul", "AMBIGUO", -1),
        ("Hola mundo", "NEUTRAL", -1),
        ("Recuerda esto", "AMBIGUO", -1),
    ]
    
    print(f"{'Frase':<28} | {'Tipo':<12} | {'Esperado':<8} | 🚪 Gate")
    print("-" * 70)
    
    correct = 0
    total_labeled = 0
    
    with torch.no_grad():
        for texto, tipo, expected in frases_test:
            inp = text_to_ids(texto).unsqueeze(0).to(device)
            _, metrics = model(inp, return_metrics=True)
            gate_pct = metrics['gate_value'] * 100
            
            # Clasificación
            if gate_pct > 50:
                emoji = "🟢 ABIERTO"
                predicted = 1
            elif gate_pct > 20:
                emoji = "🟡 PARCIAL"
                predicted = 0.5
            else:
                emoji = "🔴 CERRADO"
                predicted = 0
            
            # Accuracy
            if expected != -1:
                total_labeled += 1
                if (expected == 1 and gate_pct > 50) or (expected == 0 and gate_pct < 50):
                    correct += 1
                    mark = "✓"
                else:
                    mark = "✗"
            else:
                mark = "?"
            
            exp_str = "OPEN" if expected == 1 else "CLOSE" if expected == 0 else "?"
            print(f"{texto:<28} | {tipo:<12} | {exp_str:<8} | {emoji} {gate_pct:>6.2f}% {mark}")
    
    print("-" * 70)
    if total_labeled > 0:
        accuracy = correct / total_labeled * 100
        print(f"\n🎯 Accuracy: {correct}/{total_labeled} = {accuracy:.1f}%")
    
    return correct, total_labeled


def analyze_evolution(history):
    """Muestra la evolución del aprendizaje."""
    
    print(f"\n{'='*65}")
    print(f"📈 EVOLUCIÓN DEL APRENDIZAJE")
    print(f"{'='*65}\n")
    
    if not history['gate_ruido']:
        print("No hay datos de evolución.")
        return
    
    print("Paso    | Gate Ruido | Gate Import | Diferencia | ¿Aprendió?")
    print("-" * 65)
    
    for i, (gr, gi) in enumerate(zip(history['gate_ruido'], history['gate_importante'])):
        paso = i * 100
        diff = gi - gr
        
        if diff > 30:
            status = "✅ SÍ"
        elif diff > 10:
            status = "🟡 Parcial"
        else:
            status = "❌ No"
        
        print(f"{paso:>6}  | {gr:>9.1f}% | {gi:>10.1f}% | {diff:>+9.1f}% | {status}")
    
    print("-" * 65)
    
    inicio = history['gate_importante'][0] - history['gate_ruido'][0]
    final = history['gate_importante'][-1] - history['gate_ruido'][-1]
    
    print(f"\n📊 Resumen:")
    print(f"   Diferencia inicial: {inicio:+.1f}%")
    print(f"   Diferencia final:   {final:+.1f}%")
    print(f"   Cambio:             {final - inicio:+.1f}%")
    
    if final > 30:
        print(f"\n   🎉 ¡ÉXITO! El gate discrimina entre ruido e información importante")
    elif final > 10:
        print(f"\n   🟡 Parcial: El gate muestra algo de discriminación")
    else:
        print(f"\n   ⚠️ El gate no discrimina suficientemente")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Entrenar
    model, history = train_detector()
    
    # Testear
    correct, total = test_model(model, DEVICE)
    
    # Analizar
    analyze_evolution(history)
    
    # Guardar
    save_path = 'models/dynamic_chat_detector_v2.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'accuracy': correct / total if total > 0 else 0,
    }, save_path)
    print(f"\n💾 Modelo guardado en: {save_path}")
    
    print(f"\n{'='*65}")
    print("🏁 EXPERIMENTO v2 COMPLETADO")
    print(f"{'='*65}")
