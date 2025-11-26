#!/usr/bin/env python3
"""
🔮 INFINITO UI - Beta v0.1
===========================

Interfaz gráfica con Streamlit que muestra:
- Chat Central estilo WhatsApp/ChatGPT
- Monitor del Gate Visual (barra de importancia)
- Memoria en Vivo (sidebar con JSON en tiempo real)
- Métricas IIT completas (PHI, Coherence, Integration)

Ejecutar con: streamlit run app.py
"""

import streamlit as st
import torch
import torch.nn as nn
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Infinito Beta", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .memory-card {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border-left: 3px solid #667eea;
    }
    .gate-high { color: #00ff88; font-weight: bold; }
    .gate-medium { color: #ffcc00; }
    .gate-low { color: #ff6b6b; }
    .iit-metric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 8px;
        padding: 8px;
        margin: 4px 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURACIÓN ---
MODEL_PATH = "models/dynamic_chat_detector_v2.pt"
DB_FILE = "memoria_infinito_ui.json"

# API Key de OpenAI - Múltiples fuentes (prioridad: secrets > env > input manual)
API_KEY = ""
try:
    # 1. Primero intentar Streamlit Secrets (para Streamlit Cloud)
    API_KEY = st.secrets.get("OPENAI_API_KEY", "")
except:
    pass

if not API_KEY:
    # 2. Luego variable de entorno (para local)
    API_KEY = os.environ.get("OPENAI_API_KEY", "")

# =============================================================================
# MODELO INFINITO CON IIT (Versión simplificada para UI)
# =============================================================================

class InfinitoUIModel(nn.Module):
    """Modelo simplificado para la UI con métricas IIT."""
    
    def __init__(self, vocab_size=256, hidden_dim=64, num_layers=2, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Importance Gate (el "portero")
        self.importance_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Memory Gate
        self.memory_gate = nn.Parameter(torch.tensor(-2.0))
        
        # IIT Metrics simulados
        self.phi_weight = nn.Parameter(torch.tensor(0.5))
        
        # Output
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, return_metrics=False):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden = self.token_embedding(input_ids)
        hidden = hidden + self.position_embedding[:, :seq_len, :]
        
        # Transformer
        for layer in self.layers:
            hidden = layer(hidden)
        
        # Query para métricas
        query = hidden.mean(dim=1)  # [batch, hidden]
        
        # Calcular métricas IIT
        importance_logit = self.importance_gate(query)
        importance = torch.sigmoid(importance_logit)
        
        memory_gate_value = torch.sigmoid(self.memory_gate)
        
        # Simular PHI y coherencia basados en la varianza de hidden states
        hidden_var = hidden.var(dim=1).mean()
        phi = torch.sigmoid(self.phi_weight * hidden_var * 10)
        coherence = torch.sigmoid(hidden.mean() * 5 + 0.5)
        
        # Complejidad basada en la entropía aproximada
        logits = self.output(hidden)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        complexity = torch.sigmoid(entropy / 5)
        
        if return_metrics:
            metrics = {
                'importance': importance.mean().item(),
                'gate_value': importance.mean().item() * 100,  # Porcentaje
                'memory_gate': memory_gate_value.item(),
                'phi': phi.item(),
                'coherence': coherence.item(),
                'complexity': complexity.item(),
                'integration': (phi.item() + coherence.item()) / 2
            }
            return logits, metrics
        
        return logits, None


def text_to_ids(text, seq_len=64):
    """Convierte texto a IDs ASCII."""
    ids = [ord(c) % 256 for c in text]
    if len(ids) < seq_len:
        ids = ids + [0] * (seq_len - len(ids))
    else:
        ids = ids[:seq_len]
    return torch.tensor([ids])


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_memories():
    """Carga memorias del JSON."""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_memory(text, metrics):
    """Guarda una memoria con métricas IIT."""
    memories = get_memories()
    
    # Detectar categoría
    category = detect_category(text)
    
    entry = {
        "id": len(memories) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "content": text,
        "category": category,
        "importance": f"{metrics['importance'] * 100:.1f}%",
        "phi": f"{metrics['phi']:.3f}",
        "coherence": f"{metrics['coherence']:.3f}",
        "combined_score": f"{(metrics['importance'] + metrics['phi']) / 2 * 100:.1f}%"
    }
    memories.append(entry)
    
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)
    
    return entry


def detect_category(text):
    """Detecta la categoría de la información."""
    t = text.lower()
    if any(x in t for x in ['me llamo', 'mi nombre', 'soy ']):
        return "👤 Identidad"
    elif any(x in t for x in ['contraseña', 'clave', 'password', 'pin']):
        return "🔐 Credencial"
    elif any(x in t for x in ['teléfono', 'email', 'correo', 'dirección']):
        return "📞 Contacto"
    elif any(x in t for x in ['mi primo', 'mi hermano', 'mi madre', 'mi padre', 'mi amigo', 'mi esposa', 'mi hijo']):
        return "👨‍👩‍👧 Familia"
    elif any(x in t for x in ['recuerda', 'no olvides', 'mañana', 'cita']):
        return "📌 Recordatorio"
    elif any(x in t for x in ['me gusta', 'prefiero', 'favorito']):
        return "❤️ Preferencia"
    return "📝 General"


def es_pregunta(text):
    """Detecta si es una pregunta."""
    t = text.lower().strip()
    if '?' in t or '¿' in t:
        return True
    starts = ['que ', 'qué ', 'como ', 'cómo ', 'cual ', 'cuál ', 
              'quien ', 'quién ', 'donde ', 'dónde ', 'cuando ', 
              'cuándo ', 'sabes ', 'recuerdas ', 'dime ', 'cuanto ', 'cuánto ']
    for s in starts:
        if t.startswith(s):
            return True
    return False


def get_category_bonus(category):
    """Bonus de importancia por categoría."""
    bonuses = {
        "👤 Identidad": 0.5,
        "🔐 Credencial": 0.6,
        "📞 Contacto": 0.4,
        "👨‍👩‍👧 Familia": 0.4,
        "📌 Recordatorio": 0.5,
        "❤️ Preferencia": 0.3,
        "📝 General": 0.0
    }
    return bonuses.get(category, 0.0)


def construct_prompt():
    """Construye el prompt con memoria para GPT."""
    memories = get_memories()
    
    if not memories:
        memory_block = "NO TIENES RECUERDOS PREVIOS."
    else:
        # Ordenar por PHI (los más integrados primero)
        sorted_mems = sorted(memories, key=lambda x: float(x.get('phi', '0').replace(',', '.')), reverse=True)
        
        memory_block = "🧠 MEMORIA A LARGO PLAZO (ordenada por integración PHI):\n"
        for mem in sorted_mems[-15:]:
            memory_block += f"  • {mem['content']} [{mem['category']}] (PHI:{mem['phi']}, Imp:{mem['importance']})\n"
    
    return f"""Eres Infinito, un asistente con MEMORIA PERSISTENTE basada en teoría IIT (Integrated Information Theory).

{memory_block}

INSTRUCCIONES:
1. USA la memoria para personalizar respuestas
2. Si preguntan algo en tu memoria, RESPONDE DIRECTAMENTE
3. Los recuerdos con PHI alto son más "integrados" en tu consciencia
4. Sé breve y útil. Responde en español."""


# =============================================================================
# INICIALIZACIÓN DEL MODELO
# =============================================================================

@st.cache_resource
def load_model():
    """Carga el modelo (cached para no recargar)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = InfinitoUIModel(
        vocab_size=256,
        hidden_dim=64,
        num_layers=2,
        num_heads=4
    ).to(device)
    
    # Intentar cargar pesos entrenados
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Cargar solo pesos compatibles
            model_dict = model.state_dict()
            compatible = {k: v for k, v in state_dict.items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(compatible)
            model.load_state_dict(model_dict, strict=False)
            st.sidebar.success(f"✅ Modelo cargado ({len(compatible)} params)")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Usando modelo sin entrenar: {e}")
    else:
        st.sidebar.info("🆕 Modelo inicializado (sin pesos pre-entrenados)")
    
    model.eval()
    return model, device


def analyze_importance(text, model, device):
    """Analiza la importancia de un texto con el modelo IIT."""
    inp = text_to_ids(text).to(device)
    with torch.no_grad():
        _, metrics = model(inp, return_metrics=True)
    
    # Añadir bonus por categoría
    category = detect_category(text)
    category_bonus = get_category_bonus(category)
    
    # Score combinado
    combined = metrics['importance'] + category_bonus + (metrics['phi'] * 0.3)
    metrics['combined_score'] = min(combined, 1.0)
    metrics['category'] = category
    metrics['category_bonus'] = category_bonus
    
    return metrics


# =============================================================================
# INTERFAZ GRÁFICA
# =============================================================================

# Inicializar modelo
model, device = load_model()

# Estado de la sesión
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "🔮 Hola. Soy **Infinito**. Tengo memoria selectiva basada en IIT. Cuéntame algo importante y observa cómo mi Gate decide si guardarlo."}
    ]

# Historial de análisis IIT (NUEVO - para mostrar permanentemente)
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []

if "openai_client" not in st.session_state:
    # Inicializar OpenAI automáticamente con la API key configurada
    if API_KEY and API_KEY.startswith("sk-"):
        try:
            from openai import OpenAI
            st.session_state["openai_client"] = OpenAI(api_key=API_KEY)
        except Exception as e:
            st.session_state["openai_client"] = None
    else:
        st.session_state["openai_client"] = None

# --- SIDEBAR: MEMORIA EN VIVO ---
with st.sidebar:
    st.markdown("## 🧠 Memoria Viva")
    st.caption("Lo que la IA decide guardar en disco:")
    
    # Estado de OpenAI
    if st.session_state.get("openai_client"):
        st.success("✅ OpenAI GPT-3.5 conectado")
    else:
        with st.expander("⚙️ Configuración OpenAI", expanded=True):
            api_key_input = st.text_input(
                "API Key", 
                value="",
                type="password",
                help="Tu clave de OpenAI para respuestas con GPT"
            )
            if api_key_input and api_key_input.startswith("sk-"):
                try:
                    from openai import OpenAI
                    st.session_state["openai_client"] = OpenAI(api_key=api_key_input)
                    st.success("✅ OpenAI conectado")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    st.divider()
    
    # Mostrar memorias
    memories = get_memories()
    
    if not memories:
        st.info("💭 Memoria vacía. Cuéntame algo importante...")
    else:
        st.metric("📊 Total Recuerdos", len(memories))
        
        # Mostrar últimas memorias (las más recientes arriba)
        for mem in reversed(memories[-8:]):
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{mem['category']}**")
                    st.caption(f"📅 {mem['timestamp']}")
                with col2:
                    st.markdown(f"PHI: `{mem['phi']}`")
                
                st.text_area(
                    label=f"mem_{mem['id']}", 
                    value=mem['content'], 
                    height=60, 
                    disabled=True,
                    label_visibility="collapsed"
                )
                st.divider()
    
    # Botón para borrar memoria
    if st.button("🗑️ Formatear Memoria", use_container_width=True):
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            st.rerun()


# --- ÁREA PRINCIPAL ---
st.markdown('<h1 class="main-header">🔮 Infinito Beta v0.1</h1>', unsafe_allow_html=True)
st.markdown("Experimentando con **Memory Gates Dinámicos** + **Teoría IIT de Consciencia**")

# Métricas en tiempo real (header)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🖥️ Device", device.upper())
with col2:
    st.metric("📚 Memorias", len(get_memories()))
with col3:
    st.metric("🤖 LLM", "GPT-3.5" if st.session_state.get("openai_client") else "Simulado")
with col4:
    st.metric("🧠 Modelo", "IIT-Enhanced")

st.divider()

# Layout de dos columnas: Chat + Panel de Análisis
chat_col, analysis_col = st.columns([2, 1])

# --- COLUMNA DE CHAT ---
with chat_col:
    st.markdown("### 💬 Chat")
    
    # Container para el chat con scroll
    chat_container = st.container(height=400)
    
    with chat_container:
        # Mostrar mensajes anteriores
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

# --- COLUMNA DE ANÁLISIS IIT (Permanente) ---
with analysis_col:
    st.markdown("### 🔬 Monitor IIT en Vivo")
    
    # Mostrar historial de análisis (los más recientes arriba)
    if st.session_state.analysis_history:
        for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
            with st.expander(
                f"{'🟢' if analysis['saved'] else '🟡' if analysis['is_question'] else '🔴'} {analysis['text'][:30]}...", 
                expanded=(i == 0)  # Solo el más reciente expandido
            ):
                # Barras de progreso
                st.caption("**Combined Score**")
                st.progress(min(analysis['combined'], 1.0))
                
                # Métricas en grid
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Gate", f"{analysis['importance']*100:.1f}%", 
                             delta=f"+{analysis['category_bonus']*100:.0f}%" if analysis['category_bonus'] > 0 else None)
                with col2:
                    st.metric("PHI (Φ)", f"{analysis['phi']:.3f}")
                
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Coherence", f"{analysis['coherence']:.3f}")
                with col4:
                    st.metric("Complexity", f"{analysis['complexity']:.3f}")
                
                # Estado
                if analysis['saved']:
                    st.success(f"💾 Guardado: {analysis['category']}")
                elif analysis['is_question']:
                    st.warning("❓ Pregunta - Consultando memoria")
                else:
                    st.info("🔇 No guardado - Información trivial")
                
                st.caption(f"🕐 {analysis['timestamp']}")
    else:
        st.info("💭 Escribe algo para ver el análisis IIT aquí...")

st.divider()

# Input del usuario (fuera de las columnas, ancho completo)
if prompt := st.chat_input("Escribe algo... (ej: 'Me llamo Enrique' o '¿Cómo me llamo?')"):
    # Mostrar mensaje del usuario en el chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # --- PROCESAMIENTO INFINITO ---
    # 1. Análisis del Gate
    metrics = analyze_importance(prompt, model, device)
    importance = metrics['importance']
    combined = metrics['combined_score']
    is_question = es_pregunta(prompt)
    
    # Decisión de guardar
    should_save = (combined > 0.3 or metrics['category_bonus'] > 0.3) and (not is_question)
    
    # Guardar en historial de análisis (para el panel permanente)
    analysis_entry = {
        'text': prompt,
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'importance': importance,
        'combined': combined,
        'phi': metrics['phi'],
        'coherence': metrics['coherence'],
        'complexity': metrics['complexity'],
        'category': metrics['category'],
        'category_bonus': metrics['category_bonus'],
        'is_question': is_question,
        'saved': should_save
    }
    st.session_state.analysis_history.append(analysis_entry)
    
    # Guardar en memoria si es importante
    if should_save:
        save_memory(prompt, metrics)
    
    # --- GENERAR RESPUESTA ---
    full_response = ""
    
    # Usar OpenAI si está disponible
    if st.session_state.get("openai_client"):
        try:
            system_prompt = construct_prompt()
            
            response = st.session_state["openai_client"].chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            full_response = response.choices[0].message.content
            
        except Exception as e:
            full_response = f"❌ Error con OpenAI: {e}"
    else:
        # Respuesta simulada sin OpenAI
        memories = get_memories()
        
        # Buscar nombre en memoria
        name = None
        for mem in memories:
            if '👤' in mem.get('category', ''):
                content = mem['content'].lower()
                for pattern in ['me llamo ', 'mi nombre es ', 'soy ']:
                    if pattern in content:
                        idx = content.find(pattern) + len(pattern)
                        name = mem['content'][idx:].split()[0].strip('.,!?')
                        break
        
        t = prompt.lower()
        if any(x in t for x in ['hola', 'buenos', 'hey']):
            full_response = f"¡Hola{' ' + name if name else ''}! ¿En qué puedo ayudarte?"
        elif 'cómo me llamo' in t or 'como me llamo' in t:
            full_response = f"Te llamas **{name}**." if name else "No me has dicho tu nombre aún."
        elif 'qué sabes' in t or 'que sabes' in t:
            if memories:
                phi_avg = sum(float(m.get('phi', '0').replace(',', '.')) for m in memories) / len(memories)
                full_response = f"Tengo **{len(memories)} recuerdos**. PHI promedio: {phi_avg:.3f}"
            else:
                full_response = "Aún no sé nada de ti. ¡Cuéntame!"
        elif should_save:
            full_response = f"✅ Entendido. He guardado eso en mi memoria como **{metrics['category']}**."
        else:
            full_response = "Entendido. (Configura OpenAI para respuestas más inteligentes)"
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Recargar para mostrar todo
    st.rerun()

# --- FOOTER ---
st.divider()
st.caption("🔮 **Infinito UI** - Memoria selectiva basada en IIT (Integrated Information Theory)")
st.caption("💡 Tip: Prueba diciendo 'Me llamo [tu nombre]' y luego pregunta '¿Cómo me llamo?'")
