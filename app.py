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

# Cargar variables de entorno desde .env (para local)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv no instalado, usar otras fuentes

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
MODEL_PATH = "models/dynamic_chat_detector_v3.pt"  # v3: entrenado con 10k+ datos híbridos
DB_FILE = "memoria_permanente.json"  # Ahora usa el archivo vectorial

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
# MOTOR VECTORIAL (Búsqueda Semántica)
# =============================================================================
import numpy as np
import time

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_RETRIES = 3
RETRY_DELAY = 1  # segundos

def get_embedding(text, client, retries=MAX_RETRIES):
    """Convierte texto en un vector de 1536 números con reintentos."""
    text = text.replace("\n", " ")
    for attempt in range(retries):
        try:
            return client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Backoff exponencial
            else:
                st.warning(f"⚠️ Error obteniendo embedding: {str(e)[:50]}")
                return None
    return None

def cosine_similarity(v1, v2):
    """Calcula qué tan parecidos son dos vectores."""
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# =============================================================================
# MODELO INFINITO CON IIT - ARQUITECTURA COMPLETA
# =============================================================================

class IITGuidedMemory(nn.Module):
    """Memoria con priorización por PHI (standalone para Streamlit)."""
    
    def __init__(self, hidden_dim, memory_slots=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots
        
        # Parámetros de memoria
        self.threshold_logit = nn.Parameter(torch.tensor(0.0))
        self.memory_keys = nn.Parameter(torch.randn(memory_slots, hidden_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(memory_slots, hidden_dim) * 0.02)
        
        # Buffers para estadísticas
        self.register_buffer('phi_scores', torch.zeros(memory_slots))
        self.register_buffer('attention_scores', torch.zeros(memory_slots))
        self.register_buffer('access_count', torch.zeros(memory_slots))
        self.register_buffer('timestamps', torch.zeros(memory_slots))
        self.register_buffer('global_time', torch.tensor(0.0))
        self.register_buffer('write_count', torch.tensor(0))
        
        # Proyecciones
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def read(self, query, top_k=5):
        q = self.query_proj(query)
        k = self.key_proj(self.memory_keys)
        scores = torch.matmul(q, k.T) / (self.hidden_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        result = torch.matmul(weights, self.memory_values)
        return result, weights
    
    def write(self, content, priority=1.0):
        idx = int(self.write_count.item()) % self.memory_slots
        with torch.no_grad():
            self.memory_values.data[idx] = content.mean(dim=0) if content.dim() > 1 else content
            self.phi_scores[idx] = priority
            self.write_count += 1
    
    def get_statistics(self):
        return {'utilization': (self.write_count.item() / self.memory_slots), 'size': self.memory_slots}


class ImprovedIITMetrics(nn.Module):
    """Métricas IIT de 4 componentes (standalone para Streamlit)."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Pesos para cada componente
        self.weight_temporal = nn.Parameter(torch.tensor(0.25))
        self.weight_integration = nn.Parameter(torch.tensor(0.25))
        self.weight_complexity = nn.Parameter(torch.tensor(0.25))
        self.weight_attention = nn.Parameter(torch.tensor(0.25))
        
        self.phi_projector = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_states, attention_weights=None):
        batch_size = hidden_states.shape[0]
        
        # Temporal coherence
        if hidden_states.shape[1] > 1:
            temporal = 1 - torch.mean(torch.abs(hidden_states[:, 1:] - hidden_states[:, :-1]))
        else:
            temporal = torch.tensor(0.5)
        
        # Integration
        mean_state = hidden_states.mean(dim=1, keepdim=True)
        integration = 1 / (1 + torch.mean((hidden_states - mean_state) ** 2))
        
        # Complexity
        complexity = torch.sigmoid(hidden_states.std())
        
        # Attention diversity
        if attention_weights is not None and attention_weights.numel() > 0:
            attention_div = torch.mean(-attention_weights * torch.log(attention_weights + 1e-10))
        else:
            attention_div = torch.tensor(0.5)
        
        # PHI combinado
        phi = self.phi_projector(hidden_states.mean(dim=1)).sigmoid().mean()
        
        return {
            'temporal_coherence': temporal,
            'integration_strength': integration,
            'complexity': complexity,
            'attention_diversity': attention_div,
            'phi': phi,
            'coherence': temporal
        }


class InfinitoUIModel(nn.Module):
    """Modelo completo IIT para la UI - Compatible con pesos entrenados."""
    
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
        
        # IIT Components (compatibles con pesos entrenados)
        self.memory = IITGuidedMemory(hidden_dim)
        self.iit_metrics = ImprovedIITMetrics(hidden_dim)
        
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
        
        # IIT Metrics
        iit_results = self.iit_metrics(hidden, None)
        
        # Calcular importancia con el gate
        importance_logit = self.importance_gate(query)
        importance = torch.sigmoid(importance_logit)
        
        memory_gate_value = torch.sigmoid(self.memory_gate)
        
        # Extraer métricas
        phi = iit_results['phi'] if torch.is_tensor(iit_results['phi']) else torch.tensor(iit_results['phi'])
        coherence = iit_results['coherence'] if torch.is_tensor(iit_results['coherence']) else torch.tensor(iit_results['coherence'])
        complexity = iit_results['complexity'] if torch.is_tensor(iit_results['complexity']) else torch.tensor(iit_results['complexity'])
        
        logits = self.output(hidden)
        
        if return_metrics:
            metrics = {
                'importance': importance.mean().item(),
                'gate_value': importance.mean().item() * 100,
                'memory_gate': memory_gate_value.item(),
                'phi': phi.item() if torch.is_tensor(phi) else float(phi),
                'coherence': coherence.item() if torch.is_tensor(coherence) else float(coherence),
                'complexity': complexity.item() if torch.is_tensor(complexity) else float(complexity),
                'integration': iit_results.get('integration_strength', torch.tensor(0.5)).item() if torch.is_tensor(iit_results.get('integration_strength', 0.5)) else 0.5
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


def save_memory(text, metrics, openai_client=None):
    """Guarda una memoria con métricas IIT y vector semántico."""
    memories = get_memories()
    
    # Detectar categoría (antes de transformar)
    category = detect_category(text)
    
    # Transformar primera persona a tercera (excepto para identidad)
    # "El viernes voy a montar" → "El viernes Enrique va a montar"
    if category != "👤 Identidad":  # No transformar "Me llamo X"
        text_to_save = transform_first_to_third_person(text)
    else:
        text_to_save = text
    
    # Generar vector si tenemos cliente OpenAI
    vector = None
    if openai_client:
        vector = get_embedding(text_to_save, openai_client)
    
    entry = {
        "id": len(memories) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "content": text_to_save,
        "category": category,
        "score": f"{metrics['importance'] * 100:.1f}%",
        "importance": f"{metrics['importance'] * 100:.1f}%",
        "phi": f"{metrics['phi']:.3f}",
        "coherence": f"{metrics['coherence']:.3f}",
        "combined_score": f"{(metrics['importance'] + metrics['phi']) / 2 * 100:.1f}%"
    }
    
    # Añadir vector solo si se generó
    if vector:
        entry["vector"] = vector
    
    memories.append(entry)
    
    with open(DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)
    
    return entry


def semantic_search(query, openai_client, top_k=3):
    """Busca los recuerdos más relevantes semánticamente."""
    memories = get_memories()
    if not memories or not openai_client:
        return []
    
    # Vectorizar la pregunta
    query_vector = get_embedding(query, openai_client)
    if not query_vector:
        return []
    
    # Comparar con todos los recuerdos que tengan vector
    scored = []
    for mem in memories:
        if 'vector' in mem:
            sim = cosine_similarity(query_vector, mem['vector'])
            scored.append((sim, mem))
    
    # Ordenar por similitud
    scored.sort(key=lambda x: x[0], reverse=True)
    
    return [(score, mem) for score, mem in scored[:top_k]]


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
    
    # Signos de interrogación explícitos
    if '?' in t or '¿' in t:
        return True
    
    # Palabras interrogativas al inicio
    starts = ['que ', 'qué ', 'como ', 'cómo ', 'cual ', 'cuál ', 
              'quien ', 'quién ', 'donde ', 'dónde ', 'cuando ', 
              'cuándo ', 'sabes ', 'recuerdas ', 'dime ', 'cuanto ', 'cuánto ',
              'a que ', 'a qué ', 'a cual ', 'a cuál ', 'a donde ', 'a dónde ',
              'por que ', 'por qué ', 'para que ', 'para qué ',
              'en que ', 'en qué ', 'de que ', 'de qué ', 'con que ', 'con qué ']
    for s in starts:
        if t.startswith(s):
            return True
    
    # Patrones de pregunta sobre información guardada
    question_patterns = ['me gusta ir', 'suelo ir', 'voy a', 'tengo que',
                         'cuál es mi', 'cual es mi', 'qué es mi', 'que es mi',
                         'cómo se llama', 'como se llama', 'cuándo es', 'cuando es']
    # Si pregunta sobre SÍ MISMO en forma de pregunta
    if any(p in t for p in question_patterns) and any(w in t for w in ['qué', 'que', 'cuál', 'cual', 'cuándo', 'cuando', 'a qué', 'a que']):
        return True
    
    return False


def get_user_name():
    """Extrae el nombre del usuario de las memorias guardadas."""
    memories = get_memories()
    import re
    for mem in memories:
        content = mem.get('content', '').lower()
        # Buscar patrones como "me llamo X" o "mi nombre es X"
        patterns = [
            r'me llamo\s+([a-záéíóúñ]+)',
            r'mi nombre es\s+([a-záéíóúñ]+)',
            r'soy\s+([a-záéíóúñ]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
    return None


def transform_first_to_third_person(text, user_name=None):
    """
    Transforma oraciones de primera persona a tercera persona.
    Ejemplo: "El viernes voy a montar" → "El viernes Enrique va a montar"
    """
    if not user_name:
        user_name = get_user_name()
    
    # Si no sabemos el nombre, usar "El usuario"
    subject = user_name if user_name else "El usuario"
    
    import re
    t = text.strip()
    original = t
    
    # Transformaciones de primera persona a tercera
    transformations = [
        # Verbos comunes en primera persona → tercera
        (r'\bvoy a\b', f'{subject} va a'),
        (r'\bvoy\b', f'{subject} va'),
        (r'\btengo\b', f'{subject} tiene'),
        (r'\bquiero\b', f'{subject} quiere'),
        (r'\bnecesito\b', f'{subject} necesita'),
        (r'\bestoy\b', f'{subject} está'),
        (r'\bsoy\b', f'{subject} es'),
        (r'\bhago\b', f'{subject} hace'),
        (r'\bpuedo\b', f'{subject} puede'),
        (r'\bsé\b', f'{subject} sabe'),
        (r'\bveo\b', f'{subject} ve'),
        (r'\bcreo\b', f'{subject} cree'),
        (r'\bpienso\b', f'{subject} piensa'),
        (r'\bme gusta\b', f'A {subject} le gusta'),
        (r'\bme encanta\b', f'A {subject} le encanta'),
        (r'\bprefiero\b', f'{subject} prefiere'),
        (r'\bsuelo\b', f'{subject} suele'),
        (r'\bvivo\b', f'{subject} vive'),
        (r'\btrabajo\b', f'{subject} trabaja'),
        (r'\bestudio\b', f'{subject} estudia'),
        (r'\bjuego\b', f'{subject} juega'),
        (r'\bleo\b', f'{subject} lee'),
        (r'\bescucho\b', f'{subject} escucha'),
        (r'\bmiro\b', f'{subject} mira'),
        (r'\bcomo\b', f'{subject} come'),
        (r'\bbebo\b', f'{subject} bebe'),
        (r'\bduermo\b', f'{subject} duerme'),
        (r'\bcorro\b', f'{subject} corre'),
        (r'\bcamino\b', f'{subject} camina'),
        (r'\bmanejo\b', f'{subject} maneja'),
        (r'\bconduzco\b', f'{subject} conduce'),
        (r'\bsalgo\b', f'{subject} sale'),
        (r'\bllego\b', f'{subject} llega'),
        (r'\bvengo\b', f'{subject} viene'),
    ]
    
    for pattern, replacement in transformations:
        t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
    
    # Transformar pronombres posesivos DESPUÉS de los verbos
    # Solo si hubo cambios en verbos (significa que era primera persona)
    if t != original:
        possessive_transforms = [
            (r'\bmi\b', 'su'),
            (r'\bmis\b', 'sus'),
            (r'\bme\b', 'le'),
            (r'\bconmigo\b', f'con {subject}'),
        ]
        for pattern, replacement in possessive_transforms:
            t = re.sub(pattern, replacement, t, flags=re.IGNORECASE)
        return t
    return text


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


def construct_prompt(user_query=None, openai_client=None):
    """Construye el prompt con memoria para GPT usando búsqueda semántica."""
    
    # Si tenemos query y cliente, hacer búsqueda semántica
    if user_query and openai_client:
        results = semantic_search(user_query, openai_client, top_k=5)
        
        if results:
            memory_block = "🔍 RECUERDOS RELEVANTES (búsqueda semántica):\n"
            for score, mem in results:
                memory_block += f"  • [{score:.2f}] {mem['content']} [{mem.get('category', 'General')}]\n"
        else:
            memory_block = "No encontré recuerdos relacionados con este tema."
    else:
        # Fallback: mostrar últimas memorias
        memories = get_memories()
        if not memories:
            memory_block = "NO TIENES RECUERDOS PREVIOS."
        else:
            sorted_mems = sorted(memories, key=lambda x: float(x.get('phi', '0').replace(',', '.')), reverse=True)
            memory_block = "🧠 MEMORIA A LARGO PLAZO:\n"
            for mem in sorted_mems[-10:]:
                memory_block += f"  • {mem['content']} [{mem.get('category', 'General')}]\n"
    
    return f"""Eres Infinito, un asistente con MEMORIA PERSISTENTE y búsqueda semántica vectorial.

{memory_block}

🚨 REGLAS ABSOLUTAS (NUNCA VIOLAR):

1. **SOLO USA INFORMACIÓN EXPLÍCITA** - Si algo NO está escrito exactamente en los recuerdos, NO lo sabes
2. **NO COMBINES RECUERDOS** - Cada recuerdo es 100% independiente. NUNCA mezcles datos de diferentes recuerdos
3. **PREFERENCIA ≠ EVENTO** - "Le gusta ir al mediodía" NO significa "va a ir". Solo indica preferencia habitual
4. **FECHA/HORA ESPECÍFICA** - Si preguntan "¿cuándo?" o "¿el viernes?", solo responde si ESA fecha está en UN recuerdo específico
5. **NO INFERIR** - Si un recuerdo dice "A le gusta X" y otro dice "B hace X el viernes", NO concluyas que A hace X el viernes
6. **PERSONA CORRECTA** - Verifica que el SUJETO de la pregunta coincide con el SUJETO del recuerdo
7. **DISTINGUE ROLES** - "El restaurante DE Juan" significa que Juan es DUEÑO, no que Juan VA al restaurante

📌 TIPOS DE INFORMACIÓN:
- HECHO: "El viernes voy a montar en bici" → EVENTO con fecha específica
- PREFERENCIA: "A mi padre le gusta ir al restaurante" → HÁBITO de MI PADRE (no del dueño)
- IDENTIDAD: "Mi padre se llama Juan" → DATO permanente
- PROPIEDAD: "El restaurante de mi hermano" → Mi hermano es DUEÑO del restaurante

❌ ERRORES PROHIBIDOS:
- Pregunta: "¿A qué hora le gusta ir a Juan al restaurante?"
- Recuerdos: "A mi padre le gusta ir al restaurante de mi hermano al mediodía" + "Mi hermano se llama Juan"
- ❌ INCORRECTO: "A Juan le gusta ir al mediodía" (¡JUAN ES EL DUEÑO, no el que va!)
- ✅ CORRECTO: "No tengo información de que a Juan le guste ir al restaurante. Juan es el dueño del restaurante Sake Izakaya. Quien va al mediodía es tu padre."

- Pregunta: "¿El viernes va mi padre al restaurante?"
- Recuerdos: "El viernes voy en bici con mi padre" + "A mi padre le gusta ir al restaurante al mediodía"
- ❌ INCORRECTO: "Sí, el viernes va" (INVENTADO - ningún recuerdo dice eso)
- ✅ CORRECTO: "No tengo información de que tu padre vaya al restaurante el viernes. Solo sé que le gusta ir al mediodía normalmente."

- Pregunta: "¿Cuándo va Andrés a montar en bici?"
- Recuerdos: "Mi primo Andrés monta en bici" + "El viernes voy en bici con mi padre"
- ❌ INCORRECTO: "Andrés va el viernes" (MEZCLÓ recuerdos de diferentes personas)
- ✅ CORRECTO: "Solo sé que a tu primo Andrés le gusta montar en bici, pero no tengo guardado cuándo específicamente."

⚠️ ANTES DE RESPONDER, VERIFICA:
1. ¿Quién es el SUJETO de la pregunta? (ej: "Juan")
2. ¿Quién es el SUJETO del recuerdo? (ej: "mi padre", no Juan)
3. ¿Coinciden? Si NO, no uses ese recuerdo.

Responde de forma natural en español. Si no tienes la información, di claramente "No tengo esa información guardada"."""


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
                    st.markdown(f"**{mem.get('category', '📝 General')}**")
                    st.caption(f"📅 {mem.get('timestamp', 'N/A')}")
                with col2:
                    st.markdown(f"PHI: `{mem.get('phi', '0.000')}`")
                
                st.text_area(
                    label=f"mem_{mem.get('id', 0)}", 
                    value=mem.get('content', ''), 
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
                    st.success(f"💾 Guardado con Vector: {analysis['category']}")
                elif analysis['is_question']:
                    st.warning("❓ Pregunta - Búsqueda Semántica")
                else:
                    st.info("🔇 No guardado - Información trivial")
                
                # Mostrar resultados de búsqueda semántica si existen
                if 'semantic_results' in analysis and analysis['semantic_results']:
                    st.caption("🔍 **Búsqueda Semántica:**")
                    for sim, content in analysis['semantic_results']:
                        st.caption(f"  [{sim}] {content}...")
                
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
    
    # Guardar en memoria si es importante (con vector si hay OpenAI)
    if should_save:
        save_memory(prompt, metrics, st.session_state.get("openai_client"))
    
    # Guardar resultados de búsqueda semántica para mostrar
    semantic_results = []
    if st.session_state.get("openai_client"):
        semantic_results = semantic_search(prompt, st.session_state["openai_client"], top_k=3)
        analysis_entry['semantic_results'] = [(f"{s:.3f}", m['content'][:50]) for s, m in semantic_results]
    
    # --- GENERAR RESPUESTA ---
    full_response = ""
    
    # Usar OpenAI si está disponible
    if st.session_state.get("openai_client"):
        # Reintentos para GPT
        for attempt in range(MAX_RETRIES):
            try:
                # Usar búsqueda semántica para el prompt
                system_prompt = construct_prompt(prompt, st.session_state["openai_client"])
                
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
                break  # Éxito, salir del loop
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    full_response = f"❌ Error con OpenAI después de {MAX_RETRIES} intentos: {str(e)[:100]}"
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
