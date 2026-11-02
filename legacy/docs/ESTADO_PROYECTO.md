# 🧠 INFINITO - Estado del Proyecto
**Fecha:** 27 de Noviembre, 2025  
**Versión:** Beta v0.1  
**Repositorio:** [github.com/webtilians/principiodelTodo](https://github.com/webtilians/principiodelTodo)

---

## 🎯 Visión: "Brain Motor for LLMs"

INFINITO es un **motor cerebral** para LLMs que proporciona:
- **Memoria selectiva** basada en Integrated Information Theory (IIT)
- **Aprendizaje continuo** sin olvido catastrófico (LoRA + Replay Buffer)
- **Objetivos persistentes** con disparo automático/manual
- **Búsqueda semántica** (RAG) con vectores OpenAI

---

## 🏗️ Arquitectura Actual

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ TrivialityGate│  │  IIT Gate    │  │   GoalManager        │   │
│  │ (Red Neuronal)│  │ (PHI,Φ,Coh) │  │ (Objetivos)          │   │
│  │ 100% accuracy │  │              │  │ 3 activos            │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────┘   │
│         │                  │                                      │
│         ▼                  ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │               DECISIÓN: ¿GUARDAR?                          │  │
│  │  if (not trivial) AND (combined > 0.3 OR category > 0.3)  │  │
│  │     AND (not pregunta OR pregunta_interés)                │  │
│  └────────────────────────────────────────────────────────────┘  │
│                              │                                    │
│         ┌────────────────────┼────────────────────┐              │
│         ▼                    ▼                    ▼              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐    │
│  │ Neural Memory│     │Vector Store │     │   Replay        │    │
│  │ (LoRA+Golden │     │(OpenAI Emb) │     │   Buffer        │    │
│  │  Seed 54%)   │     │1536 dims    │     │  40 experiencias│    │
│  └─────────────┘     └─────────────┘     └─────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Modelos Entrenados

| Modelo | Archivo | Descripción | Uso |
|--------|---------|-------------|-----|
| **TrivialityGate** | `models/triviality_gate.pt` | Red neuronal que detecta saludos/cortesías | Filtrar "Hola", "ok", etc. |
| **IIT Gate v3** | `models/dynamic_chat_detector_v3.pt` | Modelo principal con métricas IIT | Calcular PHI, Coherence |
| **Golden Seed 54%** | `models/super_golden_seed_54percent.pt` | Semilla optimizada (54.35% mejora) | Base congelada para LoRA |

---

## 🧠 Componentes Principales

### 1. **TrivialityGate** (NUEVO)
```python
# Red neuronal para detectar frases triviales
# Reemplaza el diccionario hardcoded (chapuza anterior)
class TrivialityGate(nn.Module):
    # Embedding → Transformer → MLP → Sigmoid
    # Output: 0.0 = trivial, 1.0 = importante
```
- **Accuracy:** 100%
- **Entrenamiento:** 48 triviales + 27 importantes
- **Triviales → 0.000, Importantes → 1.000**

### 2. **Neural Memory Manager** (LoRA + Replay)
```python
# Aprendizaje continuo sin olvido catastrófico
class NeuralMemoryManager:
    - Base: Golden Seed 54% (CONGELADO)
    - LoRA Adapters: 8,192 params entrenables
    - Replay Buffer: 40 experiencias
    - Consolidación: cada 50 interacciones
```

### 3. **GoalManager** (Objetivos Persistentes)
```python
# Gestión de objetivos con disparo automático
class GoalManager:
    - Tipos: reminder, learning, habit, project
    - Prioridades: 1 (baja) a 4 (crítica)
    - Disparo: por fecha o keywords
    - Persistencia: goals.json
```

### 4. **Vector Engine** (RAG)
```python
# Búsqueda semántica con OpenAI embeddings
- Modelo: text-embedding-3-small (1536 dims)
- Similitud: Coseno
- Almacén: memoria_permanente.json
```

---

## 📊 Estado de Datos

| Recurso | Cantidad | Descripción |
|---------|----------|-------------|
| **Objetivos** | 3 | Recordatorios activos |
| **Replay Buffer** | 40 | Experiencias para consolidación |
| **Memorias Vectoriales** | 1+ | Con embeddings de 1536 dims |

---

## 🔄 Flujo de Procesamiento

```
Usuario escribe "Hola"
       │
       ▼
[1] TrivialityGate → importance=0.000 → ❌ TRIVIAL
       │
       └─→ "🧠 No guardado - Trivial (detectado por NN)"

Usuario escribe "Me llamo Enrique"
       │
       ▼
[1] TrivialityGate → importance=1.000 → ✓ IMPORTANTE
       │
       ▼
[2] IIT Gate → PHI=0.5, Coherence=0.8, Combined=0.65
       │
       ▼
[3] Categoría → "👤 Identidad" → Bonus=0.5
       │
       ▼
[4] Decisión → combined > 0.3 → ✅ GUARDAR
       │
       ├─→ Vector Store (embedding OpenAI)
       ├─→ Neural Memory (LoRA learning)
       └─→ Replay Buffer (consolidación)
```

---

## 🛠️ Archivos Clave

```
principiodelTodo/
├── app.py                      # UI principal Streamlit
├── src/
│   ├── goal_manager.py         # Gestor de objetivos
│   ├── neural_memory.py        # LoRA + Replay Buffer
│   ├── lora_adapter.py         # Implementación LoRA
│   └── vector_engine.py        # Búsqueda semántica
├── models/
│   ├── triviality_gate.pt      # Gate de trivialidades
│   ├── dynamic_chat_detector_v3.pt  # Gate IIT
│   └── super_golden_seed_54percent.pt  # Base para LoRA
├── train_triviality_gate.py    # Entrenamiento del Gate
├── goals.json                  # Objetivos persistentes
└── memoria_permanente.json     # Vectores + contenido
```

---

## 📈 Últimos Commits

```
6a128bc - Refactor: Reemplazar diccionario de trivialidades por red neuronal
b1ccd07 - Fix: Añadir filtro de frases triviales
af19c7f - feat: Add Neural Memory with LoRA + Replay Buffer
b59476d - feat: Add GoalManager + Update README with Brain Motor vision
627a14d - feat: Mejoras en el sistema de memoria inteligente
```

---

## 🎯 Próximos Pasos Sugeridos

1. **Entrenar IIT Gate** con más datos para mejorar discriminación importance
2. **Añadir más datos al TrivialityGate** para cubrir edge cases
3. **Implementar consolidación automática** en background
4. **Dashboard de métricas** para visualizar PHI en tiempo real
5. **Integración con más LLMs** (Gemini, Claude, Grok)

---

## 🚀 Cómo Ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Configurar OpenAI API Key
echo "OPENAI_API_KEY=sk-..." > .env

# Lanzar la app
streamlit run app.py
```

---

*Generado automáticamente el 27/11/2025*
