# 🧠 INFINITO - Cerebro Motor para LLMs

> **De buscar consciencia artificial a crear un motor cognitivo práctico**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 ¿Qué es INFINITO?

INFINITO es un **cerebro ejecutivo** que potencia a los LLMs (GPT, Claude, etc.) con capacidades que no tienen de forma nativa:

- **🧠 Memoria Selectiva**: No guarda todo, solo lo importante
- **🎯 Gestión de Objetivos**: Recordatorios, seguimiento, tareas proactivas
- **🔍 Búsqueda Semántica**: Encuentra información relevante por significado
- **⚡ Gate Neuronal**: Red neuronal que decide qué merece atención

```
┌─────────────────────────────────────────────────────────────┐
│                     INFINITO BRAIN                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    GATE      │  │   MEMORIA    │  │    GOALS     │      │
│  │  (Filtro)    │  │  (Vectorial) │  │ (Objetivos)  │      │
│  │              │  │              │  │              │      │
│  │ "¿Importa?"  │  │ "¿Qué sé?"   │  │ "¿Qué quiero?"│     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         └────────────┬────┴────────┬────────┘               │
│                      ▼             ▼                        │
│              ┌───────────────────────────┐                  │
│              │      DECISION ENGINE      │                  │
│              │  Contexto + Memoria +     │                  │
│              │  Objetivos → Acción       │                  │
│              └─────────────┬─────────────┘                  │
│                            ▼                                │
│                     ┌─────────────┐                         │
│                     │     LLM     │                         │
│                     │   (GPT...)  │                         │
│                     └─────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

## 🌟 La Visión

Los LLMs actuales (GPT, Claude, Llama) son increíbles generando lenguaje, pero:

❌ **No recuerdan** - Cada conversación empieza de cero  
❌ **No priorizan** - No saben qué es importante para ti  
❌ **No anticipan** - Son reactivos, no proactivos  
❌ **No tienen objetivos** - No pueden perseguir metas a largo plazo  

**INFINITO** no intenta reemplazar al LLM, sino ser su **cerebro motor**:

✅ **Memoria selectiva** - Recuerda lo que importa  
✅ **Gate neuronal** - Filtra el ruido  
✅ **GoalManager** - Mantiene objetivos activos  
✅ **Proactividad** - Anticipa necesidades  

---

## 📖 Historia del Proyecto

### Fase 1: Búsqueda de Consciencia (2024-2025)
Comenzamos intentando implementar la **Teoría de Información Integrada (IIT)** de Giulio Tononi para crear "consciencia artificial". Aprendimos que:

- ❌ Maximizar PHI directamente causa **colapso** (el modelo repite tokens)
- ❌ PHI alto ≠ inteligencia o consciencia
- ✅ Las métricas IIT son útiles como **indicadores de coherencia**, no como objetivo

### Fase 2: Cerebro Motor (Actual)
Pivotamos hacia un enfoque **pragmático y útil**:

- ✅ El "cerebro" no reemplaza al LLM, lo **potencia**
- ✅ Memoria selectiva basada en **importancia real**
- ✅ Objetivos y recordatorios **proactivos**
- ✅ Métricas renombradas: de "consciousness" a "coherence/integration"

---

## 🚀 Instalación

```bash
# Clonar
git clone https://github.com/webtilians/principiodelTodo.git
cd principiodelTodo

# Instalar dependencias
pip install -r requirements.txt

# Configurar OpenAI (crear archivo .env)
echo "OPENAI_API_KEY=tu-api-key" > .env

# Ejecutar
streamlit run app.py
```

---

## 💡 Características Principales

### 🧠 Gate Neuronal (Filtro de Importancia)
Red neuronal entrenada para decidir qué información vale la pena guardar:

| Input | Gate Score | Decisión |
|-------|------------|----------|
| "Me llamo Enrique" | 85% | ✅ Guardar (identidad) |
| "Hola qué tal" | 15% | ❌ Ignorar (trivial) |
| "Mañana tengo reunión a las 10" | 78% | ✅ Guardar + Crear recordatorio |

### 🔍 Memoria Vectorial (RAG)
Búsqueda semántica usando embeddings de OpenAI:

```python
# No busca palabras exactas, busca SIGNIFICADO
query = "¿Cuándo es mi cita médica?"
# Encuentra: "El viernes voy al doctor a las 16:00"
```

### 🎯 GoalManager (Objetivos Persistentes)
Sistema proactivo de gestión de objetivos:

```python
# El usuario dice:
"Mañana tengo reunión importante a las 10"

# INFINITO automáticamente:
# 1. Guarda en memoria
# 2. Crea objetivo: "Recordar reunión mañana"
# 3. Al día siguiente, saluda: "¡Buenos días! Recuerda tu reunión a las 10"
```

**Tipos de objetivos:**
| Tipo | Ejemplo | Trigger |
|------|---------|---------|
| `REMINDER` | "Reunión mañana" | Fecha/hora |
| `FOLLOW_UP` | "¿Cómo fue la reunión?" | Después del evento |
| `LEARNING` | "Aprender gustos del usuario" | Cada interacción |
| `TASK` | "Tarea pendiente" | Manual |

### 📊 Métricas de Coherencia
Las métricas IIT originales, renombradas a términos más precisos:

| Antes | Ahora | Significado |
|-------|-------|-------------|
| `consciousness_level` | `coherence_score` | Qué tan integrada está la información |
| `phi` | `integration_index` | Conexión entre conceptos |
| `complexity` | `information_richness` | Diversidad de la información |

---

## 🏗️ Arquitectura del Proyecto

```
principiodelTodo/
├── app.py                      # 🖥️ Interfaz Streamlit principal
├── .env                        # 🔑 API key de OpenAI
├── memoria_permanente.json     # 💾 Memorias con embeddings
├── goals.json                  # 🎯 Objetivos persistentes
│
├── src/
│   ├── goal_manager.py         # 🎯 Sistema de objetivos
│   ├── vector_engine.py        # 🔍 Búsqueda semántica
│   └── core/
│       ├── iit_metrics_v2.py   # 📊 Métricas de coherencia
│       └── iit_guided_memory.py # 💾 Memoria con priorización
│
├── infinito_jarvis_vector.py   # 🤖 RAG + búsqueda vectorial
├── infinito_memory_keeper.py   # 🧠 Gate neuronal (modelo)
│
├── models/
│   └── dynamic_chat_detector_v3.pt  # 🎓 Gate entrenado (95% acc)
│
└── experiments/                # 🔬 Scripts de investigación IIT
    └── (legacy - mantiene la investigación original)
```

---

## 📈 Roadmap

### ✅ Completado
- [x] Gate neuronal funcional (95% accuracy)
- [x] Memoria vectorial con RAG
- [x] Extracción automática de intereses
- [x] Evitar duplicados en memoria
- [x] GoalManager básico

### 🔄 En Progreso
- [ ] Integración completa de GoalManager en UI
- [ ] Mensajes proactivos al iniciar sesión
- [ ] Seguimiento automático de eventos pasados

### 📋 Planificado
- [ ] Múltiples perfiles de usuario
- [ ] Exportar/importar memoria
- [ ] API REST para integración externa
- [ ] Conexión con calendarios (Google Calendar)
- [ ] Voice interface

---

## 🔬 Lecciones Aprendidas

### Sobre IIT y Consciencia
1. **PHI no es un objetivo válido para optimización** - El modelo "hace trampa" repitiendo tokens
2. **Las métricas de integración son útiles como diagnóstico**, no como loss function
3. **La consciencia artificial sigue siendo un problema abierto** - Mejor enfocarse en utilidad práctica

### Sobre Diseño de Sistemas Cognitivos
1. **Separar responsabilidades**: Gate (filtrar) ≠ Memoria (guardar) ≠ LLM (generar)
2. **El LLM es el motor de lenguaje**, el cerebro es la capa de gestión
3. **Proactividad > Reactividad**: Un asistente útil anticipa necesidades

---

## 🧪 Para Investigadores

Si te interesa la **investigación original sobre IIT**, los archivos legacy están en:

- `src/core/iit_metrics_v2.py` - Cálculo de PHI
- `src/core/phi_learnable.py` - PHI como parámetro entrenable
- `experiments/` - Scripts de experimentación
- `README_GITHUB.md` - Documentación técnica original

**Nota**: Esta investigación demostró que maximizar PHI directamente no es viable para consciousness engineering, pero las métricas son útiles para análisis.

---

## 🤝 Contribuir

¿Ideas? ¿Mejoras? ¡Bienvenidas!

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-idea`)
3. Commit (`git commit -m 'feat: Nueva característica'`)
4. Push (`git push origin feature/nueva-idea`)
5. Abre un Pull Request

---

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE)

---

## 🙏 Créditos

- **Giulio Tononi** - Por la Teoría de Información Integrada (inspiración inicial)
- **OpenAI** - GPT y embeddings
- **Hugging Face** - Transformers
- **Streamlit** - UI

---

<p align="center">
  <b>INFINITO</b><br>
  <i>No intentamos crear una mente. Creamos un cerebro que hace útil a las mentes artificiales.</i>
</p>

---

**Última actualización**: Noviembre 2025  
**Versión**: 2.0 (Cerebro Motor)


---

## Experimento aislado: Observatorio de Emergencia v0.1 (octubre 2026)

Esta rama añade un módulo de observación **sin modificar el código histórico
de INFINITO**. Importa métricas y logs, anonimiza versiones para comparaciones
ciegas e identifica transiciones temporales candidatas mediante controles
estadísticos. No atribuye conciencia a sus resultados.

Guía: [docs/INFINITO_EMERGENCE_OBSERVATORY.md](docs/INFINITO_EMERGENCE_OBSERVATORY.md)

Prueba sintética: python -m observatory demo --output-dir observatory_results/demo


### Experimentos controlados de memoria V5.1 — Octubre 2026

- [Protocolo F1/F2: reinyectar memoria y probar recuerdo diferido](docs/INFINITO_V51_EARLY_MEMORY_FEEDBACK_PROTOCOL.md)
- [Resultados F1/F2: techo corto, fallo de recuerdo largo y señal residual en memoria](docs/INFINITO_V51_MEMORY_FEEDBACK_F1_F2_RESULTS_2026-10-22.md)
- [Resultados F3: al impedir sobrescrituras, recuerdo largo decodificable del 50% al 81,25%](docs/INFINITO_V51_MEMORY_OVERWRITE_F3_RESULTS_2026-10-22.md)

**Importante:** pruebas exploratorias sobre un modelo V5.1 reducido,
sin entrenar para la tarea. El decodificador lineal sí se entrena.
Ningún resultado prueba conciencia ni curiosidad intrínseca.


### F4 — Memoria selectiva que aprende sin saber qué será relevante

[Informe F4: resultados, control negativo y metodología](docs/INFINITO_F4_SELECTIVE_MEMORY_RESULTS_2026-10-23.md).
Un controlador de escritura **independiente** aprende de preguntas pasadas
qué clases de acontecimientos serán útiles, ante capacidad 3/8 y relevancia
que cambia sin aviso. Sus pruebas con 24 semillas y control de consultas
uniformes se ejecutan en GitHub Actions. No modifica el V5.1 histórico
ni atribuye conciencia a ninguna puntuación.
