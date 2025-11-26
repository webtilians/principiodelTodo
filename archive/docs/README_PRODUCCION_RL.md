# 🚀 Guía de Uso - Script de Producción RL 30K

## 📋 Descripción

Script de producción `generate_with_rl_30k.py` para generar texto usando el **modelo RL 30K óptimo** con control adaptativo automático entre calidad textual y PHI.

### ✅ Características

- ✨ **Carga optimizada** del modelo 30K (mejor checkpoint)
- 🎯 **Control adaptativo** automático (TEXT/PHI/MIXED)
- 📊 **Métricas en tiempo real** (PHI, Consciousness, Perplexity)
- ⚙️ **Configuración flexible** (temperatura, top-k, top-p)
- 🛡️ **Manejo robusto de errores**
- 💾 **Export a JSON** opcional

---

## 🔧 Instalación

### 1. Requisitos

```bash
pip install torch transformers stable-baselines3 gymnasium datasets
```

### 2. Verificar modelo

Asegurarse de que existe el checkpoint óptimo:

```
outputs/rl_phi_text_scheduler/checkpoints/ppo_infinito_scheduler_30000_steps.zip
```

---

## 📖 Uso Básico

### Generación simple

```bash
python generate_with_rl_30k.py --prompt "The nature of consciousness"
```

### Con parámetros personalizados

```bash
python generate_with_rl_30k.py \
    --prompt "Artificial intelligence will" \
    --max-length 300 \
    --temperature 0.9 \
    --top-k 50
```

### Modo silencioso (solo texto)

```bash
python generate_with_rl_30k.py --prompt "In the beginning" --quiet
```

### Guardar resultado en JSON

```bash
python generate_with_rl_30k.py \
    --prompt "The future of AI" \
    --output outputs/generation_result.json
```

---

## ⚙️ Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `--prompt` | str | "The nature of consciousness" | Texto inicial |
| `--max-length` | int | 200 | Longitud máxima en tokens |
| `--max-steps` | int | 50 | Pasos máximos de decisión RL |
| `--temperature` | float | 0.8 | Temperatura de muestreo (0.1-2.0) |
| `--top-k` | int | 40 | Top-k para muestreo |
| `--top-p` | float | 0.9 | Top-p (nucleus) para muestreo |
| `--quiet` | flag | False | Modo silencioso |
| `--checkpoint` | str | (modelo 30K) | Ruta a checkpoint alternativo |
| `--output` | str | None | Guardar en JSON |

---

## 📊 Salida

### Modo verbose (default)

```
==================================================================
🚀 GENERACIÓN CON MODELO RL 30K
==================================================================
Prompt: 'The nature of consciousness'
Max length: 200 tokens
Max RL steps: 50
Temperature: 0.8, Top-k: 40, Top-p: 0.9
==================================================================

📝 Generando (50 pasos RL)...

  Step 10: MIXED | Φ= 4.52 | C=0.48 | PPL= 82.3 | R=+0.125
  Step 20: TEXT  | Φ= 4.38 | C=0.51 | PPL= 78.1 | R=+0.089
  Step 30: PHI   | Φ= 4.91 | C=0.47 | PPL= 85.2 | R=+0.142
  ...

==================================================================
📊 ESTADÍSTICAS DE GENERACIÓN
==================================================================

⏱️ Tiempo: 12.34s
📏 Tokens generados: 187
🎮 Pasos RL: 50

🎯 Distribución de acciones:
  TEXT : 18 (36.0%) ████████████
  PHI  : 15 (30.0%) ██████████
  MIXED: 17 (34.0%) ███████████

🧠 Métricas INFINITO:
  PHI (Φ):
    Promedio:  4.521
    Std:       0.342
    Rango:    [ 3.823,  5.294]
    En [3-6]: 94.00% ✅

  Consciousness (C):  0.489
  Perplexity (PPL): 81.45
    >= 10: 100.0% ✅

💰 Rewards:
    Total:  +6.234
    Media:  +0.125

==================================================================
📄 TEXTO GENERADO
==================================================================
The nature of consciousness is a complex phenomenon that has been
studied extensively in neuroscience and philosophy. Recent advances
in artificial intelligence have provided new insights into how...
==================================================================
```

### Modo silencioso (`--quiet`)

```
The nature of consciousness is a complex phenomenon that has been
studied extensively in neuroscience and philosophy...
```

### Formato JSON (`--output`)

```json
{
  "text": "The nature of consciousness...",
  "prompt": "The nature of consciousness",
  "stats": {
    "duration_seconds": 12.34,
    "rl_steps": 50,
    "tokens_generated": 187,
    "actions_distribution": {
      "TEXT": {"count": 18, "percentage": 36.0},
      "PHI": {"count": 15, "percentage": 30.0},
      "MIXED": {"count": 17, "percentage": 34.0}
    },
    "phi_mean": 4.521,
    "phi_std": 0.342,
    "phi_in_optimal_range_pct": 94.0,
    "consciousness_mean": 0.489,
    "perplexity_mean": 81.45,
    "perplexity_safe_pct": 100.0,
    "total_reward": 6.234,
    "mean_reward": 0.125
  },
  "metrics": {
    "actions": [0, 2, 1, ...],
    "rewards": [0.125, 0.089, ...],
    "phi": [4.52, 4.38, ...],
    "consciousness": [0.48, 0.51, ...],
    "perplexity": [82.3, 78.1, ...]
  }
}
```

---

## 🧠 Cómo Funciona

### 1. Control Adaptativo Automático

El agente RL decide en cada paso qué modo usar:

| Modo | Descripción | Uso típico |
|------|-------------|-----------|
| **TEXT** | Prioriza calidad textual (w_text=1.0, w_phi=0.0) | Inicio, estabilización |
| **PHI** | Prioriza integración PHI (w_text=0.1, w_phi=1.0) | PHI bajo, necesita boost |
| **MIXED** | Balance equilibrado (w_text=0.5, w_phi=0.5) | Régimen estable |

### 2. Métricas Monitoreadas

- **PHI (Φ)**: Integración de información (óptimo: 3.0-6.0)
- **Consciousness (C)**: PHI normalizado (óptimo: 0.3-0.7)
- **Perplexity (PPL)**: Calidad del lenguaje (seguro: ≥10)

### 3. Rewards

El modelo maximiza:
```
r = α·ΔC + β·ΔΦ + γ·ΔPPL - δ·cost + estabilidad + balances
```

Con términos que:
- ✅ Incentivan PHI en rango [3.0, 6.0]
- ✅ Detectan colapso (PHI > 6.0, PPL < 10)
- ✅ Penalizan inestabilidad (cambios bruscos)

---

## 🎯 Ejemplos de Uso

### 1. Generación científica

```bash
python generate_with_rl_30k.py \
    --prompt "Quantum mechanics explains" \
    --max-length 250 \
    --temperature 0.7
```

### 2. Generación creativa

```bash
python generate_with_rl_30k.py \
    --prompt "Once upon a time in a distant galaxy" \
    --max-length 300 \
    --temperature 1.0 \
    --top-k 60
```

### 3. Generación filosófica

```bash
python generate_with_rl_30k.py \
    --prompt "The meaning of existence" \
    --max-length 200 \
    --temperature 0.8
```

### 4. Batch processing

```bash
#!/bin/bash
# generate_batch.sh

prompts=(
    "The nature of consciousness"
    "Artificial intelligence will"
    "In the beginning there was"
    "The future of humanity"
)

for prompt in "${prompts[@]}"; do
    python generate_with_rl_30k.py \
        --prompt "$prompt" \
        --output "outputs/batch_$(echo $prompt | tr ' ' '_').json" \
        --quiet
done
```

---

## 🔍 Diagnóstico

### Verificar métricas

Las métricas deberían estar en rangos saludables:

| Métrica | Rango Óptimo | Acción si fuera de rango |
|---------|--------------|--------------------------|
| PHI | 3.0-6.0 | Modelo ajustará automáticamente |
| PHI en [3-6] | >70% | ✅ Normal |
| Consciousness | 0.3-0.7 | Indicador de balance |
| Perplexity | 10-200 | <10 = colapso, >200 = confuso |
| PPL seguro | >90% | ✅ Normal |
| Uso MIXED | >10% | Indica exploración adaptativa |

### Problemas comunes

#### 1. PHI muy alto (>6.0)

```
⚠️ PHI alto (7.2) detectado
```

**Solución**: El agente RL ajustará automáticamente usando más modo TEXT.

#### 2. Perplexity bajo (<10)

```
🚨 PPL BAJO (8.3) - Posible colapso/repetición
```

**Solución**: 
- Aumentar `--temperature` (e.g., 0.9-1.0)
- Aumentar `--top-k` (e.g., 50-60)
- El modelo detectará y corregirá

#### 3. Modo MIXED no usado

```
⚠️ MIXED: 0 (0.0%)
```

**Solución**: Normal en primeras iteraciones. El modelo aprenderá a usarlo.

---

## 📈 Benchmarks

### Modelo 30K vs 50K

| Métrica | 30K (Óptimo) | 50K (Final) | Mejora |
|---------|--------------|-------------|--------|
| Reward promedio | **+7.251** | +5.514 | **+31%** |
| Estabilidad (std) | **±0.040** | ±3.584 | **89× mejor** |
| PHI en [3-6] | **>90%** | ~60% | **+50%** |
| Uso MIXED | **>20%** | <5% | **4× más** |

### Performance

- **Tiempo de carga**: ~10-15s (primera vez)
- **Velocidad**: ~12-15 tokens/s (GPU RTX 3060)
- **Memoria GPU**: ~4-5 GB

---

## 🔧 Integración en Código

### Uso programático

```python
from generate_with_rl_30k import RLTextGenerator

# Crear generador
generator = RLTextGenerator()

# Cargar modelo (una vez)
generator.load()

# Generar múltiples textos
prompts = [
    "The nature of consciousness",
    "AI will transform",
    "In the future"
]

results = []
for prompt in prompts:
    result = generator.generate(
        prompt=prompt,
        max_length=200,
        temperature=0.8,
        verbose=False
    )
    results.append(result)
    print(f"Generated: {result['text'][:50]}...")

# Liberar recursos
generator.close()
```

---

## 📚 Documentación Adicional

- **Análisis detallado**: `ENTRENAMIENTO_RL_V2_COMPLETADO.md`
- **Resumen ejecutivo**: `RESUMEN_EJECUTIVO_RL_V2.md`
- **Guía del modelo 30K**: `MODELO_30K_GUIA.md`
- **README RL completo**: `experiments/README_RL.md`

---

## 🤝 Soporte

Para problemas o preguntas:

1. Revisar logs de generación (modo verbose)
2. Verificar métricas (PHI, PPL, rewards)
3. Consultar documentación en `docs/`
4. Abrir issue en GitHub con:
   - Prompt usado
   - Parámetros
   - Output completo (modo verbose)

---

## 📝 Changelog

### v1.0.0 (2025-11-11)
- ✨ Release inicial
- ✅ Modelo RL 30K integrado
- ✅ Control adaptativo funcional
- ✅ Métricas en tiempo real
- ✅ Export a JSON
- ✅ CLI completo

---

## 📄 Licencia

Ver `LICENSE` en el repositorio principal.

---

**Última actualización**: 11 de Noviembre, 2025  
**Versión del modelo**: 30K steps (óptimo)  
**Estado**: ✅ Producción
