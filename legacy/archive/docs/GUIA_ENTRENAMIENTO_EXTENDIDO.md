# 🎓 Guía de Entrenamiento Extendido

Esta guía explica las 3 formas de entrenar más el modelo para mejorar la calidad del texto.

---

## 📋 Resumen de Opciones

| Opción | Qué entrena | Tiempo estimado | Mejora esperada | Comando |
|--------|-------------|-----------------|-----------------|---------|
| **1. Continuar RL** | Agente RL desde 30K | 2-4 horas | +10-20% calidad decisiones | `continue_training_rl.py` |
| **2. Entrenar base** | Modelo INFINITO base | 4-8 horas | +30-50% calidad texto | `train_base_more.py` |
| **3. Nuevo RL completo** | Todo desde cero | 6-12 horas | Control total | `train_phi_text_scheduler.py` |

---

## 🔄 OPCIÓN 1: Continuar RL desde 30K (Más rápido)

**¿Cuándo usar?**
- Quieres mejorar las decisiones TEXT/PHI/MIXED
- El texto generado es aceptable pero las decisiones no son óptimas
- Quieres resultados rápidos (2-4 horas)

### Comando básico:
```bash
python continue_training_rl.py --timesteps 50000
```

### Comando con más control:
```bash
python continue_training_rl.py \
  --checkpoint outputs/rl_phi_text_scheduler/checkpoints/ppo_infinito_scheduler_30000_steps.zip \
  --timesteps 100000 \
  --output outputs/rl_continued_100k \
  --save-freq 5000 \
  --eval-freq 5000
```

### Parámetros:
- `--timesteps`: Pasos adicionales a entrenar (50K-200K recomendado)
- `--checkpoint`: Checkpoint base (default: 30K óptimo)
- `--output`: Directorio de salida
- `--save-freq`: Cada cuántos pasos guardar checkpoint
- `--eval-freq`: Cada cuántos pasos evaluar

### Ventajas:
✅ Más rápido (continúa desde donde quedó)  
✅ No pierde el progreso del 30K  
✅ Mejora decisiones adaptativas  

### Desventajas:
⚠️ No mejora la calidad base del texto  
⚠️ Puede sobre-especializarse si entrenas demasiado  

---

## 📚 OPCIÓN 2: Entrenar Modelo Base (Mejor calidad)

**¿Cuándo usar?**
- El texto tiene demasiada repetición
- Quieres mejorar la coherencia y fluidez
- No te importa entrenar 4-8 horas

### Comando básico:
```bash
python train_base_more.py --epochs 20
```

### Comando optimizado (recomendado):
```bash
python train_base_more.py \
  --epochs 30 \
  --batch-size 16 \
  --lr 2e-4 \
  --lambda-phi 0.6 \
  --lora-r 8 \
  --patience 5
```

### Parámetros importantes:
- `--epochs`: Número de épocas (20-50 recomendado)
  - 20 épocas: ~4 horas, mejora moderada
  - 30 épocas: ~6 horas, mejora significativa
  - 50 épocas: ~10 horas, mejora máxima
  
- `--lambda-phi`: Balance entre texto y PHI
  - 0.3: Prioriza calidad de texto
  - 0.6: Balance óptimo (recomendado)
  - 1.0: Prioriza PHI (puede sacrificar fluidez)

- `--lora-r`: Capacidad de adaptación
  - 4: Mínimo, rápido
  - 8: Óptimo (recomendado)
  - 16: Máximo, más lento

### Para continuar desde checkpoint:
```bash
python train_base_more.py \
  --epochs 20 \
  --checkpoint models/checkpoints/infinito_phase2_best.pt
```

### Ventajas:
✅ Mejora drásticamente la calidad del texto  
✅ Reduce repeticiones  
✅ Texto más coherente y natural  

### Desventajas:
⚠️ Tarda más tiempo  
⚠️ Requiere re-entrenar RL después  

---

## 🚀 OPCIÓN 3: Entrenamiento RL Completo Nuevo

**¿Cuándo usar?**
- Quieres cambiar hiperparámetros del RL
- El modelo actual no converge bien
- Quieres experimentar con configuraciones

### Comando básico:
```bash
python experiments/train_phi_text_scheduler.py --timesteps 100000
```

### Comando con configuración personalizada:
```bash
python experiments/train_phi_text_scheduler.py \
  --timesteps 150000 \
  --inner-steps 5 \
  --max-steps 50 \
  --batch-size 4 \
  --lr 3e-4 \
  --save-freq 5000 \
  --eval-freq 5000
```

### Parámetros clave:
- `--timesteps`: Total de pasos (100K-200K recomendado)
- `--inner-steps`: Pasos internos por acción (3-10)
  - Menor = decisiones más rápidas
  - Mayor = decisiones más informadas
  
- `--max-steps`: Máximo pasos por episodio (30-100)
  - Menor = episodios cortos, aprende rápido
  - Mayor = episodios largos, más contexto

- `--lr`: Learning rate (1e-4 a 5e-4)
  - Menor = más estable pero lento
  - Mayor = más rápido pero inestable

### Ventajas:
✅ Control total de hiperparámetros  
✅ Puedes experimentar  
✅ Explorar nuevas configuraciones  

### Desventajas:
⚠️ Tarda más (empieza de cero)  
⚠️ Puede no mejorar sobre el 30K  
⚠️ Requiere conocimiento técnico  

---

## 🎯 Recomendación Estratégica

### Para mejora rápida (2-4 horas):
```bash
# 1. Continuar RL 50K pasos más
python continue_training_rl.py --timesteps 50000
```

### Para mejor calidad (4-8 horas):
```bash
# 1. Entrenar base 30 épocas
python train_base_more.py --epochs 30

# 2. Luego entrenar RL nuevo con base mejorado
python experiments/train_phi_text_scheduler.py --timesteps 100000
```

### Para investigación (1-2 días):
```bash
# 1. Base extendido (50 épocas)
python train_base_more.py --epochs 50

# 2. RL largo (200K pasos)
python experiments/train_phi_text_scheduler.py --timesteps 200000

# 3. Continuar RL con fine-tuning
python continue_training_rl.py --timesteps 100000
```

---

## 📊 Monitoreo del Entrenamiento

### Ver progreso en tiempo real:
```bash
# Para RL
python check_progress.py

# Para modelo base
# Revisar archivo training_log*.txt
```

### Analizar resultados:
```bash
# Después de entrenar RL
python analyze_rl_detailed.py

# Probar generación
python generate_with_rl_30k.py --prompt "Test" --max-length 100
```

---

## ⚙️ Configuración Avanzada

### Modificar rewards del RL:

Edita `experiments/train_phi_text_scheduler.py` líneas 100-108:

```python
"reward_weights": {
    "alpha": 1.0,   # ΔC (consciousness) - Mayor = prioriza consciousness
    "beta": 0.5,    # ΔΦ (phi) - Mayor = prioriza PHI
    "gamma": 0.1,   # Δperplexity - Mayor = prioriza fluidez
    "delta": 0.2,   # coste - Mayor = penaliza cambios
},
```

**Ejemplos:**
- **Priorizar calidad texto**: alpha=0.5, beta=0.3, gamma=0.3, delta=0.1
- **Priorizar PHI**: alpha=0.3, beta=1.0, gamma=0.1, delta=0.2
- **Balance**: alpha=1.0, beta=0.5, gamma=0.1, delta=0.2 (actual)

### Modificar arquitectura base:

Edita `train_v5_2_gpt2_lora.py` para cambiar:
- Número de capas INFINITO
- Tamaño de embeddings
- Configuración de LoRA
- Tamaño de memoria

---

## 🧪 Testing Durante Entrenamiento

### Cada 10K pasos RL:
```bash
python test_rl_generation.py
```

### Cada 5 épocas base:
```bash
python generate_phase2_text.py \
  --checkpoint models/checkpoints/infinito_phase2_best.pt \
  --prompt "Test prompt" \
  --max-length 100
```

---

## 💾 Gestión de Checkpoints

### Espacio necesario:
- Checkpoint RL: ~50MB cada uno
- Checkpoint base: ~500MB cada uno
- Logs y métricas: ~100MB por entrenamiento

### Limpieza:
```bash
# Eliminar checkpoints intermedios (dejar solo best y cada 20K)
# Manual, revisar directorio outputs/
```

---

## 🚨 Troubleshooting

### "CUDA out of memory"
```bash
# Reducir batch size
python train_base_more.py --batch-size 8  # En vez de 16

# O para RL, reducir inner_steps
python continue_training_rl.py --timesteps 50000  # Usa config por defecto
```

### "Entrenamiento no converge"
- Reducir learning rate: `--lr 1e-4`
- Aumentar paciencia: `--patience 10`
- Verificar que CUDA funciona: `python check_cuda.py`

### "Texto sigue con repeticiones"
- Necesitas entrenar el modelo BASE más épocas
- Aumentar repetition penalty en generación:
  ```bash
  python generate_with_rl_30k.py \
    --prompt "Test" \
    --max-length 100 \
    --repetition-penalty 2.0  # Aumentar de 1.2 a 2.0
  ```

---

## 📈 Resultados Esperados

### Después de continuar RL 50K (+30K = 80K total):
- Reward: +7.5 a +8.0 (vs +7.25 actual)
- Mejora en decisiones: +15-20%
- Tiempo: 2-4 horas

### Después de entrenar base 30 épocas:
- Perplexity: 40-60 (vs ~95 actual)
- Menos repeticiones: -50%
- Tiempo: 4-8 horas

### Después de pipeline completo (base + RL):
- Texto mucho más fluido y natural
- Decisiones adaptativas óptimas
- PHI estable en [3-6]
- Tiempo total: 8-16 horas

---

## 📝 Notas Finales

1. **Guardar progreso**: Todos los scripts guardan checkpoints automáticamente
2. **Interrumpir seguro**: Ctrl+C guarda el último checkpoint
3. **GPU requerida**: Todos los entrenamientos requieren CUDA
4. **Memoria**: Mínimo 8GB VRAM recomendado
5. **Paciencia**: El entrenamiento profundo lleva tiempo, pero vale la pena

---

## 🎯 Comandos Quick Start

```bash
# OPCIÓN RÁPIDA (2-4h): Continuar RL
python continue_training_rl.py --timesteps 50000

# OPCIÓN CALIDAD (4-8h): Entrenar base
python train_base_more.py --epochs 30

# OPCIÓN COMPLETA (8-16h): Base + RL
python train_base_more.py --epochs 30 && \
python experiments/train_phi_text_scheduler.py --timesteps 100000
```

¡Buena suerte con el entrenamiento! 🚀
