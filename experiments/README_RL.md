# 🎮 Sistema RL para INFINITO - Scheduler Φ vs Texto

## 📖 Descripción

Sistema de **Aprendizaje por Refuerzo (RL)** que controla dinámicamente el balance entre optimización de **texto** y **PHI (Φ)** en INFINITO.

Un agente **PPO** (Proximal Policy Optimization) aprende a decidir cuándo priorizar:
- **Modo TEXTO**: Optimizar calidad del lenguaje (w_text=1.0, w_phi=0.0)
- **Modo PHI**: Optimizar integración de información (w_text=0.1, w_phi=1.0)
- **Modo MIXTO**: Balance equilibrado (w_text=0.5, w_phi=0.5)

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────┐
│         Agente RL (PPO)                     │
│  ┌───────────────────────────────────────┐  │
│  │  Observa: [C, Φ, loss_text, loss_phi, │  │
│  │           memory_util, time]           │  │
│  └───────────────────────────────────────┘  │
│                    ↓                         │
│  ┌───────────────────────────────────────┐  │
│  │  Decide: acción ∈ {TEXT, PHI, MIXED}  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│         INFINITO (InfinitoGPT2Hybrid)       │
│  ┌───────────────────────────────────────┐  │
│  │  Ajusta pesos: w_text, w_phi          │  │
│  │  Loss = w_text·loss_LM + w_phi·loss_Φ │  │
│  └───────────────────────────────────────┘  │
│                    ↓                         │
│  Entrena N pasos con configuración actual   │
│                    ↓                         │
│  Devuelve: métricas actualizadas            │
└─────────────────────────────────────────────┘
                    ↓
          r = α·ΔC + β·ΔΦ + γ·Δppl - δ·cost
```

## 🎯 Recompensa (Versión Mejorada v2)

El agente maximiza una recompensa compuesta **mejorada** con términos adicionales:

```
r = α·ΔC + β·ΔΦ + γ·Δperplexity - δ·cost + estabilidad + balances
```

### Términos Básicos
- **α·ΔC**: Mejora en "consciousness" (PHI normalizado)
- **β·ΔΦ**: Mejora en PHI absoluto
- **γ·Δperplexity**: Mejora en perplexity (negativo si empeora)
- **δ·cost**: Penalización por uso de memoria (proxy de compute)

### Términos Mejorados (NUEVO)
- **Estabilidad PHI**: Penaliza cambios bruscos (|ΔΦ| > 1.0) → -0.5·(|ΔΦ| - 1.0)
- **Balance PHI**: Mantiene Φ ∈ [3.0, 6.0] óptimo
  - Φ < 3.0: penaliza -0.3·(3.0 - Φ)
  - Φ > 6.0: penaliza **fuerte** -0.6·(Φ - 6.0) ← Evita colapso Fase 2
  - Φ ∈ [3.0, 6.0]: bonus +0.1
- **Límites Perplexity**: Detecta colapso/confusión
  - PPL < 10: colapso → -1.0·(10 - PPL)/10
  - PPL > 200: confuso → -0.3·(PPL - 200)/100
- **Balance Consciousness**: C ∈ [0.3, 0.7] óptimo → bonus +0.05

**Pesos por defecto**: `α=1.0, β=0.5, γ=0.1, δ=0.2`

**Objetivo**: Evitar el colapso de Fase 2 (Φ > 8 + repeticiones) incentivando rangos óptimos.

## 📦 Instalación

### Dependencias

```bash
pip install gymnasium>=0.29.0
pip install stable-baselines3>=2.0.0
pip install tensorboard>=2.13.0
```

O instalar todas las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

## 🚀 Uso

### 1. Entrenar Agente RL

Entrenar un agente PPO durante 100K timesteps:

```bash
python experiments/train_phi_text_scheduler.py \
    --timesteps 100000 \
    --inner-steps 5 \
    --max-steps 100 \
    --batch-size 4 \
    --lr 3e-4 \
    --output-dir outputs/rl_phi_text_scheduler
```

**Parámetros**:
- `--timesteps`: Total de timesteps de entrenamiento (default: 100,000)
- `--inner-steps`: Pasos de INFINITO por step RL (default: 5)
- `--max-steps`: Pasos máximos por episodio RL (default: 100)
- `--batch-size`: Batch size para INFINITO (default: 4)
- `--lr`: Learning rate del agente PPO (default: 3e-4)
- `--output-dir`: Directorio de salida
- `--save-freq`: Frecuencia de checkpoints (default: 10,000)
- `--eval-freq`: Frecuencia de evaluación (default: 5,000)

### 2. Monitorear Entrenamiento

Ver logs con TensorBoard:

```bash
tensorboard --logdir outputs/rl_phi_text_scheduler/tensorboard
```

Métricas disponibles:
- `rollout/ep_rew_mean`: Recompensa promedio
- `rollout/ep_len_mean`: Longitud promedio de episodios
- `train/policy_loss`: Loss de la política
- `train/value_loss`: Loss de la función de valor
- `train/entropy_loss`: Entropía de la política

### 3. Ejecutar Demostración

Ejecutar INFINITO controlado por el agente entrenado:

```bash
python experiments/run_infinito_with_scheduler.py \
    --model outputs/rl_phi_text_scheduler/ppo_infinito_scheduler_final.zip \
    --episodes 3 \
    --max-steps 100 \
    --output-dir outputs/rl_demo
```

Esto generará:
- Logs de ejecución en consola
- Gráficos de métricas por episodio en `outputs/rl_demo/`

### 4. Usar en Script Personalizado

```python
from stable_baselines3 import PPO
from src.rl.infinito_rl_env import InfinitoRLEnv

# Configurar entorno
env_config = {
    "inner_steps": 5,
    "max_steps": 100,
    "model_kwargs": {
        "use_lora": True,
        "lambda_phi": 0.3,
    },
}

# Crear entorno
env = InfinitoRLEnv(config=env_config)

# Cargar agente entrenado
model = PPO.load("path/to/model.zip")

# Ejecutar
obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # Modo actual
    mode = ["TEXT", "PHI", "MIXED"][action]
    print(f"Modo: {mode}, Reward: {reward:.4f}")
```

## 📊 Resultados Esperados

### Entrenamiento

Durante el entrenamiento, el agente debería:

1. **Fase Inicial (0-20K timesteps)**:
   - Exploración aleatoria
   - Recompensas fluctuantes
   - No hay patrón claro

2. **Fase de Aprendizaje (20K-60K timesteps)**:
   - Recompensas comienzan a subir
   - Aparecen patrones en acciones
   - Agente aprende qué funciona

3. **Fase de Convergencia (60K-100K timesteps)**:
   - Recompensas estables
   - Política consistente
   - Balance adaptativo emergente

### Comportamiento del Agente

El agente debería aprender patrones como:

- **Inicio de episodio**: Preferir modo **TEXTO** para estabilizar
- **PHI bajo**: Cambiar a modo **PHI** para aumentar integración
- **PHI alto pero perplexity alta**: Modo **MIXTO** para balancear
- **Cerca de breakthrough (C > 0.6)**: Mantener modo actual

## 🔬 Componentes

### InfinitoRLEnv

Entorno Gymnasium para RL:

```python
class InfinitoRLEnv(gym.Env):
    """
    Espacio de acciones: Discrete(3)
      0 → Modo TEXTO
      1 → Modo PHI
      2 → Modo MIXTO
    
    Espacio de observaciones: Box(6)
      [C, Φ, loss_text, loss_phi, memory_util, time_norm]
    """
```

**Métodos clave**:
- `reset()`: Reinicia INFINITO para nuevo episodio
- `step(action)`: Ejecuta acción y devuelve (obs, reward, done, info)
- `_compute_reward()`: Calcula recompensa basada en mejoras

### InfinitoGPT2Hybrid (Modificado)

Modelo INFINITO con soporte RL:

**Nuevos métodos**:
```python
model.set_loss_weights(w_text, w_phi)  # Cambiar pesos dinámicamente
model.get_current_metrics()             # Obtener métricas actuales
model.update_current_metrics(...)       # Actualizar métricas internas
```

**Atributo**:
```python
model.loss_weights = {"text": 1.0, "phi": 0.3}  # Pesos actuales
```

## 📈 Hiperparámetros

### PPO

- **Policy**: MLP (Multi-Layer Perceptron)
- **Hidden layers**: [128, 128] (actor), [128, 128] (critic)
- **Learning rate**: 3e-4
- **Discount (γ)**: 0.99
- **GAE (λ)**: 0.95
- **PPO clip**: 0.2
- **Entropy coef**: 0.01
- **Value function coef**: 0.5

### INFINITO (para RL)

- **LoRA r**: 4 (reducido para velocidad)
- **LoRA alpha**: 16
- **Memory slots**: 128 (reducido para velocidad)
- **Sequence length**: 128 (reducido para velocidad)
- **Batch size**: 4 (pequeño para RL)

## 🐛 Troubleshooting

### Error: "stable-baselines3 not found"

```bash
pip install stable-baselines3
```

### Error: "gymnasium not found"

```bash
pip install gymnasium
```

O si usas gym antiguo:

```bash
pip install gym
```

### CUDA Out of Memory

Reducir:
- `batch_size` en env_config
- `inner_steps` (menos pasos por step RL)
- `lora_r` en model_kwargs

### Entrenamiento muy lento

Reducir:
- `max_steps` por episodio
- `inner_steps` por step
- `total_timesteps` de entrenamiento

## 📝 Notas

1. **Tiempo de entrenamiento**: ~2-4 horas para 100K timesteps (GPU)
2. **Memoria GPU**: ~6-8GB con configuración por defecto
3. **Checkpoints**: Guardados cada 10K timesteps en `output_dir/checkpoints/`
4. **Mejor modelo**: Guardado en `output_dir/best_model/` según evaluación

## 🔮 Experimentos Sugeridos

1. **Pesos de recompensa**: Probar diferentes α, β, γ, δ
2. **Inner steps**: Variar cantidad de pasos INFINITO por step RL
3. **Arquitectura PPO**: Probar redes más grandes/pequeñas
4. **Curriculum learning**: Empezar con episodios cortos, aumentar gradualmente
5. **Multi-objetivo**: Añadir más componentes a la recompensa

## 📚 Referencias

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [IIT Theory](http://integratedinformationtheory.org/)

---

**Última actualización**: 11 de Noviembre, 2025  
**Versión**: 1.0.0 - Sistema RL Scheduler Φ vs Texto
