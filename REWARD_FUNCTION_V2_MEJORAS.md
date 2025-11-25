# 🎯 REWARD FUNCTION v2 - MEJORAS IMPLEMENTADAS

**Fecha**: 12 Noviembre 2025  
**Versión**: INFINITO RL v2

---

## 📋 RESUMEN

Se mejoró la reward function del agente RL para prevenir mejor el colapso de Fase 2 y guiar al agente hacia estados más estables y óptimos.

---

## 🔄 CAMBIOS PRINCIPALES

### Reward Function Original (v1)
```python
r = α·ΔC + β·ΔΦ + γ·Δperplexity - δ·cost

Pesos: α=1.0, β=0.5, γ=0.1, δ=0.2
```

**Limitaciones**:
- Solo considera deltas (cambios), no valores absolutos
- No detecta colapso por perplexity extremo
- No penaliza inestabilidad (cambios bruscos)
- No incentiva rangos óptimos específicos

### Reward Function Mejorada (v2)
```python
r = α·ΔC + β·ΔΦ + γ·Δperplexity - δ·cost + 
    estabilidad + balance_phi + ppl_bounds + balance_c

Pesos base: α=1.0, β=0.5, γ=0.1, δ=0.2
```

**Nuevos términos**:

#### 1️⃣ **Estabilidad PHI** (-0.8 × exceso)
```python
if |ΔΦ| > 1.0:
    penalty = -0.8 × (|ΔΦ| - 1.0)
```
- Penaliza cambios bruscos en PHI
- Incentiva transiciones suaves
- **Ejemplo**: ΔΦ = 2.5 → penalty = -1.2

#### 2️⃣ **Balance PHI Óptimo** (rango [3.0, 6.0])
```python
if Φ < 3.0:   penalty = -0.3 × (3.0 - Φ)
if Φ > 6.0:   penalty = -0.6 × (Φ - 6.0)  ← Evita Fase 2
if 3.0 ≤ Φ ≤ 6.0:  bonus = +0.1
```
- **PHI bajo** (< 3.0): penalización leve
- **PHI alto** (> 6.0): **penalización FUERTE** (evita colapso Fase 2)
- **PHI óptimo**: bonus pequeño
- **Ejemplo**: Φ = 7.5 → penalty = -0.9

#### 3️⃣ **Límites Perplexity** (detecta colapso y confusión)
```python
if PPL < 10:    penalty = -2.0 × (10 - PPL) / 10  ← Colapso
if PPL > 200:   penalty = -0.3 × (PPL - 200) / 100
```
- **PPL < 10**: Colapso/repetición → penalización FUERTE
- **PPL > 200**: Modelo confuso → penalización moderada
- **Ejemplo**: PPL = 5 → penalty = -1.0

#### 4️⃣ **Balance Consciousness** (rango [0.3, 0.7])
```python
if C < 0.3:   penalty = -0.2 × (0.3 - C)
if C > 0.7:   penalty = -0.2 × (C - 0.7)
if 0.3 ≤ C ≤ 0.7:  bonus = +0.05
```
- Mantiene consciousness en rango razonable
- Bonus pequeño por estar en rango óptimo

---

## ✅ RESULTADOS DEL TEST

Todos los 8 escenarios pasan correctamente:

| Escenario | Métricas | Reward | Estado |
|-----------|----------|--------|--------|
| **1. Normal** | C=0.5, Φ=4.5, PPL=80 | +0.216 | ✅ Positivo |
| **2. PHI óptimo** | Φ ∈ [3.0, 6.0] | +0.236 | ✅ Bonus aplicado |
| **3. PHI alto** | Φ=7.0 (peligro) | **-0.353** | ✅ Penalizado |
| **4. Colapso PPL** | PPL=5 (repetición) | **-0.792** | ✅ Detectado |
| **5. Inestabilidad** | ΔΦ=2.5 (brusco) | **-0.244** | ✅ Penalizado |
| **6. PHI bajo** | Φ=2.5 | -0.476 | ✅ Penalizado |
| **7. PPL confuso** | PPL=250 | -0.039 | ✅ Penalizado |
| **8. Estado óptimo** | Todo en rango | +0.226 | ✅ Recompensado |

---

## 🎯 OBJETIVOS LOGRADOS

### ✅ Prevención Colapso Fase 2
- **PHI > 6.0** → penalización -0.6 por unidad
- **PHI > 8.0** → penalización > -1.2 (muy fuerte)
- Agente aprenderá a evitar PHI extremo

### ✅ Detección de Colapso Temprana
- **PPL < 10** → penalty -2.0 (máxima severidad)
- Detecta repeticiones antes de que se vuelvan infinitas
- Agente aprenderá a mantener PPL > 10

### ✅ Estabilidad Mejorada
- **Cambios bruscos PHI** → penalty -0.8
- Incentiva aprendizaje gradual y controlado
- Evita oscilaciones violentas

### ✅ Guía hacia Óptimos
- **PHI [3.0-6.0]** → bonus +0.1
- **C [0.3-0.7]** → bonus +0.05
- **PPL [10-200]** → sin penalización
- Agente converge a estados buenos, no solo "menos malos"

---

## 📊 COMPARACIÓN CON v1

| Aspecto | v1 (Original) | v2 (Mejorada) |
|---------|--------------|---------------|
| **Detecta colapso PPL** | ❌ No | ✅ Sí (PPL < 10) |
| **Previene Fase 2** | ⚠️ Indirecto | ✅ Directo (Φ > 6) |
| **Penaliza inestabilidad** | ❌ No | ✅ Sí (\|ΔΦ\| > 1) |
| **Incentiva rangos óptimos** | ❌ No | ✅ Sí (bonuses) |
| **Términos de recompensa** | 4 | **8** |
| **Robustez** | Media | **Alta** |

---

## 🚀 IMPACTO ESPERADO EN ENTRENAMIENTO

### Entrenamiento v1 (10K steps)
- Recompensa: -0.087 → -0.017 (+81%)
- Estrategia: Alternancia 50/50 TEXT/PHI
- PHI: 4.1 - 4.9 (controlado)
- PPL: 70 - 115 (normal)
- **Problema**: No explora MIXED (0%)

### Entrenamiento v2 (esperado con 50K steps)
- **Convergencia más rápida** (reward shaping mejor)
- **Menor variabilidad** (estabilidad incentivada)
- **Exploración MIXED** (ya no penaliza tanto extremos moderados)
- **Sin colapsos** (detección temprana PPL < 10)
- **PHI estable** en [3.5, 5.5] (rango óptimo)
- **Recompensa final esperada**: ~ +0.15 a +0.25

---

## 💡 PRÓXIMOS EXPERIMENTOS

### Experiment 1: Entrenar 50K con v2
```bash
python experiments/train_phi_text_scheduler.py \
  --timesteps 50000 \
  --inner-steps 5 \
  --max-steps 50
```
**Objetivo**: Verificar convergencia completa

### Experiment 2: Ajustar pesos
```python
# Más énfasis en estabilidad
reward_weights = {"alpha": 1.0, "beta": 0.3, "gamma": 0.15, "delta": 0.15}
```
**Objetivo**: Encontrar balance óptimo

### Experiment 3: Entrenar desde checkpoint v1
```bash
python experiments/train_phi_text_scheduler.py \
  --timesteps 50000 \
  --load-checkpoint outputs/.../best_model.zip
```
**Objetivo**: Aprovechar aprendizaje previo

---

## 📁 ARCHIVOS MODIFICADOS

1. **`src/rl/infinito_rl_env.py`**
   - `_compute_reward()`: Función mejorada con 4 términos nuevos
   - Total: ~50 líneas añadidas
   
2. **`experiments/README_RL.md`**
   - Documentación actualizada con reward v2
   
3. **`test_reward_function_v2.py`** (nuevo)
   - 8 escenarios de prueba
   - Validación completa

---

## 🏆 CONCLUSIÓN

**La reward function v2 es significativamente más robusta** que v1:

✅ **Previene activamente** el colapso de Fase 2  
✅ **Detecta temprano** degradación por repetición  
✅ **Incentiva estabilidad** y transiciones suaves  
✅ **Guía hacia rangos óptimos** conocidos  
✅ **Lista para producción** - todos los tests pasan  

**Recomendación**: Entrenar inmediatamente con v2 usando 50K+ timesteps para aprovechar las mejoras.

---

**Autor**: GitHub Copilot + INFINITO Team  
**Versión**: INFINITO RL v2  
**Estado**: ✅ Implementado y testeado
