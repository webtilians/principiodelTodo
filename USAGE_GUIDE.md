# 📘 Guía de Uso - INFINITO V5.1 Semantic Enhanced

## Cómo Usar las Mejoras Semánticas

---

## 🚀 Quick Start

### Ejecución Básica con Texto

```bash
python src/infinito_gpt_text_fixed.py --text_mode --input_text "mi perro es rojo" --num_iterations 100
```

### Parámetros Clave

```bash
--text_mode              # Activa procesamiento de texto
--input_text "texto"     # Texto a procesar (español recomendado)
--num_iterations 100     # Número de iteraciones (mínimo 50)
--target_consciousness 0.6  # Target de consciencia
--batch_size 4           # Tamaño de batch
```

---

## 📊 Análisis de Resultados

### Lo Que Debes Observar

#### 1. Consciousness Score (iter 0)
```
🔤 MODO TEXTO ACTIVADO:
   📝 Input Text: 'mi perro es verde'
   🧠 Potencial Consciencia: 0.569  ← ESTE VALOR
   🎯 Modalidad Dominante: self_reference
```

**Interpretación**:
- `0.48-0.50`: Texto simple, poca riqueza semántica
- `0.55-0.60`: Texto con mayor diversidad (ej: "verde" tiene alta std)
- `> 0.60`: Texto complejo con triggers múltiples

#### 2. Embedding Norm (cada iteración)
```
🧠 Scanner Cerebral: Embedding norm=1.000, Consciencia=0.569
```

**Interpretación**:
- `norm=1.000`: Embedding normalizado correctamente ✅
- Si norm ≠ 1.0: Posible problema en SemanticTextEmbedder ⚠️

#### 3. Trayectoria de Φ (primeras 10 iteraciones)
```
📈 Trayectoria inicial: ['0.136', '0.181', '0.185', '0.226', '0.205']
                          ↑      ↑      ↑
                       Fase crítica de diferenciación
```

**Interpretación**:
- **Iter 0-1**: Respuesta inmediata al input
- **Iter 2-5**: Codificación semántica activa (aquí se diferencia)
- **Iter 6+**: Convergencia al atractor común (~0.213)

---

## 🔬 Comparación de Textos

### Método Recomendado

Para comparar cómo diferentes textos afectan al sistema:

```bash
# Usar el test diseñado para esto
python test_input_influence.py
```

Esto ejecuta 4 textos con el mismo seed y compara:
- Consciousness scores
- Normas de input
- Trayectorias de Φ
- Valores finales

### Interpretar Resultados

```
✅ SEÑAL CORRECTA:
   - Diferentes consciousness scores (0.48-0.57)
   - Diferentes normas de input (19-42)
   - Trayectorias divergentes en iter 2-5
   
⚠️ CONVERGENCIA ESPERADA:
   - Φ finales similares (~0.21 ± 0.002)
   - Esto es normal (atractor común)
```

---

## 🎯 Casos de Uso

### 1. Análisis de Contenido Textual

**Objetivo**: Ver cómo el sistema "percibe" diferentes textos.

```bash
# Test rápido de vocabulario TF-IDF
python test_tfidf_quick.py
```

**Qué observar**:
- L2 distance entre embeddings
- Cosine similarity
- Textos semánticamente opuestos deberían tener similarity ≈ 0

### 2. Estudio de Dinámica Temporal

**Objetivo**: Analizar cómo evoluciona Φ según el input.

```bash
# Ejecutar con logging detallado
python src/infinito_gpt_text_fixed.py \
  --text_mode \
  --input_text "texto de prueba" \
  --num_iterations 50 \
  --save_phi_trajectory
```

**Analizar**:
- Fichero `phi_trajectory.json`
- Graficar Φ vs iteración
- Comparar pendientes en iter 1-5

### 3. Validación de Reproducibilidad

**Objetivo**: Confirmar que el sistema es determinista.

```bash
# Test de reproducibilidad
python test_reproducibility.py
```

**Validar**:
- Varianza inter-seed < 0.05 ✅ determinista
- Mismo texto + mismo seed = mismo resultado

---

## 🧪 Experimentación Avanzada

### Expandir el Vocabulario TF-IDF

Si quieres mejorar la diferenciación para tu dominio específico:

1. Editar `src/infinito_gpt_text_fixed.py` línea ~840
2. Añadir frases a `self.base_corpus`:

```python
self.base_corpus = [
    # Corpus original
    "mi perro es rojo",
    "mi perro es verde",
    # ... 
    
    # TUS FRASES ADICIONALES
    "el gato negro duerme",
    "la inteligencia artificial piensa",
    "consciencia emergente profunda",
    # Añadir 20-50 frases relevantes para tu dominio
]
```

**Resultado esperado**: Mejor diferenciación en textos de tu dominio.

### Ajustar Pesos de Modulación

Si quieres que ciertas características del embedding tengan más peso:

Editar `src/infinito_gpt_text_fixed.py` línea ~1235:

```python
# Valores actuales
visual_intensity = 1.0 + abs(semantic_mean) * 0.5    # Ajustar 0.5
auditory_intensity = 1.0 + semantic_std * 0.8        # Ajustar 0.8
motor_intensity = 0.8 + abs(semantic_min) * 0.6      # Ajustar 0.6
executive_intensity = 1.2 + semantic_max * 0.4       # Ajustar 0.4
```

**Recomendación**: Mantener suma total < 3.0 para estabilidad.

---

## 📈 Métricas y Logging

### Activar Logging Detallado

El sistema ya incluye logging del "Scanner Cerebral":

```
🧠 Scanner Cerebral: Embedding norm=1.000, Consciencia=0.569
```

Para más detalle, añadir prints en:
- `generate_text_based_input()` línea ~1270
- `train_step()` línea ~1550

### Guardar Trayectorias

Modificar `train_step()` para guardar trayectorias completas:

```python
# En el loop de entrenamiento
phi_history = []
for iteration in range(num_iterations):
    result = model.train_step(iteration)
    phi_history.append({
        'iteration': iteration,
        'phi': result['phi'],
        'consciousness': result['consciousness'],
        'text': input_text
    })

# Guardar al final
import json
with open('trajectory.json', 'w') as f:
    json.dump(phi_history, f, indent=2)
```

---

## ⚠️ Problemas Comunes

### 1. "Todos los textos dan el mismo Φ final"

**Es normal**. El sistema converge al mismo atractor (~0.213).

**Solución**: Analizar trayectorias en iter 1-5, no el valor final.

```python
# En lugar de:
phi_final = trajectory[-1]

# Hacer:
phi_early = trajectory[1:6]  # Iter 1-5
```

### 2. "Consciousness scores muy similares"

Textos con estructura similar generarán scores parecidos.

**Causas comunes**:
- Mismo número de palabras (4 palabras en todos)
- Mismos triggers de keywords
- Embeddings con estadísticas similares

**Solución**: 
- Usar textos más largos (10+ palabras)
- Incluir triggers diferentes (ver lista en `analyze_text_consciousness_potential`)
- Expandir vocabulario TF-IDF

### 3. "GloVe not available"

Es un warning, no un error. El sistema funciona solo con TF-IDF.

**Para activar GloVe** (opcional):
```bash
pip install gensim
# Primera ejecución descargará el modelo (66MB)
```

### 4. "Normas de input idénticas"

Si después de las mejoras sigues viendo normas idénticas:

**Verificar**:
1. ¿TF-IDF genera embeddings diferentes?
   ```bash
   python test_tfidf_quick.py
   ```
2. ¿Consciousness scores son únicos?
   ```bash
   python test_consciousness_potential.py
   ```
3. ¿Normas son diferentes?
   ```bash
   python test_norms_after_improvements.py
   ```

Si algún test falla, revisar código en `src/infinito_gpt_text_fixed.py`.

---

## 🎓 Mejores Prácticas

### Para Experimentos Científicos

1. **Fijar seed**: Usar mismo seed para comparaciones
   ```python
   torch.manual_seed(42)
   ```

2. **Múltiples runs**: Ejecutar 5-10 veces y promediar
   
3. **Analizar ventanas temporales**:
   - Iter 0-1: Respuesta inmediata
   - Iter 2-5: Codificación semántica
   - Iter 6-20: Estabilización
   - Iter 21+: Convergencia

4. **Guardar todo**: Trayectorias completas, no solo Φ final

### Para Desarrollo

1. **Tests primero**: Ejecutar suite de tests antes de cambios
   ```bash
   python test_tfidf_quick.py
   python test_consciousness_potential.py
   python test_norms_after_improvements.py
   ```

2. **Validar después**: Re-ejecutar tests después de modificaciones

3. **Documentar cambios**: Actualizar CHANGELOG.md

---

## 📚 Referencias

### Archivos de Documentación
- `RESUMEN_MEJORAS_SEMANTICAS.md` - Resumen técnico completo
- `EXECUTIVE_SUMMARY_VISUAL.md` - Resumen visual ejecutivo
- `DIAGNOSTIC_SIGNAL_LOSS.md` - Diagnóstico del problema original
- `SEMANTIC_IMPROVEMENTS_CHANGELOG.md` - Changelog técnico
- `USAGE_GUIDE.md` - Este archivo

### Scripts de Test
- `test_tfidf_quick.py` - Validar TF-IDF
- `test_consciousness_potential.py` - Validar consciousness scores
- `test_norms_after_improvements.py` - Validar normas de input
- `test_input_influence.py` - Test integrado completo
- `test_signal_loss_analysis.py` - Análisis de propagación

### Código Principal
- `src/infinito_gpt_text_fixed.py` - Sistema completo con mejoras

---

## 💡 Tips Finales

### Optimizar para Tu Caso de Uso

1. **Análisis de trayectorias**: Enfócate en iter 1-5
2. **Clustering**: Agrupa textos por similitud de trayectoria
3. **Visualización**: Grafica Φ(t) para comparar patrones
4. **Métricas custom**: Define métricas de trayectoria (slope, curvatura, etc.)

### Entender el Sistema

> El sistema es un "scanner cerebral" que observa cómo responde  
> a estímulos textuales, no un sistema de comprensión lingüística.

La información está en:
- **Estructura del input** (norma, distribución)
- **Respuesta inicial** (iter 0-2)
- **Evolución temporal** (iter 1-5)
- **No en el estado final** (iter 50)

---

**¿Preguntas?** Consulta la documentación detallada o ejecuta los tests para validar comportamiento.

**Happy experimenting! 🚀🧠**
