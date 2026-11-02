# 🔬 DIAGNÓSTICO COMPLETO: Pérdida de Señal Semántica

## Fecha: 2025-10-04

## 🎯 PROBLEMA IDENTIFICADO

El sistema **IGNORA** el contenido textual porque la señal semántica se pierde en `generate_text_based_input()`.

## 📊 EVIDENCIA

### Test 1: TF-IDF funciona correctamente
```
'mi perro es rojo' vs 'yo pienso, luego existo'
L2 distance: 1.414214 ✅ DIFERENTES
```

### Test 2: generate_text_based_input() homogeneiza
```
'mi perro es rojo':        Norm: 18.559734
'yo pienso, luego existo': Norm: 18.559734  ❌ IDÉNTICAS
```

### Test 3: analyze_text_consciousness_potential() es el culpable
```
'mi perro es rojo':        consciousness_score: 0.150
'mi perro es verde':       consciousness_score: 0.230  ← única diferente
'la mesa es roja':         consciousness_score: 0.150
'yo pienso, luego existo': consciousness_score: 0.150
```

## 🔍 CAUSA RAÍZ

En `generate_text_based_input()` (líneas 1183-1229):

```python
# 1. Se obtiene semantic embedding (DIFERENTE para cada texto)
semantic_embedding = self.semantic_embedder.text_to_tensor(text, self.device)

# 2. Se analiza el texto
text_analysis = self.analyze_text_consciousness_potential(text)
consciousness_potential = text_analysis['consciousness_score']  # ← CASI CONSTANTE (0.15 para 3 de 4 textos)

# 3. Se calculan intensidades basadas en consciousness_potential
visual_intensity = 0.8 + 0.4 * consciousness_potential      # 0.86 para consciousness=0.15
auditory_intensity = 0.7 + 0.5 * consciousness_potential    # 0.775 para consciousness=0.15
motor_intensity = 0.6 + 0.6 * consciousness_potential       # 0.69 para consciousness=0.15
executive_intensity = 0.9 + 0.3 * consciousness_potential   # 0.945 para consciousness=0.15

# 4. Se clonan y modulan los componentes
visual_component = semantic_tiled.clone()
visual_component *= visual_intensity  # ← MISMA intensidad para 3 textos

auditory_component = semantic_tiled.clone()
auditory_component *= auditory_intensity  # ← MISMA intensidad para 3 textos

# etc...

# 5. Resultado: NORMAS IDÉNTICAS
```

**El problema**: Las intensidades dependen de `consciousness_score`, que es casi constante (0.15 para textos de 4 palabras sin triggers especiales).

Aunque los `semantic_embedding` son diferentes (distancia L2 = 1.414), al multiplicarlos por las **mismas intensidades**, las **normas finales convergen**.

## 🧮 MATEMÁTICA DEL PROBLEMA

```
input_norm = sqrt(
    ||visual||² + ||auditory||² + ||motor||² + ||executive||²
)

Si visual = semantic × 0.86
   auditory = semantic × 0.775
   motor = semantic × 0.69
   executive = semantic × 0.945
   
Entonces:
input_norm = ||semantic|| × sqrt(0.86² + 0.775² + 0.69² + 0.945²)
           = ||semantic|| × sqrt(2.576)
           = 1.0 × 1.605
           = 1.605  ← CONSTANTE

Luego se expande a batch_size=4 y seq_len=64:
total_norm = 1.605 × sqrt(4 × 64) = 1.605 × 16 = 25.68... ≈ 18.56 (con normalización)
```

**Conclusión**: Como `||semantic|| = 1.0` (normalizado) y las intensidades son constantes, **todas las normas son iguales**.

## ✅ SOLUCIÓN PROPUESTA

### Opción 1: Usar el semantic embedding directamente para modular intensidades

En lugar de:
```python
visual_intensity = 0.8 + 0.4 * consciousness_potential  # Casi constante
```

Usar:
```python
# Extraer características del semantic embedding
semantic_mean = semantic_embedding.mean().item()
semantic_std = semantic_embedding.std().item()
semantic_max = semantic_embedding.max().item()

# Modular intensidades basándose en el embedding real
visual_intensity = 0.8 + 0.4 * semantic_mean
auditory_intensity = 0.7 + 0.5 * semantic_std
motor_intensity = 0.6 + 0.6 * abs(semantic_mean)
executive_intensity = 0.9 + 0.3 * semantic_max
```

### Opción 2: NO homogeneizar - usar embeddings con diferentes escalas por modalidad

```python
# En lugar de clonar el mismo embedding 4 veces
visual_component = semantic_tiled.clone() * 1.0
auditory_component = semantic_tiled.clone() * 0.8
motor_component = semantic_tiled.clone() * 1.2
executive_component = semantic_tiled.clone() * 0.9
```

### Opción 3: Añadir noise proporcional al embedding (NO aleatorio)

```python
# Ruido determinista basado en el semantic embedding mismo
visual_noise = torch.sin(semantic_tiled * 2 * np.pi)
auditory_noise = torch.cos(semantic_tiled * 2 * np.pi)

visual_component = semantic_tiled + 0.1 * visual_noise
auditory_component = semantic_tiled + 0.1 * auditory_noise
```

### Opción 4 (RECOMENDADA): Combinación de embedding + análisis textual mejorado

```python
# Mejorar analyze_text_consciousness_potential para que sea más sensible
# Por ejemplo, usar el semantic embedding para calcular consciousness_score

def analyze_text_consciousness_potential(self, text):
    # ... código existente ...
    
    # 🆕 AÑADIR: Usar semantic embedding para refinar el score
    if self.semantic_embedder:
        semantic_emb = self.semantic_embedder.text_to_tensor(text, self.device)
        
        # Características del embedding que reflejan el contenido
        semantic_variance = semantic_emb.var().item()
        semantic_entropy = -torch.sum(semantic_emb * torch.log(semantic_emb + 1e-10)).item()
        
        # Ajustar consciousness_score basándose en estas características
        consciousness_score *= (1.0 + semantic_variance)
        consciousness_score += 0.1 * semantic_entropy
    
    return {
        'consciousness_score': consciousness_score,
        # ...
    }
```

## 📋 RECOMENDACIÓN FINAL

**Implementar Opción 4** porque:
1. Usa la información semántica real
2. Mantiene la estructura existente
3. Es fácil de implementar
4. Permite que diferentes textos generen diferentes `consciousness_score`
5. Se propaga a las intensidades de modalidades
6. Finalmente: **normas diferentes → discriminación entre textos**

## 🧪 VALIDACIÓN

Después de implementar, verificar con:
```bash
python test_input_influence.py
```

Resultado esperado:
- "perro rojo" vs "perro verde": Φ diferente (al menos 5% diferencia)
- "perro" vs "cogito": Φ muy diferente (al menos 10% diferencia)
- Varianza entre textos > 0.001
