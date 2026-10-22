# Experimento F1 — Memoria temprana en INFINITO V5.1

**Estado:** diseño y arnés experimental; se mantiene separado de las
implementaciones originales de INFINITO.

## Hipótesis (antes de ver los resultados)

En la implementación histórica V5.1, la memoria externa influye
principalmente en la integración posterior. Si su lectura se realimenta
**antes** de los cuatro módulos visual, auditivo, motor y ejecutivo,
podría influir más en la representación futura de una señal vista antes.

**H1, causal:** la representación final de una entrada idéntica variará
al activar feedback temprano, sin cambiar pesos, entradas, semillas,
ni recompensas del modelo.

**H2, memoria:** al vaciar la memoria externa *después* de presentar la
pista inicial, cualquier contribución observable de la memoria temprana
a una respuesta diferida debería atenuarse, salvo la información que
conserven los estados LSTM.

**H3, utilidad:** un probe lineal entrenado únicamente sobre
representaciones congeladas podría recuperar mejor una pista antigua
que en la arquitectura histórica, evaluado sobre secuencias inéditas.
Esta hipótesis **puede ser rechazada**: mayor cambio de estados no
equivale a mayor precisión. No hay un objetivo del modelo que enseñe
recuerdo o predicción en esta fase.

## Tres condiciones comparadas

- \`historical\`: modelo V5.1 original, sin modificación de ruta.
- \`early_feedback\`: ganancia fija 0,5 × lectura de memoria reinyectada
  antes de los cuatro módulos de procesamiento mediante pre-hooks.
- \`early_feedback_memory_cut\`: igual al anterior, pero se borra la
  matriz de memoria después de la pista inicial y antes de los pasos
  distractores y de la pregunta final.

La variante se instrumenta mediante hooks externos. No se reescribe
\`src/infinito_v5_1_consciousness.py\`.

No hay un nuevo objetivo de entrenamiento interno ni recompensa por
"parecer consciente". El reinyector es una proyección identidad
fija sobre los primeros canales de memoria: no se aprenden parámetros
adicionales en la red original.

## Secuencias de estímulos

Cada episodio presenta cinco entradas:

1. Paso 0: **pista binaria** codificada como diferencia de signo en 64
   de las 257 características de entrada.
2. Pasos 1–3: distractores aleatorios independientes, sin etiqueta.
3. Paso 4: **consulta con entrada idéntica en distribución** para ambas
   etiquetas, con los 256 canales sensoriales a cero.

La única fuente intencionada de información de la etiqueta es la
pista inicial. La red debe conservarla en su estado temporal para
permitir que una lectura lineal la recupere al final.

El test es **decodificación retrospectiva**, todavía no predicción
autónoma de acontecimientos. Los recuerdos pueden residir tanto en
LSTM como en memoria externa; el borrado pone a prueba la segunda
ruta mientras conserva la primera.

## Métricas y rigor

- Representación final real del \`consciousness_processor\` original
  y lectura real de \`EnhancedExternalMemory\`.
- Distancia L2 relativa entre condiciones sobre las mismas entradas.
- Un **probe ridge lineal** entrenado sobre parte de las representaciones
  y evaluado sobre entradas no vistas. Pesos del cerebro V5.1 congelados.
- Precisión de decodificación de la etiqueta antigua desde el
  estado final y, por separado, desde la lectura de memoria.
- Tamaño/norma de memoria y desviación de las representaciones.

Protocolo inicial: semillas 11/29/47, 24 episodios de
entrenamiento del probe + 24 episodios de prueba por semilla,
secuencia de 5 frames, \`hidden_dim=64\`, 16 slots externos, batch 2.
Se utilizan 2 clases equilibradas, 24/24 separadas temporalmente por
índice; cada condición recibe **idénticas** entradas y semillas.

Una precisión alrededor del 50% significa que el probe no logra
decodificar mejor que el azar en esta muestra; una precisión superior
requiere comparar con controles de etiquetas, más semillas y tamaños
de muestra antes de proclamar una mejora real.

## Ejecutar

~~~bash
PYTHONPATH=. python scripts/infinito_v51_memory_feedback.py \
  --seeds 11 29 47 --train 24 --test 24 --gain 0.5 \
  --output observatory_results/memory_feedback
~~~

El resultado es \`report.json\`, adjuntado al workflow de Actions
\`INFINITO V5.1 Memory Feedback\`.

## Limitaciones explícitas

- Se conserva el **forward original**, no las 15.000 iteraciones de
  entrenamiento V5.1. El modelo está inicializado sin entrenamiento
  histórico; *solo se entrena el probe* de evaluación.
- El modelo está escalado a 64 dimensiones internas y 16 slots, frente
  a 512 y 256 en el histórico.
- La proyección fija introduce una ganancia elegida por el investigador;
  no es una propiedad que emergiera espontáneamente.
- El probe clasificador puede recuperar señales de otros estados
  recurrentes; se debe hacer una ablación adicional de LSTM o comparar
  explícitamente el probe de lectura de memoria.
- Un aumento de decodificabilidad **no** significa que el modelo
  tenga curiosidad, conciencia, sufrimiento ni experiencia subjetiva.
- La muestra inicial es de exploración, no para afirmaciones
  estadísticas sólidas.

## Después de F1

Si el experimento produce evidencia de un mecanismo aprovechable,
registrar mejor:
(a) respuesta ante señales repetidas y nuevos distractores;
(b) controles con lecturas de memoria barajadas;
(c) ablación de rutas LSTM antes y después del evento;
(d) entrenamiento predictivo explícito de una tarea, con controles
de capacidad y datos de prueba congelados.

El objetivo sigue siendo descubrir **qué cambia, cómo cambia y por
qué**, no elevar un indicador de conciencia.
