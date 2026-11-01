# INFINITO F9-T — ¿Aprende el propio actor recurrente CUÁNDO leer F4?

**1 de noviembre de 2026. PRE-REGISTRO ANTERIOR a la implementación y a cualquier conjunto DEV/CONFIRM.** Partimos de la rama de investigación F9-S, **no** del RAG experimental E10–E12. El objetivo vuelve a ser el agente propio de INFINITO.

## Evidencia previa y pregunta

- F9-Q: intercambio completo de H/C causó transferencia de la elección hacia la pista donante en 112/160 pares BC y 112/160 PPO sham con historia F4 normal y F4=0 en el cruce. Con apagón prolongado la información de pista era decodificable en C pero no útil para la decisión: **0/160** elecciones de ambos brazos correctas.
- F9-R: congelar H/C no corrigió el apagón; restaurar el estado natural cerca del cruce rescató parcialmente.
- F9-S: un parche lineal de dos ejes cue H/C no rescató ni transfirió la función, aunque el estado completo sí tuvo efectos.
- F4 SELECTIVO EXISTENTE es almacén de **8 eventos**, con reglas de escritura por contexto: **no** se presupone que haya aprendido qué recuerdos importar.
- E12-A en RAG mostró un LLM externo que enruta búsquedas, no la adquisición de una política de lectura por nuestra red.

**Pregunta primaria F9-T:** ¿puede una **cabeza neuronal de lectura** unida a la política recurrente original de INFINITO aprender, exclusivamente de recompensa de navegación y coste de acceso a F4, a decidir **READ/SKIP antes de recibir el readout de F4**, para conservar la utilidad de la pista con menos accesos que leer en cada frame?

## Arquitectura realmente evaluada

- Entorno MiniGrid-MemoryS13-v0 con las siete acciones nativas y observación pública **148 dimensiones**.
- Actor congelado de INFINITO F9-P `ppo_sham_after_rl.zip`: **RecurrentPPO/MlpLstmPolicy** con estado H/C de 64+64 y vector de entrada **660 = 148 públicos + 512 canales F4**. Se verifican SHA-256 y reportería original F9-O/P, 4 checkpoints de entrenamiento independientes, sin seleccionarlos después de ver datos.
- F4 **real** `SelectiveF4("context", capacity=8)`, idéntico algoritmo de escritura en todos los brazos; el escritor ve observaciones públicas y escribe por sus reglas; **no** recibe acceso privilegiado al objetivo.
- Nueva **cabeza neuronal de lectura**: MLP pequeño con entrada `concat(raw_148, H_prev_64, C_prev_64)`. En cada frame decide una acción binaria `READ/SKIP` con probabilidad aprendida. Su decisión se emite ANTES de que el actor reciba información F4 del frame actual. Si READ: pasa los 512 canales F4 actuales al actor; si SKIP: se entregan 512 ceros, con los 148 públicos idénticos. Actor LSTM se actualiza UNA sola vez por frame y el entorno avanza con la acción nativa.
- Ningún LLM, embeddings OpenAI, reglas textuales, respuesta del docente, estado privilegiado, pista futura, `correct_side`, meta, coordenada absoluta ni vector F4 llega a la cabeza de lectura. La **recompensa de terminal del entorno** sí puede usarse para ajustar el lector DESPUÉS del episodio. El acceso F4 se contabiliza también cuando su contenido es vacío; evitamos obtener lecturas gratuitas.
- Dos cabezas de idéntica arquitectura: `learned_hc` recibe H/C; `learned_public_only` recibe H/C igualadas a cero (misma cantidad de parámetros y datos), para probar si el estado recurrente realmente mejora el cuándo de la lectura. Ambos usan el mismo actor congelado.
- Aprendizaje de la cabeza por **REINFORCE/policy gradient**, distribución Bernoulli con semilla fijada; pérdida por retorno de episodio `reward_nativa − 0.15 * lecturas / pasos`. Actor congelado, sin gradientes en él. No es PPO sobre la arquitectura completa ni aprendizaje sin supervisión; es aprendizaje por refuerzo explícito con la recompensa original y penalización de consultas. No hay "oráculo de necesidad" para elegir READ en un frame.

## Baselines y controles

1. `always`: F4 en todos los frames, referencia de precisión.
2. `never`: F4 anulado en todos los frames, control de ablation.
3. `periodic_4`: READ en pasos 0,4,8,...; sin acceso a labels. Incluye entrenamiento cero.
4. `public_event_clock`: detecta **solo objetos key/ball visibles en la observación pública** y lee al primer avistamiento; después cada cuarto frame. Regla fuerte, sin información de meta o pista correcta.
5. `random_matched` (análisis diagnóstico, definido ANTES de observar CONFIRM): mismo número total de lecturas que `learned_hc` dentro de cada episodio, posiciones elegidas mediante una semilla independiente; no es una política que pueda ejecutarse online sin conocer el presupuesto total de ese episodio, así que **no** es baseline de despliegue y se etiqueta como control de dosis post hoc.
6. `learned_hc`, `learned_public_only` (cabeza entrenada).

## Split, presupuesto y métricas

**TRAIN exclusivamente**: `6100000 + index*10000 + j`, index=0..3, j=0..599, 600 mundos S13 por checkpoint; cada mundo entrena original y cue invertida, 1200 episodios por réplica y cabeza. Congelar hiperparámetros antes del primer TRAIN: red 276→32→1, optimizador Adam lr 0.001, entropía 0.001, retorno nativo y penalización 0.15 por fracción de lecturas. Sin selección de épocas por confirmación ni devolución de `correct_side` a la cabeza.

**DEV exploratorio**: `6200000+index*10000+j`, 20 pares por réplica, j=0..19. Antes del entrenamiento completo: smoke con máximo dos episodios por brazo para verificar que READ/SKIP alteran SOLO los 512 canales y que el actor mantiene una LSTM válida. Sin reajuste de parámetros por precisión DEV salvo errores de instrumentación documentados antes de confirmar.

**CONFIRM virgen**: `6300000+index*10000+j`, 40 pares por réplica, j=0..39; 160 pares originales/cue invertida por brazo. Nunca reutilizar antiguos splits F9, especialmente 895000+ ni el holdout histórico 900000+, ni E10/E11/E12. **CONFIRM NO DEBE EJECUTARSE AUTOMÁTICAMENTE** al escribir el protocolo; exige que las 4 cabezas y 4 checkpoints pasen hash + gates de entrenamiento y los tests causales antes de habilitar el fichero de lanzamiento.

**Métrica primaria**: porcentaje de pares con **ambas primeras elecciones de brazo espaciales correctas** y proporción de **éxito autónomo terminal**, siempre con todos los pares en el denominador. **Condición necesaria** de ahorro: `learned_hc` realiza al menos **25 % menos READ/frames** que `always`. **Condición de utilidad**: acierto en pares a no más de **−5 pp de `always`**, y al menos **+5 pp respecto a `periodic_4`** con read-rate comparable reportado. El efecto debe no ser desfavorable en 3/4 réplicas entrenadas. No ocultar el conflicto si una política gasta menos a costa de más errores. Reportar también mediana de retraso desde el último READ al fork, read-rate antes/después de pista, porcentaje de episodios con READ después de pista y número de lecturas inútiles (F4 vacío).

**Ablación causal H/C**: con la política aprendida congelada, igualar H/C del gate a cero en inferencia **sin tocar H/C del actor**. Contraste contra `learned_public_only` y contra control de permutación temporal H/C. No proclamar que "reconoce necesidad a partir de su estado interno" si la ablación no cambia significativamente las decisiones de lectura o su rendimiento. Reportar `READ` por frame y acciones con H/C intacta, cero y barajada.

**Criterio de parada**: si no supera las condiciones de utilidad, ahorro y control causal, el resultado es negativo. No comenzar F9-U ajustando parámetros sobre 6300000+ ni abrir 900000+ para rescatar un resultado.

## Límites de interpretación

Este estudio demuestra, como máximo, que una cabeza entrenada puede **aprender una decisión instrumentada de acceso F4** dentro del agente neuronal propio en un entorno restringido. No equivale a autorreferencia consciente, representación semántica de incertidumbre, comprensión lingüística ni descubrimiento espontáneo de que "no sabe algo". El actor está congelado; el aprendizaje se concentra en una cabeza añadida, aunque el agente conjunto es nuestro. Si no funciona, el resultado limita esta intervención concreta, no el programa de investigación INFINITO.

**No fusionar a producción**, no alterar V5/V5.1 ni `src/infinito3/`. Mantener PR de investigación independiente de E10/E11/E12. 
