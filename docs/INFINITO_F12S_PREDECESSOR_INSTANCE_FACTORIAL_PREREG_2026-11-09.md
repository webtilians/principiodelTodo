# INFINITO F12-S — Identidad del antecedente inmediato × instancia del modelo

**Prerregistro prospectivo, 9 de noviembre de 2026. Congelado antes de implementar el código F12-S, realizar pruebas nuevas o ejecutar inferencias F12-S.**

## Pregunta y origen

F12-R (PR #143, run 42656891385) reprodujo `O21-start → {"action":"q2"}` 12/12 veces al reutilizar la instancia después de `O21-disabled`; al reiniciarla respondió `{"action":"q0"}` 12/12. F12-Q usó `O0-disabled` como último antecedente y obtuvo `q0` en todos los tratamientos. **El último antecedente y el tipo de instancia no fueron cruzados en F12-R/Q; esta prueba factorial sí los cruza** con una misma prehistoria controlada. No se crean mundos, no se ajustan instrucciones ni se consume el banco de 24 mundos nuevos de transferencia F12-P-T.

Hipótesis principal contrastable: *la respuesta q2 requiere conjuntamente el antecedente inmediato `O21-disabled` y reutilización de la instancia que lo procesó*. Alternativas preregistradas: efecto de instancia no específico al antecedente, efecto de antecedente independiente de instancia, ausencia de diferencia, y falta de reproducibilidad.

## Fuente, constancia y tratamiento

**Fuente:** los seis prompts diagnósticos de F12-N reconstruidos con `observatory.f12o_repeatability_gate.build_anchors()`. Misma cadena exacta `SYSTEM`, SHA del target `O21-start` = `b3a52c881af625e9a56166841a334998ec6a366bbb957cb1eabf3c23f099e105`. Todas las condiciones utilizan la prehistoria **inmutable de 12 llamadas** `(O0-start,O0-disabled,O7-start,O7-disabled,O21-start,O21-disabled)×2` en una instancia recién creada. Después se emite **una sola llamada antecedente** variable, a continuación el target **O21-start**, siempre en posición 2 de su bloque. No se cambia ningún otro prompt ni se concatenan historiales como mensajes.

**Factor A — antecedente inmediato:** `O21-disabled` o `O0-disabled`. Ambos son prompts congelados del banco consumido. Sus longitudes/tokenizaciones pueden diferir; por ello el factor representa la *identidad completa del prompt antecedente*, no su significado semántico puro.

**Factor I — estado de instancia:** `reuse` (misma instancia realiza 12 llamadas + antecedente + target); `fresh` (tras 12 llamadas + antecedente, destruir contexto e inicializar una nueva instancia `FrozenGGUF` en el **mismo proceso hijo** y pasar a ella el target como su primera llamada). Este contraste identifica reinicio de contexto de instancia; no reinicia la biblioteca nativa ni la máquina.

**4 celdas congeladas**: 0 O21-disabled/reuse, 1 O21-disabled/fresh, 2 O0-disabled/reuse, 3 O0-disabled/fresh.

**4 trabajadores independientes** `W0..W3`, cada uno evalúa una réplica de las 4 celdas, **un proceso hijo OS completamente nuevo por celda**, orden contrabalanceado según cuadrado latino:
- W0: `[0,1,2,3]`
- W1: `[1,3,0,2]`
- W2: `[2,0,3,1]`
- W3: `[3,2,1,0]`

Por tanto cada celda aparece una vez en cada posición de ejecución entre los cuatro workers; no hay arrastre de estado entre celdas. Total: **4×4×(12 antecedentes fijos +1 antecedente variable +1 target) = 224 llamadas reales**, desglosadas en **208 anteriores al target y 16 target**. El número de llamadas y posición de target se iguala en las cuatro celdas; cada target se evalúa una sola vez por proceso hijo.

## Modelo y observabilidad

Congelado `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, HF revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, pesos SHA256 `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, llama-cpp-python `0.3.36`, CPU `n_threads=4`, `n_ctx=4096`, `n_batch=256`, `seed=0`, `temperature=0`, `top_p=1`, `max_tokens=96`, `chat_format=chatml`. Usar `FrozenGGUF` sin edición. Registrar pesos, backend/versión, worker, PID nuevo por celda, orden, respuestas raw de **cada llamada**, SHA del SYSTEM/prompt, prompt/completion tokens, instante y duración, `model.calls`, `generator.n_tokens`, `n_ctx()` y tipo de cache si disponible. **No inventar hits KV**: `kv_cache_hits_available=false` si llama.cpp no los ofrece. Distinguir esta posible persistencia de contexto de memoria/experiencia de INFINITO: no se escribe ni recupera memoria personal ni se modifican políticas.

## Gates y decisión — congelados antes de resultados

- **S0 ejecución real y completa:** 4 workers únicos; 16 celdas únicas worker×tratamiento; 16 PID hijos por estudio, sin reutilización dentro de worker; Latin square respetado; 224 llamadas, 208 antecedentes y 16 targets; todas con SHA/raw/acción válida; mismos 6 prompts congelados, prehistoria exacta, temperatura y sin verdad privada. Falta => `INVALID_S_PROTOCOL`.
- **S1 pesos, procedencia y reinicio:** exactitud peso/revisión/backend, CPU, n_threads, n_ctx y llamadas; `reuse` target llama #14 en la instancia y `fresh` target #1; mismo target O21-start y SHA para todos; ningún mundo F12-P-T consultado. Error => `INVALID_S_PROTOCOL`.
- **S2 concordancia literal:** para cada celda, las cuatro réplicas en cuatro workers producen raw idéntico (no solo la acción). Fallo => `REPLICATE_VARIANCE` y no declarar interacción confirmada.
- **S3 patrón conjunto principal:** entre las 4 réplicas de cada celda se obtiene exactamente `O21/reuse = q2`, `O21/fresh=q0`, `O0/reuse=q0`, `O0/fresh=q0`. Con S0–S2 PASS => `INTERACTION_REPLICATED` en este modelo/prompt concreto. Como control de magnitud, publicar la interacción en proporción de q2: `[p(q2|O21,reuse)-p(q2|O21,fresh)]-[p(q2|O0,reuse)-p(q2|O0,fresh)]` (esperada +1).
- **S4 interpretación alternativa:** si S0–S2 PASS pero S3 falla, clasificar datos como `INSTANCE_GENERAL_SIGNAL`, `PREDECESSOR_SIGNAL`, `NO_SIGNAL` o `MIXED_CONTRAST` según tabla observada; **no** cambiar gates. Publicar siempre los cuatro RAW y contraste. La causa interna física (cache hits, atención o sampling) queda **no demostrada** sin instrumentación específica.
- **S5 límites:** no reutilizar los 24 mundos de transferencia, no modificar Producto, no confundir persistencia del contexto de inferencia con aprendizaje del agente, mantener R/Q/P sellados, evitar afirmaciones de generalización a otros modelos o hardware.

**Estado final:** `INVALID_S_PROTOCOL` si S0/S1/S5 fallan; `REPLICATE_VARIANCE` si S2 falla; `INTERACTION_REPLICATED` si S3 pasa; de lo contrario etiqueta alternativa S4. No se modificarán ni criterios ni prompts tras ver resultados. El PR permanecerá DRAFT. Un efecto replicado solo permite diseñar confirmación independiente en otro entorno; no habilita integración de Producto ni necesariamente la transferencia de F12-P.
