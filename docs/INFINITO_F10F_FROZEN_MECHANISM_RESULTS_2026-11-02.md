# INFINITO F10-F — Auditoría interna de topk5, prioridades de firma y cabeza congelada

**2026-11-02. Resultado técnico prospectivo: `F10F_MECHANISM_AUDIT_COMPLETE`. NO es un éxito cognitivo ni confirmación del mecanismo.**

- [Prerregistro F10-F](./INFINITO_F10F_FROZEN_MECHANISM_PREREG.md) fijado en commit `34637e5e3c745d921a28127bb2e6169eb7181d69` **ANTES de escribir generador, auditor y abrir semillas nuevas**.
- [GitHub Actions #41458774516](https://github.com/webtilians/principiodelTodo/actions/runs/41458774516) **SUCCESS**: 4 réplicas, tests previos, pesos originales congelados validados SHA, reconstrucción exacta de logits `READ` y `SKIP`, agregador científico stdlib independiente.
- Artefactos: `f10f-actor-0` ... `f10f-actor-3` con **cada uno de los cinco slots por estado**, `unary`, `pairwise_rank`, `attention_logit`, `attention_weight`, pertenencia al último firmado, resultados de todos los patches y 432 filas por réplica; `f10f-independent-mechanism-audit` con totales auditados.

## Diseño y límites

Se mantienen absolutamente congelados los actores **originales F10-C `relational_read` y `unary_read_control`**, recuperados del run **41447171697**, manifest y checkpoint SHA verificados. Ningún entrenamiento ni actualización de modelo, ninguna cabeza READ. Se evalúan **432 nuevos mundos** con semillas `17_200_000+10_000*i+j`, i0..3, j0..107, 4 historias factoriales públicas por mundo (**1728 estados dependientes**). El generador replica el 2×2 de F10-E en **semillas nuevas** (informes X/C firmados/no firmados antes/después, k3/5/8, tiempo early/middle/late); NO es validación de una nueva familia de tarea ni confirmación independiente de generalización.

**Paridad de instrumentación**: diferencia máxima de logit READ y SKIP frente al `RelationalActor.forward` original = **0 en las cuatro réplicas**. El probe no cambió ni una capa ni un peso; cada registro se identifica por índice de slot original del FIFO.

## Hallazgo A — el selector topk5 no ordena por fecha ni acreditación

El algoritmo original usa `torch.topk(match,5)`, donde `match` significa **coincidencia de la clave solicitada**, no firma ni timestamp. En los **432 pares** `S_before → S_after`, el conjunto de cinco posiciones topk **no cambió en ninguna réplica** (0/432 cambios). Es una propiedad arquitectónica y fue corroborada por la auditoría, no una prioridad temporal aprendida por topk.

Cuando la clave tiene solo **3 o 5 informes**, el registro ganador está siempre dentro de los cinco slots seleccionados: 1152/1152 casos. Con **8 informes**, solo está en **336/576 = 58,3%** de estados, porque el ranking topk no sabe qué cinco informes son relevantes para verdad acreditada. Esto crea un **fallo de selección comprobable**, pero H/GRU aún pueden contener información y esa ausencia **no prueba que la decisión tenga que fallar**.

## Hallazgo B — score de versiones firmadas, atención e inversión temporal

Cada slot recibe `score_i=4×(own_i+pairwise_rank_i)`; `own` contiene la función unaria aprendida de firma, timestamp y fuente, y `pairwise_rank` el promedio de comparaciones de ese registro contra los otros seleccionados (siete features públicos). La red usa `softmax(score)` para obtener pesos y una lectura escalar agregada de valor y firma. **No existe una regla programada «elige el firmado con timestamp más reciente»**.

En la variante crítica **`S_after`**:

| Réplica | Ganador firmado incluido en topk5 | Ganador recibe la mayor atención | Atención media total al ganador* |
|---|---:|---:|---:|
| 0 | 86/108 | 19/108 | **15,2%** |
| 1 | 91/108 | 87/108 | **63,3%** |
| 2 | 88/108 | 43/108 | **19,3%** |
| 3 | 91/108 | 24/108 | **24,5%** |
| **Total** | **356/432 = 82,4%** | **173/432 = 40,0%** | **30,6%** |

\*Media sobre todos los casos de la celda, asignando atención 0 si el ganador quedó fuera de topk.

Por tanto, **el ganador suele estar presente pero NO suele ser el registro más atendido**; existe una heterogeneidad enorme entre políticas con semillas distintas. Al invertir el orden `S_before→S_after` sin cambiar el conjunto topk, la variación media del peso del ganador, en los pares donde ambos ganadores están incluidos, fue **−0,013, −0,011, +0,013 y −0,131** por réplica. No hay inversión temporal uniforme del patrón de atención.

**Datos brutos por slot**: los JSON adjuntos a Actions registran para los 1728 casos cada `own`, `pairwise_rank`, `attention_logit` y `attention_weight`; estas métricas NO estaban disponibles en F10-E. Las medias de atención aquí provienen del agregado real del probe, no de suposiciones.

## Hallazgo C — tres clases exclusivas de error, parche de evidencia correcto y límite de atribución

El registro ganador se obtiene **SOLO para el auditor** de la regla del último firmado público. El modelo original nunca recibe esa selección. Definimos tres diagnósticos excluyentes sobre errores READ auténticos:

- **A `MISSING_WINNER`**: ganador no seleccionado por topk.
- **B `HEAD_OR_HIDDEN_LIMIT`**: ganador seleccionado, pero proporcionar al `head` *solo* ese registro, manteniendo exactamente el H recurrente original, **no** corrige la respuesta.
- **C `MIXING_OR_WEIGHTING_SENSITIVE`**: ganador seleccionado, y sustituir lectura agregada por lectura onehot del ganador **sí** corrige la respuesta.

**IMPORTANTE: B NO prueba un defecto aislado de `head`**. El onehot del ganador es un input artificial **fuera de la distribución de lecturas durante entrenamiento**; la cabeza puede haber aprendido a interpretar mezclas con una calibración distinta, o interactuar con el estado GRU. No promover B a evidencia de «falta razonamiento».

En la celda difícil `S_after` (108 mundos por réplica):

| Réplica | READ correcto | Errores A: ganador fuera | Errores B: onehot no corrige | Errores C: mezcla sensible | Dominante preregistrado |
|---|---:|---:|---:|---:|---|
| 0 | 35/108 | 19 | **54** | 0 | B |
| 1 | 78/108 | **17** | 0 | 13 | A |
| 2 | 39/108 | 20 | **49** | 0 | B |
| 3 | 25/108 | 16 | **48** | 19 | B |
| **Total** | **177/432 = 41,0%** | **72** | **151** | **32** | **B en 3/4 réplicas** |

De **255 errores S_after**, **28,2%** son A, **59,2%** B, **12,5%** C. Hay **transferencia parcial de puntuación atencional en la réplica 1**, pero las otras tres no interpretan correctamente el onehot artificial.

Sobre **todas las cuatro variantes**: READ original **1347/1728 = 78,0%**; `SKIP` mismo actor **858/1728 = 49,7%**; control unario congelado **1127/1728 = 65,2%**. Los errores READ se distribuyen en A **89**, B **260**, C **32**, 381 errores totales. La réplica 1 muestra `PATCH_WINNER_ONEHOT` exitoso cuando el ganador ya fue seleccionado, pero 0 y 2 fallan **en los 432/432 casos** de onehot artificial con ganador; esto apoya la advertencia sobre calibración/representación, sin identificar responsable individual.

## Parches diagnósticos: por qué «filtrar no firmados» tampoco es la respuesta general

Usando la regla de firma del **observador** para renormalizar atención sobre los slots firmados ya seleccionados (`PATCH_SIGNED_ONLY`), en `S_after` se mejoró de **177/432 a 299/432**. Sin embargo, aplicarlo **indiscriminadamente a las cuatro variantes** empeora el resultado global de **1347/1728** a **826/1728**. Por eso el cambio NO puede presentarse como solución del agente ni justificarse como hotfix de F10-C.

Forzar el ganador a entrar entre los cinco slots (`PATCH_WINNER_INCLUDE` con nuevo score original) dejó `S_after` en **192/432** (solo +15 aciertos netos frente al original), mientras proporcionar solo el ganador al head tras inclusión obtuvo **160/432**. Estos parches son pruebas asistidas por información privilegiada del evaluador, no políticas operativas. En particular, mejorar la presencia de ganador **no garantiza** resolver la predicción, y una lectura onehot puede degradarla al alterar la distribución interna aprendida.

## Resultado y decisión

**Estado técnico formal: `F10F_MECHANISM_AUDIT_COMPLETE`.** Cuatro modelos congelados, 432 mundos nuevos, 1728 variantes, paridad exacta, cinco checkpoints? **Dos checkpoints F10-C por réplica** (relacional y unario), todos con SHA verificado. Sin reentrenar ni abrir `17_300_000+`, `16_300_000+` ni confirmaciones previas.

**Interpretación científica**: hemos aislado un **límite estructural del selector por clave** al haber >5 informes, y un segundo cuello de botella relacionado con cómo se puntúan/mezclan/evalúan las versiones firmadas. Los errores al interpretar lecturas onehot asistidas por observador no separan por sí solos el `head` del estado recurrente. La heterogeneidad de replicas 0/2 frente a 1 es esencial y debe mantenerse visible.

**Siguiente hipótesis posible, en un estudio prospectivo SEPARADO (no ejecutado):** comparar modelos de IGUAL capacidad entrenados en NUEVOS mundos con (1) un extractor que elija último firmado mediante operación diferenciable temporal acreditada y (2) el mismo extractor con supervisión de lectura desacoplada de la cabeza. Los criterios deben exigir generalización a número variable de registros, causalidad READ−SKIP y fidelidad a la actualización firmada incluso con mensajes posteriores no firmados. Evitar usar un `winner` elegido por evaluador como input del actor, y no volver a tocar pesos F10-C. No entrenar cabeza READ hasta demostrar selección y uso auténtico sin asistencia.
