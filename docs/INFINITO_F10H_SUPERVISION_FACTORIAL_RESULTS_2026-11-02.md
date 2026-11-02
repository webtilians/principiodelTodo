# INFINITO F10-H — Factorial de supervisión con varias versiones acreditadas

**2026-11-02. Resultado final:** `F10H_TECHNICAL_COMPLETE` y **`F10H_FACTORIAL_SUPERVISION_FAIL`** según puertas preregistradas. Hallazgo positivo concreto: **las dos supervisiones aisladas (selección o interpretación) alcanzan 100% de selección y decisión operacional OOD en 4/4 réplicas, y la combinación de ambas no las mejora**. Ningún gate se cambió después de evaluar.

[Prerregistro F10-H](./INFINITO_F10H_SUPERVISION_FACTORIAL_PREREG.md) fijado prospectivamente en **`02e9f0d056e312016ade0fa43a31a69e5c38d705`** (clarificación preimplementación de fuentes invariables `41238d77cfdeb6eac990ccfaaf1dce18c0063bfa`); [GitHub Actions #41515632693](https://github.com/webtilians/principiodelTodo/actions/runs/41515632693) **SUCCESS técnico**: 4 tareas TRAIN+DEV/OOD, 5 checkpoints por réplica y auditoría independiente de todas las filas, pesos y gates. Artifacts del run: `f10h-actor-0` … `f10h-actor-3` (20 checkpoints .pt + training.json + rows.json + run.log) y `f10h-independent-preregistered-audit` con `f10h-audit.json`. El agregado independiente no entrena ni elige modelos.

## Aislamiento experimental

Cuatro brazos con **exactamente la misma arquitectura unaria de F10-G**, 7458 parámetros (sin nuevo modelo): `joint` (solo BCE de decisión), `selection_only` (más CE del índice del último firmado en TRAIN), `interpretation_only` (más BCE de la cabeza al recibir aisladamente el valor del último firmado en TRAIN) y `both` (ambas pérdidas). Control GRU-only de parámetros cercanos, sin memoria externa. Las cuatro variantes comparten pesos iniciales e idéntico orden de minibatches **dentro de cada réplica**, 20 épocas, batch64, lr=.002, 640 pasos y 40.960 presentaciones por brazo. **Etiquetas y costes adicionales NO están igualados entre objetivos**, porque son precisamente la intervención causal de entrenamiento; mismo número de pesos/pasos no implica mismo cómputo.

Cuatro réplicas × [512 mundos TRAIN `19_100_000+`, 128 mundos DEV `19_200_000+`, 128 mundos OOD `19_400_000+`] × 4 variantes factuales por mundo; **2048 mundos TRAIN, 512 DEV y 512 OOD** (8192 / 2048 / 2048 decisiones relacionadas). Cada historia contiene 48 informes, 24 claves, timestamps públicos independientemente del orden físico y una consulta binaria XOR. TRAIN/DEV 7 o 9 informes de clave, OOD 12 o 16. **Mínimo cuatro versiones acreditadas y al menos tres cambios de valor firmado en todas las variantes**, extendiendo hasta 6–9 informes firmados OOD. Fuente A/B es ruido IID en TRAIN/DEV y cuatro patrones balanceados en OOD; la fuente permanece vinculada al evento al manipular firma de X. Solo `S_after` exige cambiar al valor contradictorio X, que entonces es el último firmado. Winner obtenido de firma y tiempo públicos; no entra en inferencia READ autónoma, sí como **supervisión auxiliar durante entrenamiento** para los brazos indicados.

No se han reutilizado TRAIN/DEV/OOD F10-G, F10-F ni modelos anteriores. Confirmación `19_300_000+` sellada y sin ejecutor. Los cinco checkpoints de cada réplica se guardaron **ANTES** de generar DEV/OOD.

## Hallazgo 1 — READ OOD: una sola supervisión adicional es suficiente en esta tarea

| Réplica | Solo decisión (`joint`) | + selección `selection_only` | + interpretación `interpretation_only` | + ambas `both` | GRU-only |
|---|---:|---:|---:|---:|---:|
| 0 | **48,83%** | **100%** | **100%** | **100%** | 48,44% |
| 1 | **100%** | **100%** | **100%** | **100%** | 53,91% |
| 2 | **100%** | **100%** | **100%** | **100%** | 50,39% |
| 3 | **52,15%** | **100%** | **100%** | **100%** | 54,10% |
| **Total descriptivo** | **75,24%** | **100% (2048/2048)** | **100% (2048/2048)** | **100% (2048/2048)** | **51,71%** |

Tanto `selection_only` como `interpretation_only` lograron también **100% de selección del índice acreditado y 100% de interpretación con valor aislado** en los cuatro OOD. No es solo un `oracle_patch`: **READ normal, sin suministrar ganador, obtuvo 100%**; el parche privilegiado se evaluó por separado. El resultado `both` implica 4/4 variantes correctas en **los 512 mundos base OOD**, incluyendo los dos valores de k y los cuatro cambios de régimen fuente. No afirmar 100% poblacional fuera de este generador acotado.

La supervisión del **selector** puede producir una lectura consistente que permita aprender a la cabeza mediante BCE de la acción final. Inversamente, supervisar la **interpretación del valor aislado** puede estabilizar gradientes y facilitar aprender qué informes atender con la sola BCE de la decisión. **Ambas explicaciones son compatibles con datos; la instrumentación no prueba las rutas neuronales de aprendizaje.**

## Hallazgo 2 — comparación preregistrada entre pérdidas

| Réplica | G1 transferencia de `both` | G2 beneficios independientes de ambas pérdidas | G3 complementariedad de `both` |
|---|---|---|---|
| 0 | **PASS** | **PASS** | FAIL |
| 1 | **PASS** | FAIL | FAIL |
| 2 | **PASS** | FAIL | FAIL |
| 3 | **PASS** | **PASS** | FAIL |
| **Gate total** | **PASS 4/4** | **FAIL 2/4** | **FAIL 0/4** |

- **G1 PASS 4/4**: `both` supera selección/READ 90%, todos los estratos/variantes 85%, 4/4 mundos 80%, y los límites inferiores bootstrap de READ−SKIP >+0,20, READ−SHUFFLE >+0,10.
- **G2 FAIL 2/4**: las réplicas 0 y 3 muestran ganancia de selección `selection_only−joint` **+93,16 pp**, IC95 positivos, y de interpretación diagnóstica `interpretation_only−joint` **+51,37 pp** y **+43,95 pp**, IC95 positivos. En réplicas 1 y 2 el conjunto `joint` ya alcanza 100% en ambas métricas: delta e IC son exactamente cero. Se requerían al menos **3/4** réplicas. Los efectos techo **no autorizan revisar la puerta tras observar datos**.
- **G3 FAIL 0/4**: `both` no gana a `selection_only` ni a `interpretation_only` en ninguna réplica: `both−best(single)=0`, IC [0;0] siempre. Además, `both−joint` es +51,17 pp en réplica0, 0 en 1/2 y +47,85 pp en 3, no cumple >=3/4. **No hay evidencia de complementariedad positiva de las dos pérdidas en esta familia.** El resultado global formal queda `F10H_FACTORIAL_SUPERVISION_FAIL`.

## Hallazgo 3 — memoria causal y ausencia de oráculo en READ

`both` READ **2048/2048** frente al mismo actor SKIP con H recurrente idéntico, que acierta **1064/2048 (51,95%)**: **+48,05 pp**. Por réplica la ventaja READ−SKIP y sus IC95 bootstrap agrupados por mundo:

| Réplica | Delta READ−SKIP | IC95% mundo base |
|---|---:|---|
| 0 | **+50,78 pp** | [+46,48; +55,08] pp |
| 1 | **+49,22 pp** | [+44,92; +53,52] pp |
| 2 | **+48,44 pp** | [+44,53; +53,13] pp |
| 3 | **+43,75 pp** | [+39,45; +48,05] pp |

La auditoría confirmó todos los IC inferiores >20pp y el gate READ−SHUFFLE >10pp por réplica; los valores completos de SHUFFLE, ERASE_SIGN, ERASE_TIME, selección/decisión, OOD por fuente/k y todas las filas están en `f10h-audit.json`. Las ablaciones modifican el canal externo solamente, no borran los mismos tokens del historial GRU. `oracle_patch` es evaluación privilegiada, no acción del modelo en READ.

## Integridad y riesgos

**20 checkpoints** originales de F10-H verificados SHA. **20.480 filas** DEV/OOD y **12.288 hashes únicos TRAIN/DEV/OOD** verificados sin intersección entre namespaces y sin diferencias de historias/etiquetas/ganador entre brazos. Las 4 variantes de cada mundo se agrupan para inferencia; no tratar 2048 estados OOD como mundos independientes. Workflow técnicamente SUCCESS aunque los gates G2/G3 fallen. Los cambios de fuente fueron declarados antes de datos y no se introdujeron entrenamientos extra ni criterios nuevos.

**Limitaciones determinantes:**
1. Los mundos tienen como máximo 9 versiones firmadas, 48 eventos, clave explícita, timestamps perfectos, firmas sintéticas y respuesta binaria XOR; C/X siguen siendo los dos últimos candidatos firmables aunque se exijan >=3 cambios acreditados anteriores. No se prueba prioridad entre arbitrarias fuentes reales o historias naturales.
2. Cada pérdida auxiliar usa durante TRAIN el **índice de la respuesta correcta derivado de la regla pública**. Que una señal baste para aprender no equivale a aprender cuándo consultar ni crear políticas propias sin supervisor.
3. `joint` ya aprende 100% en 2/4 réplicas con el mismo número de pasos. Sin replicar sobre tareas más variadas, las diferencias de estabilidad entre inicializaciones podrían depender de esta distribución particular.
4. Identidad de parámetros, datos y pasos no iguala cómputo de backward ni coste de anotaciones. F10-H utiliza únicamente el **unario**, no confirma ventaja del relacional F10-C ni F10-G; sus modelos no son comparables directamente porque las familias de mundos difieren.
5. IC bootstrap por mundo describen variabilidad de mundos en cada réplica; cuatro inicializaciones no justifican certeza universal sobre robustez de optimización ni tasas de error poblacionales.

## Decisión

**Conservar PR borrador, no abrir confirmación ni retocar pérdidas/épocas después de ver OOD.** La pregunta F10-G **sí se ha respondido parcialmente**: para esta familia difícil y un solo actor, cualquiera de los dos auxiliares por separado bastó para eliminar los fallos observados de selección/interpretación; no hay beneficio adicional medible de combinarlos. Sin embargo **la supervisión explícita sigue siendo necesaria para la consistencia 4/4 de estas réplicas**, y la versión solo-decisión presenta bifurcación 2/4 éxitos y 2/4 cerca del azar.

El siguiente experimento debería preguntar si pueden aprender esas señales **sin etiquetas del ganador** (objetivo autosupervisado, contrafactual basado únicamente en observaciones y recompensa, sin derivar etiqueta directa del evaluador), y confirmar transferibilidad a historias cuyo último ganador NO esté siempre entre C/X identificables. Eso sería un estudio F10-I nuevo, con mundos más adversariales y preregistro distinto: **no ejecutarlo sobre estos OOD**.
