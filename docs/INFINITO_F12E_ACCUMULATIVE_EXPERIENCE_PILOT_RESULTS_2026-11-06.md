# INFINITO F12-E — Resultados piloto con LLM real 7B, memoria episódica y estrategias inferidas
**Fecha: 2026-11-06.** Estado técnico: **SUCCESS**. Estado científico: **TRAIN-heldout diagnóstico, resultados mixtos, NO-GO a confirmación o integración**. No se debe convertir una hipótesis que cita fuentes equivocadamente en política validada.

## Procedencia y garantías
- [Prerregistro prospectivo](INFINITO_F12E_ACCUMULATIVE_EXPERIENCE_PREREG_2026-11-06.md), commit **f1e92f9ac884061907501cc896b818005bc6aeb4**, guardado ANTES de implementar F12-E, entrenar o ver resultados.
- Implementación `observatory/f12e_experience.py`, tests `tests/test_f12e_experience.py`, workflow `.github/workflows/infinito-f12e-experience.yml`. No modificar `src/infinito3` ni el servidor personal; almacén real `SQLiteCognitiveMemoryStore(path=":memory:")` y `HashEmbeddingProvider` de INFINITO.
- [**GitHub Actions #42157861095**](https://github.com/webtilians/principiodelTodo/actions/runs/42157861095), commit de lanzamiento **b74bd3622e44dc7452334fbfba28f24255e91235**: contratos de tests **SUCCESS**, inferencia Qwen7B real y comparación de brazos **SUCCESS**, artefacto `f12e-frozen7b-sqlite-experience-pilot` ID **13475793425**, SHA256 ZIP **567dfe105d81e0754bf3bab5342cd66e203109edfbb00960636fae1e54bd1ff5**, retención 30 días. Contiene `pilot_report.json` completo (pasos, feedback, respuestas crudas, SHA prompt, fuentes y coste) y `summary.json`.
- Modelo EXACTAMENTE el de F12-D: `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, HF revision **8911e8a47f92bac19d6f5c64a2e2095bd2f7d031**, hash de pesos **65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423**, `llama-cpp-python==0.3.36`, CPU, temperatura 0, sin fine-tuning ni cambios de pesos. **111 inferencias reales**, 59.499 tokens de entrada, 656 de salida, 1.006,84 s acumulados de inferencia del LLM. No confundir con el tiempo total del workflow.
- Se ejecutó **una sola réplica piloto**: 6 casos TRAIN `S0..S5` (3 perfiles opacos `nacre/cinder/spruce`, R1 luego R2), seguida de una síntesis con el MISMO LLM y 6 casos heldout `H0..H5` (servicios nuevos, mismas combinaciones visibles de clave y revisión), cada uno evaluado con presupuestos de **2 o 3 acciones** bajo A0, A1, A2. 36 episodios de evaluación emparejados. El actor nunca recibe la compatibilidad interna del simulador.

## Experiencias realmente observadas — TRAIN

| Caso | Perfil | Revisión | Acciones elegidas por el LLM | Verificación positiva |
|---|---|---|---|---|
| S0 | nacre | R1 | inspect → verify (fallida) → apply_delta (aceptada) → verify | **Sí** |
| S1 | cinder | R1 | inspect → verify (fallida) → apply_delta (rechazada) → apply_sigma (aceptada) | **No; faltó verify** |
| S2 | spruce | R1 | inspect → verify (fallida) → apply_delta (rechazada) → apply_sigma (aceptada) | **No; faltó verify** |
| S3 | nacre | R2 | inspect → verify (fallida) → apply_delta (rechazada) → apply_sigma (aceptada) | **No; faltó verify** |
| S4 | cinder | R2 | inspect → verify (fallida) → apply_delta (aceptada) → verify | **Sí** |
| S5 | spruce | R2 | inspect → verify (fallida) → apply_delta (aceptada) → verify | **Sí** |

**3/6 tareas TRAIN terminadas con verificación positiva**, y seis episodios reales almacenados como EPISODIC. Los episodios fallidos contienen **información causal útil** (rechazo de `delta` seguido de aceptación de `sigma`) que puede recuperarse, pero no deben presentarse como verificación final.

### Hallazgo crítico: falsa procedencia de la estrategia sintetizada

Tras TRAIN, Qwen7B generó una única hipótesis que afirmaba, en sustancia, que la reparación `delta` era menos eficaz y que `sigma` había funcionado **en S4 y S5**, citando expresamente `S4, S5`. **Esto es FALSO SEGÚN LAS TRAZAS**: ambas experiencias S4/S5 utilizaron `apply_delta`, recibieron aceptación de `delta` y un `verify` positivo; **no ejecutaron `sigma`**. Las experiencias S1/S2/S3 sí mostraron aceptación de `sigma` pero ninguna terminó con `verify` positiva.

El filtro preregistrado de promoción sintáctica marcó `minimum_evidence_met=true` porque existían 3 episodios exitosos y se citaban 2 de ellos; también declaró **`substantive_content_verified=false`**. Es decir: cumplió la prueba de **existencia de fuentes**, no de que la **afirmación esté respaldada por las fuentes**. Se guardó únicamente como memoria `SEMANTIC` con estado explícito `unverified_observed_hypothesis`, no como política verdadera del producto. **No hay evidencia de aprendizaje de una política correcta; la cita inventada es un fallo de fundamentación y procedencia**.

No se recalculan umbrales ni se modifica el código tras leer el heldout. Este control factual no estaba en el gate previo, y debe describirse como **auditoría posterior**: no una confirmación preespecificada ni un nuevo resultado de ejecución.

## Resultados independientes dentro de TRAIN: tres brazos y dos presupuestos

| Brazo | Tight, hasta 2 acciones | Slack, hasta 3 acciones | Total |
|---|---:|---:|---:|
| **A0 — Sin memoria** | **0/6** | **0/6** | **0/12** |
| **A1 — INFINITO episódico** | **1/6** | **5/6** | **6/12** |
| **A2 — Episodios + estrategia del LLM** | **3/6** | **3/6** | **6/12** |

A1 en slack mejoró respecto A0 **+5/6 éxitos** en tareas con un presupuesto de tres pasos. Sin embargo, con dos pasos **solo logró +1/6**, por debajo del criterio preregistrado de mejora de al menos +2/6. No se debe declarar PASS general de la utilidad de memoria bajo el protocolo principal.

A2 obtuvo **+2/6 frente a A1 en tight** (3 frente a 1), cumpliendo formalmente el subcriterio numérico `P3_strategy_incremental_success_tight`, pero **retrocedió −2/6 en slack** (3 frente a 5). Además la estrategia global era **factualmente incorrecta** y la superioridad tight no prueba que el sistema haya aprendido una regla válida. En total A1 y A2 empataron en 6/12.

Desglose por revisión, dentro de 3 casos R1 y 3 R2 por presupuesto:

| Presupuesto / brazo | R1 | R2 |
|---|---:|---:|
| 2 pasos A0 | 0/3 | 0/3 |
| 2 pasos A1 | 0/3 | 1/3 |
| 2 pasos A2 | 2/3 | 1/3 |
| 3 pasos A0 | 0/3 | 0/3 |
| 3 pasos A1 | 2/3 | **3/3** |
| 3 pasos A2 | 2/3 | **1/3** |

La degradación A2 frente A1 en R2 con tres pasos es particularmente importante en un experimento de memoria temporal: la simplificación de la estrategia favoreció indebidamente `sigma` donde correspondía `delta`. **La obsolescencia y los cambios de versión son obstáculos reales; la memoria comprimida puede destruir información esencial.**

## Qué sucedió en cada caso de heldout

| Caso | Revisión | Tight A0/A1/A2 | Slack A0/A1/A2 |
|---|---|---|---|
| H0 (nacre) | R2 | fallo / **éxito** / **éxito** | fallo / **éxito** / **éxito** |
| H1 (cinder) | R1 | fallo / fallo / **éxito** | fallo / **éxito** / **éxito** |
| H2 (spruce) | R2 | fallo / fallo / fallo | fallo / **éxito** / fallo |
| H3 (nacre) | R1 | fallo / fallo / fallo | fallo / fallo / fallo |
| H4 (cinder) | R2 | fallo / fallo / fallo | fallo / **éxito** / fallo |
| H5 (spruce) | R1 | fallo / fallo / **éxito** | fallo / **éxito** / **éxito** |

A0 usó **`inspect → verify`** inmediatamente en todas las tareas (sin haber reparado) y con tres pasos añadió usualmente `apply_delta` demasiado tarde, después de la verificación fallida; esto muestra **una debilidad del agente y de la plantilla de tarea sin memoria**, y amplifica la diferencia A1–A0. **No concluir que INFINITO generaliza mejor a incidencias reales** ni atribuir automáticamente todo el fracaso A0 a falta de información: no se ha aislado la política previa del LLM.

El retrieval de A1 utilizó `SQLiteCognitiveMemoryStore.search` verdadero más un filtro por **perfil público**, y conservó **dos experiencias por perfil** (R2 y R1), mostrando primero la revisión nueva aunque el caso consultado fuese R1. A2 tuvo acceso a episodios similares, pero cedió parte del mismo límite de contexto (1.200 caracteres) a la hipótesis sintetizada. El modelo no modificó recuerdos durante evaluación; solo pueden cambiar contadores de lectura del almacén.

## Coste observado de evaluación (sin incluir entrenamiento ni síntesis)

| Brazo | Llamadas a herramientas | Tokens de entrada | Tokens de salida | Éxitos |
|---|---:|---:|---:|---:|
| A0 | 30 | 11.470 | 156 | 0 |
| A1 | 29 | 18.285 | 156 | 6 |
| A2 | 27 | 17.054 | 151 | 6 |

A2 usa **1.231 tokens de entrada menos que A1** en el agregado (−6,7%) y dos llamadas menos, pero las seis victorias **no son las mismas** ni la política es correcta; esto no satisface el objetivo explícito de conocimiento validado. En tight, A2 consumió **7.274** tokens de entrada frente a **7.079** de A1 (2,8% más), no el ahorro de ≥15% preregistrado para alternativa a P3. En slack A2 consumió menos pero perdió dos victorias. No se midieron costes financieros de plataforma; tiempos CPU reales constan por decisión.

## Gates prospectivos: resultado literal sin mover umbrales

| Gate | Resultado | Interpretación |
|---|---|---|
| P0 LLM congelado, condiciones emparejadas, registros sin fuga | **PASS** | SHA de pesos correcto; 36 ejecuciones; contenido de recuerdos inmutable durante heldout |
| P1 ≥3 TRAIN exitosos y ≥2 citas de fuente exitosa | **PASS sintáctico** | **Falla la validez factual** de la afirmación citada, detectada mediante auditoría posterior |
| P2 memoria A1 ≥A0+2/6 en tight | **FAIL** | 1 vs 0 |
| P3 A2 ≥A1+2/6 tight sin empeorar revisiones | **PASS numérico** | 3 vs 1, pero estrategia citada incorrectamente y slack peor |
| P3 alternativa ≥15% menos tokens sin perder éxitos | **FAIL** | tight +2,8% de tokens |
| P4 reporting R1/R2 | **PASS** | Se desglosan ambas revisiones |

**Resultado global preregistrado: NO-GO para confirmación o integración.** P2 FAIL y la síntesis incumple el objetivo cualitativo de procedencia fiable a pesar de superar la comprobación formal de citas. No reajustar el modelo, los recuerdos, el prompt o los umbrales sobre H0..H5 ya consumidos. No reinterpretar este piloto como fracaso de todas las arquitecturas de aprendizaje con memoria: la prueba muestra ventajas episódicas en slack y un problema concreto de generación/verificación de estrategias.

## Decisión recomendada

El siguiente experimento científico, si se abre, debe investigar **validación causal y semántica de las estrategias inducidas**: para cada proposición concreta, extraer la acción, revisión y resultado alegados, verificar que aparecen realmente en los eventos citados, exigir una evidencia de éxito/rechazo coherente y abstenerse si faltan ejemplos. Comparar un selector que **no promociona información contradicha** con A1 episódico y con un control simple de cronología, bajo el mismo presupuesto y **nuevos perfiles/reglas y particiones independientes**. No convertir el validador en un oráculo que conozca el mapa del simulador. El validador solo puede comprobar **lo que registran los recuerdos**, nunca el resultado contrafactual de una acción no ejecutada.

Es necesario corregir también un riesgo metodológico: el contexto bruto de los episodios puede truncarse a 1.200 caracteres; en un próximo estudio registrar y auditar expresamente **texto exacto mostrado** a A1/A2 además de SHA, para comprobar cobertura efectiva de las dos versiones. F12-E guarda IDs/sha, pero no el texto renderizado literal de cada prompt de evaluación (sí sus trazas y respuesta), lo que limita auditoría independiente.

No se abrieron DEV/CONFIRM de F11, no se modificó `src/infinito3`, memoria MCP de usuario, claves ni sistemas externos. PR F12-E se mantiene en borrador.
