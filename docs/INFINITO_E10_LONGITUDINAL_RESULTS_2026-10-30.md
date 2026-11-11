# INFINITO E10 — Resultados confirmatorios de utilidad longitudinal de memoria (fase offline)

**30 de octubre de 2026 — evaluación completa, protocolo E10-A superado dentro de la tarea sintética definida.** No significa que todo E10 o que la autonomía de INFINITO estén demostrados.

- [PR #28 — E10](https://github.com/webtilians/principiodelTodo/pull/28), borrador sobre \`master\`.
- [Protocolo y criterios congelados antes de DEV y confirmación](./INFINITO_E10_LONGITUDINAL_MEMORY_PROTOCOL.md).
- [GitHub Actions #41106177869](https://github.com/webtilians/principiodelTodo/actions/runs/41106177869): integridad **SUCCESS**, desarrollo **SUCCESS**, confirmación **SUCCESS**.
- [Datos agregados versionados, incluidos ocho índices, tasas e intervalos](./INFINITO_E10_RESULTS_2026-10-30.json).
- [Artifact confirmatorio con los 128 mundos/pares, decisiones y contextos](https://github.com/webtilians/principiodelTodo/actions/runs/41106177869/artifacts/13002100493), más [artifact DEV](https://github.com/webtilians/principiodelTodo/actions/runs/41106177869/artifacts/13002150404). Los artifacts tienen retención temporal; el código del generador, protocolo y semillas está versionado en Git para regeneración. No copiar los tokens/UUID efímeros como si fueran datos de resultados científicos adicionales.

## Qué pregunta responde y cuál no

**E10-A (offline):** ¿puede el núcleo ACTUAL de INFINITO 3.0 **conservar y actualizar hechos relevantes** a pesar de distractores y de perder el historial de conversación, suministrándolos a un lector de decisiones fijo, con menos contexto que una recuperación indiscriminada?

**NO es un experimento con un LLM real**, ni prueba de razonamiento natural o de selección semántica compleja. Las decisiones proceden de \`DecisionReader\`, una regla determinista común que únicamente ve los mensajes realmente enviados al adapter \`CognitiveLoop\`, con los dos posibles valores y las reglas de acción en el prompt actual. La etiqueta secreta nunca llega al lector; se usa solo para puntuar. Por ello los resultados miden la **entrega de evidencia útil para una regla de decisión**, no una capacidad amplia de comprensión o agencia.

## Método y controles

- Misma pareja contrafactual de historias por mundo, cada una con **un dato distinto** entre dos opciones, mismo número/secuencia de **16 distractores** antes de primera decisión, y **16 distractores** más tras una actualización de la preferencia al valor opuesto. Las preguntas son exactamente idénticas entre las dos variantes. Dos decisiones por historia (antes y después del cambio) y **cuatro decisiones correctas por par** para acreditar memoria. El agente debe invertir la acción al invertir la pista; elegir siempre la misma no puntúa.
- Dominios limitados a **tres predicados que el núcleo ya reconoce**: \`mi color favorito\`, \`mi bici\`, \`vivo en\`. Estas plantillas son deliberadamente próximas a los patrones de hecho estructurado de INFINITO 3.0 y facilitan al núcleo vigente la extracción de hechos. Esto **favorece estructuralmente a INFINITO** frente a controles sin extracción ni supersesión: se reconoce, no se oculta.
- Cuatro brazos comparten un **lector y presupuesto de 180 tokens estimados** de contexto y \`top_k=8\` en el mismo \`CognitiveLoop\`. \`no_memory\`: nunca persiste; \`fifo_8\`: almacena 8 observaciones recientes; \`rag_hash\`: guarda observaciones y recupera por solapamiento léxico y vector hash, **no embeddings semánticos reales**, sin reglas de contradicción; \`infinito3\`: código de producción congelado \`CognitiveEngine.persistent\` + \`RuleBasedMemoryGate\` + \`SQLiteCognitiveMemoryStore\` + \`GeneralizedContextBuilder\`.
- El historial corto está deshabilitado al decidir para medir memoria externa entregada al modelo. En INFINITO se cierra y reabre físicamente SQLite entre las dos etapas. Baselines conservan sus estructuras de memoria entre etapas, en igualdad funcional; no se les obliga a borrar recuerdos cuando el núcleo los preserva.
- **DEV:** 36 mundos pareados \`1200000+\`, 3 índices de generación; tests pasaron. Confirmación abre \`1300000+index*1000+j\`, index 0..7 y j 0..15, **128 mundos pareados**, 256 historias ×2 decisiones = **512 acciones por brazo**, completamente separados de desarrollo. No se cambió el código ni se afinó el núcleo entre DEV y confirmación. Ninguna semilla F9 se abrió.
- Los ocho índices no son ocho políticas entrenadas (el lector es fijo). La prueba de incertidumbre realiza 1000 remuestreos de diferencias de índices de generación, semilla 1730. **No extrapolar esos intervalos a diversidad de usuarios, modelos o dominios**.

## Resultados principales — NUEVA CONFIRMACIÓN

| Brazo | Pares con las cuatro decisiones correctas | Acciones individuales correctas | Pares con las dos inversiones contrafactuales |
|---|---:|---:|---:|
| Sin memoria | 0/128 (0%) | 0/512 (0%) | 0/128 |
| FIFO 8 | 0/128 (0%) | 0/512 (0%) | 0/128 |
| RAG hash | **7/128 (5,47%)** | **300/512 (58,59%)** | **97/128** |
| **INFINITO 3.0 sin cambios** | **128/128 (100%)** | **512/512 (100%)** | **128/128** |

La diferencia entre **97 pares de inversión** RAG y solo **7 pares enteramente correctos** es importante: cambiar la decisión ante dos variantes históricas no garantiza que se use la **última** información correcta tras una contradicción. Los 300 aciertos individuales del RAG tampoco equivalen a 300 unidades independientes.

**Contexto medio (tokens estimados):** INFINITO 43,66; RAG hash 64,43; FIFO 86,26; sin memoria 0. Estos tokens incluyen el formato del contexto que cada sistema entrega, no el coste real de un proveedor LLM.

**Controles negativos sin hecho:** una ejecución de ausencia de dato por brazo, las cuatro respuestas fueron \`ABSTENER\`. Esto comprueba integridad básica, **no estima de forma robusta una tasa general de alucinaciones**.

## Contrastes preregistrados

- Frente a **FIFO 8**: diferencia **+100 puntos porcentuales** en parejas completas; por índice **[+1,+1,+1,+1,+1,+1,+1,+1]**; bootstrap de índices [100, 100] pp.
- Frente a **RAG hash**: diferencia **+94,53 puntos porcentuales**; por índice **[+93,75,+100,+81,25,+93,75,+100,+87,5,+100,+100] pp**; intervalo bootstrap de índices **[+89,84; +98,44] pp**.
- Ambos contrastes positivos en **8/8 índices**, ganancia agregada >10 pp y límite bootstrap >0. **El criterio de avance específico de este protocolo E10-A se supera**.

En DEV: INFINITO 36/36 pares; FIFO 0/36, RAG hash 0/36. **No se entrenó el sistema con estos resultados ni se corrigieron sus reglas antes de abrir confirmación**. Los resultados confirmatorios quedaron registrados íntegros.

## Qué aprendemos con confianza dentro del banco

1. En datos que encajan con los **predicados estructurados actuales**, SQLite + detección de contradicciones + Context Builder de INFINITO permite que el lector utilice correctamente el **hecho vigente** después de muchas observaciones y de un reinicio de historial.
2. Sin memoria externa ni conservación más allá de ocho eventos, se pierde la información necesaria. Esto es un contraste de capacidad explícitamente construido, no una demostración de que ninguna memoria pequeña pueda resolver casos similares.
3. La recuperación híbrida hash de observaciones sin tratamiento explícito de contradicción suele entregar información útil para la primera decisión, pero falla con frecuencia en el ciclo completo tras sobrescritura.
4. El resultado mide utilidad de **recuperación/persistencia contextual y supersesión de hechos**, no aprendizaje de importancia. La regla del gate y el esquema de hechos ya estaban programados.

## Lo que NO debemos afirmar

- **No** se ha probado contra un RAG semántico fuerte con extracción de hechos, actualización de preferencias, filtrado de versiones y presupuesto exactamente igual. Esa comparación es esencial antes de declarar superioridad general de INFINITO frente al estado del arte.
- **No** se ha utilizado un LLM de verdad; el lector fijo no comete errores de interpretación propios del lenguaje natural.
- **No** se ha probado generalización a expresiones no presentes en los patrones actuales, distractores adversariales, contradicciones ambiguas, referencias temporales complejas, múltiples hechos interdependientes ni aprendizaje autónomo de relevancia.
- **No** son trayectorias verdaderamente independientes de un agente en un entorno físico: las respuestas se evalúan, pero no modifican dinámicas futuras. Dos decisiones por historia no validan todavía días de conversaciones naturales.
- **No** hay selección aprendida de experiencias E11 ni recuperación por iniciativa propia E12. E10-A usa consultas que solicitan el dato al final.

## Decisión de investigación y siguiente E10 antes de E11

**E10-A pasa el criterio local, pero NO da licencia para saltarse la validación decisiva de memoria práctica.** Conservar el núcleo actual, sin añadir redes neuronales, y preparar una **segunda confirmación genuinamente independiente en E10-B** usando (i) un LLM real común a todos los brazos, (ii) un baseline RAG semántico competente con el mismo tratamiento de actualización y (iii) expresiones y tareas de memoria menos alineadas con patrones hechos a mano, con los mismos presupuestos y controles de doble decisión.

Ese ensayo requerirá **nuevo protocolo, nuevos casos y partición virgen**, no afinar parámetros del lector sobre la confirmación \`1300000+\`, que queda **consumida** para fines confirmatorios. Los 900000+ históricos de F9 permanecen cerrados y no son parte de E10.

Si E10-B no muestra ventaja frente a ese baseline sólido, aplicar la condición de parada de la hoja de ruta: una única intervención diagnóstica y, si no hay diferencia, conservar una solución sencilla y seguir a E11 solo cuando esté justificado.
