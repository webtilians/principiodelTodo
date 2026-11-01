# INFINITO E10-C — Auditoría única de la tubería de memoria (DIAGNÓSTICO, NO CONFIRMACIÓN)

**1 de noviembre de 2026.** Protocolo registrado ANTES de abrir nuevos fixtures.

## Motivación y alcance
E10-B, confirmado con LLM real, resultó **negativo**: INFINITO 3.0 2/24 pares completos frente a RAG semántico cronológico 24/24; la ablación de gate abierto consiguió 14/24. La inspección POST-HOC de los 96 turnos confirmatorios reveló evidencias ausentes y abstenciones pese a evidencias presentes. **No se reabrirán ni se usarán los casos 1800000+ para ajustar el núcleo.**

Este único diagnóstico identifica el primer punto observable en que una declaración vigente deja de estar disponible, siguiendo exactamente la tubería ACTUAL: **captura mediante gate → fila activa SQLite → resultado \`memory_store.search\` top-8 → \`ContextPacket.items\` → mensaje de referencia que realmente recibiría el LLM.** También examina las razones del gate, estado de supersesión y familia. No ejecuta LLM, no modifica producción y **no demuestra** que arreglar una etapa cambie decisiones reales. El desempeño del lector LLM ya fue medido en E10-B.

## Fixture nuevo, no conjunto confirmatorio
- **12 mundos nuevos de DESARROLLO**, \`1900000+i*1000+j\`, i=0..1, j=0..5; seis familias E10-B (avisos, formato, escalado, cadencia, color, ciudad), con sujetos, hechos y distractores diferentes a 1700000+/1800000+.
- Dos pistas contrafactuales, dos fases (antes/después de cambio): **48 decisiones diagnósticas por brazo**.
- Mismas 12+12 observaciones distractoras, actualización contradictoria, reinicio de historial y cierre/reapertura SQLite, \`top_k=8\`, presupuesto **320 tokens estimados** de evidencia.
- Tres brazos \`semantic_rag\` (referencia positiva de búsqueda), \`infinito3\` (original congelado) y \`infinito3_gate_open\` (ablación ya fijada). Mismo \`text-embedding-3-small\` mediante API real para los tres; cache de vectores. No simular semántica en el resultado principal.
- **No LLM real:** en esta auditoría la generación se sustituye con un RecordingLLMAdapter que solo permite inspeccionar el \`LLMRequest\` generado; sus respuestas **no se miden como decisiones**. Tests usan hash fake y se etiquetan instrumentales, nunca resultados semánticos.
- \`OPENAI_API_KEY\` se comprueba en Actions. Si falta, la auditoría real queda **bloqueada**; no se publica un resultado hash como semántico.

## Trazas congeladas, por brazo / mundo / pista / momento
1. **Gate en ingestión:** \`should_store\`, \`kind\`, \`reasons\`, \`importance\` de la declaración vigente; no se presupone que evaluar el gate equivalga a éxito en SQLite.
2. **DB:** declaración idéntica en registros \`all(include_inactive=True)\`; número de filas activas, \`status\` activo/superseded, presencia en conjunto activo. Para RAG sin SQL, \`all()\`.
3. **Retrieve:** \`CognitiveDecision.context\` es la lista \`search(query,top_k=8)\` que el núcleo consumió en el turno (NO ejecutar una búsqueda adicional que altere el estado).
4. **Select:** \`ContextPacket.items\` y \`memory_id\` del recuerdo identificado; si está recuperado y no seleccionado, registrar \`ContextPacket.diagnostics\` (filtros, top single, reglas).
5. **Transmit:** comprobación del texto exacto de la declaración vigente en los mensajes de referencia de \`LLMRequest\`, excluyendo la pregunta actual que contiene alternativas y no evidencia pasada.

Métrica descriptiva: cuántas de las 48 decisiones por brazo tienen la evidencia vigente en cada etapa; identificar las primeras pérdidas exclusivas (no capturada, supersedida/inactiva, no recuperada, recuperada-no-seleccionada, seleccionada-no-transmitida), con desagregación por las seis familias y por momento, además de control de secuenciación de datos. Cada pérdida se clasifica por evidencia **real**, no por coincidencia casual del valor en la pregunta.

## Reglas y cierre
No escribir \`src/infinito3/\`, no entrenar ni hacer búsqueda de hiperparámetros, no generar nuevas confirmaciones E10-D/F9, no inferir capacidad general. Registrar resultados brutos, manifest SHA, informe de causas con al menos una hipótesis falsable y **una sola** posible mejora acotada si emerge. La corrección, si se propone, va en **otra rama** y necesita nuevo protocolo y nuevos mundos independientes; \`1800000+\` sigue consumido. Si la traza no permite asignación causal, decirlo y conservar RAG como baseline.

No se alterarán métricas de E10-B ni se llamará «nueva confirmación» a este diagnóstico.
