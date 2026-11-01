# INFINITO E12-A — Resultado final: agente decide cuándo recuperar recuerdos

**1 de noviembre de 2026 · Ejecución completada técnicamente · Hipótesis preregistrada NO SUPERADA.**

## Procedencia científica

- [Protocolo congelado antes de implementar el experimento](./INFINITO_E12A_AUTONOMOUS_SEARCH_PROTOCOL.md) — commit \`7886b0b51eb38be600c41e58a91cc686afb4f32e\`.
- [GitHub Actions **#41324785450**: preflight, DEV real y confirmación prospectiva **SUCCESS**](https://github.com/webtilians/principiodelTodo/actions/runs/41324785450); SHA de ejecución \`136785312c88757e1e2901532938a83040af6a9e\`.
- [JSON de resumen y criterio registrado](./INFINITO_E12A_REAL_SEARCH_RESULTS_2026-11-01.json).
- [Artifact ORIGINAL confirmatorio: 32 mundos, cuatro brazos, respuestas LLM, contextos y costes](https://github.com/webtilians/principiodelTodo/actions/runs/41324785450/artifacts/13090368563). El artifact tiene retención limitada de GitHub Actions.
- [Artifact DEV original: ocho mundos](https://github.com/webtilians/principiodelTodo/actions/runs/41324785450/artifacts/13091220002).
- [PR #34, borrador y código de producción INTACTO](https://github.com/webtilians/principiodelTodo/pull/34).

## Pregunta

El LLM, viendo **solo** una pregunta y su memoria de trabajo, debe elegir **responder**, **ABSTENER** o **SEARCH**. El archivo externo (15-16 documentos por mundo) permanece inaccesible hasta ejecutar SEARCH. Si busca, el recuperador usa \`text-embedding-3-small\` real, búsqueda híbrida coseno+léxico, top-4, contexto cronológico de máximo 260 tokens estimados; posteriormente el mismo LLM \`gpt-5.6-luna\` genera la respuesta. Cada búsqueda del router exige una SEGUNDA llamada al LLM, que se contabiliza.

Cuatro brazos: \`never_search\`, \`always_search\`, \`fixed_heuristic\` (consulta si falta el dato visible o está marcado como \`no_verificado\`) y \`llm_router\` (elección explícita mediante un primer turno del mismo LLM). Ninguna política accede al archivo para decidir cuándo buscar. Los mundos incluyen los casos \`fresh\`, \`archived\`, \`stale\`, \`unknown\`, en igual proporción. N=32 mundos nuevos (8 por clase), con DEV independiente n=8, sin selección de semillas según resultados.

## Resultado: 32 mundos CONFIRMATORIOS

| Estrategia | Aciertos | Precisión | Búsquedas | Llamadas LLM | Tokens de entrada del LLM | Utilidad (acierto − 0,08×búsqueda) |
|---|---:|---:|---:|---:|---:|---:|
| Nunca buscar | 16/32 | 50,00% | 0 | 32 | 5.849 | 0,50000 |
| Buscar siempre | 32/32 | 100,00% | 32 | 32 | 10.159 | 0,92000 |
| **Regla fija fuerte** | **32/32** | **100,00%** | **24** | **32** | **9.086** | **0,94000** |
| **Agente LLM (enruta)** | **31/32** | **96,875%** | **23** | **55** | **13.752** | **0,91125** |

El agente evitó **9 de 32 búsquedas**, un ahorro de **28,125%** frente a buscarlas siempre, sin perder más de 5 puntos de precisión (**−3,125 pp** frente a always-search). No obstante, perdió **1 respuesta** y necesitó **23 llamadas adicionales al LLM**, con **35,4% más tokens de entrada** respecto a always-search (13.752 frente a 10.159) y **51,4% más** frente a la regla fija (9.086). Por tanto **no se ha demostrado ahorro total de cómputo o coste monetario**. El ahorro de búsquedas externas sí existe, pero en esta configuración la política fija lo consigue de forma aún más eficiente.

**Valor de la utilidad:** la regla fija obtiene **0,94**, el router **0,91125**; la diferencia router−regla fija es **−0,02875**, cuando el criterio preregistrado exigía **+0,05**. **La hipótesis E12-A falla**, incluso aunque se hayan alcanzado las restricciones de precisión y reducción de búsquedas.

## Desglose por tipo de información: ocho casos de cada tipo

| Clase | Nunca buscar | Buscar siempre | Regla fija | Router del LLM |
|---|---:|---:|---:|---:|
| Hecho fresco/verificado en trabajo | **8/8** | **8/8** | **8/8** | **8/8**, 0 búsquedas |
| Solo en el archivo semántico | 0/8 | **8/8** | **8/8** | **7/8**, 7 búsquedas |
| Hecho de trabajo antiguo/no verificado, versión nueva archivada | 0/8 | **8/8** | **8/8** | **8/8**, 8 búsquedas |
| No existe información verdadera para responder | **8/8** | **8/8** | **8/8** | **8/8**, 8 búsquedas |

**Control de obsolescencia:** en los ocho mundos \`stale\`, el router sí buscó y no dio por vigente el valor antiguo. **Control de invención:** en los ocho mundos \`unknown\`, el router buscó y respondió \`ABSTENER\`; antes de consultar no tenía acceso a una señal que distinguiera \`unknown\` de \`archived\`.

### Fallo individual verificado en artifact (descriptivo, NO ajuste)

- Semilla: **5201007**, clase \`archived\`.
- Consulta sobre \`Faro-435-Q7\`, con opciones \`malva\` / \`ocre\`; respuesta correcta \`SOL\` correspondiente a \`ocre\`.
- Memoria de trabajo solo decía que se revisaron materiales para un asunto diferente; **no proporcionaba el hecho consultado**.
- El router del LLM decidió **\`ABSTENER\` SIN buscar**, aunque el protocolo de instrucción le exigía \`SEARCH\` ante información ausente.
- La regla fija consultó el archivo y recuperó literalmente el registro vigente de \`Faro-435-Q7\`, respondió **\`SOL\`** con una llamada al LLM.

Esto identifica un error del propio controlador de herramientas: confundió «ahora no tengo evidencia» con «no hay evidencia accesible en el archivo». La respuesta final no fue una alucinación, pero sí una **abstención indebida**.

## Criterio preregistrado, evaluado sin correcciones

- No perder más de 5 pp de acierto vs siempre buscar: **PASA** (−3,125 pp).
- Reducir búsquedas al menos un 20% vs always-search: **PASA** (28,125%).
- Superar regla fija fuerte por ≥5 pp de utilidad: **FALLA** (−2,875 pp).
- No más de un error stale o invención unknown: **PASA** (0 y 0).
- **GATE GLOBAL: NO PASA.**

No declarar avance por cumplir solo parte de las condiciones. La semilla \`5200000+\` confirmatoria está **consumida**. No debe utilizarse para corregir un prompt y luego presentarse una ejecución sobre la misma partición como confirmación independiente.

## Interpretación y límites

**Sí:** un LLM real puede utilizar señales disponibles para dirigir el acceso a memoria persistente en la mayor parte de estos episodios, evitando búsquedas superfluas y previniendo confusiones entre dato antiguo y vigente.

**No:** no demuestra un controlador mejor que la regla fija. Tampoco demuestra *consciencia de la incertidumbre*, aprendizaje de estrategias, introspección fiable ni autonomía de objetivos. En este benchmark, la respuesta de buscar se desprende de marcadores explícitos de disponibilidad/verificación; la heurística los explota plenamente.

**Falta una prueba sobre ambigüedad genuina**: cuando no hay etiquetas explícitas, la memoria tiene contradicciones parcialmente inferibles o las búsquedas cuestan lo suficiente para justificar el router a pesar de una llamada adicional, la pregunta vuelve a ser relevante, pero ese sería un protocolo **nuevo** y no una reinterpretación de E12-A.

**Limitaciones de medición:** muestra pequeña balanceada artificialmente (32 mundos), modelo/plantillas específicos; los tokens de entrada API varían por prompt pero no se han traducido a coste real de tokens/servicio externo; embedding cache compartida contabilizada pero no representa el costo incremental de un repositorio real de gran escala. La utilidad usa una penalización fijada en términos abstractos de tasa de acierto, no euros.

## Siguiente decisión

**Cerrar E12-A sin añadir un parche de prompt ni retocar esos 32 casos.** Conservar el RAG que busca de forma controlada por regla determinista cuando la ausencia/no vigencia es explícita. El salto necesario para INFINITO sería estudiar un problema verdaderamente nuevo: identificación de lagunas o conflictos **sin etiquetas fiables y sin lectura anticipada del archivo**, con un control fuerte y presupuesto de llamadas igualado. No iniciar ese experimento por inercia ni proclamar inteligencia autónoma; diseñar preregistro separado (E12-B) solo si resuelve un fallo de uso real que la heurística no puede manejar.

**Producción \`src/infinito3/\` permanece sin cambios; PR en borrador.**
