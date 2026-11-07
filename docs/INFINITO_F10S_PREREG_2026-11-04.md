# INFINITO F10-S — Consistencia temporal entre procesos de memoria MCP

**PRERREGISTRO PROSPECTIVO 2026-11-04. Este archivo se publica ANTES de cualquier código de ejecución o prueba F10-S.** Código real fijado a master commit `2e563ddb194cb3a0193b77a22cc97575a8baba73`; repositorio `webtilians/principiodelTodo`. F10-R quedó INCONCLUSIVE por semillas solapadas, por eso este ensayo usa espacios de semilla disjuntos y auditoría separada. No se modificará la memoria del usuario.

## Hipótesis y alcance

Dos procesos **independientes** que exponen el **servidor MCP auténtico** `infinito3.mcp_server` sobre **exactamente el mismo archivo SQLite sintético** deben ofrecer un estado coherente en nuevas llamadas: visibilidad de escrituras, sustitución de hechos exclusivos (color favorito), borrado selectivo de memorias y su historia, persistencia tras reiniciar un tercer proceso y ausencia de resurrección. Un agente que conserve literalmente una respuesta previa en su contexto **seguirá conservándola**; no atribuir al servidor la eliminación retroactiva de respuestas ya vistas. La propiedad de seguridad exigida es que nuevas consultas MCP después de delete/update devuelvan el estado actual.

**Advertencia:** prueba de coherencia de servidor, NO de comprensión autónoma por un modelo lingüístico; los agentes son clientes MCP y la elección de la herramienta es explícita. La producción remota HTTPS es otra topología y queda fuera del estudio. No inferir seguridad multiusuario ni borrado de caches de LLM.

## Diseño fijo

4 réplicas r=0..3 con SQLite temporal independiente por réplica. Cada réplica abre dos clientes MCP stdio (procesos A y B simultáneos), inicializa y verifica que ambos `tools/list` incluyan `remember`, `recall`, `list_memories`, `forget`. Todos los clientes fijan `INFINITO_MEMORY_DB` a la ruta SQLite efímera del run. Ninguna llamada a MCP personal, red remota, API de producción o bóveda.

Seis familias f=0..5, cuatro casos j=0..3 por réplica, **24 casos/r, 96 en total**, semillas `32_100_000 + 10_000*r + 100*f + j` estrictamente disjuntas entre réplicas y familias. Los identificadores de hechos se crean a partir de la semilla y se usan para aislar cada operación. Confirmación independiente `32_300_000+` **SELLADA**. No reintentos ni modificación de casos tras ver resultados.

0. **Visibilidad**: A guarda una decisión sintética única; B la recupera por recall y la encuentra en list_memories. Debe aparecer una sola vez y con el texto idéntico.
1. **Actualización exclusiva**: A guarda `Mi color favorito es cobalto<seed>.`, B lo ve; A guarda `Mi color favorito es esmeralda<seed>.`, B consulta de nuevo y solo debe ver la versión nueva en recall y list_memories. Auditor SQLite cuenta 1 registro ACTIVE de ese predicado, anterior SUPSERSEDED o eliminado. Después se puede limpiar por su ID para no interferir con la siguiente muestra; la limpieza se registra y NO interviene en los resultados.
2. **Borrado selectivo**: A guarda dos enunciados sintéticos relacionados (objetivo y distractor); B consulta objetivo; A usa ID obtenido de list_memories para forget(target); B consulta otra vez. Objetivo ausente en nuevas respuestas y filas SQLite incluso inactivas, distractor presente; ninguna eliminación ajena.
3. **Reinicio**: A guarda un enunciado, B lo consulta, se inicia un tercer proceso C y debe recuperarlo; A lo borra, C se cierra, se inicia un proceso D (reinicio), que no debe recuperar el enunciado; el distractor, si lo hay, sigue presente.
4. **Concurrencia**: A y B envían a la vez dos recuerdos distintos mediante asyncio.gather; ambos deben quedar visibles sin pérdidas. A y B actualizan simultáneamente el mismo atributo exclusivo `mi color favorito`, con dos valores distintos; después de las confirmaciones debe quedar exactamente UNA versión ACTIVA, sin dos valores concurrentes activos. No se exige cuál gana (intercalación no determinista); se registran errores de bloqueo si aparecen. Limpieza posterior del atributo exclusiva no cuenta como éxito.
5. **Contexto obsoleto, revalidación**: B recupera un dato antes del borrado por A y guarda una copia literal de esa respuesta en la memoria del cliente evaluador (control de caché obsoleta). Después, el texto antiguo aún aparece en la copia local —ESTO ES ESPERADO Y NO es fallo de MCP— pero B, al volver a llamar recall/list_memories, debe dejar de obtenerlo y un proceso reiniciado tampoco debe verlo.

## Datos, observabilidad y auditoría

Registrar por caso el texto de las respuestas MCP, IDs seleccionados, acciones, semillas, eventos de error, conteo de filas SQLite `memories` y `forget_log` tras la operación; no registrar datos personales. No hacer cambios a código del producto. Auditor independiente lee los artefactos de los jobs, SIN importar MCP, el learner ni sqlite3, verifica semillas únicas y disjuntas, SHA de reports, conteos por familia, resultado de cada gate y errores con pasos. No reescribir resultados después. El orden de las familias y variantes es determinista; seed no se pasa al servidor salvo como sufijo del texto sintético.

## Puertas científicas fijadas ANTES

- **G0 integridad:** 4/4 réplicas con procesos reales A/B, DB compartida solo dentro de réplica y aislada de otras, 24 casos por réplica, 6×4, semillas sin colisiones, SHA y sin acceso a datos reales.
- **G1 visibilidad:** familia 0, 4/4 casos por réplica PASS; >=3/4 réplicas.
- **G2 actualización:** familia 1, 4/4 casos por réplica PASS (ninguna versión antigua activa/visible); >=3/4 réplicas.
- **G3 borrado selectivo:** familia 2, 4/4 PASS (sin objetivos visibles ni restos en filas SQLite y sin daño al distractor); >=3/4.
- **G4 reinicio y revalidación:** familias 3 y 5, 8/8 PASS, y control de caché local obsoleta explícito; >=3/4.
- **G5 concurrencia:** familia 4, 4/4 PASS por réplica (sin errores MCP/SQLite y única versión activa de atributo); >=3/4.
- **SEGURIDAD (hard gate):** no escritura/lectura fuera de TEMP DB, no datos personales, borrar target sin pérdida de distractores; 4/4 réplicas. Cualquier violación real implica FAIL de seguridad.

**F10S_COHERENCE_PASS** solo si integridad y seguridad 4/4 y G1..G5 >=3/4 cada uno. Si ejecución con evidencia íntegra incumple un gate: **F10S_COHERENCE_FAIL**, conservando cada fallo como diagnóstico. Error de entorno/artefactos/código de prueba: **F10S_INCONCLUSIVE**, sin alterar protocolos. No utilizar 100% en un simulador como garantía para agentes de lenguaje natural, no modificar core automáticamente.

## Decisión posterior

Un FAIL reproducible desencadena experimento F10-S-FIX con hipótesis causal e implementación aislada en rama distinta, más confirmación sellada; NO ajustar el caso F10-S original para que pase. Un PASS limitado indica servidor coherente en estas condiciones, pero no que un agente LLM sepa revalidar su contexto y olvido; próximo estudio debería medir respuestas de modelos con/ sin refresh sobre memoria compartida. Mantener master intacto y PR DRAFT.