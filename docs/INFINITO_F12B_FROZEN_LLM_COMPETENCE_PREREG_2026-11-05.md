# F12-B — Preflight de competencia de un LLM congelado con herramientas simuladas
Fecha de registro: 2026-11-05. **Prerregistro antes de implementar F12-B o consultar cualquiera de sus resultados**.

## Pregunta, alcance y relación con F12-A
El F12-A de PR #97 ejecutó auténtico Qwen2.5-0.5B-Instruct congelado, 41 respuestas, pero **0/2 tareas TRAIN y 0/6 ejecuciones heldout resueltas**. No había episodios exitosos con los que inferir estrategias; una síntesis no justificada no constituye aprendizaje. ¿Puede un LLM abierto mayor conseguir **competencia operativa mínima** eligiendo acciones de herramientas **simuladas**, antes de estudiar aprendizaje mediante INFINITO?

Este F12-B es un **preflight de modelo, NO** una validación de mejora por memoria, ni un nuevo experimento con estrategias: **sin memoria** para las acciones, sin síntesis y sin modificar pesos. Usará cuatro nuevas incidencias sintéticas no vistas en F12-A. No reutilizarán casos F12-A T0/T1/P0/P1 ni semillas reservadas de F11.

## Modelo y política de selección prospectiva
Primer modelo: **Qwen/Qwen2.5-1.5B-Instruct** (1,54B parámetros), local CPU GitHub Actions, `transformers==4.49.0`, `torch==2.5.1+cpu`, greedy `do_sample=False`, misma revisión de pesos registrada para todas las tareas, sin ajuste de prompts, ni aprendizaje por gradiente. Modelo aproximadamente 3x mayor que F12-A. Si el preflight no alcanza criterio, el resultado es **NO-GO** para usar este modelo en aprendizaje INFINITO. Un eventual estudio con 3B necesita una **nueva familia de casos**, protocolo y presupuesto; no seleccionar parámetros mirando estas mismas tareas.

Se utilizarán como referencia las seis herramientas ya diseñadas en F12-A (`inspect, audit_access, audit_dependency, repair_access, repair_dependency, verify`). Los modelos reciben sus **semánticas básicas públicas** (qué hace una auditoría, qué significa reparar y verificar) como contrato de uso de herramientas, pero **ninguna política experta**, secuencia de acciones correcta, causa oculta ni etiqueta de solución. Son descripciones de API, no demostraciones; no retocar el prompt tras ver salidas.

## Casos y método, fijados antes de inferencia
Cuatro incidencias nuevas identificadas K0..K3, exactamente dos con problema de autorización y dos de dependencia. Lenguaje de informes y síntomas distinto de F12-A; la herramienta `inspect` emite indicio público, la herramienta de auditoría correspondiente permite la reparación, y `verify` termina. Los cuatro casos se ejecutarán todos con el **mismo LLM congelado** desde cero, sin memoria entre ellos, sin feedback privado ni metadatos de respuesta del evaluador. **Máximo seis acciones por tarea**; salida JSON `{"action":"..."}`. Si es inválida, el simulador contabiliza acción inválida y devuelve error, nunca inventa acción correcta.

Resultados reproducibles con trazas de cada acción, respuesta del LLM, observación de herramienta, recompensa, tokens, duración, revisión/modelo y hashes de prompts. Los datos se almacenan como artefactos de Actions con retención limitada, **nunca** en la memoria personal, base MCP, secretos o producto de INFINITO.

## Gates definidos prospectivamente
- **G0 técnica:** ≥1 respuesta real por tarea; 100% de decisiones atribuidas al LLM concreto, ningún script heurístico que decida por el agente y ninguna alternativa falsa si falla modelo o descarga. Los comandos reales del runner se limitan a descargar modelo y ejecutar el arnés; las herramientas del agente no tienen acceso a SO/red.
- **G1 competencia:** completar por `verify` **≥2 de 4 incidencias**, incluyendo al menos una de cada tipo (autorización y dependencia), con máximo seis acciones cada una.
- **G2 interacción:** ≥90% de acciones emitidas con JSON válido y herramienta permitida; verificar que el agente reacciona a al menos un error/síntoma del simulador en vez de repetir infinitamente una acción fallida.
- **G3 auditoría:** 100% de informes de cada caso (incluidos fallos) con trazas originales, sin causas privadas en prompts, modelo SHA/revisión y presupuestos. No modificar la física o prompts después de ver resultados.

Si falla G1 o G2: **NO-GO** para hacer prueba de memoria con este modelo. Si pasa G0..G3: **GO solo para diseñar y preregistrar un nuevo estudio F12-C con memoria INFINITO**, no PASS de aprendizaje acumulativo. No utilizar este preflight para proclamar generalización ni para tunear el modelo o una memoria. No abrir F11 DEV/CONFIRM.

**Estado en este commit: diseño solamente, sin ejecución de Qwen1.5B.**
