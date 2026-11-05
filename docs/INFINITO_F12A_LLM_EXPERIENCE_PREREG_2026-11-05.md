# INFINITO F12-A — Aprendizaje de estrategias desde experiencias con LLM real congelado
**Prerregistro prospectivo — 2026-11-05. Solo diseño al confirmarse este archivo; ningún dato F12-A observado.**

## Pregunta
¿Puede un agente basado en un LLM fijo y la memoria SQLite real de INFINITO resolver nuevas tareas con herramientas simuladas mejor cuando dispone de estrategias que él mismo sintetizó desde experiencias observadas, frente a disponer solo de episodios completos, con el mismo LLM y límite de contexto?

El estudio es **distinto de F11-A/B/C**, que usaron controladores tabulares y mostraron ventajas inexistentes ante controles competentes. No abre sus semillas DEV/CONFIRM ni reutiliza sus datos. No se entrenan pesos ni se ejecutan herramientas reales.

## Proveedor y congelación
Usar el endpoint real GitHub Models (`https://models.github.ai/inference/chat/completions`) con GITHUB_TOKEN y permiso `models: read`; modelo por defecto `openai/gpt-4o-mini` en todas las variantes, temperatura 0, respuesta compacta, mismo system prompt, catálogo y número máximo de 5 decisiones por tarea. Registrar modelo declarado, modelo real reportado, uso/tokens si existe, latencia, respuesta en crudo y errores. Sin éxito de API verificable, **BLOCKED**, nunca sustituir por mock haciéndolo pasar por LLM real. Se permite usar mock SOLO en tests sin resultados científicos.

## Experiencias, aislamiento y condiciones
- Incidencias sintéticas de operaciones informáticas con 6 herramientas simuladas: `inspect`, `audit_access`, `audit_dependency`, `repair_access`, `repair_dependency`, `verify`. Todas las llamadas alteran SOLO estado del simulador en memoria; ninguna shell, red de proyecto, base de datos o cuentas externas.
- El objetivo de cada tarea es recuperar un servicio y verificarlo. Los casos tienen fallos de acceso o dependencia, costes de pasos y señales públicas de herramientas. El actor no ve la causa oculta ni el evaluador, solo el objetivo, nombres de herramientas y feedback.
- Ejecución TRAIN: agentes con LLM sin memoria completan tareas sintéticas, almacenando exactamente `(objetivo, herramientas, observaciones, acciones, recompensas observadas, resultado)` como recuerdos EPISODIC mediante `SQLiteCognitiveMemoryStore` de INFINITO y `HashEmbeddingProvider` local. Las recompensas proceden del simulador, no de anotación experta.
- Tras TRAIN, una llamada al **mismo LLM congelado** lee episodios y redacta reglas **hipotéticas y revisables**, citando hechos observables; prohibido suministrar reglas correctas del simulador, feedback privado o demostraciones expertas. Guardarlas como recuerdos SEMANTIC de INFINITO y conservar literalmente los episodios fuente.
- Tres variantes de evaluación en tareas distintas de TRAIN: **A0** LLM sin memoria, **A1** LLM con memoria EPISODIC recuperada de INFINITO, **A2** LLM con el mismo presupuesto contextual pero mostrando los resúmenes de estrategias inferidas desde la experiencia (y episodios si caben). No convertir retrospectivamente la síntesis en políticas correctas.
- Mismas tareas, catálogo de herramientas, modelo, temperatura, máximo de acciones, seed público y límite de 1200 caracteres de referencia por variante. Guardar tamaños de recuerdos, prompt tokens, llamadas y tiempo.
- No elegir estrategias, cambiar prompts o configurar tareas tras observar resultados de evaluación. El texto recuperado es **dato no confiable**, no instrucciones privilegiadas.

## Particiones y ejecución
- Un único **smoke/piloto end-to-end** con dos tareas de experiencia y dos tareas holdout diferentes (identificadores internos reservados). El pilotaje se usa para verificar que el LLM participa realmente y que los recuerdos influyen en los prompts; NO sirve para pasar un gate científico.
- No se permite comparar métricas si hay fallos 401/403/429, modelo inestable, cambios de temperatura, cotas diferentes o ejecución parcial: **BLOCKED/INCONCLUSIVE** según causa.
- Futuro DEV y CONFIRM requerirán un protocolo posterior con tareas de familias diferentes, al menos cuatro réplicas, intervalos pareados agrupados por tarea y variantes de memoria con coste equivalente. Las semillas de esos conjuntos permanecen sin abrir en este piloto.

## Criterios del piloto definidos antes de datos
- **P0:** 100% de decisiones de acción de las tres variantes provienen de respuestas no vacías de un LLM remoto real; cero fallback heurístico oculto.
- **P1:** 2 episodios TRAIN almacenados en INFINITO SQLite; al menos un recuerdo sintetizado si las observaciones tienen material causal; integridad origen/citas; cero entradas privadas.
- **P2:** todos los episodios de evaluación y sus errores quedan en logs, incluyendo éxito objetivo y costes. No cambio de modelos, herramientas, datos de train o prompts entre los brazos.
- **NO PASA evidencia científica** por ganar alguna condición en dos casos. Si A0/A1/A2 empatan al techo, el entorno se considera demasiado fácil y exige diseño externo más exigente antes de una evaluación independiente. Si síntesis falla o contradice experiencias, conservar sin corregir a mano.

## Límites
En este estudio 'experiencias reales' significa **experiencias realmente ejecutadas por un LLM** con herramientas **simuladas**, no interacciones con cuentas, datos de clientes o sistemas reales. Un LLM congelado significa sin fine-tuning: un identificador API estable no garantiza pesos internos publicados ni idéntico muestreo oculto. No incorporar estrategias al servidor MCP ni al producto antes de pruebas independientes. GitHub Actions puede tener restricciones de modelo/cupo.

**Antes de los resultados, estado: SOLO PRERREGISTRO.**
