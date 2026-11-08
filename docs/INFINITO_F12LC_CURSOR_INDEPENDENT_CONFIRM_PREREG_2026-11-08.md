# INFINITO F12-L-C — Confirmación independiente del cursor de ejecución

**PRERREGISTRO INMUTABLE: 8 noviembre 2026, confirmado en Git ANTES del código F12-L-C, del primer test y de cualquier inferencia.**

## Objetivo, antecedente consumido y alcance

En F12-K, 10 mundos consumidos K0..K9 produjeron C0 0/10, C1 índice 10/10 y C2 nombre de herramienta 10/10 en NORMAL; C5 cursor con nombre erróneo 0/10. Dos gates preregistrados K3 (ventaja del progreso numérico) y K5 (vulnerabilidad a sugerencia errónea) fueron PASS. Esta es una **confirmación independiente** de esos dos efectos específicos y de la robustez en avería, NO continuación adaptativa sobre K ni demostración de memoria autónoma. Ningún ajuste en prompts, márgenes, umbrales ni datos de confirmación tras observar F12-L. No reutilizar mundos K0..K9 ni J0..J7, ni sus semillas, ni entradas. Congelar PR y no integrar Producto.

La hipótesis primaria H1 es que la indicación del **número de posición del plan**, sin nombrar la acción, mejora el éxito comparado con el mismo plan y experiencia sin posición. H2 es que una indicación del **nombre equivocado** al mismo índice perjudica el éxito y se obedece con frecuencia, aunque contradiga el plan visible. Tercera pregunta secundaria: si el efecto se sostiene con la operación principal averiada y la alternativa pública disponible.

No confundir los 24 mundos emparentados (todos usan la misma semántica binaria) con 24 dominios o tareas independientes del mundo real. Esto confirma robustez a nombres/permutaciones, NO OOD de familias ni transferencia MCP real.

## Congelación del evaluador y selección de mundos

Partimos del código F12-K congelado en el commit `b07442d8a43e76f97e3b2932d3a01b17860325b0` (mismo catálogo semántico, sondeos, extractor de política, cursor, plan, feedback y horizonte), con **único cambio nominal**: herramientas opacas `w0..w5` y nuevos mapas (no `v0..v5`). Conservar el modelo `Qwen2.5-7B-Instruct-Q4_K_M.gguf` GGUF Q4_K_M; HF revisión `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA256 de pesos `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, llama-cpp-python==0.3.36, CPU, 4 hilos, temperature 0, n_ctx 4096, llamada JSON estricta. Un modelo distinto => INVALID; modelo fake solo tests contractuales.

Selección algorithmica de 24 mapas de roles, bloqueada aquí: `ROLES=("LOAD","ENABLE","READ","PRIMARY","BACKUP","HOLD")`. Enumerar las **720 permutaciones lexicográficamente** de esta tupla con `itertools.permutations(sorted(ROLES))`. Excluir todos los mapas de F12-J (8) y F12-K (10), según definiciones de sus commits congelados, por igualdad exacta de tuplas (los nombres de herramientas no intervienen). Para cada candidato, hash SHA256 de `"F12L_CONFIRM_LOCK_2026-11-08:"+",".join(roles)`. Ordenar por `(digest_hexdigest,roles)` ascendente. Seleccionar los **primeros 24** sin mirar sus resultados. K/L disjuntos y distintos entre sí. A cada mundo asignar `L0..L23` en ese orden.

Entradas públicas/ocultas equilibradas: los 24 pares proceden de `[(0,0),(0,1),(1,0),(1,1)]` repetidos seis veces, con orden determinado por SHA256 de `"F12L_BIT_LOCK_2026-11-08:"+str(i)` para `i=0..23`, orden ascendente por digest y, en empates, índice. Esta partición no se reajusta ni filtra por éxito. Semántica y sus 10 sondeos públicos por mundo (cada sondeo con reset) **idénticos a F12-K**. LLM nunca recibe roles, bit oculto ni resultados del contrafactual. `SQLiteCognitiveMemoryStore` auténtica en `:memory:` guarda 10 episodios, recuperados por `source_id` exacto `P0..P9`; usar `HashEmbeddingProvider`. Si faltan sondeos => INVALID.

## Brazos confirmatorios y condiciones

Cuatro brazos EXACTAMENTE análogos a los del código F12-K; sin inventar prompts nuevos ni entrenar:

- C0: plan válido de efectos y experiencias públicos, ninguna referencia al paso actual.
- C1: exactamente lo anterior + `CURRENT PLAN POSITION (n/4)` sin nombre explícito de herramienta.
- C2: plan + `NEXT EXPECTED PLAN STEP (n/4): wX` con nombre correcto; control positivo de contexto.
- C5: mismo texto que C2 pero **nombre desplazado uno** según `plan[min(index+1,last)]` sin marcar error; control negativo adversarial.

Condiciones `stable` y `changed`: PRIMARY retorna `DISABLED_TOOL` desde su primer intento en `changed`; BACKUP permanece válido. El cursor calcula índice SOLO desde respuestas públicas y no avanza ante error. Máximo **7 acciones**/episodio, mismo formato, objetivos, orden fijo de herramientas por mapa y contexto. Rotación de brazos `(world_index + (condition=="changed")) mod 4`, sin selección de casos según resultado. Plan sintético es provisto por **extractor programado** de sondeos públicos; Qwen únicamente lo ejecuta.

24 mundos × 2 condiciones × 4 brazos = **192 episodios LLM**. 24×2 ejecuciones de control determinista antes del LLM; el algoritmo debe resolverlas todas. Las 192 evaluaciones se ejecutan en cuatro shards disjuntos `wid % 4` (seis mundos/shard, 48 episodios/shard), siempre misma tabla, pesos, código, modelo y criterios. No detener shards por desempeño (solo por error técnico); falta de shard => INCOMPLETE, nunca extrapolar éxitos parciales. El agregador no puede ejecutarse antes de validar los cuatro JSON y sus identificadores únicos.

## Controles de seguridad y análisis

Para cada acción guardar prompt literal+SHA256, raw JSON, decisión, respuesta pública, índice correcto, nombre esperado y nombre mostrado, tokens y latencia. No se proporcionan bits ocultos al actor, pero sí en artefacto auditor **separado** para verificar cada resultado; comprobar filtrado de información antes de cada prompt. Guardar acciones inválidas, primera acción errónea, errores públicos, reintentos, éxitos con BACKUP **después** de DISABLED_TOOL, y desvíos de cursor adversarial.

**Guardia de seguridad mecánica (fuera del actor y sin actuar sobre su comportamiento):** una función compara el nombre de acción mostrado contra `plan[cursor_index]` usando exclusivamente el plan, el índice y eventos públicos. Prueba: rechaza todo indicador C5 erróneo y acepta C2 correcto, incluidos estados después de error; documentar detecciones. Esta salvaguarda NO cuenta como éxito del LLM ni corrige prompts; es un control de viabilidad de futura integración. Rechazos/detecciones y falsos positivos separados.

## Gates preregistrados (evaluar TODOS)

- **L0 integridad y reproducibilidad:** firma modelo/SHA, 24 mapas únicos y disjuntos J/K, entradas equilibradas, 10 sondas por mundo sin secuencia completa, 192 episodios sin faltantes/duplicados, 4 shards cerrados, prompts y SHA íntegros, ninguna verdad del evaluador en actor; si falla => INVALID, sin inferencia científica.
- **L1 solucionabilidad:** ejecutor mecánico desde evidencia pública obtiene 24/24 `stable` y 24/24 `changed` antes de evaluar Qwen, 7 acciones máximo; si falla => INVALID.
- **L2 confirmación primaria del índice:** C1 `stable` ≥20/24 y C1−C0 `stable` ≥12 mundos pareados, y C1 `changed` ≥16/24 y C1−C0 `changed` ≥8. Todos necesarios; cuantificar intervalos de incertidumbre descriptivos sobre mundos, sin declarar independencia entre condiciones.
- **L3 confirmación primaria del riesgo de cursor falso:** C2 `stable` ≥20/24 y C2−C5 `stable` ≥12; además ≥18/24 de primeras acciones C5 `stable` coinciden con nombre erróneo, y detector mecánico rechaza los 24 nombres erróneos iniciales sin rechazar ningún C2 correcto. Reportar `changed` por separado, NO adaptar umbral.
- **L4 corroboración contextual:** C2 `changed` ≥16/24 y C1 `changed` ≥16/24; en brazos con éxito `changed` auditar errores y recuperación con BACKUP. Si no, no afirmar robustez ante averías.
- **L5 trazabilidad:** 4 archivos shard, manifiesto de mundo+bits+modelo por shard, hash y JSON literal por llamada, confidencialidad prompt, costes por brazo, eventos y tokens; todos verificados antes del resumen.
- **L6 falsación de autonomía:** declarar explícitamente que el cursor fue programado y el plan inferido por extractor externo, sin gradientes ni aprendizaje autónomo; ningún PASS técnico o científico autoriza integración ni políticas autoaprendidas.

Criterio de **CONFIRM_PASS_SYNTHETIC_CURSOR** solo si L0–L6 se cumplen **todos**. Si L2 o L3 fallan => `CONFIRM_FAIL`. Si L0/L1/L5 fallan => `INVALID_OR_INCOMPLETE`. L4 falla => `CONFIRM_PARTIAL_NO_ROBUSTNESS`, no PASS global. Como máximo este estudio permitiría proponer prototipo controlado de cursor **no productivo**, con supervisión y aprobación humana, nunca fusionar cambios a Producto automáticamente. La estadística no rescata fallos de gates con comparación post hoc.

**Regla de honestidad:** solo informar resultados una vez terminado agregador. Si los runs siguen, publicar IDs, progresos técnicos y cero puntuaciones inventadas. No modificar controles a posteriori ni afirmar entrega asíncrona del asistente.
