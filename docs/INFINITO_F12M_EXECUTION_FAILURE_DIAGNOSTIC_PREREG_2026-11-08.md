# F12-M — Diagnóstico causal de selección, terminación y recuperación (INFINITO)

**Prerregistro prospectivo: 2026-11-08; este commit se confirma antes de implementar F12-M, ejecutar tests F12-M o consultar resultados de mundos nuevos.** F12-J, F12-K y F12-L-C están consumidos; sus resultados solo motivan hipótesis. No reutilizar sus mapas, nombres de herramientas ni entradas por mundo. El estudio NO demuestra aprendizaje autónomo ni autoriza integración en Producto.

## Hipótesis y separaciones causales

En F12-L-C se observó exploratoriamente: C2 (índice+herramienta) acierta primeras acciones mejor que C1 (solo índice); algunas trayectorias alcanzan el valor objetivo pero no ejecutan `verify`; C1 con `changed` a menudo utiliza BACKUP antes del primer error, mientras C2 sufre error público y a veces se recupera. No confirmar retroactivamente esos análisis.

F12-M separa experimentalmente:
- **H_S selección:** identificar nominalmente la siguiente herramienta en el cursor mejora la primera decisión y la finalización frente a dar solo el número de paso.
- **H_T terminación:** un **supervisor mecánico público** que ejecuta `verify` cuando el cursor alcanza el último paso puede rescatar fallos por falta de terminación; NO se atribuye la acción a Qwen ni se consulta el bit objetivo.
- **H_R recuperación:** después de una avería **forzada antes de ceder el control al LLM**, mostrar el nombre de BACKUP como siguiente acción mejora la recuperación frente a mostrar el nombre de PRIMARY inoperativa o solo el índice. Éste es un **rescate externo dirigido**, no revisión autónoma.

## Modelo, memoria, motor y mapas congelados

Mismo motor de efectos públicos y horizonte=7 de F12-L-C sin tocar semántica, plan programado desde 10 sondeos, estado por episodio independiente, misma capa real de `SQLiteCognitiveMemoryStore` en RAM, misma búsqueda por `source_id` exacto; mismo extractor `[ENABLE, READ, PRIMARY, verify]` y alternativa pública BACKUP. Conservamos `Qwen2.5-7B-Instruct` GGUF `Q4_K_M` con SHA256 de pesos `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, revisión HF `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, `llama-cpp-python==0.3.36`, 4 hilos CPU, temp 0, contexto 4096, JSON estricto sin reparador. Todo resultado con backend falso es TEST_ONLY.

**Nuevos nombres opacos `z0..z5`; mapas F12-M (N0..N23):** enumerar permutaciones lexicográficas de `sorted(("LOAD","ENABLE","READ","PRIMARY","BACKUP","HOLD"))`. Excluir todas las tuplas congeladas de F12-J (8), F12-K (10) y F12-L-C (24). Ordenar el resto por `(sha256("F12M_DIAGNOSTIC_LOCK_2026-11-08:"+",".join(roles)),roles)`, tomar primeros 24. Nunca filtrar por resultados. Balance de inputs: `[(0,0),(0,1),(1,0),(1,1)]*6`, reordenar índices por SHA256 de `"F12M_BITS_2026-11-08:"+str(i)` ascendente. Conservar los 10 sondeos públicos sin `verify`; preflight externo debe resolver los 24 mundos con y sin fallo.

## Brazos, intervención y presupuestos

**Episodios compartidos de selección y control de terminación**: 24 mundos × 2 condiciones × 3 brazos = **144 episodios**, duración máxima 7 acciones:
- `S0`: exactamente la mecánica de C1 F12-L-C, muestra solo `CURRENT PLAN POSITION (i/4)`, misma política válida y memoria.
- `S1`: exactamente la mecánica de C2 F12-L-C, muestra `NEXT EXPECTED PLAN STEP (i/4): zX`. Es el control *sin supervisión* del ensayo de terminación.
- `T1`: misma instrucción y prompts que S1 hasta alcanzar el último índice del cursor. Si `expected_index(public_history)==3`, un supervisor externo llama `verify` **en lugar de llamar a Qwen**; registra `origin="supervisor"` y los 0 tokens y 0 segundos asociados. Si `verify` devuelve falso, la ejecución prosigue hasta agotar 7 acciones; no mostrar verdad oculta. No invocar supervisor antes de llegar al índice 3 ni forzar acciones previas. La comparación primaria es S1 frente a T1 pareada por mundo+condición, **no** atribuir a Qwen el éxito supervisado. El contexto y las llamadas al LLM previas a la intervención deben ser idénticos salvo posibles variaciones numéricas del backend, que se auditan.

**Recuperación condicionada a un mismo fallo público**: 24 mundos × 3 brazos = **72 episodios** adicionales SOLO en `changed`:
- Ejecutar **prefijo controlado y registrado** `ENABLE, READ, PRIMARY`; la última respuesta es `{"error":"DISABLED_TOOL"}`. Este prefijo NO cuenta como decisiones del LLM y **no puede ocultarse** en el historial del prompt; los tres reciben los mismos datos públicos y el mismo estado real. Desde ahí quedan como máximo **4 decisiones de Qwen** (no hasta 7 adicionales); esas llamadas consumen el presupuesto restante del episodio, horizonte total siete.
- `R0`: cursor con índice y nombre PRIMARY averiada, exactamente como S1.
- `R1`: cursor solo con índice, exactamente como S0; alternativa BACKUP presente en contexto.
- `R2`: mismo cursor+índice que R0, pero el nombre visible para el paso de inversión es BACKUP, solo **después de la respuesta pública de error**. No cambiar el plan ni historial. Esto mide un controlador externo que propone recuperación, NO una reparación autónoma aprendida. El texto adicional debe permanecer idéntico entre R0/R2, únicamente cambia el identificador, de longitud 2.

**Total 216 episodios Qwen**, 4 shards de 6 mundos (54 episodios por shard); orden de brazos determinista rotado por mundo/condición/estudio. No publicar decisión científica a partir de un shard, exigir los cuatro. El número de `calls` de Qwen será menor que el de pasos por supervisor y prefijos controlados; contabilizar por separado. No cambiar los nombres/orden tras ver resultados.

## Auditorías obligatorias

Registrar prompts completos con SHA, salida JSON cruda, modelo y pesos, índice correcto calculado solo de eventos públicos, próximo nombre mostrado, fuente de acción (`llm`/`supervisor`/`controlled_prefix`), eventos, error público, reintentos, `verify`, tokens/latencias reales y estado de éxito del simulador solo en artefacto auditor. Guardar si antes de `verify` hay cursor en índice 3, si `verify` devuelve `false`, primeros errores y utilización de BACKUP **tras** un `DISABLED_TOOL` frente a elección anticipada sin error. Clasificar fallos: `wrong_first_action`, `never_reached_cursor_3`, `reached_cursor_3_no_verify`, `verify_false`, `repeated_disabled`, `other` (categorías solapables, no inferir causa causal solo del recuento). Un guard que solo comprueba plan e índice público puede validar sugerencias de acción, pero NO actuar como oráculo del objetivo. Publicar auditoría de 24 mapas únicos, entradas balanceadas y todos los 216 identificadores estudio/mundo/condición/brazo.

## Gates congelados

- **M0 validez:** modelo real y SHA/versiones congelados en los 4 shards, exactamente 24 mapas y entradas bloqueados, 216 episodios sin duplicados, prompt e historial sin roles ni bit privado, IDs completos de sondeos, 4 shards y costes/procedencia. En cualquier incumplimiento => `INVALID_OR_INCOMPLETE`.
- **M1 solucionabilidad preLLM:** 24/24 referencias estables y 24/24 changed válidas en 7 pasos, además 24 prefijos de error público `DISABLED_TOOL` funcionales sin simular error hacia el LLM. Si falla => `INVALID_REFERENCE`.
- **M2 selección:** `S1` primera acción igual al primer paso válido en ≥22/24 mundos estables y ventaja frente `S0` en primeras acciones ≥4/24; además `S1` éxitos estables ≥18/24 y mejora ≥4 frente `S0`. Reportar el efecto en changed como secundario; sin M2, no afirmar causalidad de nombre del paso.
- **M3 supervisor de finalización:** en al menos una condición T1 supera S1 ≥4/24 mundos de éxito neto; y el supervisor efectivamente actúa (≥4 casos), sin `verify` forzado antes de índice 3 y cero acceso a bits privados. Si no, no adjudicar recuperación de éxito a terminación.
- **M4 recuperación posterror:** `R2` ≥18/24 éxitos y ≥6/24 sobre `R0`, con ≥18 ejecuciones correctas de BACKUP *después* del error, `R1` siempre evaluado y presupuesto de 4 decisiones tras prefijo. Si no, no declarar beneficio del nombre alternativo; ningún brazo prueba aprendizaje de recuperación.
- **M5 auditoría de integridad:** para todos los episodios fuentes `origin` veraces, presupuesto global de 7 acciones y 4 en R, 0 tokens supervisor, eventos exactos, errores forzados públicos y prompts con SHA válidos, no fuga de roles/bit oculto. Si falla => INVALID. Publicar costes por brazo.
- **M6 validez de interpretación:** no adjudicar aprendizaje, autonomía, transferencia de familias ni integración al Producto. `M2–M4` se interpretan como efectos específicos de **estructura externa**; no como reemplazo de una confirmación independiente.

**Decisión:** si M0/M1/M5 PASS, y al menos dos de M2/M3/M4 PASS => `PILOT_MECHANISM_GO_TO_NEW_CONFIRM`. Si solo uno => `PILOT_MIXED`, si cero => `PILOT_NO_GO`. Ningún resultado de F12-M será denominado `CONFIRM_PASS`; los mundos N0–N23 quedarán consumidos tras primera inferencia. No integrar producto ni reutilizar mundos para tuning o re-confirmación.
