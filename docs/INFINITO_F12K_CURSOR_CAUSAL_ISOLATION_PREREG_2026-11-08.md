# INFINITO F12-K — ¿Qué componente del cursor explica la ejecución?

**Prerregistro prospectivo, 8 noviembre 2026.** Este archivo se crea y confirma en Git **antes de escribir/ejecutar código F12-K, observar salidas de su LLM o abrir evaluaciones**. Una corrección técnica anterior a la primera inferencia se registrará sin alterar tratamientos ni resultados. F12-J (PR #128, J0–J7) es únicamente antecedente consumido y **no** sirve para ajustar F12-K. No abrir DEV/CONFIRM de F11 ni integrar Producto.

## Problema, causalidad y limitaciones

En F12-J B0 (plan sin cursor) 0/8 frente a B1 (cursor: índice y siguiente herramienta exacta) 8/8 tanto estable como con avería; B2 (mismo cursor + instrucción de reparación) 0/8. No sabemos si B1 ayuda por indicar estado, nombrar la próxima acción o encuadre lingüístico. El presente experimento compara tratamientos en los **mismos diez mundos sintéticos nuevos**, con Qwen2.5-7B congelado, mismo plan y episodios públicos, **sin entrenar ni aprender pesos**. El plan es extraído por un algoritmo externo de sondeos ejecutados; no es una política generada por Qwen. En ningún caso describir las mejoras como aprendizaje autónomo.

## Entorno congelado

Seis herramientas **opacas y nuevas** `v0..v5` + `verify` representan seis efectos en un registro binario: LOAD bit público, ENABLE, READ bit oculto (requiere ENABLE), PRIMARY invertir bit, BACKUP invertir bit, HOLD. La meta es verificar el complemento del bit oculto. Un cambio predeterminado hace que PRIMARY devuelva `{"error":"DISABLED_TOOL"}` cada vez, a partir del primer intento; BACKUP siempre conserva su efecto. El evaluador conoce los roles privados, pero el LLM/cursor/SQLite solo reciben los resultados de acciones realmente observadas.

Diez asignaciones nuevas por `v0..v5` (orden de roles privado al evaluador):
- K0: LOAD READ ENABLE BACKUP HOLD PRIMARY
- K1: PRIMARY ENABLE LOAD HOLD READ BACKUP
- K2: BACKUP READ LOAD PRIMARY ENABLE HOLD
- K3: ENABLE HOLD PRIMARY READ LOAD BACKUP
- K4: READ LOAD HOLD BACKUP PRIMARY ENABLE
- K5: HOLD PRIMARY READ ENABLE BACKUP LOAD
- K6: LOAD BACKUP PRIMARY READ HOLD ENABLE
- K7: PRIMARY HOLD ENABLE BACKUP LOAD READ
- K8: READ BACKUP HOLD LOAD ENABLE PRIMARY
- K9: BACKUP ENABLE HOLD PRIMARY READ LOAD

Entradas públicas/ocultas por mundo, en orden, sin selección posterior: `(0,0),(1,0),(0,1),(1,1),(0,1),(1,0),(1,1),(0,0),(0,0),(1,1)`.

Los diez sondeos públicos por mundo (cada uno con reset) y el extractor de efectos son el mismo **contrato experimental** que en F12-J, pero **sobre estos mapas nuevos**: LOAD 0, LOAD 1, ENABLE→READ hidden0, ENABLE→READ hidden1, LOAD→PRIMARY 0/1, LOAD→BACKUP 0/1, READ sin ENABLE (error), LOAD→HOLD. Sondeos externos no decididos por LLM, sin secuencia completa de solución. Requerir que SQLiteCognitiveMemoryStore real recupere todos los IDs exactos `P0..P9` y que el extractor genere `[ENABLE,READ,PRIMARY,verify]` + BACKUP por evidencia pública, sin leer roles internos. El ejecutor mecánico debe obtener 10/10 normal y 10/10 con PRIMARY deshabilitado **antes** de ninguna inferencia; esto es preflight de solucionabilidad, no un brazo LLM.

## Tratamientos congelados

Dos condiciones por mundo: `stable` y `changed`. Se ejecutan seis brazos con **estado reiniciado**, mismo sistema, catálogo, plan completo, alternativa y diez experiencias, máximo siete decisiones del modelo por episodio; orden de brazos rotado deterministamente por mundo/condición. **120 episodios LLM** (=10 mundos×2 condiciones×6 brazos). Formato de acción estricto JSON sin recuperación de parsing inválido.

La función base del cursor computa el **índice real** a partir solo de acciones y respuestas públicas. Avanza una posición si la herramienta esperada devuelve éxito, o si BACKUP con efecto observable reemplaza PRIMARY después de error; no avanza ante error, nunca consulta bit oculto/resultado del evaluador.

- **C0 referencia:** plan e historial, ninguna referencia al paso actual; instrucción neutra de elegir próxima herramienta.
- **C1 solo progreso:** plan e historial + `CURRENT PLAN POSITION (n/4)`, sin dar nombre explícito de herramienta. El agente puede deducirla del plan visible.
- **C2 índice+nombre:** réplica lógica de B1 de F12-J, `NEXT EXPECTED PLAN STEP (n/4): vX`; avance basado en respuestas públicas.
- **C3 longitud neutra:** idéntico a C2 + apéndice sin contenido operativo nuevo. En origen **igualar caracteres** del apéndice al texto de reparación C4 (relleno con espacios); registrar tokens reales porque igualar caracteres NO implica igualar tokens. C3 aísla en parte la longitud/estructura del prompt.
- **C4 reparación verbal:** idéntico a C2 + texto fijo de F12-J: `If the current inversion step returns error DISABLED_TOOL, select the alternative inversion tool supported by the public episodes, then verify. Do not retry a permanently unavailable operation.` No cambiar su wording tras evaluar.
- **C5 cursor desfasado:** mostrar el **índice público correcto** pero un **nombre de herramienta erróneo**, tomado siempre de `plan[min(index+1, last)]`; no marcar al LLM que está equivocado. Esto prueba dependencia ciega del nombre sugerido, no una evaluación de cómo entrenar el cursor. Registrar si detecta contradicción entre plan, índice e historial.

Se mantiene la línea de cursor idéntica en C2,C3,C4,C5 salvo la intervención explícita. C1 y C0 no reciben acción siguiente. No instruir al modelo con el mapa secreto. Ningún tratamiento puede recibir el resultado de los otros.

## Métricas y decisiones predefinidas

Unidad primaria **mundo**, no cada llamada. Informar éxitos /10 separados estable/cambiado y por los 10 pares, secuencia de herramientas exacta, primeras acciones divergentes, correspondencia con cursor, intentos repetidos, uso de BACKUP **después** de `DISABLED_TOOL`, tokens de entrada/salida, tiempo y longitud de prompt por tratamiento. No usar 20 pares como 20 mundos independientes.

**K0 integridad:** mismos pesos Qwen2.5-7B Q4_K_M, HF revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA256 `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, `llama-cpp-python==0.3.36`, CPU, temp 0, contexto 4096, 120 episodios, memoria real solo sintética, sin fugas privadas, 10 sondas/mundo, todos los prompts y decisiones con SHA y texto literal. Falla => INVALID/BLOCKED.

**K1 solucionabilidad:** el ejecutor mecánico 10/10 en ambas condiciones; ninguna solución completa en sondeos. Falla => INVALID, no rescatar con otra política.

**K2 causa del nombre sugerido:** C2 ≥8/10 éxitos estables y mejora ≥3 mundos frente a C1 estable. En cambiado, informar por separado; sin K2, no afirmar que nombrar acción sea crucial, aunque C2 supere C0.

**K3 información de progreso:** C1 mejora C0 ≥3 mundos estables y C1 ≥6/10; si falla, el número de paso no demuestra utilidad independiente.

**K4 contenido adicional frente a longitud:** C3 no pierde >1 mundo frente a C2 **en ambas condiciones**, mientras C4 pierde ≥3 mundos frente a C3 en **al menos una condición**. Informar además razón de tokens reales C4/C3; si excede 1.10, declarar comparación parcialmente confusa en coste de tokenización aun si K4 se cumple. Si C3 y C4 caen igual, no atribuir regresión a contenido semántico.

**K5 cursor erróneo:** C5 pierde ≥3 mundos frente a C2 en condición estable y, entre todos los mundos, acumula al menos tres primeras acciones que coinciden con la herramienta incorrectamente sugerida. Si no, no afirmar obediencia ciega.

**K6 reparación:** con `changed`, registrar uso de BACKUP después del error y éxito. Cualquier mejora del brazo C2 frente a C0 **no** se atribuirá a revisión autónoma si el cursor/alternativa ya fue suministrado. No exigir superioridad de C4 cuando F12-J mostró regresión; K4 mide precisamente ese mecanismo.

**K7 auditoría y costes:** todos los tratamientos registran traza, prompt, SHA, tokens, latencia, entradas y salidas; los mensajes de cursor dependen solo de eventos públicos. Comparar consumos y falsos positivos, sin declarar eficiencia de producto a partir de simulación.

**GO solo para un nuevo estudio confirmatorio** si K0/K1/K7 PASS y al menos una de K2/K3/K4/K5 demuestra su contraste causal específico con trazas coherentes, sin interpretación post hoc como replicación confirmatoria; si ninguna => PILOT_NO_GO. Gates individuales deben reportarse aunque otros fallen. Mantener PR borrador, sin merge ni integración en INFINITO productivo.
