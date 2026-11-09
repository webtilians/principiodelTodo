# INFINITO F12-P — Historial de inferencia y transferencia de la formulación «acción admisible»

**Prerregistro prospectivo — 9 de noviembre de 2026. ESTE commit debe quedar congelado antes de cualquier código o inferencia de F12-P.** El estudio se basa en F12-O `REPLAY_PASS` con prompts agrupados y `PILOT_MIXED` sobre una sola familia binaria. Prohibido modificar hipótesis, scripts, mapas, umbrales o exclusiones después de observar resultados F12-P. Código de Producto sin cambios, PR en borrador.

## Fase H — ¿El orden/historial previo altera la salida del modelo?

Seis prompts **consumidos** del estudio F12-N, reconstruidos exactamente con `observatory/f12o_repeatability_gate.py::build_anchors`: O0/O7/O21 en start/disabled, usando exactamente el mismo SYSTEM/plan/sondeos y sha del primer prompt O21 `b3a52c881af625e9a56166841a334998ec6a366bbb957cb1eabf3c23f099e105`. **No** usar nuevos mundos para esta fase.

Ejecutar **dos procesos frescos independientes** (workers 0 y 1), una sola instancia de Qwen7B por proceso. En cada worker, someter las mismas seis anclas intercaladas en tres órdenes **bloqueados**: (I) lista lexicográfica O0-start, O0-disabled, O7-start, O7-disabled, O21-start, O21-disabled; (R) su inversa; (S) índices `[2,5,0,3,1,4]`. Repetir cada secuencia **dos veces** en el mismo proceso sin reinstanciar el modelo: 3 secuencias ×2 ciclos ×6 prompts = **36 inferencias/worker**, 72 totales. Conservar datos de orden, posición, secuencia, raw completo, tokenización, hashes de prompt/sistema, CPU, backend, modelo, versión y peso.

Model Qwen2.5-7B-Instruct GGUF Q4_K_M congelado con HF revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA256 de pesos `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, llama-cpp-python `0.3.36`, CPU 4 hilos, n_ctx 4096, seed0, temp0, top_p1, max tokens96. `FrozenGGUF` de F12-F. No warm-up escondido. Arranque de worker limpio. `PYTHONHASHSEED=0`. No usar mocks como evidencia científica.

**Gates H0 y H1:**
- H0 integridad: exactamente 72 inferencias, 12 por ancla, dos workers y todos los bloques/ordenes completos, prompt SHA O21 igual al antiguo, parámetros reales iguales y raw JSON con acción válida. Si no => INVALID_HISTORY.
- H1 literal: cada ancla devuelve **una sola cadena de bytes cruda** en las 12 emisiones, cruzando órdenes/posiciones/workers. Si difiere => `HISTORY_VARIANCE_DETECTED`, **no ejecutar Fase T** y conservar todos los artefactos. Si pasa => `HISTORY_PASS_FOR_ANCHORS` y permitir fase T; no declarar determinismo universal. Reportar coincidencia de acciones por separado del raw.

## Fase T — Transferencia a otras familias, no permutaciones binarias

**SOLO si H0/H1 PASS y nunca en caso contrario.** Nuevo simulador controlado con **DOS familias funcionales distintas de los registros binarios** de F12-J..O:

1. **Autorización de operación**: `OPEN → AUTHORIZE → SUBMIT → verify`; `SUBMIT` devuelve `PROVIDER_DOWN` en changed, alternativa pública `SUBMIT_BACKUP`. Herramientas extra `AUDIT`, `RESET`, `DUMMY`. `AUTHORIZE` sin `OPEN` devuelve `NO_SESSION`, commit sin autorización devuelve `UNAUTHORIZED`.
2. **Dependencias de despliegue**: `SCAN → RESOLVE → LOCK → APPLY → verify`; `APPLY` devuelve `CHANNEL_OFFLINE` en changed, alternativa pública `APPLY_BACKUP`. Herramientas extra `NOOP`, `WIPE`. `RESOLVE` sin `SCAN` da `NO_MANIFEST`, `LOCK` sin `RESOLVE` da `NO_DEPENDENCIES`, `APPLY` sin `LOCK` da `NOT_LOCKED`.

Siete herramientas con nombres opacos `r0..r6` en cada familia, con mapeos papel→nombre permutados. `verify` como operación pública adicional, con resultado cierto solo si se completaron las operaciones, sin exponer verdad privada del evaluador al actor. El estado público y feedback de los pasos anteriores se muestran al LLM. El controlador de cursor únicamente accede a `plan`, alternativa acreditada y eventos públicos, no `World.roles`, mundo interno, respuesta futura ni oráculo.

**Mapas nuevos fijos:** por familia, enumerar lexicográficamente `permutations(sorted(roles))` con los siete nombres de rol específicos de esa familia, ordenarlos por SHA256 de `"F12P_TRANSFER_2026-11-09:"+family+":"+join(roles)` y desempate roles, usar los **primeros 12** por familia; exactamente 24 tareas (12 auth y 12 deps). Las familias y sus siete roles son nuevas respecto a F12-J..O, sin reutilizar sus seis-roles mundos. Herramientas y mapeos excluyen selector por rendimiento. Ningún cambio de mapas tras ver datos.

Experiencias públicas: **6 sondeos por mundo** en instancia independiente, sin ejecutar `verify`. Deben permitir derivar sin datos privados el plan válido y BACKUP: para autorización [OPEN], [OPEN,AUTHORIZE], [OPEN,AUTHORIZE,SUBMIT], [OPEN,AUTHORIZE,SUBMIT_BACKUP], [AUTHORIZE], [AUDIT]; para dependencias [SCAN], [SCAN,RESOLVE], [SCAN,RESOLVE,LOCK], [SCAN,RESOLVE,LOCK,APPLY], [SCAN,RESOLVE,LOCK,APPLY_BACKUP], [APPLY]. Guardar y recuperar en `SQLiteCognitiveMemoryStore` genuina, `source_id=P0..P5`, construir plan desde acciones y respuestas públicas, controles que detecten registro contaminado/sondeo incoherente. No existe aprendizaje de política desde el LLM: este estudio mide únicamente ejecución guiada.

**Brazos y parámetros fijos (misma estructura de cursor):**
- E `NEXT EXPECTED PLAN STEP (i/n): rX. This cursor advances only after a successful public tool response.`
- A `NEXT ADMISSIBLE ACTION (i/n): rX. This cursor advances only after a successful public tool response.`
- N `NEXT ACTION (i/n): rX. This cursor advances only after a successful public tool response.` control de redacción más corto.

Todos los brazos recomiendan la herramienta de BACKUP **solo después** de un error público de PRIMARY, según el mapa de plan y prueba de experiencias. No cambiar la posición (paso de SUBMIT/APPLY pendiente); avanzar solo tras herramienta con respuesta de éxito. No incluir instrucciones narrativas de reparación. Misma cabecera, CATALOG, plan, sondeos, historial y system salvo estas palabras.

**Tres condiciones**, `stable` sin avería, `changed` avería PRIMARY encontrada naturalmente, `forced` prefijo mecánico controlado hasta incluir error real `PROVIDER_DOWN` o `CHANNEL_OFFLINE` antes del primer LLM. El prefijo usa exclusivamente el plan público, cuenta dentro del máximo total de **7 acciones** por episodio y queda registrado como `controlled_prefix`, sin asignar tokens LLM; el resto del presupuesto pertenece al actor. 24 tareas × 3 condiciones × 3 brazos = **216 episodios** (6 mundos/shard por 4 shards, 54 episodios/shard). Orden de brazos rotado por `(world_index+condition_index)%3`. `verify` debe ser acción emitida por el LLM.

**Registros y gates T:**
- T0 integridad: H0/H1 previamente PASS, modelo exacto, 24 mapas nuevos y únicos, 12/familia, 6 sondas por mundo y origen SQLite, 216 episodios sin duplicados y 4 shards completos, prompts SHA, raw, fuente, tokens, parámetros. Si no `INVALID_TRANSFER`.
- T1 solucionabilidad pública: la política mecánica desde sondeos supera **24/24 stable y changed** (sin consultar estado privado del cursor) y todos los 24 prefijos forced causan error público real del tipo de la familia. Si falla INVALID.
- T2 intervención auditable: cero conmutaciones en stable, todas las conmutaciones requieren error anterior autenticado de PRIMARY y BACKUP acreditada, ningún otro factor del prompt difiere salvo redacción y su coste de tokenización. Si falla INVALID.
- T3 robustez de A: en **cada familia por separado** A resuelve `stable >=10/12`, `changed>=9/12`, `forced>=10/12`; no extrapolar entre familias si una falla.
- T4 contraste A vs E: A−E gana **≥3 mundos netos** en `changed` de **cada familia** y gana **≥1 mundo neto** en `forced` de cada familia, sin pérdida neta en stable, y al menos dos pares discordantes a favor de A por familia. N se reporta pero no cambia umbrales.
- T5 auditoría y costes: ningún leakage de roles o estado oculto a LLM, ninguna llamada de prefijo atribuida a actor, respetar presupuesto total siete y contar ambos errores distintos. Si el ratio de tokens promedio por decisión A/E por escenario/familia excede 1,10, señalar `PROMPT_COST_CONFOUNDED` y bloquear cualquier etiqueta GO causal.

Estado `TRANSFER_PILOT_GO_TO_NEW_CONFIRM` solo si T0–T5 PASS y sin confusor de coste; `TRANSFER_MIXED` si T0,T1,T2,T5 PASS pero T3/T4 falla (no modificar gate). `INVALID_TRANSFER` si integridad o proceso inválidos. `BLOCKED_BY_HISTORY` si historia no pasa y **sin consumo de mundos T**. Aun un PASS sería solo piloto dentro de dos simuladores, no uso MCP real, aprendizaje autónomo ni validación de Producto.

## Control de producto

No cambiar código de servidor ni MCP de producción; no merge automático, no inventar resultados ni porcentajes, no prometer entrega futura. Registrar resultados solo desde artefactos agregados, con hashes, limitaciones y enlaces GitHub.
