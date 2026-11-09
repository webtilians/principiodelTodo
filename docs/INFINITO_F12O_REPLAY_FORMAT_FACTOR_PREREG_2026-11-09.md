# INFINITO F12-O — Repetibilidad antes de contrastar estructura vs. redacción del cursor

**PRERREGISTRO PROSPECTIVO — 2026-11-09.** Se congela en un commit ANTES de escribir código F12-O o emitir inferencias. El fallo de N2 en F12-N O21 (mismo SHA de primer prompt y dos acciones distintas) es el motivo de la fase diagnóstica. **Primero ejecutar REPLAY; solo si pasa el gate congelado se permitirá usar mundos nuevos del FACTOR.** No cambiar gates, mundos, textos o condiciones después de observar inferencias. PR DRAFT, no tocar Producto ni atribuir aprendizaje autónomo.

## Preguntas y restricciones

R: con Qwen2.5-7B-Instruct GGUF Q4_K_M congelado, temperature=0, ¿cuán reproducible es la salida **literal**, no solo la acción, para un **prompt y system idénticos** en un proceso y en dos procesos frescos en GitHub Actions? La prueba puede detectar variaciones; pasar no demuestra determinismo universal. Se analiza además si cambia respuesta por reinicio del proceso y posible CPU distinta.

F: en nuevos mundos, una vez superada R, ¿qué aporta (a) separar el **estado del plan** de la **recomendación**, y qué aporta (b) escribir `NEXT ADMISSIBLE ACTION` frente a `NEXT EXPECTED PLAN STEP`? El estudio factorial no permite atribuir diferencias a layout puro, porque la longitud/tokenización del texto puede variar. Medir esos costes, balancear orden de ejecución, y no equiparar cambio de redacción con aprendizaje.

Congelar modelo: repo `bartowski/Qwen2.5-7B-Instruct-GGUF`, filename `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, HF revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA256 bytes `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, llama-cpp-python `0.3.36`, CPU 4 threads, n_ctx 4096, n_batch 256, seed 0, temp=0, top_p=1, max_tokens=96, chatml, same SYSTEM. No fine-tuning, prompts sin datos privados, `SQLiteCognitiveMemoryStore` real con 10 sondas públicos por mundo. Modelos Fake solo pruebas técnicas.

## Fase R: comprobación PREVIA de reproducibilidad (mundos consumidos de F12-N: SOLO diagnóstico)

Reconstruir los **seis prompts ancla** con código F12-N sellado en `299c593923c41fcfe09f875d35f122af23381d37`: mundos `O0,O7,O21` × estados (a) comienzo `stable`, (b) tras prefijo controlado `ENABLE,READ,PRIMARY→DISABLED_TOOL` usando el motor, las diez sondas y `prompt_for(..., mode="F0")`. La meta se fija literal como `"Use the hidden bit, invert it, and verify success."`; el SYSTEM es el mismo F12-N; no incluir estado privado. Verificar obligatoriamente SHA256 de `O21 stable start` = `b3a52c881af625e9a56166841a334998ec6a366bbb957cb1eabf3c23f099e105` visto en el artefacto F12-N; si no coincide, INVALID_PROTOCOL antes de inferencia.

Lanzar dos workers `replay_0` y `replay_1`, **procesos Python frescos** con instalación y runner idénticos. En cada uno, cargar una sola instancia del modelo, ejecutar cada uno de los 6 prompts **5 veces consecutivas**, sin cambiar contexto ni SYSTEM, sin hacer reset de instancia entre repeticiones (30 inferencias por worker, **60 inferencias**). Orden fijo `O0-start, O0-disabled, O7-start, O7-disabled, O21-start, O21-disabled`; no incluir otros prompts distractores. Registrar SHA del system/prompt, bytes exactos raw, acción JSON, modelo/CPU/versiones, índice repetición, trabajador y tiempos/tokens.

**Gate R0 integridad:** 6 × 5 × 2 = 60 salidas, 6 prompts ancla únicos, SHAs idénticos entre runs y las cinco repeticiones, modelo/pesos/backend/params verificados, sistema idéntico, checksum O21 conocido, respuestas no vacías y JSON acción permitida. Cualquier falla => `INVALID_REPLAY`.
**Gate R1 literal:** para cada ancla, las **10** cadenas crudas entre los dos procesos son idénticas (no basta acción semántica); cualquier diferencia => `REPLAY_VARIANCE_DETECTED`. No ejecutar Fase F en esa run si R0 o R1 fallan. No seleccionar solo los prompts estables tras ver la variación, no cambiar número de repeticiones tras ver datos. Esta muestra limitada no demuestra determinismo en otros CPUs o ejecuciones.

## Fase F: 2×2 factorial en mundos inéditos y controles negativos

**Solo si R0 y R1 PASS.** Desde código F12-N pero herramientas **`s0..s5`** en lugar de q, mismas seis funciones opacas y 10 sondeos públicos, plan externo acreditado `[ENABLE,READ,PRIMARY,verify]`, BACKUP acreditada por sondeos, misma regla adaptativa desde un evento auténtico `PRIMARY→DISABLED_TOOL`: paso permanece en 3/4 pero la recomendación cambia a BACKUP. Ningún brazo ejecuta herramientas automáticamente.

24 mundos inéditos `V0..V23`: todas las permutaciones lexicográficas de `sorted(("LOAD","ENABLE","READ","PRIMARY","BACKUP","HOLD"))`, excluyendo tuplas exactas de F12-J, K, L-C, M, N; ordenar candidatos por `(sha256("F12O_FACTOR_LOCK_2026-11-09:"+",".join(roles)),roles)`, elegir primeras 24, sin filtrar resultados. Balancear bit público/oculto 6 por cada (0,0),(0,1),(1,0),(1,1), reordenar usando `(sha256("F12O_BITS_LOCK_2026-11-09:"+str(i)),i)` en los 24 índices de la lista de pares (valores asociados a índice seleccionado), sin ver respuestas.

Dos factores cruzados, MISMO sistema y resto de prompt, mismo `observed` y `valid plan`:
- `A00` inline + EXPECTED (control de formato F0/F1): `NEXT EXPECTED PLAN STEP (i/4): sX. This cursor advances only after a successful public tool response.`
- `A01` split + EXPECTED: `PLAN POSITION (i/4). NEXT EXPECTED PLAN STEP: sX. This cursor advances only after a successful public tool response.`
- `A10` inline + ADMISSIBLE: `NEXT ADMISSIBLE ACTION (i/4): sX. This cursor advances only after a successful public tool response.`
- `A11` split + ADMISSIBLE (réplica de F3, alias nuevos): `PLAN POSITION (i/4). NEXT ADMISSIBLE ACTION: sX. This cursor advances only after a successful public tool response.`

TODOS cuatro brazos aplican exactamente la MISMA transición de PRIMARY a BACKUP solo después de un error público genuino + evidencia de BACKUP; si no hay error deben mantener la instrucción y recomendar el plan vigente.

Condiciones `stable` sin avería, `changed` avería natural, `forced` con prefijo externo `ENABLE,READ,PRIMARY→DISABLED_TOOL` y **4 acciones LLM** restantes. Horizonte global 7; prefijo cuenta 3 acciones y no recibe tokens LLM. 24 mundos ×3 condiciones ×4 brazos = **288 episodios**, divididos en cuatro shards cada uno 6 mundos/72 episodios. Rotar orden de brazos con `(wid + condition_index) % 4` como F12-N. Preflight determinista 24/24 stable y 24/24 changed y autenticidad del prefijo, todo antes de Qwen.

**Métricas:** éxito verificado, recuperación tras error y BACKUP usada después del error, PRIMARY reintentada tras error, agotamiento de presupuesto, errores de JSON, tokens por brazo, latencia, hashes de prompts y RAW, costes por mundo y estrato. Trazabilidad por acción, fuente LLm vs prefijo, estado del cursor y prueba de respaldo público.

## Gates F congelados y decisión

- **F0 integridad:** real weights pinned, 24 mapas V disjuntos de J/K/L/M/N, entradas equilibradas, 288 episodios únicos y cuatro shards completos, SHA de prompts y JSON bruto completos, no verdad privada. Si falla => `INVALID_FACTOR`.
- **F1 solvabilidad:** referencia 24/24 ambas condiciones, 24/24 prefijos error real, 10 sondas públicos completos sin verify; si falla => INVALID.
- **F2 reglas de intervención idénticas:** para todos brazos en `stable`, cero switches; y en cualquier brazo los switches se producen solo tras error PRIMARY real cuando posición==3/4. No usar truth del evaluador en prompts. Si falla => INVALID.
- **F3 beneficio descriptivo robusto de A11:** `A11` ≥22/24 éxitos en *ambas* `changed` y `forced`, y ≥22/24 en `stable`; si falla => `PILOT_NO_GO` sin prometer mejora.
- **F4 hipótesis factorial principales:** `split_effect` = (A01−A00 + A11−A10)/2 y `semantic_effect` = (A10−A00 + A11−A01)/2, calculadas sobre números de éxitos por 24 mundos, en cada escenario. Un efecto obtiene señal `SUPPORTED_IN_PILOT` si alcanza ≥4 éxitos netos (promedio de dos comparaciones) en `changed` **y** ≥2 en `forced`, sin ser negativo en stable. Los contrastes exactos y pares discordantes siempre se publican, sin reinterpretar un efecto como el otro; no se puede declarar ambos si solo interacción produce mejora. Interacción = (A11−A01)−(A10−A00), reportar sin gate adicional.
- **F5 interpretabilidad:** no modificar mundos ni umbrales tras ver datos; si R0/R1 no PASS, nunca ejecutar F; no atribuir mejora a texto puro si diferencias de tokenización o coste exceden ±10%, reportar como `TEXT_AND_COST_CONFOUNDED`. Resultado factorial preliminar, no confirmación independiente ni nueva política aprendida.

Estado `PILOT_FACTOR_GO_TO_NEW_CONFIRM` solo si F0,F1,F2,F3 pasan y al menos uno de los dos efectos F4 aparece soportado en piloto, sin confusor de coste >10%; `PILOT_MIXED` si F3 pasa pero no el gate de efecto o hay confusor; `PILOT_NO_GO` si F3 falla; `INVALID_FACTOR` si F0/F1/F2 fallan. En ningún caso fusionar en Producto; estudio F en los mismos mapas que sus cuatro brazos, no generalización OOD. Esto es investigación de **capa de ejecución guiada**, no aprendizaje autónomo de políticas.

**Nota de operación:** workflow: contratos → dos trabajos de REPLAY → agregador gate R0/R1 → cuatro shards F (solo si gate PASS) → agregador F. Si REPLAY falla, los 24 mundos V permanecen **no evaluados** (y no se simulan resultados).
