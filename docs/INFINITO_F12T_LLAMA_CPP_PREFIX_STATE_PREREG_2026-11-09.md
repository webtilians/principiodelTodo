# INFINITO F12-T — Ruta de reutilización de prefijo y estado de inferencia en llama.cpp

**PRERREGISTRO PROSPECTIVO, 9 noviembre 2026 — congelar ESTE documento antes de implementar F12-T, hacer inferencias Qwen para F12-T o cambiar sus gates.** Se apila sobre F12-S sin fusionar ni tocar Producto.

## Motivación y fundamento contrastable en código congelado

F12-S PR #144 run 42670942459 logró `INTERACTION_REPLICATED`: con el mismo prompt objetivo `O21-start`, `O21-disabled` como antecedente inmediato y misma instancia dio `{"action":"q2"}` 4/4; con `O0-disabled` o instancia nueva dio `{"action":"q0"}` 12/12. Sigue sin demostrarse un mecanismo interno.

La **versión fijada** `llama-cpp-python v0.3.36` contiene, en `llama_cpp/llama.py::Llama.generate`, detección de `longest_prefix` entre `self._input_ids` y el nuevo prompt tokenizado. Si hay prefijo positivo y `_ctx.kv_cache_seq_rm(-1, reuse_prefix, -1)` devuelve éxito, reutiliza prefijo y decodifica sufijo. Si devuelve falso, vuelve a `self.reset()` + evaluación completa. `Llama.reset()` pone `n_tokens=0` y `_requires_eval=True`; `llama_cpp/_internals.py::LlamaContext.kv_cache_clear()` delega a `llama_memory_clear(..., True)`. Estos son hechos **de código**, NO evidencia de causa todavía. Fuente exacta pública: https://github.com/abetlen/llama-cpp-python/blob/v0.3.36/llama_cpp/llama.py y https://github.com/abetlen/llama-cpp-python/blob/v0.3.36/llama_cpp/_internals.py.

La hipótesis principal T: **la ruta de reutilización automática del prefijo, en una instancia reutilizada después de `O21-disabled`, es necesaria para observar `q2`**. Un observador de llamadas KV que no modifica comportamiento debe dejar `q2`, mientras que impedir *solo* la primera eliminación de sufijo KV de prefijo para forzar el fallback a evaluación completa debe devolver `q0`. Si la manipulación no intercepta una llamada positiva real, el brazo es INVALID y no se atribuye mecanismo.

## Secuencias inmutables, modelo, aleatorización

SOLO los seis prompts diagnósticos ya consumidos F12-N (fuente `observatory.f12o_repeatability_gate.build_anchors()`), mismo SYSTEM y hash `O21-start` **`b3a52c881af625e9a56166841a334998ec6a366bbb957cb1eabf3c23f099e105`**. En cada celda cargar **una instancia** FrozenGGUF, efectuar exactamente las 12 llamadas previas fijas de F12-S `(O0-start,O0-disabled,O7-start,O7-disabled,O21-start,O21-disabled)×2`, ejecutar un último antecedente **`O21-disabled` o `O0-disabled`**, intervenir inmediatamente antes del target, ejecutar target una única vez y cerrar el proceso. No concatenar mensajes, no cambiar texto, no usar nuevos mundos.

Qwen2.5-7B-Instruct-Q4_K_M.gguf real, SHA `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, HF revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, llama-cpp-python==`0.3.36`, chatml, CPU4 threads, n_ctx4096, n_batch256, seed0, temp0, top_p1, max_tokens96. Ningún fine-tune; modelo, pesos, SYSTEM, prompts, temperatura, presupuesto y hardware runner `ubuntu-24.04` fijados entre celdas.

### Factor P (2 antecedentes) × factor I (5 intervenciones antes del target)

Antecedentes `O21-disabled` y `O0-disabled`.

Intervenciones, aplicadas exclusivamente en la **misma instancia que ha procesado los antecedentes**:
- **NATIVE**: no tocar estado; baseline F12-S. Registrar `n_tokens` y `_requires_eval` antes/después. Esperado q2 para O21, q0 para O0.
- **OBSERVE**: envolver temporalmente `model.generator._ctx.kv_cache_seq_rm` con un *spy* que registra las llamadas y delega exactamente al método original (sin alterar resultado). Esperado mismo raw que NATIVE. Registrar el número `p0>0` de eliminación KV y booleano devuelto; **no** llamarlo porcentaje de hits.
- **BLOCK_PREFIX**: mismo wrapper, pero **solo la primera solicitud** `kv_cache_seq_rm(seq,p0,p1)` donde `p0>0` y `p1==-1` devuelve `False` **sin ejecutar** eliminación real; ello impide la optimización de prefijo y fuerza `generate` a tomar el camino de reset+decode completo. Las demás llamadas delegan al original. Se auditará `blocked_positive_request_count==1` (si 0 o >1, INVALID_INTERVENTION); el código original sigue intacto. Esperado O21=q0, O0=q0 si reutilización de prefijo es necesaria.
- **RESET**: llamar explícitamente `model.generator.reset()` antes del target (n_tokens pasa a 0, _requires_eval True); esperar q0/q0. No equivale a limpiar manualmente la KV **antes** de la llamada, aunque el siguiente `eval` elimina entradas.
- **CLEAR_RESET**: llamar `model.generator._ctx.kv_cache_clear()` (API exacta fijada) y después `model.generator.reset()` antes del target; esperar q0/q0. Es un control de contexto limpio adicional; no se utilizará `kv_clear` sin reset porque dejaría punteros/contadores inconsistentes.

**10 celdas**; **3 trabajadores independientes** W0–W2, cada uno ejecuta cada celda una vez en proceso hijo OS nuevo. Orden de celdas por W0 ascendente 0..9, W1 inverso 9..0, W2 rotación fija [3,4,5,6,7,8,9,0,1,2]. Entre celdas no se conserva estado. Total **30 targets**, **390 inferencias previas** (30×13) y **420 inferencias reales**. NO confundir los tres workers con muestras estadísticas de hardware independiente, las máquinas pueden tener distinta CPU.

## Instrumentación y controles

Para cada inferencia: ID antecedente, SHA de SYSTEM y prompt, raw completo, acción válida, temperatura/seed/revisión/backend/pesos, métricas input/output, hora de inicio y duración, `model.calls`, `generator.n_tokens`, `_requires_eval`, tipo de cache, `n_ctx()`. Para `OBSERVE` y `BLOCK_PREFIX`, capture exacto de invocaciones a `kv_cache_seq_rm` con `seq,p0,p1,returned,blocked`; restaurar método original en `finally` y nunca dejar wrapper activo después de target. `n_tokens` es proxy, no contador de hits; no hay hit rate confiable. Documentar que se manipula API interna de llama-cpp-python, no una API pública estable.

Ningún dump de KV con claves privadas, ninguna verdad de evaluador, ningún acceso a servidor MCP Productivo. Solo se emplean prompts antiguos consumidos. No ajustar después de inferencia.

## Gates (congelados) y decisión

- **T0 protocolo:** 3 workers únicos, 10 celdas por worker, 30 OS hijos distintos dentro de cada worker, 420 llamadas reales (390 antecedentes/30 target), todos los registros completos, 2 antecesores×5 intervenciones, hash target/sistema constantes, todos los 6 prompts originales. Si falla => `INVALID_T_PROTOCOL`.
- **T1 runtime real:** Qwen GGUF real y pesos/revisión exactos, backend 0.3.36, parámetros congelados, original sin parche permanente, 13 llamadas previas antes de target y exactamente 14 llamadas si misma instancia, estado de modelo auditable. Los brazos RESET/CLEAR_RESET deben tener `n_tokens==0` y `_requires_eval==True` antes del target. Si falla => INVALID.
- **T2 instrumentación:** OBSERVE debe interceptar ≥1 intento de reutilización de prefijo positivo y delegarlo; BLOCK_PREFIX exactamente 1 bloqueo positivo sin invocar original y fallback completa (debe verse después llamada a `kv_cache_seq_rm` con `p0==0`). Si cualquier brazo no intercepta prefijo válido, => `INVALID_INTERVENTION`, no inferir mecanismo. NATIVE/RESET/CLEAR_RESET no bloquean ni observan llamadas espurias. Comparar índice `p0`, estado y respuesta, no identificar `p0` con cantidad real de KV hits.
- **T3 reproducibilidad:** las 3 respuestas RAW por celda son literalmente iguales y OBSERVE coincide con NATIVE por antecedente. Si falla => `REPLICATE_VARIANCE`, sin claim causal.
- **T4 hipótesis principal:** todas las 3 réplicas de O21 bajo NATIVE y OBSERVE => q2; O21 con BLOCK_PREFIX, RESET, CLEAR_RESET => q0; las cinco intervenciones con antecedente O0 => q0. T0–T3 PASS y patrón completo => `PREFIX_PATH_CAUSAL_SIGNAL`. Este estado significa solo que **la rama optimizada de prefijo afecta al output en las condiciones ensayadas**, NO que KV estuviera corrupta ni que se haya identificado una operación física defectuosa.
- **T5 alternativas y límites:** si T0–T3 pasan, publicar diferencias exactas de cada tratamiento; sin patrón T4 => `MIXED_MECHANISM` o `NO_EFFECT_OBSERVED` según outputs, sin mover umbrales. Si pasa RESET pero no BLOCK_PREFIX, no atribuir causa solo al prefijo. Ningún resultado autoriza cambiar Producto ni entrenar el LLM. Los 24 mundos de transferencia F12-P-T siguen reservados.

**Estado:** `INVALID_T_PROTOCOL` si T0/T1 falla; `INVALID_INTERVENTION` si T2 falla; `REPLICATE_VARIANCE` si T3 falla; `PREFIX_PATH_CAUSAL_SIGNAL` si T4 PASS; en otro caso `MIXED_MECHANISM` o `NO_EFFECT_OBSERVED`. En todos los casos conservar artefactos y código congelados. La inferencia de mecanismo está condicionada a que el efecto NATIVE sobreviva a instrumentación OBSERVE y a que BLOCK_PREFIX fuerce realmente la ruta de evaluación completa.
