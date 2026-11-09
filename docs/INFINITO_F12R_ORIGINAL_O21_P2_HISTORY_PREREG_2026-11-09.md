# INFINITO F12-R — Reproducción del cambio q0→q2 con O21 en segunda posición

**PRERREGISTRO PROSPECTIVO · 2026-11-09 · congelar antes de implementar código F12-R y antes de inferencia.**

## Antecedente y contraste principal

F12-P-H run 42617573328 reprodujo `O21-start` exactamente igual (SHA `b3a52c881af625e9a56166841a334998ec6a366bbb957cb1eabf3c23f099e105`) pero con salida `{"action":"q2"}` en el **bloque inverso**; había dos ciclos previos del bloque normal de 6 prompts. F12-Q run 42625830712 dio `{"action":"q0"}` 16/16, pero allí O21 ocupaba posiciones 4 o 6 en procesos de ensayo nuevos, jamás posición 2 tras las 12 llamadas previas originales. **F12-R no busca aumentar rendimiento, sino reproducir y localizar la dependencia del historial.** No consumir los 24 mundos de transferencia reservados a F12-P-T, no ejecutar simuladores nuevos, no cambiar Producto, no atribuir memoria cognitiva o aprendizaje al modelo.

Se usarán SOLO seis prompts ya consumidos de F12-N, obtenidos sin tocar implementaciones congeladas desde `observatory.f12o_repeatability_gate.build_anchors()`, mismo SYSTEM, checksum fijo de O21 y pesos Qwen2.5-7B-Instruct Q4_K_M SHA `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, HF revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, `llama-cpp-python==0.3.36`, chatml, CPU n_threads=4, n_ctx=4096, n_batch=256, seed=0, temp=0, top_p=1, max_tokens=96.

## Congelación de secuencias

Nombres exactos: A=`O0-start`, B=`O0-disabled`, C=`O7-start`, D=`O7-disabled`, E=`O21-start` (objetivo), F=`O21-disabled`.

- Prehistoria `H0`: **cero** llamadas previas.
- Prehistoria `HP`: **12 llamadas** antes del bloque de prueba: `(A,B,C,D,E,F)` repetida dos veces. Reproduce EXACTAMENTE lo que antecede al primer bloque inverso en F12-P.
- Prehistoria `HR`: **12 llamadas** antes del bloque de prueba: `(F,E,D,C,B,A)` repetida dos veces. Es el control de orden contrabalanceado, no un control de semántica equivalente.

- Posición `P2`: ejecutar `F,E`. Se reproduce el **primer tramo de bloque inverso** F12-P, con E en posición 2 y el último antecedente F.
- Posición `P4`: ejecutar `A,C,F,E`. Mantiene F como antecedente inmediatamente anterior, pero cambia posición y dos antecedentes extra. **P2 frente a P4 confunde posición con cantidad y contenido de antecedentes**: reportar como `position_and_extra_history`, no causalidad pura de posición.

- Instancia `Ireuse`: la instancia Qwen que ejecutó prehistoria y prefijo conserva estado hasta emitir E.
- Instancia `Ifresh`: la instancia que ejecuta prehistoria y prefijo se destruye, `gc.collect()`, se carga una **instancia nueva** antes de E. Se registran `model.calls=1` tras E, y se reconoce que la biblioteca llama.cpp sigue cargada en el mismo proceso OS; no equiparar a reiniciar el proceso entero.

Cruzar las tres prehistorias × dos posiciones × dos instancias = **12 celdas**. Ejecutar cada celda en su **propio subproceso OS recién iniciado**, dos workers separados `W0` y `W1`: W0 celdas índice 0..11, W1 11..0, sin cambios de prompt/configuración. **24 salidas E en total. Conteo exacto preregistrado:** por worker, 8 celdas con 12 llamadas de prehistoria =96, 12 celdas con 1 o 3 llamadas de prefijo =24, más 12 targets =132 inferencias; total **264 llamadas reales** (240 antecedentes y 24 E). `HP,P2,Ireuse` es la reproducción primaria F12-P: exactamente 12 normales + F + E en una instancia. Todas las celdas realizan `E` una sola vez y registran raw literal. Dos workers en máquinas GitHub Actions `ubuntu-24.04` (sin garantizar CPUs idénticas).

El formato de llamada es independiente `model.chat(SYSTEM,prompt)` (no historial de mensajes de chat concatenado). Si vemos arrastre de inferencia, es del runtime de la misma instancia, no de una conversación textual explícita.

## Auditoría

Por cada llamada registrar: `anchor_id`, SHA del prompt y SYSTEM, raw, acción parseada, métricas de tokens y tiempo, contador de llamadas antes/después, variables de entorno y metadata del modelo congelado. Capturar `generator.n_tokens` antes/después, tipo de cache y `generator.n_ctx()`, registrando `kv_cache_hits_available=false` si no existe métrica fiable. Guardar IDs de procesos hijos, worker, orden de celda, instancia nueva vs reutilizada, longitud de prehistoria, posición y secuencia literal. No interpretar `n_tokens` como hit rate. Sin nuevas reglas de autocontrol ni cambio de temperature.

## Gates predefinidos

- **R0 completitud:** dos workers con 12 celdas únicas cada uno, subproceso nuevo por celda, 264 llamadas reales (240 antecedentes + 24 E), conteo por celda exacto, fuente única O0/O7/O21, hashes y salidas JSON validas; si falla => `INVALID_R_PROTOCOL`.
- **R1 modelo/procedencia:** pesos, backend, revisión, temp0, n_threads4, n_ctx4096, max_tokens96, n_batch/seed solicitados, integridad `SYSTEM`/SHA del target O21; `Ifresh` calls target=1 y `Ireuse` calls target=(nº antecedentes+1), anotación de KV sin inventar hits; si falla => `INVALID_R_PROTOCOL`.
- **R2 repetición por celda:** las dos réplicas de cada celda tienen exactamente el mismo RAW de E; si falla `INCONCLUSIVE_REPLICATE_VARIANCE`; no inferir efectos causales de diferencias no repetibles.
- **R3 reproducción directa:** en `HP,P2,Ireuse`, ambas réplicas generan exactamente `{"action":"q2"}`. Si sucede, `ORIGINAL_SEQUENCE_REPRODUCED`; si no, `ORIGINAL_SEQUENCE_NOT_REPRODUCED`, sin invalidar otros controles.
- **R4 aislamiento exploratorio:** contrastes emparejados de `HP` vs `H0`, `HR` vs `HP`, `P2` vs `P4`, `Ireuse` vs `Ifresh`, manteniendo otros factores fijos. No atribuir exclusividad a `P` por su confusión conocida. Etiquetar señales solo si R0,R1,R2 PASS. Las 12 celdas se publican completas incluso si no reproduce q2.
- **R5 límites:** no usar herramientas del evaluador privado como entrada, no cambiar prompts ni original F12-P/Q, no consumir ni desbloquear mundos reservados F12-P-T, no confundir diferencias de runtime con aprendizaje de políticas, no integrar Producto.

**Estado**: `REPRODUCED_WITH_CONTROLS` cuando R0,R1,R2,R3,R5 PASS; `NO_REPRODUCTION` cuando R0,R1,R2,R5 PASS, R3 FAIL; `INCONCLUSIVE_REPLICATE_VARIANCE` si R0,R1,R5 PASS pero R2 FAIL; `INVALID_R_PROTOCOL` ante fallo R0/R1/R5. Ningún resultado constituye todavía prueba de causalidad universal ni de comportamiento en otros dispositivos/CPU. Si se observa `q2` dependerá de hardware/contexto operativo que requerirá confirmación.

**Revisión antes de commit:** cantidades exactas 12 celdas/worker; HP/HR tienen 8 celdas×12=96 antecedentes largos; sumatoria prefijo P2 y P4 = 6×1+6×3=24; 12 targets =132/worker; dos workers =264. Las 24 tareas transferibles siguen sin usarse.
