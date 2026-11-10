# S0-I-R — Enmienda exclusivamente técnica de ejecución, previa a repetir inferencias

**Fecha:** 2026-11-10. **Experimento científico congelado:** protocolo S0-I commit `da079145`, 12 TRAIN, 24 PILOT, 48 CONFIRM, tres brazos, Qwen2.5-7B y su SHA256, métricas y umbrales idénticos. Esta enmienda **no revisa la hipótesis, mundos, selección de acciones, memoria ni umbrales**.

## Evidencia de bloqueo original
- [Actions #42715586396](https://github.com/webtilians/principiodelTodo/actions/runs/42715586396): contratos 84 PASS + 1 XFAIL anterior; job de inferencia CANCELLED al límite `timeout-minutes:55`, a las 23:47 UTC del 9 de noviembre; empezó a las 22:52 UTC. La salida no incluye informe final ni métricas A0/A1/A2. La cancelación es **BLOCKED técnico**, no resultado de la hipótesis.
- Trabajo secuencial anterior: 24 mundos PILOT ×3 brazos = 72 inferencias Qwen, cada una con contexto tokenizado comparable, más 36 llamadas TRAIN OAuth/MCP. Podría pasar el límite de una sola máquina.

## Cambio de ejecución únicamente
- Se ejecutan **tres shards PILOT paralelos** indexados 0,1,2. Cada shard recibe los mundos originales `PILOT[index::3]`, ocho mundos por shard; NO crea nuevos mundos ni cambia su contenido. Dentro de cada mundo se conservan A0/A1/A2 en el orden original. 24 inferencias por shard; **72 totales**.
- Cada shard inicializa modelo con los mismos pesos fijados SHA256, parámetros, biblioteca y semilla de S0-I; rehace en un clon temporal 12 episodios TRAIN, `36` consultas OAuth/MCP, misma `AgentMemory` SQLite y misma tabla A2. Ningún entrenamiento ni dato de otro shard circula entre shards. Se contabilizan **36 TRAIN por réplica de shard**; ya no se comunica el recuento TRAIN original como si solo existiera una réplica.
- Un ensamblador determinista separado valida SHA de código esperado en el runner, IDs de todos los mundos, 3 brazos por mundo, formato, 1 llamada MCP por brazo, SQLite observado, presupuesto tokenizado y descarta shards faltantes/duplicados. Agrega las filas por orden original y evalúa **la función `pilot_gate` original, sin modificaciones**.
- **Solo si** PILOT y contratos pasan, ocho shards CONFIRM independientes indexados 0..7, seis mundos cada uno `CONFIRM[index::8]`: 18 inferencias Qwen por shard, **144 totales**. El ensamblador aplica `confirm_gate` original y emite GO/NO-GO científico. CONFIRM jamás se ejecuta con piloto NO-GO/INVALID/BLOCKED.
- El runtime por shard se limita a 55 minutos, como antes; el workflow global puede durar más por descargar y preparar el modelo. Se conservan contenedores `--network none`, read-only, 4 CPU, 6500 MB; descargas solo en fase previa al aislamiento, modelo montado read-only. No se cambia el mecanismo de evaluación.
- Antes de usar Qwen se ejecutan contratos de coherencia de partición, coste, sumas y gates con resultados **sintéticos fabricados para tests**, nunca considerados resultados científicos.

## Riesgo metodológico residual
El modelo se inicializa desde cero por shard y se invoca con `temperature=0` y `seed=0`; en principio sus respuestas son por solicitud, pero el orden global de llamadas cambia. Registrar explícitamente esta variación y verificar determinismo técnico cuando sea posible. Si aparecen diferencias no atribuibles a esa condición, no afirmar equivalencia experimental. Repeticiones de TRAIN en cada shard aumentan llamadas MCP totales del laboratorio sin alterar experiencias ni presupuestos comparados dentro de cada mundo.

**Decisión de bloqueo:** un job fallido, parcial, sin reportes, diferencia de cobertura, discrepancia de modelo o gates de presupuesto produce `BLOCKED/INVALID`, nunca GO. No alterar la fuente `security_lab/s0i.py` ni el protocolo preregistrado. Mantener PR draft, sin fusionar ni desplegar.
