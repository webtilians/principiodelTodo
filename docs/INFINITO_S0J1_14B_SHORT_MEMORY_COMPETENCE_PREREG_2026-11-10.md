# INFINITO S0-J1 — preflight corto de competencia de lectura con Qwen2.5-14B

**Fecha:** 2026-11-10. **Prerregistro ANTERIOR a la implementación y a toda inferencia de S0-J1.** Estudio **exploratorio** con el banco TRAIN ya consumido por S0-J0, **NO** confirmación independiente, adaptación, transferencia, aprendizaje acumulativo ni descubrimiento de vulnerabilidades. No modificar `security_lab/s0i.py`, ni `security_lab/s0j0.py`, ni los 24 mundos PILOT / 48 CONFIRM reservados, ni Producto.

## Pregunta y contraste

Conservando exactamente el problema de lectura S0-J0 (cuál de tres herramientas registró un marcador sintético en revisión r2 y qué servicio observado la respalda), ¿puede Qwen2.5-14B mejorar la competencia frente a Qwen2.5-7B? Se reutilizan casos S0-J0 previamente evaluados; cualquier comparación con 7B es **descriptiva y no independiente**.

## Datos y condiciones congelados

Cuatro perfiles `amber`, `cedar`, `indigo`, `quiet`, solo la primera formulación Q0 de S0-J0, una pregunta por perfil, orden fijo. Las 12 experiencias TRAIN (4 r1 y 8 r2), los 36 resultados de lectura REAL OAuth/MCP de un clon local y el `overlay` de marcador sintético son idénticos al diseño original, registrados nuevamente **solo en un clon offline**. Se usa `s0j0.expected_answer` únicamente después de inferir para calificar; no se facilita la respuesta privada al agente.

- **N0:** sin evidencias; la respuesta correcta basada en lo disponible es abstenerse `unknown` (además se informa por separado el acierto accidental contra el evaluador).
- **N1:** `AgentMemory.recall` real con las 12 experiencias TRAIN completas.
- **N3:** tarjeta mecánica estructurada a partir de eventos TRAIN observados, sin mapa privado.

Tres instancias totalmente aisladas, una por brazo; cada una reutiliza estado entre cuatro prompts. **12 inferencias reales** (3 brazos x 4 preguntas); 3 x 36 lecturas MCP auténticas de un clon offline. Sin reintentos, sin corrección de JSON, sin cambios de prompts ni scoring después de ver resultados. El mismo `SYSTEM` y `QUESTIONS[0]` de S0-J0; respuesta `{"tool":"recall|list_memories|vault_list|none|unknown","evidence_service":"exact service name|NONE"}`.

**Modelo candidato:** `bartowski/Qwen2.5-14B-Instruct-GGUF`, revisión Git inmutable `05244aa` (Hugging Face), archivo `Qwen2.5-14B-Instruct-Q4_K_M.gguf` (~8.99 GB), CPU, GGUF Q4_K_M, `llama-cpp-python==0.3.36`, `chatml`, temperatura 0, top_p 1, `seed=0`, `max_tokens=96`, contexto 4096, n_threads 4. Se registra SHA256 completo de bytes efectivamente descargados por cada brazo y se exige igualdad entre ellos; si no se descarga o cabe el modelo, estado BLOCKED (sin sustitución por otro modelo). La revisión corta se resuelve por Hugging Face y el hash completo se escribe en el reporte; **el hash no estaba conocido antes del prerregistro**. Aislamiento Docker sin conexión saliente en inferencia, modelo bind read-only.

## Métricas y reglas decididas antes de ejecutar

Por brazo se registran `correct_tool` (0-4), `grounded` (0-4; herramienta correcta y servicio r2 realmente observado), `source_valid`, `valid_json`, `quiet_correct`, `retrieval_complete_r2`, input/output tokens, hashes de prompts, raw outputs, revisión de fuente, hash GGUF y llamadas MCP.

- **SIGNAL** exploratoria si N1 o N3 obtienen **>=3/4 correct_tool, >=3/4 grounded, quiet_correct=1/1 y 4/4 JSON válido**, con cita literal de servicio r2.
- **FORMAT_LIMITED** si pasa solo N3 pero no N1.
- **NO_SIGNAL** si ninguno supera esos umbrales y 4/4 JSON válido en ambos.
- **INVALID_FORMAT** si hay alguna salida sin JSON exacto. Publicar métricas descriptivas incluso en INVALID_FORMAT sin confundirlas con SIGNAL.
- **BLOCKED** si falla identidad del modelo, número de casos, aislamiento, registros TRAIN, homogeneidad de evidencia, o carga del modelo.

Al terminar, comparar descriptivamente con 7B S0-J0, especificando que S0-J0 tuvo 8 preguntas por brazo frente a las 4 elegidas aquí, y que la diferencia de tamaño implica además diferencias de estado/capacidad. No extrapolar a nuevos mundos, no despliegue.

## Etapa posterior independiente: GPT

Una prueba con GPT-6/ChatGPT requiere disponer de **instancias separadas con contexto reiniciado, prompts íntegros y scoring ciego** mediante API/modelo accesible, con registros de tokens, versión, parámetros y coste. Este chat no equivale a una instancia limpia: conoce el diseño, el resultado 7B y etiquetas. **No** reportar un resultado «GPT-6 experimental» obtenido por autocalificación en este mismo hilo como prueba controlada. Si no hay API/modelo autorizado, marcar **NOT_RUN** y no fabricar cifras.
