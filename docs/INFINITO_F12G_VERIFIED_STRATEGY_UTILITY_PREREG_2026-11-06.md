# INFINITO F12-G — ¿Mejoran las decisiones las estrategias verificadas?

**Prerregistro prospectivo: 2026-11-06. Este commit se crea ANTES de código, entrenamiento, evaluación y lectura de cualquier resultado F12-G.**

## Pregunta y límites
F12-E observó mejora de la memoria episódica frente a ninguna memoria, pero su estrategia con citas falsas degradó decisiones. F12-F mostró que un filtro determinista sobre **trazas públicas** reconoce falsos apoyos y distingue versiones, pero no probó utilidad en tareas nuevas. **F12-G pregunta si mostrar al actor estrategias redactadas por el LLM y filtradas por evidencia (A3) mejora los éxitos/costes, con igual modelo y recursos, frente a memoria episódica fuerte (A1), estrategias SIN validar (A2) y un control fuerte de cronología (A4).** No confundir contrato de validador con aprendizaje de políticas generalizables.

### Modelo y ejecución
Mismo LLM real y congelado de F12-D/E/F: `bartowski/Qwen2.5-7B-Instruct-GGUF`, archivo `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, HF revisión `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA256 pesos `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, llama-cpp-python 0.3.36 CPU, temperature=0, n_ctx=4096, n_threads=4; sin fine-tuning, API secreta ni modelos auxiliares. No usar resultados de F12-E/F para escoger prompts mirando U.

**Única réplica piloto TRAIN/heldout**, no CONFIRM, no prueba causal fuera de esta familia. GH Actions guarda toda evidencia. No abrir F11 DEV/CONFIRM.

### Entorno y observación
Familia sintética nueva de reparaciones **opacas**: seis experiencias TRAIN `W0..W5` para tres perfiles públicos `coral, willow, quartz` con versiones R1/R2 (R1 antes que R2). Las operaciones `apply_delta` y `apply_sigma` invierten compatibilidad entre R1 y R2. Solo las herramientas simuladas conocen la compatibilidad correcta y responden **accepted/rejected**; `verify` exige reparación compatible previa. El actor no recibe la tabla de reglas, ni un ejemplo experto, ni contrafactuales no ejecutados.
- El LLM ejecuta cada TRAIN sin memoria, máximo 5 acciones, y `SQLiteCognitiveMemoryStore(path=":memory:")` de INFINITO almacena el historial fiel `(source_id, public_profile, public_revision, goal, actions, tool_feedback, verified_success)` en recuerdos `EPISODIC`. Los eventos íntegros se conservan en el artefacto.
- Una llamada al **mismo** LLM por cada recuerdo produce una afirmación JSON `source_id, profile, revision, operation, observed_outcome, verified_after_operation`. Debe representar acciones que realizó, no soluciones de un observador externo. Se guardan tanto propuestas válidas como fallidas; no se corrige un JSON malformado.
- Un verificador determinista **sin acceso al mapa privado del simulador**, idéntico en semántica al C2 de F12-F, comprueba acción realmente ejecutada, aceptación observada y `verify` posterior. Conservar solo proposiciones de éxito apoyadas, con revisión y fuente. Las aceptadas se guardan en `SEMANTIC` como `observed_verified_claim`: son **hechos históricos** acreditados, NO políticas transferibles garantizadas.
- **Seis servicios heldout nuevos** `U0..U5`, misma distribución de perfiles/versiones vistos durante TRAIN, servicios y objetivos distintos. Exactamente 3 en R1, 3 en R2. Como las combinaciones son conocidas, evalúa **reutilización temporal en servicios nuevos**, no generalización a perfiles/estructuras OOD. No reutilizar W/U como confirmación.

### Brazos y contrafactuales ejecutados
**Cinco condiciones** con idéntico LLM, prompt de sistema, temperatura, herramientas, límite de memoria `1000 caracteres`, máximo de acciones y feedback visible:
- `A0`: sin memoria (techo inferior descriptivo, no rival principal).
- `A1`: memoria episódica **competente**, recuperada del SQLite real por similitud y filtro exacto de perfil público. Se muestran **ambas revisiones** R2 y R1 con texto compacto fiel de TODAS las acciones, observaciones de aceptación/rechazo y verificación; la tarea actual debe seleccionar la revisión apropiada. Nunca elegir una acción a escondidas desde la lógica del evaluador.
- `A2`: solo afirmaciones **no verificadas** generadas por el LLM, con ID y revisión públicos, aunque sean falsas; sin apoyo ni acción correcta del evaluador. Si la salida no es JSON válido, el brazo carece de esa afirmación, **no se rescata automáticamente**.
- `A3`: mismas propuestas de A2, pero **solo** afirmaciones aceptadas por C2, fuente y revisión. Si el filtro no promueve propuestas, contexto vacío; evitar fallback episódico que confunda el efecto.
- `A4`: baseline fuerte de cronología basado **solo en eventos públicos**: de los registros, extraer para cada perfil+revisión la última operación aceptada y verificada; mostrar la proposición y su fuente. Este control no usa predicción del LLM ni la tabla privada. Debe igualar o superar una mera reconstrucción de la misma evidencia; si supera A3, no decir que se ha demostrado una nueva capacidad cognitiva.

Se evalúa cada U en **tight=2 acciones** y **slack=3 acciones**, con todas las cinco condiciones (6×2×5=60 episodios). `apply_delta/sigma`, `verify`, `inspect` son las únicas herramientas; las inválidas gastan presupuesto. La memoria TRAIN se congela **antes** de evaluación, no se actualiza con observaciones de U. El orden de brazos rota de forma determinista por caso y presupuesto para mitigar confusión de ejecución. Todos los límites de contexto son iguales; los tokens efectivamente consumidos se comparan, **no se exige artificialmente que A0 reciba texto de relleno**.

### Gates y lectura antes de resultados
- **G0 integridad:** modelo real SHA correcto; ≥3/6 episodios TRAIN con `verify`; 6 recuerdos en SQLite; 6 propuestas LLM con JSON válido o estado de abstención; ningún dato `_compatible` en memoria, síntesis o prompts; 60 evaluaciones completas; contexto ≤1000 caracteres y presupuestos idénticos.
- **G1 validez de evidencia:** A3 nunca muestra afirmaciones contradichas por fuente observada. Se informa nº propuestas válidas, falsas, rechazadas y cobertura de las tres combinaciones/revisiones. Si A3 carece de una afirmación en más de 2/6 combinaciones, reportar **coverage-limited**, no inferir que validar es malo.
- **G2 utilidad incremental tight:** A3 supera a **A1 por ≥2 éxitos de 6** y no empeora ni R1 ni R2. Exigir también A3 ≥ A2 en tight. Si no, **NO-GO** de ventaja cognitiva de estrategias verificadas.
- **G3 frente a control cronológico fuerte:** A3 supera a A4 por ≥2/6 en tight, **O** con iguales o más éxitos ahorra ≥15% de tokens de prompt en tight. Si A4 iguala A3 al mismo coste, **NO se demuestra ventaja incremental del validador/LLM**.
- **G4 robustez y coste:** A3 no pierde ≥2 éxitos frente A1 en slack; informar resultados por versión R1/R2, tokens de entrada/salida, tiempos CPU, nº acciones, acciones innecesarias y coste de síntesis/verificación. Si falla, NO-GO aunque G2 pase.
- **G5 seguridad de inferencias:** no atribuir mejora a políticas aprendidas si la evidencia se limita a copiar un hecho de TRAIN; el caso es sintético y con combinaciones conocidas. Ningún `DEV/CONFIRM` F11, ningún cambio al producto/MCP ni lectura de datos personales.

**GO solo para diseñar estudio confirmatorio independiente** si G0–G5 alcanzan criterios y existe variación real en candidatos válidos/incorrectos. Los números de un piloto de seis casos NO son un PASS científico; cualquier confirmación posterior exige perfiles NUEVOS, reglas diferentes y ≥4 réplicas. **No modificar prompts, cotas, filtros o tareas tras ver U0–U5.** En un empate o saturación, publicar **NO-GO** y no ejecutar otro banco equivalente para obtener porcentajes mejores.

## Anticipación de falsos positivos
A3 podría ganar a A1 solo por **resumir** recuerdos a una forma más fácil de leer: A4 controla parte de esta confusión. Asimismo, si las propuestas LLM están todas fundamentadas, A2 y A3 serán idénticos: se reportará ausencia de intervención efectiva. Si A0 falla por `inspect` innecesario, esa es una limitación del prompt, no demostración de inteligencia en memoria.

**Estado de este commit: SOLO protocolo prospectivo; ningún código ni resultado F12-G.**
