# INFINITO F12-F — Verificación de afirmaciones de políticas contra experiencias observadas

**Prerregistro prospectivo, 2026-11-06. Este commit precede CUALQUIER código, generación y evaluación F12-F.**

## Hipótesis
F12-E (PR #104, [run 42157861095](https://github.com/webtilians/principiodelTodo/actions/runs/42157861095)) confirmó un riesgo: un LLM 7B citó episodios exitosos S4/S5 para atribuir a SIGMA un éxito que en realidad correspondía a DELTA. La mera existencia del ID de la fuente **NO prueba que la proposición sea cierta**. F12-F aislará el **verificador de procedencia** que contrasta afirmaciones con observaciones visibles, y medirá errores de aceptación y rechazo frente a controles simples.

### Pregunta causal
¿Un **validador determinista de trazas públicas**, sin conocer las reglas verdaderas del simulador, evita promover afirmaciones contradichas por sus fuentes, sin rechazar excesivamente afirmaciones realmente observadas? ¿Qué pierde frente a una política de aceptación por cita y otra de preferir solo los informes más recientes?

**No evaluar aún la ventaja en éxito de un agente ni llamar "aprendizaje" al filtro determinista.** F12-F es una prueba de guardarraíl epistemológico para una futura F12-G de políticas y generalización.

## Entorno, modelo y memoria
- Modelo real congelado EXACTAMENTE Qwen2.5-7B-Instruct Q4_K_M usado en F12-D/E, `bartowski/Qwen2.5-7B-Instruct-GGUF`, archivo `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, revisión HF `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA256 pesos `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, `llama-cpp-python==0.3.36`, CPU, sin fine-tuning.
- Memoria real `SQLiteCognitiveMemoryStore(path=":memory:")` de INFINITO con `HashEmbeddingProvider`, exclusivamente datos sintéticos de F12-F; nunca MCP o memoria del usuario.
- **8 episodios nuevos** de interacción con herramientas (V0..V7): cuatro perfiles públicos inéditos (`ember, moss, lunar, harbor`) con versiones públicas R1 y R2. Para cada perfil, reglas simuladas invierten qué operación es compatible entre revisiones; la compatibilidad privada solo vive en `SimulatedTools`. Cada episodio actor usa hasta 5 decisiones, sin memoria/etiquetas de oráculo, con `inspect, apply_delta, apply_sigma, verify`. Orden R1 completo antes de R2. Los resultados observados (acción, texto público de aceptación/rechazo, verificaciones posteriores) se guardan como EPISODIC en SQLite. Este banco V **no reutiliza** S/H de F12-E, M de F12-D, K/L anteriores ni DEV/CONFIRM F11.
- El mismo LLM emite afirmaciones **estructuradas** a partir de **cada episodio** y una síntesis global del conjunto de los ocho episodios: `{source_id, profile, revision, operation, observed_outcome, verified_after_operation}`. Se solicita JSON para hacer cada proposición auditable; no se le proporcionan acciones correctas ni etiquetas del evaluador. Respuestas malformadas se registran y se rechazan, sin reparar automáticamente el JSON. Una afirmación debe citar una fuente existente; **tener estructura no implica ser verdadera**. El contenido libre heredado F12-E no se considera comprobado; requeriría normalización explícita separada antes de aceptarse.

## Validadores, sin mapa de verdad
Todas las condiciones reciben EXACTAMENTE los mismos candidatos y los mismos registros de experiencia observados. Solo se consulta la tabla de memoria pública (sin `_compatible` ni tabla de reglas), sin ejecutar acciones contrafactuales ni API externa:
- **C0 Cita:** acepta una proposición cuando `source_id` existe en SQLite; reproduce el fallo F12-E.
- **C1 Recencia simple:** acepta una proposición citada solo si la fuente es del último registro del mismo perfil (R2), sin verificar operación ni feedback. Se evalúa el coste de descartar verdades históricas R1.
- **C2 Evidencia:** exige concordancia exacta de source ID, perfil, revisión, operación realmente ejecutada, resultado afirmado (aceptado/rechazado) y evento público `verify` posterior a la aceptación. Una afirmación con resultado **aceptado** y verificación positiva **posterior** puede obtener estatus `PROMOTABLE_OBSERVED_SUPPORT`; las observaciones negativas se guardan como evidencia negativa pero **nunca** se promueven como política de éxito.
- **C2 en uso actual:** adicionalmente rechaza como `STALE_FOR_TARGET` la reutilización de una afirmación válida para R1 cuando el objetivo de ejecución tiene revisión R2, y viceversa. Esta condición no niega la verdad histórica, solo su aplicabilidad. No consultar reglas privadas.

## Conjunto de pruebas prospectivo
1. **Pruebas adversariales construidas exclusivamente desde trazas públicas** (no son inferencias del LLM): para cada episodio con al menos una acción de mantenimiento, construir una afirmación correcta sobre una acción ejecutada y variaciones programadas: operación cambiada, resultado invertido, revisión falseada, cita inexistente, `verified_after_operation` falseado. Se registran etiquetas con el mismo historial visible usado por el verificador; son pruebas de especificación y **no evidencia independiente de inteligencia**. Contar falsos aceptados y falsos rechazados en C0/C1/C2.
2. Evaluar también las afirmaciones **efectivamente propuestas por Qwen7B**, sin filtrado manual ni reintentos condicionados a sus contenidos: cuántas pasan la comprobación de cita, de recencia, de correspondencia factual y de soporte positivo verificable. Registrar abstenciones/JSON inválido.
3. Regresión F12-E con un test local de afirmación estructurada: S4/S5 justifican `delta` aceptado y verificado, **no** `sigma`. Esta regresión se usa solo para diseño y tests, **NO** como confirmación independiente.

## Gates definidos antes de resultados
- **G0 procedencia/aislamiento:** verificación de revisión/SHA exactos del modelo real, todos los episodios procedentes de respuestas reales, cero `_compatible`/mapeo verdadero en los argumentos de C0/C1/C2, `SQLiteCognitiveMemoryStore` auténtico, registros íntegros.
- **G1 seguridad de verificación:** C2 debe tener **cero promociones incorrectas** en adversariales y en auditoría de afirmaciones propuestas por LLM; una afirmación con fuente inexistente/contradictoria nunca se promociona. Si no existe ningún episodio verificado, reportar **INCONCLUSIVE** para el criterio de promoción.
- **G2 conservación:** C2 debe mantener **>=95 % de controles positivos apoyados** con aceptación y `verify` posterior, y rechazar >=95 % de mutaciones con desacuerdo factual. Denominadores públicos por tipo de mutación.
- **G3 comparador:** informar false-positive/false-negative de C0 y C1 sobre el mismo conjunto; C1 puede rechazar indebidamente R1 por recencia. No atribuir a C2 superioridad causal si la prueba se deriva de su propia semántica.
- **G4 utilidad/alcance:** documentar cuántas propuestas de Qwen7B resultan promovibles y cuál es la tasa de abstención. **NO** declarar mejora de éxito del agente, aprendizaje de pesos, transferencia OOD ni efectividad de producto.

Si todos los gates técnicos son válidos, resultado **PASS de guardarraíl técnico en este banco**, no PASS científico ni entrada al producto. Un estudio futuro sobre mejoras cognitivas necesita problemas nuevos, cuatro réplicas, comparación con RAG episódico fuerte, intervención del verificador y nuevos datos de confirmación. Si las afirmaciones son demasiado triviales, si el extractor LLM no emite ninguna o si C2 solo demuestra tautológicamente el predicado de su diseño, declararlo limitado.

## Materiales y custodia
Artefactos de Actions con trazas públicas, propuestas LLM sin corregir, decisiones de los tres validadores, tasas por estrato y costes. El evaluador guarda la compatibilidad privada **fuera** de la memoria y de los validadores. No hay contenido de usuarios ni acceso a redes/herramientas desde el actor. Se preservan F11 DEV/CONFIRM y todos los holdout anteriores.

**Estado al registrar: solo el protocolo; aún no existen código F12-F, inferencias ni resultados.**
