# INFINITO F12-C — Preflight independiente con LLM congelado de 3B cuantizado
**Prerregistro científico/técnico; 2026-11-06. Este archivo se confirma ANTES de crear código F12-C o ejecutar cualquiera de sus casos.**

## Motivación
F12-A (0,5B) produjo 0/8 ejecuciones resueltas en tareas de operaciones simuladas. F12-B (1,5B) produjo 1/4 en nuevos casos, con acciones válidas pero fallos de planificación de precondiciones. F12-B quedó NO-GO para probar memoria. Objetivo F12-C: verificar **competencia autónoma mínima de un LLM mayor**, NO demostrar aprendizaje, de modo que una futura comparación con memoria de INFINITO tenga un agente capaz de acumular experiencias exitosas.

## Modelo y presupuesto fijados prospectivamente
- **Qwen2.5-3B-Instruct**, modelo abierto **GGUF Q5_K_M** del repositorio `bartowski/Qwen2.5-3B-Instruct-GGUF`, archivo exacto `Qwen2.5-3B-Instruct-Q5_K_M.gguf` (~2,22 GB). Se fija el commit Hugging Face obtenido antes de descargar el modelo en la ejecución, se registra SHA256 del archivo descargado, y se prohíben sustituciones. Solo llama.cpp por medio de `llama-cpp-python`, CPU, sin GPU ni claves de servicio, usando modo ChatML de Qwen y `temperature=0`, máximo 96 tokens generados por llamada. `n_ctx=4096` (cualquier truncamiento es fallo técnico, nunca se corta información relevante para salvar un caso), `n_threads=4`. Pinar versión de librería en workflow.
- **Seis casos nuevos** `L0..L5` no vistos en F12-A/B: exactamente tres de autorización y tres de dependencias de servicio, informes/síntomas inéditos. Una sola ejecución por caso, en orden `L0..L5`. Sin selector, memoria, RAG, estrategias, fine-tuning, recuperación de pesos adaptados, acceso a historias de casos anteriores ni feedback de casos anteriores.
- **Seis herramientas simuladas**: inspect, audit_access, audit_dependency, repair_access, repair_dependency, verify. Solo estados internos en memoria del proceso. Semántica pública de cada herramienta, requisitos de auditoría y objetivo de verificar, pero **sin secuencias de solución, etiquetas del evaluador o ejemplos expertos**.
- Máximo **6 acciones por incidencia**; cada llamada de herramienta se selecciona exclusivamente mediante una respuesta real del LLM (JSON `{"action":"..."}`), sin arreglar formatos incorrectos mediante reglas ocultas. Todos los prompts y respuestas han de quedar auditables con hashes y trazas literales. La causa privada `hidden_fault` y los contadores del evaluador nunca se pasan al LLM. Modelo idéntico para los seis casos, weights congelados, sin acceso a herramientas externas ni lectura de archivos por el agente.
- Si la infraestructura GGUF/CPU falla, modelo desconocido, descarga inválida, proveedor responde vacío o se produce OOM: clasificar BLOCKED y no sustituir por mock. Los tests técnicos pueden usar fake marcado explícitamente, sin llamarlo resultado de LLM real.

## Gates antes de cualquier ejecución real
1. **G0 procedencia:** 100 % de las decisiones provienen del LLM GGUF realmente ejecutado, modelo SHA256 + HF revision + versión de backend registrados y sin fallback heurístico. 
2. **G1 competencia:** completar **al menos 4/6** incidencias **y al menos 2/3 en cada tipo** (autorización y dependencia), culminando con `verify` positivo dentro de las seis acciones.
3. **G2 uso de herramientas:** >=90 % JSON válido con herramienta permitida; reacción documentada ante al menos una observación de error/progreso, sin reiniciar artificialmente episodios.
4. **G3 trazabilidad:** 100 % de casos en informes, incluyendo fracaso, pasos, observaciones, recompensas, respuesta textual, coste de tokens y tiempo, presupuestos. Cero acceso de actor a metadatos privados.
5. **G4 aislamiento:** ninguna lectura/escritura de memoria personal/MCP/producto, solo herramientas del simulador, sin DEV/CONFIRM de F11 y sin reutilizar casos K0..K3 para tuning.

Si todos G0-G4 pasan: **GO para diseñar un experimento posterior de aprendizaje con INFINITO y tareas nuevas**, no PASS de aprendizaje ni generalización. Si G1 o G2 fallan: **NO-GO** con este modelo y este contrato, sin repetir los mismos casos para ajustar prompts. Si hay fallo de infraestructura: BLOCKED, no atribuir fracaso al modelo. Un resultado mixto solo sirve para diagnosticar; no se abrirán casos científicos reservados hasta preregistrar nuevo estudio.

## Riesgos declarados
- La cuantización puede degradar algo la calidad y el número de parámetros no determina por sí solo competencia.
- El entorno continúa siendo una familia sintética simple de incidencias, no problemas reales de producción; las herramientas y sus precondiciones están definidas por investigadores, pero ninguna política correcta está insertada en el actor.
- El paso de 1,5B a 3B no es un A/B causal aislado: cambian modelo y casos. No afirmar ganancia atribuible al tamaño.
- No ajustar thresholds después de ver resultados, ni inferir inteligencia general de seis casos.

**Estado en este commit: sin código F12-C, sin inferencia ni resultados.**
