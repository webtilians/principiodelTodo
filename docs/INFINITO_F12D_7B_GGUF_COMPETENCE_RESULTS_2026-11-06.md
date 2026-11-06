# INFINITO F12-D — Resultado del preflight con un LLM 7B auténtico congelado

**Fecha: 2026-11-06.** GitHub Actions **SUCCESS técnico**; gates de competencia F12-D **G0–G4 PASS**. **GO para PRERREGISTRAR UN ESTUDIO NUEVO CON MEMORIA**, NO es PASS de aprendizaje ni validación OOD.

## Orden prospectivo y evidencia

1. Prerregistro de F12-D guardado **antes de código e inferencia**: `docs/INFINITO_F12D_7B_GGUF_COMPETENCE_PREREG_2026-11-06.md`, commit **d90cde21aa548a92f1c43038f3f84cc0a9e27f8e**. Modelo Qwen2.5-7B-Instruct, GGUF Q4_K_M, ocho tareas nuevas M0–M7, gates >=6/8 **y >=3/4 por categoría**, mismo catálogo/prompts/semántica de F12-C.
2. Código, tests y workflow: `observatory/f12d_7b_competence.py`, `tests/test_f12d_7b_competence.py`, `.github/workflows/infinito-f12d-7b-competence.yml`. Se reutilizó el arnés sintético de F12-C, sin modificar la física ni copiar casos consumidos.
3. [**GitHub Actions run #42152303314**](https://github.com/webtilians/principiodelTodo/actions/runs/42152303314), commit de lanzamiento **e169d5e2423d755058fa4a4e335f4e1e9eae20bf**: tests de aislamiento y fake backend PASS; inferencia 7B real CPU PASS; subida de evidencias PASS. **Una ejecución de evaluación**, sin ajustes postresultados.
4. Artefacto `f12d-qwen-7b-q4-frozen-competence`, ID **13473890754**, digest ZIP **sha256:4640d9f99274842de2a4cabe0c3282f877295364f224dcccd435d380ee3e2a34**; `preflight_report.json` conserva las respuestas literales, observaciones y retorno de cada paso; `summary.json` resume modelo/gates. Retención 30 días.

## Identidad del modelo y coste medido

- Modelo `bartowski/Qwen2.5-7B-Instruct-GGUF`, archivo exacto `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, cuantización Q4_K_M, formato ChatML, `llama-cpp-python==0.3.36` CPU.
- Revisión del repositorio HF: **8911e8a47f92bac19d6f5c64a2e2095bd2f7d031**.
- SHA256 calculado de **los pesos reales descargados y ejecutados**: **65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423**.
- Parámetros: `n_ctx=4096`, cuatro hilos CPU, `temperature=0`, `max_tokens=96`, seed 0; pesos congelados, sin ajuste de gradientes, ningún LLM remoto ni secreto.
- **32 llamadas** de generación = 32 decisiones de herramienta, **13.675 tokens de entrada**, **176 tokens de salida**, **305,65 segundos** de inferencia acumulados en CPU. No equivale a tiempo total de workflow ni coste monetario de API.

## Resultados por incidencia

| ID | Tipo de fallo real (oculto al agente) | Decisiones ejecutadas por el LLM | Resultado |
|---|---|---|---|
| M0 | Autorización | inspect → audit_access → repair_access → verify | **PASS** |
| M1 | Dependencia | inspect → audit_dependency → repair_dependency → verify | **PASS** |
| M2 | Autorización | inspect → audit_access → repair_access → verify | **PASS** |
| M3 | Dependencia | inspect → audit_dependency → repair_dependency → verify | **PASS** |
| M4 | Autorización | inspect → audit_access → repair_access → verify | **PASS** |
| M5 | Dependencia | inspect → audit_dependency → repair_dependency → verify | **PASS** |
| M6 | Autorización | inspect → audit_access → repair_access → verify | **PASS** |
| M7 | Dependencia | inspect → audit_dependency → repair_dependency → verify | **PASS** |
| **Total** | **4 acceso + 4 dependencia** | **32/32 acciones válidas** | **8/8 (100 %)** |

Todas las tareas llegaron a un `verify` con éxito observado; **no** se cuentan reparaciones parciales como éxitos. Cada tarea se resolvió en exactamente cuatro pasos, dentro del presupuesto de seis.

## Gates preregistrados

| Gate | Exigencia | Resultado |
|---|---|---|
| G0 | LLM real, SHA de pesos, revisión y ejecución verificable; sin fallback | **PASS** |
| G1 | >=6/8 y >=3/4 por cada una de dos categorías | **PASS: 8/8; 4/4 y 4/4** |
| G2 | >=90 % llamadas JSON válidas y reacción a feedback | **PASS: 32/32 válidas** |
| G3 | Trazas, costes y procedencia completos de ocho casos | **PASS** |
| G4 | Sin memoria INFINITO/MCP, producto, aprendizaje ni DEV/CONFIRM | **PASS de contrato** |

**Resultado operativo preregistrado: `GO_PREREGISTER_NEXT_INFINITO_MEMORY_STUDY`.** Esto NO significa que el agente haya aprendido o que INFINITO sea responsable de sus aciertos; **no se usó INFINITO** en el preflight.

## Interpretación crítica y próximos experimentos

**Se detectó saturación del banco:** las ocho tareas se reducen a dos secuencias, `inspect → audit_access → repair_access → verify` o `inspect → audit_dependency → repair_dependency → verify`. Aunque las superficies lingüísticas y servicios son distintos y el agente eligió las acciones en función de feedback visible, la **estructura latente es idéntica** y el modelo acertó en todas. Es un resultado fuerte de **competencia mínima en esta familia sintética sencilla**, **no** prueba de robustez general en incidencias reales ni comparación causal contra 3B.

Esta saturación implica que **F12-E NO DEBE usar más instancias de la misma plantilla para medir mejora por memoria**: el LLM sin memoria ya tiene éxito del 100% y no hay margen para observar diferencias. Se necesita una **nueva familia de tareas, con decisiones menos triviales y consecuencias que permitan aprender estrategias a partir de éxitos/fallos observados**, incluyendo:
- dependencia de información pasada no reintroducida en el enunciado, para que una memoria útil sea relevante;
- cambios de condiciones y obsolescencia, para que recuperar experiencias viejas sin juicio pueda perjudicar;
- casos nuevos fuera del entrenamiento, presupuestos de modelo y herramientas iguales en todas las variantes;
- comparación A0 sin memoria, A1 RAG episódico y A2 estrategia **inferida por el LLM** de experiencias guardadas en `SQLiteCognitiveMemoryStore`; control fuerte A1 y coste token/latencia; **ninguna regla correcta proporcionada manualmente**;
- política de promoción: no convertir automáticamente hipótesis procedentes exclusivamente de experiencias fallidas en «estrategias correctas».

Los criterios de GO de F12-D **solo autorizan diseñar y preregistrar** ese futuro estudio, no ejecutar sin nuevo protocolo o integrar el producto. Reutilizar M0–M7 para tuning después de estos resultados invalidaría su carácter de holdout. No abrir las particiones reservadas de F11.

**Conclusión:** por primera vez en F12 hemos identificado un actor congelado capaz de completar autónomamente todas las tareas de competencia **de esta familia**. La investigación puede pasar al problema central: demostrar o refutar si la memoria de INFINITO le aporta una mejora frente al mismo LLM sin memoria y con un RAG episódico competente. No se afirma aún esa mejora.

**Estado del PR:** experimento aislado, en borrador, no fusionado, `src/infinito3`, servicios MCP, secretos y datos personales sin cambios.
