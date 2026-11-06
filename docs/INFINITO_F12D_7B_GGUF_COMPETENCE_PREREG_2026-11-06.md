# INFINITO F12-D — Preflight independiente de competencia con LLM congelado de 7B

**Prerregistro prospectivo — 6 de noviembre de 2026.** Este documento queda versionado **antes del código F12-D y antes de abrir cualquier caso M0–M7**.

## Pregunta y decisión que cambiaría

F12-C terminó con Qwen2.5-3B-Instruct Q5_K_M: **4/6 tareas** pero solo **1/3 de autorización**, por tanto **NO-GO** respecto a la competencia mínima preregistrada. Falló en dos casos por no verificar tras reparar. ¿Un LLM 7B real congelado puede resolver de manera autónoma y verificable suficientes incidencias nuevas como para servir de actor en un futuro estudio de aprendizaje mediante la memoria de INFINITO?

Este estudio **NO prueba memoria, aprendizaje o transferencia**; solo un **preflight de competencia** antes de decidir si abrir otra línea. No reutilizar K0–K3 (F12-B), L0–L5 (F12-C), T/P (F12-A), ni semillas F11 DEV/CONFIRM.

## Modelo y ejecución congelados

- Modelo: **Qwen2.5-7B-Instruct** abierto, exactamente el GGUF `Qwen2.5-7B-Instruct-Q4_K_M.gguf` de `bartowski/Qwen2.5-7B-Instruct-GGUF`, cuantización Q4_K_M, archivo documentado ~4,68 GB. El peso real se descarga una sola vez, se registra revisión SHA de Hugging Face y **SHA256 completo del archivo**, se congela el modelo durante todas las tareas. Sin reentrenamiento ni actualización de pesos.
- Motor: `llama-cpp-python==0.3.36` CPU con wheel precompilada para `ubuntu-24.04`, `huggingface_hub==0.29.3`; `n_ctx=4096`, `n_threads=4`, `temperature=0`, `top_p=1`, `max_tokens=96`, `chat_format=chatml`, `seed=0`. No reemplazar modelos si descarga/ejecución falla: marcar **BLOCKED**. No consumir archivos privados ni claves API.
- **El mismo prompt de sistema, catálogo, lógica de observación, parseo JSON y herramientas de F12-C** (se copia sin modificar), salvo identificadores, nombres y reportes de los casos nuevos. No modificar prompt ni precondiciones después de ver resultados de F12-C o F12-D.
- Seis herramientas **in-memory** del simulador F12-C: `inspect`, `audit_access`, `audit_dependency`, `repair_access`, `repair_dependency`, `verify`. Una reparación exige auditoría previa correcta; éxito únicamente cuando `verify` confirma recuperación. Máximo **seis acciones por caso**; una respuesta inválida consume la acción, no se repara por código.
- Ocho incidencias de prueba `M0..M7` con vocabulario y servicios nuevos: **cuatro autorizaciones y cuatro dependencias**. Orden fijo M0..M7; el evaluador conoce la causa, pero el actor solo el objetivo y los resultados públicos elegidos. Las tareas son **nuevas instancias de la misma familia sintética**, no OOD estructural.

## Controles, criterios cerrados y resultados

- Referencia histórica descriptiva: F12-C 3B obtuvo 4/6, acceso 1/3 y dependencia 3/3. **No es A/B causal**, porque F12-D utiliza tareas diferentes y cuantización distinta. No usar como estimación de mejora debida al tamaño.
- **G0 procedencia:** todas las decisiones generadas por el LLM 7B cargado realmente, modelo SHA, revisión HF, versión de motor y número de llamadas registrados; ningún script/heurística que elija acciones.
- **G1 competencia:** `>=6/8` casos con verificación positiva, incluyendo **>=3/4 en autorización y >=3/4 en dependencias**. No basta con haber reparado si no termina con `verify`.
- **G2 herramientas y feedback:** `>=90%` de JSON válidos para una de seis herramientas, y al menos una reacción a resultado observado (incluidas señales de fallo); sin resultados ocultos.
- **G3 auditoría:** 100% de ocho casos con respuestas textuales, observaciones, resultados, retornos, tokens, latencia, presupuestos, SHA de prompt y SHA del modelo. Guardar artefacto de GitHub Actions.
- **G4 aislamiento:** cero memoria de usuario, memoria INFINITO, MCP, modificación de `src/infinito3` o productos; sin DEV/CONFIRM, sin entrenar políticas.

**GO únicamente si pasan G0–G4:** escribir un NUEVO prerregistro de estudio de aprendizaje de experiencias con memoria INFINITO (RAG episódico vs estrategia sintetizada, mismos tokens) sobre tareas nuevas y particiones independientes. **NO-GO** si falla G1/G2 aunque pase el CI técnico. **BLOCKED** si no se llega a ejecutar el modelo real por infraestructura. No ajustar tras resultados ni repetir M0–M7 para optimizar.

## Riesgos y límites

La cuantización Q4 puede reducir precisión respecto a pesos sin cuantizar; una VM estándar puede tener limitaciones de RAM, red o cuota; registrar sin sustituir por otro actor. La muestra de ocho casos es pequeña, y la familia simple permite ciertas rutinas preexistentes en el LLM: aun pasando G1 no demostraríamos aprendizaje ni habilidades en producción. La subida de 3B a 7B junto al cambio de banco **no aísla causalmente el tamaño**.

**Estado en este commit: solo diseño; ninguna inferencia F12-D ejecutada.**
