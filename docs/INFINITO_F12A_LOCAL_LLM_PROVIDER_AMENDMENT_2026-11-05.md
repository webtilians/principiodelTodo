# F12-A — Enmienda prospectiva de proveedor para el piloto local
**Fecha 2026-11-05. Confirmada ANTES de implementar el proveedor local y de ejecutar ninguna inferencia local.**

El experimento F12-A original (prerregistro commit 04e90b1a) intentó GitHub Models con modelo openai/gpt-4o-mini y token efímero. GitHub Actions run 42052897453 falló al intentar leer JSON; el run de diagnóstico 42053094521 confirmó: **HTTP 200, Content-Type text/plain, cuerpo de 4 bytes**, hash prefijo 19d8c9921b99. No se ejecutó ninguna inferencia verificable del LLM, ni TRAIN, ni evaluación, ni se generó una experiencia real. Los tests con FakeDecisionBackend no son resultados científicos.

## Cambio permitido antes de datos locales
Se sustituye el proveedor para **este único piloto nuevo** por el LLM abierto **Qwen/Qwen2.5-0.5B-Instruct** (Hugging Face), ejecutado con transformers/torch **localmente en el runner GitHub Actions CPU**, sin API key de OpenAI ni costes de inferencia de proveedor. Modelo fijo para A0/A1/A2 y la síntesis, `do_sample=False`, pesos en modo `eval()`, sin gradient steps ni cambios de pesos. Registrar modelo real, versión de transformers/torch, revisión HF y hashes de checkpoint si los aporta el hub, input/output tokens, tiempo, CPU/RAM. Sin acceso a Hugging Face o si la carga falla, registrar `BLOCKED_LOCAL_MODEL`, no reemplazar por respuestas predeterminadas ni mock. Este modelo pequeño puede ser incapaz de usar herramientas; eso limita los resultados pero NO autoriza ajustar prompts una vez vistas las evaluaciones.
Se mantiene la física y casos sintéticos del protocolo original, los tres brazos y el presupuesto; no comparar resultados del proveedor bloqueado con el nuevo.
Los contratos de tests técnicos continúan utilizando FakeDecisionBackend solo para comprobar integridad, nunca como LLM real.
La selección de Qwen2.5-0.5B-Instruct es por tamaño viable en CPU, descarga pública y capacidad básica de emitir JSON; modelo pequeño, **no representante de todos los LLMs**.
**Gates P0–P2 sin cambios:** una experiencia real requiere salida de un modelo efectivamente ejecutado y registrado; si el agente falla todas las tareas, reportar saturación inferior y no inferir ausencia general de capacidad de aprender.

Al firmar este texto **no se han abierto tareas de evaluación con el nuevo proveedor**.
