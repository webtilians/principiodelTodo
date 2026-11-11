# INFINITO Cognitive Diary P0 — especificación retrospectiva de la demostración

**Estado metodológico:** documento técnico elaborado durante la implementación; NO es un preregistro prospectivo ni una confirmación científica independiente.

**Objetivo:** demostrar que un diario sintético de un único día puede reconstruirse sin discrepancias, registrar decisiones/hipótesis/pruebas procedentes de fuentes concretas, auditar una omisión de un agente subordinado, conservar la evolución de una reflexión y dejar una pauta solo propuesta hasta decisión humana.

**Comparador:** reconstrucción determinista del guion en un proceso limpio, sin memoria persistente ni LLM. El diario guardado debe coincidir bit a bit con un segundo guion nuevo y `--replay` debe aceptar solo el guion exacto. No es un control científico de mejora frente a RAG o modelos sin memoria; esa comparación requiere P1 separado.

**Datos:** un día totalmente inventado el 15 de enero de 2026, con tres informes de herramientas ficticias, dos de ellos verificados, uno no verificado. No contiene recuerdos de la cuenta personal ni claves de INFINITO.

**Criterios de aprobación técnica:** ocho tests, incluidos reproducción idéntica, cadena de procedencia, detección de manipulación, fuente omitida en una supervisión que consulta originales, revisión vinculada a reflexión anterior, pauta que no puede aplicarse sin autorización humana, bloqueo de cruce de tenant/proyecto, olvido transitivo y ausencia de coste/modelo real. La evaluación científica sobre aprendizaje debe seguir marcada como **NO EVALUADA** independientemente de PASS técnico.

**Decisión posterior:** si P0 pasa, valorar un P1 de modelos reales *solo* con datos sintéticos y límites monetarios explícitos, sin despliegue ni acceso a recuerdos cifrados sin sesión y consentimiento.
