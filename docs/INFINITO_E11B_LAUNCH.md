# INFINITO E11-B — Lanzamiento único de comparación real

**Fecha:** 1 noviembre 2026.

Protocolo preregistrado antes de la implementación en `c349049d2e3c1bef98fc1238a6f697ca4516b90c`.
[Tests de integración y regresiones E11-B](https://github.com/webtilians/principiodelTodo/actions/runs/41319242377) **SUCCESS**.
[PR #33](https://github.com/webtilians/principiodelTodo/pull/33) borrador, sin modificación del núcleo de producción.

**Único cambio experimental:** política que mantiene cuatro recuerdos de un RAG común.
Mismo LLM real `gpt-5.6-luna`, embeddings reales `text-embedding-3-small`,
misma representación, presupuesto máximo 260 tokens de evidencia, orden temporal.

Este fichero habilita **una sola secuencia**: preflight de proveedor real, congelación de cinco aprendices de E11-A, cuatro parejas DEV nuevas `4100000+`, validación integral de API/slots y, solo entonces, confirmación `4200000+` con 15 pares predictivos y 15 de demanda invertida. Controles uniformes `4400000+` solo offline y negativos de abstención con API real.

El criterio de avance se considera fallido si la política aprendida no supera el mejor baseline práctico **o** si pierde más de 5 puntos frente a reservorio cuando cambia la demanda. **No ajustar después de observar el resultado.** Si no cumple ambos, cerrar E11 y continuar con E12.

Los identificadores confirmatorios E10, E11-A y F9 permanecen sin abrir/reutilizar en este proceso.
