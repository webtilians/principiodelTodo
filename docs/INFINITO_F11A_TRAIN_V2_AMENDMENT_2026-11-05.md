# F11-A — enmienda prospectiva de TRAIN v2 (5 noviembre 2026)

**Antes del código v2 y ANTES de generar nuevos datos**. El prerregistro de F11-A está en `docs/INFINITO_F11A_PREREG_2026-11-05.md` (PR #86 fusionado). El TRAIN v1 de PR #87 está archivado: el entorno quedó saturado (A1=A2=A3=4095/4096 en probes in-sample) y A1 agregó valores por acción, por lo que era un falso baseline episódico.

## Intervención metodológica autorizada
Este documento registra una modificación deliberada del simulador y el control A1 **antes de DEV o CONFIRM**:

- **Mundos de acciones permitidas opacas:** A..F, tareas con objetivos públicos, dos prerrequisitos (carga y material), ruta, costes de fallo, penalizaciones temporales y ruido; las operaciones cambian de significado al producirse una ruptura no anunciada en mitad de los episodios de TRAIN.
- **Crédito temporal:** preparación mediante operaciones cuyo efecto aparece después de 1–2 pasos. El agente observa consecuencias, no las reglas de transición ni las colas internas de operaciones pendientes. Se registra retraso y versión de reglas **solo para auditoría**, nunca como input del actor.
- **A1 competidor episódico de verdad:** conserva trazas completas (sin estadísticas persistentes por estado/acción), recupera vecinos de experiencias reales en cada decisión y usa la siguiente acción del episodio recuperado. Los presupuestos de consulta/cantidad de datos se documentan; no hay una política A1 agregada.
- **A2 efectos:** modelo de transiciones estocásticas observables con búsqueda acotada y memoria de efectos; sin almacenar procedimientos reutilizables.
- **A3 opciones:** extrae secuencias condicionales únicamente de experiencias propias exitosas; reutiliza, penaliza y descarta opciones tras contradicciones, manteniendo el mismo modelo de efectos que A2 para fallback. El colector de experiencias es compartido por todos (incluye una pequeña proporción de episodios seleccionados por A3, proporcionados igualmente a A1 y A2).
- **A0:** controlador reactivo sin memoria. **A4:** heurística informada solo como techo diagnóstico, no entrenador; no afirmar optimalidad matemática.
- **No hay entrenamiento de LLM ni gradientes**: se entrena aprendizaje tabular/eventos y opciones de agente sintético. No se integra en producto.

## Igualdad, perturbaciones y criterios de validez
- Cuatro réplicas r0..r3, únicamente semillas TRAIN `61_100_000 + 10_000*r + i` con `0<=i<512`; hasta 30 episodios/mundo. Un máximo fijo de decisiones por episodio (aquí 12).
- Experiencias recolectadas con la **misma política exploratoria compartida**, misma cantidad e idénticas trazas para A1/A2/A3. La parte elegida por A3 se contabiliza como coste de exploración común, no regalado a A3. No usar puntuaciones del evaluador para actualizar.
- Probes TRAIN sobre mundos vistos, en fase de reglas antigua y cambiada. Son diagnósticos in-sample contaminados; registrar éxito, retorno, acciones, cambios de opción, abortos, bytes de episodios y opciones.
- Auditoría de origen de efectos retardados, de ruptura real de reglas, de imposibilidad de acceso a roles/semilla del simulador, ausencia de memoria personal, y hash de experiencias.
- La comparación con v1 es solo descriptiva: cambia la física y el control A1, **no** equivale a una mejora causal del agente.
- No usar ni abrir ninguna semilla DEV `61_200_000+`, CONFIRM `61_300_000+`, ni combinaciones reservadas `61_400_000+`. Ninguna cifra de TRAIN se usará como PASS científico.
- Antes de DEV revisar si todos los brazos A1/A2/A3 tienen costes comparables (G5), adaptación realmente inducida (G3), y un entorno no saturado que permita inferencia. Si alguno falla, NO-GO; conservar todos los hallazgos.

## Registro de cambios respecto al documento inicial
F11-A original preveía efectos retardados/cambios, pero no fijaba mecanismo de transición, acción ni recuperación. v1 los omitió. Este protocolo hace explícita la operacionalización prospectivamente. No alterar umbrales G0–G5 del documento original después de observar DEV. Si esta versión tampoco resulta científicamente válida, mantener DEV/CONFIRM sellados y publicar otro protocolo prospectivo separado.

**Estado en este commit: documento solamente. Ningún entrenamiento v2, ninguna muestra DEV/CONFIRM.**
