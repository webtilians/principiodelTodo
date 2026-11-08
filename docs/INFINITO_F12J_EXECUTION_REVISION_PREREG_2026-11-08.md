# F12-J — Ejecución y revisión ante fallos de una política válida

**Prerregistro prospectivo, 8/11/2026. Independiente de F12-I: se congela antes de escribir código y de observar resultados F12-J.**

## Pregunta científica y separación de capacidades
F12-H generó un plan potencialmente ejecutable que el actor no siguió. F12-J no vuelve a medir inducción: da al mismo Qwen2.5-7B una política **válida obtenida de sondeos públicos**, y pregunta (1) si la ejecuta, (2) si cambia a una alternativa observada cuando el plan queda invalidado por un fallo explícito. Así se separa inducción vs ejecución vs revisión. Comparar un plan recordado y las mismas evidencias con distintos grados de señalización de seguimiento; **no confundir ayuda del prompt con aprendizaje autónomo**.

Mismo modelo/hash/pesos/backend que F12-F/H/I (Qwen2.5-7B Q4_K_M, SHA256 `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, llama-cpp-python 0.3.36, temperatura 0). Sin fine-tuning. SQLite INFINITO auténtica en RAM, solo datos sintéticos. Ningún ajuste con H/Q, S/H/V/W/U/Y/I ni con resultados futuros de J.

## Mundo, herramientas y entrenamiento público
Seis herramientas opacas `u0..u5` más `verify` implementan LOAD, ENABLE, READ, FLIP_PRIMARY, FLIP_BACKUP, HOLD. El catálogo público declara solo acciones y mensajes `value`, `ack`, `error`, `verified`. El registro y bit oculto son privados. La meta heldout es invertir el bit oculto y verificar.

Ocho asignaciones de roles fijas, evaluator-only, por `u0..u5`:
- J0 LOAD ENABLE READ PRIMARY BACKUP HOLD
- J1 ENABLE READ PRIMARY BACKUP HOLD LOAD
- J2 READ PRIMARY BACKUP HOLD LOAD ENABLE
- J3 PRIMARY BACKUP HOLD LOAD ENABLE READ
- J4 BACKUP HOLD LOAD ENABLE READ PRIMARY
- J5 HOLD LOAD ENABLE READ PRIMARY BACKUP
- J6 ENABLE BACKUP LOAD PRIMARY HOLD READ
- J7 READ HOLD PRIMARY ENABLE BACKUP LOAD

El investigador ejecuta, por mundo, sondeos parciales **separados**, nunca el plan objetivo: LOAD con bit público 0/1; ENABLE→READ con oculto 0/1; LOAD→PRIMARY con público 0/1; LOAD→BACKUP con público 0/1; READ sin ENABLE (error); LOAD→HOLD. Se almacenan las diez trazas *públicas* en SQLite. Un extractor **determinista desde esas trazas** identifica las dos operaciones que invierten ambos valores, la secuencia ENABLE→READ y el plan principal [enable,read,primary,verify]. La política válida se establece por preflight mecánico, sin consultar verdad privada para inferirla, e incluye que hay una alternativa observada. Si el extractor falla, INVALID: nunca entregar un plan experto a mano para rescatar el experimento.

## Intervención y evaluación
Se evalúan dos condiciones por cada uno de ocho mundos, cuatro brazos × dos condiciones × ocho mundos = **64 episodios**:
- ESTABLE: PRIMARY y BACKUP operan según lo observado.
- CAMBIO: en la primera aplicación de PRIMARY, y desde ese momento, devuelve `{"error":"DISABLED_TOOL"}`. BACKUP sigue disponible; esto no se anuncia hasta que el actor lo ejecuta. El objetivo y los efectos restantes no cambian. Esta es una mutación fija, no una respuesta elegida para castigar al agente.
Cada evaluación usa dos bits públicos/ocultos por mundo precongelados, balanceados, sin repetir instancias de F12-I/H. Horizonte 7, un episodio por mundo×condición×brazo, estados nuevos y mismo orden de brazos rotado.

Los brazos reciben **idénticos** objetivos públicos, herramientas, política primaria válida, registro de experiencias públicas (incluyendo BACKUP) y presupuesto de contexto 3500 caracteres:
- B0: actor libre, política escrita como referencia, historial observado. Decide acciones sin puntero de seguimiento.
- B1: actor con índice/cursor calculado *solo de confirmaciones públicas*, señalando cuál es el próximo paso esperado del plan; sin consejo explícito sobre errores.
- B2: cursor idéntico a B1 y **una instrucción de reparación predefinida**: si la operación indicada devuelve DISABLED_TOOL, consultar la alternativa de inversión documentada por las experiencias. Es un andamiaje prescrito, no habilidad autónomamente aprendida; si mejora, hablar de obediencia y replanificación inducida por instrucciones.
- B3: ejecutor programado transparente del plan y de la alternativa, a partir de los efectos observados, sin ningún LLM. Control superior que prueba solucionabilidad con la misma evidencia y horizonte.

Los cursor/status son calculados de eventos observados y se muestran con SHA de prompt. B1 y B2 difieren **solo** en la instrucción adicional de revisar; registrar ese cambio y evitar atribuir diferencias a memoria. Ningún brazo recibe el rol oculto ni los resultados contrafactuales. El bit del evaluador nunca pasa al actor ni a su memoria.

Métricas por mundo×condición: verificación positiva, seguimiento del paso, primera divergencia, repeticiones, cambio a BACKUP tras error, llamadas, tokens, tiempo, abandono. Preflight B3 debe lograr 8/8 en ambas condiciones antes de ejecutar cualquier LLM; es un test mecánico de contrato, no datos de rendimiento de Qwen.

## Gates anteriores a datos
- J0 integridad: modelo y hash correctos, 10 sondeos públicos/mundo, 8 mapas precongelados, 64 evaluaciones y episodios no contaminados; cero acceso a verdad privada del actor; fallos => INVALID/BLOCKED.
- J1 entorno solucionable: B3 alcanza 8/8 estable y 8/8 cambio, con BACKUP efectivamente observado antes; si no => INVALID.
- J2 ejecución: B1 mejora B0 en ESTABLE al menos 3 de 8 casos, y B1 ≥6/8; si no, no hay prueba de que el cursor mejore obediencia.
- J3 revisión: B2 supera B1 en CAMBIO al menos 3/8, y B2 ≥6/8; contar como recuperación únicamente un BACKUP realmente usado **después** del error público, no casual.
- J4 no sacrificio: B2 ≥B1 en ESTABLE, sin gastar >25% más llamadas ni tokens que B1 en ESTABLE. Si falla, no reivindicar una mejora global.
- J5 atribución: inspeccionar las trazas literales y explicar desviaciones y posibles coincidencias; no reclamar política aprendida ni habilidad OOD entre familias.

Un PASS solo permitiría diseñar confirmación con mundos nuevos y comparadores adicionales. No fusionar producto, no reinterpretar resultados de F12-I, no ajustar según J0..J7 consumidos. Falta de J2/J3 => NO-GO de eficacia; estabilidad sin cambio no prueba revisión.
