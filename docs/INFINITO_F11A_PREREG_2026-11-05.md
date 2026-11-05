# INFINITO F11-A — aprender y reutilizar políticas a partir de experiencias

**Prerregistro prospectivo, 2026-11-05. ESTADO: SOLO DISEÑO; NO se ha entrenado, ejecutado DEV ni abierto CONFIRM.**
Este archivo debe quedar confirmado en Git ANTES de implementar el generador, el entrenamiento o consultar sus resultados. Ningún umbral se cambiará después de ver DEV. La confirmación es de una implementación congelada, no de un diseño cambiante.

## Pregunta causal
¿Puede un agente de INFINITO **inducir una política reutilizable** a partir de pares (observación, acción, resultado), y usarla de forma ventajosa en situaciones nuevas o tras un cambio de reglas, **sin políticas preprogramadas por tarea ni respuestas del evaluador**?

Se pretende avanzar respecto a F10-P-C: allí EventEffects alcanzó 1990/2048=97,17% en la misma familia sintética (3/4 réplicas pasaron G0/G1; r2 falló estrato S_after). F10-P-C utilizaba acciones, representación, búsqueda y física entregadas por el investigador. No es demostración de invención de herramientas ni de aprendizaje autónomo de políticas.

## Entorno y contrato de observación
- Mundos totalmente sintéticos de tareas secuenciales con recursos, costes, fallos, efectos retardados y metas variables. El catálogo de operaciones **permitidas** lo da el entorno por seguridad. La estrategia, las reglas de composición, la prioridad y la decisión de cuándo abandonarlas **no** se suministran al actor.
- El actor solo recibe descripción de objetivo, señales observables de estado e historial de resultados de acciones propias. No recibe semilla, transición interna del simulador, política óptima, recompensa futura ni etiquetas del evaluador.
- Cada episodio deja un registro con timestamp, contexto público, acción, observación posterior y recompensa observada; mismo flujo y retención para todos los brazos. Recompensas y penalizaciones son explícitas y comunes, sin reglas de shaping específicas para favorecer F11-A.
- Entrenamiento por ensayo-error **solo en el simulador**, con máximo fijo de decisiones/episodio. No ejecutar acciones sobre herramientas externas, MCP de usuario, bóveda, servicios pagos ni datos privados.

## Brazos, misma información y presupuesto
- **A0 — Reactivo**: controlador con el mismo vocabulario de acciones y feedback actual, sin experiencias persistidas ni políticas.
- **A1 — Episódico / RAG competente**: almacena los mismos episodios y recupera experiencias relevantes, pero no crea reglas generales persistentes. Misma ventana y presupuesto de recuperación.
- **A2 — Modelo de efectos/planificador**: aprende de los mismos eventos un predictor de efectos; planifica sin almacenar políticas reutilizables. Comparación principal para aislar la abstracción política frente a memorias de transición. Cuando sea comparable, incluir un F10-P-C congelado como referencia secundaria, sin adaptarlo a las familias nuevas.
- **A3 — Política aprendida**: aprende *opciones* / condiciones de activación, acción y condiciones de terminación como hipótesis revisables, usando exclusivamente éxito/fracaso de sus propias decisiones. Puede explorar y abandonar políticas pero no acceder a la verdad del simulador.
- **A4 — Oráculo informado**: óptimo computado por evaluador, diagnóstico de techo; no debe servir de entrenamiento ni ser considerado evidencia de aprendizaje.

Emparejar semillas, episodios y flujos observables por réplica; controlar por número de llamadas, límites de pasos, bytes de memoria, coste de inferencia y tiempo. No permitir que un brazo tenga exploración extra encubierta. Congelar modelo base, embedding, tamaño/latencia de ranker, versionado, formato de eventos y las capacidades de herramientas entre brazos. Si se introduce LLM real, ejecutar evaluación secundaria con un modelo idéntico y el mismo límite de tokens en A1/A2/A3, y marcarla no confirmada salvo protocolo separado.

## Transferencias y cambios de entorno
Tres conjuntos de prueba disjuntos:
1. **Intercalación conocida, instancias nuevas:** nuevas configuraciones y perturbaciones de las mismas familias, para medir generalización de estados.
2. **Composición nueva:** combinaciones de subobjetivos y efectos nunca vistas en TRAIN, sin inventar primitivas ni dar planes nuevos.
3. **Cambio de regla:** después de aprender, invertir o modificar una relación causal observable; medir si el actor detecta el deterioro y sustituye una política anticuada.

Agregar controles de invariancia semántica: nombres de operaciones permutados de forma consistente, correlaciones espurias invertidas y observaciones irrelevantes. Excluir casos que podrían resolverse por pistas del identificador de familia o de la semilla. Auditar que los casos de composición nuevos no se filtren por plantillas casi duplicadas.

## Particiones prospectivas (no abrir hasta el estado correcto)
R0..R3 con RNG independientes. Espacios de semillas propuestos y reservados para F11-A: TRAIN 61_100_000 + 10_000*r, DEV 61_200_000 + 10_000*r, CONFIRM 61_300_000 + 10_000*r; familias compuestas reservadas 61_400_000 + 10_000*r. Antes de generar, verificar que no aparecen en otros experimentos ni se solapan por hashes de caso; si hay colisión, detener y emitir nueva versión de protocolo **antes** de cualquier apertura de DEV.
- TRAIN: 512 mundos base/r, hasta 30 episodios por mundo; no ejemplos de política óptima.
- DEV: 128 mundos base/r por cada uno de los tres conjuntos; exclusivamente ajuste de hiperparámetros, selección de implementación y pruebas de integridad.
- CONFIRM: 256 mundos base/r por conjunto, cuatro réplicas; se abre **una sola vez** tras fijar commit de código, modelos, costes, presupuesto y umbrales. Evaluar resultados vinculados por mundo base, no como observaciones independientes.

## Variables registradas
Éxito por mundo; retorno acumulado normalizado respecto a rango público de tareas; coste por éxito; pasos hasta mejora; recuperación tras cambio de regla (área bajo curva de 20 episodios y recompensa de los últimos cinco); tasa de políticas inducidas reutilizadas, abandonadas y revisadas; errores de sobre-generalización; violaciones de condiciones de seguridad; bytes, latencia y tokens/calls. Guardar trazas completas y SHA de generador/actor/snapshots. Reportar siempre resultados por réplica, familia y estrato.

## Hipótesis / gates congelados
Auditor independiente sin importar al actor ni código de entrenamiento, bootstrap de 2.000 remuestreos agrupados por mundo base con IC95 y contrastes pareados. Resultado global PASS requiere >=3/4 réplicas para G1,G2,G3,G4, y G0/G5 universalmente.

- **G0 Integridad y seguridad (obligatorio):** separación de TRAIN/DEV/CONFIRM, auditoría de no fuga de estado interno/oráculo, mismos budgets y acción válida; cero violaciones de límites de acción. FAIL técnico => INCONCLUSIVE, no ajustar semillas después de abrir resultados.
- **G1 Ventaja de política reutilizable:** A3 supera A2 y A1 en éxito de instancias nuevas al menos +10 puntos porcentuales por réplica, y el límite inferior del IC95 pareado de A3-A2 >0.
- **G2 Transferencia composicional:** A3 supera A2 por >=5 puntos en composiciones nuevas; IC95 inferior >0; A3 no pierde >5 puntos respecto a A2 en ningún estrato preregistrado.
- **G3 Adaptación al cambio:** en episodios postcambio A3 mejora A2 en >=10 puntos de retorno normalizado por mundo base en área acumulada de 20 episodios, IC95 inferior >0, y reduce el uso de políticas antiguas tras evidencia contradictoria.
- **G4 Causalidad de la política:** ablación solo de políticas aprendidas de A3 (manteniendo experiencias y predictor) empeora éxito >=10 puntos; la política correcta elegida debe ser trazable a experiencia observable previa, no a plantilla del simulador.
- **G5 No regresión operativa (obligatorio):** ningún acceso a memoria personal o herramientas reales, pérdidas de datos/forget 0, coste total A3 <=1,25x A2 con presupuesto de inferencia idéntico y logs de operaciones reproducibles.

Un resultado inferior a cualquier gate => **F11A_FAIL**; fallos de instrumentación o fuga => **F11A_INCONCLUSIVE**. No reinterpretar un control peor como mejora si también viola los presupuestos. Una réplica fallida queda publicada íntegra.

## Decisiones predefinidas y límites
- PASS: evidencia acotada de aprendizaje y reutilización de políticas en familias sintéticas; preparar estudio separado con señales textuales/herramientas reales sintéticas en modo shadow. **No integrar directamente en producto**.
- FAIL: analizar qué gate falló; archivarlo y plantear hipótesis nueva sin reciclar semillas confirmatorias.
- INCONCLUSIVE: solucionar solo defecto técnico y re-prerregistrar si cualquier conjunto reservado ya fue observado.
- Ni un PASS permite afirmar inteligencia general, invención autónoma de nuevas operaciones, transferencia a dominios abiertos o autonomía segura en producción.

## Orden de implementación
1. Revisar seguridad de project scopes en PR #85 sin mezclar su código con este estudio.
2. Implementar harness F11-A en `observatory/`, con unit tests e instrumentos para los cinco brazos.
3. Validar presupuestos, invariantes, ceguera y auditor en datos puramente de prueba (no DEV/CONFIRM).
4. Abrir DEV, congelar artefactos/versiones, realizar preflight ciego, abrir CONFIRM una sola vez y publicar todos los estratos y fallos.

**Este documento es un protocolo, NO son resultados de un experimento.**
