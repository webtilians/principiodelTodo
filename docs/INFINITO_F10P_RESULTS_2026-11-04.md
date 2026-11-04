# INFINITO F10-P — resultados originales: efectos desde respuestas PÚBLICAS de herramientas

**4 noviembre 2026. Resultado científico preregistrado: F10P_OBSERVABLE_EFFECTS_PASS. Auditoría técnica completa (todos los 4 entrenamientos y auditor independiente SUCCESS).** El resultado se obtiene exclusivamente dentro de la MISMA familia sintética de mundos de F10-N; no hay confirmación independiente en la partición 28300000+, que sigue SELLADA.

## Cronología, preregistro y preservación

1. [Prerregistro prospectivo F10-P](INFINITO_F10P_PREREG_2026-11-04.md), **commit 14a44d4ce3c6fd4314a1c15c68318c401175f99c**, anterior a todo código/dato F10-P. Precisión prospectiva del umbral de identificación de una transición (mínimo 10 observaciones, consistencia modal >=90%, abstenerse en efecto desconocido): **commit f30af41d7776887090d63cd90b26f3e0bcdfe5b3**, también ANTES del primer archivo de código del learner.
2. **Un único lanzamiento original**: [GitHub Actions run #41888234405](https://github.com/webtilians/principiodelTodo/actions/runs/41888234405), commit de lanzamiento **f91abc2fb26c6bcd32a820d04b284688bc190748**. Los 4 trabajos TRAIN/DEV/OOD y el auditor independiente terminaron SUCCESS. No hubo cambios a modelos ni umbrales tras abrir los casos.
3. Modelos/habilidades **ORIGINALES F10-L**: cinco SHA por cada una de 4 réplicas, obtenidos del [run #41685499939](https://github.com/webtilians/principiodelTodo/actions/runs/41685499939), verificados antes/después. F10-N, F10-N-C (PASS), F10-O (FAIL) y producto permanecieron sin modificación. PR #71 DRAFT; sin merge a master.
4. Artefactos originales del run #41888234405: f10p-original-public-events-0 a -3 (logs TRAIN públicos, 7 memorias por réplica SHA-congeladas ANTES de DEV/OOD, predicciones completas) y f10p-independent-public-event-science (JSON completo de la auditoría, intervalos y gates). La auditoría separada NO importó Torch, generador de mundos ni learner.

## Pregunta científica exacta

**F10-O (FAIL)**: el GRU con solo respuesta/recompensa terminal obtuvo 55,18% OOD, mientras un control tabular con estados internos consiguió 96,73%; pero la arquitectura y observabilidad diferían simultáneamente. No era posible atribuir esa diferencia solo a información privada.

F10-P compara un **MISMO EventEffects factorizado**, idéntico planificador, conteos, umbrales y programas reales, mientras cambia exclusivamente la máscara de salida P/S/R/X. Las herramientas se definen para devolver al ejecutarse P→valor público cargado, S→ACK de selección sin índice, R→valor leído (solo tras ejecutar READ) y X→valor transformado. Estas salidas se derivaron para el entorno experimental de su ejecutor y son respuestas EXTERNAS bajo un contrato de interfaz diseñado, NO observaciones del mundo natural sin diseño. El aprendiz E100/E50/E0/noX/shuffleX **no recibe el registro interno before/after, index winner, contenidos F4, true label, máscara de selección ni read_observed privado**. La auditoría reconstruyó sus conteos desde los logs de SOLO op/output público.

EventEffects reconstruye P(bit público→valor devuelto), X(valor observable previo→valor devuelto), S ACK y R RETURN desde eventos realmente ejecutados. Fija una abstracción de valor de trabajo y ACK a partir de la interfaz, y explora los 85 programas predefinidos usando las dos hipótesis del bit que devolvería R (0/1) y la meta SOURCE/FLIP aportada por el investigador. Nunca se le proporcionó directamente que X invierte, ni qué programa era óptimo. **SRX nunca fue ejecutado durante TRAIN**.

## Diseño y presupuesto

TRAIN 28100000+, DEV 28200000+, OOD 28400000+ (MEMORY1 nunca vista), CONFIRM 28300000+ SELLADA. Cuatro réplicas, tres contextos TRAIN (PUBLIC0, PUBLIC1, MEMORY0), 3072 casos/r ×16 pases ×2 intentos = **98.304 programas realmente ejecutados por réplica**, el **MISMO HISTORIAL** compartido por todos los brazos. Total **393.216 episodios reales**, no siete veces ese número. 84 cadenas permitidas TRAIN, SRX excluida.

Cada caso OOD pertenece a 128 mundos/r con cuatro variantes vinculadas; OOD 512 casos/r, 2048 casos/brazo en las 4 réplicas. 2000 remuestreos bootstrap de 128 mundos base/r (las cuatro variantes de un mundo permanecen juntas) con semillas y umbrales previamente fijados.

| Brazo | Lo que el MISMO learner recibe durante entrenamiento |
|---|---|
| E100 | 100% de salidas públicas P/S/R/X |
| E50 | 50% de salidas al azar OMITIDAS (missing, nunca interpretadas como cero) |
| E0 | 0% de salidas intermedias, solo feedback terminal |
| E_noX | P/S/R observables, pero NINGÚN output X |
| E_shuffleX | P/S/R intactas; bits devueltos por X aleatorizados independientemente |
| B_reward | Baseline separado de puntuaciones aditivas por programa/contexto |
| I_privileged | Control diagnóstico EffectMemory que SÍ recibe before/after interno original |

**Importante**: E100 y los otros cuatro E usan exactamente el mismo modelo y búsqueda. B e I son controles de mecanismos distintos, no comparaciones del mismo algoritmo.

## Resultados OOD originales y programa elegido

| Réplica | E100 | E50 | E0 terminal | E_noX | E_shuffleX | B_reward | I_privileged | SRX elegido por E100 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | **100%** | **100%** | 50,39% | 50,39% | 50,39% | 51,17% | 100% | 512/512 |
| 1 | **100%** | **100%** | 57,81% | 57,81% | 57,81% | 52,34% | 100% | 512/512 |
| 2 | **90,43% (463/512)** | **90,43%** | 47,66% | 47,66% | 47,66% | 49,22% | 90,43% | 512/512 |
| 3 | **100%** | **100%** | 49,22% | 49,22% | 49,22% | 50,78% | 100% | 512/512 |
| **Pooled descriptivo** | **97,61% (1999/2048)** | **97,61%** | **51,27% (1050/2048)** | **51,27%** | **51,27%** | **50,88% (1042/2048)** | **97,61%** | **2048/2048** |

**Observación decisiva:** con solo salidas de herramientas, E100 descubre y elige **S→R→X** en TODOS los 2048 casos de la combinación MEMORY1 jamás entrenada, pese a **cero ejecuciones de SRX en TRAIN**. Bajo el mismo modelo, ocultar completamente las salidas, ocultar específicamente X o corromper los outputs X hace que el planificador no identifique la transformación con >=90% de consistencia y seleccione el programa vacío. Esto verifica la necesidad de feedback adecuado, **pero también refleja la fuerte regla de confianza y la abstracción de estado diseñadas**, no una identificación causal irrestricta.

**E50 igual a E100**: al omitir ~50% de los resultados de herramientas queda todavía un volumen abundante de observaciones deterministas: aproximadamente 131.4–131.7 mil eventos observados por réplica en E50 frente a 263 mil en E100. Por ello E50 identifica igualmente los efectos. **No se ha probado tolerancia a muestras pequeñas, pérdidas dependientes del resultado o errores sistemáticos de herramientas.**

Réplica 2: E100 DEV MEMORY0=**87,50%**, inferior al umbral G0 individual del 90%; las otras tres réplicas cumplen los tres DEV al 100%. En OOD r2=90,43%, IC95 por mundo [87,50%,93,16%], y S_after=71,09%, próximo al límite de 70%; no ocultar la fragilidad. La calidad del selector F10-L original congelado afecta también al control I, que da idéntica precisión. Los otros 3 CIs por mundo son [100%,100%].

### Contrastes preregistrados y causalidad

**E100−B recompensa**: +48,83pp (r0), +47,66pp (r1), +41,21pp (r2), +49,22pp (r3), todos con límite inferior IC95 por mundo claramente >0 (mínimo r2 +36,13pp).

**E100−E0 / E_noX / E_shuffleX**: respectivamente alrededor de +42,19 a +50,78pp entre réplicas; los tres contrastes G4 pasan en las 4 réplicas con IC inferiores >0. No introducir información privilegiada en E100: los eventos de herramienta contienen únicamente nombre de la operación y resultado público y se observa el valor R solo **DESPUÉS** de READ.

**Intervenciones causales sobre inferencia congelada**:
- Barajar los contenidos de F4 con mismo plan: cae E100 entre **43,36 y 49,41 puntos** según réplica, con IC inferior >0.
- Desactivar X en ejecución con mismo plan: cae E100 entre **80,86 y 100 puntos**, con IC inferior >0.
- La precisión del control con before/after I_privileged no mejora respecto E100 en esta familia. Esto demuestra **suficiencia del feedback público diseñado** para esta tarea, NO que feedback natural sin contrato equivalga a estado privilegiado.

### Puertas científicas prospectivas

| Gate | Resultado |
|---|---|
| G0 E100 competencia conocida DEV >=90% en los tres contextos | **PASS 3/4** |
| G1 SRX nunca entrenada y OOD/CI/estratos con E100 | **PASS 4/4** |
| G2 dependencias causales de F4 y X en ejecución | **PASS 4/4** |
| G3 ventaja sobre puntuaciones B con mismas experiencias | **PASS 4/4** |
| G4 necesidad causal de outputs intermedios, X observable y no aleatorio, mismo algoritmo | **PASS 4/4** |
| G5 E50 precisión OOD>=80% (diagnóstico) | **PASS 4/4** |
| G6 I_privileged precisión OOD>=80% (diagnóstico) | **PASS 4/4** |

**SCIENCE STATUS: F10P_OBSERVABLE_EFFECTS_PASS**, conforme al contrato G0∧G1∧G2∧G3∧G4, sin cambiar ninguna regla al ver OOD. **No es una confirmación independiente**; CONFIRM 28300000+ permanece SELLADA.

## Implicaciones reales e integración

F10-P corrige una laguna de F10-O: en este simulador, con **el mismo planificador/modelo** y las mismas interacciones, el resultado depende de que los eventos públicos sean informativos, especialmente el output de la transformación X. **No demuestra que una red genérica infiera estados latentes a partir de solo recompensa final** (F10-O permanece FAIL). Tampoco demuestra descubrir la semántica de salida, tipos de argumento o representación abstracta de herramientas desconocidas: esas interfaces fueron predefinidas; el propio planificador sigue explorando solo 85 programas de hasta tres operaciones y sabe la meta de paridad XOR/FLIP.

El contrato de herramienta de laboratorio **expone directamente el nuevo valor después de P/R/X**. Muchas herramientas reales no devuelven así su estado o su efecto. No extrapolar éxito a MCP de memoria ni a razonamiento sobre mensajes de usuario. No se han probado concurrencia, temporales ruidosos, errores de herramientas, feedback retrasado, olvido de datos derivados, adversarios, persistencia multiusuario ni operaciones nuevas. No utilizar información real de usuarios ni la bóveda.

**NO-GO para integrar un planificador autónomo aprendido en el núcleo de INFINITO.** Si una confirmación nueva revalida F10-P con las memorias originales congeladas, sería razonable un siguiente estudio de **salidas reales de herramientas en modo shadow, con esquema explícito y datos totalmente sintéticos**, sin escritura a memoria, borrado ni herramientas sensibles. Solo después, con aislamiento, coste real y evaluación contra el producto actual bajo igual presupuesto de tokens, considerar una funcionalidad experimental feature-flagged off por defecto.

**STOP F10-P original**. No ajustar umbrales ni volver a entrenar para obtener mejores porcentajes. Mantener PR DRAFT, producto/master intactos. Si se confirma, preregistrar nuevo ensayo de inferencia únicamente usando las siete memorias originales SHA congeladas sobre 28300000+; no abrir esas semillas ahora.
