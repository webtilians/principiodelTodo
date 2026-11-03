# INFINITO F10-N — resultado original: memoria de efectos propone una secuencia nunca ejecutada

**3 de noviembre de 2026. Resultado científico: F10N_EFFECT_MEMORY_COMPOSITION_PASS.** Resultado de OOD preregistrado sobre nueva familia de semillas, NO confirmación independiente posterior. La partición F10-N de confirmación 26300000+ permanece SELLADA. No se ha entrenado con la tarea MEMORY+FLIP1 ni ejecutado SRX durante TRAIN.

## Cronología, procedencia y reparación técnica acotada

- **Prerregistro ANTES de generar cualquier código o dato F10-N**: [INFINITO_F10N_EXPERIENCE_EFFECT_MEMORY_PREREG_2026-11-03.md](INFINITO_F10N_EXPERIENCE_EFFECT_MEMORY_PREREG_2026-11-03.md), commit **2992972edaa8c955b6cc4e623acab6d39d1cd0d0**. Gates G0–G4, semillas, presupuestos, operaciones, controles y reserva de SRX definidos entonces.
- **Ejecución ORIGINAL congelada**: [Actions #41714113859](https://github.com/webtilians/principiodelTodo/actions/runs/41714113859), commit del lanzamiento **2ea33e63cab5af7b781c35a9c349224e30a7e7f9**. Las cuatro réplicas completaron TRAIN y DEV/OOD correctamente y subieron las cuatro colecciones originales de experiencias por SHA. El job estadístico encontró un fallo exclusivamente de programación del auditor tras procesar r0: sobrescribía accidentalmente la variable g0 del manifiesto con un booleano. Esto explica el estado técnico global FAILURE de este primer run; no invalida los cuatro modelos y sus registros congelados.
- **Corrección técnica documentada** en [INFINITO_F10N_AUDIT_ONLY_CORRECTION.md](INFINITO_F10N_AUDIT_ONLY_CORRECTION.md), commit **a0d1675891adac323d78c6a7825ad112acbad39d**: renombrar solamente el booleano local como g0_pass. Sin cambios a criterios, bootstrap, generación, pesos, memorias, datos, resultados de inferencia o código científico de F10-N.
- **Auditoría reejecutada SIN reentrenar NI reevaluar**: [Actions #41714672031](https://github.com/webtilians/principiodelTodo/actions/runs/41714672031), SUCCESS. Descargó SOLO los cuatro artefactos originales del run 41714113859 y el manifiesto original de modelos F10-L del run **41685499939**. Verificó que el código de mundos, aprendizajes, memoria y prerregistro permanecía idéntico al lanzamiento original; auditó todos los logs y cálculos, y guardó el JSON íntegro **f10n-audit-only-original-immutable-results**. No importó PyTorch ni abrió ningún mundo nuevo en la recuperación.
- **Operadores originales inmutables:** se verificaron los 20 SHA checkpoint originales F10-L en cada réplica, incluido SELECT y XOR, sin modificarlos. No se tocaron master ni src/infinito3, ni se usaron datos Bemula o confirmaciones antiguas. Rama F10-N y PR #66 permanecen en borrador.

## Hipótesis intervenida

El controlador F10-M seleccionaba entre 85 programas usando pesos aprendidos de recompensas, pero no conservaba una memoria explícita de los **efectos de cada acción**. F10-N conserva operaciones P (carga público), S (elige slot con modelo F10-L), R (lee bit del slot), X (aplica XOR congelado). El entorno devuelve un estado de trabajo observacional **después de ejecutar cada operación** y recompensa al finalizar el episodio.

**SRX nunca se ejecutó durante TRAIN**: se impidió deliberadamente esa cadena en todos los contextos TRAIN, dejando las otras 84 disponibles (incluidas acciones redundantes, inválidas y con orden erróneo). El actor estudia PUBLIC0, PUBLIC1 y MEMORY0, pero nunca MEMORY1. Para planificar una consulta MEMORY1, la memoria de efectos predice qué harían las distintas operaciones con dos hipótesis sobre el bit F4 (0/1), sin consultar su valor real, y escoge la secuencia que satisface el objetivo en ambas.

**Cuatro condiciones y exposición:**
- A: tabla explícita de recompensas por contexto/programa, con registros de éxito y coste.
- B: extrapolación de la tabla de retornos entre contextos, sin aprender transiciones.
- C: el **MISMO HISTORIAL EXACTO de programas ejecutados de A/B**, pero construye un mapa de efectos de operación antes→después y busca un programa nuevo utilizando ese conocimiento.
- D: modelo de efectos idéntico a C pero con ensayos dirigidos por incertidumbre/cobertura (50%) + ensayos uniformes (50%), en los mismos mundos y presupuesto.

Por réplica, 16 épocas ×3072 ejemplos TRAIN ×2 intentos ejecutados = **98.304 programas realmente ensayados** por stream (uniforme compartido A/B/C y dirigido D). En las cuatro réplicas hay **786.432 episodios ejecutados distintos entre los dos streams**; no contar tres veces las mismas observaciones compartidas de A/B/C. Todos los programas, recompensas, estados antes/después de cada operación y costes se conservaron en logs comprimidos originales. Cadenas de origen F4 no se entregan antes de ejecutar R.

## Rendimiento OOD: combinación MEMORY+FLIP1 jamás entrenada

512 decisiones por réplica, 128 mundos base ×4 variantes vinculadas; 2.048 predicciones por brazo.

| Réplica | A: mejor recompensa histórica | B: extrapolación de recompensa | C: efectos observados, exploración uniforme | D: efectos, exploración dirigida | C, programa OOD |
|---|---:|---:|---:|---:|---|
| 0 | 0,00% | 51,56% | **100,00%** | **100,00%** | SRX (512/512) |
| 1 | 0,00% | 50,78% | **100,00%** | **100,00%** | SRX (512/512) |
| 2 | 10,35% | 45,70% | **89,65%** (459/512) | **89,65%** | SRX (512/512) |
| 3 | 0,00% | 50,78% | **100,00%** | **100,00%** | SRX (512/512) |
| **Total descriptivo** | **2,59%** | **49,71%** | **97,41% (1995/2048)** | **97,41%** | **SRX 2048/2048** |

**Hallazgo:** el modelo de efectos C eligió **S→R→X en TODAS las 2.048 decisiones OOD**, aunque esta cadena tuvo **cero ejecuciones reales de entrenamiento**. Los errores de r2 provienen de fallos del operador congelado de selección en casos más difíciles, no de elegir un programa inadecuado. Su IC95 por 2.000 bootstrap de mundos fue **[86,33%,92,58%]**; las otras tres réplicas 100% [100%,100%]. Estos intervalos son por **mundo base**, no por filas correlacionadas.

### Contraste causal con el MISMO historial

Ventaja C−B, IC95 por mundo:

| Réplica | Efecto C−B | IC95 |
|---|---:|---|
| 0 | **+48,44 pp** | [+44,53; +52,73] pp |
| 1 | **+49,22 pp** | [+44,92; +53,52] pp |
| 2 | **+43,95 pp** | [+38,47; +49,61] pp |
| 3 | **+49,22 pp** | [+44,92; +53,52] pp |

El mismo conjunto de intentos no resulta útil para el programa inédito si solo se almacenan sus puntuaciones, pero sí permite construirlo cuando se retienen los efectos de las operaciones. Esta diferencia favorece específicamente **el mecanismo de memoria de transiciones + búsqueda**, no exclusivamente la cantidad de feedback o el número de episodios.

### Ablaciones de evidencia y transformación (brazo C)

| Réplica | Caída de precisión al permutar F4 | Caída al desactivar XOR |
|---|---:|---:|
| 0 | **47,46 pp**, IC95 [40,62;54,30] | **100 pp** |
| 1 | **49,02 pp**, IC95 [42,38;55,86] | **100 pp** |
| 2 | **46,09 pp**, IC95 [39,26;52,54] | **79,30 pp**, IC95 [73,05;85,16] |
| 3 | **51,56 pp**, IC95 [44,73;58,59] | **100 pp** |

Las cuatro réplicas superaron G2. Por tanto, el controlador no se limitó a seleccionar SRX como cadena decorativa: su resultado depende causalmente del contenido leído y de la transformación.

### Tareas vistas DEV (brazo C)

| Réplica | PUBLIC0 | PUBLIC1 | MEMORY0 |
|---|---:|---:|---:|
| 0 | 100% | 100% | 100% |
| 1 | 100% | 100% | 100% |
| 2 | 100% | 100% | **84,38%** |
| 3 | 100% | 100% | 100% |

**G0 pasa 3/4**. La réplica 2 no alcanza 90% en MEMORY0, y debe conservarse como una debilidad. En OOD sí pasa G1 con 89,65% y sus estratos cumplen límites preregistrados.

## Gates científicos originales

| Criterio | Resultado |
|---|---|
| G0: competencia >=90% en cada tarea vista | **PASS 3/4** |
| G1: propuesta y precisión de SRX nunca ejecutado, OOD >=80%, IC inferior>70%, estratos>=70% | **PASS 4/4** |
| G2: contenido F4 y XOR causalmente necesarios, caídas>=20 pp e IC inferior>0 | **PASS 4/4** |
| G3: memoria de efectos C vence tabla aditiva B por>=15pp con mismo historial | **PASS 4/4** |
| G4: exploración dirigida D vence C por>=5pp | **FAIL 0/4** |

**Global original F10N_EFFECT_MEMORY_COMPOSITION_PASS**, conforme al contrato G0∧G1∧G2∧G3. NO se ha demostrado una ventaja de dirigir la exploración. A pesar de su igual precisión final, D ejecutó más acciones intermedias y produjo menos errores de precondición en TRAIN que C; es diagnóstico, no ventaja final confirmada.

## Alcance real, sesgos y próximo paso

**Sí se ha demostrado en este entorno:** recordar efectos observados de operaciones ejecutadas permite sintetizar por búsqueda una secuencia operativa nueva, no ejecutada durante TRAIN, y aprovechar memoria F4 y XOR; almacenar solo recompensas por programa/contexto no produjo esa transferencia con los mismos datos.

**No se ha demostrado**: que el agente inventara operaciones o descubriera por sí solo la semántica del objetivo; que aprendiera la representación de su estado desde observaciones crudas; que no dependiera de instrumentación del entorno, una interfaz de operadores, simular F4=0/1 o búsqueda exhaustiva entre solo 85 candidatos; que replantee sus decisiones después de leer; ni utilidad automática en LLM, tarea real o producto comercial. **La tarea y el espacio de estados fueron diseñados para que el mecanismo pudiera reutilizar efectos.** La propia selección F10-L aún falla en r2.

**STOP F10-N.** No tunear ni reentrenar después de OOD ni abrir mundos adicionales para mejorar el resultado. CONFIRM **26300000+ SELLADA**; próximo paso indicado: preregistrar una **confirmación independiente de pesos y memorias originales** en ese namespace, con mismo protocolo y criterios, sin volver a aprender efectos. Solo después estudiar inferencia de estados desde observaciones no instrumentadas o planificación reactiva de operaciones.
