# INFINITO F10-N-C — confirmación interna de síntesis de SRX con memorias originales congeladas

**Fecha: 3 de noviembre de 2026. Resultado científico preregistrado: F10N_CONFIRM_PASS. Resultado técnico: F10NC_TECHNICAL_COMPLETE (todos los trabajos SUCCESS).** Es confirmación en semillas independientes de la MISMA familia sintética, no validación externa ni evidencia de invención de operadores.

## Cronología y artefactos antes de confirmar

1. **Prerregistro PROSPECTIVO**: [INFINITO_F10N_CONFIRM_PREREG_2026-11-03.md](INFINITO_F10N_CONFIRM_PREREG_2026-11-03.md), commit **da441ba1d2f6d02a02dad896ac2574d9a8fdbfc1**, registrado ANTES de crear los módulos de confirmación o generar un único mundo 26300000+. Establece resultados originales, G0 prerequisito, criterios G1–G4, 2000 bootstrap por mundo y tamaño exacto de muestra.
2. **Código original F10-N inmutable**: commit base **6d3e7bd8753f3ad5d1d4a6e44208866bac036298**; los archivos de mundo, memoria de efectos, experimento, modelos F10-L y núcleo del producto se compararon sin cambios de contenido. Los 4 bundles F10-N de memorias originales y experiencias proceden únicamente del **[run 41714113859](https://github.com/webtilians/principiodelTodo/actions/runs/41714113859)**. Los 12 snapshots de memoria SHA fueron guardados antes de abrir DEV/OOD originales. [Auditoría original corregida SIN reentrenar](https://github.com/webtilians/principiodelTodo/actions/runs/41714672031) verificó integridad de los bundles y resultado previo G0 3/4.
3. **20 checkpoints originales F10-L** (5×4) del **[run 41685499939](https://github.com/webtilians/principiodelTodo/actions/runs/41685499939)**. Los SHA se verificaron frente al manifiesto original F10-L y frente a las referencias SHA de entrenamiento F10-N. No se entrenó modelo, política, tabla ni memoria en la confirmación.
4. **Antes de cualquier mundo nuevo CONFIRM**, los tests reprodujeron bit a bit 2048 ejemplos OOD abiertos previamente (26400000+) y el evaluador **F10-N original inalterado** reprodujo bit a bit las 5120 filas de DEV/OOD originales por réplica: programas, predicciones, costes, puntuaciones y SHA de caso. Solamente después se habilitó la sustitución de entrada por las semillas nuevas.
5. **[Run confirmatorio único #41719604626](https://github.com/webtilians/principiodelTodo/actions/runs/41719604626)**, commit del lanzamiento **a04b9a2295027a96c85db243f93233309b7bdee0**. Las cuatro réplicas y el auditor estadístico independiente completaron SUCCESS técnico. Los trabajos evaluaron **el mismo código original observatory.f10n_experiment.evaluate**, insertando los nuevos casos solo en su interfaz dataset, y dejaron las doce memorias/20 políticas idénticas bit a bit antes y después.
6. **Artefactos originales**: \`f10nc-original-confirm-0..3\` (casos/predicciones confirmatorios y registros de regresión); \`f10nc-independent-original-confirm-audit\` contiene el JSON íntegro \`f10nc-confirm-science.json\` con las réplicas, SHA, intervalos y gates fijos. PR #69 en borrador; master y producto intactos.

## Nueva partición de confirmación

**26300000 + 10000 × réplica + j**, réplicas 0..3 y mundos base j=0..127; **128 mundos por réplica × 4 variantes** U_before, U_after, S_before, S_after = 512 casos por réplica, 2048 casos confirmatorios por condición/estrategia, **8192 predicciones** entre los cuatro brazos. Distribución idéntica a la OOD original: 9/11 informes, 4/5 firmados, cuatro modos de procedencia y un informe contradictorio cuya acreditación/fecha cambia entre variantes. **Solo la variante S_after altera el último valor firmado**. Los cuatro casos de un mundo son dependientes; el bootstrap de 2000 remuestreos mantiene el mundo base completo. La tarea sigue siendo MEMORY+FLIP1, ausente de TRAIN y DEV originales.

Los cuatro mecanismos NO aprendieron nada nuevo: A memoria de puntuaciones; B extrapolación aditiva de puntuaciones; C memoria de transiciones y búsqueda de 85 programas; D la misma memoria con exploración dirigida en el TRAIN original. **SRX tuvo cero ejecuciones en TRAIN**, según los logs independientes originales. El planificador de C recibe esquema de estado, catálogo y objetivo diseñados por el investigador; NO recibe la respuesta ni el dato F4 verdadero antes de READ.

## Resultado de confirmación con pesos y memorias inmutables

| Réplica | A recompensa | B puntuación aditiva | C memoria de efectos | D efectos dirigidos | SRX elegido por C | C-G1 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 0,00% | 44,92% | **100,00% (512/512)** | 100,00% | 512/512 | PASS |
| 1 | 0,00% | 50,00% | **100,00% (512/512)** | 100,00% | 512/512 | PASS |
| 2 | 12,30% | 47,66% | **87,70% (449/512)** | 87,70% | 512/512 | **FAIL individual** |
| 3 | 0,00% | 48,83% | **100,00% (512/512)** | 100,00% | 512/512 | PASS |
| **Total descriptivo** | **3,08% (63/2048)** | **47,85% (980/2048)** | **96,92% (1985/2048)** | **96,92%** | **2048/2048** | **3/4 PASS** |

**IC95 de C por mundo**: r0 [100,100]%; r1 [100,100]%; r2 **[83,98%,90,82%]**; r3 [100,100]%. La precisión global 1985/2048 es **descriptiva**; no implica 2048 observaciones independientes para intervalos.

**Excepción preregistrada r2**: la precisión individual de C es 87,70% y supera el umbral general >=80% y el límite inferior IC95 >70%, pero su estrato **S_after = 69,53125% (89/128)** incumple por poco el mínimo **>=70%** definido antes de abrir CONFIRM. Los otros tres estratos de r2 U_before/U_after/S_before obtuvieron **93,75%**. N=9 y N=11 superaron 87%; fuentes iid=88,28%, signed_source0=100%, signed_source1=78,13% y alternating=84,38%. **NO declarar G1 PASS en r2 ni modificar el 70%.**

## Controles causales y contraste de memoria

| Réplica | C − B con mismas experiencias | IC95 C−B por mundo | Caída por shuffle F4 | Caída por desactivar X |
|---|---:|---|---:|---:|
| 0 | **+55,08 pp** | [+50,78; +58,98] pp | **46,09 pp** | **100 pp** |
| 1 | **+50,00 pp** | [+45,70; +53,92] pp | **50,00 pp** | **100 pp** |
| 2 | **+40,04 pp** | [+34,38; +45,70] pp | **35,94 pp** | **75,39 pp** |
| 3 | **+51,17 pp** | [+46,88; +55,08] pp | **47,66 pp** | **100 pp** |

Los límites inferiores IC95 preregistrados de todos los contrastes causales son mayores que cero en las cuatro réplicas: en r2 shuffle F4 [28,90;43,16] pp y X disabled [68,36;82,03] pp. Por tanto, ni leer F4 ni aplicar XOR son operaciones decorativas: alterarlas reduce realmente la precisión.

La ventaja de C frente a B, **usando las mismas experiencias originales**, se reproduce: aproximadamente **+40 a +55 puntos**. C y D dan EXACTAMENTE la misma precisión: no se demuestra que dirigir la exploración mejore el resultado con este presupuesto, igual que en la OOD original.

## Gates congelados, sin interpretación retrospectiva

| Criterio | Resultado confirmación |
|---|---|
| Prerrequisito original G0, competencia conocida | PASS 3/4 |
| **C-G1** nueva secuencia SRX, precisión/CI y todos los estratos | **PASS 3/4** |
| **C-G2** uso causal F4+XOR | **PASS 4/4** |
| **C-G3** memoria de efectos C > memoria de puntuaciones B | **PASS 4/4** |
| C-G4 exploración dirigida D > uniforme C (diagnóstico) | **FAIL 0/4** |

**Estado científico: F10N_CONFIRM_PASS**, según la regla prospectiva original (G0 y C-G1, C-G2, C-G3 con >=3/4; C-G4 no requisito). **Estado técnico: F10NC_TECHNICAL_COMPLETE** y todos los jobs SUCCESS.

## Qué se ha establecido y qué no

**Resultado reproducido:** un agente dotado de memoria explícita de los efectos realmente observados de operaciones ejecutadas puede, con búsqueda sobre un catálogo definido, **proponer y usar una secuencia nunca ejecutada durante entrenamiento** y superar un control de memoria explícita de puntuaciones a igualdad de experiencias. La dependencia causal de información y transformación se repite con nuevos mundos y los mismos artefactos de aprendizaje.

**Límites decisivos:** la familia de mundos, la estructura SOURCE/FLIP, los registros de trabajo instrumentados, la función de objetivo XOR, la biblioteca de operaciones y el mecanismo de búsqueda exhaustiva (85 programas) fueron proporcionados por los investigadores. No demuestra que el agente haya inventado nuevas primitivas ni aprendido la semántica de la meta desde observaciones crudas; no replantea online su programa tras READ; no demuestra una ventaja de exploración dirigida; no garantiza transferencia a entornos nuevos fuera de esta distribución, a un LLM real ni al producto comercial. La réplica 2 sigue exhibiendo una debilidad con evidencia acreditada contradictoria y cambios de versión S_after.

**STOP F10-N-C:** no retocar modelos, memorias, exploración, semillas o umbrales tras el resultado; conservar la PR en borrador y no fusionar con master. El siguiente problema científico debe aislar qué ocurre al retirar la instrumentación privilegiada de estados o al requerir planificación reactiva, siempre preregistrando una nueva familia de mundos.
