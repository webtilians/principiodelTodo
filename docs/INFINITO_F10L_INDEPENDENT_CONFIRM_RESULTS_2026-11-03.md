# INFINITO F10-L-C — Confirmación independiente en 23300000+ con modelos originales

**3 de noviembre de 2026. ESTADO TÉCNICO: \`F10LC_TECHNICAL_COMPLETE\`; ESTADO CIENTÍFICO PRE-REGISTRADO: \`F10L_CONFIRM_PASS\`.** Confirmación **interna de semillas independientes** en la misma familia de mundos sintéticos; NO es validación externa de arquitecturas cognitivas generales.

## Trazabilidad, congelación y cronología

- **Prerregistro de confirmación original**, escrito **antes del generador y antes de abrir CONFIRM**: [\`INFINITO_F10L_INDEPENDENT_CONFIRM_PREREG_2026-11-03.md\`](INFINITO_F10L_INDEPENDENT_CONFIRM_PREREG_2026-11-03.md), commit **\`dea0b788aeef3ae449c4062da828fae24fb37457\`**. No se cambiaron los gates G3/G4 después de ver resultados; son **exactamente los umbrales de F10-L**.
- **F10-L código original y estado de modelo congelado**: commit **\`b5c32cbfb50a24414607e3a24e30e139ca92fede\`**, origen de esta rama y de la PR #63. **Entrenamiento y evaluación OOD originales**: [GitHub Actions #41685499939](https://github.com/webtilians/principiodelTodo/actions/runs/41685499939), 20 checkpoints SHA, 5 modelos por réplica y 4 réplicas, más auditoría independiente previa \`f10l-independent-g0-original/f10l-g0.json\`.
- **Confirmación independiente**: [GitHub Actions #41699416468](https://github.com/webtilians/principiodelTodo/actions/runs/41699416468), commit lanzado **\`c21881d3668b03e1a85f7868286e4a40fe2bf603\`**. Las cuatro réplicas y la auditoría externa de datos completaron **SUCCESS técnico**, sin reentrenar ningún modelo. Todos los valores de peso y SHA concordaron **exactamente** con el manifiesto anterior a CONFIRM. En esta rama, \`observatory/f10l_train.py\`, \`observatory/f10l_worlds.py\`, \`observatory/f10l_stage2.py\` y ambos auditores antiguos permanecieron **sin modificación byte a byte**, igual que \`src/infinito3\`.
- **Comprobación anti-cambio de procedimiento ANTES de abrir CONFIRM**: el generador nuevo reprodujo **bit a bit** los 2048 casos OOD antiguos cuando se suministraron sus semillas originales; y el método \`f10l_stage2.run\` **original inalterado** produjo idénticas predicciones con el generador viejo y el nuevo para OOD anterior. Solo se sustituyó la fuente de datos en una llamada acotada, nunca el actor, el procedimiento de inferencia ni el cálculo de las intervenciones.
- **PR en borrador**: [#63](https://github.com/webtilians/principiodelTodo/pull/63). No modificar ni fusionar en \`master\`/producto por defecto. Artefactos de confirmación: \`f10lc-original-confirm-0..3\`, y auditor \`f10lc-original-independent-confirmation-audit/f10lc-original-audit.json\`.

## Distribución de confirmación y controles

**NUEVAS semillas, sin reutilizar**: \`23300000 + 10000*r + j\`, r=0..3, j=0..127. **128 mundos por réplica × 4 variantes = 512 decisiones por réplica; 2048 decisiones confirmatorias agrupadas en 512 mundos base.** Dentro de cada grupo, \`U_before/U_after/S_before/S_after\` difieren solo en firma/timestamp; únicamente \`S_after\` cambia el valor acreditado. Mismos N=9/11 registros y cuatro formas de procedencia que OOD anterior; orden de almacenamiento y tiempo desacoplados, distractores posteriores no firmados. **No** se han realizado nuevas actualizaciones de pesos, pasos de aprendizaje, selección de checkpoints, cambios de arquitectura, recalibración de thresholds ni retuning de hiperparámetros tras abrir los datos.

Protocolo congelado de inferencia: modelo \`sel_uniform_pair\` → READ valor del slot elegido → modelo \`xor_two_actions\` con flip=1 → acción binaria. Control \`shuffle\` cambia los valores F4 manteniendo el selector; control \`no_flip\` desactiva transformación; \`oracle_slot\` solo diagnóstico y control simbólico último firmado+XOR como techo. Comparaciones con métodos alternativos originales también se conservaron en filas completas.

## Confirmación cuantitativa

| Réplica | Precisión de la selección | Composición | IC95% precisión por mundo | ¿C-G3? | ¿C-G4? |
|---|---:|---:|---|---|---|
| 0 | 100,00% | **100,00%** (512/512) | [100;100]% | PASS | PASS |
| 1 | 100,00% | **100,00%** (512/512) | [100;100]% | PASS | PASS |
| 2 | 84,57% | **88,09%** (451/512) | [84,77;91,02]% | **FAIL** | PASS |
| 3 | 100,00% | **100,00%** (512/512) | [100;100]% | PASS | PASS |
| **Total descriptivo** | **96,14%** | **97,02% (1987/2048)** | Ver IC por réplica | **3/4 PASS** | **4/4 PASS** |

**C-G3 COMPOSICIÓN**: criterio original accuracy>=80%, límite inferior IC95 bootstrap **>70%**, subgrupos por N/fuente/variante **>=70%**, cumplido en al menos **3/4 réplicas**. Se cumple: **PASS 3/4**. En réplica 2, pese al 88,09% global, la variante **\`S_after\` solo alcanza 64,06% (82/128)**, por debajo del mínimo preregistrado 70%; por ello **G3 individual r2 FAIL**, no reinterpretar retrospectivamente. La siguiente variante más difícil de r2 es por fuente \`signed_source1\` 77,34% (cumple mínimo), y por N ambos 9/11 superan 87%.

### Ablaciones de dependencia causal

| Réplica | Normal | Barajar valores F4 | Desactivar flip | Caída F4 | Caída flip |
|---|---:|---:|---:|---:|---:|
| 0 | 100,00% | 50,00% | 0% | **50,00 pp** | **100 pp** |
| 1 | 100,00% | 45,70% | 0% | **54,30 pp** | **100 pp** |
| 2 | 88,09% | 48,83% | 11,91% | **39,26 pp** | **76,17 pp** |
| 3 | 100,00% | 46,09% | 0% | **53,91 pp** | **100 pp** |

**C-G4 REUTILIZACIÓN CAUSAL**: original caída >=+20 puntos tanto con \`shuffle\` como con \`no_flip\`, y **IC95 bootstrap por mundo con límite inferior >0** para ambos, >=3/4 réplicas. **PASS 4/4**. En la réplica r2, F4 efecto +39,26 pp, IC95 [32,42;45,70]pp, y flip +76,17 pp, IC95 [69,92;82,03]pp. **Ningún modelo necesita que alteremos su política para producir estas caídas**: son consecuencias de intervenir sobre información y transformación en inferencia.

Bootstrap: exactamente **2000 remuestreos por los 128 mundos base** por réplica, agrupando siempre sus cuatro versiones, seeds fijos \`241000 + 100*r + offset\`. La precisión global 1987/2048 es **descriptiva** y no equivale a 2048 observaciones independientes para intervalos.

## Estado preregistrado y límites

- **C-G3**: **PASS 3/4**.
- **C-G4**: **PASS 4/4**.
- **GLOBAL \`F10L_CONFIRM_PASS\`**, según el contrato de confirmación original, sin modificar gates.
- **Técnico SUCCESS**, verificación SHA/modelo/regresión incluida.

Esto confirma *internamente* y en semillas nuevas de la misma familia de simulaciones que dos módulos aprendidos previamente mediante recompensas locales y congelados pueden **reutilizarse funcionalmente** para resolver una tarea compuesta no entrenada, cuando **el investigador especifica el encadenamiento SELECT → READ → XOR**. Las ablaciones hacen improbable que el éxito se explique únicamente por ignorar el valor leído o el flag flip.

**NO se demuestra**: invención de operaciones cognitivas, autoorganización de cadenas de operaciones, aprendizaje sin feedback informacional (en XOR binario ambas recompensas revelan la respuesta), superioridad del RL emparejado sobre convencional (G1/G2 de F10-L original siguen FAIL), robustez perfecta (R2 falla S_after), ni transferencia a tareas abiertas o al producto de memoria de asistentes.

**STOP F10-L-C**: no abrir otras semillas por conveniencia, no elegir otro checkpoint, ni retocar umbrales a posteriori. Para avanzar hacia políticas nuevas, diseñar F10-M prospectivamente en mundos distintos donde el agente tenga que **escoger por sí mismo una secuencia de operaciones** en lugar de recibir el cableado, manteniendo los resultados de F10-L y F10-L-C congelados.
