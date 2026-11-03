# INFINITO F10-J — Exploración obligada READ/SKIP no descubre una política útil (resultado independiente)

**3 de noviembre de 2026.** **\`F10J_TECHNICAL_COMPLETE\`** / **\`F10J_STRUCTURED_EXPLORATION_FAIL\`**, sin cambio alguno de gates, costes, entrenamiento o mundos después de ver resultados. **G1/G2/G3/G4: FAIL 0/4 réplicas cada uno**. La ejecución entera, incluida auditoría estadística independiente, terminó **SUCCESS técnico** en [GitHub Actions #41636104900](https://github.com/webtilians/principiodelTodo/actions/runs/41636104900), commit de código \`813e0b53cb073e34460862cc99371fee625c1ebd\`.

## Trazabilidad y controles

- Prerregistro original antes de implementar/generar F10-J: [\`INFINITO_F10J_STRUCTURED_EXPLORATION_PREREG.md\`](./INFINITO_F10J_STRUCTURED_EXPLORATION_PREREG.md) commit \`9a90692bf0b3f1af69f10df174f82da29a1023e6\`; aclaración de balanceo tipo×longitud×fuente **antes de cualquier TRAIN/DEV/OOD** commit \`ed679200b1916ee643ec68095f2c03742b362c3b\`. Los gates fijados no se ajustaron.
- Nuevos TRAIN \`21100000+\` (512 mundos/réplica), DEV \`21200000+\` (128), OOD \`21400000+\` (128); cuatro contrafactuales por mundo. El namespace de **confirmación \`21300000+\` permanece sellado y no generable**. No se utilizaron mundos antiguos de F10-I ni datos de producto.
- Mismo actor GRU/selector/gate/acción en **5 brazos × 4 réplicas**, inicialización igual por réplica, **20 checkpoints SHA**, 640 actualizaciones y **81.920 episodios de entrenamiento realmente ejecutados por modelo**, con 10% de corrupción exógena de reward compartida por pares, no información privilegiada para A-D. E recibe explícitamente winner/verdad como techo controlado.
- Auditoría independiente de entrenamiento verifica 12.288 hashes distintos de TRAIN/DEV/OOD, 20 SHA de pesos, **20.480 filas de predicción DEV/OOD** (5×4×2×128×4), mismas historias y etiquetas entre brazos, igual presupuesto e integridad de splits. Bootstrap 2.000 por **mundo base**, agrupando sus cuatro variantes en cada remuestreo. Artefactos originales \`f10j-original-0..3\` y \`f10j-original-fixed-gates-and-bootstrap\` disponibles en el [run #41636104900](https://github.com/webtilians/principiodelTodo/actions/runs/41636104900).
- Los valores de los informes **NO están en el estado recurrente**: en INFORMATIVE se necesita recuperar un único valor de F4; en PUBLIC la respuesta está en el enunciado y una lectura solo añade coste. Ambos tipos, k=12/16 y fuentes OOD se cruzan equilibradamente en las 16 celdas, 8 mundos base por celda y réplica. En TEST pre-TRAIN se confirmó que cambiar los valores F4 no puede cambiar la salida SKIP.

## Resultado OOD: precisión por tipo (1.024 decisiones informativas y 1.024 públicas por brazo)

| Brazo | INFORMATIVE (necesita F4) | PUBLIC (no necesita F4) | READ INFORMATIVE | Selección winner INFORMATIVE |
|---|---:|---:|---:|---:|
| A. \`on_policy\`, REINFORCE convencional | **52,34%** | **68,46%** | **0%** | 22,07% |
| B. \`forced_gate\`, un READ/otro SKIP realmente ejecutados por caso | **52,73%** | **70,02%** | **0%** | 5,18% |
| C. \`forced_gate_slot_explore\`, B + 50% de slots uniformes | **52,64%** | **72,85%** | **0%** | 5,57% |
| D. \`paired_credit\`, C + diferencia de recompensa por parejas | **54,30%** | **74,22%** | **0%** | 9,47% |
| E. \`supervised_ceiling\` (etiquetas privilegiadas) | **100%** | **100%** | **100%** | **100%** |

Cifras agrupadas descriptivas 4 réplicas, 256 decisiones INFORMATIVA + 256 PÚBLICA OOD por réplica y brazo. En PUBLIC el control E ejecuta SKIP 100%. En los cuatro brazos no supervisados **todas las decisiones de ambos tipos son SKIP**.

Por réplica, precisión OOD en INFORMATIVE:

| Réplica | A | B | C | D | E |
|---|---:|---:|---:|---:|---:|
| 0 | 53,12% | 54,69% | 54,69% | 55,47% | 100% |
| 1 | 49,22% | 49,61% | 50,00% | 51,56% | 100% |
| 2 | 51,56% | 53,91% | 55,08% | 53,91% | 100% |
| 3 | 55,47% | 52,73% | 50,78% | 56,25% | 100% |

**Intervenciones causales:** En INFORMATIVE, comparar \`force_read\` con \`force_skip\` después del entrenamiento produce de forma agregada:
- A: **52,44%** contra **52,34%**.
- B: **52,44%** contra **52,73%**.
- C: **52,54%** contra **52,64%**.
- D: **54,00%** contra **54,30%**.
- E: **100%** contra **48,93%**.

Ni siquiera **forzar READ en inferencia** rescata a los RL, porque selector y cabeza de acción tampoco han aprendido a convertir el valor recuperado en un beneficio funcional. En E sí lo demuestra la diferencia de ~+51 puntos y la degradación fuerte al permutar valores F4.

## Gates prospectivos, sin reinterpretación

El prerregistro exigía:
- **G1 AUTONOMOUS_INFORMATION**: C o D, >=3/4 réplicas, INFORMATIVE accuracy>=75%, winner>=65%, READ>=80% y sensibilidad causal READ−SKIP>=+20pp con IC95 inferior>0, además de PUBLIC accuracy>=90%, SKIP>=80%, y mínimos por k/fuente informativa.
- **G2** B−A >=+10pp en INFORMATIVE e IC inferior>0 en >=3/4.
- **G3** C−B >=+8pp e IC inferior>0 en >=3/4.
- **G4** D−C >=+5pp e IC inferior>0 en >=3/4.

**Resultados:** \`G1_C_or_D=0/4\`, \`G2_B_minus_A=0/4\`, \`G3_C_minus_B=0/4\`, \`G4_D_minus_C=0/4\`. Global **\`F10J_STRUCTURED_EXPLORATION_FAIL\`**. Ninguna mejora de A→B, B→C o C→D tiene soporte suficiente conforme a los umbrales prospectivos e intervalos por mundo. Aunque D alcanza 56,25% en una réplica, no representa utilidad autónoma de F4.

Ejemplo de los intervalos de confianza por mundo de R0: B−A **+1,56 pp, IC95 [−1,56,+4,69]**; C−B **0 pp, IC95 [−2,34,+2,34]**; D−C **+0,78 pp, IC95 [0,+2,34]**. No satisface G2/G3/G4.

## Diagnóstico: la exploración aislada no basta

Este experimento mejora el diagnóstico de F10-I, pero **no** resuelve el problema.

1. En A (on-policy), las lecturas muestreadas durante TRAIN caen hacia ~2–9% según réplica en la época 20. La política evalúa SKIP 100%, reproduciendo colapso de lectura.
2. En B, C y D, los **rollouts de entrenamiento sí mantuvieron READ=50%** por diseño desde la época 1 hasta la 20 (un READ, un SKIP por estado), pero la probabilidad aprendida de lectura OOD informativa llega solo a **1–2%**. **No se puede atribuir el fracaso simplemente a que el entorno dejó de ofrecer experiencias de lectura.**
3. El selector sigue fallando: C acierta el slot último firmado ~5,6% y D ~9,5% en OOD INFORMATIVE. Sin elegir el slot y aprender cómo su valor se relaciona con el éxito, observar READ no ofrece ventaja consistentemente.
4. Hay también **aprendizaje insuficiente de una regla pública trivial**: en PUBLIC, donde basta leer un bit del enunciado y no se necesita F4, los RL alcanzan 68–74%, no 100%. Por tanto, parte del obstáculo está en la propagación/eficiencia de la señal de RL sobre la arquitectura, no solo en el descubrimiento del valor de memoria.
5. El control E aprende totalmente (100% ambos tipos, usa READ solo cuando útil), demostrando capacidad representacional **bajo etiquetas privilegiadas**. No demuestra que el mismo protocolo RL pueda hacerlo.

La combinación de selección difícil, interpretación del valor y crédito por recompensa final produce un problema de aprendizaje **coordinado**: las tres políticas deben mejorar conjuntamente para que READ tenga retorno neto mayor que SKIP. Obligar un READ a ocurrir mantiene experiencia, pero no le entrega a las cabezas una señal local suficiente para aprovechar el reporte adecuado. La posibilidad de gradientes no informativos, configuraciones de entropía insuficientes y sesgos de arquitectura queda abierta; no afirmar causa única confirmada.

## Siguiente hipótesis potencial, NUEVO experimento F10-K (NO ejecutado aquí)

Separar la adquisición de una *capacidad de interpretación de evidencia* de la decisión de *consultar evidencia*, sin introducir etiquetas del ganador. Se puede diseñar un **aprendizaje temporal/auxiliar a partir de la consecuencia observada**, con:
- Un paso de **lectura exploratoria contrastiva del mismo registro** bajo intervención READ/SKIP para generar experiencia identificable, manteniendo todos los canales de observación y presupuesto equivalentes.
- Un objetivo **autosupervisado y verificable por el entorno** que aprenda a predecir consecuencias de observaciones *realmente vistas*, evitando índice winner oráculo y etiquetas de acción del evaluador.
- Controles aislados para (i) interpretar valor leído, (ii) seleccionar registro acreditado, (iii) gate READ/SKIP, manteniendo la cadena de crédito causal y mundo/capacidad iguales.
- Tareas públicas fáciles como **control positivo**: si el RL no aprende un bit público con >=95%, no interpretar sus fallos de memoria como problemas específicos de memoria.
- Nuevo TRAIN/DEV/OOD/CONFIRM preregistrado; no reajustar F10-J tras sus OOD. La transferencia al producto requiere experimentación independiente y nunca usar Bemula-Lite como entrenamiento.

**STOP F10-J**. Conservar PR de investigación en borrador, reportar resultado negativo, no abrir \`21300000+\`, no fusionar automáticamente en \`master\`, no ajustar hiperparámetros para obtener un PASS.
