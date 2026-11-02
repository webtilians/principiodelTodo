# INFINITO F10-I — Aprendizaje por experiencia sin supervisión del ganador

**2026-11-02. Resultado técnico: \`F10I_TECHNICAL_COMPLETE\`. Resultado científico preregistrado: \`F10I_EXPERIENTIAL_LEARNING_FAIL\` — G1/G2/G3 FAIL 0/4.**

**Hallazgo principal:** el refuerzo ordinario, el refuerzo con dos trayectorias del mismo mundo y ese mismo refuerzo con predicción de observación futura **convergen a NO CONSULTAR F4 en inferencia greedy OOD (0% de lecturas en las cuatro réplicas de los tres brazos RL)**. La arquitectura idéntica con etiquetas privilegiadas TRAIN consigue selección 100% y decisión 100% en 4/4 réplicas. El problema observado es de **exploración y asignación de crédito al valor de lectura**, pero los datos no demuestran que esa sea la única causa ni descartan otros algoritmos de RL.

## Prerregistro, ejecución original y corrección puramente técnica

- [Prerregistro F10-I](INFINITO_F10I_EXPERIENTIAL_LEARNING_PREREG.md), commit **\`1e146c47b5ac8c76019484368c448b197f374802\`**. Dos correcciones de especificación *antes de abrir cualquier F10-I*: \`780444c826f31263409b3177b77edc47b6ce0985\` conserva un distractor posterior sin firma en todas las variantes y \`e72f6b423bb898e5e0e29ee28e628f9c70365e82\` aclara que el retraso de 2–4 pasos es descuento, **sin** decisiones ni observaciones intermedias. Ninguna puerta científica cambió.
- **[Ejecución original de entrenamiento #41530837321](https://github.com/webtilians/principiodelTodo/actions/runs/41530837321)**: entrenó 16 modelos, guardó 16 checkpoints finales SHA, generó y evaluó DEV/OOD, pero falló DESPUÉS en el serializador de \`rows.json\` por un contador \`numpy.int64\`. RUN técnico FAILURE y estado conservado; no alterar interpretación histórica.
- Un primer intento de recuperar resultados [#41531538404](https://github.com/webtilians/principiodelTodo/actions/runs/41531538404) falló en la comprobación sintáctica PREVIA a cargar checkpoints, por caracteres \`\\n\` literales en un helper Python añadido para serializar JSON. No ejecutó inferencia ni entrenamiento.
- **[Recuperación definitiva de SOLO inferencia original congelada #41531768717](https://github.com/webtilians/principiodelTodo/actions/runs/41531768717)**: **SUCCESS 4/4 recuperación + auditoría independiente**. Descargó los checkpoints **originales** del run fallido, verificó sus SHA, confirmó sin cambios el actor \`f10i_actor.py\`, generador \`f10i_worlds.py\` y AST de \`train\`/\`reward\`/\`main\` frente al commit original \`1350b6381616f3a6f9c9bc9e4a38927f31fd1497\`. Se corrigió **exclusivamente serialización de metadatos NumPy**, en commits \`5d828e623ebe9d55089ea33479d4fd7f344346ae\`, \`fb990194d8e6567bb077d7434ecf3449c21043f7\`, \`e6a77b1e2697fc3876fa8e8950d1a02971f95c75\`. El recuperador llamó a \`evaluate\` SIN ejecutar \`train\`, con los mismos DEV/OOD ya abiertos en el run fallido y sin escoger ni cambiar checkpoints. No se reajustó ningún hiperparámetro tras mirar DEV/OOD.
- Artifacts del run de recuperación: \`f10i-recovered-0..3\` (16 checkpoints + 16 informes/curvas TRAIN + 16×1024 filas de predicción DEV/OOD); [\`f10i-fixed-gate-original-frozen-audit\`](https://github.com/webtilians/principiodelTodo/actions/runs/41531768717) con \`f10i-audit.json\` y todos los IC bootstrap.

Esta es una **repetición de inferencia sobre los mismos mundos**, NO segunda confirmación independiente. **CONFIRM \`20300000+\` permanece sellada.**

## Qué compararon los cuatro brazos

Una misma arquitectura de GRU, política Bernoulli READ/SKIP, selector Categorical de un único registro firmado/no firmado y acción binaria interpretando el valor leído. El selector no recibe valores del registro antes de seleccionar. Los cuatro brazos tienen exactamente los mismos parámetros e inicialización por réplica.

| Brazo | Señal de aprendizaje durante TRAIN |
|---|---|
| A: \`rl_standard\` | Recompensa terminal real +1/−1 y coste −0,05 por leer; baseline EMA de retornos previos |
| B: \`rl_pair\` | Diferencia de recompensas de dos trayectorias ejecutadas independientemente del MISMO mundo, sin oráculo |
| C: \`rl_pair_predict\` | B + predicción del sensor posterior con probabilidad de 20% de error y bono intrínseco pequeño |
| D: \`supervised_ceiling\` | Etiqueta correcta de índice vigente y acción correctas en TRAIN, un **techo privilegiado** |

Los tres RL **NO utilizan \`winner\` ni \`truth\` como targets supervisados**. La recompensa binaria de acierto revela información indirecta sobre la respuesta tras actuar, y el sensor ruidoso de C está correlacionado con la verdad: **no afirmar ausencia absoluta de información correctiva**.

Cada brazo/réplica tuvo exactamente **81.920 rollouts completos**, 20 épocas, 640 actualizaciones, 40.960 presentaciones de estados y dos trayectorias por estado. Son 4 réplicas × 4 brazos = **16 checkpoints originales**. TRAIN \`20100000+\` 512 mundos por réplica; DEV \`20200000+\` y OOD \`20400000+\` 128 mundos cada uno por réplica, cuatro variantes contrafactuales por mundo, **2.048 casos OOD dependientes pertenecientes a 512 mundos base**. TRAIN k6/9→OOD k12/16, varios informes firmados, distractores posteriores no firmados, timestamps públicos, fuentes alteradas y una consulta binaria final. **No se abrió la confirmación \`20300000+\`.**

El retraso 2–4 está representado por \`γ^delay\` pero no por múltiples decisiones temporales; **F10-I NO evalúa credit assignment temporal profundo**. El generador es sintético y no valida autonomía sobre tareas abiertas.

## Resultado operacional OOD

| Brazo | R0 | R1 | R2 | R3 | Total |
|---|---:|---:|---:|---:|---:|
| A: RL estándar | 46,68% | 47,27% | 47,66% | 48,44% | **47,51% (973/2048)** |
| B: RL emparejado | 48,83% | 47,46% | 47,66% | 48,44% | **48,10% (985/2048)** |
| C: RL + predictor | 46,29% | 47,27% | 49,41% | 48,44% | **47,85% (980/2048)** |
| D: control supervisado privilegiado | **100%** | **100%** | **100%** | **100%** | **100% (2048/2048)** |

Baselines públicos: moneda reproducible **50,39%**, última posición de almacenamiento **49,22%**, último timestamp ignorando firma **50,39%**, algoritmo simbólico «último firmado» **100%** por definición, este último NO aprendido.

**READ/SKIP aprendido:** los tres RL eligen **SKIP en 2048/2048 consultas OOD (0% READ)**. Probabilidad media de consultar, antes de umbral greedy: A **2,88%**, B **2,43%**, C **3,00%**. El control privilegiado elige READ en **100%** y tiene probabilidad media de READ **99,92%**.

**Selector sin haber leído:** diagnóstico independiente del gate, porcentaje de elecciones de slot ganador: A **6,45%**, B **0%**, C **0%**; control privilegiado **100%**. Sin embargo, leer a la fuerza no salva políticas cuyo selector/interpretación nunca se entrenaron adecuadamente: A \`forced_read\` **47,66%**, B **48,24%**, C **48,24%**, frente a SKIP **47,51%**, **48,10%**, **47,85%**. Abolir el valor de memoria, firma o timestamp externos no cambia la decisión greedy de los tres RL, pues no la consultan.

El control privilegiado ofrece contraste causal auténtico: READ/FORCED_READ **100%**, FORCED_SKIP **49,17%**, SHUFFLE de valores externos **43,85%**, ERASE_SIGN **50,39%** y ERASE_TIME **51,17%**, con mismo historial GRU dentro de cada intervención. En A/B/C NO existe utilidad F4 consistente.

Las curvas TRAIN sí muestran un patrón específico de exploración fallida: el porcentaje de lecturas **muestreadas** comienza alrededor de 45–55% en la época 1 y cae a ~1–7% en la época 20; el gate greedy OOD termina siempre en SKIP. En algunas réplicas las acciones muestreadas superan ligeramente el azar TRAIN mientras no generalizan a OOD, por lo que **no afirmar que aprendieron a utilizar F4**.

## Intervalos y puertas científicas prospectivas

Prerregistro: G1 exige en >=3/4 réplicas que B **o** C alcance >=75% OOD, >=70% por variante/k/fuente, >=65% selección, diferencia \`forced_read−forced_skip\` >=+15pp con IC95 inferior positivo. G2 exige B−A >=+8pp e IC inferior>0 en >=3/4. G3 exige C−B >=+5pp e IC inferior>0 en >=3/4.

| Réplica | B−A precisión OOD | IC95% por mundo | C−B precisión OOD | IC95% por mundo |
|---|---:|---|---:|---|
| 0 | +2,15 pp | [−1,37; +5,66] | −2,54 pp | [−6,25; +1,17] |
| 1 | +0,20 pp | [−3,52; +3,71] | −0,20 pp | [−5,27; +4,69] |
| 2 | 0 pp | [0;0] | +1,76 pp | [−5,08; +8,40] |
| 3 | 0 pp | [0;0] | 0 pp | [0;0] |

Cada IC usa **2.000 remuestreos por mundo base**, agrupando cuatro variantes del mismo mundo; no tratar las 2048 decisiones OOD como observaciones independientes.

- **G1 autonomía:** FAIL 0/4 B y FAIL 0/4 C.
- **G2 beneficio de comparar dos trayectorias:** FAIL 0/4.
- **G3 beneficio de predicción experiencial:** FAIL 0/4.
- **Estado formal \`F10I_EXPERIENTIAL_LEARNING_FAIL\`**, aunque auditoría técnica \`F10I_TECHNICAL_COMPLETE\`.

La ventaja marginal B−A agregada de +0,59 pp es descriptiva; no supera el gate y no está sólidamente respaldada por IC replicados. C no supera consistentemente a B.

## Diagnóstico causal y límite de inferencia

El fracaso parece comenzar en la etapa de **EXPLORACIÓN DEL VALOR DE LEER**. READ tiene un coste cierto de 0,05 pero su beneficio inicial es difícil de descubrir: tanto selector como cabeza de interpretación tienen que coordinarse para que leer produzca recompensas superiores. Cuando la política reduce la probabilidad de READ, pierde experiencias útiles para aprender selección y el aprendizaje se **autoamplifica hacia SKIP**.

La comparación B usa dos trayectorias **independientemente muestreadas** del mismo estado, NO fuerza que una lea y otra omita, ni que consulten dos registros diferentes. Una vez que READ se vuelve raro, ambas trayectorias normalmente eligen SKIP y **no permiten estimar el valor contrafactual de consultar**. El resultado negativo no demuestra que las diferencias emparejadas sean inútiles en general: identifica una debilidad específica del diseño de exploración.

C introduce un predictor postacción con dato ambiental ruidoso y un pequeño objetivo intrínseco; **no proporcionó una señal suficiente para sostener la exploración del canal de memoria** con estos coeficientes y presupuesto, pero no refuta el uso de aprendizaje predictivo en otros entornos.

El 100% del control D demuestra que la capacidad de la misma arquitectura era suficiente bajo supervisión privilegiada, no que RL pueda alcanzar ese techo mediante el procedimiento probado. Se evalúa únicamente un entorno sintético binario de informes firmados, no políticas cognitivas generales ni creatividad autónoma.

## Decisión y siguiente pregunta

**STOP F10-I.** Conservar este PR en borrador, no retocar coste de READ, coeficientes de entropía o reward, períodos de entrenamiento ni generar nuevos OOD para «rescatarlo». No abrir confirmación.

El siguiente experimento prospectivo candidato, F10-J, debería **forzar exploración informacional comparativa SIN etiquetas de ganador** al inicio: crear pares de experiencias donde una trayectoria ejecuta obligatoriamente READ y otra SKIP en el mismo mundo, **con igual coste de interacción**, y usar el delta REAL de recompensas para aprender una política de lectura independiente de la de selección. Una segunda condición forzaría consultar dos registros distintos, sin señalar cuál es correcto. Comparar contra el REINFORCE original con el MISMO número de episodios y sin usar retornos de acciones no ejecutadas, preservando NO lectura como acción libre en evaluación. Fijar umbrales/semillas nuevos antes de TRAIN y utilizar entornos donde la ventaja de información sea variable para que READ no sea siempre útil. El objetivo sería **descubrir el valor de la información mediante experiencia**, no imponer una política de lectura privilegiada. No ejecutar F10-J en estos mismos mundos.
