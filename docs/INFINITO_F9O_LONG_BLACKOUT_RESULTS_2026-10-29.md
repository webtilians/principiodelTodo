# INFINITO F9-O — Resultado prospectivo: seis cortes F4 continuos frente a seis fragmentados

**Fecha:** 29/10/2026. **Estado:** completado, con resultado primario **NO CONFIRMADO**.

- [GitHub Actions #40957255987](https://github.com/webtilians/principiodelTodo/actions/runs/40957255987): **SUCCESS** (integridad, cuatro entrenamientos, gate de ocho modelos, cuatro evaluaciones nuevas y agregado).
- [PR #22](https://github.com/webtilians/principiodelTodo/pull/22), borrador apilado sobre F9-N #21; sin merge.
- [JSON íntegro agregado](./INFINITO_F9O_RESULTS_2026-10-29.json) (incluye números por semilla, ambos grupos, BC/PPO y todos los controles).
- [Artifact agregado CI #12935190542](https://github.com/webtilians/principiodelTodo/actions/runs/40957255987/artifacts/12935190542). Los cuatro artifacts por semilla contienen todos los pares originales e invertidos, no solo agregados.
- [Protocolo preregistrado y enmiendas antes de confirmación](./INFINITO_F9O_LONG_BLACKOUT_PROTOCOL.md).

## Pregunta y respuesta breve

**Pregunta:** con idénticas seis exposiciones de readout F4 suprimido por historia de imitación, ¿enseñar seis pasos *consecutivos* (continuous6) en vez de tres intervalos de dos (fragmented6) mejora la **retención y utilización autónoma** de la pista en S13 cuando F4 desaparece después de observarla?

**Respuesta confirmatoria:** **No hay evidencia consistente de mejora en el desenlace primario tras PPO.** Con F4 anulada siempre que la pista inicial ya no está a la vista, de **160 pares S13** la política fragmentada toma **ambas primeras salidas correctas en 11**; la continua en **14**. Deltas por semilla **[0, −11, 0, +14]** pares de 40, es decir mejora en **solo 1 de 4 semillas**, y no alcanza el criterio preregistrado (>=3/4 y ventaja agregada). El éxito de episodios individuales aumenta, pero se explica descriptivamente por mayor llegada al cruce, no por una ventaja de decisión condicionada a haber llegado.

Se trata de un resultado **negativo del criterio de memoria**, aunque el experimento técnico y el CI hayan sido exitosos.

## Integridad del diseño

- **Ocho políticas realmente nuevas**, 4 inicializaciones × dos brazos \`fragmented6\` y \`continuous6\`; 380.552 parámetros, entrada 660, LSTM 64, store F4 contextual de ocho eventos. Por cada semilla, **hash de pesos iniciales idéntico**, igual arquitectura, mismas acciones, recompensas y mundos.
- **600 mundos S13 TRAIN nuevos por índice**, \`500000+index*10000+j\`, dos variantes key↔ball por mundo = **1200 historias teacher por semilla**. Misma ordenación y 40 épocas BC, con **7200/7200 lecturas efectivamente anuladas** en cada brazo/semilla (seis por historia, cero historias incompletas). Calendarios: continuo offsets 0–5 vs fragmentado 0,1,3,4,6,7, a partir de la primera detección pública de cualquier key/ball + dos pasos.
- Guardados checkpoints **después de BC** y **después de PPO limpio**; PPO ejecutó **20.096 pasos por política** (20.000 solicitados), sin máscaras en ninguno de los dos brazos; las trayectorias PPO pueden divergir aunque el presupuesto sea el mismo.
- Los **ocho checkpoints y los dos stages por modelo** pasaron gate separado de SHA-256 / exposición exacta / parámetros / pasos ANTES de abrir confirmación.
- Confirmación nueva \`600000+index*10000+j\`, **160 pares S13** y **80 pares S7 por brazo/condición**, distinta de train 500000+, F9-M 300000+, F9-N 400000+ y holdout histórico 900000+, que permaneció cerrado. Ninguna política reentrenada usando confirmación.
- Problemas de instrumentación anteriores a la confirmación: el primer horario de ocho pasos no cabía en 61 de 1200 historias S13, por lo que se modificó a **seis pasos realmente iguales**; posteriormente dos aserciones heredadas seguían comprobando ocho, fueron corregidas y cubiertas con tests. La ejecución final terminó con **11/11 jobs exitosos**. No se usó confirmación para arreglar estos errores.

## Primario, tras BC+PPO: S13, F4 ausente cuando la pista no se ve

**Denominadores:** 40 mundos pareados por semilla; 4 semillas; **160 pares / 320 episodios por brazo**. El evaluador cuenta pares con **ambas primeras decisiones correctas** incluyendo los episodios que nunca llegan al cruce.

| Indicador | Fragmentado6 | Continuo6 |
|---|---:|---:|
| **Ambas primeras elecciones correctas** | **11/160 (6,88 %)** | **14/160 (8,75 %)** |
| Llega al cruce | 240/320 (75,0 %) | **320/320 (100,0 %)** |
| Entra en una salida | 240/320 (75,0 %) | **320/320 (100,0 %)** |
| Primera decisión correcta entre quienes se comprometen | **131/240 (54,58 %)** | 173/320 (54,06 %) |
| Primera decisión correcta, sobre todos los episodios | 131/320 (40,94 %) | **173/320 (54,06 %)** |
| Finaliza con recompensa positiva | 131/320 (40,94 %) | **173/320 (54,06 %)** |
| Timeouts | 80/320 (25,0 %) | **0/320** |
| Pasos F4 anulados en evaluación (media por episodio) | **150,01** | **11,18** |

| Semilla | Fragmentado6: ambas/40 | Continuo6: ambas/40 | Diferencia |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 11 | 0 | −11 |
| 2 | 0 | 0 | 0 |
| 3 | 0 | 14 | +14 |
| **Total** | **11/160** | **14/160** | **+3** |

**Interpretación:** el tratamiento continuo consigue alcanzar la bifurcación con más frecuencia y evita los timeouts, pero la tasa de *primera elección acertada entre quienes llegaron a comprometerse* es prácticamente igual (~54 %). Esta comparación condicionada **no es un estimador causal** porque quienes llegan difieren entre grupos. Dado que los pares correctos apenas aumentan y solo en una semilla, **la mejora global no demuestra mayor memoria autónoma de la identidad de la pista**.

Importante: las políticas recibieron la misma **dosis de máscaras durante la imitación TRAIN**, no durante la confirmación autónoma. Los diferentes números de pasos F4 anulados en evaluación (150 frente a 11) se deben a trayectorias, bucles y timeouts distintos. Sería incorrecto afirmar «misma exposición» en la fase de evaluación.

## Antes de PPO, únicamente BC (secundario; no seleccionar retrospectivamente)

| S13 con F4 ausente | Fragmentado6 tras BC | Continuo6 tras BC |
|---|---:|---:|
| Ambas primeras elecciones correctas | **18/160** | **42/160** |
| Por semilla | [0, 12, 6, 0] | [18, 13, 0, 11] |
| Diferencias por semilla | — | [+18, +1, −6, +11] |
| Llegada al cruce | 311/320 | 315/320 |
| Primera elección correcta entre compromisos | 135/311 (43,41 %) | 168/315 (53,33 %) |
| Éxito | 135/320 (42,19 %) | 168/320 (52,50 %) |

Existe **una señal preliminar** después de BC: +24 pares con ambas decisiones correctas, ganancia en tres de cuatro semillas, y llegada casi igual entre brazos. Pero el protocolo declaró **post-PPO** como resultado principal. Después de PPO limpio, la ventaja decae a +3 pares, con una única semilla positiva. **No es científicamente válido sustituir el desenlace central por BC** porque resulte más favorable. El mecanismo exacto de la pérdida tras PPO requeriría investigación fuera de este conjunto confirmatorio ya consumido.

## Robustez temporal breve, navegación normal y transferencia S7 (post-PPO)

| Tarea / condición | Fragmentado6: dos primeras correctas | Continuo6: dos primeras correctas |
|---|---:|---:|
| **S13 normal**, F4 disponible | 147/160 (91,88 %) | **153/160 (95,63 %)** |
| **S13 fixed_4**, F4 suprimida t=3..6 | 126/160 (78,75 %) | **150/160 (93,75 %)** |
| **S13 cue_lost**, F4 suprimida tras pista | **11/160 (6,88 %)** | **14/160 (8,75 %)** |
| S7 normal | 4/80 | 0/80 |
| S7 fixed_4 | 0/80 | 0/80 |
| S7 cue_lost | 0/80 | 0/80 |

En S13/fixed_4 la exposición de evaluación **sí es idéntica** (4/4 pasos efectivos en 320/320 episodios por brazo): la política continua mejora pares correctos **+24/160**, con ventaja por semilla **[+12, +2, +4, +6]**, positiva 4/4. Esto es evidencia de **tolerancia a interrupciones breves con F4 restaurada**, no de independencia de F4 durante el recorrido. También normal S13 es alto en ambos grupos.

**Transferencia S7 es mala** al haber entrenado exclusivamente en S13: 4/80 vs 0/80 pares de dos primeras decisiones correctas con F4 normal, y ninguno bajo cortes. No afirmar generalización entre tamaños.

## Conclusión científica y siguiente hipótesis

**Resultado primario: no confirmado.** Enseñar solo seis pasos continuos de ausencia de F4 en **BC**, seguidos de PPO con F4 íntegra, **no logra un uso de memoria autónoma estable de la identidad de la pista en S13 bajo pérdida prolongada de F4**, comparado con igual dosis fragmentada. Hay ganancia de llegada y robustez fuerte a interrupciones cortas, pero no un éxito consistente en pares cue-invertidos con F4 largamente ausente.

La diferencia BC → PPO hace razonable **formular** un nuevo ensayo, no modificar este: entrenar PPO con interrupciones igualadas también durante interacción, posiblemente bajo currículo cuidadosamente presupuestado, con un conjunto confirmatorio nuevo aún no utilizado, y mantener la métrica de ambos primeros compromisos correctos. Ninguna afirmación de conciencia, comprensión semántica general ni límites universales de la memoria. Cuatro semillas, misma familia MiniGrid y cobertura de cortes dependiente de la conducta en evaluación limitan generalización.

**Proveniencia:** el JSON adjunto versiona tanto métricas completas por semilla como recuentos exactos y efecto descriptivo del modelo. La confirmación \`600000+\` ahora está **consumida**, no puede volver a usarse para ajustar el entrenamiento y declararlo confirmatorio.
