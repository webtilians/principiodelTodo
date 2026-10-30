# INFINITO F9-P — Informe confirmatorio: PPO prolongado sin F4 destruye la decisión por pista

**29/10/2026. Resultado principal: NO CONFIRMADO, con efecto desfavorable.** [GitHub Actions #40960267485](https://github.com/webtilians/principiodelTodo/actions/runs/40960267485) ha terminado **SUCCESS, 11/11 jobs**: tests, cuatro parejas PPO, gate de los ocho modelos, cuatro confirmaciones independientes y agregado. [PR #23](https://github.com/webtilians/principiodelTodo/pull/23) apilada sobre F9-O #22, borrador sin fusionar. [Resultados agregados versionados](./INFINITO_F9P_RESULTS_2026-10-29.json) y [artifact de agregado #12935293301](https://github.com/webtilians/principiodelTodo/actions/runs/40960267485/artifacts/12935293301).

## Pregunta de INFINITO

F9-O había observado en los modelos después de **BC**, antes de PPO, una señal de capacidad para elegir las **dos salidas correctas** de dos historias que solo difieren por la pista inicial. Posteriormente PPO, que siempre tenía F4 disponible, disminuyó esa señal. ¿Conservaría PPO esa habilidad si se entrenase con **largos intervalos sin leer F4**?

La respuesta en los nuevos mundos de F9-P es **NO** para el tratamiento específico. La política \`ppo_blackout\` aprende a **llegar al cruce y tomar una salida fija**, no a recuperar qué salida corresponde a la pista original. No atribuirle «memoria», «dependencia de H» ni «comprensión» por obtener 50 % de aciertos en episodios sueltos.

## Integridad, nuevos mundos y contrastes

- Punto de partida idéntico por índice: los mismos cuatro **F9-O continuous6_after_bc.zip** (fuente Actions #40957255987) usados como ancla BC y cargados dos veces para PPO, con SHA-256 verificado. Cuatro réplicas independientes de entrenamiento, arquitectura context8 LSTM H=64, 380.552 parámetros, observaciones públicas de 148 más lectura F4 de 512, siete acciones nativas, recompensas originales.
- Dos grupos PPO con **20.096 pasos reales** cada uno y los mismos hyperparámetros y pesos iniciales por índice. Nueva población TRAIN S13 \`550000+index*10000+j\` alternando cue original e invertida, totalmente distinta del TRAIN previo (500000+). Las trayectorias de PPO pueden divergir.
- Reloj exógeno global: **112 posiciones consecutivas de cada 128 pasos** programadas para ambos agentes, **17.584 oportunidades** por agente. \`ppo_sham\` no elimina F4 útil (efectivos 0); \`ppo_blackout\` anula los 512 canales F4 en esas posiciones: efectivos **17.581/17.584** para semilla 0 y **17.584/17.584** para 1,2,3. Ni escritor, ni selector, ni imagen pública, ni reward cambiaron. «Igualadas» están las oportunidades de reloj y presupuesto de pasos, **no** la cantidad de contenido F4 disponible, que es precisamente la intervención.
- Barrera SHA-256 y parámetros de ocho modelos PPO y cuatro checkpoints de arranque BC pasó **ANTES** de abrir confirmación.
- Partición confirmatoria inédita \`700000+index*10000+j\`, 40 mundos pareados S13 por semilla (**160 pares / 320 episodios por grupo**), y 20 S7 (**80 pares**), en ninguno de los rangos de confirmación F9-M (300000+), F9-N (400000+), F9-O (600000+) ni holdout histórico (900000+). La confirmación \`700000+\` queda ahora **consumida**.
- Protocolo y criterio [congelados antes de entrenar](./INFINITO_F9P_PPO_MEMORY_PRESERVATION_PROTOCOL.md). **No** se ha modificado el criterio para acomodar los resultados. Las cuatro semillas son las réplicas de políticas, no los 320 episodios.

## Primario: ambos primeros compromisos correctos por par S13 con F4 ausente

El evaluador F9-N registra primera pista efectivamente vista, llegada al cruce, **primera entrada al brazo upper/lower**, éxito y timeouts. Para evaluar memoria de identidad, se invierte la pista en el **mismo mundo**, manteniendo geometría. El denominador incluye los pares donde no llega al cruce ni decide.

**Condición primaria:** \`cue_lost_continuous\`, leer F4=0 después de haber visto la pista inicial y en cada frame donde ya no es visible; si se vuelve a ver, ese frame puede recibir F4 de nuevo. Cada episodio es autónomo, sin acciones teacher en confirmación.

| Métrica S13 | Ancla BC, sin PPO | PPO con F4 (sham) | PPO con cortes largos |
|---|---:|---:|---:|
| **Ambas primeras salidas correctas, pares** | **36/160 (22,50 %)** | **20/160 (12,50 %)** | **0/160 (0 %)** |
| Pista observada | 320/320 | 298/320 | 282/320 |
| Llega al centro de la bifurcación | 316/320 (98,75 %) | 263/320 (82,19 %) | **320/320 (100 %)** |
| Primera entrada a un brazo | 316/320 | 263/320 | 320/320 |
| Primera salida correcta, todos los episodios | 163/320 (50,94 %) | 130/320 (40,63 %) | **160/320 (50 %)** |
| Primera salida correcta entre episodios con compromiso | 163/316 (51,58 %) | 130/263 (49,43 %) | **160/320 (50 %)** |
| Timeouts | 4/320 | 57/320 | **0/320** |
| Pares con lados opuestos al primer compromiso | 65/160 | 37/160 | **0/160** |

| Índice | Ancla BC correctos /40 | Sham /40 | Blackout largo /40 |
|---|---:|---:|---:|
| 0 | 15 | 0 | 0 |
| 1 | 14 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 7 | 20 | 0 |
| **Total** | **36** | **20** | **0** |

El tratamiento tiene deltas preregistrados frente al sham **[0, 0, 0, −20]** por semilla; **0/4 semillas mejoran**. Criterio de mejora ≥3/4 **NO ALCANZADO**. Más aún, **ninguno** de los 160 pares del grupo blackout elige lados opuestos en sus dos historias, aunque la pista cambie; es un patrón consistente con **elección de salida independiente de la identidad**, es decir una estrategia fija, no con memoria operativa de esa pista.

### Distinción que importa para INFINITO

**Llegar al cruce (100 %) no implica recordar una pista (0/160 pares).** Una métrica de éxito individual de 50 % se obtiene sin distinguir cue original/invertida. F9-P vuelve a demostrar por intervención y contrafactual por qué la métrica central de memoria NO debe ser recompensa, velocidad ni navegación.

La fracción de elección correcta condicionada al compromiso también sufre selección por trayectorias distintas; no se la debe leer como efecto causal puro. En blackout llegan todas las observaciones a comprometidas, así que aquí la tasa 50 % describe directamente la elección; en sham no llegan/deciden todas.

## Intervenciones secundarias: F4 íntegra, corte corto y estado H/C

| Dos primeras salidas correctas S13 | BC sin PPO | PPO sham | PPO blackout |
|---|---:|---:|---:|
| \`normal\`, F4 íntegra | **146/160** | 104/160 | 29/160 |
| \`fixed_4\`, quitar F4 t=3..6 y restaurar | **160/160** | 103/160 | 18/160 |
| \`cue_lost_continuous\`, pérdida prolongada | **36/160** | 20/160 | **0/160** |

El PPO blackout empeora también el rendimiento **cuando F4 sí está disponible**: con F4 normal resuelve 29/160 pares frente a 104/160 del control y 146/160 del ancla BC. Por tanto, no se debe concluir que la técnica aprendió independencia y solo falló una prueba extrema: **degradó la decisión dependiente de la identidad** en general.

**Ablación de estado recurrente en el cruce:** manteniendo \`cue_lost_continuous\`, reiniciar H/C justo al entrar en la bifurcación da **21/160 pares** en sham (frente a 20/160 sin reset) y **0/160** en blackout (frente a 0/160). La intervención se aplicó realmente en 263 episodios del sham y 320 del blackout. En el tratamiento **no queda ventaja de memoria que pueda atribuirse a H/C**; un reset sin cambio sobre un agente colapsado **no prueba** que el LSTM nunca contenga información útil en otros estados/experimentos. El reset puede alterar orientación/navegación además de memoria, por lo que no es una decodificación selectiva de identidad.

**S7 transferencia:** ambos PPO dan **0/80 pares** cue-lost; ancla BC 3/80. Con F4 normal ambos PPO 0/80, y ancla BC 7/80. El entrenamiento S13 no generaliza de forma robusta a S7.

## Conclusión científica

**F9-P refuta la hipótesis concreta de que un PPO con blackout exógeno del 87,5 % del tiempo, partiendo de los checkpoints BC de F9-O, preservaría una decisión autónoma por memoria de la pista.** Al contrario, bajo pérdida F4 prolongada la política cambia a una **estrategia fija** que llega al cruce sin utilizar cue, lo que explica el 50 % de éxito individual y 0 % de pares contrafactuales correctos. Con F4 normal también hay degradación fuerte, señal de que el tratamiento perjudica la política, no solamente su resistencia al blackout.

F9-O enseñó una señal parcial tras BC y F9-P muestra que ni PPO limpio ni este PPO con blackout largo la mantienen de forma consistente. **No equivale** a demostrar que sea imposible conservar memoria interna con RL: quizá el blackout global del 87,5 % (que a veces oculta la pista antes de ser codificada en F4), la distribución de episodios de PPO, el equilibrio de recompensa o el valor de exploración sean inadecuados. Esas son **hipótesis nuevas, no explicaciones confirmadas**.

### Siguiente pregunta metodológica sin desviarse del objetivo

En vez de aumentar dificultad o perseguir éxito por navegación, una investigación futura debería diseñar un **punto de decisión de recuerdo controlado**: mismo historial y geometría hasta la bifurcación, F4 ausente, manipulaciones separadas de estado H/C e identidad de la pista, para distinguir la existencia de una representación, su causalidad para la acción y su preservación bajo PPO. Si se decide entrenar de nuevo, preregistrar otro conjunto **nuevo** y usar criterios de **pares con ambas primeras decisiones correctas**, no seleccionar modelos según retorno.

No hay evidencia en estas pruebas de conciencia, agencia general ni acceso a recuerdos deliberados: son experimentos de memoria funcional limitada en una tarea MiniGrid. Este resultado **negativo** queda versionado sin ajustes posteriores sobre los mundos \`700000+\`.
