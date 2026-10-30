# INFINITO F9-M — Resultado de entrenar nuevas políticas con dropout temporal F4

**Fecha:** 29/10/2026. **Estado:** completado; [GitHub Actions #40950579984](https://github.com/webtilians/principiodelTodo/actions/runs/40950579984) **SUCCESS**, integrity + 4 train + 4 confirm + aggregate. [PR #20](https://github.com/webtilians/principiodelTodo/pull/20), borrador sobre F9-L #19. [JSON versionado completo](./INFINITO_F9M_RESULTS_2026-10-29.json); [artifact original CI #12932250604](https://github.com/webtilians/principiodelTodo/actions/runs/40950579984/artifacts/12932250604).

## Resumen ejecutivo y criterio anterior al resultado

Se entrenaron **8 políticas nuevas** (4 semillas independientes × 2 brazos: dropout real vs sham igualado). La hipótesis primaria predefinida era aumentar el éxito autónomo S13 bajo interrupción F4. El agregado confirma una **mejora descriptiva de 26,875 puntos porcentuales**: sham 13,75 % vs dropout 40,625 %, **positiva en 4/4 semillas**. En S7, bajo interrupción se mejora del 41,25 % al 67,5 %.

**Sin embargo no se demuestra robustez autónoma de memoria en la tarea larga**: en S13/blackout el agente dropout resuelve **ambos cues correctamente en solo 1/80 pares** (sham: 0/80). Además, el blackout llega a aplicarse en el 87,5 % de episodios del dropout vs 97,5 % en sham, por divergencias de las propias trayectorias. La diferencia agregada no equivale a un contraste causal condicionado a igual exposición al blackout.

El criterio preregistrado de mejorar >=3 de 4 semillas se cumple; la comprobación sustantiva de ambas salidas correctas arroja una **mejora pequeña, 1/80**, insuficiente para afirmar que el sistema utiliza memoria interna de forma autónoma y fiable en S13.

## Diseño y procedencia

- Dos grupos \`sham\` y \`dropout\` por semilla (índices 0–3); nuevos pesos, **380.552 parámetros**, input 660, LSTM 64, ocho eventos de memoria externa F4 (512 canales, 64 activos y 448 padding). Las políticas de cada par tenían SHA-256 **idéntico de pesos iniciales**.
- Ambos grupos recibieron los mismos 600 mundos S7 train con ambos cues, 40 épocas BC, las mismas acciones y el mismo entrenamiento RecurrentPPO, **20.096 pasos PPO efectivamente ejecutados** por política (20.000 solicitados). PPO puede experimentar trayectorias divergentes al cambiar su política, así que igualar número de pasos NO iguala las transiciones vistas.
- Schedule de blackout en ambos grupos, 50 % episodios y longitudes 2 o 4 pasos tras el primer objeto key/ball visto en frame **público** más dos observaciones. Arm \`sham\` enmascara solo padding que ya es cero; arm \`dropout\` enmascara los 512 canales F4 **entregados al actor** sin modificar el almacenaje ni el selector. No existe señal de docente, cue correcto, reward futuro o geometría privilegiada en el scheduler.
- **Enmienda antes de entrenar y abrir confirmación:** el primer disparador previsto («ningún key/ball visible») no activó máscaras durante smoke tests. Fue corregido de manera visible a un reloj de dos pasos después de la **primera detección pública**; no se utilizaron resultados científicos para elegirlo. Dos runs CI de integridad previos fallaron; los tests de la ejecución final pasaron antes del entrenamiento.
- Una nueva partición prospectiva de **20 mundos emparejados por semilla y tamaño** usa \`300000 + index*10000 + j\`; es independiente de las particiones train, development y el holdout histórico (900000+), que no se abrió. La confirmación comenzó **solo cuando los ocho entrenamientos y sus hashes/budgets pasaron**. Se usaron S7 y S13 con esta partición; no se reentrenó nada al ver confirmación.
- Probes H: ridge lineal alpha fija 10, 80 pares S7 **train** por semilla. Fijo en H-natural o readaptado al corte; evaluación S7/S13 únicamente en la nueva partición. Probes no actualizan política.

## Primaria: navegación autónoma, nuevos mundos

Cada fila son **160 episodios por brazo** (4 políticas × 20 mundos × 2 cues); 80 pares. Comparar cada semilla, no tratar 160 episodios como 160 réplicas de entrenamiento.

| Tarea | F4 entregada al actor | Sham éxito | Dropout éxito | Diferencia | Sham ambos correctos | Dropout ambos correctos |
|---|---|---:|---:|---:|---:|---:|
| S7 | Normal | 90,625 % | **100 %** | **+9,375 pp** | 65/80 | **80/80** |
| S7 | Eliminada tras ver pista | 41,25 % | **67,5 %** | **+26,25 pp** | 5/80 | **28/80** |
| S13 | Normal | 59,375 % | **61,875 %** | +2,5 pp | 28/80 | 35/80 |
| **S13** | **Eliminada tras ver pista** | **13,75 %** | **40,625 %** | **+26,875 pp** | **0/80** | **1/80** |

Variación por semilla S13 **blackout**:

| Índice | Sham | Dropout | Diferencia |
|---|---:|---:|---:|
| 0 | 30 % | 50 % | +20 pp |
| 1 | 25 % | 37,5 % | +12,5 pp |
| 2 | 0 % | 37,5 % | +37,5 pp |
| 3 | 0 % | 37,5 % | +37,5 pp |

La tasa media de intervención **realmente aplicada** en S13/blackout no es igual: sham 97,5 %, dropout 87,5 %, a causa de trayectorias diferentes (algunos episodios no llegan al estado que dispara el corte). Se registró en lugar de excluir episodios. Timeout: sham **72,5 %**, dropout **18,75 %**. De ahí que parte de la mejora de éxito pueda deberse a que el dropout **evita bloqueos o alcanza la salida más a menudo**, no necesariamente a recuperar la pista correcta.

En S13/normal la media cambia poco y existe gran heterogeneidad: dropout menos sham por semilla = **+20, +10, −52,5, +32,5 puntos**. Por eso no se debe describir la navegación limpia como uniformemente preservada o mejorada.

En S7/blackout, **ambos correctos pasa de 5 a 28 pares/80**, una evidencia conductual más relevante que el éxito global. En S13/blackout ambos correctos pasa de 0 a 1: sigue siendo prácticamente ausencia de resolución de pistas contrafactuales por pares.

## Secundaria: primera decisión en bifurcación emparejada, recorrido guiado

F4 original restaurada justo en el fork; en cada par el prefijo de acciones del profesor, la observación actual y la F4 del fork son idénticos entre las pistas opuestas. Son diagnósticos de política bajo teacher forcing, NO navegación autónoma.

| Tarea | Condición antes del fork | Sham primera acción correcta | Dropout primera acción correcta | Sham H decodificable (probe adaptado) | Dropout H decodificable |
|---|---|---:|---:|---:|---:|
| S7 | Normal | 97,5 % | 100 % | 100 % | 100 % |
| S7 | Corte desde post-cue | 51,875 % | **93,75 %** | 100 % | 100 % |
| S7 | Corte desde mitad | 63,75 % | **99,375 %** | 100 % | 100 % |
| S7 | Corte último paso | 86,25 % | **100 %** | 100 % | 100 % |
| S13 | Normal | 87,5 % | 92,5 % | 83,75 % | **95,625 %** |
| S13 | Corte desde post-cue | 50 % | **60 %** | 55,625 % | 64,375 % |
| S13 | Corte desde mitad | 50 % | **75 %** | 61,25 % | **86,25 %** |
| S13 | Corte último paso | 80,625 % | 80,625 % | 65,625 % | **90 %** |

Especialmente interesante: **S13/cut_halfway**, con acción correcta 50 %→75 % y H decodificable 61,25 %→86,25 %, sobre los 160 casos; **ambas acciones correctas** por par 6/80→40/80. Bajo estas historias guiadas, el aprendizaje del dropout sí parece haber mejorado la disponibilidad y el uso de información histórica frente al control.

Pero la gran distancia entre **40/80 pares correctos guiados en S13/cut_halfway** y **1/80 pares de éxito autónomo con borrado de F4 tras la pista** demuestra que esta robustez **no se ha trasladado de forma suficiente a la navegación integral**. En autonomía también importan llegar a la bifurcación, duración del episodio y estabilidad de trayectoria sin profesor.

## Conclusión honesta

**Sí:** con la arquitectura y presupuesto igualados, introducir blackout temporal F4 durante BC y PPO **mejora el comportamiento bajo interferencia** en los nuevos mundos. Esta comparación no recicla los checkpoints F9-I ni la partición de development. La mejora se observa en las cuatro semillas S13 del resultado primario y es fuerte en S7; la decisión guiada y el probe H S13 apoyan una representación más útil bajo algunos cortes.

**No todavía:** el resultado no demuestra que la LSTM **sustituya funcionalmente** a F4 en navegación autónoma S13; no hay suficientes pares key/ball resueltos bajo borrado prolongado. La ganancia bruta en acierto S13 puede reflejar menor atasco, sesgo de una salida y distinta exposición al blackout, no recuperación correcta de la pista. No declarar éxito de memoria autonóma S13 ni conciencia.

**Límites:** cuatro réplicas de entrenamiento, tarea MiniGrid, mundos confirmatorios generados por el mismo simulador, intervención adaptada a lo visto por el evaluador, probes S7 de entrenamiento, teacher forcing fuera de distribución, no nueva evaluación en otro entorno/semillas de entrenamiento adicionales. La partición prospectiva **ya fue abierta** por F9-M: no se debe usar ahora para elegir hiperparámetros y volver a llamarla confirmación.

**Próximo protocolo posible (NO ejecutado):** separar la perturbación de F4 de los errores de navegación iniciales mediante un currículo de mundos largos S13 o tareas nuevas con enseñanza exclusivamente en train, incluir exposición al blackout exógena idéntica para todos los modelos (control de cobertura), registrar decisiones en bifurcación durante navegación verdaderamente autónoma, y definir un segundo conjunto confirmatorio nuevo antes de ajustar. Nunca reabrir la partición 300000 para seleccionar diseños.
