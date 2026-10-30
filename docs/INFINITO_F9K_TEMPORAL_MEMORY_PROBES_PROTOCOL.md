# INFINITO F9-K — Protocolo predefinido: ¿qué información de la pista permanece en LSTM?

Fecha: 29 de octubre de 2026. Rama basada en la PR #17 (F9-J); PR #16 sigue siendo su base, PR #15 es auditoría alternativa. **Este documento se fija antes de ver los resultados F9-K.**

## Pregunta y predicciones

F9-J encontró observación actual y readout externo F4 idénticos en la bifurcación, pero cambiar únicamente el estado recurrente cambió la siguiente acción en 31/40 pares S13 con prefijo del profesor. F9-K pregunta si un observador lineal *fijo* puede recuperar cuál de las dos pistas contrafactuales se vio previamente a partir de (1) hidden+cell LSTM H, (2) readout externo M, (3) H+M y (4) observación pública actual O. No equivale a demostrar representación semántica general ni conciencia.

**Predicción:** si el estado LSTM transporta la identidad de la pista decodificable, H deberá superar el 50 % en pares invertidos con observación actual idéntica tras dejar de ver la pista, especialmente en el cruce. Una separación positiva puede reflejar una política específica de la tarea; no presupone memoria conceptual. Si H no es linealmente decodificable, ello no demuestra ausencia de información (podría requerir un decodificador no lineal).

## Diseño congelado

- Checkpoints originales **F9-I context8 y zero**, cuatro semillas 0–3, 660 canales, 64 unidades de hidden y celda, siete acciones nativas, capacidad externa de ocho eventos (512 canales con padding). Fuente: [F9-I Actions #40928547873](https://github.com/webtilians/principiodelTodo/actions/runs/40928547873). SHA-256 y report.json validados por semilla. **Sin nuevo entrenamiento del agente.**
- El entrenador del *probe* usa exclusivamente **80 mundos S7 train pareados por semilla**, cada uno con su pista normal e invertida. La política F9-I original ya había visto mundos de la partición train durante su entrenamiento; esto no es un nuevo conjunto externo. No se abre heldout.
- Evaluación: **10 mundos pareados S7 development** y **10 S13 development** por semilla. El *probe* se ajusta sobre S7 train y se transfiere a S13 development **sin ninguna muestra S13 para entrenar el probe**. El propio agente ya fue entrenado solo con S7, por lo que S13 sigue siendo zero-shot para la política y para el probe, condicionado a estas familias de mundos.
- Las dos variantes contrafactuales usan exactamente las mismas acciones del profesor hasta el primer instante en que sus siguientes acciones difieren. La etiqueta binaria indica **identidad real del objeto visto (key o ball)**, leída únicamente por el evaluador; la pista original puede ser aleatoriamente cualquiera de las dos. La versión invertida lleva necesariamente la etiqueta opuesta. Ni la etiqueta ni el observador se pasan al actor o selector F4. Es una clasificación diagnóstica de una historia guiada, **no éxito de navegación autónoma**.
- Por cada par: comprobar acciones y visibilidad idénticas antes del cruce; captura en **post_cue** (primer frame después de la última pista visible), **halfway** (mitad discreta del intervalo entre ese frame y el cruce) y **fork** (primer instante de acción correcta diferente). No seleccionar instantes basándose en aciertos del probe. Si no existe intervalo posterior a la pista, fallar y documentarlo.
- A cada instante se captura el estado H **anterior** a procesar la observación actual, pero posterior a procesar la historia precedente, y el readout M **después de** actualizar F4 con la observación actual, igual que F9-J. En los tres instantes la observación O de ambos miembros debe ser *bit a bit* idéntica. Si no lo es, **falla el experimento sin descartar pares**.
- Decodificador lineal ridge: regresión a etiquetas -1/+1; centrado y escala aprendidos solo sobre train; regularización fija alpha=10.0; clasificación con umbral 0 sin ajuste en dev. Se resuelve en espacio dual para no alterar la capacidad de cada modalidad. Se entrena un probe nuevo para cada (semilla, etapa, modalidad), nunca se actualizan pesos de la política original.
- Modalidades de context8: **H** (128), **M** (512), **H+M** (640), **O** (148). En zero se evalúan **H** y **O** como controles. Las dimensionalidades son diferentes **por diseño de diagnóstico**: comparar accuracies no permite atribución causal exclusiva a la capacidad.
- Registrar por semilla, tarea, etapa y modalidad la **exactitud en 20 episodios**, **pares con ambas etiquetas correctas de 10** y **pares con predicción invertida**, además de igualdad de observaciones, igualdad externa, distancia euclídea H y pasos desde la última pista. Mostrar resultados malos y buenos; no elegir etapas según rendimiento. Describir incertidumbre entre **cuatro semillas de entrenamiento**, sin tratar 80 episodios como 80 entrenamientos independientes.

## Falsadores y límites

- Si **O** predice la inversión por encima del azar en los checkpoints tras pista, existe un problema de observaciones no emparejadas o de código: el test debe fallar por igualdad de O antes de estimar el probe.
- Si **M** sigue siendo decodificable en el fork de F9-J donde los readouts se igualan, hay que auditar cómo se han generado los pares o el entrenamiento del probe; no concluir que F4 conserva el primer evento.
- Si **H_zero** también decodifica, la diferencia de pista puede formarse a partir del input público sin F4; comparar ambos agentes, no adjudicar el efecto únicamente a F4.
- Un buen probe no prueba que la política **utilice** cada dimensión decodificable: la atribución causal local sigue siendo la de F9-J. Un éxito del probe tampoco demuestra abstracción o generalización abierta.
- Ningún dato development se usa para entrenar, elegir alpha, umbral o etapas. Sin inferencias causales entre políticas entrenadas por separado. Sin cambios al PC/VPS.

## Criterio de lectura

1. H reproduce la etiqueta por encima del azar y mantiene predicciones invertidas en **S13/fork** con O y M iguales: evidencia de información histórica recuperable en el estado recurrente que transfiere a S13 bajo el prefijo guiado.
2. H+M mejora H a etapas tempranas pero no en fork: compatible con F4 como soporte temporal; **no** demuestra que F4 sea innecesaria.
3. Evidencia insuficiente o no lineal: diseñar después una prueba no lineal predefinida y una prueba causal temporal; no repetir selectivamente hasta lograr un porcentaje alto.

GitHub Actions generará JSON por cada semilla y un agregado; informes de resultados únicamente después de CI real.
