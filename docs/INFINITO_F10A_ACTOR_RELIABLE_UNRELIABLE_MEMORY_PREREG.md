# INFINITO F10-A — Nueva generación de actores ante evidencia externa auténtica, engañosa o ausente

**Preregistro del 2026-11-02 ANTES de escribir código específico de entrenamiento o abrir nuevos mundos.** Nueva línea experimental tras F9-Z `SUPPORT_NOT_ACHIEVED`. No se amplía ni se entrena aún la cabeza READ. **Los actores F9-P originales permanecen como artefactos históricos congelados; estos nuevos actores NO son confirmaciones de F9-V/W/X/Y/Z.**

## Pregunta y alcance

¿Entrenar desde el mismo checkpoint F9-O de imitación un actor recurrente para tomar la decisión correcta en tres estados de evidencia externa —F4 legítima, F4 con identidad falsa y F4 vacía— produce un nuevo actor más robusto que un control entrenado en las mismas trayectorias pero sin degradar F4? Y ¿esa nueva generación dispone de soporte causal `D=+1` y `D<=0` en **cada uno** de los 16 estratos actor×cue×historia, en mundos DEV distintos de TRAIN, para poder plantear después un predictor del beneficio de READ?

No se supone que el actor pueda identificar de forma infalible un recuerdo falso si la observación y su historia no contienen un indicador fiable. Este experimento **no añade una marca oracular de autenticidad**, por lo que la correcta respuesta ante recuerdos contradictorios puede depender de recordar la pista en H/C. Si se resuelve solo con H/C, el experimento puede no obtener soporte para READ. Este resultado negativo se reportará, no se maquillará.

## Medio y etiquetas

Entorno nativo `MiniGrid-MemoryS13-v0`, 7 acciones, entrada 660 = 148 observación pública + 512 canales de F4 `SelectiveF4(context,capacity=8)`. Cuatro réplicas independientes, fuente **F9-O `continuous6_after_bc.zip`** (Actions #40957255987), cada checkpoint verificado con informe, hashes y comparadores de F9-O; misma inicialización y parámetro base para cada par de brazos de réplica. Sin cambiar la política ni el código del F4 original. **Esta es una tarea de entrenamiento de robustez nueva, no de la misma distribución de experiencia on-policy que F9-P.**

Para cada mundo TRAIN se generan **ambas pistas** key/ball por intervención de cue original; se mantiene constante geometría, posiciones, ruta docente hasta el cruce y secuencia de acciones con etiqueta de entrenamiento. El docente BFS puede ver la solución **solo durante TRAIN**. El actor solo recibe vista pública, F4 y su H/C y no recibe `correct_side`, `D` ni privilegios del docente.

## Factorial de entrenamiento (secuencias enteras, sin filtrado)

- **TRAIN**: `12_100_000 + 10_000*index + j`, `j=0..127` (128 mundos por réplica, 512 mundos únicos).
- **DEV**: `12_200_000 + 10_000*index + j`, `j=0..31` (32 mundos nuevos por réplica, 128 mundos únicos); `12_300_000+` reservado CONFIRM sin ejecutor.
- Para cada mundo: 2 cue inversos × 2 regímenes F4 historial (normal / F4=0 en prefijo pospista hasta cruce) × 3 condiciones en la fase pospista `clean`, `adversarial`, `blank`. Total **1536 secuencias TRAIN por réplica**, todas y cada una incluidas (4×1536 = 6144 secuencias de entrenamiento; hay mundos repetidos por factorial y NO son independientes). El valor de `D` no se calcula en TRAIN para decidir inclusión.
- `clean`: readout original de SelectiveF4.
- `adversarial`: tras salir de vista la pista, invertir únicamente los dos canales de identidad key/ball de cada slot **ocupado** del readout de F4; se conservan tiempos, orden, color, ocupación y vista pública. La escritura en el almacén mantiene los eventos auténticos; se interviene la **lectura**. Es una corrupción físicamente artificial declarada, sin marca externa de origen.
- `blank`: lectura F4=0 tras la pista; almacén preservado. En `postcue_off` los valores F4 se anulan únicamente antes de llegar al fork, manteniendo el frame del cruce como condición de lectura libre (análoga a F9-Q).
- El mismo docente de acciones/frames y la misma geometría para los tres valores de evidencia; no hay acción de un oráculo condicionada a si el recuerdo es limpio o falso.

## Brazos y presupuesto fijo

Cada réplica parte de exactamente el mismo F9-O BC verificado, con idéntico `RecurrentPPO MlpLstmPolicy`, LSTM64, ancho660 y sus pesos iniciales, sin un nuevo tamaño de red.
- `evidence_mixed`: entrenamiento BC sobre las 1536 secuencias factoriales TRAIN, con los tres tipos reales de F4.
- `evidence_clean_control`: **mismas 1536 trayectorias, acciones, orden y estados públicos**, pero se presenta siempre F4 `clean` como input de entrenamiento en lugar de las condiciones `adversarial`/`blank`. Las tres copias están presentes para igualar la dosis de supervisión, pero no igualan la distribución de observaciones; se documentará explícitamente.
- Optimización idéntica de las dos políticas desde el mismo estado inicial, `torch.manual_seed(121000+index)`, mezcla determinista y emparejada de secuencias, **10 épocas completas** BC, Adam `lr=0.001`, batch64, grad clipping0.5. Sin PPO posterior, sin selección de época por DEV. Guardar ambos checkpoints con SHA y pérdidas por época. El control no es una política completamente independiente de memoria: también usa F4 auténtica. El checkpoint F9-O original se evalúa además como tercera referencia sin entrenamiento F10-A.

## Evaluación de actores ANTES de cualquier cabeza READ

Para cada mundo DEV y cada par cue×regime, construir el mismo prefijo docente contrafactual hasta el primer fork (autonomía solamente después del fork), desde H/C congruente con la historia observada. Cada modelo se compara en **tres intervenciones en el mismo estado público y H/C**:
- `SKIP`: todos los canales F4 a 0 en fork y continuación.
- `READ_clean`: readout auténtico en fork, F4=0 después.
- `READ_adversarial`: misma estructura de slots y metadatos, pero identidad key/ball invertida únicamente en el fork, F4=0 después.
Evaluar primera salida escogida: `D_clean = first_correct(READ_clean) - first_correct(SKIP)`; `D_adversarial = first_correct(READ_adversarial) - first_correct(SKIP)`. Guardar por mundo, réplica, cue, historia, brazo, condición, y conteos `+1/0/-1`, sin postseleccionar positivos. Informar también navegación autónoma end-to-end desde el reset (por el actor entrenado, sin tutor) en DEV, evaluada sin corrupción y con corrupción, para evitar confundir éxito al primer cruce con capacidad real de recorrido.

## Gates y límites preregistrados

1. **Contrato de entrenamiento:** fuentes F9-O verificadas, anchos/arquitectura/pesos iniciales coinciden, misma cantidad de ejemplos/épocas/batches y acciones entre brazos, manipulación exclusiva del readout F4, todos los mundos fijados consumidos, y cero etiquetas privilegiadas de DEV en entrenamiento.
2. **Robustez de evidencia (DEV exploratorio):** `evidence_mixed` debe tener **>= 0** diferencia de primeras salidas correctas frente al control clean en `READ_clean`, y **>= +5/128** en `READ_adversarial`, en **al menos 3 de las 4 réplicas**; además la corrección `READ_adversarial` del mixto debe exceder SKIP del mismo mixto en >=0 réplicas (esto último es diagnóstico, no criterio). Los conteos se publican sin filtrar independientemente de gate.
3. **Soporte causal para el *futuro* lector:** en `evidence_mixed`, para los **16 estratos** actor×cue×historia, al menos **4 mundos D_clean=+1 y 4 mundos D_clean<=0** de 32 DEV. La separación por actor no es opcional. Si falla una sola celda: `INSUFFICIENT_POSITIVITY`, **no entrenar lector**. Evaluar si también hay `D_adversarial=-1` en cada estrato, pero ello no se exige como soporte de D_clean. No mezclar decisiones + de clean con - de adversarial para fabricar soporte.
4. Las puertas 2 y 3 se reportan por separado. `EVIDENCE_ROBUSTNESS_NOT_ESTABLISHED` si falla gate 2; `INSUFFICIENT_POSITIVITY` si falla gate 3; `ACTOR_DEV_FEASIBLE` solo si pasa contrato y **ambos** gates. **DEV no confirma resultados independientes**. Ningún cambio de arquitectura o entrenamiento tras mirar DEV puede abrir `12_300_000+`: requerirá experimento nuevo.

## Validez, riesgos y NO afirmaciones

La F4 falsa está fabricada por una intervención del observador y es potencialmente indistinguible de la verdadera sin cotejo con H/C. El actor puede adquirir una heurística de no leer: en ese caso será robusto a la corrupción pero no se habrá creado valor de memoria; por eso el gate de positividad es independiente del de robustez. El protocolo mantiene el mismo problema de prefijo docente: no afirmar navegación plenamente autónoma, memoria contextual abstracta ni inteligencia general. Este es un **nuevo entrenamiento de actores**, explícitamente separado de los experimentos F9 y de la nueva fase futura de aprendizaje del lector.

**No reutilizar** anteriores confirmaciones `8300000+`, `9300000+`, `9600000+`, `10300000+`, ni semillas previas `11100000+` piloto F9-Z. Reservar `12300000+` sin acceso en esta fase.
