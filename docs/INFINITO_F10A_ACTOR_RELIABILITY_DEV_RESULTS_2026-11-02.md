# INFINITO F10-A — Nueva generación de actores: aprendizaje con memoria auténtica, falsa y ausente

**Resultado DEV del 2026-11-02, no confirmación independiente.** Protocolo preregistrado **antes de implementar y ejecutar**: commit `e44c30704ec9af347da7a3dfe7cbc8d8c64a5065`, [protocolo](./INFINITO_F10A_ACTOR_RELIABLE_UNRELIABLE_MEMORY_PREREG.md). [GitHub Actions #41435811684](https://github.com/webtilians/principiodelTodo/actions/runs/41435811684) finalizado **SUCCESS técnico** en cuatro réplicas y el agregado. Los cuatro reportes DEV, informes de entrenamiento y ocho checkpoints nuevos se conservan como artifacts de este run.

## Hipótesis y límites

F9-Z fue `SUPPORT_NOT_ACHIEVED`: variar capacidad F4 y duración READ de cuatro actores F9-P congelados no produjo positividad en las 16 celdas. En F10-A se volvió al checkpoint **F9-O `continuous6_after_bc.zip` de cada réplica**, sin tocar los actores originales F9-P. Se compararon dos actores con la misma arquitectura `RecurrentPPO MlpLstmPolicy` de 660 entradas, LSTM64 y 380552 parámetros:
- **Mixto:** entrenamiento con F4 legítima, identidad key/ball invertida después de la pista, o vacía.
- **Control limpio:** las mismas trayectorias, acciones docentes, número de ejemplos, inicialización y 10 épocas BC; readout F4 siempre legítimo (salvo el apagón historial común).

Se generaron **128 mundos TRAIN nuevos por réplica `12100000+`**, 1536 secuencias por réplica y brazo (2 cues × 2 historias × 3 calidades); prueba DEV **32 mundos nuevos `12200000+` por réplica**. La cabeza READ NO se entrenó. La escritura real de F4 y la vista pública no se cambiaron; la intervención invierte solo la identidad de registros ocupados en la lectura. Sin etiquetas de autenticidad: el agente debe recurrir a sus otras señales si quiere resistir contradicciones. La tarea usa un docente BFS para los ejemplos TRAIN y para el **prefijo hasta el primer cruce** en la evaluación causal; se informa además navegación autónoma desde el reset.

## Resultado 1 — Primer cruce causal: robustez por supresión del valor de F4

Los valores siguientes agregan **cuatro réplicas × 32 mundos × dos cues × dos regímenes = 512 estados de primer cruce** por grupo:

| Grupo | Correctos SKIP /512 | Correctos READ limpio /512 | Correctos READ falso /512 | D limpio: D+ / D0 / D− |
|---|---:|---:|---:|---|
| F9-O BC original (sin reentrenar) | 346 | 374 | 374 | 28 / 484 / 0 |
| **Actor mixto** | **461** | **457** | **465** | **3 / 502 / 7** |
| Actor control limpio | 354 | **495** | 361 | 141 / 371 / 0 |

La política mixta sufre mucho menos que el control ante un F4 falso (**465 vs 361 primeros cruces correctos**), pero también deja prácticamente de extraer beneficio causal de F4 limpio (**3 D+ frente a 141 D+ del control**, y **7 D−**). Su precisión con READ limpio es **457**, inferior incluso a su propio SKIP (**461**). Esto es **compatible con la estrategia de apoyarse casi por completo en H/C y desatender F4**, aunque estos conteos no identifican por sí solos un mecanismo interno exacto. La intervención READ falsa no incluyó una señal de autenticidad que el actor pudiera leer en el fork.

### Resultado por réplica

| Réplica | READ limpio mixto | READ limpio control | Delta limpio | READ falso mixto | READ falso control | Delta falso | Positivos D_clean mixto por sus 4 celdas |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 | 116 | 128 | −12 | 116 | 87 | +29 | 0 / 0 / 0 / 0 |
| 1 | 115 | 128 | −13 | 118 | 83 | +35 | 0 / 0 / 0 / 1 |
| 2 | 122 | 124 | −2 | 122 | 104 | +18 | 0 / 0 / 0 / 0 |
| 3 | 104 | 115 | −11 | 109 | 87 | +22 | 0 / 0 / 0 / 2 |

Orden de celdas: `key-normal, ball-normal, key-postcue_off, ball-postcue_off`. Son 32 mundos por réplica y por celda. Las cuatro diferencias de robustez ante F4 falso favorecen al mixto, **pero las cuatro diferencias ante F4 limpio favorecen al control**. La condición necesaria preregistrada era no perder con F4 limpio y ganar >=5 aciertos falsos en >=3 réplicas. El número de réplicas que supera ambos requisitos es **0/4**.

## Resultado 2 — Soporte causal para un futuro predictor READ

El preregistro exigía, **en las 16 celdas actor × cue × historia**, >=4 mundos con `D_clean=+1` y >=4 con `D_clean<=0`. Los actores mixtos producen **3 positivos en total sobre 512 estados**, repartidos únicamente en `ball+postcue_off` de réplicas 1 (1) y 3 (2). **Ninguna** de las 16 celdas alcanza 4 positivos. Puerta de soporte: **`INSUFFICIENT_POSITIVITY`**.

La otra intervención, `D_adversarial`, se reporta por separado; no se mezcla con la etiqueta D_clean para aumentar artificialmente soporte. Los mundos DEV nunca se seleccionaron mediante D ni se usaron para reentrenar actores.

## Resultado 3 — Capacidad autónoma sin prefijo docente

Cada brazo tuvo **4 actores × 32 mundos × 2 cues = 256 episodios por calidad**; son episodios distintos emparejados por mundo, no 256 mundos independientes:

| Actor | Éxito autónomo F4 limpio /256 | Éxito autónomo F4 adversarial /256 |
|---|---:|---:|
| F9-O BC original | 244 | 166 |
| **Mixto** | **256** | **185** |
| Control limpio | **256** | **182** |

Los **dos nuevos actores** resuelven todos los episodios autónomos con F4 limpio en este DEV. Con F4 falsa, el mixto solo aventaja al control en **3/256** y empeora claramente en la réplica 0 (38/64 vs 52/64); en otras réplicas gana (1: 53 vs 51; 2: 49 vs 37; 3: 45 vs 42). Por tanto es injustificado afirmar una ventaja autónoma uniforme del entrenamiento robusto. El éxito con F4 limpio tampoco implica que la política use la memoria: el contrafactual demuestra que el mixto toma casi las mismas decisiones sin leer.

## Gates preregistrados

- **Contrato técnico y split:** PASS. Inicialización de pesos y acciones docentes iguales entre brazos; 4×2 entrenamientos; 10 épocas por entrenamiento; sin modificar actor histórico ni selector; TRAIN/DEV disjuntos; tests e informes correctos.
- **Robustez sin sacrificar READ limpio:** **FAIL, 0/4 réplicas**.
- **Positividad de D_clean en las 16 condiciones:** **FAIL, 0/4 réplicas con sus cuatro celdas soportadas**.
- Resultado científico exacto del agregador: **`EVIDENCE_ROBUSTNESS_NOT_ESTABLISHED_AND_INSUFFICIENT_POSITIVITY`**. Un `SUCCESS` de GitHub Actions solo certifica que el experimento se completó técnicamente.
- **STOP:** No entrenar cabezas READ, no ajustar retrospectivamente épocas, arquitectura, tipos de corrupción ni umbrales; no abrir `12300000+` confirmatorio.

## Interpretación y siguiente hipótesis separada

La dificultad no es simplemente entrenamiento insuficiente del actor o de la cabeza. Si todas las secuencias llevan el mismo objetivo incluso cuando la F4 es falsa, el actor puede minimizar errores evitando F4 y manteniendo la pista en la memoria recurrente. Eso da **robustez aparente ante evidencia adversarial** sin cultivar el uso selectivo de memoria externa. Si queremos que el agente *necesite* memoria externa y pueda **verificar su fiabilidad**, la siguiente tarea nueva deberá incluir **información no reconstruible solo a partir de H/C actual** y **procedencia contrastable desde observaciones públicas históricas**, o secuencias con cambio de objetivo documentado y evidencia temporal. Se debe demostrar experimentalmente que eliminar F4 empeora con evidencia legítima mientras permitir F4 falsa sin control perjudica. Se requieren nuevos checkpoints, un protocolo diferente y mundos nuevos. **Esa siguiente fase NO se ha ejecutado en F10-A.**

No se han reabierto confirmaciones históricas F9-V `8300000+`, F9-W `9300000+`, F9-X `9600000+`, F9-Y `10300000+`, F9-Z `11100000+`. Reserva F10-A `12300000+` sellada.
