# INFINITO F9-Y — Valor causal D+ condicionado a pista/historia y transferencia entre combinaciones

**2026-11-02. Resultado del desarrollo, NO confirmación.** Protocolo científico congelado **antes de código, TRAIN/DEV y resultados**: commit `8555f8e720c31d25731d9ae1848bc2bbd393b9b3`, [protocolo](./INFINITO_F9Y_CONDITIONAL_VALUE_PREREG.md). Ejecución [GitHub Actions #41368960928](https://github.com/webtilians/principiodelTodo/actions/runs/41368960928) finalizada **SUCCESS técnico**: 4 réplicas, 12 cabezas/replica, tests, TRAIN y DEV separados, agregado y gate. **Gate científico: `INCONCLUSIVE_SUPPORT`.**

## Datos y congelación

Los cuatro actores F9-P originales y F4 se mantuvieron congelados con procedencia y código original validados. Se abrieron solamente TRAIN `10_100_000+10_000*index+j, j=0..127` y DEV independiente `10_200_000+10_000*index+j, j=0..31`: **2048 estados TRAIN y 512 estados DEV**, cada mundo con cue key/ball × historia F4 normal/apagón. Cada `D` procede de intervenciones READ/SKIP idénticas salvo una lectura F4 en el primer cruce. El prefijo sigue siendo docente; la continuación es autónoma solo tras el cruce.

**No se ha abierto** CONFIRM F9-Y `10_300_000+`; tampoco F9-V `8300000+`, F9-W `9300000+`, F9-X `9600000+`, F9-T `6300000+`, F9-U `7300000+` ni histórico `900000+`. Ninguna etiqueta de `D` de DEV se usó en aprendizaje o en selección de umbral/épocas.

## Resultado 1 — utilidad predictiva con las cuatro combinaciones disponibles en TRAIN

Los predictores MLP entrenados sobre las **cuatro combinaciones** originales, siempre evaluados sobre nuevos mundos DEV, obtuvieron:

| Predictor | Log-loss natural DEV (menor es mejor) | READ/512 |
|---|---:|---:|
| `deploy_hc` (148 público + 128 H/C, sin etiquetas cue/historia) | **0,1071** | **48** |
| `deploy_public` (148 público, H/C cero) | 0,3102 | 0 |
| `controlled_hc` (+ cue/historia privilegiados) | **0,0929** | 48 |
| `controlled_nuisance` (cue/historia privilegiados, sin H/C) | 0,2202 | 0 |

En `deploy_hc`, las 48 lecturas fueron **44 útiles y 4 neutras**, sin perjudiciales, concentradas en índices 0 y 3. Los modelos `controlled_*` son solamente **comparadores diagnósticos** porque cue/historia proceden de etiquetas del observador y no son inputs desplegables. Estos números indican que **H/C aporta información predictiva adicional en mundos nuevos con combinaciones ya vistas en TRAIN**. No implican extrapolación a combinaciones no vistas ni robustez en los cuatro actores.

## Resultado 2 — generalización LOCO (objetivo científico central)

Se entrenó, para cada actor y cada celda, un par nuevo de modelos `loco_hc` / `loco_nuisance` con **tres** de las cuatro combinaciones cue×historia, excluyendo al 100% la cuarta de TRAIN. El par usó arquitectura, etiquetas privilegiadas de cue/historia, inicialización y presupuesto idénticos, diferenciándose **solo en recibir H/C**. Luego se evalúa exactamente la celda inédita sobre mundos DEV.

**Soporte prospectivo obligatorio por celda:** >=4 positivos D+ y >=4 no positivos en DEV agregado, más >=2 actores con >=2 positivos en esa celda.

| Combinación omitida de TRAIN | D+ /128 en DEV | D+ por actor 0/1/2/3 | ¿Soporte? | Δ log-loss (nuisance − H/C) |
|---|---:|---|---|---:|
| key + normal | 38 | 19 / 0 / 0 / 19 | **SÍ** | **−0,1592** |
| ball + normal | 9 | 8 / 0 / 1 / 0 | NO (solo un actor >=2) | −0,4614 |
| key + apagón | 1 | 0 / 0 / 1 / 0 | NO | +0,0695 |
| ball + apagón | 7 | 0 / 7 / 0 / 0 | NO (solo un actor >=2) | −0,0020 |

El diseño exigía **al menos dos celdas con soporte**; solo **una** lo tiene. Resultado científico preregistrado `INCONCLUSIVE_SUPPORT`, **no PASS** ni confirmación de la hipótesis. En la única celda evaluable, H/C generaliza **peor que el control privilegiado**: pérdidas `1,2388` vs `1,0796`; Δ = −0,1592 nats/estado, IC95% bootstrap por mundos estratificado por actor **[−0,1794, −0,1395]**. Esto es evidencia descriptiva contra transferencia en ese estrato, sin rescatar el gate general.

En todos los 512 estados LOCO (incluyendo estratos sin suficiente soporte), `loco_hc` pierde frente a `loco_nuisance`: log-loss **0,5814 vs 0,4431**. Realiza **49 READ: 2 útiles, 6 perjudiciales y 41 neutras**; el control nuisance no realiza READ. El `loco_hc` falla así en transferencia de selección bajo combinación omitida. Estos datos no deben presentarse como un test general bien identificado de 4 estratos, ya que faltan positivos en tres de ellos.

## Resultado 3 — por qué falla la interpretación causal amplia

Los TRAIN positivos D+ están distribuidos de forma marcadamente específica por actor y por estrato:

| Réplica | key-normal | ball-normal | key-apagón | ball-apagón |
|---|---:|---:|---:|---:|
| 0 | 60 | 40 | 0 | 0 |
| 1 | 0 | 0 | 0 | 33 |
| 2 | 0 | 6 | 12 | 0 |
| 3 | 60 | 0 | 0 | 0 |

En la réplica 3, al dejar fuera key-normal no queda **ningún positivo** en TRAIN. En la 1 ocurre lo mismo al dejar fuera ball-apagón. La distribución de utilidad de F4 es extremadamente dependiente del actor y de las condiciones: una clasificación sin evidencia positiva extrapolable puede ser imposible para este protocolo, independientemente de la arquitectura del predictor. No se debe resolver seleccionando retrospectivamente las dos réplicas favorables ni subiendo la capacidad del modelo tras ver los resultados.

La permutación H/C dentro del mismo actor/estrato para el predictor LOCO **no aporta el criterio causal exigido**; en key-normal la log-loss **baja** en lugar de subir al permutar: `shuffle_loss - original_loss = −0,16065`, IC95% **[−0,24597, −0,08371]**. Esto es compatible con una asociación H/C específica de estratos conocidos que resulta engañosa al transferir; el intercambio entre mundos puede generar muestras fuera de distribución y no es por sí solo prueba de qué semántica existe.

## Juicio científico y próxima decisión

1. **Hipótesis más acotada, apoyada en DEV:** existe señal predictiva en H/C cuando todas las combinaciones están representadas en TRAIN y los mundos DEV son nuevos. La mejora es descriptiva, no tiene confirmación propia independiente.
2. **Hipótesis nueva y fuerte, NO demostrada:** la información predice el valor causal de READ bajo combinaciones no entrenadas. El gate fue `INCONCLUSIVE_SUPPORT`; la única celda evaluable arroja desempeño adverso. Nunca abrir `10300000+` para retocar/revalidar esta variante.
3. **Siguiente cuello de botella es la POSITIVIDAD experimental, no el tamaño de la MLP**: crear un entorno causal contrabalanceado donde cada actor disponga de casos D+ y D<=0 para cada combinación, manteniendo comparadores matched y documentando las modificaciones, o cambiar tarea/actor de modo preregistrado. Seleccionar casos de acuerdo a D de DEV no es aceptable.
4. El prefijo docente y una sola READ impiden inferir autonomía de navegación o una valoración longitudinal de memoria.

**No fusionar esta rama a producción**; archivar resultado y diseñar la futura manipulación de soporte en experimento independiente, sin reusar el holdout cerrado F9-Y.
