# INFINITO F10-F — Auditoría mecanicista de selección, atención y salida (SIN entrenamiento)

**Prerregistro prospectivo 2026-11-02, antes de crear el generador de F10-F y antes de evaluar mundos `17_200_000+`.** F10-E mostró fallos severos `S_after` y `S_before`; la pregunta ahora NO es volver a medir una nota global ni mejorar hiperparámetros, sino determinar en qué etapa aparece el error de decisión de los modelos congelados.

## Pregunta y límites causales

Para cada política F10-C original `relational_read` (cuatro réplicas independientes del run `41447171697`, con checkpoint SHA verificado), evaluar:

1. **Selector (topk5)**: ¿incluye el registro firmado más reciente de la clave? ¿Qué otros cuatro registros selecciona?
2. **Puntuación y mezcla**: si incluye el ganador, ¿qué `own`, `pairwise_rank`, `logits` y atención softmax recibe frente a otros registros firmados/obsoletos?
3. **Cabeza de decisión**: si el ganador está presente pero la decisión original es incorrecta, ¿el mismo GRU/HEAD sin cambiar pesos responde correctamente si se sustituye **solo la lectura agregada** por la información de ese único registro firmado?

Los puntos 1 y 2 son **mediciones internas, no causalidad**. El punto 3 es una **intervención artificial y privilegiada para DIAGNÓSTICO**, pues el investigador selecciona un registro usando la regla pública «última firma». El agente real NO recibe esa selección, ni esas etiquetas ni esas salidas modificadas. No presentar accuracy del *patch* como rendimiento operativo o confirmación. Un parche exitoso no demuestra qué algoritmo podría aprender autónomamente a elegir ese registro.

## Semillas, mundos y fuentes congeladas

- Usar **nuevas** semillas `17_200_000 + 10_000*i+j`, índices `i=0..3`, `j=0..107`, 108 mundos base por réplica. Son **432 mundos base disjuntos** y cada uno contiene cuatro variantes contrafactuales `U_before`, `U_after`, `S_before`, `S_after`: **1728 estados de evaluación** dependientes (no 1728 mundos independientes).
- Para conservar un test controlado del mecanismo **reutilizar la especificación pública del factorial F10-E, NO sus semillas ni sus DEV**: 24 claves y 60 eventos por historial; cambios entre variantes solo credencial de X y posiciones relativas C/X; k∈{3,5,8} y tiempos `early,middle,late`, 12 mundos por combinación `k×timing` y réplica. Todos los 4 contra-historiales de un mundo comparten consulta final; información verdadera = último informe firmado público. Esto **NO es una nueva familia de tareas ni una confirmación independiente**; es una réplica mecanicista en semillas nuevas creada después de ver F10-E.
- Checkpoints F10-C **relacional** y **unario** originales por réplica del [run 41447171697](https://github.com/webtilians/principiodelTodo/actions/runs/41447171697), verificados por SHA contra sus manifiestos. Comparar salida binaria unaria READ como referencia, pero los diagnósticos de etapas son del relacional. No tocar ni reentrenar pesos; no cargar mundos F10-C/D/E antiguos. No abrir confirmaciones `17_300_000+` ni anteriores.

## Reproducción exacta del forward antes de interpretar

Instrumentar una copia analítica del forward original `RelationalActor.forward` sin alterar el actor. Reconstruir `GRU`, `torch.topk(match,5,dim=1)`, `unary`, tensor `pairwise` de siete features, `rank`, logits `(rank+own)*4`, softmax, `read_value`, `read_sign`, `available`, `head`. Exigir, para **cada caso**:
- Diferencia absoluta entre `actor.forward(history, log)` y la reconstrucción del logit <= **1e−5**, tanto para READ como para SKIP (con el mismo estado GRU). Error de paridad => FAIL **técnico**, no inferir mecanismos.
- Los cinco índices seleccionados corresponden a índices originales del FIFO, no al orden de timestamps del valor. Se conserva cada registro seleccionado: índice de slot, timestamp, valor, firma, fuente, `own`, `pairwise_rank`, `logit`, peso, y bandera de si coincide con el registro acreditado más reciente.
- En la consulta, determinar el **winner** únicamente de eventos públicos (`argmax timestamp` sobre registros de la clave con firma=1). Esto es un criterio de auditoría basado en la definición de tarea, no un input al modelo.

## Parches diagnósticos fijos (NO políticas operativas)

Conservar exactamente el historial público y el estado GRU del forward original, además del `head` congelado. Evaluar:
- `READ_real`: forward real sin parches.
- `READ_skip`: log externo cero, mismo GRU.
- `PATCH_SIGNED_ONLY`: manteniendo los cinco slots topk reales y los `rank` aprendidos, renormalizar softmax únicamente sobre registros firmados y válidos. No alterar el score relativo de registros firmados; cambia solo la contribución de no firmados. Puede quedarse sin ganador si `topk` lo excluyó.
- `PATCH_WINNER_ONEHOT` **solo si winner ya está entre topk5**: sustituir únicamente `read_value,read_sign,available` por el ganador (valor y firma de un informe público); alimentar el mismo `GRU` y `head` sin modificación.
- `PATCH_WINNER_INCLUDE` solo cuando winner estaba **fuera** de topk5: sustituir el quinto slot topk por el ganador, recomputar puntuaciones y atención originales con el mismo actor; no reparar la cabeza ni cambiar firma/valor/timestamp. Además registrar un `winner_onehot` tras inclusión como prueba de la cabeza. Si ya estaba presente, `include` debe coincidir exactamente con READ_real.
- La construcción de patches **utiliza un registro ganador seleccionado externamente por la regla de verdad** y es por ello una prueba causal asistida por observador. **La regla no llega al agente original**, y no puede contarse como generalización aprendida.

## Variables y clasificación prospectiva de errores

Para cada caso guardar la respuesta verdadera, probabilidad READ/ SKIP/ unaria, predicción, `winner_slot`, `selected_slots`, `winner_in_topk`, `winner_top_attention`, `winner_weight`, contribución relativa de otros firmados, `selected_signed_count`, scores de cada slot, resultados de todos los patches, y contraste vinculado al mundo base.

Definir **tres clases exclusivas de errores READ_real**:
- **A. MISSING_WINNER**: el último registro firmado **no figura en los cinco slots**. Esta es una insuficiencia de entrada del selector para razonar sobre ese registro directamente, aunque H recurrente siga disponible.
- **B. HEAD_OR_HIDDEN_LIMIT**: ganador seleccionado, pero `PATCH_WINNER_ONEHOT` también predice incorrectamente. Esto **no prueba** un fallo exclusivo del head: H, calibración de head y combinación entre ambos siguen confundidos.
- **C. MIXING_OR_WEIGHTING_SENSITIVE**: ganador seleccionado, patch winner_onehot corrige el error. La decisión es **sensible** a la mezcla de registros, sin demostrar por sí sola que el modelo sea capaz de aprender un selector correcto.

Estratificar las clases por réplica, `U_before`, `U_after`, `S_before`, `S_after`, k y timing. Publicar todas las clases incluidas las neutras/indecidibles. Registrar `PATCH_SIGNED_ONLY` y `PATCH_WINNER_INCLUDE` como diagnósticos adicionales por su población elegible. Guardar datos por mundo, sin filtrar casos por acierto.

### Hipótesis descriptivas y criterio de lectura (sin convertirlas en éxito del actor)

En cada réplica, sobre los **errores READ_real `S_after`** (denominador real de fallos, nunca todos los casos), observar si A, B o C supera el **50%** de esos errores. Solo etiquetar `DOMINANT_A`, `DOMINANT_B` o `DOMINANT_C` si hay >=10 errores y la categoría >50%; en otro caso `MIXED_OR_UNDERPOWERED`. La contribución mayoritaria de A no elimina el efecto de H; B no es evidencia de una arquitectura de reasoning necesariamente insuficiente; C no identifica si el problema es firma, timestamp o interacción de ambos. Anotar si >=3/4 réplicas identifican una misma clase dominante, **sin promover eso a confirmación de mecanismo ni a autorización para entrenar**.

Calcular además dos contrastes emparejados entre variantes **con historial público modificado**: cambios de `winner_slot`/índices `topk` y de peso del ganador entre `S_before`/ `S_after`, y entre `U_after`/`S_after`; por par reportar solo si había ganador en ambas selecciones. No inferir invariancia de H entre variantes, porque el propio orden/firma altera el GRU.

## Gate técnico y STOP

`F10F_MECHANISM_AUDIT_COMPLETE` solo si se verifica 100% paridad de logits READ/SKIP, fuentes congeladas verificadas, todas las 4 réplicas/108 mundos×4 variantes sin exclusiones, sin entrenamiento, sin nueva confirmación, y se publican tanto casos correctos como negativos. `F10F_TECHNICAL_AUDIT_FAIL` si falta lo anterior. **Este gate certifica la auditoría, NO que un modelo haya aprendido una regla cognitiva.** No reajustar el selector, cambiar número 5, entrenar heads, ni reutilizar mundos `16_200_000+` o abrir `17_300_000+` tras observar diagnósticos. Si la auditoría revela fallo o empate, siguiente hipótesis en experimento posterior y splits nuevos.
