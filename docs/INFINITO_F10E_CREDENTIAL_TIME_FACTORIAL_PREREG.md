# INFINITO F10-E — Factorial causal de credencial, precedencia y contradicción

**Prerregistro prospectivo · 2026-11-02 · ANTES de implementar generador, evaluador o abrir mundos nuevos.** Nueva pregunta motivada por F10-D `signed|revised`, sin tratar su DEV como confirmación. **No se entrenan nuevos modelos ni una cabeza READ.**

## Pregunta falsable
¿La dificultad del actor relacional congelado para actualizar información acreditada se debe a reconocer el token de procedencia, a establecer qué informe es más reciente, o a **combinar ambos**? El experimento identifica **errores conductuales** por intervenciones públicas controladas; no identifica por sí solo circuitos internos de red. Todos los datos temporales y credenciales son accesibles al actor. Un «firmado» es un indicador *sintético observable*, no verificación criptográfica.

## Factorial completo y respuesta correcta
En cada mundo base, con valor binario inicial `b` aleatorio para una clave pública `q`, hay una evidencia inicial `S(b)` y dos registros posteriores **C=S(b)** (confirmación auténtica) y **X=(1-b)`** (revisión contradictoria). Se realizan cuatro variantes emparejadas **sin modificar pesos, clave solicitada, contenido de eventos, fuentes, relleno ni sus posiciones fijas de calendario**:

| Variante | Firma de X | Orden entre X y C | Última evidencia acreditada | Verdad |
|---|---|---|---|---|
| `U_before` | X sin firma | X antes de C | C=S(b) | b |
| `U_after` | X sin firma | C antes de X | C=S(b) | b |
| `S_before` | X firmada | X antes de C | C=S(b) | b |
| `S_after` | X firmada | C antes de X | X=S(1-b) | 1-b |

`S_after` es la única condición donde debe cambiar la respuesta. Reordenar implica recolocar los informes observados y actualizar su timestamp público al nuevo lugar, **sin cambiar el identificador de fuente asociado a cada informe**. Cambiar firma modifica **solo el bit público de credencial** de X. Las cuatro variantes de un mundo tienen la misma pregunta final; cada una contiene su propio historial completo y la F4 escrita desde ese historial. **Para comparar READ con SKIP dentro de una variante**, ambos comparten *exactamente* ese mismo historial/estado recurrente y solo se anula el log externo F4.

Este 2×2 NO es una intervención que mantenga fija la historia GRU entre variantes: credencial y orden forman parte de la experiencia pública; no puede interpretarse como manipulación aislada de un subespacio interno. Sí permite discriminar cambios conductuales sensibles a una sola variable externa frente a una conjunción.

## Nuevos mundos, cruces y distractores
- **Cuatro réplicas con modelos congelados**, una réplica por conjunto de checkpoints históricos F10-C/F10-B. Mundo base `16_200_000+10_000*i+j`, `i=0..3`, `j=0..107`; en total **108 mundos nuevos por réplica**, 432 por cuatro réplicas y **4 variantes factoriales cada uno = 1728 evaluaciones de inferencia por modelo**, emparejadas por mundo. `16_300_000+` reservado **sin ejecutor ni uso**. No reutilizar 13/14/15 millones ni abrir antiguas confirmaciones.
- Entrada **idéntica dimensionalmente a F10-C**: 24 claves, 60 informes públicos de 29 características + consulta final, log F4 observacional FIFO64 de 28 características y respuesta binaria. Query final nunca revelada en el prefijo de 60. **La clave se sortea una vez por mundo**; cada variante conserva idéntica clave.
- Cantidad de informes de la clave consultada **k∈{3,5,8}** y ubicación del par de informes contendientes **`early`, `middle`, `late`**; cruzar los 9 tratamientos por **12 mundos por réplica y celda**, con `k=3,5,8` y posiciones del par en índices 0-based globales `(34,35), (39,49), (48,59)`. Hay informes iniciales firmados de 24 claves en índices 0..23; hasta cinco informes extras de la clave consultada se sitúan **antes del par**, con credencial/valor variables pero sin firmar nunca el valor contrario a b. Los restantes hasta 60 se rellenan con claves distractoras ajenas a la consulta, valores y firmas públicas independientes. No seleccionar mundos por desempeño.
- El mundo base y las cuatro variantes tienen la **misma cantidad total de informes (60)** y el mismo número de registros para la consulta. Los cambios en la historia pública están limitados al bit de credencial de X o el intercambio X↔C entre los dos índices fijados. Se balancea el valor inicial b∈{0,1} por sorteo aleatorio y se reporta por b; la identidad A/B de fuentes de X y C se sortea independientemente, **permanece vinculada al informe durante el intercambio** y se reporta. No existe fuente fija que garantice verdad.
- Las cuatro variantes se evalúan como contrafactuales **del mismo mundo base**; agrupación bootstrap por mundo base, no por 4 variantes independientes. Cada variante tiene única respuesta derivable del último informe firmado público y se verifica con una función independiente.

## Modelos y comparadores congelados
Cargar originales **F10-C** run `41447171697`: `relational_read`, `unary_read_control`, `recurrent_only_control`; y **F10-B** run `41443064754`: `memory_available`, `recurrent_only`. Todos los checkpoints con SHA respecto a sus respectivos manifests; cero gradientes/optimizador/updates. La comparación primaria de políticas es **relacional READ vs F10-C unario READ** y vs GRU-only. El actor relacional tiene matching por clave y **topk5 fijo**, algo que podría limitar k8. Ningún modelo conoce etiquetas de credencial correctas aparte del token que ve, ni el resultado del operador READ/SKIP.

Por variante medir: READ auténtico, SKIP con F4 a cero, control unario con F4, recurrente sin F4, F10-B histórico con F4. Además controles deterministas basados EXCLUSIVAMENTE en informes públicos: «último informe cualquiera» y **«último firmado»** que es la regla exacta/techo, no una inteligencia aprendida. Reportar todas las probabilidades/decisiones para auditoría. No es preciso volver a correr ataques `stale`/`forged`: las intervenciones factoriales ya varían explícitamente experiencia pública y memoria escrita a partir de ella.

## Métricas fijadas antes de observar resultados
Por actor y mundo, porcentaje de mundos con **las cuatro respuestas correctas**; acierto en cada una de `U_before,U_after,S_before,S_after`; acierto por `k∈{3,5,8}` y por `posición∈{early,middle,late}` (cada bin 36 mundos ×4 variantes por actor); exactitud comparadores y baselines; D_clean=correct(READ)−correct(SKIP) para cada variante y global; bootstrap IC95% agrupado por los **108 mundos** con 2000 resamples.

Contrastes de **cambio de clasificación** por mundo:
- **Credencial útil**: predicción `U_after` = b y predicción `S_after` = 1−b (**debe cambiar**).
- **Precedencia útil**: predicción `S_before` = b y `S_after` = 1−b (**debe cambiar**).
- **Control estable de credencial**: `U_before` = `S_before` = b (**no debe cambiar**).
- **Control estable de precedencia**: `U_before` = `U_after` = b (**no debe cambiar**).

Estas pruebas separan posibles déficits de *sensibilidad* y *especificidad* de credencial/tiempo. Fallar ambas pruebas útiles cuando los controles neutros pasan es **compatible con** dificultad de conjunción; **no demuestra** qué unidades internas representan cada operación. Se reportan además los casos de modelos que simplemente predicen b siempre (techo 75% en el factorial) y cambios de probabilidad aunque no cambie la clase.

## Gates prospectivos de DEV factorial (no confirmación)
- **G1 exactitud robusta de cuatro estados:** en >=3/4 réplicas el relacional debe acertar >=80% de cada variante por separado, >=70% de los 108 mundos con **4/4 correctos**, y >=75% dentro de cada bin de k y posición (promediando las cuatro variantes). El criterio de 4/4 evita que 75% global por «contestar b siempre» cuente como aprendizaje.
- **G2 contrastes causales de conducta:** en >=3/4 réplicas, éxito en cada uno de los **cuatro contrastes** credencial útil, precedencia útil, control neutro de credencial y control neutro de orden de al menos **80% de mundos**. Reportar por separado; no elevar a éxito global si falla una sola.
- **G3 utilidad causal de memoria:** en >=3/4 réplicas, diferencia `READ−SKIP >=+0.10` sobre los 432 casos/actor, límite inferior IC95% bootstrap por mundo >0, y READ frente a F10-C recurrent-only >=+0.05. **No se exige superioridad al unario** para el objetivo diagnóstico; se reporta y no se infiere mecanismo específico por ventaja general.
- Estado `F10E_FACTORIAL_DIAGNOSTIC_PASS` únicamente si G1&G2&G3; en otro caso `F10E_FACTORIAL_DIAGNOSTIC_FAIL`, con celdas/contrastes que fallaron sin ocultarlos.

## STOP
Solo evaluación congelada; no ajustar F10-C, no entrenar cabezas READ, no cambiar el generador o umbrales después de ver DEV, no abrir `16_300_000+` ni holdouts F9/F10 anteriores. Si aparece un fallo **técnico antes de ejecutar ningún mundo**, documentar corrección mínima conservando la genealogía del protocolo. Si aparece un fallo causal, registrar como resultado negativo. El test demuestra respuestas bajo cambios públicamente observables de tiempo y firma, no validez criptográfica ni transferencia a LLM, RAG o F4 MiniGrid.
