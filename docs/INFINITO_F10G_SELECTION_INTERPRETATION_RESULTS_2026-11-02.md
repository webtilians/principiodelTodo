# INFINITO F10-G: selección e interpretación, resultado prospectivo

**2026-11-02 — `F10G_TECHNICAL_COMPLETE`, `F10G_DECOUPLING_FAIL`, `F10G_ARCHITECTURE_FAIL`.**

Los dos selectores con supervisión desacoplada consiguen selección y decisión perfectas en los nuevos mundos evaluados. No se demuestra superioridad relacional: el unario empata. Tampoco se cumple la mejora preregistrada en las cuatro réplicas: el relacional conjunto de la réplica 0 ya resuelve todos los casos. El fallo científico se conserva aunque la ejecución técnica sea correcta.

## Registro y trazabilidad

- [Prerregistro](INFINITO_F10G_SELECTION_INTERPRETATION_PREREG.md), commit **eedc1abf639f1f8e6763294bdfcf3010b0e0b30e**, publicado antes de implementar F10-G.
- Código ejecutado: **a11f525af2acf42769962d4ec6b95468526a3ece**. Cuatro procesos CPU locales, un hilo por proceso, sin reintentos, búsqueda ni selección de checkpoints. Cinco modelos finales por réplica; cada conjunto de cinco se guardó antes de evaluar esa réplica.
- [Datos, 20 checkpoints y logs completos](artifacts/f10g-frozen-run.zip), [manifiesto SHA256 y presupuestos](artifacts/f10g-manifest.json), [auditoría detallada](artifacts/f10g-audit.json).
- Después de evaluar sólo se reforzó el auditor para exigir igualdad de hashes, etiquetas y bases entre brazos. No cambió el generador, actor, entrenamiento, evaluación ni ningún gate. No se volvió a entrenar.
- No se importaron generadores anteriores ni se cargaron pesos o filas DEV F10-F. No se abrieron confirmaciones, incluidas las reservadas `18300000+`. El estudio está motivado por F10-F: independencia de datos no significa confirmación independiente de la hipótesis.

## Qué se comparó

Factorial selector **unario/relacional** × entrenamiento **sólo por decisión/auxiliar desacoplado**. El selector aprende con firma, tiempo y procedencia pública; nunca recibe valores ni la respuesta del evaluador como entrada. Examina todos los candidatos de la clave. La cabeza recibe la lectura, el estado recurrente y una instrucción binaria pública: devolver el valor vigente o invertirlo (XOR).

Los auxiliares añaden, sólo en TRAIN, supervisión del índice vigente y una pérdida de interpretación con el valor vigente aislado. Esta información deriva de la regla pública, pero es **supervisión adicional**. En evaluación READ la selección y decisión son autónomas, sin introducir el ganador conocido. El parche `oracle_patch` es un diagnóstico separado, no un resultado operacional.

Por réplica: 512 mundos TRAIN, 128 DEV y 128 OOD; cada mundo tiene cuatro variantes emparejadas. Total: **2.048 mundos TRAIN, 512 DEV, 512 OOD**, equivalentes a 8.192/2.048/2.048 estados. Las cuatro variantes no son observaciones independientes. TRAIN/DEV tienen 3 o 5 informes de la clave; OOD tiene 8 o 12. Cada historial contiene 32 registros, 24 claves posibles, posiciones de almacenamiento aleatorias y timestamps independientes del orden de almacenamiento. Hay uno o dos informes acreditados; las otras versiones y claves actúan como distractores.

Los cuatro brazos factoriales tienen **7.458 parámetros**; GRU-only, **7.588 (+1,74%)**. Todos reciben 640 pasos y 40.960 presentaciones de ejemplos por réplica, sin early stopping. Misma inicialización y orden de minibatches dentro del factorial. No hay gate READ/SKIP entrenado.

## Resultado operacional

Cada celda DEV/OOD agrega 2.048 decisiones, descriptivamente, conservando las réplicas abajo.

| Modelo | DEV READ | OOD READ | Selección OOD | Interpretación con ganador OOD |
|---|---:|---:|---:|---:|
| Relacional conjunto | 62,70% | 62,74% | 40,97% | 62,11% |
| Relacional auxiliar | **100%** | **100%** | **100%** | **100%** |
| Unario conjunto | 62,94% | 64,21% | 46,97% | 64,40% |
| Unario auxiliar | **100%** | **100%** | **100%** | **100%** |
| GRU-only | 51,22% | 51,56% | No aplica | No aplica |

La selección y el parche del control GRU no son capacidades medibles de ese modelo: no posee selector ni canal de lectura. Los campos internos comunes de sus filas son marcadores auxiliares y no deben interpretarse como rendimiento de selección.

Ambos modelos auxiliares aciertan las cuatro variantes de cada uno de los 512 mundos OOD, en ambos valores de k. No implica una tasa poblacional exactamente igual a 100% ni una regla universal aprendida.

| Réplica | Relacional conjunto OOD | Relacional auxiliar OOD | Unario conjunto OOD | Unario auxiliar OOD | GRU OOD |
|---|---:|---:|---:|---:|---:|
| 0 | 100% | 100% | 54,69% | 100% | 54,69% |
| 1 | 49,61% | 100% | 49,41% | 100% | 50,59% |
| 2 | 52,73% | 100% | 52,73% | 100% | 50,20% |
| 3 | 48,63% | 100% | 100% | 100% | 50,78% |

La variabilidad entre inicializaciones es sustancial. Por ejemplo, el unario conjunto de la réplica 0 selecciona correctamente en 83,98% de OOD pero decide bien sólo en 54,69%; el parche con ganador alcanza 56,05%. Esto separa mediciones de selección e interpretación, sin atribuir causalmente todo el fallo a una única capa.

## Ablations y bases

OOD del relacional auxiliar: READ **100%**, SKIP **48,93%**, valores permutados **43,65%**, firmas borradas **56,93%**, tiempos borrados **75%**. SKIP y la permutación mantienen exactamente el historial/estado recurrente. La permutación es un desplazamiento cíclico fijo entre los candidatos; puede conservar valores repetidos. Borrar metadatos sólo interviene el canal externo, no el historial: es estrés del canal, no eliminación global de la información.

IC95 emparejados por mundo, 2.000 remuestreos, diferencias en puntos porcentuales:

| Réplica | READ−SKIP | READ−permutación | Auxiliar−conjunto relacional |
|---|---:|---:|---:|
| 0 | +52,54 [48,44; 56,84] | +58,98 [52,34; 66,21] | **0 [0; 0]** |
| 1 | +51,56 [47,27; 55,86] | +58,79 [52,15; 65,82] | +50,39 [46,09; 54,69] |
| 2 | +50,00 [45,70; 54,30] | +55,27 [48,44; 62,11] | +47,27 [42,77; 51,56] |
| 3 | +50,20 [45,70; 54,49] | +52,34 [45,31; 58,99] | +51,37 [47,07; 55,47] |

Las bases OOD alcanzan: moneda reproducible **49,80%**, último registro por posición **51,37%**, máximo timestamp ignorando firma **57,03%**, regla simbólica acreditada **100%**. La última es un techo programado, no una capacidad aprendida. Las bases no tienen presupuesto de entrenamiento comparable.

## Gates, costes y límites

Las cuatro réplicas del relacional auxiliar superan los umbrales de selección, decisión por variante, interpretación, cuatro variantes correctas y dependencia causal de READ. **La réplica 0 incumple el umbral de mejora auxiliar−conjunto ≥5 puntos con IC inferior positivo**. Por eso `F10G_DECOUPLING_FAIL`. El contraste relacional auxiliar−unario auxiliar es **0 [0; 0] en las cuatro réplicas**: `F10G_ARCHITECTURE_FAIL`.

Igualar parámetros y pasos no iguala cómputo. La estimación de operaciones densas por ejemplo/forward es **624.512** para el relacional, **402.304** para el unario y **413.882** para GRU. Cuenta dos operaciones por multiplicación-acumulación; excluye activaciones, pooling, pérdidas, backward y optimizador. El auxiliar añade una pasada de cabeza (~2.304 operaciones densas), CE y sus gradientes. No es una medición completa de FLOPs. El relacional cuesta aproximadamente 1,55× en esta estimación.

Tiempo de entrenamiento acumulado sobre cuatro réplicas: relacional conjunto **36,29 s**, relacional auxiliar **37,95 s**, unario conjunto **17,72 s**, unario auxiliar **18,57 s**, GRU **17,45 s**. Son tiempos locales con procesos concurrentes, no un benchmark de velocidad normalizado. El manifiesto contiene tiempos individuales y versiones del entorno.

El efecto observado corresponde al **paquete de dos pérdidas auxiliares**: no distingue por sí solo el beneficio de supervisar selección del de entrenar interpretación aislada. Tampoco prueba que ampliar arquitectura sea necesario. El generador tiene a lo sumo dos firmas, una interpretación binaria XOR, 32 eventos y una sola consulta; no demuestra transferencia a cadenas arbitrarias de revisiones, lenguaje natural o tareas abiertas. La supervisión adicional, el coste cuadrático relacional y el efecto de inicialización siguen siendo factores explícitos.

## Reproducción y comprobaciones

Cuatro pruebas cubren verdad pública, pares, dominios de semillas, capacidad, invariancia a valores del selector, intervenciones y permutación de almacenamiento. Todas pasan. El auditor independiente del forward verifica los 20 SHA de checkpoints, 20.480 filas de evaluación, 12.288 hashes únicos de estados TRAIN/DEV/OOD, igualdad de mundos entre brazos y los gates fijados.

Para auditar sin entrenar: extraer el ZIP en una carpeta y ejecutar `python -m observatory.f10g_audit --root CARPETA --output audit.json` (NumPy 2.4.6). El workflow F10-G repite esta auditoría, verifica el SHA del archivo y exige identidad con el resultado publicado; un FAIL científico no hace fallar una auditoría técnicamente correcta.

Para reproducir desde cero en carpetas nuevas: `python -m observatory.f10g_experiment --index R --output CARPETA/R`, R=0..3; después ejecutar el auditor. Usar las versiones registradas en los manifiestos de entrenamiento. Es reproducción de los mismos datos abiertos, no otro experimento independiente.

**Decisión:** mantener el PR en borrador. No reajustar este DEV/OOD ni abrir confirmaciones. El resultado respalda utilidad de la supervisión explícita en esta tarea acotada, pero conserva los dos fallos científicos preregistrados.
