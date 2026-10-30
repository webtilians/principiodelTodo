# INFINITO F9-J — Resultados de atribución: LSTM frente a memoria F4 externa

**Fecha:** 29-10-2026. **Status:** desarrollo exploratorio, cuatro políticas congeladas F9-I context8 y cuatro controles zero congelados; **cero entrenamiento nuevo**. La [ejecución F9-J #40942089024](https://github.com/webtilians/principiodelTodo/actions/runs/40942089024) terminó correctamente (tests, 4 réplicas y agregación). [PR #17](https://github.com/webtilians/principiodelTodo/pull/17). Holdout no abierto.

## Reproducción del control histórico

Antes de interpretar intervenciones, se verificó SHA-256 de los ocho checkpoints originales context8/zero y se exigió igualdad EXACTA en resultados de desarrollo normal S7 y S13. Los baselines context8 S7/S13 coinciden por semilla con el experimento F9-I.

| Agente autónomo F9-I congelado | S7 éxito (ambos correctos / 40 pares) | S13 éxito (ambos correctos / 40 pares) |
|---|---:|---:|
| Context8 sin cambios | **96,25 % (37/40)** | **72,5 % (21/40)** |
| Zero, misma arquitectura, readout siempre vacío | 50 % (0/40) | 35 % (0/40) |

Los resultados se basan en 80 episodios por tamaño/condición, repartidos en cuatro semillas de entrenamiento originales × diez pares original/invertido.

## Prueba 1: intervenciones en la navegación autónoma

| Intervención en el mismo checkpoint context8 | S7 éxito | Pares S7 acertados | S13 éxito | Pares S13 acertados |
|---|---:|---:|---:|---:|
| Ninguna | **96,25 %** | 37/40 | **72,5 %** | 21/40 |
| Reset únicamente del estado LSTM en la primera bifurcación detectada | 46,25 % | 0/40 | 46,25 % | 7/40 |
| Poner a cero únicamente el readout F4 en una observación del cruce | 92,5 % | 34/40 | 66,25 % | 17/40 |
| Reset LSTM + readout F4 cero en el cruce | 52,5 % | 2/40 | 46,25 % | 7/40 |
| Readout F4 cero siempre que la pista ya fue vista y no está visible | 51,25 % | 5/40 | **16,25 %** | **0/40** |

El reseteo LSTM se aplicó en 100 % de episodios S7 y 90 % S13. Borrar F4 una vez al cruzar se aplicó en 100 % S7 y 90 % S13; borrar F4 después de ver la pista se aplicó en 100 % S7 y 97,5 % S13.

Control de perturbación general: la LSTM zero (sin memoria externa informativa) mantuvo 50 % S7 y 35 % S13 con o sin reset al cruce, y 0 pares acertados en ambos. Esto reduce, sin eliminar, la explicación de fallo de navegación genérico ante reset.

**Interpretación:** en el agente context8 el estado recurrente tiene una influencia importante sobre decisiones al cruce; una perturbación puntual del readout externo produce una caída menor. PERO el readout F4 durante el camino sigue importando: eliminarlo tras observar la pista lleva la tasa S13 del 72,5 % al 16,25 % y elimina sus 21 pares acertados. Es posible que la LSTM actualice sus estados con información de los slots externos a lo largo del camino, incluso si el evento inicial fue expulsado. Las ablaciones de cero y el reset LSTM son fuera de distribución y pueden tener efectos generales: no demuestran por sí solos una codificación semántica.

## Prueba 2: misma bifurcación, intercambio factorial H/M

Dos historias (pista original e invertida) se recorrieron con **el mismo prefijo de acciones nativas del profesor**, hasta el primer instante en que la siguiente acción correcta difería. La política recibe solo observaciones públicas y actualiza el propio almacenamiento F4 y estado recurrente con cada imagen; la identidad de la acción correcta no entra en el actor.

En los 40 pares evaluados por tamaño se comprobaron:

| Medida antes de la siguiente acción en la bifurcación | S7 | S13 |
|---|---:|---:|
| Observación actual idéntica | 40/40 | 40/40 |
| Readout F4 de ocho eventos idéntico | **40/40** | **40/40** |
| Estados oculto/celda LSTM diferentes | 40/40 | 40/40 |
| Política elige ambas acciones correctas de profesor | 37/40 | 31/40 |
| Cambiar SOLO el estado recurrente H modifica la acción | **37/40** | **31/40** |
| Cambiar SOLO el readout F4 M modifica la acción | **0/40** | **0/40** |

La comparación H_original/M_original frente a H_invertida/M_original usa **exactamente el mismo vector de observación actual y el mismo vector F4**. En los pares positivos, solo el estado LSTM varía y la acción cambia en la dirección requerida.

Esto es evidencia **causal y local, bajo el recorrido guiado**, de que el estado recurrente afecta a la decisión condicionada por la pista y puede transportar la diferencia entre ambas historias más allá de la retención del evento original en la memoria externa.

No es un experimento confirmatorio ni evalúa autonomía del prefijo guiado. Una entrada oculta diferente no garantiza una representación general; el resultado específico es el cambio de acción con la intervención.

## Conclusión y siguiente paso

El resultado orienta hacia una arquitectura en la que la memoria externa **proporciona contexto durante la experiencia** y la LSTM **integra parte de ese contexto para las decisiones**. No basta con conservar el primer evento ni sería correcto concluir que F4 es innecesaria: sin F4 durante el recorrido la transferencia S13 colapsa.

Una prioridad para F9-K sería medir qué información sobre la pista se puede recuperar de H y de la memoria externa a varios instantes (tras verla, mitad del recorrido, antes del cruce), con un **probe** entrenado solo en escenarios de train y evaluado en escenarios development completamente separados. Repetirlo con capacidad y arquitectura igualadas y controles zero. No abrir heldout.

**Alcance y límites:** F4 es Event/Store/FIFO con selección contextual predefinida, no LearnedRelevance ni F5; cuatro semillas y parejas de desarrollo reutilizadas; resets y ceros son fuera de distribución; el teacher-forced probe tiene historia guiada y no sustituye éxito autónomo. Sin cambios PC/VPS y sin abrir test reservado.
