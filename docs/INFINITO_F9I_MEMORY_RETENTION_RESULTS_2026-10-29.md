# INFINITO F9-I — Resultados: ¿olvido por FIFO o fallo de decisión a largo plazo?

Fecha: 29 de octubre de 2026. **Desarrollo, no confirmación**. Cuatro semillas congeladas F9-H. Ningún entrenamiento nuevo.

## Proveniencia y prueba de integridad

[GitHub Actions F9-I #40928315965](https://github.com/webtilians/principiodelTodo/actions/runs/40928315965) **Success**: tests de integridad, reconstrucción de las 160 trayectorias originales F9-H (80 S7, 80 S13), comparación del hash del readout de memoria de **cada paso**, acciones, poses, contadores de eventos y ocupación, y verificación SHA-256 de los ocho checkpoints originales F9-H. [Datos agregados y originales](https://github.com/webtilians/principiodelTodo/actions/runs/40928315965).

Fuente congelada: [F9-H #40925227783](https://github.com/webtilians/principiodelTodo/actions/runs/40925227783). La identidad del evento relacionado con la pista se determina SOLO por un observador de evaluación que intercambia la pista a igual pose y localiza los píxeles afectados. NO se presenta esa etiqueta al agente ni a los selectores FIFO/First16. El análisis no abre el holdout.

### Diagnóstico de retención con idéntica capacidad de 64 eventos

| Medida | S7 | S13 |
|---|---:|---:|
| Episodios | 80 | 80 |
| Pista vista por el actor | 80 | 78 |
| FIFO conserva el primer evento de pista hasta la última acción | 76 | 62 |
| First16 conserva el primer evento hasta la última acción | 80 | 78 |
| FIFO conserva AL MENOS UN evento de pista hasta la última acción | 80 | 75 |
| First16 conserva AL MENOS UN evento de pista | 80 | 78 |
| FIFO alcanzó los 64 slots | 4 | 18 |
| Éxitos F9-H original | 76 | 31 |
| Episodios terminados por timeout | 4 | 15 |

First16 mantiene fijados los primeros 16 eventos vistos, conservando el mismo Store, payload, readout y capacidad que FIFO. Se trata de una regla determinista sin etiquetas, NO LearnedRelevance. Conserva mejor el primer evento en recorridos que exceden el presupuesto, pero no es una política entrenada.

### Lo decisivo: S13 falla aun con la pista almacenada

La tabla cruzada, calculada sobre los episodios originales reproducidos (no inferida de la media):

| Estado S13 | Episodios | Al menos un evento de pista seguía en FIFO |
|---|---:|---:|
| Éxito | 31 | 30 |
| Fracaso | 49 | **45** |
| Timeout | 15 | 12 |
| Llegó a cualquiera de las dos salidas | 65 | 63 |
| Llegó a una salida incorrecta | 34 | **33** |

Así, en **33 de 34 elecciones de salida erróneas** había todavía por lo menos un evento de la pista disponible. Esto restringe drásticamente la hipótesis «la explicación principal es la expulsión por FIFO»: en el régimen observado los problemas de navegación/uso del historial pesan más.

La **última acción** no siempre corresponde a una bifurcación si hubo timeout; por ello se presenta la separación entre salida equivocada y timeout. Que un evento siga presente no garantiza que el agente sepa identificar cuál es relevante, ni que recuerde el orden correcto.

## Intervenciones sobre el modelo original CONGELADO

Mismas cuatro semillas; **cinco mundos emparejados por semilla** para un total de 20 pares/40 episodios por tamaño y condición. La manipulación cambia SOLO los dos canales [key,ball] del readout de UN evento con correspondencia a la pista inicial, sin tocar la imagen actual, otros recuerdos, pesos ni posiciones. El comparador «no pista» modifica un único evento no asociado a ese píxel.

| S7 / política congelada | Éxito | Pares ambos correctos | Cambios de salida |
|---|---:|---:|---:|
| FIFO sin intervención | **95 %** | 18/20 | 18/20 |
| FIFO + primer evento de pista alterado | **22,5 %** | 1/20 | 13/20 |
| FIFO + primer evento no-pista alterado | **92,5 %** | 17/20 | 17/20 |
| First16 sin intervención | 95 % | 18/20 | 18/20 |

Esto constituye una evidencia mucho más específica de que la política S7 responde al contenido de un evento histórico relevante. Hay que distinguir entre **sensibilidad causal de la política a un evento** y prueba de un mecanismo semántico general. La identidad del evento se identifica con un observador privilegiado de la *ablación*, no con un selector que conozca el futuro. El control no-pista solo cambia un evento diferente; el efecto exacto de una inversión fuera de distribución no constituye una comparación semántica perfectamente emparejada.

| S13 / política congelada | Éxito | Pares ambos correctos |
|---|---:|---:|
| FIFO sin intervención | **42,5 %** | 0/20 |
| FIFO + primer evento de pista alterado | 42,5 % | 3/20 |
| FIFO + primer evento no-pista alterado | 37,5 % | 2/20 |
| First16 sin intervención | 42,5 % | 0/20 |

En S13, alterar recuerdos provoca respuestas inconsistentes pero **no muestra utilización competente de la pista**. Incluso la conservación mejorada First16 no cambia la puntuación del modelo previamente entrenado con FIFO. Esta comparación First16 es una intervención *fuera de la distribución de entrenamiento*, **NO** una evaluación justa de un selector First16 entrenado independientemente.

## Interpretación y siguiente decisión de investigación

El fracaso S13 no se explica principalmente por una simple falta de slots: 33/34 salidas erróneas ocurren con algún evento de la pista retenido. La prioridad experimental debería ser diferenciar:
- El agente recuerda eventos pero no distingue **qué evento** corresponde al indicio inicial tras muchos pasos/objetos.
- El agente no domina una política de navegación S13 porque solo ha sido entrenado en S7; esto puede generar fallo aun con memoria intacta.
- La saturación FIFO importa especialmente en loops y timeouts; First16 sirve de control para estos casos, no prueba por sí solo una solución general.

**Propuesta F9-J:** currículo multi-longitud con **presupuestos y dimensiones idénticos**, S7 y longitud variable en entrenamiento, evaluaciones S13 y longitudes mayores de desarrollo no vistas; comparar FIFO, First16 y una selección contextual aprendida con recompensa ex post válida (sin truth de la pista). Para llamar LearnedRelevance al mecanismo hay que especificar una correspondencia de *consulta retrospectiva observable* y validar que su feedback no filtra el objetivo del simulador: la implementación actual de LearnedRelevance necesita el identificador de una query y **no hay tal etiqueta pública en MiniGrid**. Hasta resolverlo, sería engañoso presentarla como una condición ya probada.

**Límites:** tarea concreta MiniGrid, semillas/escenarios de desarrollo reutilizados, 4 entrenamientos previos, solo 5 pares por semilla en las intervenciones congeladas. No resultado confirmatorio ni superioridad general de INFINITO. Nada ejecutado en PC o VPS; sin heldout.
