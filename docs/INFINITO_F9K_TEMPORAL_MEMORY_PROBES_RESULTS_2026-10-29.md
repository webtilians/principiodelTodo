# INFINITO F9-K — Resultados: decodificación temporal de la pista en LSTM y F4

**Fecha:** 29-10-2026. **Estado:** ejecución de desarrollo completa y aprobada.
**CI final:** [GitHub Actions #40946831488](https://github.com/webtilians/principiodelTodo/actions/runs/40946831488), **success** (integridad, cuatro semillas, agregado).
**JSON agregado:** [artifact #12928654315](https://github.com/webtilians/principiodelTodo/actions/runs/40946831488/artifacts/12928654315).
**PR:** [#18](https://github.com/webtilians/principiodelTodo/pull/18) (borrador sobre F9-J #17).

## Pregunta

¿Podemos reconstruir, mediante un decodificador lineal entrenado **solo en S7 train**, la identidad efectiva de una pista observada anteriormente, a partir del estado LSTM congelado, de F4 externa o de la observación actual? Esto complementa la atribución causal local F9-J pero no la sustituye.

## Integridad y diseño

- **Cuatro checkpoints originales context8 F9-I y sus cuatro zero emparejados**, cada uno verificado por SHA-256 y report.json desde [F9-I #40928547873](https://github.com/webtilians/principiodelTodo/actions/runs/40928547873); ningún peso de política fue entrenado ni actualizado.
- Único ajuste permitido: decodificador lineal ridge, alpha fijo = 10, con 80 **pares** de S7 train por semilla, incluyendo ambas identidades de pista. No se eligieron etapas, umbral o regularización con los resultados dev.
- Etiqueta correcta: objeto visto **key o ball**, obtenido solo por un observador separado. El objeto original varía entre mundos; jamás se etiquetó «original/invertido» como si fuese un contenido constante. La etiqueta nunca entra a la política ni al selector.
- Evaluación independiente del probe: 10 pares S7 development y 10 pares S13 development por semilla (40 pares y 80 casos por tarea). **Ningún ejemplo S13 entrena el probe**.
- Prefijos guiados idénticos hasta la primera acción diferente del profesor; tres instantes *post_cue*, *halfway* y *fork*. Observaciones actuales idénticas en **40/40 pares de cada etapa por tarea**. En el fork, F4 externa también idéntica en **40/40 pares S7 y 40/40 S13**.
- Se evaluaron todas las condiciones H (hidden+cell LSTM, 128), M (readout externo, 512), H+M (640), O (imagen pública actual, 148), y H/O en el agente zero sin F4. El conjunto heldout se mantuvo cerrado.
- El actor usa observaciones públicas y su propio estado, sin información privilegiada. El evaluador posee etiquetas solo para supervisar el probe y auditar la igualdad entre pares.

## S13: transferencia del decodificador entrenado solo en S7

Porcentaje correcto sobre 80 casos; cuatro semillas. Valor de azar para etiqueta equilibrada = 50 %.

| Fuente decodificadora | Post-cue (1 paso) | Halfway (6 pasos) | Fork (11 pasos) |
|---|---:|---:|---:|
| **H: estado LSTM, agente context8** | 78,75 % | 80 % | **96,25 %** |
| M: readout F4, agente context8 | **100 %** | **100 %** | **50 %** |
| H+M, agente context8 | 95 % | 91,25 % | 96,25 % |
| O: observación actual, context8 | 50 % | 50 % | 50 % |
| **H: estado LSTM del agente zero** | 50 % | 50 % | **50 %** |
| O: observación actual, zero | 50 % | 50 % | 50 % |

**Resultado crítico:** en S13/fork, H-context8 identificó correctamente el objeto pasado en **77/80 casos** y resolvió **ambas identidades en 37/40 pares** (predicción invertida correcta en 37/40). Por semilla: **100 %, 100 %, 95 %, 90 %**. En ese mismo instante, la observación pública O y la memoria F4 M eran idénticas entre ambas historias; O y M quedaron en 50 % y cero pares con ambas correctas. El control H-zero obtuvo 50 %.

En el **tramo anterior** al fork, M continuaba diferenciando ambas pistas en los 40/40 pares S13, y fue perfectamente decodificable. Por eso es incorrecto concluir que la LSTM retuvo la pista **durante 11 pasos sin recibir información F4**. El estado recurrente pudo seguir integrando readouts externos hasta muy cerca del fork.

## S7: tarea corta y comparación zero

| Fuente decodificadora | Post-cue (1 paso) | Halfway (3 pasos) | Fork (5 pasos) |
|---|---:|---:|---:|
| **H-context8** | **100 %** | **100 %** | **100 %** |
| M-context8 | 100 % | 100 % | 50 % |
| H+M-context8 | 100 % | 100 % | 100 % |
| O-context8 | 50 % | 50 % | 50 % |
| **H-zero** | 100 % | 91,25 % | **87,5 %** |
| O-zero | 50 % | 50 % | 50 % |

H-zero en S7/fork: semillas 100 %, 65 %, 85 %, 100 %; 30/40 pares con ambas identidades correctas. Por tanto, la memoria recurrente puede portar información del cue sin readout F4 en S7, pero **la decodificación lineal entrenada en S7 no transfirió a S13 para zero**. Diferentes checkpoints y diferencias de distribución impiden afirmar que F4 causa *por sí sola* toda la diferencia.

## Relación con F9-J

F9-J, con prefijos guiados, mostró cambio causal de acción en **31/40 pares S13** al intercambiar solo el estado recurrente con observación/F4 actuales idénticos. F9-K localiza una señal lineal del **contenido key/ball** en H-context8 del fork en **37/40 pares S13**; esa discrepancia no es contradicción: *decodificar información* y *ejecutar la acción correcta* no son la misma prueba. La navegación autónoma F9-J context8 S13 fue 72,5 %, y **no** debe sustituirse por el 96,25 % del probe guiado.

## Qué queda demostrado y qué no

**Sí, en esta distribución de desarrollo:** la identidad de una pista pasada se puede decodificar linealmente del estado recurrente H en el fork largo, con observación y readout F4 idénticos entre historias. La F4 era informativa inmediatamente después y a mitad de camino, lo que apunta a un papel dinámico externo→interno. La comparación H-context8/H-zero evidencia una fuerte diferencia de generalización del probe entre políticas.

**No:** que el estado H mantuviese el objeto sin apoyo de F4 durante once pasos; que esos vectores sean una memoria semántica reutilizable; que la política siempre utilice toda la información decodificable; que esta representación se generalice fuera de las tareas MiniGrid S7/S13; que el resultado sea confirmatorio o una medida de conciencia. Los recorridos son *teacher-forced* y pueden ser fuera de distribución respecto a la política autónoma.

**Limitaciones:** cuatro semillas de política, 40 pares por tarea, desarrollo reutilizado, tamaños temporales condicionados a estas tareas, probe lineal fijo, diferencia de dimensiones/modalidades, ajuste del probe en S7 train previamente visto por el actor. No se abrió heldout.

## Siguiente experimento recomendado: F9-L

Interrumpir experimentalmente **solo el canal F4** desde distintos puntos previos al fork (después de la pista, medio camino, un paso antes) usando checkpoints congelados y prefijos de acciones idénticos, y **medir simultáneamente** si la información sigue decodificable de H y si el cambio de acción depende de ella. Incluir controles zero y perturbaciones temporales neutras. El objetivo sería separar *almacenamiento recurrente persistente* de *reconstrucción reciente gracias a F4*. Fijar controles y ventanas antes de ejecutar y reconocer la posible perturbación fuera de distribución.

**Resultados F9-K obtenidos directamente del agregado de CI, sin seleccionar solo las semillas favorables.**
