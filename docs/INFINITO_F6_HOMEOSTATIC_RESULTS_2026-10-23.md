# INFINITO F6 — Resultados de aprendizaje de consecuencias futuras

**Ejecución confirmada:** https://github.com/webtilians/principiodelTodo/actions/runs/39867529437  
**Rama:** experiment/f6-homeostatic-world-model  
**Código:** scripts/f6_homeostatic_rl.py  
**Protocolo previo:** docs/INFINITO_F6_HOMEOSTATIC_MODEL_BASED_RL_PROTOCOL.md  
**Pruebas:** tests/test_f6_homeostatic_rl.py (7 superadas)  
**Datos:** artefacto infinito-f6-homeostasis-rl-heldout (report.json completo).

## Qué ejecutamos realmente

Un mundo *sintético*, observable, con variables de sed (déficit), fatiga,
inventario de agua y calor. Las cuatro acciones posibles son esperar,
beber, descansar y buscar agua. El coste homeostático está explícitamente
programado; el agente **no siente sed**.

El modelo se entrenó para predecir el estado siguiente a partir de
transiciones generadas mediante exploración aleatoria. Desconoce las
ecuaciones de transición reales y calcula consecuencias aproximadas
para escoger acciones con planificación recedente a tres pasos.

Configuración: 8 semillas independientes; 110 episodios exploratorios
de 48 pasos por semilla = 5.280 transiciones por semilla; 12
episodios futuros de 48 pasos por semilla y régimen.
Modelos congelados durante el test. Se usaron las mismas
semillas de evaluación en todas las políticas de un régimen.

## Resultados: test sin cambio de mecanismos

**Coste medio por paso, menor es mejor.**

| Política | Coste medio | Fracción pasos con sed elevada |
|---|---:|---:|
| Reglas reactivas hechas por humano | **0,3748** | **0,0%** |
| Modelo aprendido + horizonte 3 | **0,7247** | 11,26% |
| Aleatorio | 1,4426 | 11,85% |
| Q-learning tabular (mismos datos, representación distinta) | 1,7733 | 26,41% |
| Modelo aprendido + horizonte 1 | 3,0177 | 59,48% |
| Horizonte 3 con etiquetas de acción permutadas en modelo | 6,7800 | 87,41% |

Diferencia de recompensa media *a favor* de horizonte 3, pareada
por semilla, con intervalo bootstrap exploratorio agrupado:

- Horizonte 3 menos horizonte 1: **+2,2930** de recompensa
  por paso, intervalo del 95% **[+2,0154; +2,5421]**.
- Horizonte 3 menos pronósticos permutados: **+6,0554**
  **[+5,6793; +6,3657]**.
- Horizonte 3 menos Q tabular: **+1,0486**
  **[+0,5724; +1,5191]**.
- Horizonte 3 menos reglas humanas: **−0,3498**
  **[−0,7975; +0,0383]**.
- Horizonte 3 menos aleatorio: **+0,7179**
  **[+0,4228; +0,9825]**.

El RMSE medio de predicción de la transición dinámica en una
secuencia independiente fue **0,02326** unidades normalizadas.

**La hipótesis principal se sostuvo en este simulador**:
anticipar tres pasos ayuda más que un paso, y desordenar
las predicciones afecta gravemente. Sin embargo, **un
control reactivo hecho a mano superó a la arquitectura
aprendida en coste medio**. El intervalo frente al control
humano incluye 0; no se reivindica una ventaja del agente.

Ejemplo local comprobado con el predictor calibrado:
ante sed=0,2, fatiga=0,1, agua=0,05 y calor=0,
la versión de un paso eligió DESCANSAR y la de tres
pasos BUSCAR AGUA. No es una instrucción programada
en el planificador; se desprende de consecuencias
previstas a diferentes horizontes.

## Cambio de mecanismos físico desconocido (OOD)

En pruebas posteriores se redujo la eficacia de beber,
descansar y obtener agua, y aumentó el efecto del calor.
No se reentrenó el modelo.

| Política | Coste con física cambiada | Fracción de sed elevada |
|---|---:|---:|
| Reglas humanas | **1,9496** | 34,48% |
| Horizonte 3 | 3,3011 | 63,91% |
| Horizonte 1 | 3,5521 | 71,05% |
| Aleatorio | 4,0144 | 62,63% |
| Q tabular | 4,4265 | 66,71% |
| Predicciones permutadas | 6,8476 | 88,63% |

Diferencia horizonte 3 vs horizonte 1: +0,2510,
intervalo bootstrap **[−0,0060; +0,5436]**;
la evidencia ya no permite concluir robustamente
que la planificación más larga sea mejor en este
régimen nuevo.

**Resultado negativo importante:** el agente falla
cuando las consecuencias reales se alejan del
modelo aprendido; su control de sed es peor que
una heurística conocida por el diseñador.

## Qué hemos aprendido / qué no

Sí:
- Puede aprender cambios de estado asociados a
  acciones **sin conocer las reglas del simulador**.
- Con ese modelo, planifica secuencias futuras y
  obtiene mejores consecuencias que las de un
  horizonte puramente inmediato dentro del régimen
  de entrenamiento.
- Las perturbaciones del conocimiento de acción
  (pronósticos intercambiados) perjudican la
  conducta de manera medible.

No:
- No decide su objetivo interno ni adquiere
  experiencia subjetiva.
- No demuestra curiosidad espontánea, búsqueda
  activa de información ni autoconciencia.
- No hay memoria episódica selectiva F4 implementada
  dentro de este simulador. El modelo sintetiza
  transiciones pasadas mediante regresión.
- No garantiza mejor rendimiento frente a una
  estrategia reactiva basada en conocimiento experto.
- No generaliza bien a cambios importantes en
  las leyes físicas sin reentrenamiento.

## Decisión sobre la siguiente hipótesis

La próxima prueba **no** debería consistir en
añadir un parámetro que mejore el score del
mismo mundo.

La nueva hipótesis comprobable es:

> Ante incertidumbre sobre qué acción funciona,
> ¿puede un agente elegir una experiencia que
> reduzca su error de predicción antes de ejecutar
> acciones más costosas?

Eso se puede estudiar mediante exploración
dirigida por incertidumbre (valor de la
información), comparada con exploración aleatoria
y con un agente sin exploración, y replicada bajo
cambios del entorno. Es una forma de **curiosidad
instrumental funcional**, no de sentir curiosidad.

## Reproducción

Python 3.10+ y NumPy; sin CUDA, servidor, Godot ni
las dependencias completas de V5.1.

    PYTHONPATH=. python -m unittest discover -s tests -p 'test_f6_homeostatic_rl.py' -v

    PYTHONPATH=. python scripts/f6_homeostatic_rl.py --seeds 8 --train-episodes 110 --test-episodes 12 --horizon 48

Los artefactos de Actions contienen report.json
con resultados por semilla y comparaciones pareadas.
Todos los cambios siguen aislados en la rama F6.
