# INFINITO F6 — Anticipación de consecuencias y regulación de necesidades

**Hipótesis fijadas antes de los resultados confirmatorios.** Rama exclusiva experiment/f6-homeostatic-world-model, partiendo de experiment/f5-ecodispatch-predictive-memory. No se modifica V5.1 histórica, EcoDispatch ni master.

## Pregunta

¿Puede un agente aprender de su experiencia los efectos de acciones desconocidas, imaginar sus consecuencias y escoger acciones que regulen variables internas mejor que una decisión puramente inmediata?

Es un experimento de **control basado en modelos** (*model-based RL / model predictive control*) inspirado en la función de coste prospectivo de EcoDispatch. No hay conciencia ni sed fenomenológica simulada.

## Mundo

Observaciones: déficit de agua ("sed"), fatiga, inventario de agua, contexto de calor visible. Acciones: **esperar, beber, descansar, buscar agua**. La dinámica que transforma observaciones y acciones se codifica **solo en el simulador**, no en el planificador. Una búsqueda puede consumir energía y aumentar sed ahora pero suministrar agua útil después. Beber solo funciona si hay suficiente inventario.

Las magnitudes de las acciones y de los procesos físicos varían por semilla, y las observaciones contienen ruido. La temperatura puede cambiar independientemente del agente.

**Objetivo explícito:** minimizar una penalización de déficit de agua, fatiga y desabastecimiento, junto al coste de actuar. El objetivo se programa; no aparece espontáneamente.

## Aprendizaje con presupuesto equitativo

Cada semilla de entrenamiento recoge 110 episodios × 48 pasos bajo **política exploratoria aleatoria**. Se entrena un modelo lineal ridge por acción para aproximar el siguiente estado a partir de observación y acción.

El agente puede consultar ese modelo, pero **no** las ecuaciones del simulador. El objetivo de entrenamiento de la dinámica es predecir estados siguientes; el objetivo de decisión es minimizar coste homeostático.

Durante las pruebas los pesos de modelo y del baseline Q se congelan, evitando filtración de futuro.

## Políticas

1. **Planificación 3 pasos**: imagina consecuencias de 64 secuencias posibles y maximiza recompensa descontada; hipótesis principal.
2. **Planificación 1 paso**: decide por recompensa inmediata usando exactamente el mismo modelo predictivo; aísla el efecto de mirar al futuro.
3. **Dinámica permutada**: intercambia deliberadamente pronósticos entre acciones; control de dependencia del conocimiento causal aprendido.
4. **Q-learning tabular fuera de política**: aproximación model-free entrenada con las mismas transiciones, a título exploratorio; resolución de estado y entrenamiento distintos.
5. **Reglas reactivas escritas a mano**: control fuerte privilegiado por conocimiento humano de los efectos de las acciones; no es un baseline de aprendizaje con información equiparable.
6. **Aleatorio**: referencia inferior.

## Test retenido y cambio de mecanismo

Ocho semillas, cada una con doce episodios de evaluación (48 pasos) en **dos** entornos: parámetros del mismo régimen de calibración y un **cambio no anunciado** en eficacia de acciones y calor.

Los entornos se emparejan por misma semilla entre políticas. No se entrena en los episodios de evaluación.

**Métrica primaria:** coste homeostático medio por paso (menor es mejor). Secundarias: fracción de pasos con sed elevada (>0.85), RMSE del modelo dinámico en transiciones nuevas y diferencias por semilla con bootstrap agrupado.

## Condición de continuación, fijada antes de ver la prueba

Para considerar validada la *ventaja específica de anticipar*:

- La planificación a tres pasos debe mejorar la de un paso en el test sin cambio de dinámica, con intervalo de diferencias entre semillas enteramente favorable.
- Debe superar también la versión con pronósticos intercambiados.
- Debe informarse del resultado frente a reglas reactivas, aleatorio y Q-learning, incluso cuando estos sean superiores.
- Su comportamiento ante el cambio de mecanismos debe mostrarse por separado: una mejora de entrenamiento no demuestra robustez bajo distribuciones nuevas.

## Límites

No se modela curiosidad auto-originada. Las observaciones de entrenamiento son exploradas aleatoriamente, los objetivos están programados y el mundo es pequeño y completamente observable. El agente aprende un modelo de consecuencias y planifica acciones, no aprende por sí mismo el objetivo de supervivencia.

Este experimento **no constituye** reproducción V5.1 ni aplicación literal de teoremas k-server/k-median de OpenAI; únicamente traslada el principio de penalizar consecuencias futuras de EcoDispatch a una política secuencial.

El siguiente paso, solo si las pruebas lo justifican, sería memoria selectiva de transiciones relevantes y experimentación epistemológica, con controles de capacidad y sin recompensa arbitraria por "curiosidad".
