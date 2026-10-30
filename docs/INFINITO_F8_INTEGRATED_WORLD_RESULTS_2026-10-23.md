# INFINITO F8 — Laboratorio integrado de predicción e información

**Fecha:** 23-10-2026  
**Estado:** implementación exploratoria probada en GitHub Actions, conectada al portal web.  
**Rama:** `experiment/f8-laboratory-portal` (no se modifica `master`).  
**Código:** [observatory/active_world_f8.py](../observatory/active_world_f8.py)  
**Workflow:** [infinito-f8-integrated.yml](../.github/workflows/infinito-f8-integrated.yml)  
**Verificación:** https://github.com/webtilians/principiodelTodo/actions/runs/39927537597

## Pregunta científica

¿Puede un agente con una función de utilidad fija **aprender empíricamente las consecuencias de sus acciones** y decidir **si adquirir información costosa** sobre el estado oculto del mundo, sin que el planificador reciba la solución?

Esto reúne dos mecanismos anteriormente aislados:

- F6: representar acciones y sus consecuencias mediante un modelo de transición, y planificar con horizonte temporal.
- F7: calcular el valor esperado de información de un sensor, restando su coste.

**No equivale a una nueva red V5.1**: es un banco de pruebas independiente para decidir qué mecanismos merecen ser integrados después.

## Mundo sintético

Variables internas observables: sed, fatiga, agua disponible y calor.

Cinco acciones: esperar, beber, descansar, buscar agua en el oeste y buscar agua en el este. El mundo opera bajo uno de dos regímenes ocultos; la eficacia de las búsquedas depende del régimen. El agente **no recibe el régimen verdadero**. Existe una pista gratuita débil y dos sensores opcionales: uno habitualmente fiable (94 %), otro ruido (50 %). Cada consulta cuesta **0,045** unidades de recompensa.

El contexto relevante puede cambiar sin avisar en mitad de un episodio. La física se genera por semilla y es **idéntica** en la calibración y la prueba estacionaria de esa semilla; los resultados de evaluación utilizan otro flujo de ruido. Esta separación corrige el problema de emparejamiento detectado en F6.

### Entrenamiento y toma de decisiones

1. Por semilla, recoger 1.800 transiciones de entrenamiento estratificadas por régimen, acción, calor y disponibilidad de agua. Las etiquetas de régimen son visibles **solo en el experimento de calibración offline**; el planificador de test no dispone de ellas.
2. Aprender medias de cambios de sed, fatiga y agua para cada contexto y acción. El estimador predictivo **no conoce los coeficientes originales del mundo**.
3. Calibrar fiabilidad de pista y de sensores con observaciones anteriores.
4. En test, mantener una creencia probabilística sobre el régimen a partir de la pista y de los resultados experimentados al buscar recursos.
5. Planificar con horizonte de dos pasos mediante secuencias comunes de acciones. Si obtener un sensor permite cambiar a una secuencia con mayor utilidad esperada, y la mejora supera su coste, realizar la consulta.
6. Observar el resultado y actualizar la **creencia de régimen** para la decisión siguiente. En esta primera versión, el estimador de transiciones aprendido en calibración permanece fijo.

**Limitación de diseño que no debe ocultarse:** el agente conoce de antemano que existen exactamente dos regímenes. Esta primera F8 no aprende a descubrir cuántos mundos posibles existen, y su modelo físico no se recalibra online. El aprendizaje en test ocurre en la **inferencia del régimen**, no en los parámetros del modelo físico.

## Evaluación

- 12 semillas independientes.
- 10 episodios inéditos por semilla y escenario, 42 pasos por episodio.
- Tres escenarios: `stable`, `switch` (régimen cambia sin anuncio), `null` (sensores y pista no informan del régimen).
- Ocho políticas con el mismo mundo/ruido/estímulos potenciales: valor de información adaptativo (VOI), nunca inspeccionar, inspeccionar siempre, inspección aleatoria, VOI sin memoria entre pasos, inspeccionar ruido, reglas reactivas escritas por humano y control oracular (conoce el régimen).
- Métrica primaria: recompensa neta media por paso; mayor es mejor. Se calcula la diferencia de recompensa media **emparejada por semilla** y un bootstrap exploratorio de semillas, no de pasos.

## Primer resultado en CI

| Política | Estable | Régimen cambia | Sensores/pista sin información |
|---|---:|---:|---:|
| **VOI + predicción** | **-0,12513** | **-0,13251** | **-0,14619** |
| Nunca inspeccionar | -0,13626 | -0,14314 | -0,14619 |
| Inspección aleatoria | -0,14404 | -0,15112 | -0,16829 |
| Inspeccionar siempre | -0,16014 | -0,16236 | -0,18921 |
| VOI sin memoria entre pasos | -0,15973 | -0,15668 | -0,35269 |
| Inspeccionar ruido | -0,18029 | -0,18783 | -0,19156 |
| Reglas reactivas (heurística fija, oeste) | -1,49540 | -1,19558 | -1,58662 |
| **Oráculo de régimen** | **-0,11357** | **-0,11468** | **-0,10907** |

### Diferencias pareadas (recompensa, mayor es mejor)

- Estable, VOI – no inspección: **+0,01113** (IC bootstrap 95 % exploratorio **[+0,00684; +0,01642]**).
- Estable, VOI – aleatoria: **+0,01891** (**[+0,01744; +0,02041]**).
- Régimen cambiado, VOI – no inspección: **+0,01064** (**[+0,00708; +0,01468]**).
- Régimen cambiado, VOI – aleatoria: **+0,01862** (**[+0,01596; +0,02122]**).
- Escenario nulo, VOI – no inspección: **0,00000**, porque la política aprendida **no inspeccionó** con sensores sin utilidad.
- El VOI se aproxima, pero no alcanza, al control oracular. El controlador reactivo aquí es deliberadamente un baseline imperfecto: no usa el contexto oculto y siempre busca en el oeste cuando falta agua. Su bajo rendimiento **no** debe venderse como demostración general de superioridad cognitiva.

**Fracción de inspecciones VOI:** 1,92 % en estable, 1,81 % con cambio y 0 % en nulo. Inspecciona poco pero, cuando lo hace, el beneficio supera el coste bajo el modelo aprendido.

Los intervalos son exploratorios, con 12 semillas. Repetir más pasos/semillas evalúa reproducibilidad **dentro del simulador**; no elimina la limitación de que la estructura de dos regímenes y la función de utilidad se programaron.

## Qué sí / qué no se ha logrado

**Sí:** en un entorno parcialmente observable, el mismo agente utiliza conocimiento aprendido de transiciones para planificar y usa incertidumbre para comprar observaciones con coste. Se ha medido en pruebas independientes de la calibración, con controles de información inútil y comparaciones emparejadas.

**No:** el agente no aprende metas propias; no tiene recuerdos autobiográficos como V5.1; no descubre mundos más allá de los dos definidos; no adapta online las leyes físicas; no demuestra emergencia inédita, experiencia subjetiva o conciencia. Es una combinación reproducible de control predictivo y decisión Bayesiana ya conocidos.

**Nota metodológica:** la implementación concreta y las cifras de este primer benchmark se han afinado y verificado en esta misma iteración; la prueba es **exploratoria**, no una confirmación independiente preregistrada.

## Repetición con más potencia en tu portal

Inicia INFINITO Laboratory en esta rama; elige **F8 · Mundo activo integrado** y un perfil.

- **Rápido:** 4 semillas, 320 transiciones, 3 episodios, horizonte 18.
- **Estándar:** 12 semillas, 1.800 transiciones, 10 episodios, horizonte 42.
- **Largo:** 48 semillas, 5.000 transiciones, 24 episodios, horizonte 60.

Para replicación genuina, cambia el parámetro `seed-offset` (por ejemplo 10000, 20000, 30000) sin ajustar el código con los resultados de esas pruebas. Se guardan `report.json` y logs por ejecución, descargables en ZIP.

Sin portal:

```bash
python -m observatory.active_world_f8 --seeds 12 --train 1800 \
  --test-episodes 10 --horizon 42 --seed-offset 1000 \
  --output observatory_results/f8_active_world
```

## Criterio para el experimento siguiente

Antes de fusionar con INFINITO Brain o V5.1, introducir **F8b** como experimento separado: cambios inéditos en la física, actualización online del modelo físico sin etiquetas de régimen, sensores que se degradan sin aviso, y pruebas con distribuciones de contextos no vistas. Exigir mejoría de utilidad real frente a baselines con el mismo presupuesto de datos, memoria y cálculo.

Si los baselines conocidos explican todo, registrar esa conclusión y cambiar la hipótesis, en vez de añadir una falsa «capa de conciencia».
