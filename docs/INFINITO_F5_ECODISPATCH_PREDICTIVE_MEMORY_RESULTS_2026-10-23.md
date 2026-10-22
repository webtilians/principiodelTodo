# INFINITO F5 — EcoDispatch para memoria predictiva · resultados

**Fecha:** 23 octubre 2026  
**Rama:** experiment/f5-ecodispatch-predictive-memory  
**Protocolo previo:** [F5 protocol](INFINITO_F5_ECODISPATCH_PREDICTIVE_MEMORY_PROTOCOL.md)  
**Código:** [predictive_memory.py](../observatory/predictive_memory.py)  
**Tests:** [test_predictive_memory.py](../tests/test_predictive_memory.py)  
**GitHub Actions verificado:** https://github.com/webtilians/principiodelTodo/actions/runs/39866272153

## Diseño realizado

Este experimento es independiente de V5.1: **no está integrado en su memoria neuronal**. El mundo simulado emite 12 tipos de acontecimientos en orden aleatorio y un contexto (4 posibles). Tres tipos causan casi todo el resultado futuro en cada contexto, y los principales causantes son distintos según el contexto. Se eligen aleatoriamente para cada semilla.

Se entrena un regresor ridge contextual sobre **960 episodios anteriores etiquetados y completamente registrados**. Esa observación completa durante entrenamiento es una ventaja explícita: **en evaluación** cada política ve **un evento a la vez**, no la consecuencia futura, y solo conserva 3 de los 12. Luego el mismo predictor intenta recuperar el resultado desde los eventos retenidos. La API pública AdmissionController.observe(j, x) impide pasarle resultados futuros.

La política inspirada en EcoDispatch prioriza el descenso de la pérdida predictiva esperada; aproxima el potencial del error omitido mediante sumas cuadráticas aditivas, más 0.04×magnitud de evento como utilidad inmediata. No incorpora ningún nuevo teorema de OpenAI ni hereda garantías generales de k-median/k-server.

**Comparadores:** FIFO, reservorio aleatorio, magnitud (lambda=0), predictor global sin contexto y dos intensidades predictivas (lambda=.25 y lambda=1). El selector con pesos verdaderos y el predictor que puede ver los doce eventos solo son diagnósticos, no competidores con el mismo presupuesto.

## Resultados verificados (MSE; menor es mejor)

20 semillas; cada semilla contiene 240 casos inéditos **por escenario**. Cada caso se entrega a todas las políticas. El resultado real se revela solo para evaluar. Los valores MSE de escenarios distintos no se deben comparar entre sí porque su distribución difiere.

| Política | Estable | Cambio oculto de reglas | Aleatorio | Interacción no lineal |
|---|---:|---:|---:|---:|
| Predictiva contextual λ=1 | **0.1460** | 11.0850 | 6.0297 | **0.5131** |
| Predictiva contextual λ=.25 | 0.1910 | 11.0369 | 5.9911 | 0.5588 |
| Magnitud / λ=0 | 1.9011 | 9.4134 | 5.9775 | 2.2184 |
| Sin contexto | 1.7060 | 9.4913 | 5.9468 | 2.0667 |
| FIFO | 4.2626 | **7.1275** | **5.8581** | 4.7462 |
| Reservorio | 4.3293 | 7.1716 | 5.8680 | 4.7027 |
| Selector oracular (diagnóstico) | 0.1453 | 6.9086 | 5.9107 | 0.5128 |
| Predictor completo (diagnóstico) | 0.1294 | 11.0816 | 6.0600 | 0.4931 |

Los diagnósticos oraculares **no** constituyen límites rigurosos sobre el mejor MSE. En el escenario de cambio de reglas, el predictor sigue equivocado incluso cuando recibe los doce eventos.

### Criterio previo: mundo estable

La política λ=1 superó los controles fijados antes de ejecutar el benchmark.

- Contra FIFO, diferencia MSE −4.1166 (IC bootstrap exploratorio, semillas emparejadas: −4.2910 a −3.9460).
- Contra λ=0, diferencia −1.7551 (IC −1.8392 a −1.6683).
- Contra estimador global, diferencia −1.5600 (IC −1.6194 a −1.4956).
- Acierto del signo del resultado futuro: 94,5% con λ=1, 63,9% con FIFO.

Esta gran ganancia es **en gran parte esperada por construcción**: el mundo genera relaciones mayoritariamente lineales por contexto y el estimador utiliza la familia de modelos apropiada. No hay una emergencia espontánea.

### Resultado negativo esencial: cambio oculto

Al cambiar las relaciones causales justo antes de evaluar, sin reentrenar, λ=1 empeora a **11.0850**, mientras FIFO queda en **7.1275**. Diferencia λ=1 − FIFO: +3.9575 (IC +3.7291 a +4.2063). Estudiar el futuro con un modelo descalibrado puede perjudicar.

### Control negativo genuino

El control aleatorio recibe **tanto un historial de entrenamiento aleatorio como resultados futuros aleatorios independientes de eventos**. λ=1 arroja 6.0297 frente a 5.8581 de FIFO, sin ventaja predictiva. El modelo sobreajusta algo de ruido.

### Prueba de no linealidad

Se añade una interacción no lineal fuera de la clase del predictor. λ=1 aún obtiene 0.5131 frente a 4.7462 de FIFO gracias al componente lineal fuerte. Esto **no** prueba aprendizaje de interacciones.

## Limitaciones y decisión

No prueba conciencia, curiosidad, metas internas, aprendizaje autónomo de semántica ni funcionamiento de V5.1. Es un controlador simbólico con supervisión pasada y memoria limitada en inferencia; el potencial de predicción es una aproximación aditiva y se descalibra cuando cambia el mundo.

**Resultado científico principal:** en un entorno predecible, anticipar consecuencias guía bien la conservación de recuerdos; cuando las leyes cambian, el mismo mecanismo se vuelve perjudicial. El siguiente estudio necesitaría incorporar detección de cambio/estimación de incertidumbre y entrenamiento con observación parcial, antes de la integración neuronal con V5.1.

**Integridad:** la investigación está aislada de master, de EcoDispatch y de las implementaciones históricas.
