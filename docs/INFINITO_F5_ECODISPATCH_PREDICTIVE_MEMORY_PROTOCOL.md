# INFINITO F5 — EcoDispatch para memoria predictiva · protocolo previo

**Estado:** diseño congelado antes de ejecutar el benchmark.  
**Rama exclusiva:** `experiment/f5-ecodispatch-predictive-memory`; rama madre: `experiment/emergence-observatory-v0-1`.

## Pregunta verificable

¿Puede un estimador de consecuencias futuras aprender qué experiencias conservar, aun cuando todos los tipos de acontecimientos aparecen con igual frecuencia, el orden varía y la utilidad cambia según el contexto?

Esta F5 es **un experimento independiente**, no una nueva versión del modelo V5.1 y no pretende medir conciencia.

## Transferencia matemática desde EcoDispatch

EcoDispatch puntúa una decisión con coste inmediato y cambio de cobertura futura.

Aquí cada evento observado de categoría j y magnitud x_j ocupa un slot; la memoria M guarda como máximo k eventos. Un modelo del mundo, entrenado **solo con episodios anteriores**, aprende coeficientes \(\hat\beta_{c,j}\) para predecir un resultado futuro Y bajo contexto c.

Potential de predicción perdida si no almacenamos un evento:

\[
\widehat\Phi(M,c,x)=\sum_{j \notin M} \widehat\beta_{c,j}^2 x_j^2.
\]

Es una aproximación **aditiva** del error cuadrático futuro; los términos cruzados y correlaciones pueden invalidarla. También es equivalente a una cobertura ponderada de distancia 0/1 (caso discreto simple de k-median); **no se implementa ni hereda** ningún algoritmo de aproximación de OpenAI.

Se acepta/expulsa un evento según su beneficio de reducir \(\widehat\Phi\), más una pequeña utilidad inmediata de novedad común a políticas. El caso λ=0 ignora el potencial futuro. En todos los casos el predictor de salida final usa exactamente el mismo modelo de coeficientes aprendido y solo lee hasta k eventos retenidos.

## Entorno con consecuencias, no consultas de claves

- Doce clases de eventos; cada episodio presenta las 12 una vez, en orden aleatorio, con un valor gaussiano firmado independiente.
- Cuatro contextos observables; cada contexto activa **tres causas principales distintas** de los demás (selección aleatoria oculta al selector). El objetivo futuro es suma ponderada de los eventos causales más ruido observacional.
- El contexto está disponible antes de decidir; nunca se proporciona la etiqueta «importante», los coeficientes verdaderos ni el resultado futuro hasta terminar el episodio.
- Memoria limitada a **3 de 12 eventos**. No se permite consultar los valores descartados durante la predicción.
- Para cada una de **20 semillas independientes**, se generan 960 episodios de calibración, 240 episodios inéditos de evaluación y un escenario de ruptura donde la relación causa-contexto cambia tras entrenar.
- Entrenamiento del modelo predictivo mediante ridge lineal contextual sobre logs completos del **periodo de calibración**. Se explicita que el registro completo de calibración es un privilegio de entrenamiento offline; la restricción de memoria rige la decisión prospectiva de evaluación, no esa adquisición previa.
- Control negativo: resultados futuros independientes de todos los eventos, con varianza equivalente; el sistema no debería conseguir una ganancia predictiva fiable sobre un predictor constante.

## Baselines con capacidad idéntica

- FIFO (tres eventos más recientes).
- Reservorio aleatorio (tres eventos equiprobables).
- Sorpresa/λ=0 (magnitud observada, sin previsión de consecuencias).
- Política global sin contexto (efecto medio por categoría).
- Potencial futuro contextual λ=0.25 y λ=1; **λ=1 es la comparación primaria**.
- Selector oracular con coeficientes verdaderos y predictor completo sin límite: **diagnósticos**, no políticas competidoras implementables.

## Medidas y criterio de continuación

**Principal:** MSE del resultado real futuro al predecir desde los 3 recuerdos retenidos, en ejemplos no vistos y emparejados. Secundarias: error absoluto, acierto de signo, porcentaje de varianza explicada, cobertura de potencial verdadero.

**Éxito mínimo para justificar integrar a V5.1:** λ=1 debe tener MSE menor que FIFO, sorpresa λ=0 y política global en media por semilla, con intervalo bootstrap de diferencias emparejadas entre **semillas** completamente favorable, Y esa supuesta ventaja no debe aparecer bajo resultados aleatorios.

**Límites explícitos:** aun si se supera, el entorno es sintético, lineal, diseñado para que una parte de la utilidad sea aprendible por contexto. El experimento demostraría optimización predictiva controlada, no emergencia, conciencia o aprendizaje autónomo de significado.

**Diagnóstico de fallo:** se incluirá cambio oculto de reglas sin reentrenamiento (concept drift). Si la política sigue apostando por predicciones obsoletas, debe fallar; es una condición de frontera semejante a la reversión de EcoDispatch al pasar de demanda sintética a histórica.

El código histórico de INFINITO y el repositorio EcoDispatch no se alteran.
