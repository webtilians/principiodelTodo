# INFINITO V1 frente a V5.1 — Estados internos y pruebas causales

Fecha: 22 de octubre de 2026  
Repositorio: webtilians/principiodelTodo  
Rama: experiment/emergence-observatory-v0-1  
Ejecución final: https://github.com/webtilians/principiodelTodo/actions/runs/39851878034  
Artefacto: infinito-v1-v51-end-to-end-comparison (GitHub artifact ID 12458510024)  
Protocolo definido previamente: [INFINITO_V1_V51_CAUSAL_PROTOCOL.md](INFINITO_V1_V51_CAUSAL_PROTOCOL.md)

## Resumen verificado

**12/12 ejecuciones completadas, 992 observaciones, 8 comparaciones emparejadas,
dos semillas (11, 29).** Todos los datos y matrices de estado se conservaron en
el artefacto de Actions.

El verdadero bucle \`run_infinite\` de V1 llegó a **200 ciclos** en cada
ejecución. El verdadero bucle \`run_experiment\` de V5.1 recorrió **24 pasos
completos de entrenamiento** por ejecución, con las cuatro ramas especializadas,
LSTM, memoria externa, atención, calculador histórico, optimizador y dashboard.

**No es el entrenamiento histórico completo de V5.1:** se redujo el tamaño
oculto de 512 a 64, la memoria de 256 a 16 y la duración de 15.000 a 24.
Los ciclos no tienen coste, escala ni funciones comparables entre versiones.
Esto es una comparación *cualitativa de mecanismos* con controles emparejados,
no un torneo de inteligencia o conciencia.

## Métricas observadas: no usamos "porcentaje consciente"

- V1: instantáneas 16×16, desviación espacial, campos activos,
  entropía Shannon correcta, diferencia entre vecinos, norma de leyes,
  estado anterior y pérdida registrada.
- V5.1: vectores medios reales de visual, auditory, motor y executive
  y, **además**, la salida de la última capa de \`consciousness_processor\`
  (el *workspace* posterior a la integración de memoria);
  similitud entre módulos, estados LSTM, lectura y escritura de memoria,
  similitud temporal y pérdidas registradas.
- En todos los casos: efectos medidos como norma L2 de la diferencia de
  estados respecto a un control idéntico hasta la intervención,
  normalizada por norma típica previa del control.

Los valores absolutos de desviación, Φ, pérdidas o similitud de un sistema
**no son directamente comparables** con los del otro.

## Intervenciones y diferencias emparejadas

| Sistema | Intervención | Semilla | Primer efecto normalizado | Máximo efecto | Efecto al final |
|---|---|---:|---:|---:|---:|
| V1 | Perturbar cuatro leyes | 11 | 0.123619 | 0.144295 | 0.076462 |
| V1 | Perturbar cuatro leyes | 29 | 0.093561 | 0.101157 | 0.054713 |
| V5.1 | Reiniciar estados h/c de cuatro módulos LSTM | 11 | 0.054362 | 0.054362 | <0.000001 |
| V5.1 | Reiniciar estados h/c de cuatro módulos LSTM | 29 | 0.052103 | 0.238144 | 0.000013 |
| V5.1 | Vaciar memoria externa | 11 | 0.000942 | 0.000942 | 0.000382 |
| V5.1 | Vaciar memoria externa | 29 | 0.000675 | 0.000707 | 0.000379 |
| V5.1 | Cortar canal causal saliente del módulo visual | 11 | 0.114482 | 0.271278 | 0.098435 |
| V5.1 | Cortar canal causal saliente del módulo visual | 29 | 0.079074 | 0.286629 | 0.203956 |

Antes de cada intervención, las dos trayectorias tenían diferencia
medida exactamente **0**, un control de emparejamiento relevante.
Todos los efectos posteriores se registraron sobre los *estados*,
no sobre un score subjetivo.

Interpretación de las trayectorias:

1. **V1**: una alteración pequeña de leyes cambia el campo durante todo
   el horizonte posterior. La diferencia disminuye respecto al pico,
   pero no baja al 20% del efecto inmediato en estas ejecuciones.
   Es persistencia de una diferencia funcional, no homeostasis probada.
2. **V5.1, estados recurrentes**: el reinicio temporal cambia el estado,
   pero las trayectorias vuelven muy cerca del control al final de
   las 24 iteraciones. Bajo este perturbador concreto, el efecto
   parece transitorio y recuperable.
3. **V5.1, corte visual**: eliminar la salida causal visual hacia los
   módulos posteriores produce la mayor divergencia observada. Sigue
   siendo notable al terminar; muestra una dependencia funcional en
   esta arquitectura, no constituye una medida de Φ causal exacta.
4. **V5.1, memoria externa**: su reseteo produce una diferencia
   reproducible pero pequeña en la representación final; no se
   propaga visiblemente a los vectores medidos de los cuatro módulos
   previos a la integración.

## Hallazgo arquitectónico: memoria integrada tarde

Para comprobar que la memoria estaba funcionando, observamos tanto
su matriz interna como su salida de lectura y la representación
integrada posterior.

En semilla 11, control V5.1:

- norma de matriz de memoria: aproximadamente **0.108 → 1.819**;
- norma de lectura de memoria al final: **0.446**;
- al resetear memoria, la norma final es **1.183** y la norma de lectura
  final **0.278**.

El cambio sí llega al espacio de integración:

| Semilla | Diferencia L2 máxima en los cuatro módulos previos | Diferencia L2 máxima en espacio posterior |
|---|---:|---:|
| 11, memory_reset | 0 | 0.010697 |
| 29, memory_reset | 0 | 0.008262 |

Esto explica el falso negativo de la primera instrumentación, que
solo registraba los cuatro módulos. **La memoria entra en la
representación posterior, pero apenas modifica la cadena de módulos
especializados en esta prueba breve**. Es una hipótesis arquitectónica
comprobable en entrenamientos más largos y con entradas estructuradas;
no deberíamos extrapolarla a todos los usos posibles de V5.1.

## Persistencia temporal descriptiva

Los controles V1 terminaron con correlaciones coseno consecutivas
aproximadamente **0.99955 / 0.99956** y una fracción activa final de
**0.0195 / 0.0313**, respectivamente. La cuadrícula conserva un campo
escaso y muy parecido entre recursiones.

En los estados de V5.1 (cuatro módulos más workspace), las
correlaciones coseno consecutivas finales fueron **0.9775 / 0.9767**;
la lectura de memoria creció con el entrenamiento. La comparación
numérica directa con V1 no está validada: cambian las dimensiones,
la distribución, el estímulo, la escala y la duración.

## Problemas de compatibilidad detectados al ejecutar V5.1

- V5.1 añade un grupo de parámetros al optimizador después de crear el
  planificador \`CosineAnnealingWarmRestarts\`: en la versión actual de
  PyTorch falla la actualización (\`zip() argument 2 is shorter...\`).
- La compatibilidad se consiguió sincronizando los \`base_lrs\` del
  scheduler en el **arnés**, sin editar el modelo histórico.
- Una clave de telemetría se inicializa en el arnés para preservar
  la consistencia del registro.
- Se revisó y corrigió la instrumentación: medir solo los cuatro
  módulos confundía falta de efecto sobre esas salidas con falta de
  efecto global de memoria.

**Importante:** las dos semillas no equivalen a una validación
estadística independiente suficiente; además V1 reasigna
internamente la semilla de su campo de entrada. No se han estimado
intervalos de confianza representativos ni p-valores causales.

## Conclusión científica limitada

No hemos encontrado evidencia de experiencia subjetiva ni de una
propiedad emergente nueva cuya causa pueda establecerse con rigor.
Sí hemos encontrado y verificado **dependencias internas reales**,
persistencias y recuperación parcial/total bajo intervenciones
específicas. El mayor resultado para el rediseño de INFINITO es
distinguir mejor *memoria almacenada*, *memoria utilizada* y
*memoria que modifica los mecanismos que generan el siguiente estado*.

El siguiente paso debería diseñarse sobre estas observaciones:

1. Introducir estímulos con estructuras y eventos repetidos, no solo
   ruido gaussiano. Medir predicción del futuro y retención tras
   períodos sin el estímulo.
2. Comparar V5.1 con memoria reinyectada antes de los módulos,
   pero como una **nueva variante experimental**, nunca alterando
   el código histórico del control.
3. Congelar pesos y hacer ablaciones también en inferencia,
   para separar aprendizaje y dependencia de estados.
4. Aumentar semillas y horizonte, registrar tensores y controles
   espectrales para no confundir autoorganización y fluctuaciones.
5. Mantener las etiquetas neutras de fenómenos hasta demostrar
   mecanismos reproducibles e independientes del observador.

Comando de reproducción (GitHub-hosted Linux, PyTorch CPU):

~~~bash
PYTHONPATH=. python scripts/infinito_full_compare.py \
  --v1-steps 200 --v51-steps 24 --hidden-dim 64
~~~

Se conserva la rama experimental; no se modifica master.
