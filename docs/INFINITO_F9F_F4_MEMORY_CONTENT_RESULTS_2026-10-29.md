# INFINITO F9-F — Auditoría de información conservada en F4

Fecha: 29 de octubre de 2026. Análisis de **desarrollo**, retrospectivo, SIN nuevos entrenamientos ni inferencia.

Fuente: cuatro artefactos originales del [run F9-E #40908055330](https://github.com/webtilians/principiodelTodo/actions/runs/40908055330), 4 semillas. Código: observatory/f9f_memory_content_audit.py y tests/test_f9f_memory_content_audit.py. Los checkpoints congelados se comparan con sus SHA-256, y el test reservado sigue cerrado.

## Resultado exacto en S7

| Medida de los 40 pares con pista original/invertida | Antes de PPO | Después de PPO |
|---|---:|---:|
| Ambos escenarios observaron la pista | 40/40 | 40/40 |
| Readout F4 idéntico en el primer avistamiento | 40/40 | 40/40 |
| F4 ya guardaba **llave y pelota** en ese momento | 40/40 | 40/40 |
| Readout F4 idéntico antes de la última acción | 40/40 | 40/40 |
| Toda la secuencia de readouts F4 idéntica | 40/40 | 40/40 |
| Trayectorias de acciones/posiciones idénticas | 40/40 | 40/40 |
| Acierto en ambas variantes del par | 0/40 | 0/40 |

**Diagnóstico de representación:** VisibleF4.encode de F9-E almacena cada objeto observado de tipo key o ball mediante Event(key=object_id*ncolors+color_id, payload=0) sobre FIFO/Store, con capacidad para tres identidades distintas. No conserva el papel de pista, el orden contextual semántico ni su relación con las salidas. La imagen inicial ya puede contener llave y pelota. Al intercambiar cuál era la pista, los doce valores de la memoria **no cambian** en ninguno de los 40 mundos S7.

Para esos pares y esas trayectorias, **ningún decodificador que lea solamente esos 12 canales podría distinguir qué variante de pista ocurrió**. Esto NO significa que el actor entero no tenga información: la vista simbólica actual y el estado recurrente de LSTM se proporcionan por otros canales. Tampoco prueba nada negativo sobre LearnedRelevance, F5 o mecanismos cognitivos completos que NO han sido incorporados aquí.

## Transferencia S13 (40 pares, solo tras PPO)

| Medida | Resultado |
|---|---:|
| Pista observada en ambas variantes | 38/40 |
| Mismo readout al primer avistamiento, sobre total de pares | 24/40 |
| Mismo readout antes de la última acción | 26/40 |
| Misma secuencia completa de readouts | 26/40 |
| Ambos tipos key y ball en F4 antes de la última acción | 40/40 |
| Misma secuencia de acciones y posiciones | 36/40 |

En S13 la memoria **puede diferir** por cambios de visibilidad/recorrido. Una diferencia no prueba que se haya codificado la pista relevante. El último paso de un episodio agotado por tiempo no debe confundirse con una decisión correcta en el cruce.

## Límites e integridad

- El análisis usa exclusivamente el historial de F9-E ya generado; ninguna acción, observación, recompensa, política o entorno nuevo. No se calcula significación estadística a partir de pares pertenecientes a las mismas cuatro semillas de entrenamiento.
- F4 aquí significa el Store y FIFO originales, no el selector aprendido LearnedRelevance ni F5.
- Este resultado demuestra pérdida de **información discriminante en el readout F4 actual**, pero no imposibilidad del algoritmo completo.
- Los 40 pares S7 de F9-E provenían de 4 semillas y 10 mundos de desarrollo por semilla; no se abre el holdout.

## F9-G recomendado (NO ejecutado aquí)

El siguiente paso es **diseñar memoria de eventos con contexto observable**, que pueda codificar qué objeto se observó y en qué instante/posición *egocéntrica* de la imagen, sin filtrar etiquetas «pista», éxito, coordenadas absolutas ni metas del simulador. Antes de gastar más cálculo, exigir por tests que intercambiar la pista inicial cambie el nuevo readout incluso cuando ambos tipos están presentes y que la distinción perdure hasta la bifurcación. Igualar capacidad y parámetros, conservar controles de lectura a cero y corrupción de identidad/contexto. Solo después entrenar la política con comparación causal completa y las siete acciones nativas.

En esta fase no se cambia el PC ni VPS y no se abren semillas retenidas.
