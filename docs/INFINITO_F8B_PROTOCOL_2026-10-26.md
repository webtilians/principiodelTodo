# F8b — Adaptación online de la dinámica física

Fecha: 26-10-2026. Estado: **protocolo preparado; implementación y ejecución pendientes**.
Este documento no contiene resultados F8b ni registra el experimento en el catálogo.
Se prepara después de comprobar despliegue, ejecución y persistencia del laboratorio.

## Pregunta y alcance

¿Actualizar online un modelo de transición mejora el retorno neto cuando cambian
coeficientes físicos que el modelo calibrado de F8 no puede explicar cambiando
solamente su creencia entre dos regímenes conocidos?

Separar este efecto del valor de comprar información. Primera comparación: idéntica
política de observación y planificación, cambiando solo actualización física sí/no.
Después evaluar la interacción con inspección activa. No integrar todavía en V5.1.

## Mundo e información permitida

Conservar las cinco acciones de F8 y las variables observables. Reutilizar exactamente
la física y la calibración inicial por semilla entre políticas. En evaluación, cambiar
sin aviso la eficacia de buscar agua y los costes de fatiga; los valores posteriores
no deben ser solo el intercambio de los dos regímenes de F8. El agente nunca recibe
el instante del cambio, sus coeficientes ni etiquetas verdaderas de régimen en test.

El aprendizaje online solo accede a (observación previa, acción, observación posterior),
recompensa y sensores realmente comprados. Emitir y registrar cada predicción antes
de actualizar con el resultado. El planificador no puede llamar al simulador verdadero.
La calibración offline etiquetada heredada de F8 debe seguir declarándose como tal;
no presentar este diseño como aprendizaje completo sin supervisión.

## Implementación propuesta

Módulo separado `observatory/active_world_f8b.py`. Modelo de deltas por acción/contexto,
con actualización incremental y olvido exponencial; mantener capacidad, representación
y horizonte de planificación iguales al modelo congelado. Una actualización por
transición observada, sin simulaciones privilegiadas ni reentrenamiento con datos futuros.

Elegir coeficiente de olvido y regularización solo con semillas de desarrollo. Antes
de la evaluación reservada, guardar un manifiesto con SHA, configuración, dependencias,
semillas y huella del protocolo. Este documento aún es un diseño: los hiperparámetros
no están fijados y no se afirmará que la prueba es confirmatoria hasta congelarlos.

## Escenarios y comparaciones

Escenarios separados: estable; salto físico oculto; deriva gradual; degradación de
sensor; cambio combinado; sensores puramente aleatorios. No mezclar el fallo de sensor
con el cambio físico en la comparación primaria.

Políticas: modelo congelado; mismo modelo con actualización online; actualización
sin olvido; actualización con observaciones temporalmente barajadas como control;
y oráculo de física solo como techo de referencia, nunca como comparación equitativa.
Cruzar congelado/online con nunca inspeccionar y VOI en análisis secundarios. Comparar
inspección aleatoria con presupuesto predefinido, sin ajustar su frecuencia mirando test.

Usar ruido potencial indexado por semilla, episodio, paso y canal: las decisiones de
una política no deben desplazar el generador aleatorio de las demás. Mantener iguales
datos iniciales, memoria máxima y presupuesto de planificación; registrar tiempo de
cómputo por política porque actualizar tiene un coste adicional.

## Evaluación y criterio de decisión

- Desarrollo: semillas 20000–20015. Evaluación reservada propuesta: 30000–30047.
  Comprobar que no se hayan usado antes; si se usaron, elegir otro rango antes de mirar resultados.
- Inicio técnico: 4 semillas de desarrollo, 3 episodios y 24 pasos. No es evidencia científica.
- Evaluación propuesta: 48 semillas, 20 episodios y 80 pasos por escenario; validar primero
  coste/memoria en el servidor compartido. No iniciar una ejecución larga automáticamente.
- Métrica primaria: diferencia emparejada por semilla del retorno neto medio por paso
  online menos congelado en cambio físico abrupto, con idéntica política de observación.
- Secundarias: retorno tras cambio, error predictivo antes de actualizar, coste de sensores,
  deterioro en estable, recuperación temporal, uso de memoria y coste computacional.
- Intervalos bootstrap sobre semillas, no sobre pasos; guardar todas las semillas,
  distribuciones y diferencias. Una comparación primaria; el resto se etiqueta exploratorio.
- Criterio propuesto a congelar: límite inferior del IC 95% primario mayor que cero,
  diferencia media de al menos 0,005 por paso, y límite inferior en estable mayor que -0,005.
  Si no se cumple, informar el fallo y no reajustar umbrales sobre evaluación reservada.

## Pruebas previas y conexión al portal

Comprobar: ausencia de acceso a etiquetas/coficientes ocultos; predicción anterior a
actualización; igualdad de mundos y ruido potencial; reproducibilidad; actualización
nula equivalente a congelado; controles negativos; límites de memoria; JSON finito;
resumen y archivo descargable; ejecución/cancelación y persistencia del trabajo.

Solo después: registrar `f8b` en catálogo, resumidor, etiquetas y tests; publicar en una
rama experimental, desplegar y comprobar que aparece y ejecuta el perfil técnico.
Separar resultados exploratorios de cualquier evaluación reservada. No inferir
conciencia ni emergencia subjetiva de una mejora del control predictivo.
