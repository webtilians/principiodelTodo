# INFINITO F7 — ¿Investigar o actuar? Valor instrumental de información

**Pre-registro:** publicado antes de ejecutar la batería completa. Rama `experiment/f7-active-information-value`. No afecta a master, EcoDispatch, ni V5.1.

## Auditoría de la trayectoria

- F1/F2/F3: estados internos y lectura de memoria no son equivalentes; bloquear escrituras selectivamente recuperó capacidad de decodificación, pero el criterio de escribir era un oráculo externo.
- F4: una política simbólica aprende demandas por categoría en un entorno que por diseño premia memorizar frecuencias. No demuestra relevancia semántica ni conciencia.
- F5: protocolo de potencial predictivo, no integración demostrada en el modelo neuronal.
- **F6 tiene un problema de control:** la función World inicializa los parámetros de dinámica usando `seed//1000`. En entrenamiento utiliza `i+3` (y, al variar el identificador de episodio, la mayoría de episodios usa el mismo bloque); en evaluación utiliza `i+50401`. Ambos comparten **distribución de parámetros**, no idénticos coeficientes. Por tanto la etiqueta previa "in_distribution" significa *misma familia de distribuciones*, no mundo idéntico. Además, el Q tabular comparte transiciones, pero no necesariamente una puesta a punto equivalente. No inferimos superioridad general de arquitectura.
- Una mejora de RL bajo función de utilidad programada no prueba motivos internos autónomos, ni emociones, ni conciencia.

## Hipótesis principal

¿Puede un agente estimar la utilidad de comprar información **antes de observarla**, renunciar a exploraciones sin valor y mejorar sus decisiones frente a políticas que nunca exploran, siempre exploran o inspeccionan al azar?

Es una tarea pequeña de **decisión parcialmente observable con información costosa**. Conoce que existen dos posibles estados ocultos, pero **aprende de episodios de experiencia** cuál de dos instrumentos está relacionado con ellos. Ninguna acción conoce directamente el estado actual antes de tomar la decisión.

### Mundo y costes

- Estado oculto binario z. El agente recibe un indicador gratuito ruidoso y el nivel de consecuencias (bajo/alto).
- La fiabilidad del indicador depende del tipo de situación (fácil/difícil), observable, pero las fiabilidades específicas se aprenden.
- Dos posibles inspecciones; su identidad informativa cambia entre semillas, y una devuelve ruido completamente independiente del estado (**noisy TV**).
- Una inspección cuesta 0,12 unidades de utilidad, y solo puede realizarse una antes de elegir una de las dos acciones finales.
- Si la acción final coincide con z, utilidad +stake; en caso contrario, -stake. `stake` es observable y es 0,2 o 1,0.
- Tras actuar, el agente recibe feedback de acierto/fallo, del que puede deducir qué estado tenía ese episodio. Esto llega **después** de cerrar todas las decisiones.

### Entrenamiento y test

- Calibración inicial exclusivamente con inspecciones asignadas aleatoriamente, para que todas las políticas compartan la **misma información**; se registran observación, inspección elegida y feedback final.
- Aprendizaje Beta-binomial de precisión de cada inspección y del indicador gratuito, sin etiquetas explícitas del instrumento valioso.
- Durante test, emparejar exactamente los mismos episodios latentes y observaciones potenciales; cada agente solo puede ver los sensores que solicita.
- Prueba con reglas estables, otra donde cambia cuál instrumento es fiable sin aviso y un control totalmente sin información (ambos inspecciones puramente aleatorias).
- La política que aprende en línea recibe solamente sus inspecciones efectivamente realizadas y el feedback final. Se compara con la misma política congelada.

### Políticas

- **VOI adaptativa:** comparar utilidad esperada de actuar inmediatamente frente al valor de inspeccionar cada sensor, menos coste; actualizar la estimación de fiabilidades según observaciones y feedback posterior.
- **VOI congelada:** misma fórmula, pero deja de aprender tras calibración.
- **Nunca inspeccionar**: decide usando solo el indicador.
- **Inspección aleatoria**: inspecciona al azar en la mitad de las ocasiones; integra evidencia observada en su respuesta.
- **Inspección siempre**: elige el sensor que estima más preciso, incluso cuando no compensa pagar por él.
- **Alta sorpresa / noisy TV:** elige deliberadamente el sensor históricamente menos predecible como ejemplo de curiosidad mal diseñada.
- **Oráculo de fiabilidades:** únicamente un límite de referencia; conoce las probabilidades generadoras, pero no el estado oculto de cada episodio.

### Métrica de éxito o abandono

Primaria: recompensa media real **neta del coste de información**, por episodio, promediada a nivel de semilla; bootstrap emparejado entre semillas, no entre muestras del mismo mundo. Secundarias: precisión de acción final, fracción de inspecciones, fracción de inspecciones de ruido y evolución tras cambio de sensor útil.

Criterio de éxito: en el mundo estable, VOI adaptativa debe superar al no inspeccionar, al azar y al «siempre inspeccionar», sin obtener ventaja espuria cuando ambos sensores son ruido. Ante cambio, debe adaptarse mejor que la política congelada tras un periodo de observación. Si no cumple, **F7 se clasifica como negativo** y no se integrará en INFINITO.

El mundo es diseñado para que alguna información tenga utilidad, y el algoritmo Bayesiano parte de una familia de sensores razonable. Esto es estudio de decisiones epistémicas instrumentales, **no** un descubrimiento nuevo, ni emergencia espontánea, ni experiencia subjetiva. Se compara explícitamente con oráculo para mostrar cuánto queda sin aprender.

## Relación con EcoDispatch y el resto de INFINITO

EcoDispatch decide si asumir ahora un coste para preservar cobertura futura. Aquí se decide si pagar por una observación para reducir el riesgo de una acción futura. Son **principios de decisión análogos**, pero no una aplicación directa de las garantías de k-median/k-server.

Si F7 funciona, la extensión científicamente justificable sería aplicar el mismo criterio a la necesidad de información de un predictor de consecuencias del F6 **después de subsanar las limitaciones de emparejamiento y validación**. No se debería fusionar automáticamente en V5.1 por obtener una métrica favorable.
