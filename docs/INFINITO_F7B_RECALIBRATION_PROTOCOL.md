# F7b — Recalibración activa ante errores inesperados (exploratorio)

**Creado después de revisar F7; NO debe presentarse como confirmación preregistrada.** Se ejecuta exclusivamente como experimento nuevo para explicar el fracaso de F7 tras una inversión de sensores.

## Observación previa de F7, fechada antes de programar F7b

En 24 semillas F7: en régimen estable la política VOI obtiene recompensa neta 0,3633 frente a 0,2447 sin inspecciones. Ante inversión de sensores, su segunda mitad cae unos 0,1085 por debajo de no inspeccionar, aunque supera a una versión VOI congelada. F7 no consigue descubrir eficazmente la nueva fuente informativa.

## Hipótesis F7b

Tras varias observaciones que contradigan sistemáticamente la fiabilidad esperada de un sensor, el agente debería revisar su modelo y **elegir adquirir datos nuevos**, incluso pagando un coste inmediato, para volver a decidir mejor.

## Intervención

Control denominado reactive_recalibration:

1. Aprende las fiabilidades iniciales con **idénticas** 240 muestras de calibración de F7.
2. Realiza selección VOI normal sin acceder al cambio de reglas del entorno.
3. Tras cada inspección efectiva compara si el sensor acertó con la probabilidad predicha **antes** de conocer el estado real.
4. Si hay 18 observaciones del mismo sensor en una ventana de 24 muestras y la discrepancia media de aciertos supera 0,22 (por debajo del valor esperado), trata esto como señal de cambio; reinicia únicamente las estadísticas de precisión de sensores.
5. Reserva **40 inspecciones forzadas**, alternando los dos sensores, para identificarlos de nuevo; los resultados posteriores ajustan sus estimaciones. Este número y el umbral son elecciones del experimentador, no «curiosidad emergida».
6. Sigue sin disponer del estado oculto hasta después de la decisión final. Los episodios y parámetros físicos son los mismos que F7.

**Ojo:** esta política adicional conoce la posibilidad abstracta de cambio, incorporada por diseño. Es un método de detección de cambio + sondeo dirigido, no espontaneidad. La siguiente fase debería aprender la valoración de reabrir una investigación, no recibir esa regla preprogramada.

## Comparaciones y controles

Se compara F7b con la política original F7, la versión congelada y la política sin inspecciones, emparejadas semilla a semilla, para:

- Mundo estable (penaliza exploración innecesaria).
- Inversión sorpresiva de cuál sensor es útil (mide adaptación, especialmente últimas 100 decisiones).
- Mundo con dos sensores aleatorios (penaliza curiosidad por ruido no aprendible).

Métrica principal F7b, **exploratoria**: recompensa neta en últimas 100 decisiones de cada semilla tras el cambio, frente a no investigar y F7. Se reporta además número de detecciones, inspecciones forzadas, falsas detecciones y recompensa estable y sin información.

No se modifican los parámetros de F7 ni se vuelve a redefinir retroactivamente su hipótesis. No se construye todavía una red neuronal ni se atribuye conciencia.
