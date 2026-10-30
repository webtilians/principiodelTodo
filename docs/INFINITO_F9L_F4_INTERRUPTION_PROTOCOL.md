# INFINITO F9-L — Protocolo anterior a resultados: supervivencia recurrente sin entrada F4

**Fecha:** 2026-10-29. En continuidad estricta con F9-J (atribución) y F9-K (decodificación temporal).

## Pregunta principal

Después de observar una pista key/ball, ¿puede el estado oculto+celda LSTM (H) continuar transportando su identidad **si se corta deliberadamente el readout F4 desde un punto conocido del recorrido hasta el cruce**? ¿Conserva la política la siguiente acción correcta?

El resultado F9-K de 96,25 % en S13/fork **no probaba una duración de 11 pasos sin soporte externo**, ya que M era perfectamente decodificable en instantes intermedios. F9-L manipula ese soporte y observa los efectos.

## Diseño y controles preasignados

- Cuatro checkpoints originales F9-I context8 congelados y cuatro zero (control sin memoria externa), verificados con los SHA-256 de \`report.json\` original y reproducidos desde la ejecución [F9-I #40928547873](https://github.com/webtilians/principiodelTodo/actions/runs/40928547873).
- Recorridos MiniGrid S7 y S13 con **prefijos del profesor nativos e idénticos para el par de cues opuestos** hasta la primera decisión que difiere. Ni las acciones futuras ni el label key/ball llegan a la política. Son **historias guiadas, NO pruebas de navegación autónoma**.
- Intervenciones en la **entrada al actor** de 512 canales F4, no en el almacenamiento interno, selector ni observación pública de 148 canales. El selector sigue leyendo observaciones normales durante toda la intervención. La intervención se aplica en \`[inicio, fork)\`, y **se restablece el readout original en la observación del fork**; por tanto O y M en la decisión final son idénticos entre las dos historias y entre condiciones: las diferencias del acto en el fork proceden de los diferentes estados H previos.
- Cuatro condiciones context8:
  1. \`normal\`: nunca cortar.
  2. \`cut_post_cue\`: desde el primer frame posterior a la última visión de la pista, hasta antes del fork.
  3. \`cut_halfway\`: desde la mitad entera de la distancia post-cue→fork.
  4. \`cut_last_step\`: solo el instante \`fork-1\`.
  5. \`neutral_tail\`: desde \`post_cue\` en adelante poner a cero **solo los 448 canales de padding garantizados a cero** de context8 (control exacto de no-cambio; no controla la alteración OOD causada por cortar los primeros 64).
- Control \`zero\` sin F4, idéntico algoritmo de teacher forcing, tamaño de entrada y LSTM, sin modificar su input. No se interpretan diferencias entre checkpoints como estimadores causales puros de F4: fueron entrenados por separado.
- La distancia de interrupción hasta la bifurcación se deriva de visibilidad real de la pista en el evaluador, sin escoger según rendimiento. En cada par son idénticos el prefijo de acciones, la observación en el fork y el número de inputs suprimidos. No se filtran ejemplos con fallos o distancias desfavorables.
- Distancia de la interrupción = \`fork - cut_index\`, positiva en todos los mundos. En S7 se esperan cortes de unos cinco, tres y un paso; en S13 de unos once, seis y un paso (se registrarán los valores reales).
- Se comprueba por test que \`neutral_tail\` produce **exactamente** el mismo estado y acción que \`normal\`; F4 en fork se restituye en todas las condiciones, con memoria F4 del par idéntica para ambas identidades en el fork. También se comprueba que \`cut_post_cue\` y \`cut_halfway\` realmente retiran canales que eran informativos en el intervalo; de lo contrario el experimento falla.

## Decodificación y conducta, evitando confundir OOD con olvido

1. **Decodificador fijo de referencia:** entrenar probe ridge lineal con H-fork natural S7 train (80 pares/semilla), alpha=10, threshold=0, y reutilizarlo **sin recalibración** en los H perturbados S7/S13 development. Su fallo podría ser un desplazamiento de representación por inputs fuera de distribución: **no equivale automáticamente a olvido**.
2. **Decodificador readaptado por condición:** entrenar otro probe ridge lineal con **H-fork de S7 train bajo esa misma intervención** (80 pares/semilla) y evaluar **S7 y S13 development** (10 pares/semilla y tamaño), sin ajuste en development. Este control pregunta si permanece información linealmente recuperable incluso si el formato de H se desplazó. No se confunde con una nueva política: **los pesos originales no se cambian**.
3. **Decisión del actor en fork:** con O y F4 restauradas e idénticas entre cues contrarios, evaluar la **primera acción nativa** determinada por la política y registrar: exactitud respecto al profesor, pares con ambas acciones correctas, y pares donde la política cambia su acción entre cues. Si el probe readaptado conserva información pero la acción falla, indica desacoplamiento de representación/uso bajo la intervención, no pérdida total.
4. Reportar cambios de norma/distancia entre H de cues, tasa de restauración M, todos los resultados por semilla y longitud del corte. La diferencia LSTM no es equivalente a información decodificable; no inferir nada sólo de norma.

## Particiones, integridad y límites

- F9-I entrenó los agentes solamente en S7 train. Los probes se entrenan **exclusivamente en 80 pares S7 train por semilla**. Evaluación: los mismos 10 pares S7 development y 10 pares S13 development por semilla que F9-J/K (sin S13 para entrenamiento del probe).
- **Heldout permanece cerrado.** Semillas sólo admitidas por \`checked_seed('train'|'development', ...)\`; presupuesto máximo y fijo; no buscar mejores cutoffs o regularización tras ver resultados. Sin trabajo en PC/VPS.
- Reproducir exactamente la decodificación H-context8 natural S13/fork de F9-K por semilla: [100 %, 100 %, 95 %, 90 %] y S7/fork 100 % todas, además de H-zero S13/fork 50 %. Fallar CI en caso contrario.
- Cortar M durante la trayectoria provoca inputs fuera de distribución y altera el proceso de integración del LSTM, no sólo un bit aislado. Incluso si la acción no cambia, no garantiza retención semántica general; si cae, no puede concluirse que toda la memoria desapareció.
- Cuatro semillas son réplicas de agente, los 80 episodios emparejados no son 80 entrenamientos independientes. No abrir holdout ni construir afirmaciones sobre conciencia a partir de estos tests.

## Lectura planificada

- Si H sigue decodificable con corte desde post-cue en S13/fork **y** las acciones siguen orientadas por la pista, evidencia de retención recurrente sin actualizaciones F4 durante el intervalo (bajo prefijo forzado).
- Si el probe fijo falla pero el readaptado recupera información, indicar cambio de código interno, **no amnesia**.
- Si ambos probes fallan mientras mantener F4 hasta mitad mejora, evidencia compatible con que la actualización externa es necesaria para retención **lineal** robusta en este protocolo.
- Si H decodifica pero la acción falla, investigar qué impide convertir conocimiento interno en decisión antes de ampliar arquitectura.
