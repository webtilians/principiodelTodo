# INFINITO F9-N — Protocolo preregistrado: ¿navegar hasta el cruce o recordar cuál es la salida?

**Congelado el 29-10-2026 antes de abrir la nueva confirmación.** Continuación de F9-M, PR #20, y su confirmación ya consumida (semillas 300000+). En F9-M S13 sin F4: sham 13,75 % de éxito, dropout 40,625 %, pero solo 0/80 frente a 1/80 pares con ambas variantes correctas. F9-N estudia qué parte de la trayectoria se mejoró.

## Fuente y pregunta

- Reutilizar **EXACTAMENTE los 8 checkpoints NUEVOS F9-M**, cuatro pares sham/dropout de la ejecución [GitHub Actions 40950579984](https://github.com/webtilians/principiodelTodo/actions/runs/40950579984), recuperados por SHA-256 y \`report.json\` con \`validate_training\`. Ningún entrenamiento, ajuste de parámetros, probe ni selección de checkpoints en F9-N.
- ¿El entrenamiento de dropout mejora (A) la llegada a la bifurcación, (B) la primera elección correcta al comprometer una salida, o (C) la ejecución tras haber elegido?
- Una evaluación verdaderamente autónoma: el evaluador NO impone acciones de profesor, no resetea al cruce, no pasa ninguna variable geométrica/cue/goal al agente. Se mantienen siete acciones nativas, observación parcial 148 + readout F4 512 y las recompensas originales.

## Nueva partición: una sola apertura

- Confirmación **prospectiva F9-N**: \`seed = 400000 + index*10000 + j\`, donde \`index=0..3\`, \`j=0..39\` para **S13** (40 pares/semilla = 160 pares/condición/brazo) y \`j=0..19\` para **S7** (20 pares/semilla = 80 pares/condición/brazo). Cada par invierte key↔ball, conserva geometría y arranque.
- Las semillas no solapan S7 train 1000+, development 100000+, confirmación ya consumida F9-M 300000+, ni holdout histórico 900000+. **No volver a usar este conjunto tras evaluar para ajustar método o umbral.** El holdout histórico permanece cerrado.
- *Integridad antes de confirmación:* ejecutar tests sobre 1 mundo TRAIN (solo para semántica, no para estimar resultados) y las reglas de máscaras, validate_training y manifest de cuatro artefactos antes de abrir 400000+. Si el método falla tras abrir confirmación, publicar el fallo; no optimizar con esos ejemplos.

## Intervenciones (fijadas de antemano)

Se aplican a los **512 canales F4 suministrados al actor**; F4 almacena y selecciona normalmente, el frame público de 148 dimensiones nunca cambia. Igual calendario por cada variante dentro de un mismo episodio, pesos siempre congelados.

1. \`normal\`: F4 íntegra durante toda la navegación.
2. \`cue_lost_continuous\`: condición diagnóstica histórica F9-M: cero F4 en **cada frame** para el que el observador ya vio la pista alguna vez y **la pista inicial no está visible actualmente**. Si el agente vuelve a verla, F4 reaparece durante esa observación: reproduce exactamente la semántica de F9-J/F9-M, no una máscara irrevocable. El observador usa \`cue_visible(env)\` exclusivamente para esta intervención y métricas; **no entra al actor**. Su exposición depende de la trayectoria.
3. \`fixed_4\`: cuatro pasos desde los tiempos **t=3,4,5,6**, contados desde reset (base 0), independientemente de pista, posición, reward o conducta. Para episodios que sobreviven 7 pasos, las **oportunidades de máscara** son iguales (4) entre modelos; reportar pasos programados, efectivos y proporción de episodios con 4 aplicaciones. No suponer que el efecto/memoria es igual entre modelos si el contenido F4 es distinto. Si el episodio termina antes, mantenerlo en el denominador y reportar duración incompleta. El reloj no se retrasa si no ve la pista.

En los dos cortes, se exige que el actor vea \`encoded[:148]\` original; mismo \`SelectiveF4("context",8,81000+index)\` para todos los agentes. Se registra además cuántos pasos tenían realmente M!=0 antes de anularla.

## Etapas del recorrido, definidas por geometría del entorno SOLO EN EVALUADOR

Los nombres y definiciones se fijan antes de observar resultados; todo evento es la **primera aparición** (sin contar repeticiones de bucle):

- **Pista observada**: \`cue_visible(env)\` se vuelve true en cualquier paso, usando un entorno sombra observador que intercambia la pista; NO usar detección de cualquier objeto como sustituto de la pista inicial.
- **Entrada a la bifurcación central**: primera posición con \`x == labels["success_position"][0]\` e \`y == size//2\`, la misma definición geométrica de la posición de cruce en F9-J, ahora explicitada con y central. Se registra tiempo del primer acceso.
- **Primer compromiso hacia salida**: después del acceso al cruce central, primer estado \`x == fork_x\` e \`y != mid\`. Signo de \`y-mid\` fija **upper/lower**. Esta definición capta primero entrar en un brazo, aunque más tarde cambie de opinión o acabe en el otro. **Si se llega a un brazo sin pasar por el centro**, registrar \`arm_without_center\` y resultado final, sin excluir ni reclasificar como llegada. Si se llega al centro pero nunca se entra en un brazo, \`no_commit\`, incluso si termina en timeout.
- **Elección correcta**: \`first_committed_side == labels["correct_side"]\`; elección errónea equivalente con opuesta. El evaluador nunca añade premio ni cambia las acciones.
- **Fin del episodio**: éxito original según reward>0, salida final correcta/errónea/sin salida, timeout y tiempo tras compromiso.

No usar únicamente success o \`x==fork_x\` como sinónimo de decisión correcta. Separar \`junction_arrived\`, \`committed\`, \`first_choice_correct\`, \`final_success\`; comprobar las implicaciones y contabilizar los eventos anómalos explícitamente.

## Estadísticos preregistrados

**Primarios S13** (todas las condiciones, con especial atención a \`cue_lost_continuous\`): para cada brazo y semilla, y agregados de 160 pares / 320 episodios: 
- Pista vista; acceso al cruce central; primer compromiso entre quienes llegaron; primera elección correcta **entre todos los episodios** (sin sesgo de selección); acierto primera elección **condicional a compromiso** (con denominador y advertencia de selección), éxito final y timeout.
- Pares en los que **ambas variantes ven la pista**, **ambas alcanzan cruce**, **ambas se comprometen en lados opuestos** y **ambas escogen la salida correcta al primer compromiso**, y por separado **ambas terminan con éxito**. Publicar también números individuales por variante key/ball.
- Descomposición descriptiva de mejoras dropout−sham por semilla: \`P(llegada)\`, \`P(compromiso|llegada)\`, \`P(acierto primero|compromiso)\`, \`P(éxito)\`. Las probabilidades condicionadas pueden sufrir sesgo por población de alcanzadores distinta; **no atribuir causalidad a una diferencia condicionada**.
- Contraste de exposición fija (t=3..6) y cobertura efectiva de blackout, reportando cualquier fracaso o uso parcial de máscara. La condición fija no garantiza misma observación ni mismo número de pasos totales después de divergir.

**Secundarios S7:** mismo desglose para comprobar que no se sacrifica el caso corto. Exponer diferencias de cumplimiento del reloj y de episodios con pista no vista.

**Indicadores de evidencia descriptiva:** concluir mejora de memoria/elección autónoma S13 **solo si** hay ganancia apreciable y replicada en pares con las dos *primeras decisiones* correctas (no solo éxito bruto), además de progreso en llegada; para atribución, priorizar números absolutos y variabilidad entre cuatro semillas en vez de tests que fingen 320 réplicas independientes.

## Auditoría y límites

- No ajustar políticas, programa de intervención, semillas, métricas, límites de pasos ni criterios de éxito al conocer confirmación; pruebas de software se hacen con TRAIN o fixtures.
- *No confundir* éxito tras un timeout menor con uso de memoria; no confundir la primera acción izquierda/derecha en la intersección con el primer compromiso espacial (un giro puede no producir desplazamiento).
- La pista se intercambia cambiando también el objetivo del entorno; se exige distinta salida correcta por par.
- S13 con dropout tuvo exposición irregular a la ablación en F9-M; F9-N vuelve a reportarla para \`cue_lost_continuous\`, y la intervención exógena \`fixed_4\` sirve para aislar el **calendario** de la trayectoria (no para demostrar un estimador causal puro de memoria).
- Las cuatro semillas son las únicas réplicas independientes de entrenamiento; los mundos nuevos dan precisión sobre esos agentes, no generalización a otros sistemas.
- El nuevo conjunto de confirmación se considera **consumido tras ejecutar una vez**; cualquier experimento posterior debe preregistrar nuevos mundos, no rebuscar en 400000+.

## Resultados posibles

- Si mejora llegada pero sigue siendo casi cero el número de pares de primera decisión correcta: se aprendió principalmente navegación/evitación de bloqueo, no memoria autónoma robusta.
- Si mejora el acierto pareado al primer compromiso, incluso si algunos episodios no finalizan: evidencia más fuerte de uso de memoria a la hora de escoger, limitada a quienes llegan y no suficiente todavía para toda la trayectoria.
- Si desaparece mejora con \`fixed_4\` y cobertura igualada: la diferencia F9-M depende de calendario/alcance de interrupción más de lo previsto.
