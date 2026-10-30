# INFINITO F9-N — Resultados: navegación hasta el cruce frente a memoria de la salida (S13)

**29 de octubre de 2026.** [GitHub Actions #40953360163](https://github.com/webtilians/principiodelTodo/actions/runs/40953360163): **SUCCESS** (tests, gate de los ocho checkpoints, cuatro evaluaciones y agregado). [PR #21](https://github.com/webtilians/principiodelTodo/pull/21), borrador sobre F9-M #20. [JSON agregado versionado](./INFINITO_F9N_RESULTS_2026-10-29.json) y [artifact original del agregado #12932268331](https://github.com/webtilians/principiodelTodo/actions/runs/40953360163/artifacts/12932268331). Los 4 artefactos por semilla guardan además **todos los pares y episodios**, no solo promedios.

## Pregunta resuelta

F9-M había observado éxito autónomo S13/blackout sham **13,75 %** frente a dropout **40,625 %**, pero solo **1/80 pares** con ambas pistas correctamente resueltas por dropout (0/80 sham). F9-N separa *haber visto cue*, *llegar a bifurcación*, *comprometer por primera vez una salida*, *acertar ese primer compromiso* y *completar el episodio*, reutilizando los mismos **8 modelos F9-M congelados**, sin PPO/BC nuevos.

**Hallazgo principal:** en S13, al suprimir F4 mientras la pista inicial ya no es visible, **la ventaja del dropout es primordialmente de navegación hasta el cruce**. Llega el **81,25 %**, frente al **28,75 %** del sham (+52,5 puntos). Una vez comprometidos con un brazo, la fracción de decisiones correctas es **51,54 % dropout frente a 50 % sham**. Ambos porcentajes están cerca del azar, y no pueden interpretarse causalmente como diferencia entre la misma población: los modelos seleccionan trayectorias distintas. El éxito final sube de **14,375 % a 41,875 %**, pero se puede explicar descriptivamente por **llegar más**, no porque el agente recupere fiablemente qué pista vio.

El criterio fuerte de memoria contrafactual S13/blackout — **ambas identidades key/ball correctas al primer compromiso dentro del mismo mundo** — queda en **7/160 pares dropout frente a 0/160 sham**: mejora leve en el nivel de pares, todavía insuficiente para afirmar memoria robusta autónoma.

## Diseño preregistrado e integridad

- Ocho checkpoints **originales de F9-M**, recuperados de [run 40950579984](https://github.com/webtilians/principiodelTodo/actions/runs/40950579984); antes de abrir la nueva confirmación, \`validate_training\` verificó los **8 SHA-256**, mismas arquitecturas, 380.552 parámetros, hashes de inicio emparejados, 600 mundos S7 × dos cue, 40 épocas BC, 20.096 pasos PPO por agente. F9-N **no entrenó ni ajustó pesos o probes**.
- Partición nueva de semillas **400000 + index*10000 + j**, j=0..39 para S13: **40 pares × 4 semillas = 160 pares / 320 episodios por brazo y condición**. S7 j=0..19: **80 pares / 160 episodios por brazo/condición**. Ninguno en F9-M 300000+, train 1000+, development 100000+ o heldout histórico 900000+, que permanece cerrado.
- Autonomía nativa sin teacher forcing, con 7 acciones, observación pública 148, F4 512. El selector/almacenamiento F4 continuó funcionando, igual por brazo; únicamente se ablanó el **readout entregado al actor**.
- Tres condiciones predefinidas: \`normal\`; \`cue_lost_continuous\` que reproduce el control F9-J/F9-M (en cada frame con pista antes vista y ahora invisible, F4=0; si se vuelve a ver reaparece temporalmente); \`fixed_4\` corte exógeno estrictamente en t=3..6 desde reset, sin leer pista/objetivo/ruta.
- El evaluador solamente observa etiquetas y geometría **después** de decidir: \`junction_arrived\` al alcanzar posición central en columna de bifurcación, \`first_committed\` al entrar por primera vez a brazo superior/inferior, \`first_choice_correct\` contra la pista, \`success\` por recompensa nativa. No se descartan los atascados, que quedan en los denominadores.
- Integridad: tests de fechas/semillas, contador de máscaras, no alteración de los 148 canales, first-commit y pares invertidos; el gate de ocho checkpoints pasó; todos los cuatro jobs confirmatorios y agregado pasaron. No se volvió a entrenar ni escoger modelos tras confirmación.

## S13: descomposición de todas las etapas

Las tasas no condicionadas usan los **320 episodios** por brazo y condición, incluyendo fallos de llegada, timeouts y falta de compromiso. Las tasas \`correcto / comprometido\` muestran su denominador.

| Condición | Brazo | Cue visto | Llega al cruce | Elige brazo | Primer brazo correcto / todos | Correcto / comprometido | Éxito final | Ambos correctos primer brazo /160 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Normal | Sham | 96,25 % | 75 % | 74,06 % | 60,31 % | 81,43 % (237) | 60,31 % | 55 |
| Normal | Dropout | 94,38 % | 77,81 % | 76,56 % | 62,81 % | 82,04 % (245) | 62,81 % | 74 |
| **Pista ya no visible, F4=0** | **Sham** | 96,25 % | **28,75 %** (92) | 28,75 % | 14,38 % | **50 % (92)** | 14,38 % | **0** |
| **Pista ya no visible, F4=0** | **Dropout** | 94,38 % | **81,25 %** (260) | 81,25 % | 41,88 % | **51,54 % (260)** | 41,88 % | **7** |
| Reloj fijo 4 pasos | Sham | 96,25 % | 74,06 % (237) | 72,5 % | 59,69 % | 82,33 % (232) | 59,69 % | 51 |
| Reloj fijo 4 pasos | Dropout | 93,13 % | 81,88 % (262) | 80 % | 68,44 % | 85,55 % (256) | 68,44 % | 84 |

**Comparaciones dentro de las cuatro semillas:**
- S13 con corte posterior a la pista, mejora de **llegada** dropout−sham en **4/4 semillas**: +40; +21,25; +81,25; +67,5 puntos.
- Mejora de **éxito** en **4/4 semillas**: +17,5; +12,5; +46,25; +33,75 puntos.
- Eso **no** se acompaña de una mejora relevante del porcentaje correcto al primer compromiso cuando se condiciona a haber llegado: cerca de 50 % en ambos brazos.
- En \`fixed_4\` las **320/320 evaluaciones de cada brazo** recibieron los **4 pasos programados y efectivos de blackout** (100 % de episodios; media exactamente 4 por brazo). Así el calendario de corte se iguala entre políticas. Frente a sham, dropout tiene llegada +7,81 puntos, acierto no condicionado +8,75 puntos, primera decisión correcta en **84 vs 51 de 160 pares**, pero **una de cuatro semillas empeora mucho** (−47,5 pp en éxito), y el F4 vuelve luego a estar disponible. Esto sí indica **robustez a interrupciones breves**, no sustitución permanente de F4.
- En la intervención S13 ligada a pérdida de pista, el **número de pasos con F4 efectivamente anulado** fue muy desigual: **323,07 por episodio** sham frente a **57,41 dropout**. Es consecuencia, al menos en parte, de la mayor duración/atasco de los episodios de control. La diferencia de exposición no se corrigió seleccionando episodios.

En este MiniGrid la primera elección espacial correcta coincidió con el éxito terminal para **todas** las condiciones S7/S13 confirmadas: tras comprometer un brazo, no apareció una etapa adicional de fallo de ejecución medible por estos eventos. Por tanto, la descomposición relevante aquí consiste esencialmente en **llegar al cruce y elegir el lado correcto**. Se registraron por separado los pocos agentes que alcanzaron el centro pero nunca se comprometieron; no se contaron como elecciones correctas.

## S7: el caso corto sí mejora también el primer compromiso

| Condición | Brazo | Llega | Correcto entre comprometidos | Éxito final | Pares ambos primeros correctos |
|---|---|---:|---:|---:|---:|
| F4 normal | Sham | 92,5 % | 97,97 % | 90,625 % | 65/80 |
| F4 normal | Dropout | 100 % | 100 % | 100 % | 80/80 |
| Sin F4 cuando pista invisible | Sham | 85 % | 42,65 % | 36,25 % | 2/80 |
| Sin F4 cuando pista invisible | **Dropout** | **100 %** | **71,875 %** | **71,875 %** | **35/80** |
| Reloj fijo 4 pasos | Sham | 90,625 % | 55,86 % | 50,625 % | 6/80 |
| Reloj fijo 4 pasos | **Dropout** | **100 %** | **93,75 %** | **93,75 %** | **70/80** |

En S7 el dropout aumenta tanto la probabilidad de llegar como el acierto en la primera elección bajo ambas ablaciones. A diferencia de S13/cue_lost, los pares con **ambas pistas resueltas correctamente** sí aumentan materialmente (**2→35** en corte ligado a la pista y **6→70** en reloj fijo, sobre 80). Es compatible con **aprendizaje de robustez útil para decisiones de memoria de trayecto corto**, aunque no permite atribuirla a una neurona o componente aislado.

## Interpretación científica delimitada

1. **Respuesta F9-N sobre S13:** el gran avance de F9-M bajo F4 persistentemente ausente procede sobre todo de **reducir bloqueo y alcanzar la bifurcación**; la elección sigue cercana a azar entre quienes se comprometen. El +27,5 pp de éxito no es evidencia de resolver la identidad de pista de forma autónoma y robusta.
2. **Interrupción corta distinta:** con reloj exógeno durante 4 pasos, dropout conserva buena navegación y mejora el número de pares con ambas decisiones correctas S13, pero F4 está disponible de nuevo al decidir. Demuestra robustez a una interferencia acotada, **no** retención autónoma durante un apagón prolongado.
3. **No confundir condicionar y causar:** \`correct_given_commit\` se calcula sobre agentes que llegan y deciden, una selección de episodios que cambia con la política. Solo se puede comparar descriptivamente; la métrica de **pares ambos correctos / todos** es más rigurosa para prevenir estrategia de elegir siempre un lado.
4. La validación fue una única nueva partición prospectiva del mismo MiniGrid, no prueba fuera de distribución en juegos distintos. Cuatro semillas independientes, mundos emparejados. No abrir holdout histórico ni volver a usar 400000+ para ajustar un nuevo entrenamiento.

## Pregunta F9-O (NO ejecutada en F9-N)

Para aislar memoria de dependencia externa en S13, un próximo ensayo debería entrenar con **ventanas de bloqueo largas o currículos crecientes**, sobre mundos TRAIN nuevos, y comparar con igual exposición frente a control, pero **solo confirmar en una nueva partición preregistrada**, distinta de 300000+, 400000+ y del holdout 900000+. Mantener como desenlace central las dos primeras decisiones correctas por par en navegación autónoma; el porcentaje de éxito en una sola variante puede mejorar aun sin recordar qué salida es la correcta.
