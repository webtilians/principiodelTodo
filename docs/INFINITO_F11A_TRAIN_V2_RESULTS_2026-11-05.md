# INFINITO F11-A — TRAIN v2: efectos retardados y cambios de reglas

**5 de noviembre de 2026. Estado técnico: SUCCESS en contratos y cuatro réplicas. Estado científico: TRAIN_ONLY; NO es PASS ni FAIL del F11-A original.** No se abrió DEV, CONFIRM ni la partición de composición.

## Procedencia y separación

La enmienda prospectiva del TRAIN v2 fue registrada en el commit 6edb98166fbbe6eca0dcd792adc2b91faad2cfff, **antes del código y del entrenamiento**. Se conserva el TRAIN v1 de PR #87 y el documento preregistrado de PR #86.

Ejecución auténtica: [GitHub Actions run 42034978959](https://github.com/webtilians/principiodelTodo/actions/runs/42034978959), commit f9e206737b5d936170ad8b5b64cc1e9504a3794f. Contratos SUCCESS, r0/r1/r2/r3 SUCCESS. Artefactos f11a-train-v2-r0, -r1, -r2, -r3 con train_events.jsonl.gz, checkpoint.json y train_summary.json. El run conserva hashes y trazas; los artefactos de Actions caducan según la retención configurada.

Reproducción: python -m unittest tests.test_f11a_train_v2 -v; python -m observatory.f11a_train_v2 --replica N --worlds 512 --episodes 30 --probes 2 --out DIR.

No se modificó src/infinito3, memoria personal, servicios MCP, secretos, herramientas externas ni producción.

## Implementación

- Las acciones opacas A–F tienen preparación de carga, recolección de material, ajuste de ruta y entrega; hay acciones no productivas, costes de fallo, ruido y recursos.
- La preparación tiene **retardo de 1–2 acciones**; la cola de efectos no es visible al agente, solo su cantidad. Las recompensas aparecen al entregar, no en el instante de preparar.
- Tras 15 episodios por mundo, se **permutan sin aviso las funciones de las acciones**, manteniendo la observación pública. Fase y semilla aparecen únicamente en trazas para auditoría.
- A0: reactivo sin memoria. A1: búsqueda entre episodios crudos y copia de acción del episodio recuperado, **sin tabla persistente de política o Q**. A2: modelo de efectos observables y planificación limitada a profundidad cinco. A3: secuencias condicionales extraídas de experiencias propias exitosas, reutilizadas y penalizadas cuando fallan, con el mismo A2 como respaldo. A3_no_options: ablación de las opciones. A4: heurística informada como techo diagnóstico, no entrenador ni óptimo matemático probado.
- Las experiencias se entregan idénticas a A1/A2/A3. Aproximadamente una quinta parte de episodios de TRAIN es elegida por A3 y entregada a todos: **potencial sesgo del colector a favor de A3**.
- Se entrenaron eventos/modelos tabulares y opciones; **no se entrenó un LLM ni una red con gradientes**.

## Datos de entrenamiento

En total **2.048 mundos de TRAIN**, **61.440 episodios de exploración**, **689.659 acciones**, **12.798 episodios exitosos (20,83%)**, **294.141 efectos retardados materializados**. A3 recogió **7.621 opciones distintas** sumando réplicas.

| Réplica | Episodios | Éxitos de exploración | Acciones | Opciones |
|---|---:|---:|---:|---:|
| r0 | 15360 | 3216 | 171322 | 1869 |
| r1 | 15360 | 3255 | 172130 | 1933 |
| r2 | 15360 | 3001 | 173830 | 1868 |
| r3 | 15360 | 3326 | 172377 | 1951 |

## Diagnósticos exclusivamente sobre los mismos mundos vistos en TRAIN

Cada fase contiene 512 probes/réplica x cuatro réplicas = 2.048 episodios por brazo. Ninguno de estos datos mide generalización fuera de TRAIN.

| Agente | Reglas originales, fase 0 | Reglas cambiadas, fase 1 |
|---|---:|---:|
| A0 reactivo | 268/2048 (13,09%) | 248/2048 (12,11%) |
| A1 episódico | 56/2048 (2,73%) | 1652/2048 (80,66%) |
| A2 modelo de efectos | 56/2048 (2,73%) | 126/2048 (6,15%) |
| **A3 políticas aprendidas** | **855/2048 (41,75%)** | **1866/2048 (91,11%)** |
| A3 sin opciones | 56/2048 (2,73%) | 126/2048 (6,15%) |
| A4 informado | 2047/2048 (99,95%) | 2048/2048 (100%) |

A3 reutilizó 13.114 pasos de opciones durante probes en mundos vistos y reportó 3.110 abortos de guarda. Son hechos descriptivos sin validez causal suficiente. Los mundos están correlacionados y no representan pruebas OOD.

## Confusores graves y NO-GO científico para DEV

1. **A2 no es todavía un baseline competente para efectos retardados:** solo planifica cinco pasos, y el estado visible no describe los tipos/edades de los efectos pendientes. Que A3 lo supere puede reflejar una desventaja artificial del control.
2. **A1 sufre fuerte sesgo de actualidad:** recupera episodios exitosos recientes, que describen sobre todo la fase cambiada; por eso obtiene 80,66% en fase 1 pero apenas 2,73% en fase 0. Aunque ahora es recuperación episódica auténtica y no política agregada, debe controlarse dicho sesgo temporal antes de usarlo como RAG competente.
3. **El colector incluye acciones elegidas por A3 en 20% de los episodios:** todos ven las mismas experiencias, pero A3 puede favorecer sus propias trayectorias. Hacen falta controles de colector no sesgados.
4. **A3 se probó en mundos de entrenamiento**, reutilizando procedimientos específicos ya aprendidos, no en familias nuevas. Un algoritmo de memorización de secuencias podría explicar parte o toda la ventaja.
5. No se midió transferencia composicional, reentrenamiento online en mundos desconocidos, seguridad fuera del simulador, igualdad estricta de latencia/bytes/tokens ni se ejecutó auditor separado de los gates preregistrados G0–G5. A3_no_options implementa exactamente A2, por lo que su empate no sustituye una ablación independiente.

**Decisión: TRAIN v2 COMPLETO como prueba técnica. NO-GO para DEV ni CONFIRM con estos controles.** Antes de abrirlos: mejorar planificador A2 con creencias/atribución temporal y horizonte apropiado; comprobar A1 con recuperación temporal y presupuestos medidos; desacoplar la exploración de A3; introducir auditor técnico y pruebas de adaptación fuera de los mundos de entrenamiento, únicamente en nuevas pruebas TRAIN autorizadas. Documentar toda revisión prospectivamente sin modificar umbrales preregistrados después de observar DEV.

**No presentar los porcentajes como nueva inteligencia demostrada ni fusionar este estudio con el producto.**
