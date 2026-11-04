# INFINITO F10-P-C — confirmación independiente del aprendizaje de efectos desde salidas públicas de herramientas

**4 de noviembre de 2026. Estado científico: F10PC_CONFIRM_PASS. Estado técnico: F10PC_TECHNICAL_COMPLETE. 4 réplicas originales sin reentrenar; 2048 nuevos casos por brazo y auditor independiente finalizado.** Todos los ficheros científicos de F10-P originales y los 20 checkpoints F10-L originales permanecieron inmutables; la rama no se ha fusionado a master.

## 1. Respuesta directa

La memoria factorizada **EventEffects** de F10-P, aprendida exclusivamente a partir de respuestas externas P/S/R/X diseñadas para el experimento, volvió a proponer **S→R→X en los 2048 casos nuevos** de MEMORY+FLIP1, una secuencia que jamás ejecutó en TRAIN. Con los **mismos** snapshots originales y sin actualizar una sola transición, E100 logró **1990/2048 = 97,17%**. El control que recibe 50% de eventos omitidos aleatoriamente (E50) dio también 1990/2048; el control de puntuaciones aditivas B dio 1008/2048=49,22%. Tres réplicas fueron perfectas y la réplica 2 obtuvo 454/512=88,67%, pero incumplió un estrato preregistrado (S_after 66,41%). **G0 y G1 pasan 3/4; G2-G4 pasan 4/4; PASS global conforme a protocolo, sin rescatar r2.**

El éxito respalda **transferencia en nuevas semillas de la MISMA familia sintética** de mundos, dadas abstracción, funciones, objetivos, interfaces y búsqueda de programas por investigadores. **No demuestra** aprendizaje de semántica de herramientas desconocidas, invención autónoma de operaciones, políticas generalizables de agente completo, datos naturales ni preparación inmediata para producción.

## 2. Prerregistro, artefactos congelados, prueba temporal

- Código/memorias originales F10-P: commit **9de6adb6ff0299b361d72258b9d988f832256201**; run original [41888234405](https://github.com/webtilians/principiodelTodo/actions/runs/41888234405), PR #71. Se recuperaron los artefactos f10p-original-public-events-0, -1, -2, -3 y f10p-independent-public-event-science del run **original**, no de un nuevo entrenamiento.
- Habilidades F10-L originales: [run 41685499939](https://github.com/webtilians/principiodelTodo/actions/runs/41685499939), los cinco checkpoints por réplica y G0 original cotejados por SHA. Los 28 snapshots de siete brazos y cuatro réplicas F10-P se cotejaron con su frozen.json antes y después de abrir CONFIRM.
- Protocolo [F10-P-C preregistrado](INFINITO_F10PC_PREREG_2026-11-04.md) en **commit 2c1edcfcf8eac2bd5b8e5904216ddfe2c360b9f5**, publicado ANTES de implementar confirmación y abrir 28300000+.
- **Fallo técnico anterior a abrir CONFIRM:** [run 41895819923](https://github.com/webtilians/principiodelTodo/actions/runs/41895819923) abortó en las 4 réplicas durante el primer chequeo de metadatos por tratar como diferentes el XOR LUT (1,0) de Python y [1,0] cargado desde JSON. El arreglo **2bd8d5d6c8ac4538539f06a9419d698ab0d8b376** normalizó ambos con JSON antes de la comparación estricta. No se abrió ningún mundo 28300000+, no hubo resultados científicos ni ajustes; el error y su corrección se conservaron explícitos en el documento de lanzamiento.
- **ÚNICA ejecución confirmatoria que abrió mundos:** [run 41896086606](https://github.com/webtilians/principiodelTodo/actions/runs/41896086606), launch commit **4caf39464f8c8b7f5db3995dcae8c1ac6f639198**, cuatro jobs de réplica SUCCESS y un job de auditor SUCCESS. Se verificó el diff nulo en el código original del generador, ejecutor, modelos, producto y src/infinito3.
- La auditoría original/confirmación está en el artefacto **f10pc-original-independent-confirm-audit** del run 41896086606; los cuatro conjuntos crudos de 3584 predicciones y sus 4 intervenciones por réplica en **f10pc-original-frozen-confirm-0..3**. El auditor usa solo NumPy+JSON, no Torch, simulador ni learner. No es auditoría humana externa.

## 3. Preflight obligatorio: la confirmación no abrió mundos sin autenticación

Antes de generar ningún caso nuevo:
1. Cargar exactamente las **7 memorias originales por réplica** y **5 checkpoints originales F10-L por réplica**, y verificar todos los SHA contra manifiestos congelados del run original.
2. Reconstruir cada programa elegido, score y resultado exacto en **8960 filas originales DEV/OOD × cuatro modos por réplica**, es decir **35.840 filas DEV/OOD originales y 143.360 resultados intervenidos** reejecutados, desde los registros y habilidades originales. Concuerdan los SHA de casos, programas, slots seleccionados, recompensas y respuestas.
3. Recalcular el G0 DEV original: r0 PASS, r1 PASS, r2 **FAIL** (MEMORY0=87,50%, umbral=90%), r3 PASS; total **3/4**. No se alteró la evaluación retrospectiva ni los umbrales.
4. Solo entonces reutilizar el generador **original F10-P**, sin modificar su física: cambiar temporalmente en memoria el OOD_BASE de 28400000 a **28300000**, llamar al mismo dataset OOD y restaurar la constante. Cada fila mantiene contexto MEMORY+FLIP1, 9/11 informes y 4 regímenes de fuentes del protocolo. No hubo ajustes al predictor tras observar confirmación.

La nueva confirmación cubre **128 mundos base/r, cuatro variantes vinculadas/r = 512 casos/r**. Siete brazos E100/E50/E0/E_noX/E_shuffleX/B_reward/I_privileged, mismo registro de planificación y cuatro intervenciones por cada programa, sin recopilar nuevos episodios TRAIN.

## 4. Resultados CONGELADOS en CONFIRM

| Réplica | E100 públicas | E50 mitad observaciones | E0 terminal | E_noX | E_shuffleX | B_reward | I privilegiado | SRX elegida E100 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | **100,00% (512/512)** | 100,00% | 51,95% | 51,95% | 51,95% | 48,44% | 100,00% | 512/512 |
| 1 | **100,00% (512/512)** | 100,00% | 53,13% | 53,13% | 53,13% | 47,66% | 100,00% | 512/512 |
| 2 | **88,67% (454/512)** | 88,67% | 53,52% | 53,52% | 53,52% | 48,05% | 88,67% | 512/512 |
| 3 | **100,00% (512/512)** | 100,00% | 53,91% | 53,91% | 53,91% | 52,73% | 100,00% | 512/512 |
| **Total descriptivo** | **97,17% (1990/2048)** | **97,17%** | 53,13% (1088/2048) | 53,13% | 53,13% | 49,22% (1008/2048) | 97,17% | **2048/2048** |

**No considerar los 2048 casos independientes:** 128 mundos/r con cuatro variantes vinculadas. Bootstrap percentil IC95 con 2000 remuestreos por mundo base y RNG preregistrado.

Réplica 2: IC95 E100 por mundo **[85,35%,91,60%]**. Estrato U_before/U_after/S_before=96,09% cada uno; **S_after=85/128=66,41%**, por debajo del mínimo preregistrado 70% y por eso **G1=FAIL individual**, aunque el total 88,67% supere 80%. Su DEV original también queda G0=FAIL individual. El mismo fallo persiste en el control privilegiado I, lo que sugiere una limitación del selector F10-L congelado bajo cambios de la evidencia acreditada, no fallo de la secuencia SRX. Esta es interpretación compatible con las observaciones, no un diagnóstico causal independiente del selector.

**Contrastes pareados causales sobre los programas congelados:**
- Barajar F4: caída E100 entre **39,65 y 55,66 puntos** según réplica, IC95 inferior >0 en las cuatro.
- Desactivar XOR: caída **77,34 a 100 puntos**; IC95 inferior >0 en las cuatro.
- E100 frente a tabla aditiva B: ventaja **+40,63 a +52,34pp**, cuatro IC95 inferiores >0.
- E100 frente E0, noX y shuffleX con el mismo modelo: ventajas entre **+35,16 y +48,05pp**, todos IC95 inferiores >0 en las cuatro réplicas.
- E50 igual a E100 en las cuatro réplicas, pero debido a observaciones suficientes para identificar efectos deterministas con ese presupuesto. No prueba robustez con pocas observaciones ni omisión dependiente del resultado.

## 5. Gates científicos preregistrados

| Gate | Réplicas PASS | Global |
|---|---:|---|
| G0: old DEV >=90% cada contexto, 8960 filas y SHA preflight | **3/4** | PASS |
| G1: CONFIRM >=80%, CI inferior >70%, estratos >=70%, SRX >=80% | **3/4** | PASS; r2 S_after FAIL |
| G2: causalidad F4 shuffle y X disabled | **4/4** | PASS |
| G3: ventaja E100 sobre recompensa aditiva B | **4/4** | PASS |
| G4: E100 supera E0, E_noX y E_shuffleX manteniendo learner | **4/4** | PASS |
| G5: E50 >=80% (diagnóstico) | 4/4 | No requerido |
| G6: I_privileged >=80% (diagnóstico) | 4/4 | No requerido |

**F10PC_CONFIRM_PASS** conforme a G0∧G1∧G2∧G3∧G4, con mínimo 3/4 por gate. No retocar r2 ni realizar nueva confirmación sobre estas semillas.

## 6. Interpretación y decisión de producto

Resultado confirmado: **con un contrato explícito de salida P/S/R/X y una abstracción de trabajo diseñados por investigadores, una memoria de efectos puede aprender las transiciones de esas respuestas y componer una cadena nunca ejecutada, sin leer estados internos privilegiados ni entrenar en la combinación nueva**. Las comparaciones E100 versus E0/noX/shuffleX aíslan razonablemente la contribución de la información observable porque comparten modelo, búsquedas, experiencias y presupuesto; control I usa otro algoritmo, no aislarlo como solo información.

Limitaciones que permanecen: mismos mundos sintéticos, bits deterministas, presupuesto alto de eventos, funciones/semántica externa dadas, búsqueda exhaustiva 85 secuencias, READ hipotético binario, feedback exacto, sin nuevas herramientas ni texto libre. **No se han probado llamadas de herramientas reales, agentes con acciones online, recuperación de memoria MCP del usuario, privacidad multiusuario, latencia ni aumento de coste por tarea**. No atribuir al modelo la invención de SRX como nueva primitiva ni usar el 97,17% como KPI de producto.

**Decisión: PASS científico, NO-GO para integrar ahora el controlador en INFINITO productivo.** La siguiente prueba que ofrece utilidad de producto es un piloto **shadow/read-only** con ejecuciones reales de herramientas sobre datos completamente sintéticos y resultados públicos de APIs, manteniendo un baseline INFINITO memoria y RAG si corresponde, presupuesto comparable, y seguridad/abstención/errores. Sin leer MCP personal, sin permisos de escritura/olvido, sin bóveda, sin nuevos servicios ni costes no autorizados. Controlador en el laboratorio, PR #72 DRAFT; el usuario decide merges.
