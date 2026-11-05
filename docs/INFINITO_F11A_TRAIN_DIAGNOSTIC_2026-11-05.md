# F11-A — primer entrenamiento TRAIN-only y diagnóstico de saturación

**Fecha:** 2026-11-05  
**Estado técnico:** cuatro jobs de entrenamiento + contratos SUCCESS.  
**Estado científico:** TRAIN_DIAGNOSTIC_ONLY; ningún gate F11-A comprobado. No hay DEV ni CONFIRM, y no se debe presentar esto como mejora cognitiva.

## Procedencia exacta

- Prerregistro ya fusionado antes de cualquier implementación: PR #86, commit \`8f8a6dcf40c9f75b73b5a40207df3876c542a5b1\`.
- Código de TRAIN: rama \`experiment/f11a-train-20261105\`, primer launch commit \`ee1acb020ce578d188d3709f2c29326af8d29f08\`.
- GitHub Actions [run #42032871604](https://github.com/webtilians/principiodelTodo/actions/runs/42032871604): contratos SUCCESS; TRAIN r0..r3 SUCCESS; cuatro artefactos \`f11a-train-replica-0..3\` con \`checkpoint.json\`, \`train_summary.json\` y \`train_events.jsonl.gz\`.
- Reproduce \`python -m unittest tests.test_f11a_train -v\` y \`python -m observatory.f11a_train --replica N --worlds 512 --episodes 30 --probes 2 --out PATH\`.
- Los artefactos de GitHub Actions tienen caducidad; sus hashes de eventos figuran en los cuatro logs y sus \`train_summary.json\`.

## Qué se entrenó realmente

Cuatro réplicas independientes, 512 mundos de TRAIN/r, 30 episodios por mundo: **61.440 episodios** de exploración, **344.380 acciones** observadas, **14.188 episodios de exploración exitosos** (23,1%). A1 (memoria de experiencia), A2 (modelo de transiciones) y A3 (opciones reutilizables extraídas de experiencias exitosas) recibieron los mismos eventos de exploración. A0 es una política reactiva aleatoria y A4 utiliza el mecanismo del simulador como control privilegiado exclusivamente en probes, no en TRAIN. A3 extrajo 6.177 opciones de trayectoria exitosas entre cuatro réplicas.

La palabra *entrenamiento* aquí significa inducción de un modelo tabular de efectos y de opciones a partir de experiencias, **no** ajuste por gradiente de una red neuronal, ni entrenamiento de un LLM, ni política que inventa nuevas acciones. Las cuatro réplicas usan un generador sintético simple y una gramática de acciones fija. Se implementaron guardas sobre los estados observados para abortar una opción cuando un efecto no se produjo como se esperaba.

## Probes sobre los MISMOS mundos de TRAIN

Dos probes por mundo = 1.024 episodios por réplica y brazo. Esta medición está contaminada por la familiaridad de entrenamiento y no equivale a generalización OOD.

| Réplica | A0 reactivo | A1 episodio | A2 efectos | A3 opciones | A4 privilegiado |
|---|---:|---:|---:|---:|---:|
| r0 | 219/1024 | 1024/1024 | 1024/1024 | 1024/1024 | 1023/1024 |
| r1 | 234/1024 | 1024/1024 | 1024/1024 | 1024/1024 | 1024/1024 |
| r2 | 248/1024 | 1024/1024 | 1024/1024 | 1023/1024 | 1024/1024 |
| r3 | 203/1024 | 1023/1024 | 1023/1024 | 1024/1024 | 1024/1024 |
| **TOTAL (descriptivo, NO inferencial)** | **904/4096 (22,1%)** | **4095/4096 (99,98%)** | **4095/4096 (99,98%)** | **4095/4096 (99,98%)** | **4095/4096 (99,98%)** |

- Uso de opción A3 reportado en los 4096 probes; 463 abortos de guarda. Esta cifra **no demuestra utilidad causal**, pues A2 también obtuvo 4095/4096.
- El control A4 es un **planificador heurístico con acceso privilegiado a los roles de acción**, no una demostración de óptimo matemático.
- No se realizaron los contrastes pareados con intervalos de confianza de los gates preregistrados (reservados para CONFIRM).
- **Resultado diagnóstico relevante: saturación del entorno.** El diferencial A3–A2 es cero en TRAIN.

## Inadecuaciones de este primer prototipo, ANTES de DEV

1. Dinámica excesivamente sencilla: las acciones de carga, cambio y entrega tienen efectos inmediatos con prerrequisitos, pero todavía no existe una verdadera variable temporal oculta ni una penalización de recurso suficientemente rica. No es la familia de tareas difícil contemplada en el protocolo.
2. El control A1 actual agrega puntuaciones por estado/acción y, por tanto, se comporta como una política aprendida simple, **no como una recuperación episódica/RAG competente sin reglas persistentes**. Se debe sustituir el control A1 antes de cualquier comparación científicamente válida.
3. A3 aprende opciones de demostraciones propias exitosas del colector compartido. Falta un controlador completamente cerrado que actualice y descarte políticas con resultados de **sus propias** decisiones al cambiar las reglas, y un control de ablación real de opciones.
4. Faltan operaciones de adaptación, familias composicionales disjuntas, permutaciones semánticas de acciones, presupuestos de consulta comparables y auditor independiente de G0/G5.
5. Los mundos y episodios de los probes se derivan de TRAIN; no extrapolar el 99,98% a datos no vistos, MCP, herramientas reales o producto.

## Decisión metodológica

**NO-GO para DEV y CONFIRM con este harness tal como está.** Sí se ha probado que el pipeline TRAIN, el registro de experiencias, la inducción de opciones, las guardas y la persistencia de checkpoints se ejecutan en cuatro réplicas.

El siguiente trabajo debe hacerse **antes de mirar DEV**: enriquecer la física con efectos retardados/costes y metas composicionales; corregir A1 como recuperación episódica sin regla agregada; implementar aprendizaje y revisión online de opciones; añadir auditorías de coste, contaminación y ablación; repetir pruebas usando exclusivamente TRAIN y contract tests. Mantener inalterados los umbrales y la partición de CONFIRM preregistrados. Si el diseño sustancial de los mundos cambia respecto al documento inicial, publicar una enmienda prospectiva antes de generar DEV y conservar intactos el informe y las semillas de este piloto.

No se ha modificado el núcleo de producto (\`src/infinito3/\`), el MCP de usuarios, datos personales, secretos ni un servidor remoto. No se ha realizado merge automático.
