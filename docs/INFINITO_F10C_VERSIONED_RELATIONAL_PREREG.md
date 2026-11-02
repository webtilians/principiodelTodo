# INFINITO F10-C — Resolver precedencia entre evidencias públicas contradictorias
**Protocolo prospectivo congelado 2026-11-02, antes de implementar el generador o evaluar semillas nuevas.** Esta prueba no reutiliza TRAIN/DEV F10-B y no abre su confirmación 13_300_000+.

## Pregunta
¿Un comparador relacional entrenable de registros de la misma clave, usando solamente credencial, instante, valor y fuente públicamente observados, aprende a recuperar la **última versión acreditada** mejor que la atención unaria de F10-B? ¿Continúa READ teniendo utilidad causal cuando existen varios informes contradictorios?

## Nueva familia sintética F10-C, compatible dimensionalmente con los controles congelados
Se mantienen **24 claves, 60 informes, 29 atributos por evento, GRU64 y log observacional de 64 slots**, para permitir evaluar *sin reentrenar* los dos checkpoints congelados de F10-B. No se conserva la distribución de informes de F10-B.

En cada mundo una sola secuencia pública de 60 informes, compartida **bit a bit entre cuatro consultas que llegan al final**, sin pistas sobre qué clave se preguntará. Las 24 claves empiezan con un informe firmado, valores y fuentes A/B aleatorios. Ocho claves elegidas antes de generar informes se dividen en 4 estratos de 2 claves cada uno: (último firmado/no firmado) × (estable/revisado). A cada una se añaden 4 informes posteriores (al menos dos contradictorios); se añaden cuatro distractores a claves distintas de las consultables, dando `24 + 8×4 + 4 = 60`. Los 36 informes posteriores se intercalan al azar, respetando el orden interno de cada clave; **no se ordenan por calidad o etiqueta del resultado**. Las cuatro consultas finales seleccionan una de las dos claves de cada estrato con semilla fijada.

Tratamientos de secuencia, con `b` = valor firmado inicial:
- firmado/estable: U(!b), U(!b), S(b), S(b)
- no firmado/estable: U(!b), S(b), U(!b), U(!b)
- firmado/revisado: S(!b), U(b), S(!b), S(!b)
- no firmado/revisado: S(!b), U(b), U(b), U(b).

`S` es registro con credencial observable =1, `U` credencial observable =0; fuente A/B independiente, tiempo derivado del orden de observación. La verdad de cada consulta es la última afirmación **firmada** de su clave, reconstruible exclusivamente de los informes públicos. *El bit de firma es una premisa sintética de autenticación*, no criptografía real, ni un bit que revele directamente la respuesta. Los distractores no cambian el valor de las claves consultadas. El agente no recibe la pertenencia a estrato, el valor correcto, la revisión ni el índice de registro ganador.

**No hay necesidad lógica absoluta de F4**: el GRU64 ve toda la historia. La posible utilidad es empírica y se debe demostrar frente a SKIP y a un GRU-only competitivo.

## Diseño de modelos y entrenamiento (no retocar tras DEV)
**Cuatro réplicas independientes**, cada una entrenada desde pesos nuevos y semillas de inicialización fijadas `171000+index`. TRAIN `14_100_000+10000*index+j`, `j=0..511`, 512 mundos por réplica y 4 consultas/mundo. DEV `14_200_000+10000*index+j`, `j=0..63`, 64 mundos por réplica y 4 consultas/mundo. CONFIRM `14_300_000+` reservada y **sin ejecutor**. Ningún mundo de F10-B (13_100_000, 13_200_000, 13_300_000+) se consulta durante esta fase.

Tres brazos nuevos sobre **los mismos 2048 casos TRAIN/actor**:
1. `relational_read`: GRU64 + F4 FIFO de eventos + selector de registros con coincidencia de clave pública estructural y comparación **entrenable por pares** de credencial, orden temporal, contenido y fuente. Ni prioridad `signed` ni prioridad temporal están fijadas como regla; solo el *emparejamiento por clave* está codificado explícitamente. Mismo tipo de salida binaria.
2. `unary_read_control`: arquitectura original `EvidenceActor` F10-B (GRU64 + F4 y puntuación unaria aprendida), **reentrenada únicamente en los nuevos TRAIN F10-C** para aislar cambio de distribución; no es el checkpoint congelado.
3. `recurrent_only_control`: misma arquitectura `EvidenceActor` pero recibe log=0 durante TRAIN y DEV, también reentrenada en los mismos mundos.

Las 3 reciben las mismas observaciones, objetivos, orden de ejemplos, **12 épocas**, minibatch64, Adam lr0.002, BCEWithLogitsLoss, clip1.0, sin tuning. Las arquitecturas no tienen exactamente el mismo número de parámetros por el comparador de pares: se reportará la diferencia y **no** se atribuirá causalmente toda la mejora a la operación pairwise si no hay control de tamaño.

Comparadores adicionales **congelados**: `memory_available` F10-B y `recurrent_only` F10-B, checkpoints del [run #41443064754](https://github.com/webtilians/principiodelTodo/actions/runs/41443064754) con SHA comprobados frente a manifests originales. **No se entrena ninguna nueva cabeza READ**; el GRU ve la secuencia pública completa.

## DEV: inferencia y efectos
Para cada consulta en DEV, el *mismo actor relational_read*, su historial público y su estado GRU, se evalúa en:
- `READ_clean` log auténtico (todos los registros);
- `SKIP` log=0, sin modificar historia ni pesos;
- `READ_stale` conserva registro firmado inicial y retira versiones firmadas siguientes de la clave (intervención observer-only, no fenómeno natural);
- `READ_forged` convierte en firmado el último informe no firmado de la clave, sin tocar el historial (ataque sintético fuera de distribución).

Guardar cada caso sin seleccionar por acierto, `D=correct(READ)-correct(SKIP) ∈ {-1,0,1}` y cada estrato. Comparar exactitud con `unary_read_control`, `recurrent_only_control` y los dos F10-B originales congelados en **estos mismos mundos NUEVOS**. Reportar intervalo bootstrap del efecto READ−SKIP agrupado por mundo con 2000 remuestreos y los 4 estratos completos.

## Gates preregistrados DEV exploratorios
- Integridad: fuente y etiqueta derivadas solo de informes públicos; los cuatro casos de un mundo comparten historia exacta; 5 informes para cada clave consultada; no se ocultan D-/D0; los checkpoints F10-B originales coinciden con hashes; no se optimiza con DEV.
- Procedencia temporal: `relational_read` >=**80%** en cada uno de los cuatro estratos en >=**3 de 4** réplicas.
- Transferencia/comparador: en >=**3/4** réplicas, `relational_read` supera `unary_read_control` en al menos **+0.08 de precisión** en el estrato especialmente difícil `unsigned|revised` y en >=**+0.05 de precisión** global; comparar y reportar por separado contra F10-B congelado, sin exigir que el checkpoint F10-B transferido supere su propia distribución anterior.
- Utilidad de memoria: >=**3/4** réplicas tienen READ−SKIP >=**+0.10** global y límite inferior IC95 agrupado por mundo >0, y READ superando al `recurrent_only_control` por >=**+0.05** global; además **TODOS los 16 estratos actor×firma×revisión** tienen >=**8 D+ y 8 D<=0** sobre 64 mundos. Esto es soporte para *futuro* lector, **no** prueba de que el actor haya aprendido cuándo consultar.
- `F10C_DEV_PASS` **solo** si se satisfacen *todos* esos criterios. En otro caso `F10C_DEV_FAIL`, con detalle por criterio y estrato. Aunque PASS sería señal en tarea sintética, no confirmación independiente ni generalización LLM/INFINITO.

## STOP y honestidad
Si falla cualquier puerta: **NO** hacer tuning después de DEV sobre esas semillas, NO abrir `14_300_000+`, NO entrenar cabeza READ, NO presentar otra ejecución como confirmación. Una nueva propuesta requerirá nuevos mundos y otro protocolo. La operación pairwise, el matching estructural por clave y el token firmado son sesgos de diseño explícitos: no afirmar que el agente descubre el concepto general de verdad o seguridad criptográfica.
