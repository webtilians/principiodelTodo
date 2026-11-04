# F10-R — primer y único lanzamiento del ensayo preregistrado con MCP auténtico

2026-11-04. Protocolo prospectivo fijado antes de implementar: commit f7c6425ce6efebff4756740fc613908c37e8774e. Código del producto real fijado al master histórico f6bd6239bfc1734c04c233a5cf8cef050c6d54c2. PR #75 DRAFT. Ninguna escritura/lectura en memoria MCP personal.

Esta marca dispara un ensayo original: 4 réplicas, 100 llamadas de entrenamiento por réplica a remember/recall/list_memories/forget auténticos, 48 tareas OOD de dos subobjetivos por réplica y TRES brazos distintos con BDs SQLite temporales (política documental competente, política aprendida sobre resultados MCP y control de etiquetas permutadas). El servidor y los 4 tools proceden del producto real, no de una imitación del laboratorio.

Congelar por SHA estadísticas de las políticas aprendidas ANTES de abrir OOD. Puntuación principal: mejora incremental frente a la política documentada, con gates estrictos y controles de borrado irreversibles, actualización, preservación de hechos ajenos y costes de llamadas. Un empate no es PASS científico. CI exitoso no equivale a ganancia cognitiva. Confirmación 31300000+ reservada. Sin cambiar modelo/umbrales después de OOD, sin merge/despliegue.

## Reproducción técnica después de fallo de integridad (sin cambiar ciencia)

El primer run [41912981283](https://github.com/webtilians/principiodelTodo/actions/runs/41912981283) completó las llamadas MCP de TRAIN y OOD, pero FALLÓ en el control post-OOD de SHA porque los ficheros OOD LEARNED.json y SHUFFLED.json utilizaron los mismos nombres que los snapshots de entrenamiento congelados, sobrescribiéndolos. **Resultado: INCONCLUSIVE técnico** para ese run; se conservarán sus logs. Los mundos OOD 31400000+ ya se habían abierto: esta repetición no debe presentarse como una segunda confirmación independiente ni como una muestra ciega.

Se cambiaron exclusivamente las rutas de salida de los resultados OOD a eval_LEARNED.json, eval_SHUFFLED.json y eval_DOCS.json, además de sus referencias en auditor y CI; commits dabecb432cccfa814b4c5c5c33d0b2bec2b8c18d, dc8d12799b5c6132d744c7d23f0a5747b5b4fd22 y bb11b51f9e53e4451fb96c31e1e60adfa36ce1bd. Sin cambios a casos, semillas, controles, selección, entrenamiento, herramientas, fórmula, umbrales ni políticas. Se repite EXACTAMENTE el primer experimento para obtener una **auditoría íntegra de la misma muestra**. El FAIL/PASS científico no se rescatará mediante ajustes. Los mundos reservados 31300000+ siguen cerrados.
