# INFINITO F10-R — MCP auténtico, comparación con política documentada

**4 noviembre 2026. ESTADO F10R_INCONCLUSIVE por integridad de semillas.** No se obtuvo una mejora incremental descriptiva. No hay cambios en producto, master ni memoria personal.

## Diseño científico

Protocolo prospectivo en commit f7c6425ce6efebff4756740fc613908c37e8774e. Se ejecutó el MCP AUTÉNTICO del producto master fijado en f6bd6239bfc1734c04c233a5cf8cef050c6d54c2. Se usaron exclusivamente bases SQLite temporales, sintéticas e independientes por política y réplica. Herramientas reales: remember, recall, list_memories y forget.

Se entrenó un selector TABULAR de utilidad de llamadas sobre 100 acciones reales por réplica (cinco tipos de objetivo por cinco acciones por cuatro ejemplos), observando respuestas MCP y estados accesibles mediante list_memories. No es el mismo algoritmo EventEffects de F10-P: la categorización de intención, la asignación de argumentos y la protección de IDs para forget fueron aportadas por investigadores.

Comparador fuerte DOCS: seguir literalmente la documentación del MCP, con remember para escribir/actualizar, recall para consultar, list_memories+forget para borrar y noop para no escribir preguntas. LEARNED: escoger acción según éxito y coste observados, congelado antes del nuevo OOD. SHUFFLED: aprendizaje sobre etiquetas aleatorizadas, diagnóstico. OOD 48 workflows de dos pasos por réplica: recordar/consultar, actualizar/consultar, borrar/consultar otro, recordar/borrar, actualizar/borrar y no guardar preguntas/consultar. Bootstrap pareado prospectivo de 2000 repeticiones. G2 principal exigía mejorar a DOCS al menos 5pp con IC inferior positivo, o ahorrar 15% de tool calls manteniendo exactitud dentro de 2pp. Semillas 31300000+ de confirmación reservadas.

## Resultados observados, descriptivos

| Réplica | DOCS | LEARNED | SHUFFLED |
|---|---:|---:|---:|
| 0 | 48/48 | 48/48 | 0/48 |
| 1 | 48/48 | 48/48 | 0/48 |
| 2 | 48/48 | 48/48 | 0/48 |
| 3 | 48/48 | 48/48 | 8/48 |
| **Total** | **192/192, 100%** | **192/192, 100%** | **8/192, 4,17%** |

**DOCS: 448 llamadas MCP. LEARNED: 448 llamadas MCP. Ventaja en tareas y coste = cero.** El control SHUFFLED empeora fuertemente, pero eso solo indica dependencia de los ejemplos bajo objetivos y acciones suministrados, no invención autónoma de semántica.

Auditor independiente: G0 integridad 0/4, G1 éxito mínimo 4/4, **G2 ventaja frente a DOCS 0/4**, G3 diagnóstico shuffled 4/4, G4 cero borrados ajenos/reapariciones de datos/preguntas guardadas 4/4. Por G0, NO se puede declarar PASS ni FAIL científico definitivo: F10R_INCONCLUSIVE.

## Historial técnico e integridad

Primer run 41912981283: se ejecutó OOD pero se detectó que los ficheros OOD de LEARNED y SHUFFLED sobrescribían accidentalmente los snapshots congelados con los mismos nombres; terminó error de SHA, inconcluso. Se corrigieron solo rutas de salida, sin tocar algoritmo, datos, semillas ni umbrales. Segundo run 41913251736: las cuatro réplicas completaron con SHA conservados, pero la auditoría encontró que TRAIN se generó con un desplazamiento de 10000 por tipo de objetivo que se superpone al desplazamiento de 10000 por réplica. Ejemplo: réplicas distintas pueden compartir identificadores de casos; viola la disjunción preregistrada. La OOD ya estaba abierta, por tanto NO corregimos semillas ni hicimos un tercer intento científico.

Una auditoría SOLO de datos sobre los artefactos existentes, sin invocar el MCP ni reentrenar, terminó SUCCESS técnico en run 41913749694 y devolvió expresamente F10R_INCONCLUSIVE. Su artefacto se llama f10r-independent-inconclusive-forensic-audit. No se abrió la partición de confirmación.

## Significado y recomendación

En este laboratorio el controlador aprendido simplemente igualó una política que ya sabe usar las herramientas documentadas. Eso no demuestra beneficio de integrar un nuevo planificador. **NO-GO de integración** incluso como consecuencia de las cifras descriptivas. Para un experimento nuevo hacen falta casos que de verdad desafíen la semántica documentada: errores impredecibles, cambios de comportamiento, evidencia incompleta o variación de interfaces, y una comparación fuerte con igual presupuesto. No repetir F10-R para rescatar un PASS. PR en borrador, sin cambios en master.