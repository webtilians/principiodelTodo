# INFINITO F10-S — resultados originales: temporalidad entre agentes MCP

**4 noviembre 2026. F10S_COHERENCE_FAIL. Integridad y seguridad PASS 4/4.** Prerregistro prospectivo commit 30cc583dd2c29d6b08bcd42c4cf84c5bd89d1dd9. Unica ejecución original [GitHub Actions 41920653029](https://github.com/webtilians/principiodelTodo/actions/runs/41920653029), cuatro réplicas y auditor independiente SUCCESS. Producto auténtico en SHA 2e563ddb194cb3a0193b77a22cc97575a8baba73, sin mutar master ni MCP personal.

Dos procesos MCP stdio independientes A/B compartían una SQLite temporal sintética por réplica. Tercer/cuarto proceso fresh verificaron reinicio y borrado. 96 casos totalmente nuevos: cuatro réplicas × 6 familias ×4 casos, semillas disjuntas 32100000+10000*r+100*f+j. Reservados 32300000+ sin abrir.

| Prueba | Aciertos |
|---|---:|
| Escritura y lectura cruzada A→B | 16/16 |
| Sustitución secuencial de dato exclusivo | 16/16 |
| Olvido selectivo con purga real y distractor intacto | 16/16 |
| Nuevos procesos no resucitan un dato olvidado | 16/16 |
| Contexto local antiguo y nuevas consultas MCP coherentes | 16/16 |
| **Dos escrituras concurrentes de dato exclusivo** | **0/16** |

80/96 global descriptivo, G0 integridad 4/4; G1 4/4; G2 4/4; G3 4/4; G4 4/4; **G5 0/4**; hard gate seguridad 4/4; consecuencia F10S_COHERENCE_FAIL sin ajustes posteriores.

**Evidencia real del bug**: las dos escrituras de hechos independientes A/B sí se guardaron y permanecieron disponibles. Pero, bajo concurrencia, ambos procesos recibieron ACK Remembered de valores diferentes del atributo exclusivo color favorito, por ejemplo granate32100400 y dorado32100400. En el trace original, B recupera simultáneamente ambos recuerdos como activos. Es un fallo de coherencia entre procesos, no una memoria previamente cacheada.

**Mecanismo probable, todavía por probar causalmente**: SQLiteCognitiveMemoryStore.add selecciona la versión anterior, luego la supersede e inserta la nueva bajo una transacción diferida y un lock threading que no coordina distintos procesos. Dos procesos pueden seleccionar el mismo estado previo antes de bloquear escritor y completar ambas inserciones como ACTIVE. Candidato de corrección: BEGIN IMMEDIATE al entrar al tramo SELECT+UPDATE+INSERT para serializar escritores también entre procesos. Solo confirmar en una nueva rama y nuevas semillas, manteniendo el fallo F10-S original.

El servidor no puede borrar retroactivamente contexto ya emitido a un LLM, pero nuevas consultas MCP deben obedecer borrado; esto pasó. No se probó transporte HTTPS remoto ni gestión de caché de agentes no controlados. PR #77 DRAFT, producto/master inalterados.