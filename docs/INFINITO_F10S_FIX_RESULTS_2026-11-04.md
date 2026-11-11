# INFINITO F10-S-FIX — informe causal sobre sustitución concurrente de recuerdos exclusivos

**4 de noviembre de 2026. F10S_FIX_VALIDATED** bajo los criterios prospectivos propios de esta fase, que se definieron *después* de descubrir el fallo original F10-S pero *antes* de escribir el parche y evaluar 32_500_000+. No se trata de confirmación ciega del estudio F10-S original. No se ha fusionado ni desplegado en el servicio de memoria del usuario.

## 1. Hallazgo, hipótesis y única intervención

F10-S original PR #77 encontró, mediante 2 MCP stdio distintos sobre una SQLite sintética compartida, que **16/16 escrituras simultáneas de dos valores para `mi color favorito` dejaban ambos valores activos**, aunque dos memorias NO exclusivas escritas a la vez sí se conservaban. Estado original `F10S_COHERENCE_FAIL`: 80/96 casos en total, falla solo concurrencia. [Run original 41920653029](https://github.com/webtilians/principiodelTodo/actions/runs/41920653029); [informe F10-S](https://github.com/webtilians/principiodelTodo/blob/experiment/f10s-cross-agent-temporal-coherence-20261104/docs/INFINITO_F10S_RESULTS_2026-11-04.md).

Explicación causal propuesta: `SQLiteCognitiveMemoryStore.add` ejecuta SELECT del hecho exclusivo anterior antes de su primer UPDATE/INSERT, dentro de `with self._lock, self._conn`. El `RLock` es por instancia/proceso, mientras que `sqlite3.Connection` en modo diferido no toma el bloqueo de escritor durante SELECT; dos procesos pueden decidir ambos que no existe una versión que sustituir e insertar dos registros ACTIVE. No existe un índice único parcial por predicado/estado que lo impida.

**Tratamiento único:** insertar `self._conn.execute("BEGIN IMMEDIATE")` al entrar al bloque de escritura de `SQLiteCognitiveMemoryStore.add`, **antes de `_tick`, SELECT, UPDATE e INSERT**, con comentario explicativo. De ese modo ambos procesos se serializan por bloqueo de escritura SQLite al comprobar el valor vigente. Este cambio no altera contratos MCP, algoritmos de búsqueda, contenidos, filtrado, semántica de olvido ni schema. Commit del parche **d3a376580b5f9e19b79750013b41e3ca9df4e710**.

## 2. Protección metodológica

Prerregistro F10-S-FIX **733000c019d4712b1c1e63b56210af5c96623ff9**, publicado antes de implementar código y generar casos nuevos. Estudio **causal posterior al descubrimiento**. Comparación pareada entre producto original congelado commit `2e563ddb194cb3a0193b77a22cc97575a8baba73` y producto con un solo patch BEGIN IMMEDIATE. Mismo test MCP real y semillas **nuevas** `32_500_000+10000*r+100*f+j`, con 4 réplicas ×6 familias ×4 casos = **96 casos por brazo, 192 casos totales**. Dos procesos MCP reales A/B por brazo sobre su propia SQLite temporal independiente. No usar memoria personal, bóveda, servicio remoto HTTPS ni datos de usuario. Reservadas nuevas semillas confirmatorias `32_700_000+`.

Verificación en CI: comparación exacta del archivo de producto con master que demuestra que solo se añadió el bloque `BEGIN IMMEDIATE` y comentarios, y comprobación de que no se modificó después el código durante el estudio. Auditor independiente que **no importa MCP ni SQLite** recomputa el mismo conjunto de casos, SHA, gates y controles de seguridad desde artefactos. [Run del seguimiento causal 41922000727](https://github.com/webtilians/principiodelTodo/actions/runs/41922000727) con cuatro jobs original/tratamiento y auditor independiente todos SUCCESS.

## 3. Resultados nuevos

| Prueba | Versión original | BEGIN IMMEDIATE |
|---|---:|---:|
| Dos agentes escriben hechos independientes | 16/16 (incluido en concurrencia) | 16/16 |
| **Dos agentes actualizan simultáneamente hecho exclusivo** | **0/16** | **16/16** |
| Visibilidad de A hacia B | 16/16 | 16/16 |
| Sustitución secuencial | 16/16 | 16/16 |
| Borrado selectivo sin daño a otras memorias | 16/16 | 16/16 |
| Persistencia tras abrir nuevos procesos | 16/16 | 16/16 |
| El contexto local antiguo sigue siendo antiguo, consulta MCP nueva es vigente | 16/16 | 16/16 |
| **Resultado global (6 familias)** | **80/96** | **96/96** |

**No son 16 casos independientes de estrés arbitrario**; son 4 réplicas ×4 variantes sintéticas de un mismo tipo de carrera. El nuevo conjunto base mantiene los mismos mecanismos del test original y la intervención fue seleccionada a partir del fallo, por lo que no hay confirmación verdaderamente ajena ni prueba de carga de producción.

**Puertas preregistradas**: G0 casos aislados y emparejados 4/4; G1 concurrencia corregida 4/4; G2 sin regresiones en las cinco familias restantes 4/4; G3 ventaja causal respecto original en nuevo conjunto 4/4; G4 seguridad 4/4. `F10S_FIX_VALIDATED`.

## 4. Qué se ha solucionado y qué no

Este ensayo **sí** comprueba que adquirir el bloqueo SQLite de escritor antes de leer el hecho exclusivo evita, en las intercalaciones ensayadas, que ambas escrituras terminen como dos valores activos. La comparación control/tratamiento en las mismas condiciones, con un único cambio de código, favorece la hipótesis de la carrera transaccional.

No demuestra garantías de serialización ante NFS/múltiples hosts, una base remota fuera de este proceso, bloqueos de larga duración, conflicto multiusuario, fallos del servidor a mitad de operación, expiración de caches del LLM ni que un contexto externo borre recuerdos ya vistos. La corrección añade posible espera del writer lock y requiere prueba de rendimiento con concurrencia alta. Nada se fusiona automáticamente.

**Próximo paso previo a integrar:** conservar test de regresión permanente con dos procesos MCP sobre la misma SQLite, ejecutar suite de actualización/olvido, y pedir revisión de PR #78. Luego una confirmación independiente en semillas selladas con mayor concurrencia; sin rescatar fallos por retuning ni desplegar a la memoria conectada del usuario desde este laboratorio.