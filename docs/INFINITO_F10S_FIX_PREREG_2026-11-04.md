# INFINITO F10-S-FIX — causalidad de bloqueo SQLite antes de sustitución exclusiva

**PRERREGISTRO PROSPECTIVO 2026-11-04 ANTES de implementar la corrección, las nuevas semillas o ejecutar CI.** Experimento F10-S original separado y congelado, commit de resultados 97c4f3821883f8ba255886d85eb8c8ea0b6b240a, FAIL 16/16 de concurrencia, producto master original SHA 2e563ddb194cb3a0193b77a22cc97575a8baba73. Este estudio F10-S-FIX es **hipótesis posterior al hallazgo**, NO confirmación ciega de F10-S.

## Hipótesis causal

`SQLiteCognitiveMemoryStore.add` usa un lock `threading.RLock` por instancia y `with sqlite3.Connection` que inicia transacción diferida al primer INSERT/UPDATE; la comprobación SELECT de memoria exclusiva previa puede ocurrir en dos procesos antes de su primera escritura. Ambos devuelven ACK y se observan dos versiones activas. **Único cambio de tratamiento preregistrado:** ejecutar `BEGIN IMMEDIATE` al inicio del bloque `with self._lock, self._conn` en `add`, ANTES de `_tick`, SELECT de equivalencia y SELECT de hecho exclusivo; no modificar filtros, schemas, indexaciones, política de olvido, MCP, contratos, hecho exclusivo ni lógica de búsqueda. Debe serializar el check-and-update entre diferentes procesos, sin depender de locks Python.

## Comparación y nuevos casos

Control `BASELINE`: producto original congelado master SHA 2e563ddb194cb3a0193b77a22cc97575a8baba73. Tratamiento `BEGIN_IMMEDIATE`: mismo código con el único cambio descrito. Ambos ejecutan por proceso real MCP stdio, dos servidores independientes A/B apuntando a una base SQLite temporal independiente por brazo y por réplica. Mismos mensajes y mismos casos. Harness F10-S original copiado sin alterar la lógica de pruebas, salvo BASE semilla y ruta explícita `PYTHONPATH` del producto para elegir la variante de código. No cambiar criterios después de OOD.

Semillas **completamente nuevas** `32_500_000+10000*r+100*f+j`, r=0..3, f=0..5, j=0..3. Cuatro réplicas, 6 familias ×4 casos = **24 casos/replica/brazo, 192 casos emparejados en total**. Confirmación adicional `32_700_000+` sellada. El piloto F10-S original `32_100_000+` y confirmación `32_300_000+` no se reutilizan. Los dos brazos ejecutan secuencialmente la batería por réplica pero con bases limpias distintas; concurrencia se genera dentro de cada caso mediante dos procesos reales y `asyncio.gather`.

Revisar resultados por familia y traza MCP. SQL auditor externo por herramienta del test observa ACTIVE y purga; modelos de lenguaje y servicio HTTP remoto fuera de alcance. Ningún dato personal/vault ni MCP conectado al usuario. Grabar SHA del script y ambos códigos fuente; no fusionar automáticamente.

## Gates preregistrados

- **G0 integridad (4/4)**: dos ejecutables MCP con rutas de código origen distintas, bases temporales distintas, misma batería de 24 casos por réplica/brazo, hashes y semillas iguales y disjuntas entre réplicas, producto original inmutable y patch de un solo archivo/una sola inserción `BEGIN IMMEDIATE`.
- **G1 eficacia corregida (>=3/4)**: tratamiento completa 4/4 pruebas de concurrencia por réplica y 16/16 agregadas, sin dos versiones exclusivas activas ni pérdidas de notas independientes.
- **G2 no regresión (>=3/4)**: tratamiento mantiene 4/4 casos de cada una de las otras cinco familias por réplica, incluyendo borrados irreversibles, reinicios y revalidación.
- **G3 mejora causal (>=3/4)**: en nuevos casos, tratamiento supera al baseline en al menos **2/4** casos de concurrencia por réplica; el baseline debe reproducir algún fallo de concurrencia para atribuir el efecto. Reportar diferencias pareadas globales y cada réplica, NO ocultar que el hallazgo original motivó la hipótesis.
- **G4 seguridad (4/4)**: 0 pérdida de distractores en tratamiento, 0 reapariciones en datos SQL, 0 datos de usuario tocados; no incluir tests de red, bóveda ni operación de producción.

**SCIENCE STATUS `F10S_FIX_VALIDATED`** solo si G0/G4=4/4 y G1,G2,G3>=3/4. Si pruebas íntegras incumplen gates, `F10S_FIX_FAIL`; si error de infraestructura/artefactos, `F10S_FIX_INCONCLUSIVE`. Cualquier error se conserva; no modificar seeds/código después de ver OOD. Propuesta de merge queda reservada a revisión explícita posterior; no actualizar el servidor real del usuario.

## Interpretación

Un PASS demuestra que una transacción SQLite inmediata, en esta familia de cargas concurrentes y estos procesos, evita el fallo reproducido de versiones exclusivas activas; no demuestra linearizabilidad universal bajo cargas distribuidas, multiproceso remoto, timeouts, fallos de máquina, bloqueo de archivos NFS ni sincronización de contextos LLM ya entregados. Si PASS, sería apropiado añadir pruebas de integración concurrente en CI principal ANTES de fusionar código de producto y ejecutar una confirmación separada sobre semillas reservadas.