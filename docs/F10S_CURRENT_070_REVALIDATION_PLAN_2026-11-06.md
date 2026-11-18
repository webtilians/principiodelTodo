# F10-S-FIX sobre INFINITO v0.7.0 — Plan de revalidación previo al parche

**Fecha:** 2026-11-06. **Commit base congelado:** `129e70189a060136b3942f8dc251575f8a0b6299` (master tras merge #108 de página de memoria). Este documento se confirma antes de reintroducir la corrección F10-S-FIX en esta rama. **No contiene autorización de fusionar, publicar ni desplegar**.

## Contexto y diagnóstico
F10-S original reveló carrera entre dos procesos MCP stdio y una SQLite compartida: 16/16 escrituras simultáneas de valores exclusivos terminaron con dos valores activos. La corrección experimental PR #78, `BEGIN IMMEDIATE` dentro de `SQLiteCognitiveMemoryStore.add` antes de consultar el estado previo, mejoró 0/16→16/16 en la familia de concurrencia y 80/96→96/96 global bajo nuevos seeds (cuatro réplicas). Se aplicará **solo el mecanismo mínimo**, no los experimentos antiguos.

En `master` actual el método `add` entra con `with self._lock, self._conn` y consulta hechos exclusivos antes de que exista una transacción de escritura adquirida. Bloqueo de hilo local no serializa dos procesos. El servicio HTTP cifrado `SealedMemories` usa un lock **por instancia dentro del proceso** para leer/reescribir el fichero completo; esta corrección del núcleo SQLite **NO demuestra** atomicidad entre varios procesos HTTP. La distinción es crítica.

## Contratos antes de ejecutar
- **G0 integridad:** rama parte del SHA base exacto, diff del producto exclusivamente un `BEGIN IMMEDIATE` con comentario breve, sin cambios en endpoints, cliente, vault, secretos, esquema, servidor HTTP, permisos o memoria de usuario. Se conservará el PR #78 sin editar ni fusionar.
- **G1 concurrencia real local:** dos **MCP stdio reales y distintos** escriben valores exclusivos concurrentes a la **misma SQLite temporal**; en cada ronda, exactamente uno de los valores en disputa figura activo y todos los agentes reciben confirmación correcta. Ejecutar al menos 12 rondas según regresión independiente de PR #78 y un stress adicional de 4 procesos sobre memoria temporal, con cambios de valores repetidos.
- **G2 regresiones de producto:** correr todo `tests/test_*v3.py` del master actual, especialmente persistencia, recall, olvido, aislamiento de proyecto, servidor MCP, autenticación, cifrado, backup/restore, página web de memoria y bóveda. Instalar `.[mcp]` + pytest y librerías de pruebas adecuadas; las pruebas omitidas por falta de dependencias deben mostrarse, no contarse como PASS.
- **G3 semántica:** actualizaciones exclusivas de varios hechos, duplicados idempotentes, otros recuerdos preservados, cadena de supersession, consulta en nueva conexión, olvidar y volver a guardar. Ningún recuerdo purgado puede reaparecer en la base probada.
- **G4 límites operativos:** no atribuir al parche corrección multiworker HTTP cifrado, ni verificar secreto/backup en la máquina del usuario. Si alguna prueba HTTP multiworker demuestra pérdida de actualización, tratar como limitación BLOQUEANTE para afirmar soporte a múltiples workers. El PR permanece DRAFT, sin automerge.
- **G5 reproducibilidad:** GitHub Actions con logs, versiones de Python/SDK y artefactos; publicar resultados y números reales, nunca declarar GO científico por un workflow verde si hay skips, fallos, regresión o tests no ejecutados.

## Condiciones para recomendar fusión posterior (no realizarla aquí)
Todos G0–G3 sin errores en nuevos tests y regresión, revisar dependencias y cobertura; decidir de forma explícita si el servidor remoto usa un único worker. Probar migración y backup/restore con ficheros **sintéticos**. El desarrollador decidirá cuándo fusionar. No usar ni abrir ninguna memoria de usuario, vault o datos reales, y no desplegar desde Actions.

**Estado al registrar el plan: no hay patch ni runs de esta nueva rama.**
