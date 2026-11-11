# F10-S-FIX — revalidación de integración en INFINITO v0.7.0 (master actual)

**Fecha:** 2026-11-06  
**Conclusión:** **G0–G3 PASS** en MCP stdio local con SQLite compartida y regresiones del producto. **Preparado para revisión de código e integración restringida de SQLite, NO para despliegue automático ni afirmaciones de consistencia multiworker HTTPS**.

## 1. Rastreabilidad y alcance
- Base exacta del producto v0.7.0: `129e70189a060136b3942f8dc251575f8a0b6299`, último `master` antes de abrir la rama; ya incluía el merge #108 de la página de memoria.
- Plan prospectivo previo al parche: `docs/F10S_CURRENT_070_REVALIDATION_PLAN_2026-11-06.md`, commit `b0100c7eafb739f2ba2402edfff714cc9152b42f`.
- Parche de producto commit `fe2befe873e3f8fba27087c739056308ebf985dd`: **una línea** `self._conn.execute("BEGIN IMMEDIATE")` y comentario explicativo, antes de cualquier comprobación de hechos `SELECT` o `UPDATE/INSERT` en `SQLiteCognitiveMemoryStore.add`. Verificado en Actions por comparación del archivo completo contra el `master` de partida: cero cambios adicionales de producto.
- Rama separada `fix/f10s-current-070-revalidation-20261106` y **PR #110 DRAFT** contra master; **PR #78 anterior permanece aparcado e intacto**.
- No modifica ni usa memoria personal, servicio HTTPS de producción, bóveda, clientes, permisos, claves, endpoints o repositorios distintos.

## 2. Prueba diferencial decisiva: mismo test con y sin corrección
**Control negativo sin parche:** [GitHub Actions #42240671483](https://github.com/webtilians/principiodelTodo/actions/runs/42240671483), workflow `.github/workflows/f10s-current-070-negative-control.yml`.
1. Checkout de la rama de validación que contiene **el mismo test** que la variante corregida.
2. Reemplazar exclusivamente `src/infinito3/persistent_memory.py` por los bytes exactos del commit `master` congelado `129e7018` utilizando `git show`. Verificado: no contiene BEGIN IMMEDIATE.
3. Lanzar cuatro procesos MCP stdio genuinos, todos con la misma SQLite temporal, `remember` simultáneos de cuatro valores distintos de `favorite_color`.
4. **Falló en la PRIMERA ronda (ronda 0): cuatro valores diferentes quedaron ACTIVOS a la vez.** Traza exacta: `AssertionError: (0, ['Mi color favorito es turmalina0agente3.', 'Mi color favorito es turmalina0agente2.', 'Mi color favorito es turmalina0agente0.', 'Mi color favorito es turmalina0agente1.'])`; `assert 4 == 1`. El job clasifica y EXPECTA ese fallo; su estado SUCCESS significa **defecto de base reproducido**, NO que la rama sin parche sea válida.

**Intervención con parche:** [GitHub Actions #42239794876](https://github.com/webtilians/principiodelTodo/actions/runs/42239794876), workflow `.github/workflows/f10s-current-070-integration.yml`.
- Regresión previa de **dos procesos MCP**: 12 rondas, un único valor exclusivo activo, escritura y lectura correctas.
- Estrés nuevo de **cuatro procesos MCP reales**, cada uno conectado al mismo fichero SQLite: **12 rondas × 4 escrituras simultáneas = 48 herramientas remember**, todos los resultados correctos, **un único valor activo tras cada ronda**, las lecturas MCP concuerdan con SQLite y el estado persiste tras finalizar los procesos.
- Datos de proyectos independientes **Acme** y **Beta** preservados y aislados por filtro de proyecto tras concurrencia.
- Recuerdos exclusivos duplicados no crean múltiples valores activos; actualización sustituye y marca historia; `forget` elimina la memoria activa **y la antigua supersedida**, reapertura no resucita contenido y un nuevo hecho puede guardarse correctamente.
- **3/3 tests específicos PASS en 5,54 s**; verificación de diff mínimo PASS.

El primer intento [#42239581278](https://github.com/webtilians/principiodelTodo/actions/runs/42239581278) encontró únicamente un error de expectativa del test de olvido (esperaba borrar 1 registro y el producto borró correctamente **2**, activo + historial). Se corrigió **solo ese test** y se repitió la ejecución completa. La prueba negativa inicial [#42240469071](https://github.com/webtilians/principiodelTodo/actions/runs/42240469071) terminó anticipadamente porque el shell `bash -e` abortaba ante el error **esperado** del control; corregido el script de clasificación y relanzado en el #42240671483. Ambos incidentes de instrumentación quedaron documentados, **ninguno exigió cambiar el algoritmo del producto**.

## 3. Regresiones completas de la versión v0.7.0
**[Run #42239794876](https://github.com/webtilians/principiodelTodo/actions/runs/42239794876)**, los tres jobs finalizaron SUCCESS:
- `product-regressions (3.11)`: **274 PASSED, 1 SKIPPED**, 59,86 s.
- `product-regressions (3.12)`: **274 PASSED, 1 SKIPPED**, 78,83 s.
- `atomic-stdio-mcp`: **3 PASSED**, 5,54 s, con diff exacto y dos/four procesos.

TOTAL en ambos intérpretes: **548 ejecuciones de tests correctas y 2 omisiones de la MISMA prueba opcional** `tests/test_semantic_v3.py:35`, que exige tener instalado el modelo externo `INFINITO_EMBEDDINGS`. La prueba de embeddings **NO fue ejecutada**; no se cuenta como PASS. Se instalaron extras del producto `.[mcp,dev,semantic,vault]`, `httpx2` y `uvicorn`: la batería cubre MCP, memoria local/persistente, usuario/proyecto, cifrado remoto, cuentas/autenticación, copias/restauración, página de recuerdos y bóveda. Todos los datos son sintéticos en runners efímeros.

Otros workflows de CI de producto de esa rama (**Repository integration checks, INFINITO 3.0 Core, Demo tests, Claude Desktop extension**) también terminaron SUCCESS en commit `eaee404b7b7ca0b6dc47e4abdb96192ff88b8cd1`.

## 4. Calibración de gates y limitaciones

| Gate | Estado | Alcance |
|---|---|---|
| G0 — cambio mínimo congelado y rastreable | **PASS** | diff exacto de fuente de producto comprobado automáticamente |
| G1 — dos y cuatro MCP independientes, SQLite compartida | **PASS** | 12 rondas de cada escenario, observación con SQL y MCP; base negativa con cuatro ACTIVE |
| G2 — regresiones v0.7.0 | **PASS con 1 omisión declarada por intérprete** | 274/274 ejecutadas en Python 3.11 y 3.12 |
| G3 — actualizar, idempotencia, olvido, reapertura y proyectos | **PASS** | test adicional sintético + suite existente |
| G4 — no sobreprometer servidor HTTPS multiworker | **LÍMITE EXPLÍCITO** | No probado ni corregido por esta intervención |
| G5 — reproducibilidad con resultados del control negativo | **PASS** | run y commit fijos, logs/artefactos conservados, incidentes de CI anotados |

### Advertencia arquitectónica sobre remoto cifrado
`SealedMemories.use` en `remote.py` **desencripta una SQLite completa en RAM, ejecuta la operación y vuelve a escribir un fichero cifrado bajo un lock por instancia Python**. La transacción `BEGIN IMMEDIATE` se ejecuta sobre esa SQLite en memoria **y no serializa operaciones de diferentes procesos HTTPS sobre el mismo fichero sellado**. Tampoco se ha demostrado compatibilidad de dos replicas HTTPS compartiendo filesystem. **No presentar esta integración como solución multiworker HTTP**: para ese escenario se necesita coordinación interproceso en todo el ciclo leer/desencriptar/modificar/sellar/reemplazar, y pruebas separadas con trabajadores reales.

### Riesgos no despejados
- El stress cubre dos y cuatro procesos y 12 rondas por escenario, **no todas las intercalaciones posibles** ni carga sostenida de producción.
- No se ejecutó el test de modelo semántico opcional porque `INFINITO_EMBEDDINGS` no estaba presente.
- No se accedió al despliegue activo ni se verificó su commit, configuración de workers, copias reales ni recuperación operacional.
- No hay métricas de latencia y throughput a escala, más allá de la regresión funcional.

## 5. Decisión
**RECOMENDACIÓN: GO a revisión del PR #110 y eventual integración del pequeño parche SQLite local si el mantenedor aprueba explícitamente la fusión y se comprueba despliegue de un único worker HTTPS o se resuelve separadamente la coordinación multiworker.** No equivale a autorización de despliegue inmediato ni a corrección de toda posible condición de carrera en el servidor remoto. Mantener #110 **DRAFT**, preservar #78 y exigir backup verificable previo al release.

La incorporación de un modelo de estrategia autónoma F10/F12 es totalmente distinta y sigue **NO-GO**; no se ha tocado.

## 6. Revalidación sobre el nuevo master que llegó durante las pruebas

Durante esta auditoría se fusionó el PR #109 de pruebas de fotos. El nuevo `master` quedó en **`2671e889f619e115e77db2e2b7ec906b9a4cfd78`**, con cambios en `remote.py`, `remote_photos.py` y tests, mientras `persistent_memory.py` seguía sin el fix.

Para no confundir este cambio de base con una mejora validada, añadimos una tercera ejecución independiente de CI: **[GitHub Actions #42241258007](https://github.com/webtilians/principiodelTodo/actions/runs/42241258007)**. Su workflow `.github/workflows/f10s-latest-master-overlay.yml` no rebasa ni fusiona el PR: crea un worktree EXACTO desde commit `2671e889`, copia **solo** el parche SQLite mínimo y los dos tests F10-S, y coteja que el resultado del archivo de producto es matemáticamente equivalente a `before.replace(anchor, patch)`. Verifica además la presencia de `tests/test_remote_photos_v3.py` y ejecuta TODA la suite nueva de este commit.

Resultados finales en ese nuevo `master`:
- **Python 3.11:** **298 tests de producto PASS, 1 SKIP** (modelo `INFINITO_EMBEDDINGS` ausente), más **3/3 pruebas multiproceso PASS**, test comparativo de diff PASS. Job SUCCESS.
- **Python 3.12:** **298 tests de producto PASS, 1 SKIP** (misma prueba), más **3/3 pruebas multiproceso PASS**, diff PASS. Job SUCCESS.
- **596 ejecuciones de tests del producto PASS + 6 ejecuciones del bloque F10-S PASS; 2 SKIP**, una por intérprete del mismo módulo semántico opcional.
- Están incluidos los nuevos contratos de las herramientas de foto y los cambios recientes del servidor HTTP de #109. Ninguna línea nueva del producto fue necesaria.

Esta segunda batería sustituye como evidencia de actualidad al primer baseline `129e7018`, que permanece documentado por trazabilidad causal. **Estado final para el `master` más reciente: PASS funcional y de concurrencia SQLite**, con la excepción explícita de la prueba semántica no ejecutada y **SIN cobertura de escritura HTTPS multiworker**.

**No se ha desplegado ni fusionado el PR #110 ni se ha tocado la memoria de un usuario.**
