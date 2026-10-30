# E10-B — Autorización de lanzamiento único de confirmación real

**Fecha:** 30-10-2026.

El protocolo `docs/INFINITO_E10B_LIVE_SEMANTIC_PROTOCOL.md` está preregistrado desde commit `2d9fdcf7fe18ecc07991e2c72d95021f1a2f9935`.

El código y los tests están congelados en la rama antes de este commit de lanzamiento. Los **7 tests específicos E10-B**, **6 contratos E10-A** y **75 tests del núcleo** pasaron en el workflow offline [#41109066264](https://github.com/webtilians/principiodelTodo/actions/runs/41109066264) (SUCCESS), manteniendo `src/infinito3/` intacto.

Este fichero activa exclusivamente el workflow `infinito-e10b-real-semantic-one-shot.yml` por `push`.

Secuencia autorizada: comprobar disponibilidad real de OPENAI_API_KEY, modelo y embeddings → ejecutar seis parejas DEV `1700000+` → verificar que las respuestas llevan ID/tokens de proveedor real → abrir **una sola vez** los 24 mundos confirmatorios `1800000+`, familias preasignadas, sin selección de muestras, sin cambiar modelo ni filtros.

**La falta de API o de modelo válido detendrá el proceso.** No se sustituirá por mocks; no se atribuirá ningún resultado offline al ensayo con LLM real. Tras la primera apertura, las semillas confirmatorias quedan consumidas; cualquier corrección posterior será exploratoria.

El holdout histórico F9 `900000+` no se abrirá.
