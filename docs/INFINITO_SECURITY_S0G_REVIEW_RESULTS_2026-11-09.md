# Informe técnico preparatorio S0-G — auditoría independiente aún pendiente

**Fecha:** 2026-11-09. **Estado:** `NO-GO production / EXTERNAL REVIEW REQUIRED`. Este informe lo preparó el mismo asistente que implementó S0-F; **NO cuenta como revisión independiente** ni certifica seguridad.

## Fuentes y resultados comprobados

- Origen bajo revisión: PR #147, commit `ee9974bcadb81b8fabf3f9d729fb1ff811167fd2`.
- Dossier y bloqueos formales: [issue #149](https://github.com/webtilians/principiodelTodo/issues/149).
- Pruebas A00–A09 creadas **después** del [plan preregistrado](INFINITO_SECURITY_S0G_REVIEW_PLAN_2026-11-09.md) en commit `b85c77e`.
- PR de revisión técnica (sin alterar implementación): [#150](https://github.com/webtilians/principiodelTodo/pull/150).
- Ejecución offline [Actions #42676893969](https://github.com/webtilians/principiodelTodo/actions/runs/42676893969), salida `56 passed, 2 xfailed in 8.02s`, workflow success. Incluye tests de auditoría, S0-F y OAuth/MCP anteriores. Docker `--network none`, read-only, `cap-drop ALL`, `no-new-privileges`, rootfs solo lectura y usuario no-root.

## Casos A00–A09

| Caso | Propiedad evaluada | Resultado |
|---|---|---|
| A00 | Firma auténtica vinculada a cliente, cuenta, nonce, consulta y emisor | PASS |
| A01 | Un emisor con la misma URL alegada pero clave privada diferente solo puede engañar a un cliente que confía en el pin equivocado | PASS (depende de pin independiente) |
| A02 | Archivo de clave corrupto o con permisos de fichero 0644 rechazado sin regeneración silenciosa | PASS |
| A03 | Sustitución física de clave provoca cambio de identidad y fallo del cliente con pin anterior | PASS (demuestra ausencia de transición administrada) |
| A04 | Texto que finge autoría no altera `record_owner` del almacén | PASS; falta linaje/autoría real |
| A05 | Firmante rechaza registro etiquetado con otro propietario | PASS |
| A06 | Dos verificaciones sucesivas del mismo sobre con igual nonce dentro del TTL son aceptadas | PASS — **limitación de replay documentada** |
| A07 | Prueba firmada con clave comprometida debería invalidarse tras revocación antes del TTL | **XFAIL — NO implementado** |
| A08 | Directorio raíz con escritura por cualquiera debería rechazarse | **XFAIL — NO implementado** |
| A09 | Sustitución de key_id/clave pública con pin auténtico es rechazada | PASS |

**Interpretación correcta:** `XFAIL` significa que un requisito de seguridad deliberadamente exigido **no se cumple**; no se debe elevar la ejecución a `GO`. En particular, A07 simula reemplazar el archivo local de firma **sin** comunicar revocación a clientes: un cliente que aún tiene anclada la clave comprometida sigue aceptando pruebas anteriores dentro del TTL; falta un mecanismo de confianza/revocación. A08 muestra que comprobar 0600 en el archivo no sustituye examinar la seguridad del directorio que lo contiene.

## Bloqueos antes de producto

1. [P1] Implementar y ensayar todo el ciclo de vida Ed25519: clave actual, sucesora, retirada, revocada, emergencia por compromiso, publicación segura, políticas de retención y múltiples workers. Una clave privada en disco bajo 0600 es insuficiente como única medida operativa.
2. [P1] Definir confianza inicial/actualización de clave pública verificada fuera de la propia respuesta que se autentica. Los tests aceptan el pin del cliente como precondición; no lo distribuyen.
3. [P1] No presentar dueño de almacén como autor original ni como fuente original de un registro. Diseñar linaje separado para importaciones/compartición si el producto necesita tales afirmaciones.
4. [P2] Política anti-replay: nonce criptográficamente nuevo por petición, posible unicidad/consumo y handling de respuesta duplicada dentro del TTL; no anunciar un bloqueo de replay de un solo uso con la implementación actual.
5. [P2] Verificar permisos/directorios, revocación de OAuth, scopes, recuperación/importación/compartición, carreras concurrentes, colisión de tenants, backups y cambio de instancia sobre un **clon nuevo**.
6. [P1] Recibir **aprobación de una persona revisora de seguridad independiente**, registrada mediante GitHub Review en #147. Hasta entonces mantener PR como DRAFT, flag opt-in apagado por defecto y NO DEPLOY.

## Lo que la batería NO demuestra

No demuestra explotación de producción, fuga de memorias entre clientes, robustez tras compromiso de servidor, protección de autores originales, ni un protocolo real de distribución de confianza. La revisión automatizada se ha realizado sobre código S0-F escrito por el mismo asistente; solo prepara una evaluación externa.

**Decisión:** S0-F sirve para investigación y clon de laboratorio. El gate para integración comercial sigue explícitamente BLOQUEADO. [OWASP Key Management](https://cheatsheetseries.owasp.org/cheatsheets/Key_Management_Cheat_Sheet) requiere una política completa de generación, distribución, protección, rotación y retirada de claves.
