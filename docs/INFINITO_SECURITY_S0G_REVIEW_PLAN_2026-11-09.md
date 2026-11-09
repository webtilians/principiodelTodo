# S0-G — plan congelado de pruebas adversariales para revisión externa de S0-F

**Fecha:** 2026-11-09 · **Base auditada:** PR #147 commit `ee9974bcadb81b8fabf3f9d729fb1ff811167fd2`. Esta rama **NO modifica el código de implementación** (`src/infinito3/remote_provenance.py` ni `src/infinito3/remote.py`). Añade solo pruebas, documentación y workflow para auditores. El equipo/IA que escribió S0-F NO puede considerarse auditor independiente; se requiere revisor distinto registrado en GitHub.

## Hipótesis de seguridad

La suite K00–K15 prueba autenticidad por clave Ed25519 y aislamiento de lectura OAuth en dos cuentas, pero **no** la integridad de un proceso de rotación/compromiso, distribución de trust anchor, revocación de claves, defensa anti-replay de nonce usado ni metadatos de autor real. La siguiente suite no debe presentar un test técnico positivo como autorización de despliegue.

## Casos nuevos A00–A09 (preregistrados antes de crear tests)

- **A00:** un sobre firmado por la clave del servidor verdadero solo es aceptado con pin correcto, emisor, tenant y cliente OAuth esperados.
- **A01:** un emisor falso que dice tener igual URL de issuer puede emitir un sobre coherente; un pin auténtico lo rechaza, pero elegir el pin ofrecido por el propio falso emisor deja que pase. Muestra por qué la herramienta publica no reemplaza una raíz de confianza.
- **A02:** key file corrupto o permisos excesivos obligan a error y no regeneración.
- **A03:** restablecer/reemplazar físicamente la clave genera nueva identidad y rompe pin anterior: no hay protocolo de rotación autenticada implementado.
- **A04:** propiedad de registro en payload se limita a dueño del almacén y no verifica quien escribió un texto. Registro con atribución falsa en su propio texto sigue etiquetado por su almacén.
- **A05:** la herramienta rechaza firma de un registro etiquetado con propietario distinto a la cuenta autenticada.
- **A06:** la misma respuesta firmada puede aceptarse dos veces dentro del TTL si el cliente reutiliza exactamente el nonce; prueba **de limitación**, no fuga demostrada.
- **A07:** después de pérdida/compromiso y reemplazo físico de clave, una prueba firmada con clave vieja y pin viejo sigue verificándose durante su TTL: falta revocación. **XFAIL** contra el requisito de rechazo inmediato.
- **A08:** la raíz de almacenamiento con escritura por cualquiera no provoca fallo del firmante, a pesar de la clave individual 0600: falta endurecer directorio y threat model. **XFAIL** contra el requisito de directorio seguro.
- **A09:** la sustitución de clave por el emisor falso en respuesta no altera validez de un pin verdadero; el cliente que usa pin erróneo acepta prueba falsa. Test positivo de dependencia de pin.

## Interpretación de resultados

- El diseño registra los **controles pendientes como XFAIL explícitos**, no se cambian para obtener verde; el informe resume cuántos PASS / XFAIL y limita el gate a **AUDIT REQUIRED / NO-GO production**.
- Pruebas de rotación/revocación y anti-replay no se deben transformar en afirmaciones de vulnerabilidades explotadas de INFINITO en producción (funcionalidad está opt-in y sin desplegar).
- CI no usa red externa (`docker --network none`), cuentas/firmas de pruebas únicas, rootfs solo lectura, y corre también S0-F previo para descartar regresiones.
- Al pasar esas pruebas y solucionar pendientes, un revisor externo debe repetir en un clon nuevo con distintos tenants, cuentas recuperadas, shares, scopes, tokens revocados y claves rotadas. No reutilizar K00–K15 como confirmación.
- Riesgos/gates detallados en [issue #149](https://github.com/webtilians/principiodelTodo/issues/149).

**Bloqueo:** CI verde con `xfail` NO implica GO ni aprobación. El proceso completo requiere auditor independiente nombrado y cambios revisados con evidencias.
