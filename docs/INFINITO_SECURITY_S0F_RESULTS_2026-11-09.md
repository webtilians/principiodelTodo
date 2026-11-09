# INFINITO Security S0-F — resultados: procedencia firmada por servidor OAuth/MCP

**Fecha:** 2026-11-09 · **Estado:** GitHub Actions SUCCESS · **Decisión:** GO técnico para revisión independiente; **no desplegar ni fusionar automáticamente**.

## Fuentes verificables

- Protocolo preregistrado antes de la implementación: commit `7a42efb387f0e929fb8df34a2d1bb5b15b9ef06a`.
- PR draft #147: https://github.com/webtilians/principiodelTodo/pull/147 (apilado sobre S0-E #146, S0-D #141, S0-C #138, S0-B #137, S0-A #136).
- Primer run `42675406712` **FAIL**: error de preparación de un recuerdo sintético rechazado por el validador de memorias («Not stored: questions, requests and acknowledgements are not memories»); 39 tests pasaron y 16 casos E2E no pudieron iniciar. No se obtuvo ningún resultado K00–K15.
- Reparación solo en contenido natural de las **fixtures** de recuerdo, sin cambiar pruebas ni gates: commit `1ec69a5a98d1b331b0469ded340b4c922eba3268`.
- Run final `42675546385`: https://github.com/webtilians/principiodelTodo/actions/runs/42675546385 — **SUCCESS**, `55 passed in 5.61s` (S0-A/B/C/D/E + S0-F, incluidos K00–K15) y `28 passed in 6.14s` (suite de cuentas y OAuth/MCP). La imagen Docker se ejecutó con `--network none`, root filesystem de solo lectura, cap drop ALL, no-new-privileges y usuario no root.
- Código: `src/infinito3/remote_provenance.py`, `src/infinito3/remote.py`, `tests/test_security_lab_s0f.py`, `.github/workflows/infinito-security-s0f.yml`.

## Implementación

- Dos **nuevas herramientas** `provenance_public_key` y `recall_attested`, expuestas solo si `INFINITO_EXPERIMENTAL_ATTESTED_MCP=1` y con OAuth; sin flag no se crea la clave privada ni se habilita la característica.
- Ed25519 con clave privada generada por el proceso servidor, persistida en archivo local `0600`, con comprobación de permisos, propietario y longitud. La clave pública tiene fingerprint SHA-256 y se verifica por el cliente usando una clave **anclada** desde un canal de confianza independiente.
- Cada respuesta firmada incluye emisor, cuenta OAuth, `client_id`, nombre de herramienta, nonce aleatorio de cliente, digest SHA-256 de la consulta, TTL 120 segundos y hasta 10 registros del almacén cifrado que el servidor abrió para esa cuenta (id, contenido, fecha, `record_owner` igual al propietario del **almacén autenticado**). El texto de la memoria nunca decide su dueño.
- Verificador Ed25519 independiente rechaza sobre modificado, nonce/consulta/cliente/identidad equivocados y expiración. Las herramientas de escritura y lectura existentes se preservan.

## Evidencias del nuevo banco K00–K15

| Casos | Control | Evidencia observada |
|---|---|---|
| K00–K03 | Firmas de lecturas legítimas de tres cuentas, clave pública igual para las sesiones y tras reconstrucción del firmante, archivo 0600 | PASS |
| K04–K07 | Alterar propietario, contenido, id o lista de registros después de la firma | 4/4 rechazados |
| K08–K11 | Alterar nonce, consulta, cuenta o `client_id` | 4/4 rechazados |
| K12–K15 | Textos que falsifican propietario, HTTP sin OAuth, aislamiento entre dos cuentas, app en nivel read-only | PASS |
| Controles adicionales | Fecha caducada/futura, clave pública falsa, emisor falso, clave en disco con permisos inseguros/corrupta, intento de firmar registro de otro propietario | Rechazados |
| Regresiones anteriores | S0-A/B/C/D/E y suite `test_remote_accounts_v3.py` | 55 + 28 pruebas PASS |

## Gates

- **G0 PASS** laboratorio aislado, flag opt-in, 16 casos nuevos; falta validación externa adicional antes de producto.
- **G1 PASS** firmas auténticas validadas y clave reconstruida desde almacenamiento persistente.
- **G2 PASS** falsificaciones y cruces de contexto K04–K11 rechazados 8/8.
- **G3 PASS** autenticación y aislamiento comprobados sobre las cuentas ficticias; sin OAuth HTTP 401.
- **G4 PASS** pruebas anteriores y nuevas sin regresiones en CI.

## Límites de seguridad y de interpretación

1. **No es una prueba de autoría original.** El campo `record_owner` acredita de qué almacén cifrado de cuenta procede el registro en esta lectura; no demuestra quién escribió el contenido ni de qué fuente externa vino. Si una aplicación o el servidor insertara un registro ajeno en el almacén equivocado, el firmante podría acreditar incorrectamente la propiedad del almacén. La firma no sustituye controles de autorización, historial de procedencia por registro ni auditoría de compartición.
2. **La confianza en la clave no se autogenera.** La herramienta `provenance_public_key` devuelve un valor público; solo un cliente que lo compare con un pin obtenido por un canal confiable puede demostrar que la firma corresponde al emisor esperado. TOFU sin revisión no impide suplantación en el primer contacto. El protocolo no establece aún distribución y rotación de claves apta para producción.
3. **Clave en disco:** es deliberadamente persistente, en claro y con permisos `0600`, accesible al usuario de servicio; requiere protección de backups, controles operativos, estrategia de revocación/rotación y manejo de múltiples instancias antes de producción.
4. **Datos sintéticos únicamente.** Los 16 casos nuevos K00–K15 comprueban autenticidad e aislamiento, no un conjunto de hallazgos reales. No se repitió una inferencia LLM: en S0-F se comprueba la emisión y validación **del servidor**; el piloto S0-E previo ya mostró que el LLM solo produce falsos positivos en evidencia sin acreditar.
5. **Clon y ejecución experimental.** El PR no se fusionó ni se desplegó en servidores públicos. No afirma seguridad absoluta de INFINITO.

**Siguiente paso:** revisión de seguridad de key management, límites de tenant y semántica de autoría; confirmación en un clon separado con otras cuentas, datos y tipos de registros. Dejar el flag apagado por defecto hasta finalizar revisión.
