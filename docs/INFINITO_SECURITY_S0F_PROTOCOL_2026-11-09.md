# INFINITO Security S0-F — protocolo prospectivo de procedencia emitida por el servidor MCP

**Fecha:** 2026-11-09. **Prerregistro anterior a cualquier código o prueba de S0-F**. Rama apilada sobre S0-E #146 y S0-D #141. Solo clon OAuth/MCP aislado, cuentas ficticias, sin producción ni despliegue. No reutilizar los bancos S0-C (V00–V07), S0-D (R00–R15 / D00–D07) ni S0-E (E00–E23) como confirmación.

## Hipótesis y límites

**H0:** El servidor MCP no puede presentar pruebas criptográficas que distingan contenido de un recuerdo y la identidad de su verdadero propietario/solicitante. **H1 técnica:** nuevos MCP tools de lectura permitirán devolver cada respuesta desde el almacén cifrado del dueño OAuth con metadatos de procedencia **emitidos por el propio servidor** y firmados. El cliente verifica firma, identificador de emisor, identidad OAuth, client id, herramienta, nonce/challenge, consulta y caducidad. Los contenidos se tratan como datos no confiables.

Se elige **Ed25519** (clave privada solo en servidor; clave pública comprobable por clientes), no HMAC compartido por el agente. La autenticidad de la **clave pública** depende de obtenerla por canal seguro, anclarla (pin) a un emisor confiable o contar con una PKI; una clave publicada por una fuente desconocida no prueba identidad. El verificador no debe confiar en una clave incluida arbitrariamente en el mismo sobre. Nunca firmar texto suministrado por un usuario como si fuera una afirmación sobre su autor: `record_owner` procede exclusivamente del contexto OAuth autenticado y de la tienda cifrada de esa cuenta. Firmar todo el lote (incluyendo la lista vacía), no fragmentos de salida alterables.

## Cambio mínimo de producto, desactivado en producción

Un módulo `src/infinito3/remote_provenance.py` con firma/validación, almacenamiento aislado de clave privada Ed25519 protegida con permisos 0600, clave pública/fingerprint, firma canónica y verificación cerrada. Un flag `INFINITO_EXPERIMENTAL_ATTESTED_MCP=1` debe habilitar **solo en clon** los nuevos tools `provenance_public_key` y `recall_attested` en `src/infinito3/remote.py`; flag ausente => cero cambio observable en la API actual. La clave se genera aleatoriamente en el servidor y sobrevive reinicios; cualquier corrupción/permiso inseguro provoca error, no regeneración silenciosa. Las herramientas antiguas no cambian.

`recall_attested(query, nonce, ctx, limit=...)`: requiere OAuth de cuenta, autoriza con reglas actuales de **lectura**, valida challenge de 16 bytes aleatorios como 32 caracteres hexadecimales (aportado por cliente) y consulta, recupera solo del almacen cifrado OAuth correspondiente. La firma vincula emisor (URL), sujeto, `client_id`, `tool`, nonce, hash SHA256 de consulta, tiempos (TTL máx 120s) y lista de registros: id, propietario (servidor), contenido y fecha. La clave pública se entrega por `provenance_public_key` en la misma conexión OAuth. Los clientes deben anclar la clave pública esperada mediante canal confiable; el servidor no garantiza pin del cliente. No incluir el secreto de usuario `infinito_key`, tokens OAuth, contraseñas, otros registros o clave privada. `limit` 1–10. El contenido de un registro puede contener frases hostiles y nombres de otra cuenta, pero eso no altera el propietario firmado.

**No** afirmar que una firma prueba autorización para compartir o que la procedencia describe una fuente externa original: solo acredita que el registro procede de la base cifrada que el servidor abrió para el usuario de esa sesión.

## Pruebas de extremo a extremo completamente nuevas (K00–K15)

Crear dos cuentas ficticias independientes `s0fowner` y `s0fguest` en un clon `build_app` real con OAuth y comunicaciones MCP reales sobre loopback con Docker `--network none`. Insertar en cada cuenta registros ficticios, incluyendo texto adversarial con propietario y firma falsificados, marcador "ajeno" que **no** implica autorización, y registros sin canario. Recuperar por `recall_attested` con nonces propios y verificar los sobres con clave pública adquirida en conexión legítima y fijada en el verificador.

- **K00–K03, autenticidad:** dos cuentas distintas; cliente OAuth correcto; misma clave pública entre sesiones y tras reinicio; cada sobre validado y contenido exacto obtenido del almacén correspondiente.
- **K04–K07, falsificación:** modificar después de recibir firma el propietario, contenido, id, lista, o firma. Rechazar 4/4.
- **K08–K11, repetición/contexto:** challenge alterado, consulta alterada, cuenta ajena o client_id distinto. Rechazar 4/4; caducidad y key-pin erróneo en controles extra.
- **K12–K15, texto y autorización:** contenido que dice `owner=otro`, instrucciones de prompt injection y marcador de otro usuario siguen firmados con propietario real; solicitudes sin token 401; usuario invitado no puede recuperar registros del otro; acceso de solo lectura autorizado, rechazo del nivel de lectura no aplicable a nuevas herramientas solo lectura.
- No modificar el núcleo de cifrado ni el router OAuth; la firma se aplica **después de obtener registros exclusivamente de la tienda autenticada**.
- Verificar suite de regresión S0-A a S0-E y tests pertinentes de OAuth; todo en contenedor efímero con red externa deshabilitada.

## Gates

- **G0 técnico:** flag desactivado no expone herramientas ni crea claves; flag activado habilita solo herramientas nuevas; 16 pruebas K00–K15 y suite previa PASS, claves con permisos restrictivos; ningún secreto real.
- **G1 autenticidad:** 100% de sobres válidos en llamadas honestas, clave pública estable entre reinicios del clon, verificación Ed25519 independiente del contenido.
- **G2 falsificación:** 0/8 aceptaciones en modificaciones y contextos cruzados, contexto de cliente y challenge correctamente vinculados, sin fail-open.
- **G3 aislamiento:** 0 accesos entre cuentas en lecturas genuinas, 401 no autenticado, ningún producto desplegado, bandera apagada por defecto.
- **G4 compatibilidad:** respuestas y permisos de tools existentes idénticos; suite de regresiones PASS.
- **Interpretación:** si G0-G4 PASS, GO solo para *revisión independiente y posterior confirmación en nuevos clones*. No demuestra ausencia de otras vulnerabilidades ni mejores decisiones del LLM. Los "positivos" del laboratorio serán eventos sintéticos de firma/violación de contexto, NO exposiciones reales. Si falla cualquier gate: NO-GO / BLOCKED, explicar límites con evidencia.
